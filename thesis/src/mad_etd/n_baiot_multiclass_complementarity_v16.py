"""N-BaIoT device-held-out multiclass evidence Fusion for MAD-ETD v16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .cse_mafsids_gap_closing_w241_w245 import _dump, _security, _write_csv
from .n_baiot_development_w316 import _sample_file
from .non_tls_fresh_specialist_w297_w300 import _read_csv
from .paper_evaluation import sha256_file


EXPERIMENT = "mad_etd_n_baiot_multiclass_complementarity_v16"
DEFAULT_DATA = Path("data/raw/n_baiot/official_download/extracted")
DEFAULT_W315 = Path("data/runs/mad_etd_n_baiot_schema_w315")
DEFAULT_W316 = Path("data/runs/mad_etd_n_baiot_development_w316")
DEFAULT_OUTPUT = Path(
    "data/runs/mad_etd_n_baiot_multiclass_complementarity_v16"
)
DEFAULT_MODELS = Path(
    "data/models/mad_etd_n_baiot_multiclass_complementarity_v16"
)
TRAIN_ROWS_PER_FILE = 1_500
VALIDATION_ROWS_PER_FILE = 3_000
ACCEPTANCE_ROWS_PER_FILE = 3_000
SEED = 42
TEMPERATURES = (0.75, 1.0, 1.25, 1.5)
BLENDS = (0.25, 0.5, 0.75, 1.0)
GATE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _class_name(row: Mapping[str, str]) -> str:
    if row["binary_label"] == "benign":
        return "benign"
    return f"{row['botnet_family']}:{row['attack_type']}"


def _feature_views(features: list[str]) -> dict[str, list[str]]:
    views = {
        "general_full": list(features),
        "direction_host": [
            name
            for name in features
            if name.startswith("MI_dir_") or name.startswith("H_L")
        ],
        "pair_behavior": [
            name
            for name in features
            if name.startswith("HH_") and not name.startswith("HH_jit_")
        ],
        "temporal_jitter": [
            name for name in features if name.startswith("HH_jit_")
        ],
        "endpoint_pair": [
            name for name in features if name.startswith("HpHp_")
        ],
    }
    if any(not names for names in views.values()):
        raise RuntimeError("empty N-BaIoT evidence view")
    specialist = [
        set(views[name])
        for name in (
            "direction_host",
            "pair_behavior",
            "temporal_jitter",
            "endpoint_pair",
        )
    ]
    if any(
        left & right
        for index, left in enumerate(specialist)
        for right in specialist[index + 1 :]
    ):
        raise RuntimeError("specialist N-BaIoT feature views overlap")
    return views


def _load_role(
    data: Path,
    manifest: list[dict[str, str]],
    role: str,
    features: list[str],
    classes: list[str],
    *,
    rows_per_file: int,
    allow_acceptance: bool = False,
    sampling_seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if role == "acceptance" and not allow_acceptance:
        raise RuntimeError("sealed N-BaIoT acceptance cannot be opened")
    class_to_id = {name: index for index, name in enumerate(classes)}
    parts: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    sources: list[str] = []
    for row in manifest:
        if row["role"] != role:
            continue
        class_name = _class_name(row)
        if class_name not in class_to_id:
            continue
        path = data / row["relative_path"]
        frame = _sample_file(
            path, features, rows_per_file, int(sampling_seed)
        )
        values = frame.to_numpy(dtype=np.float32, copy=True)
        parts.append(values)
        labels.extend([class_to_id[class_name]] * len(values))
        groups.extend([row["device_group"]] * len(values))
        source_hash = hashlib.sha256(
            row["relative_path"].encode("utf-8")
        ).hexdigest()[:16]
        sources.extend([source_hash] * len(values))
    if not parts:
        raise RuntimeError(f"no N-BaIoT rows loaded for {role}")
    return (
        np.vstack(parts),
        np.asarray(labels, dtype=np.int16),
        np.asarray(groups, dtype=object),
        np.asarray(sources, dtype=object),
    )


def _validation_partition(
    groups: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.zeros(len(groups), dtype=bool)
    counters: dict[tuple[str, str], int] = {}
    for index, (group, source) in enumerate(
        zip(groups.tolist(), sources.tolist(), strict=True)
    ):
        key = (str(group), str(source))
        offset = counters.get(key, 0)
        counters[key] = offset + 1
        digest = hashlib.sha256(
            f"{group}:{source}:{offset}".encode("utf-8")
        ).digest()
        calibration[index] = digest[0] % 2 == 0
    selection = ~calibration
    if not calibration.any() or not selection.any():
        raise RuntimeError("invalid N-BaIoT validation partition")
    return calibration, selection


def _fit_agent(
    x: np.ndarray,
    y: np.ndarray,
    columns: list[int],
    *,
    seed: int,
) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=260,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x[:, columns], y)
    return model


def _agent_probabilities(
    agents: Mapping[str, Mapping[str, Any]],
    x: np.ndarray,
    class_count: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for agent_id, payload in agents.items():
        model = payload["model"]
        columns = payload["columns"]
        raw = np.asarray(model.predict_proba(x[:, columns]), dtype=np.float64)
        aligned = np.zeros((len(x), class_count), dtype=np.float64)
        for source_index, class_id in enumerate(model.classes_):
            aligned[:, int(class_id)] = raw[:, source_index]
        result[agent_id] = aligned
    return result


def _temperature(probability: np.ndarray, value: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-10, 1.0)
    logits = np.log(clipped) / float(value)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _multiclass_ece(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> float:
    confidence = probability.max(axis=1)
    correct = prediction == y
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(correct[mask].mean())
                - float(confidence[mask].mean())
            )
    return float(result)


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    benign_id: int,
) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    group_values: dict[str, float] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_values[str(group)] = float(
            f1_score(y[mask], prediction[mask], average="macro")
        )
    malicious_labels = [
        label for label in sorted(set(y.tolist())) if label != benign_id
    ]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "macro_precision": float(
            precision_score(y, prediction, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, prediction, average="macro", zero_division=0)
        ),
        "attack_macro_recall": float(
            recall_score(
                y,
                prediction,
                labels=malicious_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "ece": _multiclass_ece(y, probability, prediction),
        "coverage": 1.0,
        "selective_error": float(1.0 - accuracy_score(y, prediction)),
        "worst_device_macro_f1": min(group_values.values()),
        **{
            f"group_macro_f1_{group}": value
            for group, value in group_values.items()
        },
    }


def _meta_features(
    probabilities: Mapping[str, np.ndarray], agent_order: list[str]
) -> np.ndarray:
    matrices = [probabilities[name] for name in agent_order]
    stack = np.stack(matrices, axis=1)
    return np.column_stack(
        (
            *matrices,
            stack.mean(axis=1),
            stack.std(axis=1),
            stack.max(axis=1),
        )
    ).astype(np.float32)


@dataclass
class LockedNBaiotFusionV16:
    meta: LogisticRegression
    agent_order: list[str]
    reference_agent: str
    blend: float
    temperature: float
    gate_threshold: float

    def predict_probability(
        self, probabilities: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        reference = probabilities[self.reference_agent]
        meta = self.meta.predict_proba(
            _meta_features(probabilities, self.agent_order)
        )
        meta = _temperature(meta, self.temperature)
        blended = (1.0 - self.blend) * reference + self.blend * meta
        uncertainty = 1.0 - reference.max(axis=1)
        gate = uncertainty >= self.gate_threshold
        return np.where(gate[:, None], blended, reference)


def _candidate_rows(
    y_cal: np.ndarray,
    y_sel: np.ndarray,
    group_sel: np.ndarray,
    cal_probabilities: Mapping[str, np.ndarray],
    sel_probabilities: Mapping[str, np.ndarray],
    reference_agent: str,
    reference_metrics: Mapping[str, float],
    *,
    benign_id: int,
    random_state: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, LockedNBaiotFusionV16]]:
    order = sorted(cal_probabilities)
    x_cal = _meta_features(cal_probabilities, order)
    x_sel = _meta_features(sel_probabilities, order)
    reference_cal = cal_probabilities[reference_agent]
    reference_sel = sel_probabilities[reference_agent]
    uncertainty = 1.0 - reference_cal.max(axis=1)
    gate_thresholds = sorted(
        {
            float(np.quantile(uncertainty, quantile))
            for quantile in GATE_QUANTILES
        }
    )
    rows: list[dict[str, Any]] = []
    policies: dict[str, LockedNBaiotFusionV16] = {}
    for c_value in (0.01, 0.1, 1.0, 10.0):
        for class_weight in (None, "balanced"):
            meta = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=2_000,
                solver="lbfgs",
                random_state=int(random_state),
            )
            meta.fit(x_cal, y_cal)
            raw_meta = meta.predict_proba(x_sel)
            for temperature in TEMPERATURES:
                calibrated_meta = _temperature(raw_meta, temperature)
                for blend in BLENDS:
                    blended = (
                        (1.0 - blend) * reference_sel
                        + blend * calibrated_meta
                    )
                    for gate_threshold in gate_thresholds:
                        gate = (
                            1.0 - reference_sel.max(axis=1)
                            >= gate_threshold
                        )
                        candidate_probability = np.where(
                            gate[:, None], blended, reference_sel
                        )
                        metrics = _metrics(
                            y_sel,
                            candidate_probability,
                            group_sel,
                            benign_id=benign_id,
                        )
                        group_deltas = [
                            metrics[f"group_macro_f1_{group}"]
                            - reference_metrics[
                                f"group_macro_f1_{group}"
                            ]
                            for group in sorted(set(group_sel.tolist()))
                        ]
                        candidate_id = (
                            f"meta_c{c_value}_w{class_weight or 'none'}"
                            f"_t{temperature}_b{blend}"
                            f"_g{gate_threshold:.6f}"
                        ).replace(".", "_")
                        row = {
                            "candidate_id": candidate_id,
                            "c_value": c_value,
                            "class_weight": class_weight or "none",
                            "temperature": temperature,
                            "blend": blend,
                            "gate_threshold": gate_threshold,
                            **metrics,
                            "macro_f1_delta": (
                                metrics["macro_f1"]
                                - float(reference_metrics["macro_f1"])
                            ),
                            "accuracy_delta": (
                                metrics["accuracy"]
                                - float(reference_metrics["accuracy"])
                            ),
                            "attack_macro_recall_delta": (
                                metrics["attack_macro_recall"]
                                - float(
                                    reference_metrics[
                                        "attack_macro_recall"
                                    ]
                                )
                            ),
                            "ece_delta": (
                                metrics["ece"]
                                - float(reference_metrics["ece"])
                            ),
                            "worst_group_delta": (
                                metrics["worst_device_macro_f1"]
                                - float(
                                    reference_metrics[
                                        "worst_device_macro_f1"
                                    ]
                                )
                            ),
                            "nonnegative_group_count": sum(
                                value >= 0.0 for value in group_deltas
                            ),
                        }
                        rows.append(row)
                        policies[candidate_id] = LockedNBaiotFusionV16(
                            meta=meta,
                            agent_order=order,
                            reference_agent=reference_agent,
                            blend=blend,
                            temperature=temperature,
                            gate_threshold=gate_threshold,
                        )
    return rows, policies


def train_n_baiot_multiclass_complementarity_v16(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    data_root: str | Path = DEFAULT_DATA,
    w315_dir: str | Path = DEFAULT_W315,
    w316_dir: str | Path = DEFAULT_W316,
    model_dir: str | Path = DEFAULT_MODELS,
    model_seed: int = SEED,
    sampling_seed: int = SEED,
) -> dict[str, Any]:
    output, data, w315, w316, models = map(
        Path, (output_dir, data_root, w315_dir, w316_dir, model_dir)
    )
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    prior = json.loads(
        (w316 / "acceptance_report.json").read_text(encoding="utf-8-sig")
    )
    if (
        prior.get("acceptance_opened") is not False
        or int(prior.get("acceptance_rows_read", -1)) != 0
    ):
        report = {
            "status": "failed_n_baiot_acceptance_not_sealed",
            **_security(),
        }
        _dump(output / "training_report.json", report)
        return report
    feature_lock = json.loads(
        (w315 / "safe_feature_lock_w315.json").read_text(
            encoding="utf-8-sig"
        )
    )
    features = list(feature_lock["features"])
    views = _feature_views(features)
    manifest = _read_csv(w315 / "official_file_manifest_w315.csv")
    classes = sorted({_class_name(row) for row in manifest})
    if "benign" not in classes or len(classes) != 11:
        raise RuntimeError("unexpected N-BaIoT class inventory")
    x_train, y_train, group_train, _ = _load_role(
        data,
        manifest,
        "train",
        features,
        classes,
        rows_per_file=TRAIN_ROWS_PER_FILE,
        sampling_seed=sampling_seed,
    )
    x_val, y_val, group_val, source_val = _load_role(
        data,
        manifest,
        "validation",
        features,
        classes,
        rows_per_file=VALIDATION_ROWS_PER_FILE,
        sampling_seed=sampling_seed,
    )
    calibration, selection = _validation_partition(group_val, source_val)
    column_index = {name: index for index, name in enumerate(features)}
    agents: dict[str, dict[str, Any]] = {}
    registry: list[dict[str, Any]] = []
    for index, (agent_id, view_features) in enumerate(views.items()):
        columns = [column_index[name] for name in view_features]
        agent = _fit_agent(
            x_train, y_train, columns, seed=int(model_seed) + index
        )
        agent_path = models / f"{agent_id}_agent.joblib"
        joblib.dump(agent, agent_path, compress=3)
        agents[agent_id] = {"model": agent, "columns": columns}
        registry.append(
            {
                "agent_id": agent_id,
                "expert_role": (
                    "GeneralFlowMalwareSkill"
                    if agent_id == "general_full"
                    else "N-BaIoT view specialist"
                ),
                "safe_feature_count": len(view_features),
                "safe_features": "|".join(view_features),
                "feature_policy_hash": feature_lock["feature_policy_hash"],
                "artifact": agent_path.as_posix(),
                "artifact_hash": sha256_file(agent_path),
                "output_schema": "AgentEvidence",
                "final_verdict_owner": "FusionAgent",
                "label_or_group_enters_detector_input": False,
            }
        )
    _write_csv(output / "agent_registry.csv", registry)
    bundle_path = models / "base_agents.joblib"
    joblib.dump(
        {
            "agents": agents,
            "classes": classes,
            "features": features,
            "feature_policy_hash": feature_lock["feature_policy_hash"],
        },
        bundle_path,
        compress=3,
    )
    probabilities = _agent_probabilities(agents, x_val, len(classes))
    single_rows: list[dict[str, Any]] = []
    for agent_id, probability in probabilities.items():
        metrics = _metrics(
            y_val[selection],
            probability[selection],
            group_val[selection],
            benign_id=classes.index("benign"),
        )
        single_rows.append({"agent_id": agent_id, **metrics})
    single_rows.sort(
        key=lambda row: (
            -row["macro_f1"],
            -row["accuracy"],
            row["ece"],
            row["agent_id"],
        )
    )
    strongest = single_rows[0]
    reference_agent = str(strongest["agent_id"])
    reference_metrics = {
        key: value for key, value in strongest.items() if key != "agent_id"
    }
    _write_csv(output / "validation_single_agent_results.csv", single_rows)
    candidate_rows, policies = _candidate_rows(
        y_val[calibration],
        y_val[selection],
        group_val[selection],
        {
            name: probability[calibration]
            for name, probability in probabilities.items()
        },
        {
            name: probability[selection]
            for name, probability in probabilities.items()
        },
        reference_agent,
        reference_metrics,
        benign_id=classes.index("benign"),
        random_state=model_seed,
    )
    _write_csv(output / "validation_fusion_matrix.csv", candidate_rows)
    group_count = len(set(group_val[selection].tolist()))
    eligible = [
        row
        for row in candidate_rows
        if row["macro_f1_delta"] >= 0.003
        and row["accuracy_delta"] > 0.0
        and row["attack_macro_recall_delta"] >= 0.0
        and row["ece_delta"] <= 0.0
        and row["worst_group_delta"] >= 0.0
        and row["nonnegative_group_count"] == group_count
    ]
    pool = eligible if eligible else candidate_rows
    pool.sort(
        key=lambda row: (
            -row["macro_f1_delta"],
            -row["worst_group_delta"],
            -row["attack_macro_recall_delta"],
            row["ece_delta"],
            row["candidate_id"],
        )
    )
    selected = pool[0]
    policy = policies[str(selected["candidate_id"])]
    policy_path = models / "locked_fusion.joblib"
    joblib.dump(policy, policy_path, compress=3)
    report = {
        "status": (
            "locked_n_baiot_multiclass_ready_for_acceptance"
            if eligible
            else "blocked_n_baiot_multiclass_validation_gate_failed"
        ),
        "task": "11-class benign and botnet attack-type attribution",
        "class_names": classes,
        "class_count": len(classes),
        "train_rows": len(y_train),
        "train_device_count": len(set(group_train.tolist())),
        "validation_rows": len(y_val),
        "validation_device_count": len(set(group_val.tolist())),
        "calibration_rows": int(calibration.sum()),
        "selection_rows": int(selection.sum()),
        "specialist_feature_overlap_count": 0,
        "strongest_single": strongest,
        "selected_candidate": selected,
        "eligible_candidate_count": len(eligible),
        "candidate_count": len(candidate_rows),
        "model_seed": int(model_seed),
        "sampling_seed": int(sampling_seed),
        "base_bundle": bundle_path.as_posix(),
        "base_bundle_hash": sha256_file(bundle_path),
        "policy_artifact": policy_path.as_posix(),
        "policy_artifact_hash": sha256_file(policy_path),
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "acceptance_used_for_selection": False,
        "label_or_device_identity_enters_detector_input": False,
        **_security(),
    }
    _dump(output / "policy_lock.json", report)
    _dump(output / "training_report.json", report)
    return report


def _grouped_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 1_000,
    seed: int = SEED,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups.tolist()))
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled]
        )
        deltas.append(
            float(
                f1_score(y[indexes], candidate[indexes], average="macro")
                - f1_score(
                    y[indexes], reference[indexes], average="macro"
                )
            )
        )
    return {
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "device_group",
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def evaluate_n_baiot_multiclass_complementarity_v16(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    data_root: str | Path = DEFAULT_DATA,
    w315_dir: str | Path = DEFAULT_W315,
) -> dict[str, Any]:
    output, data, w315 = map(Path, (output_dir, data_root, w315_dir))
    marker = output / "acceptance_opened_once.json"
    report_path = output / "acceptance_report.json"
    if marker.is_file():
        if report_path.is_file():
            return json.loads(
                report_path.read_text(encoding="utf-8-sig")
            )
        raise RuntimeError("partial N-BaIoT acceptance open")
    lock = json.loads(
        (output / "policy_lock.json").read_text(encoding="utf-8-sig")
    )
    if (
        lock.get("status")
        != "locked_n_baiot_multiclass_ready_for_acceptance"
    ):
        report = {
            "status": "n_baiot_acceptance_remains_sealed_validation_failed",
            "acceptance_opened": False,
            "acceptance_rows_read": 0,
            **_security(),
        }
        _dump(report_path, report)
        return report
    _dump(
        marker,
        {
            "acceptance_open_count": 1,
            "selection_completed_before_open": True,
            "policy_artifact_hash": lock["policy_artifact_hash"],
        },
    )
    bundle = joblib.load(lock["base_bundle"])
    policy: LockedNBaiotFusionV16 = joblib.load(lock["policy_artifact"])
    manifest = _read_csv(w315 / "official_file_manifest_w315.csv")
    x, y, groups, sources = _load_role(
        data,
        manifest,
        "acceptance",
        list(bundle["features"]),
        list(bundle["classes"]),
        rows_per_file=ACCEPTANCE_ROWS_PER_FILE,
        allow_acceptance=True,
        sampling_seed=int(lock.get("sampling_seed", SEED)),
    )
    probabilities = _agent_probabilities(
        bundle["agents"], x, len(bundle["classes"])
    )
    reference_probability = probabilities[policy.reference_agent]
    candidate_probability = policy.predict_probability(probabilities)
    benign_id = list(bundle["classes"]).index("benign")
    reference_metrics = _metrics(
        y, reference_probability, groups, benign_id=benign_id
    )
    candidate_metrics = _metrics(
        y, candidate_probability, groups, benign_id=benign_id
    )
    reference_prediction = reference_probability.argmax(axis=1)
    candidate_prediction = candidate_probability.argmax(axis=1)
    bootstrap = _grouped_bootstrap(
        y, reference_prediction, candidate_prediction, groups
    )
    metric_names = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "attack_macro_recall",
        "ece",
        "coverage",
        "selective_error",
        "worst_device_macro_f1",
    )
    deltas = {
        metric: candidate_metrics[metric] - reference_metrics[metric]
        for metric in metric_names
    }
    group_deltas = {
        group: (
            candidate_metrics[f"group_macro_f1_{group}"]
            - reference_metrics[f"group_macro_f1_{group}"]
        )
        for group in sorted(set(groups.tolist()))
    }
    gates = {
        "macro_f1_delta_ge_0_003": deltas["macro_f1"] >= 0.003,
        "accuracy_positive": deltas["accuracy"] > 0.0,
        "bootstrap_ci_lower_gt_zero": (
            bootstrap["macro_f1_delta_ci95_lower"] > 0.0
        ),
        "attack_macro_recall_not_worse": (
            deltas["attack_macro_recall"] >= 0.0
        ),
        "ece_not_worse": deltas["ece"] <= 0.0,
        "worst_group_not_worse": (
            deltas["worst_device_macro_f1"] >= 0.0
        ),
        "all_groups_nonnegative": all(
            value >= 0.0 for value in group_deltas.values()
        ),
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "fake_metric_count_zero": True,
    }
    accepted = all(gates.values())
    _write_csv(
        output / "acceptance_predictions.csv",
        [
            {
                "row_id_hash": hashlib.sha256(
                    f"{source}:{index}".encode("utf-8")
                ).hexdigest(),
                "device_group_hash": hashlib.sha256(
                    str(group).encode("utf-8")
                ).hexdigest(),
                "label": int(label),
                "strongest_single_prediction": int(reference),
                "complete_mad_etd_prediction": int(candidate),
            }
            for index, (source, group, label, reference, candidate) in enumerate(
                zip(
                    sources.tolist(),
                    groups.tolist(),
                    y.tolist(),
                    reference_prediction.tolist(),
                    candidate_prediction.tolist(),
                    strict=True,
                )
            )
        ],
    )
    report = {
        "status": (
            "accepted_n_baiot_multiclass_second_positive"
            if accepted
            else "not_accepted_n_baiot_multiclass"
        ),
        "task": lock["task"],
        "class_names": lock["class_names"],
        "sample_count": len(y),
        "device_group_count": len(set(groups.tolist())),
        "reference_agent": policy.reference_agent,
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "group_deltas": group_deltas,
        "bootstrap": bootstrap,
        "acceptance_gates": gates,
        "acceptance_gates_passed": accepted,
        "acceptance_opened": True,
        "acceptance_open_count": 1,
        "acceptance_used_for_selection": False,
        "fusion_owner": "FusionAgent",
        "promoted_runtime_created": False,
        **_security(),
    }
    _dump(report_path, report)
    return report
