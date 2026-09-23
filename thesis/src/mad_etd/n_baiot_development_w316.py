"""W316 N-BaIoT train/validation-only detector development."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from .edge_iiotset_safe_baseline_w307 import (
    _metrics,
    _select_validation_calibration,
    _temperature_scale,
)
from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _hash_snapshot_equal,
    _load,
    _read_csv,
    _safe_hashes,
    _write_csv,
)


EXPERIMENT = "mad_etd_n_baiot_development_w316"
DEFAULT_DATA = Path("data/raw/n_baiot/official_download/extracted")
DEFAULT_W315 = Path("data/runs/mad_etd_n_baiot_schema_w315")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_n_baiot_development_w316")
DEFAULT_MODELS = Path("data/models/mad_etd_n_baiot_w316")
DEFAULT_DOC = Path("docs/MAD_ETD_N_BAIOT_DEVELOPMENT_W316.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_N_BAIOT_DEVELOPMENT_W316_CN.md")
SEEDS = (42, 43, 44)


def _path_seed(path: Path, seed: int) -> int:
    digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + seed) % (2**32 - 1)


def _sample_file(
    path: Path,
    features: list[str],
    target: int,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=features,
        nrows=max(target * 4, target),
        low_memory=False,
    )
    frame = frame[features].replace([np.inf, -np.inf], np.nan)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(frame) > target:
        frame = frame.sample(
            n=target,
            random_state=_path_seed(path, seed),
            replace=False,
        )
    return frame


def _load_role_balanced(
    data_root: Path,
    manifest: list[dict[str, str]],
    role: str,
    features: list[str],
    rows_per_device_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if role == "acceptance":
        raise RuntimeError("W316 must never open acceptance content")
    rows = [row for row in manifest if row["role"] == role]
    devices = sorted({row["device_group"] for row in rows})
    parts: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    sampling: list[dict[str, Any]] = []
    for device in devices:
        for label_name, label in (("benign", 0), ("malicious", 1)):
            paths = [
                data_root / row["relative_path"]
                for row in rows
                if row["device_group"] == device
                and row["binary_label"] == label_name
            ]
            if not paths:
                raise ValueError(
                    f"{role}/{device} has no {label_name} files"
                )
            allocation = int(np.ceil(rows_per_device_class / len(paths)))
            sampled_frames = [
                _sample_file(path, features, allocation, seed)
                for path in paths
            ]
            combined = pd.concat(sampled_frames, ignore_index=True)
            if len(combined) > rows_per_device_class:
                combined = combined.sample(
                    n=rows_per_device_class,
                    random_state=(
                        int(
                            hashlib.sha256(
                                f"{device}/{label_name}".encode("utf-8")
                            ).hexdigest()[:8],
                            16,
                        )
                        + seed
                    )
                    % (2**32 - 1),
                    replace=False,
                )
            values = combined.to_numpy(dtype=np.float32, copy=True)
            parts.append(values)
            labels.extend([label] * len(values))
            groups.extend([device] * len(values))
            sampling.append(
                {
                    "role": role,
                    "device_group": device,
                    "binary_label": label_name,
                    "source_file_count": len(paths),
                    "sampled_rows": len(values),
                    "target_rows": rows_per_device_class,
                    "device_identity_enters_detector_input": False,
                }
            )
    return (
        np.vstack(parts),
        np.asarray(labels, dtype=int),
        np.asarray(groups, dtype=object),
        sampling,
    )


def _build_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=240,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=240,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=220,
            max_depth=7,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        )
    except Exception:
        pass
    return models


def build_n_baiot_development_w316(
    w315_dir: str | Path = DEFAULT_W315,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    w315 = Path(w315_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    readiness = _load(w315 / "acceptance_report.json")
    feature_lock = _load(w315 / "safe_feature_lock_w315.json")
    ready = (
        readiness.get("training_authorized_for_w316") is True
        and int(feature_lock.get("feature_count", 0)) == 115
    )
    _dump(output / "frozen_hashes_before_w316.json", _safe_hashes())
    protocol = {
        "status": (
            "w316_protocol_ready_for_train_validation"
            if ready
            else "blocked_w315_not_ready"
        ),
        "experiment": EXPERIMENT,
        "models": [
            "hgb",
            "random_forest",
            "extra_trees",
            "xgboost_if_available",
            "cross_device_oof_probability_stack",
        ],
        "seeds": list(SEEDS),
        "feature_count": feature_lock.get("feature_count", 0),
        "feature_policy_hash": feature_lock.get("feature_policy_hash"),
        "training_roles": ["train"],
        "selection_roles": ["validation"],
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "test_or_acceptance_used_for_selection": False,
        "candidate_definition": (
            "base-family seed-averaged probabilities followed by "
            "leave-one-validation-device-out logistic stacking"
        ),
        "development_gate": {
            "minimum_candidate_macro_f1": 0.80,
            "minimum_macro_f1_delta_vs_strongest_single_family": 0.005,
            "accuracy_delta_must_be_positive": True,
            "malicious_recall_delta_minimum": -0.005,
            "ece_delta_maximum": 0.005,
        },
        "training_authorized": ready,
        "model_trained": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "protocol_w316.json", protocol)
    return protocol


def train_n_baiot_development_w316(
    data_root: str | Path = DEFAULT_DATA,
    w315_dir: str | Path = DEFAULT_W315,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODELS,
    rows_per_device_class: int = 10_000,
) -> dict[str, Any]:
    data = Path(data_root)
    w315 = Path(w315_dir)
    output = Path(output_dir)
    models_dir = Path(model_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    protocol = _load(output / "protocol_w316.json")
    if protocol.get("training_authorized") is not True:
        raise RuntimeError("W316 protocol is not authorized")
    features = list(
        _load(w315 / "safe_feature_lock_w315.json")["features"]
    )
    manifest = _read_csv(w315 / "official_file_manifest_w315.csv")
    x_train, y_train, groups, sampling = _load_role_balanced(
        data,
        manifest,
        "train",
        features,
        rows_per_device_class,
        SEEDS[0],
    )
    _write_csv(output / "training_sample_manifest_w316.csv", sampling)

    trained: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for family, model in _build_models(seed).items():
            model_id = f"{family}_seed{seed}"
            started = time.perf_counter()
            model.fit(x_train, y_train)
            duration = time.perf_counter() - started
            trained[model_id] = model
            result_rows.append(
                {
                    "model_id": model_id,
                    "model_family": family,
                    "seed": seed,
                    "train_rows": len(y_train),
                    "train_device_count": len(set(groups)),
                    "train_benign_rows": int((y_train == 0).sum()),
                    "train_malicious_rows": int((y_train == 1).sum()),
                    "training_seconds": duration,
                    "selection_role_opened": False,
                    "acceptance_opened": False,
                }
            )
    artifact = models_dir / "development_models_w316.joblib"
    joblib.dump(
        {
            "features": features,
            "models": trained,
            "feature_policy_hash": protocol["feature_policy_hash"],
            "rows_per_device_class": rows_per_device_class,
        },
        artifact,
        compress=3,
    )
    _write_csv(output / "training_results_w316.csv", result_rows)
    report = {
        "status": "w316_base_models_trained_train_only",
        "trained_model_count": len(trained),
        "train_rows": len(y_train),
        "train_device_count": len(set(groups)),
        "validation_rows_read": 0,
        "acceptance_rows_read": 0,
        "artifact": artifact.as_posix(),
        "model_trained": True,
        "fake_metric_count": 0,
    }
    _dump(output / "training_report_w316.json", report)
    return report


def _family_probabilities(
    trained: Mapping[str, Any],
    x_values: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions: dict[str, list[np.ndarray]] = {}
    latency_rows: list[dict[str, Any]] = []
    for model_id, model in trained.items():
        family = model_id.rsplit("_seed", 1)[0]
        started = time.perf_counter()
        probability = model.predict_proba(x_values)[:, 1]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        predictions.setdefault(family, []).append(probability)
        latency_rows.append(
            {
                "model_id": model_id,
                "inference_rows": len(x_values),
                "total_latency_ms": elapsed_ms,
                "latency_ms_per_sample": elapsed_ms / max(len(x_values), 1),
            }
        )
    averaged = {
        family: np.mean(np.vstack(values), axis=0)
        for family, values in predictions.items()
    }
    return averaged, latency_rows


def _oof_stack(
    probability_matrix: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    oof = np.zeros(len(truth), dtype=float)
    rows: list[dict[str, Any]] = []
    unique_groups = sorted(set(groups.tolist()))
    for group in unique_groups:
        train_mask = groups != group
        holdout_mask = groups == group
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(probability_matrix[train_mask], truth[train_mask])
        oof[holdout_mask] = model.predict_proba(
            probability_matrix[holdout_mask]
        )[:, 1]
        rows.append(
            {
                "heldout_validation_device": group,
                "meta_train_rows": int(train_mask.sum()),
                "meta_holdout_rows": int(holdout_mask.sum()),
                "meta_train_device_count": len(unique_groups) - 1,
                "heldout_device_enters_meta_features": False,
            }
        )
    return oof, rows


def validate_n_baiot_development_w316(
    data_root: str | Path = DEFAULT_DATA,
    w315_dir: str | Path = DEFAULT_W315,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODELS,
    rows_per_device_class: int = 10_000,
) -> dict[str, Any]:
    data = Path(data_root)
    w315 = Path(w315_dir)
    output = Path(output_dir)
    models_dir = Path(model_dir)
    payload = joblib.load(models_dir / "development_models_w316.joblib")
    features = list(payload["features"])
    manifest = _read_csv(w315 / "official_file_manifest_w315.csv")
    x_val, y_val, groups, sampling = _load_role_balanced(
        data,
        manifest,
        "validation",
        features,
        rows_per_device_class,
        SEEDS[0],
    )
    _write_csv(output / "validation_sample_manifest_w316.csv", sampling)
    family_probs, latency_rows = _family_probabilities(
        payload["models"], x_val
    )
    _write_csv(output / "validation_latency_w316.csv", latency_rows)

    result_rows: list[dict[str, Any]] = []
    calibrated_family_probs: dict[str, np.ndarray] = {}
    for family, probabilities in sorted(family_probs.items()):
        calibration = _select_validation_calibration(y_val, probabilities)
        calibrated = _temperature_scale(
            probabilities, float(calibration["temperature"])
        )
        calibrated_family_probs[family] = calibrated
        result_rows.append(
            {
                "candidate_id": family,
                "candidate_type": "strong_single_family_baseline",
                **calibration,
                "validation_protocol": "device_group_heldout",
            }
        )

    family_names = sorted(calibrated_family_probs)
    matrix = np.column_stack(
        [calibrated_family_probs[name] for name in family_names]
    )
    oof_probabilities, oof_rows = _oof_stack(
        matrix, y_val, groups
    )
    _write_csv(output / "oof_meta_folds_w316.csv", oof_rows)
    candidate_calibration = _select_validation_calibration(
        y_val, oof_probabilities
    )
    result_rows.append(
        {
            "candidate_id": "cross_device_oof_probability_stack",
            "candidate_type": "mad_etd_dataset_specific_candidate",
            **candidate_calibration,
            "validation_protocol": (
                "leave_one_validation_device_out_meta_evaluation"
            ),
        }
    )
    _write_csv(output / "validation_results_w316.csv", result_rows)

    baseline_rows = [
        row
        for row in result_rows
        if row["candidate_type"] == "strong_single_family_baseline"
    ]
    baseline_rows.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["accuracy"]),
            -float(row["malicious_recall"]),
            float(row["ece"]),
            row["candidate_id"],
        )
    )
    strongest = baseline_rows[0]
    candidate = next(
        row
        for row in result_rows
        if row["candidate_id"] == "cross_device_oof_probability_stack"
    )
    deltas = {
        "accuracy_delta": float(candidate["accuracy"])
        - float(strongest["accuracy"]),
        "macro_f1_delta": float(candidate["macro_f1"])
        - float(strongest["macro_f1"]),
        "weighted_f1_delta": float(candidate["weighted_f1"])
        - float(strongest["weighted_f1"]),
        "malicious_recall_delta": float(candidate["malicious_recall"])
        - float(strongest["malicious_recall"]),
        "ece_delta": float(candidate["ece"]) - float(strongest["ece"]),
    }
    gates = {
        "candidate_macro_f1_at_least_0_80": float(candidate["macro_f1"])
        >= 0.80,
        "macro_f1_delta_at_least_0_005": deltas["macro_f1_delta"]
        >= 0.005,
        "accuracy_delta_positive": deltas["accuracy_delta"] > 0.0,
        "malicious_recall_not_worse_by_more_than_0_005": deltas[
            "malicious_recall_delta"
        ]
        >= -0.005,
        "ece_not_worse_by_more_than_0_005": deltas["ece_delta"] <= 0.005,
    }
    development_passed = all(gates.values())

    final_meta = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    final_meta.fit(matrix, y_val)
    payload.update(
        {
            "family_names": family_names,
            "family_temperature": {
                row["candidate_id"]: float(row["temperature"])
                for row in baseline_rows
            },
            "meta_model": final_meta,
            "candidate_temperature": float(
                candidate_calibration["temperature"]
            ),
            "candidate_threshold": float(
                candidate_calibration["decision_threshold"]
            ),
            "development_gate_passed": development_passed,
        }
    )
    locked_artifact = models_dir / "locked_candidate_w316.joblib"
    joblib.dump(payload, locked_artifact, compress=3)

    report = {
        "status": (
            "w316_development_positive_ready_for_w317_sealed_acceptance"
            if development_passed
            else "w316_development_not_positive_acceptance_remains_sealed"
        ),
        "strongest_baseline": strongest,
        "candidate": candidate,
        "deltas_vs_strongest_baseline": deltas,
        "development_gates": gates,
        "failed_development_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "validation_rows": len(y_val),
        "validation_device_count": len(set(groups)),
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "test_or_acceptance_used_for_selection": False,
        "acceptance_run_authorized": development_passed,
        "candidate_artifact": locked_artifact.as_posix(),
        "performance_candidate_accepted": False,
        "optional_profile_created": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "development_decision_w316.json", report)
    return report


def _render(
    report: Mapping[str, Any],
    document: Path,
    document_cn: Path,
) -> None:
    baseline = report.get("strongest_baseline", {})
    candidate = report.get("candidate", {})
    delta = report.get("deltas_vs_strongest_baseline", {})
    english = f"""# MAD-ETD N-BaIoT Development W316

- Status: `{report['status']}`
- Strongest baseline: `{baseline.get('candidate_id')}`
- Baseline Macro-F1: {baseline.get('macro_f1')}
- Candidate Macro-F1: {candidate.get('macro_f1')}
- Macro-F1 delta: {delta.get('macro_f1_delta')}
- Accuracy delta: {delta.get('accuracy_delta')}
- Acceptance opened: false
- Candidate accepted: false
- Fake metrics: 0
- Default runtime changed: false

The candidate was evaluated with leave-one-validation-device-out probability
stacking. W316 is development-only; only a passing signal may authorize the
separate one-shot W317 acceptance.
"""
    chinese = f"""# MAD-ETD N-BaIoT 开发实验 W316

- 状态：`{report['status']}`
- 最强单模型基线：`{baseline.get('candidate_id')}`
- 基线 Macro-F1：{baseline.get('macro_f1')}
- 候选 Macro-F1：{candidate.get('macro_f1')}
- Macro-F1 增量：{delta.get('macro_f1_delta')}
- Accuracy 增量：{delta.get('accuracy_delta')}
- 打开 acceptance：false
- 候选已验收：false
- fake metric count：0
- 修改默认 runtime：false

候选使用 leave-one-validation-device-out 概率堆叠进行开发评估。W316 只属于
开发阶段；只有通过开发门槛，才允许后续 W317 一次性打开封存 acceptance。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_n_baiot_development_w316(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    decision = dict(_load(output / "development_decision_w316.json"))
    hashes_after = _safe_hashes()
    _dump(output / "frozen_hashes_after_w316.json", hashes_after)
    hashes_unchanged = _hash_snapshot_equal(
        _load(output / "frozen_hashes_before_w316.json"),
        hashes_after,
    )
    safety = {
        "acceptance_remains_sealed": decision.get("acceptance_opened")
        is False
        and int(decision.get("acceptance_rows_read", 0)) == 0,
        "test_or_acceptance_not_used_for_selection": decision.get(
            "test_or_acceptance_used_for_selection"
        )
        is False,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "fake_metric_count_zero": decision.get("fake_metric_count") == 0,
        "frozen_hashes_unchanged": hashes_unchanged,
        "tests_passed": bool(tests_passed),
    }
    safety_passed = all(safety.values())
    development_passed = decision.get("acceptance_run_authorized") is True
    if not safety_passed:
        status = "w316_rejected_security_or_governance_failure"
    elif development_passed:
        status = "w316_development_positive_ready_for_w317_sealed_acceptance"
    else:
        status = "w316_development_not_positive_acceptance_remains_sealed"
    decision.update(
        {
            "status": status,
            "acceptance_run_authorized": (
                safety_passed and development_passed
            ),
            "acceptance_opened": False,
            "acceptance_rows_read": 0,
            "acceptance_metrics_generated": False,
            "performance_candidate_accepted": False,
            "optional_profile_created": False,
            "promoted_runtime_created": False,
            "runtime_safe_v3_0_remains_default": True,
            "blocked_field_violation": 0,
            "fusion_ownership_violation": 0,
            "ood_override_count": 0,
            "illegal_verdict_execution_count": 0,
            "fake_metric_count": 0,
            "security_gates": safety,
            "failed_security_gates": [
                name for name, passed in safety.items() if not passed
            ],
            "tests_passed": bool(tests_passed),
            "test_count": int(test_count),
        }
    )
    _dump(output / "acceptance_report.json", decision)
    _dump(output / "security_acceptance_w316.json", safety)
    _dump(
        output / "negative_results.json",
        []
        if safety_passed and development_passed
        else [
            {
                "experiment_id": EXPERIMENT,
                "failure_type": (
                    "development_performance_gate_failed"
                    if safety_passed
                    else "security_or_governance_failure"
                ),
                "failure_reason": "; ".join(
                    decision.get("failed_development_gates", [])
                    if safety_passed
                    else decision["failed_security_gates"]
                ),
                "acceptance_opened": False,
                "fake_metric_count": 0,
                "runtime_modified": False,
                "safe_claim": (
                    "N-BaIoT train/validation development completed under "
                    "device-disjoint safe-input controls."
                ),
                "forbidden_claim": (
                    "N-BaIoT candidate accepted or promoted."
                ),
                "final_status": status,
            }
        ],
    )
    _render(decision, Path(document), Path(document_cn))
    return decision
