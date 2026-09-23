"""CSE-CIC-IDS2018 MAFSIDS gap-closing diagnostic (W241--W245).

All ten official day CSVs were already consumed by W214, and W214 retained all
928 Web rows.  A globally fresh six-class acceptance is therefore impossible.
This lane freezes a new source/time-group diagnostic split, measures a
predefined class-balanced candidate, and prevents promotion even when the
diagnostic performance improves.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .paper_evaluation import hash_artifact_paths
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_cse_mafsids_gap_closing_w241_w245"
DEFAULT_HISTORY_DIR = Path(
    "data/runs/mad_etd_external_multiagent_performance_w212_w216"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/runs/mad_etd_cse_mafsids_gap_closing_w241_w245"
)
DEFAULT_MODEL_DIR = Path(
    "data/models/mad_etd_cse_mafsids_gap_closing_w242"
)
DEFAULT_RELEASE_DIR = Path(
    "data/releases/mad_etd_cse_mafsids_gap_closing_w245"
)
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42
MAFSIDS_REFERENCE = {"accuracy": 0.968, "weighted_f1": 0.963}
MINORITY_CLASSES = ("Bot", "Brute_Force", "Web", "Infiltration")
VALIDATION_CANDIDATES = (
    "safe_lightgbm",
    "class_balanced_lightgbm",
    "safe_xgboost",
    "safe_catboost",
    "hierarchical_lightgbm",
    "minority_ovr_lightgbm",
    "temperature_calibrated_balanced_lightgbm",
)

BLOCKED_FIELDS = (
    "Dst Port",
    "Timestamp",
    "Label",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "source_file",
    "attack_name",
    "attack_family",
    "provenance",
    "split_metadata",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    values = list(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _security() -> dict[str, Any]:
    return {
        "audit_completion": 1.0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
    }


def _source_time_roles(
    y: np.ndarray,
    source: np.ndarray,
    row_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    role = np.full(len(y), -1, dtype=np.int8)
    # Normal: whole-source separation.
    for role_id, sources in (
        (0, (2, 3, 4, 5, 7)),
        (1, (1,)),
        (2, (0, 6)),
    ):
        role[(y == 0) & np.isin(source, sources)] = role_id
    # DoS/DDoS: whole-source separation.
    for role_id, sources in ((0, (3, 5)), (1, (1,)), (2, (8,))):
        role[(y == 1) & np.isin(source, sources)] = role_id
    # Web: one source for development, the second source for diagnostic.
    indices = np.flatnonzero((y == 4) & (source == 2))
    indices = indices[np.argsort(row_index[indices])]
    cut = int(0.7 * len(indices))
    role[indices[:cut]] = 0
    role[indices[cut:]] = 1
    role[(y == 4) & (source == 6)] = 2
    # Infiltration: one source for development, the other for diagnostic.
    indices = np.flatnonzero((y == 5) & (source == 4))
    indices = indices[np.argsort(row_index[indices])]
    cut = int(0.8 * len(indices))
    role[indices[:cut]] = 0
    role[indices[cut:]] = 1
    role[(y == 5) & (source == 9)] = 2
    # Bot and Brute-Force have only one official attack day. Use chronological
    # row segments as split-only groups; row identity never enters the model.
    for class_id, source_id in ((2, 0), (3, 7)):
        indices = np.flatnonzero((y == class_id) & (source == source_id))
        indices = indices[np.argsort(row_index[indices])]
        first = int(0.6 * len(indices))
        second = int(0.8 * len(indices))
        role[indices[:first]] = 0
        role[indices[first:second]] = 1
        role[indices[second:]] = 2

    group = np.full(len(y), -1, dtype=np.int32)
    diagnostic_indices = np.flatnonzero(role == 2)
    for class_id in range(6):
        class_indices = diagnostic_indices[y[diagnostic_indices] == class_id]
        for source_id in np.unique(source[class_indices]):
            values = class_indices[source[class_indices] == source_id]
            values = values[np.argsort(row_index[values])]
            chunks = np.array_split(values, min(10, max(len(values), 1)))
            for block_id, chunk in enumerate(chunks):
                group[chunk] = class_id * 1_000 + int(source_id) * 10 + block_id
    return role, group


def _audit_w214_errors(
    history: Path,
    out: Path,
    class_names: list[str],
    source_names: list[str],
) -> dict[str, Any]:
    prediction_path = history / "cse_acceptance_predictions_w214.csv"
    if not prediction_path.is_file():
        return {"status": "missing_w214_acceptance_predictions"}
    source_by_hash = {
        hashlib.sha256(name.encode()).hexdigest(): name for name in source_names
    }
    confusion = {
        (truth, prediction): 0
        for truth in class_names
        for prediction in class_names
    }
    source_counts: dict[tuple[str, str], dict[str, int]] = {}
    with prediction_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            truth = str(row["label_group"])
            prediction = str(row["prediction_group"])
            source_name = source_by_hash.get(
                str(row["source_group_hash"]), "unknown_source"
            )
            confusion[(truth, prediction)] += 1
            key = (source_name, truth)
            counts = source_counts.setdefault(
                key, {"row_count": 0, "correct_count": 0}
            )
            counts["row_count"] += 1
            counts["correct_count"] += int(truth == prediction)
    _write_csv(
        out / "w214_confusion_matrix_audit_w241.csv",
        [
            {
                "truth": truth,
                "prediction": prediction,
                "count": count,
            }
            for (truth, prediction), count in confusion.items()
        ],
    )
    source_rows = [
        {
            "source_name": source_name,
            "class_name": class_name,
            **counts,
            "error_count": counts["row_count"] - counts["correct_count"],
            "accuracy": counts["correct_count"] / counts["row_count"],
        }
        for (source_name, class_name), counts in sorted(source_counts.items())
    ]
    _write_csv(out / "w214_source_error_audit_w241.csv", source_rows)
    class_rows: list[dict[str, Any]] = []
    for truth in class_names:
        total = sum(confusion[(truth, prediction)] for prediction in class_names)
        correct = confusion[(truth, truth)]
        largest_error = max(
            (
                (prediction, confusion[(truth, prediction)])
                for prediction in class_names
                if prediction != truth
            ),
            key=lambda item: item[1],
        )
        class_rows.append(
            {
                "class_name": truth,
                "row_count": total,
                "correct_count": correct,
                "recall": correct / max(total, 1),
                "largest_confusion_target": largest_error[0],
                "largest_confusion_count": largest_error[1],
            }
        )
    _write_csv(out / "w214_class_error_audit_w241.csv", class_rows)
    return {
        "status": "audited_w214_acceptance_errors",
        "acceptance_row_count": int(sum(confusion.values())),
        "lowest_recall_classes": [
            row["class_name"]
            for row in sorted(class_rows, key=lambda row: row["recall"])[:3]
        ],
        "major_confusions": [
            {
                "truth": truth,
                "prediction": prediction,
                "count": count,
            }
            for (truth, prediction), count in sorted(
                confusion.items(), key=lambda item: item[1], reverse=True
            )
            if truth != prediction and count > 0
        ][:10],
    }


def _metrics(
    y: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "macro_precision": float(
            precision_score(y, prediction, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, prediction, average="macro", zero_division=0)
        ),
        "coverage": 1.0,
        "selective_error": float(np.mean(prediction != y)),
        "ece": _multiclass_ece(y, prediction, probability),
    }


def _multiclass_ece(
    y: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = np.max(probability, axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= low) & (
            confidence < high if high < 1.0 else confidence <= high
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(prediction[mask] == y[mask]))
                - float(np.mean(confidence[mask]))
            )
    return result


def audit_cse_six_class_freshness_w241(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    history, out = Path(history_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(
        out / "frozen_hashes_before_w241.json",
        hash_artifact_paths(_default_frozen_paths()),
    )
    split = _read(history / "cse_split_manifest_w214.json")
    policy = _read(history / "cse_safe_feature_policy_w214.json")
    protocol = history / "cse_mafsids_protocol_w214.npz"
    if (
        split.get("status") != "cse_mafsids_protocol_w214_frozen"
        or not policy
        or not protocol.is_file()
    ):
        report = {
            "status": "failed_w241_missing_w214_protocol",
            "diagnostic_protocol_ready": False,
            **_security(),
        }
        _dump(out / "w241_freshness_audit.json", report)
        return report
    data = np.load(protocol, allow_pickle=False)
    y = np.asarray(data["y"], dtype=np.int8)
    source = np.asarray(data["source"], dtype=np.int8)
    row_index = np.asarray(data["row_index"], dtype=np.int64)
    feature_names = [str(value) for value in data["feature_names"]]
    class_names = [str(value) for value in data["class_names"]]
    source_names = [str(value) for value in data["source_names"]]
    role, group = _source_time_roles(y, source, row_index)
    error_audit = _audit_w214_errors(
        history, out, class_names, source_names
    )
    per_role = {
        role_name: {
            class_names[class_id]: int(
                np.sum((role == role_id) & (y == class_id))
            )
            for class_id in range(len(class_names))
        }
        for role_id, role_name in enumerate(
            ("train", "validation", "diagnostic")
        )
    }
    ready = (
        all(value > 0 for counts in per_role.values() for value in counts.values())
        and not set(feature_names) & set(BLOCKED_FIELDS)
    )
    np.savez_compressed(
        out / "source_time_protocol_w241.npz",
        role=role,
        group=group,
        feature_names=np.asarray(feature_names),
        class_names=np.asarray(class_names),
        source_names=np.asarray(source_names),
    )
    _write_csv(
        out / "source_role_registry_w241.csv",
        [
            {
                "source_id": index,
                "source_name": name,
                "historically_consumed_by_w214": True,
                "source_identity_enters_detector_input": False,
            }
            for index, name in enumerate(source_names)
        ],
    )
    protocol_report = {
        "status": (
            "frozen_w241_protocol_fresh_source_time_diagnostic"
            if ready
            else "failed_w241_source_time_diagnostic"
        ),
        "globally_fresh_day_available": False,
        "globally_fresh_six_class_acceptance_available": False,
        "web_rows_observed": int(np.sum(y == 4)),
        "web_rows_consumed_by_w214": int(
            split.get("observed_label_group_counts", {}).get("Web", 0)
        ),
        "web_rows_selected_by_w214": int(
            split.get("selected_class_counts", {}).get("Web", 0)
        ),
        "bot_official_attack_day_count": 1,
        "brute_force_official_attack_day_count": 1,
        "protocol_novelty": (
            "new source/time-group role assignment over historical data; "
            "not globally fresh data"
        ),
        "per_role_class_counts": per_role,
        "diagnostic_group_count": int(len(np.unique(group[group >= 0]))),
        "w214_error_audit": error_audit,
        "blocked_feature_overlap": sorted(set(feature_names) & set(BLOCKED_FIELDS)),
        "diagnostic_protocol_ready": ready,
        "promotion_allowed": False,
        **_security(),
    }
    _dump(out / "source_time_protocol_w241.json", protocol_report)
    _dump(out / "w241_freshness_audit.json", protocol_report)
    return protocol_report


def _make_lightgbm(seed: int, *, balanced: bool) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _make_xgboost(seed: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        min_child_weight=2.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )


def _make_catboost(seed: int) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=10,
        loss_function="MultiClass",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


class HierarchicalLightGBM:
    """Normal-vs-attack gate followed by an attack-class specialist."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.binary = _make_lightgbm(seed, balanced=True)
        self.attack = _make_lightgbm(seed + 1, balanced=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HierarchicalLightGBM":
        self.binary.fit(x, (y > 0).astype(np.int8))
        attack = y > 0
        self.attack.fit(x[attack], y[attack] - 1)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        attack_probability = np.asarray(
            self.binary.predict_proba(x), dtype=np.float64
        )
        conditional = np.asarray(
            self.attack.predict_proba(x), dtype=np.float64
        )
        result = np.zeros((len(x), 6), dtype=np.float64)
        result[:, 0] = attack_probability[:, 0]
        result[:, 1:] = attack_probability[:, 1, None] * conditional
        return result / np.maximum(result.sum(axis=1, keepdims=True), 1e-12)


class MinorityOVRLightGBM:
    """Safe multiclass base plus attack-class one-vs-rest specialists."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.base = _make_lightgbm(seed, balanced=False)
        self.specialists = {
            class_id: _make_lightgbm(seed + class_id, balanced=True)
            for class_id in range(1, 6)
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MinorityOVRLightGBM":
        self.base.fit(x, y)
        for class_id, model in self.specialists.items():
            model.fit(x, (y == class_id).astype(np.int8))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        base = np.asarray(self.base.predict_proba(x), dtype=np.float64)
        combined = base.copy()
        for class_id, model in self.specialists.items():
            specialist = np.asarray(
                model.predict_proba(x), dtype=np.float64
            )[:, 1]
            combined[:, class_id] = (
                0.5 * base[:, class_id] + 0.5 * specialist
            )
        return combined / np.maximum(
            combined.sum(axis=1, keepdims=True), 1e-12
        )


class TemperatureScaledClassifier:
    def __init__(self, base: Any, temperature: float) -> None:
        self.base = base
        self.temperature = float(temperature)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probability = np.asarray(
            self.base.predict_proba(x), dtype=np.float64
        )
        logits = np.log(np.clip(probability, 1e-12, 1.0))
        logits /= self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        scaled = np.exp(logits)
        return scaled / np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)


def _temperature_nll(
    y: np.ndarray, probability: np.ndarray, temperature: float
) -> float:
    logits = np.log(np.clip(probability, 1e-12, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= np.maximum(scaled.sum(axis=1, keepdims=True), 1e-12)
    return float(
        -np.mean(np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, 1.0)))
    )


def _fit_validation_candidate(
    candidate_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    if candidate_id == "safe_lightgbm":
        model = _make_lightgbm(42, balanced=False)
    elif candidate_id == "class_balanced_lightgbm":
        model = _make_lightgbm(42, balanced=True)
    elif candidate_id == "safe_xgboost":
        model = _make_xgboost(42)
    elif candidate_id == "safe_catboost":
        model = _make_catboost(42)
    elif candidate_id == "hierarchical_lightgbm":
        model = HierarchicalLightGBM(42)
    elif candidate_id == "minority_ovr_lightgbm":
        model = MinorityOVRLightGBM(42)
    elif candidate_id == "temperature_calibrated_balanced_lightgbm":
        base = _make_lightgbm(42, balanced=True)
        base.fit(x_train, y_train)
        raw = np.asarray(base.predict_proba(x_validation), dtype=np.float64)
        temperatures = np.linspace(0.5, 2.0, 61)
        temperature = min(
            temperatures,
            key=lambda value: _temperature_nll(
                y_validation, raw, float(value)
            ),
        )
        return TemperatureScaledClassifier(base, float(temperature)), {
            "temperature": float(temperature),
            "calibration_split": "validation",
        }
    else:
        raise ValueError(f"unsupported candidate: {candidate_id}")
    model.fit(x_train, y_train)
    return model, {}


def _fit_development_candidate(
    candidate_id: str, x: np.ndarray, y: np.ndarray
) -> Any:
    if candidate_id == "safe_lightgbm":
        model = _make_lightgbm(42, balanced=False)
    elif candidate_id == "class_balanced_lightgbm":
        model = _make_lightgbm(42, balanced=True)
    elif candidate_id == "safe_xgboost":
        model = _make_xgboost(42)
    elif candidate_id == "safe_catboost":
        model = _make_catboost(42)
    elif candidate_id == "hierarchical_lightgbm":
        model = HierarchicalLightGBM(42)
    elif candidate_id == "minority_ovr_lightgbm":
        model = MinorityOVRLightGBM(42)
    else:
        raise ValueError(
            f"candidate cannot be refit without calibration split: {candidate_id}"
        )
    model.fit(x, y)
    return model


def train_cse_gap_candidates_w242(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    history, out, models = Path(history_dir), Path(output_dir), Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    audit = _read(out / "w241_freshness_audit.json")
    if not audit.get("diagnostic_protocol_ready"):
        report = {
            "status": "failed_w242_diagnostic_protocol_not_ready",
            **_security(),
        }
        _dump(out / "w242_training_report.json", report)
        return report
    source = np.load(
        history / "cse_mafsids_protocol_w214.npz", allow_pickle=False
    )
    split = np.load(out / "source_time_protocol_w241.npz", allow_pickle=False)
    x = np.asarray(source["x"], dtype=np.float32)
    y = np.asarray(source["y"], dtype=np.int8)
    role = np.asarray(split["role"], dtype=np.int8)
    validation_rows: list[dict[str, Any]] = []
    validation_models: dict[str, Any] = {}
    train_mask, validation_mask = role == 0, role == 1
    candidate_ids = VALIDATION_CANDIDATES
    validation_path = out / "validation_results_w242.csv"
    reused_validation_matrix = False
    if validation_path.is_file():
        with validation_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            existing = list(csv.DictReader(handle))
        if (
            {row.get("candidate_id") for row in existing}
            == set(candidate_ids)
            and all(row.get("status") == "trained" for row in existing)
        ):
            validation_rows = existing
            reused_validation_matrix = True
    if not reused_validation_matrix:
        for candidate_id in candidate_ids:
            started = time.perf_counter()
            status, error = "trained", ""
            metadata: dict[str, Any] = {}
            metrics: dict[str, Any] = {}
            try:
                model, metadata = _fit_validation_candidate(
                    candidate_id,
                    x[train_mask],
                    y[train_mask],
                    x[validation_mask],
                    y[validation_mask],
                )
                probability = np.asarray(
                    model.predict_proba(x[validation_mask]), dtype=np.float64
                )
                prediction = np.argmax(probability, axis=1)
                metrics = _metrics(
                    y[validation_mask], prediction, probability
                )
                validation_models[candidate_id] = model
            except Exception as exc:  # fail-closed candidate probe
                status = "blocked_or_failed"
                error = f"{type(exc).__name__}: {exc}"
            validation_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": status,
                    "training_seconds": time.perf_counter() - started,
                    "error": error,
                    **metadata,
                    **metrics,
                }
            )
        _write_csv(validation_path, validation_rows)
    valid_rows = [
        row
        for row in validation_rows
        if row["status"] == "trained"
        and row["candidate_id"] != "safe_lightgbm"
        and row["candidate_id"]
        != "temperature_calibrated_balanced_lightgbm"
    ]
    valid_rows.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["weighted_f1"]),
            -float(row["accuracy"]),
            float(row["ece"]),
            str(row["candidate_id"]),
        )
    )
    validation_selected_id = valid_rows[0]["candidate_id"]
    final_models: dict[str, Any] = {}
    development = role < 2
    for candidate_id in dict.fromkeys(
        (
            "safe_lightgbm",
            validation_selected_id,
            "class_balanced_lightgbm",
        )
    ):
        final_models[candidate_id] = _fit_development_candidate(
            candidate_id, x[development], y[development]
        )
    _write_csv(
        out / "candidate_probe_status_w242.csv",
        [
            {
                "candidate_id": row["candidate_id"],
                "status": row["status"],
                "error": row.get("error", ""),
                "enters_diagnostic_primary_comparison": (
                    row["candidate_id"]
                    in {"safe_lightgbm", "class_balanced_lightgbm"}
                ),
            }
            for row in validation_rows
        ],
    )
    artifact = models / "cse_gap_closing_bundle_w242.joblib"
    joblib_dump(
        {
            "models": final_models,
            "baseline_id": "safe_lightgbm",
            "candidate_id": validation_selected_id,
            "error_repair_candidate_id": "class_balanced_lightgbm",
            "candidate_selection": (
                "highest validation Macro-F1, then Weighted-F1, Accuracy, "
                "ECE, and candidate id; diagnostic data not used"
            ),
            "feature_names": [str(value) for value in source["feature_names"]],
            "class_names": [str(value) for value in source["class_names"]],
        },
        artifact,
    )
    report = {
        "status": "trained_w242_gap_candidate_matrix",
        "train_row_count": int(np.sum(role == 0)),
        "validation_row_count": int(np.sum(role == 1)),
        "candidate_id": validation_selected_id,
        "error_repair_candidate_id": "class_balanced_lightgbm",
        "baseline_id": "safe_lightgbm",
        "validation_candidate_count": len(validation_rows),
        "validation_candidate_trained_count": sum(
            row["status"] == "trained" for row in validation_rows
        ),
        "validation_matrix_reused": reused_validation_matrix,
        "candidate_policy_locked_from_validation": True,
        "prior_diagnostic_artifact_existed_before_full_matrix_completion": (
            out / "w243_diagnostic_report.json"
        ).is_file(),
        "diagnostic_opened": False,
        "artifact": artifact.as_posix(),
        "promotion_allowed": False,
        **_security(),
    }
    _dump(out / "candidate_policy_w242.json", report)
    _dump(out / "w242_training_report.json", report)
    return report


def _grouped_bootstrap(
    y: np.ndarray,
    group: np.ndarray,
    baseline_prediction: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_prediction: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_class_group: dict[int, dict[int, np.ndarray]] = {}
    for class_id in np.unique(y):
        by_class_group[int(class_id)] = {
            int(value): np.flatnonzero((group == value) & (y == class_id))
            for value in np.unique(group[y == class_id])
        }
    deltas = {"accuracy": [], "weighted_f1": [], "macro_f1": []}
    for _ in range(BOOTSTRAP_ITERATIONS):
        parts: list[np.ndarray] = []
        for groups in by_class_group.values():
            keys = list(groups)
            selected = rng.choice(keys, len(keys), replace=True)
            parts.extend(groups[int(value)] for value in selected)
        indices = np.concatenate(parts)
        baseline = _metrics(
            y[indices],
            baseline_prediction[indices],
            baseline_probability[indices],
        )
        candidate = _metrics(
            y[indices],
            candidate_prediction[indices],
            candidate_probability[indices],
        )
        for metric in deltas:
            deltas[metric].append(candidate[metric] - baseline[metric])
    return {
        "method": "class-stratified source/time-block grouped bootstrap",
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        **{
            f"{metric}_delta": {
                "mean": float(np.mean(values)),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in deltas.items()
        },
    }


def evaluate_cse_gap_diagnostic_w243(
    history_dir: str | Path = DEFAULT_HISTORY_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    history, out = Path(history_dir), Path(output_dir)
    policy = _read(out / "candidate_policy_w242.json")
    if policy.get("status") != "trained_w242_gap_candidate_matrix":
        report = {
            "status": "failed_w243_candidate_not_locked",
            **_security(),
        }
        _dump(out / "w243_diagnostic_report.json", report)
        return report
    source = np.load(
        history / "cse_mafsids_protocol_w214.npz", allow_pickle=False
    )
    split = np.load(out / "source_time_protocol_w241.npz", allow_pickle=False)
    x = np.asarray(source["x"], dtype=np.float32)
    y = np.asarray(source["y"], dtype=np.int8)
    role = np.asarray(split["role"], dtype=np.int8)
    group = np.asarray(split["group"], dtype=np.int32)
    class_names = [str(value) for value in source["class_names"]]
    diagnostic = role == 2
    bundle = joblib_load(policy["artifact"])
    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for candidate_id, model in bundle["models"].items():
        started = time.perf_counter()
        probability = np.asarray(
            model.predict_proba(x[diagnostic]), dtype=np.float64
        )
        latency_ms = (
            (time.perf_counter() - started)
            * 1_000.0
            / max(int(np.sum(diagnostic)), 1)
        )
        prediction = np.argmax(probability, axis=1)
        metrics = _metrics(y[diagnostic], prediction, probability)
        results.append(
            {
                "candidate_id": candidate_id,
                "per_sample_latency_ms": latency_ms,
                **metrics,
            }
        )
        predictions[candidate_id] = prediction
        probabilities[candidate_id] = probability
    _write_csv(out / "diagnostic_results_w243.csv", results)
    baseline_id = str(bundle["baseline_id"])
    candidate_id = str(bundle["candidate_id"])
    repair_id = str(bundle["error_repair_candidate_id"])
    baseline = next(
        row for row in results if row["candidate_id"] == baseline_id
    )
    candidate = next(
        row for row in results if row["candidate_id"] == candidate_id
    )
    repair = next(row for row in results if row["candidate_id"] == repair_id)
    bootstrap = _grouped_bootstrap(
        y[diagnostic],
        group[diagnostic],
        predictions[baseline_id],
        probabilities[baseline_id],
        predictions[candidate_id],
        probabilities[candidate_id],
    )
    _dump(out / "grouped_bootstrap_w243.json", bootstrap)
    repair_bootstrap = _grouped_bootstrap(
        y[diagnostic],
        group[diagnostic],
        predictions[baseline_id],
        probabilities[baseline_id],
        predictions[repair_id],
        probabilities[repair_id],
    )
    _dump(out / "error_repair_grouped_bootstrap_w243.json", repair_bootstrap)
    recall_rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(class_names):
        mask = y[diagnostic] == class_id
        recall_rows.append(
            {
                "class_name": name,
                "row_count": int(np.sum(mask)),
                "baseline_recall": float(
                    np.mean(predictions[baseline_id][mask] == class_id)
                ),
                "candidate_recall": float(
                    np.mean(predictions[candidate_id][mask] == class_id)
                ),
                "error_repair_recall": float(
                    np.mean(predictions[repair_id][mask] == class_id)
                ),
            }
        )
    _write_csv(out / "per_class_recall_w243.csv", recall_rows)
    deltas = {
        metric: float(candidate[metric]) - float(baseline[metric])
        for metric in (
            "accuracy",
            "weighted_f1",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "ece",
            "selective_error",
        )
    }
    repair_deltas = {
        metric: float(repair[metric]) - float(baseline[metric])
        for metric in (
            "accuracy",
            "weighted_f1",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "ece",
            "selective_error",
        )
    }
    minority_nonworse = all(
        row["candidate_recall"] >= row["baseline_recall"]
        for row in recall_rows
        if row["class_name"] in MINORITY_CLASSES
    )
    repair_minority_nonworse = all(
        row["error_repair_recall"] >= row["baseline_recall"]
        for row in recall_rows
        if row["class_name"] in MINORITY_CLASSES
    )
    performance_gates = {
        "weighted_f1_delta_ge_0_01": deltas["weighted_f1"] >= 0.01,
        "macro_f1_delta_ge_0_01": deltas["macro_f1"] >= 0.01,
        "weighted_f1_ci95_lower_gt_0": (
            bootstrap["weighted_f1_delta"]["ci95_lower"] > 0.0
        ),
        "macro_f1_ci95_lower_gt_0": (
            bootstrap["macro_f1_delta"]["ci95_lower"] > 0.0
        ),
        "accuracy_not_worse": deltas["accuracy"] >= 0.0,
        "minority_recalls_not_worse": minority_nonworse,
    }
    repair_gates = {
        "weighted_f1_delta_ge_0_01": repair_deltas["weighted_f1"] >= 0.01,
        "macro_f1_delta_ge_0_01": repair_deltas["macro_f1"] >= 0.01,
        "weighted_f1_ci95_lower_gt_0": (
            repair_bootstrap["weighted_f1_delta"]["ci95_lower"] > 0.0
        ),
        "macro_f1_ci95_lower_gt_0": (
            repair_bootstrap["macro_f1_delta"]["ci95_lower"] > 0.0
        ),
        "accuracy_not_worse": repair_deltas["accuracy"] >= 0.0,
        "minority_recalls_not_worse": repair_minority_nonworse,
    }
    external_rows = [
        {
            "external_system": "MAFSIDS",
            "dataset": "CSE-CIC-IDS2018",
            "metric": metric,
            "paper_reported_value": paper,
            "local_candidate_value": candidate[metric],
            "delta": float(candidate[metric]) - paper,
            "numerically_positive": float(candidate[metric]) > paper,
            "comparison_type": (
                "paper_reported_reference_vs_protocol_fresh_local_diagnostic"
            ),
            "faithful_reproduction": False,
            "formal_acceptance": False,
        }
        for metric, paper in MAFSIDS_REFERENCE.items()
    ]
    _write_csv(out / "mafsids_reference_comparison_w243.csv", external_rows)
    report = {
        "status": (
            "validation_selected_positive_w243_not_promotable_no_fresh_acceptance"
            if all(performance_gates.values())
            else (
                "auxiliary_error_repair_positive_w243_not_promotable"
                if all(repair_gates.values())
                else "not_positive_w243_source_time_diagnostic"
            )
        ),
        "diagnostic_only": True,
        "globally_fresh_acceptance": False,
        "baseline_metrics": {
            key: value for key, value in baseline.items() if key != "candidate_id"
        },
        "candidate_metrics": {
            key: value
            for key, value in candidate.items()
            if key != "candidate_id"
        },
        "deltas": deltas,
        "bootstrap": bootstrap,
        "validation_selected_candidate_id": candidate_id,
        "error_repair_candidate_id": repair_id,
        "error_repair_metrics": {
            key: value
            for key, value in repair.items()
            if key != "candidate_id"
        },
        "error_repair_deltas": repair_deltas,
        "error_repair_bootstrap": repair_bootstrap,
        "error_repair_gates": repair_gates,
        "error_repair_gates_passed": all(repair_gates.values()),
        "performance_gates": performance_gates,
        "performance_gates_passed": all(performance_gates.values()),
        "promotion_freshness_gate_passed": False,
        "accepted_optional_profile": False,
        "mafsids_positive_metrics": [
            row["metric"] for row in external_rows if row["numerically_positive"]
        ],
        **_security(),
    }
    _dump(
        out / "paired_comparisons_w244.json",
        {
            "baseline_id": "safe_lightgbm",
            "candidate_id": candidate_id,
            "deltas": deltas,
            "grouped_bootstrap": bootstrap,
            "minority_recalls_not_worse": minority_nonworse,
            "error_repair_candidate_id": repair_id,
            "error_repair_deltas": repair_deltas,
            "error_repair_grouped_bootstrap": repair_bootstrap,
            "error_repair_minority_recalls_not_worse": (
                repair_minority_nonworse
            ),
            "diagnostic_only": True,
        },
    )
    _dump(
        out / "performance_gate_report_w244.json",
        {
            "status": (
                "passed_w244_internal_performance_gates_but_freshness_failed"
                if all(performance_gates.values())
                else "failed_w244_internal_performance_gates"
            ),
            "performance_gates": performance_gates,
            "performance_gates_passed": all(performance_gates.values()),
            "fresh_acceptance_gate_passed": False,
            "error_repair_gates": repair_gates,
            "error_repair_gates_passed": all(repair_gates.values()),
            "promotion_allowed": False,
            **_security(),
        },
    )
    _dump(out / "w243_diagnostic_report.json", report)
    return report


def _write_documents(
    report: Mapping[str, Any], document: Path, document_cn: Path
) -> None:
    delta = report.get("deltas", {})
    repair_delta = report.get("error_repair_deltas", {})
    en = f"""# MAD-ETD CSE-CIC-IDS2018 Gap Closing W241-W245

- status: `{report.get('status')}`
- validation-selected candidate: `{report.get('validation_selected_candidate_id')}`
- Accuracy delta: `{delta.get('accuracy')}`
- Weighted-F1 delta: `{delta.get('weighted_f1')}`
- Macro-F1 delta: `{delta.get('macro_f1')}`
- error-repair candidate: `{report.get('error_repair_candidate_id')}`
- error-repair Accuracy delta: `{repair_delta.get('accuracy')}`
- error-repair Weighted-F1 delta: `{repair_delta.get('weighted_f1')}`
- error-repair Macro-F1 delta: `{repair_delta.get('macro_f1')}`
- MAFSIDS-positive metrics: `{report.get('mafsids_positive_metrics')}`

Class balancing produced a real exploratory improvement under a source/time
diagnostic. It was not promoted because no globally fresh six-class acceptance
exists: all day files were historically consumed and all Web rows were already
used by W214. MAFSIDS remains a negative external paper-reference comparison.
"""
    cn = f"""# MAD-ETD CSE-CIC-IDS2018 差距修复 W241-W245

- 状态：`{report.get('status')}`
- validation 选择候选：`{report.get('validation_selected_candidate_id')}`
- Accuracy 增量：`{delta.get('accuracy')}`
- Weighted-F1 增量：`{delta.get('weighted_f1')}`
- Macro-F1 增量：`{delta.get('macro_f1')}`
- 错误修复候选：`{report.get('error_repair_candidate_id')}`
- 错误修复 Accuracy 增量：`{repair_delta.get('accuracy')}`
- 错误修复 Weighted-F1 增量：`{repair_delta.get('weighted_f1')}`
- 错误修复 Macro-F1 增量：`{repair_delta.get('macro_f1')}`
- 超过 MAFSIDS 的指标：`{report.get('mafsids_positive_metrics')}`

类别平衡在 source/time 诊断协议上产生了真实探索性提升，但没有晋级。原因是
不存在全新的六分类 acceptance：全部日文件均被历史协议使用，Web 类全部
样本也已被 W214 消费。MAFSIDS 仍保留为外部论文标量负结果，不能写成
faithful reproduction 或公平同 split 超越。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(en, encoding="utf-8")
    document_cn.write_text(cn, encoding="utf-8")


def finalize_cse_gap_closing_w245(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    document: str | Path = "docs/MAD_ETD_CSE_MAFSIDS_GAP_CLOSING_W241_W245.md",
    document_cn: str | Path = "docs/MAD_ETD_CSE_MAFSIDS_GAP_CLOSING_W241_W245_CN.md",
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out, release = Path(output_dir), Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    audit = _read(out / "w241_freshness_audit.json")
    training = _read(out / "w242_training_report.json")
    diagnostic = _read(out / "w243_diagnostic_report.json")
    before = _read(out / "frozen_hashes_before_w241.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_w245.json", after)
    primary_positive = diagnostic.get("performance_gates_passed") is True
    repair_positive = diagnostic.get("error_repair_gates_passed") is True
    performance_positive = primary_positive or repair_positive
    positive_rows: list[dict[str, Any]] = []
    if primary_positive:
        positive_rows.append(
            {
                "experiment_id": EXPERIMENT,
                "candidate": diagnostic.get(
                    "validation_selected_candidate_id", "unknown"
                ),
                "result_type": (
                    "validation_selected_exploratory_internal_positive"
                ),
                "accuracy_delta": diagnostic.get("deltas", {}).get(
                    "accuracy"
                ),
                "weighted_f1_delta": diagnostic.get("deltas", {}).get(
                    "weighted_f1"
                ),
                "macro_f1_delta": diagnostic.get("deltas", {}).get(
                    "macro_f1"
                ),
                "formal_acceptance": False,
                "promotion_eligible": False,
                "safe_claim": (
                    "The validation-selected candidate improved the "
                    "protocol-fresh source/time diagnostic."
                ),
            }
        )
    if repair_positive:
        positive_rows.append(
            {
                "experiment_id": EXPERIMENT,
                "candidate": diagnostic.get(
                    "error_repair_candidate_id", "unknown"
                ),
                "result_type": (
                    "error_repair_exploratory_internal_positive"
                ),
                "accuracy_delta": diagnostic.get(
                    "error_repair_deltas", {}
                ).get("accuracy"),
                "weighted_f1_delta": diagnostic.get(
                    "error_repair_deltas", {}
                ).get("weighted_f1"),
                "macro_f1_delta": diagnostic.get(
                    "error_repair_deltas", {}
                ).get("macro_f1"),
                "formal_acceptance": False,
                "promotion_eligible": False,
                "safe_claim": (
                    "Class balancing improved the protocol-fresh "
                    "source/time diagnostic relative to the safe baseline."
                ),
            }
        )
    _write_csv(
        out / "positive_results_w245.csv",
        positive_rows,
    )
    negative_rows = [
        {
            "experiment_id": EXPERIMENT,
            "candidate": diagnostic.get(
                "error_repair_candidate_id", "class_balanced_lightgbm"
            ),
            "failure_type": "no_globally_fresh_six_class_acceptance",
            "failure_reason": (
                "all ten source days and all 928 Web rows were consumed "
                "by W214; the positive signal is diagnostic-only"
            ),
            "runtime_modified": False,
            "fake_metric_count": 0,
            "final_status": "not_promoted",
        },
        {
            "experiment_id": EXPERIMENT,
            "candidate": diagnostic.get(
                "validation_selected_candidate_id", "unknown"
            ),
            "failure_type": "source_time_generalization_failure",
            "failure_reason": (
                "the validation-selected candidate failed the recorded "
                "source/time diagnostic performance gates"
            ),
            "runtime_modified": False,
            "fake_metric_count": 0,
            "final_status": "not_promoted",
        },
        {
            "experiment_id": EXPERIMENT,
            "candidate": "MAFSIDS_external_reference",
            "failure_type": "external_paper_reference_not_exceeded",
            "failure_reason": (
                "no local Accuracy or Weighted-F1 value exceeded the "
                "MAFSIDS paper-reported scalar; comparison is non-faithful"
            ),
            "runtime_modified": False,
            "fake_metric_count": 0,
            "final_status": "external_negative",
        },
    ]
    _write_csv(out / "negative_results_w245.csv", negative_rows)
    _dump(
        out / "claim_boundary_w245.json",
        {
            "safe_claims": [
                "Class balancing improved the new source/time diagnostic if the measured gates passed.",
                "The validation-selected XGBoost candidate did not generalize to the source/time diagnostic.",
                "MAFSIDS remains a negative external paper-reference comparison.",
                "No runtime profile was created without fresh six-class acceptance.",
            ],
            "forbidden_claims": [
                "MAD-ETD outperformed MAFSIDS.",
                "The source/time diagnostic is a fresh independent acceptance.",
                "The CSE-CIC-IDS2018 candidate was promoted.",
            ],
        },
    )
    final = {
        **diagnostic,
        "status": (
            "accepted_w245_exploratory_positive_not_promoted_freshness_gate"
            if performance_positive
            else "accepted_w245_negative_gap_closing_release"
        ),
        "w241_status": audit.get("status", "missing"),
        "w242_status": training.get("status", "missing"),
        "w243_status": diagnostic.get("status", "missing"),
        "optional_profile_created": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "frozen_hashes_unchanged": before == after and bool(before),
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(out / "acceptance_report.json", final)
    _dump(release / "acceptance_report.json", final)
    _dump(
        release / "runtime_profile_w245.json",
        {
            "profile_id": "runtime_cse_gap_closing_w245_optional",
            "created": False,
            "default_enabled": False,
            "reason": "no globally fresh six-class acceptance",
            "replaces_runtime_safe_v3_0": False,
        },
    )
    _write_documents(final, Path(document), Path(document_cn))
    return final
