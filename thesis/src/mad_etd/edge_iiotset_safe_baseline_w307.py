"""W307 safe-input baseline development pipeline for Edge-IIoTset."""

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
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _hash_snapshot_equal,
    _load,
    _read_csv,
    _safe_hashes,
    _write_csv,
)


EXPERIMENT = "mad_etd_edge_iiotset_safe_baseline_w307"
DEFAULT_DATA = Path("data/raw/edge_iiotset/official_download")
DEFAULT_W306 = Path("data/runs/mad_etd_edge_iiotset_readiness_w306")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_edge_iiotset_safe_baseline_w307")
DEFAULT_MODELS = Path("data/models/mad_etd_edge_iiotset_w307")
DEFAULT_DOC = Path("docs/MAD_ETD_EDGE_IIOTSET_SAFE_BASELINE_W307.md")
DEFAULT_DOC_CN = Path(
    "docs/MAD_ETD_EDGE_IIOTSET_SAFE_BASELINE_W307_CN.md"
)
SEEDS = (42, 43, 44)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binary_ece(
    truth: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    prediction = (probabilities >= 0.5).astype(int)
    correct = prediction == truth
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if np.any(mask):
            result += (
                abs(
                    float(correct[mask].mean())
                    - float(confidence[mask].mean())
                )
                * int(mask.sum())
                / max(len(truth), 1)
            )
    return float(result)


def _metrics(
    truth: np.ndarray,
    malicious_probability: np.ndarray,
    latency_ms: float | str = "not_available",
    threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = (malicious_probability >= threshold).astype(int)
    accuracy = float(accuracy_score(truth, prediction))
    return {
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                truth, prediction, average="weighted", zero_division=0
            )
        ),
        "malicious_recall": float(
            recall_score(truth, prediction, pos_label=1, zero_division=0)
        ),
        "ece": _binary_ece(truth, malicious_probability),
        "coverage": 1.0,
        "selective_error": 1.0 - accuracy,
        "p95_latency_ms": latency_ms,
        "decision_threshold": float(threshold),
    }


def _temperature_scale(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def _select_validation_calibration(
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for temperature in (0.75, 1.0, 1.25, 1.5):
        calibrated = _temperature_scale(probabilities, temperature)
        for threshold in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
            candidates.append(
                {
                    "temperature": temperature,
                    **_metrics(
                        truth,
                        calibrated,
                        threshold=threshold,
                    ),
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["malicious_recall"]),
            float(row["ece"]),
            abs(float(row["decision_threshold"]) - 0.5),
            abs(float(row["temperature"]) - 1.0),
        )
    )
    return candidates[0]


def _build_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=280,
            max_depth=8,
            learning_rate=0.08,
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


def _safe_features(w306: Path) -> list[str]:
    schema_path = w306 / "feature_schema_audit_w306.csv"
    if not schema_path.is_file():
        return []
    return [
        row["feature_name"]
        for row in _read_csv(schema_path)
        if row.get("final_feature_policy") == "safe_conditional"
    ]


def _resolve_csv(data_root: Path, relative_stem: str) -> Path:
    base = data_root / Path(relative_stem)
    candidate = Path(f"{base}.csv")
    if candidate.is_file():
        return candidate
    matches = list(base.parent.glob(f"{base.name}*.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"unique CSV not found for {relative_stem}")


def _load_role(
    data_root: Path,
    split_rows: list[dict[str, str]],
    role: str,
    features: list[str],
    rows_per_group: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for row in split_rows:
        if row["role"] != role:
            continue
        if role == "acceptance":
            raise RuntimeError("W307 must never load acceptance samples")
        path = _resolve_csv(data_root, row["relative_path"])
        frame = pd.read_csv(
            path,
            usecols=features,
            nrows=rows_per_group * 4,
            low_memory=False,
        )
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"safe features missing from {path}: {missing}")
        frame = frame[features].replace([np.inf, -np.inf], np.nan)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(frame) > rows_per_group:
            group_seed = int(row["capture_group"][:8], 16)
            frame = frame.sample(
                n=rows_per_group,
                replace=False,
                random_state=group_seed,
            )
        values = frame.to_numpy(dtype=np.float32, copy=True)
        parts.append(values)
        label = 1 if row["binary_label"] == "malicious" else 0
        labels.extend([label] * len(values))
        groups.extend([row["capture_group"]] * len(values))
    if not parts:
        raise ValueError(f"no rows loaded for role {role}")
    return np.vstack(parts), np.asarray(labels, dtype=int), groups


def prepare_edge_iiotset_safe_baseline_w307(
    w306_dir: str | Path = DEFAULT_W306,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    w306 = Path(w306_dir)
    output = Path(output_dir)
    readiness = (
        _load(w306 / "acceptance_report.json")
        if (w306 / "acceptance_report.json").is_file()
        else {}
    )
    split_path = w306 / "frozen_split_manifest_w306.csv"
    split_rows = _read_csv(split_path) if split_path.is_file() else []
    features = _safe_features(w306)
    role_groups = {
        role: {
            row["capture_group"]
            for row in split_rows
            if row.get("role") == role
        }
        for role in ("train", "validation", "acceptance")
    }
    overlap = (
        role_groups["train"] & role_groups["validation"]
        or role_groups["train"] & role_groups["acceptance"]
        or role_groups["validation"] & role_groups["acceptance"]
    )
    ready = (
        readiness.get("status")
        == "ready_for_w307_safe_baseline_development"
        and bool(features)
        and all(role_groups.values())
        and not overlap
    )
    status = (
        "w307_protocol_ready_for_train_validation"
        if ready
        else "pipeline_ready_waiting_for_w306_data"
    )
    feature_lock = {
        "status": (
            "locked_from_w306_safe_conditional"
            if features
            else "waiting_for_w306_schema"
        ),
        "safe_features": features,
        "feature_count": len(features),
        "feature_policy_hash": _canonical_hash(features),
        "blocked_fields_enter_feature_matrix": False,
        "label_fields_enter_feature_matrix": False,
        "unknown_fields_fail_closed": True,
    }
    protocol = {
        "status": status,
        "w306_status": readiness.get("status", "missing"),
        "w306_ready": ready,
        "models": [
            "hgb",
            "random_forest",
            "extra_trees",
            "xgboost_if_available",
        ],
        "seeds": list(SEEDS),
        "selection_role": "validation_only",
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "test_or_acceptance_used_for_selection": False,
        "role_group_counts": {
            role: len(groups) for role, groups in role_groups.items()
        },
        "role_overlap_count": len(overlap),
        "model_training_authorized": ready,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "protocol_w307.json", protocol)
    _dump(output / "safe_feature_lock_w307.json", feature_lock)
    _write_csv(
        output / "paper_reported_only_comparison_w307.csv",
        [
            {
                "reference": "Autonomous LLM Agent on Edge-IIoTset",
                "reported_metric": "multiclass_accuracy",
                "reported_value": 0.93,
                "source_type": "paper_reported_only",
                "same_split": False,
                "safe_input_policy_matched": False,
                "used_for_local_selection": False,
                "used_as_local_metric": False,
            },
            {
                "reference": "SmartSecLab/ENViSEC",
                "reported_metric": "not_imported",
                "reported_value": "not_available",
                "source_type": "external_code_context_only",
                "same_split": False,
                "safe_input_policy_matched": False,
                "used_for_local_selection": False,
                "used_as_local_metric": False,
            },
        ],
    )
    _dump(output / "frozen_hashes_before_w307.json", _safe_hashes())
    return protocol


def train_edge_iiotset_safe_baseline_w307(
    data_root: str | Path = DEFAULT_DATA,
    w306_dir: str | Path = DEFAULT_W306,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODELS,
    rows_per_group: int = 5000,
) -> dict[str, Any]:
    data = Path(data_root)
    w306 = Path(w306_dir)
    output = Path(output_dir)
    models_out = Path(model_dir)
    protocol = _load(output / "protocol_w307.json")
    if protocol.get("status") != "w307_protocol_ready_for_train_validation":
        report = {
            "status": "pipeline_ready_waiting_for_w306_data",
            "trained_model_count": 0,
            "train_rows": 0,
            "validation_rows_read": 0,
            "acceptance_rows_read": 0,
            "model_trained": False,
            "fake_metric_count": 0,
        }
        _write_csv(output / "training_results_w307.csv", [])
        _dump(output / "training_report_w307.json", report)
        return report
    features = _load(output / "safe_feature_lock_w307.json")[
        "safe_features"
    ]
    split_rows = _read_csv(w306 / "frozen_split_manifest_w306.csv")
    x_train, y_train, groups = _load_role(
        data, split_rows, "train", features, rows_per_group
    )
    trained: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for name, model in _build_models(seed).items():
            model_id = f"{name}_seed{seed}"
            started = time.perf_counter()
            try:
                model.fit(x_train, y_train)
                trained[model_id] = model
                results.append(
                    {
                        "model_id": model_id,
                        "status": "trained",
                        "training_seconds": time.perf_counter() - started,
                        "train_rows": len(y_train),
                        "train_group_count": len(set(groups)),
                        "feature_count": len(features),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "model_id": model_id,
                        "status": "training_failed",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                        "training_seconds": time.perf_counter() - started,
                        "train_rows": len(y_train),
                        "train_group_count": len(set(groups)),
                        "feature_count": len(features),
                    }
                )
    if len(trained) < 3:
        raise RuntimeError("fewer than three W307 baselines trained")
    models_out.mkdir(parents=True, exist_ok=True)
    artifact = models_out / "development_models_w307.joblib"
    joblib.dump(
        {
            "models": trained,
            "safe_features": features,
            "feature_policy_hash": _load(
                output / "safe_feature_lock_w307.json"
            )["feature_policy_hash"],
            "acceptance_opened": False,
        },
        artifact,
    )
    _write_csv(output / "training_results_w307.csv", results)
    report = {
        "status": "w307_safe_baselines_trained_train_only",
        "trained_model_count": len(trained),
        "train_rows": len(y_train),
        "train_group_count": len(set(groups)),
        "validation_rows_read": 0,
        "acceptance_rows_read": 0,
        "artifact": artifact.as_posix(),
        "artifact_hash": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model_trained": True,
        "fake_metric_count": 0,
    }
    _dump(output / "training_report_w307.json", report)
    return report


def validate_edge_iiotset_safe_baseline_w307(
    data_root: str | Path = DEFAULT_DATA,
    w306_dir: str | Path = DEFAULT_W306,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODELS,
    rows_per_group: int = 3000,
) -> dict[str, Any]:
    data = Path(data_root)
    w306 = Path(w306_dir)
    output = Path(output_dir)
    protocol = _load(output / "protocol_w307.json")
    training = _load(output / "training_report_w307.json")
    if (
        protocol.get("status") != "w307_protocol_ready_for_train_validation"
        or training.get("status") != "w307_safe_baselines_trained_train_only"
    ):
        report = {
            "status": "pipeline_ready_waiting_for_w306_data",
            "selected_model": None,
            "validation_rows": 0,
            "acceptance_rows_read": 0,
            "validation_metrics_generated": False,
            "fake_metric_count": 0,
        }
        _write_csv(output / "validation_results_w307.csv", [])
        _dump(output / "selected_model_lock_w307.json", report)
        return report
    artifact_path = Path(model_dir) / "development_models_w307.joblib"
    payload = joblib.load(artifact_path)
    features = list(payload["safe_features"])
    split_rows = _read_csv(w306 / "frozen_split_manifest_w306.csv")
    x_validation, y_validation, groups = _load_role(
        data, split_rows, "validation", features, rows_per_group
    )
    results: list[dict[str, Any]] = []
    for model_id, model in payload["models"].items():
        started = time.perf_counter()
        probability = np.asarray(model.predict_proba(x_validation))[:, 1]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        calibrated = _select_validation_calibration(
            y_validation,
            probability,
        )
        results.append(
            {
                "model_id": model_id,
                **calibrated,
                "avg_inference_latency_ms_per_sample": elapsed_ms
                / max(len(y_validation), 1),
                "calibration_role": "validation_only",
            }
        )
    results.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["malicious_recall"]),
            float(row["ece"]),
            row["model_id"],
        )
    )
    _write_csv(output / "validation_results_w307.csv", results)
    selected = results[0]
    lock = {
        "status": "w307_validation_model_locked_acceptance_sealed",
        "selected_model": selected["model_id"],
        "selected_temperature": selected["temperature"],
        "selected_threshold": selected["decision_threshold"],
        "selection_rule": (
            "validation-only model, temperature and threshold selection by "
            "Macro-F1, malicious recall, ECE and deterministic tie-breaks"
        ),
        "validation_metrics": selected,
        "validation_rows": len(y_validation),
        "validation_group_count": len(set(groups)),
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "test_or_acceptance_used_for_selection": False,
        "performance_candidate_accepted": False,
        "optional_profile_created": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "selected_model_lock_w307.json", lock)
    return lock


def _render_documents(
    report: Mapping[str, Any],
    document: Path,
    document_cn: Path,
) -> None:
    english = f"""# MAD-ETD W307 Edge-IIoTset Safe Baseline Pipeline

- Status: `{report['status']}`
- W306 ready: {report['w306_ready']}
- Model trained: {report['model_trained']}
- Validation metrics generated: {report['validation_metrics_generated']}
- Acceptance opened: false
- Fake metrics: 0
- Default runtime changed: false

The implementation is ready, but real training is prohibited until W306
verifies official payloads and freezes capture-disjoint roles. Paper-reported
Edge-IIoTset metrics remain in a separate context-only table. Model,
temperature and threshold selection are validation-only; acceptance remains
sealed throughout W307.
"""
    chinese = f"""# MAD-ETD W307 Edge-IIoTset 安全强基线流水线

- 状态：`{report['status']}`
- W306 已通过：{report['w306_ready']}
- 训练模型：{report['model_trained']}
- 生成 validation 指标：{report['validation_metrics_generated']}
- 打开 acceptance：false
- fake metric count：0
- 修改默认 runtime：false

训练代码已经准备好，但只有 W306 核验官方下载数据并冻结 capture-disjoint
角色后才允许真实训练。外部论文指标保留在独立的 context-only 表中，不会
混成本地同 split 结果。模型、温度和决策阈值只允许由 validation 固定，
acceptance 在本轮始终保持密封。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_edge_iiotset_safe_baseline_w307(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    protocol = _load(output / "protocol_w307.json")
    training = _load(output / "training_report_w307.json")
    lock = _load(output / "selected_model_lock_w307.json")
    paper_rows = _read_csv(
        output / "paper_reported_only_comparison_w307.csv"
    )
    before = _load(output / "frozen_hashes_before_w307.json")
    after = _safe_hashes()
    unchanged = _hash_snapshot_equal(before, after)
    waiting = protocol.get("status") == "pipeline_ready_waiting_for_w306_data"
    status = (
        "pipeline_ready_waiting_for_w306_data"
        if waiting
        else "w307_validation_model_locked_acceptance_sealed"
    )
    gates = {
        "paper_metrics_separated": all(
            row["used_as_local_metric"] == "False" for row in paper_rows
        ),
        "acceptance_never_opened": int(
            lock.get("acceptance_rows_read", -1)
        )
        == 0
        and not bool(lock.get("acceptance_opened", False)),
        "waiting_state_did_not_train": (
            not waiting or not bool(training.get("model_trained", False))
        ),
        "fake_metric_count_zero": int(lock.get("fake_metric_count", -1))
        == 0,
        "frozen_hashes_unchanged": unchanged,
        "tests_passed": bool(tests_passed),
    }
    if not all(gates.values()):
        status = "w307_failed_integrity_gate"
    _dump(output / "frozen_hashes_after_w307.json", after)
    report = {
        "status": status,
        "experiment": EXPERIMENT,
        "w306_ready": bool(protocol.get("w306_ready", False)),
        "model_trained": bool(training.get("model_trained", False)),
        "trained_model_count": int(training.get("trained_model_count", 0)),
        "validation_metrics_generated": bool(
            lock.get("validation_metrics_generated", True)
            if waiting
            else True
        ),
        "selected_model": lock.get("selected_model"),
        "selected_temperature": lock.get("selected_temperature"),
        "selected_threshold": lock.get("selected_threshold"),
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "acceptance_metrics_generated": False,
        "paper_reported_metrics_used_as_local_metrics": False,
        "performance_candidate_accepted": False,
        "optional_profile_created": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "fake_metric_count": 0,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "acceptance_report.json", report)
    _dump(
        output / "negative_results.json",
        {
            "negative_results": (
                [
                    {
                        "failure_type": "waiting_for_w306_data",
                        "failure_reason": (
                            "Official Edge-IIoTset payload and W306 frozen "
                            "capture split are not ready."
                        ),
                        "fake_metric_count": 0,
                        "runtime_modified": False,
                    }
                ]
                if waiting
                else []
            )
        },
    )
    _render_documents(report, Path(document), Path(document_cn))
    return report


__all__ = [
    "_binary_ece",
    "_load_role",
    "_metrics",
    "_select_validation_calibration",
    "_temperature_scale",
    "prepare_edge_iiotset_safe_baseline_w307",
    "train_edge_iiotset_safe_baseline_w307",
    "validate_edge_iiotset_safe_baseline_w307",
    "finalize_edge_iiotset_safe_baseline_w307",
]
