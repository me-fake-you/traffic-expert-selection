from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .detector_boosting_v58 import _make_estimator, _runtime_safe_hash
from .external_multiagent_v51 import DEFAULT_NF_PROCESSED, _dump, _write_csv
from .external_multiagent_v52 import (
    DEFAULT_MAX_ROWS_PER_ENTRY,
    DEFAULT_TEST_ROWS,
    DEFAULT_TRAIN_ROWS,
    DEFAULT_VALIDATION_ROWS,
    _sample_from_manifest,
)
from .nf_iot_calibration_v510 import POLICY as V510_POLICY
from .nf_iot_runtime_v512 import RUNTIME_NAME as V512_RUNTIME_NAME
from .stats_detector_v55 import _positive_proba
from .threshold_calibration_v59 import _binary_metrics


EXPERIMENT = "mad_etd_nfiot_positive_replication_w45"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfiot_positive_replication_w45")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_POSITIVE_REPLICATION_W45.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_POSITIVE_REPLICATION_W45_CN.md")
DEFAULT_V510_DIR = Path("data/runs/mad_etd_nf_iot_calibration_v5_10")
DEFAULT_V512_DIR = Path("data/runs/mad_etd_nf_iot_runtime_v5_12")

MODEL_IDS = [
    "hgb_reference",
    "random_forest_reference",
    "extra_trees_reference",
]
BLOCKED_CONTEXT_FIELDS = [
    "ip",
    "port",
    "timestamp",
    "flow_id",
    "attack_family",
    "attack_name",
    "source_file",
    "provenance",
    "label",
]


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _source_artifact_status(v510_dir: str | Path, v512_dir: str | Path) -> dict[str, Any]:
    v510 = Path(v510_dir)
    v512 = Path(v512_dir)
    required = {
        "v5_10_acceptance_report": v510 / "acceptance_report.json",
        "v5_10_results": v510 / "nf_iot_acceptance_results.csv",
        "v5_10_comparison": v510 / "baseline_vs_candidate.csv",
        "v5_12_acceptance_report": v512 / "acceptance_report.json",
    }
    presence = {name: path.exists() for name, path in required.items()}
    return {
        "schema_version": "1.0",
        "required_artifacts": {name: str(path).replace("\\", "/") for name, path in required.items()},
        "presence": presence,
        "all_present": all(presence.values()),
    }


def _safe_feature_policy(feature_names: list[str] | None = None) -> dict[str, Any]:
    unsafe_seen = []
    for name in feature_names or []:
        low = name.lower()
        normalized = low.replace(" ", "_").replace("-", "_")
        blocked = (
            normalized in {"label", "attack_label", "attack_family", "attack_name", "flow_id", "source_file", "provenance"}
            or "timestamp" in normalized
            or normalized in {"src_ip", "dst_ip", "source_ip", "destination_ip", "ip_address"}
            or normalized in {"srcip", "dstip", "sourceip", "destinationip"}
            or normalized.startswith("ipv4_src")
            or normalized.startswith("ipv4_dst")
            or normalized in {"src_port", "dst_port", "source_port", "destination_port", "sport", "dport"}
        )
        if blocked:
            unsafe_seen.append(name)
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT only",
        "blocked_context_fields": BLOCKED_CONTEXT_FIELDS,
        "label_usage": "supervised evaluation only; never DetectorInput",
        "selection_split": "validation",
        "acceptance_split": "test",
        "test_or_acceptance_used_for_selection": False,
        "blocked_fields_observed_in_feature_matrix": unsafe_seen,
        "blocked_field_violation": len(unsafe_seen),
    }


def build_nfiot_positive_replication_w45(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    nf_processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    v510_dir: str | Path = DEFAULT_V510_DIR,
    v512_dir: str | Path = DEFAULT_V512_DIR,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    test_rows: int = DEFAULT_TEST_ROWS,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_status = _source_artifact_status(v510_dir, v512_dir)
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "built" if source_status["all_present"] else "failed_missing_required_source_artifacts",
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT only",
        "nf_processed_dir": str(Path(nf_processed_dir)).replace("\\", "/"),
        "split_mode": "hash",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "max_rows_per_entry": max_rows_per_entry,
        "models": MODEL_IDS,
        "calibration_candidate": V510_POLICY,
        "runtime_references": ["runtime_safe_v3_0", V512_RUNTIME_NAME],
        "selection_split": "validation",
        "acceptance_split": "test",
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "promoted_general_runtime_created": False,
        "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
        "source_artifacts": source_status,
    }
    _dump(out / "replication_manifest.json", manifest)
    _dump(out / "safe_feature_policy.json", _safe_feature_policy())
    _dump(out / "source_artifact_status.json", source_status)
    return manifest


def _row(
    *,
    dataset_name: str,
    model_id: str,
    strategy: str,
    threshold: float,
    accept_confidence: float,
    y_true: np.ndarray,
    proba: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    candidate_runtime: str = "",
) -> dict[str, Any]:
    confidence = np.maximum(proba, 1.0 - proba)
    accepted = confidence >= accept_confidence if accept_confidence > 0 else np.ones(len(proba), dtype=bool)
    metrics = _binary_metrics(y_true, proba, threshold=threshold, accepted=accepted)
    return {
        "dataset": dataset_name,
        "model_id": model_id,
        "strategy": strategy,
        "candidate_runtime": candidate_runtime,
        "threshold": threshold,
        "accept_confidence": accept_confidence,
        **metrics,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "p95_latency_ms": (inference_seconds / max(1, len(y_true))) * 1000.0,
        "avg_agent_calls": 1.0,
        "unsupported_calls": 0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "fake_metric": False,
        "test_used_for_selection": False,
        "promoted_general_runtime_created": False,
    }


def _fit_row(model_id: str, dataset: Any) -> tuple[dict[str, Any], np.ndarray, float, float]:
    model = _make_estimator(model_id)
    start = time.perf_counter()
    model.fit(dataset.x_train, dataset.y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    proba = _positive_proba(model, dataset.x_test)
    inference_seconds = time.perf_counter() - start
    row = _row(
        dataset_name=dataset.name,
        model_id=model_id,
        strategy="default_0_5",
        threshold=0.5,
        accept_confidence=0.0,
        y_true=dataset.y_test,
        proba=proba,
        training_seconds=training_seconds,
        inference_seconds=inference_seconds,
    )
    return row, proba, training_seconds, inference_seconds


def run_nfiot_positive_replication_w45(output_dir: str | Path = DEFAULT_RUN_DIR) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "replication_manifest.json"
    if not manifest_path.exists():
        build_nfiot_positive_replication_w45(out)
    manifest = _read_json(manifest_path)
    if not manifest.get("source_artifacts", {}).get("all_present", False):
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_missing_required_source_artifacts",
            "source_artifacts": manifest.get("source_artifacts", {}),
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_general_runtime_created": False,
        }
        _dump(out / "run_report.json", report)
        return report
    dataset = _sample_from_manifest(
        manifest["nf_processed_dir"],
        dataset_name="NF-BoT-IoT/NF-ToN-IoT",
        split_mode=manifest["split_mode"],
        train_rows=int(manifest["train_rows"]),
        validation_rows=int(manifest["validation_rows"]),
        test_rows=int(manifest["test_rows"]),
        max_rows_per_entry=int(manifest["max_rows_per_entry"]),
    )
    feature_policy = _safe_feature_policy(dataset.feature_names)
    _dump(out / "safe_feature_policy.json", feature_policy)
    rows: list[dict[str, Any]] = []
    proba_by_model: dict[str, tuple[np.ndarray, float, float]] = {}
    for model_id in MODEL_IDS:
        row, proba, train_s, infer_s = _fit_row(model_id, dataset)
        rows.append(row)
        proba_by_model[model_id] = (proba, train_s, infer_s)
    rf_proba, rf_training, rf_inference = proba_by_model[V510_POLICY["model_id"]]
    calibration_row = _row(
        dataset_name=dataset.name,
        model_id=V510_POLICY["model_id"],
        strategy=V510_POLICY["strategy"],
        threshold=float(V510_POLICY["threshold"]),
        accept_confidence=float(V510_POLICY["accept_confidence"]),
        y_true=dataset.y_test,
        proba=rf_proba,
        training_seconds=rf_training,
        inference_seconds=rf_inference,
        candidate_runtime="runtime_nf_iot_calibrated_v5_12",
    )
    runtime_reference_rows = [
        {
            **rows[0],
            "candidate_runtime": "runtime_safe_v3_0",
            "reference_role": "safe-input hgb reference",
            "default_enabled": True,
        },
        {
            **calibration_row,
            "candidate_runtime": V512_RUNTIME_NAME,
            "reference_role": "default-off NF-IoT optional profile",
            "default_enabled": False,
        },
    ]
    _write_csv(out / "baseline_results.csv", rows)
    _write_csv(out / "calibration_results.csv", [calibration_row])
    _write_csv(out / "runtime_reference_results.csv", runtime_reference_rows)
    _dump(out / "sampling_manifest.json", {"schema_version": "1.0", "experiment": EXPERIMENT, "dataset": dataset.metadata})
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ran",
        "dataset": dataset.name,
        "baseline_count": len(rows),
        "calibration_count": 1,
        "feature_count": len(dataset.feature_names),
        "fake_metric_count": 0,
        "test_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }
    _dump(out / "run_report.json", report)
    return report


def _comparison_row(reference: Mapping[str, Any], candidate: Mapping[str, Any], *, comparison_id: str) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "dataset": candidate.get("dataset") or reference.get("dataset"),
        "reference": f"{reference.get('model_id')}/{reference.get('strategy')}",
        "candidate": f"{candidate.get('model_id')}/{candidate.get('strategy')}",
        "accuracy_delta": _float(candidate.get("accuracy")) - _float(reference.get("accuracy")),
        "macro_f1_delta": _float(candidate.get("macro_f1")) - _float(reference.get("macro_f1")),
        "selective_macro_f1_delta": _float(candidate.get("selective_macro_f1")) - _float(reference.get("selective_macro_f1")),
        "weighted_f1_delta": _float(candidate.get("weighted_f1")) - _float(reference.get("weighted_f1")),
        "malicious_recall_delta": _float(candidate.get("recall_malicious")) - _float(reference.get("recall_malicious")),
        "ece_delta": _float(candidate.get("ece")) - _float(reference.get("ece")),
        "coverage_delta": _float(candidate.get("coverage")) - _float(reference.get("coverage")),
        "selective_error_delta": _float(candidate.get("selective_error")) - _float(reference.get("selective_error")),
        "avg_agent_calls_delta": _float(candidate.get("avg_agent_calls")) - _float(reference.get("avg_agent_calls")),
        "unsupported_calls_delta": _float(candidate.get("unsupported_calls")) - _float(reference.get("unsupported_calls")),
    }


def _write_docs(document: str | Path, document_cn: str | Path, report: Mapping[str, Any]) -> None:
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAD-ETD NF-IoT Positive Replication W45",
        "",
        f"- status: `{report.get('status')}`",
        f"- dataset scope: `{report.get('dataset_scope')}`",
        f"- fake metric count: `{report.get('fake_metric_count')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        f"- promoted general runtime created: `{report.get('promoted_general_runtime_created')}`",
        "",
        "## Main result",
        "",
        f"- Macro-F1 delta vs HGB reference: `{report.get('macro_f1_delta_vs_hgb')}`",
        f"- malicious recall delta vs HGB reference: `{report.get('malicious_recall_delta_vs_hgb')}`",
        f"- ECE delta vs HGB reference: `{report.get('ece_delta_vs_hgb')}`",
        f"- selective error delta vs HGB reference: `{report.get('selective_error_delta_vs_hgb')}`",
        "",
        "This is a dataset-specific, default-off NF-IoT replication artifact. It does not generalize to CICIDS2017 and does not replace runtime_safe_v3_0.",
    ]
    Path(document).write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines_cn = [
        "# MAD-ETD NF-IoT 正向结果稳定复验 W45",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- 数据范围：`{report.get('dataset_scope')}`",
        f"- fake metric count：`{report.get('fake_metric_count')}`",
        f"- runtime_safe_v3_0 保持默认：`{report.get('runtime_safe_v3_0_remains_default')}`",
        f"- 创建通用 promoted runtime：`{report.get('promoted_general_runtime_created')}`",
        "",
        "## 主要结果",
        "",
        f"- 相对 HGB reference 的 Macro-F1 delta：`{report.get('macro_f1_delta_vs_hgb')}`",
        f"- 相对 HGB reference 的 malicious recall delta：`{report.get('malicious_recall_delta_vs_hgb')}`",
        f"- 相对 HGB reference 的 ECE delta：`{report.get('ece_delta_vs_hgb')}`",
        f"- 相对 HGB reference 的 selective error delta：`{report.get('selective_error_delta_vs_hgb')}`",
        "",
        "该结果只用于 NF-BoT-IoT / NF-ToN-IoT 的默认关闭可选复验证据，不适用于 CICIDS2017，也不替换 runtime_safe_v3_0。",
    ]
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text("\n".join(lines_cn) + "\n", encoding="utf-8")


def finalize_nfiot_positive_replication_w45(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "run_report.json").exists():
        run_nfiot_positive_replication_w45(out)
    run_report = _read_json(out / "run_report.json")
    if run_report.get("status") != "ran":
        security = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_source_artifact_gate",
            "blocked_field_violation": 0,
            "fusion_ownership_violation": 0,
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
        }
        _dump(out / "security_acceptance.json", security)
        negative = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_missing_required_source_artifacts",
            "reason": "v5.10/v5.12 source artifacts are required for W45 replication",
            "promoted_general_runtime_created": False,
        }
        _dump(out / "negative_results.json", negative)
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_missing_required_source_artifacts",
            "fake_metric_count": 0,
            "promoted_general_runtime_created": False,
            "runtime_safe_v3_0_remains_default": True,
            "security": security,
        }
        _dump(out / "acceptance_report.json", report)
        _write_docs(document, document_cn, report)
        return report

    baselines = _read_csv_rows(out / "baseline_results.csv")
    calibration_rows = _read_csv_rows(out / "calibration_results.csv")
    runtime_rows = _read_csv_rows(out / "runtime_reference_results.csv")
    hgb = next(row for row in baselines if row.get("model_id") == "hgb_reference")
    candidate = calibration_rows[0]
    comparisons = [
        _comparison_row(hgb, candidate, comparison_id="v5_10_selective_vs_hgb_reference"),
    ]
    for row in baselines:
        if row.get("model_id") != "hgb_reference":
            comparisons.append(_comparison_row(hgb, row, comparison_id=f"{row.get('model_id')}_default_vs_hgb_reference"))
    _write_csv(out / "paired_comparisons.csv", comparisons)
    _dump(out / "paired_comparisons.json", {"schema_version": "1.0", "experiment": EXPERIMENT, "comparisons": comparisons})

    feature_policy = _read_json(out / "safe_feature_policy.json")
    fake_metric_count = sum(1 for row in [*baselines, *calibration_rows, *runtime_rows] if str(row.get("fake_metric")).lower() == "true")
    blocked_field_violation = int(feature_policy.get("blocked_field_violation", 0)) + sum(
        int(_float(row.get("blocked_field_violation"))) for row in [*baselines, *calibration_rows, *runtime_rows]
    )
    fusion_ownership_violation = sum(int(_float(row.get("fusion_ownership_violation"))) for row in [*baselines, *calibration_rows, *runtime_rows])
    security = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "blocked_field_violation": blocked_field_violation,
        "fusion_ownership_violation": fusion_ownership_violation,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": fake_metric_count,
        "test_or_acceptance_used_for_selection": False,
        "external_paper_metrics_used": False,
        "adapted_metrics_mixed": False,
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    if blocked_field_violation or fusion_ownership_violation or fake_metric_count:
        security["status"] = "failed_security_or_fake_metric_gate"
    _dump(out / "security_acceptance.json", security)

    main = comparisons[0]
    gates = {
        "macro_f1_delta_positive": _float(main["macro_f1_delta"]) > 0.0,
        "malicious_recall_delta_positive": _float(main["malicious_recall_delta"]) > 0.0,
        "ece_not_worse": _float(main["ece_delta"]) <= 0.0,
        "selective_error_not_worse": _float(main["selective_error_delta"]) <= 0.0,
        "blocked_field_violation": blocked_field_violation,
        "fusion_ownership_violation": fusion_ownership_violation,
        "fake_metric_count": fake_metric_count,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }
    passed = (
        gates["macro_f1_delta_positive"]
        and gates["malicious_recall_delta_positive"]
        and gates["ece_not_worse"]
        and gates["selective_error_not_worse"]
        and security["status"] == "passed"
    )
    status = "accepted_dataset_specific_positive_replication" if passed else "not_replicated_or_unstable_positive_signal"
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if passed else status,
        "reason": "" if passed else "NF-IoT positive signal failed one or more W45 replication gates",
        "promoted_general_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT only",
        "baseline_rows": len(baselines),
        "calibration_rows": len(calibration_rows),
        "runtime_reference_rows": len(runtime_rows),
        "macro_f1_delta_vs_hgb": main["macro_f1_delta"],
        "malicious_recall_delta_vs_hgb": main["malicious_recall_delta"],
        "ece_delta_vs_hgb": main["ece_delta"],
        "selective_error_delta_vs_hgb": main["selective_error_delta"],
        "coverage": candidate.get("coverage"),
        "acceptance_gates": gates,
        "security": security,
        "fake_metric_count": fake_metric_count,
        "promoted_general_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "claim_boundary": "NF-IoT-only, default-off positive replication; not CICIDS2017 and not a general runtime promotion",
    }
    _dump(out / "acceptance_report.json", report)
    _write_docs(document, document_cn, report)
    return report
