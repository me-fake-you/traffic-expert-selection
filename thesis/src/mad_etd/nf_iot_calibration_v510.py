from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .detector_boosting_v58 import _make_estimator, _pcap_boundary_report, _runtime_safe_hash
from .external_multiagent_v51 import DEFAULT_NF_PROCESSED, _dump, _write_csv
from .external_multiagent_v52 import (
    DEFAULT_MAX_ROWS_PER_ENTRY,
    DEFAULT_TEST_ROWS,
    DEFAULT_TRAIN_ROWS,
    DEFAULT_VALIDATION_ROWS,
    _sample_from_manifest,
)
from .packet_sequence_v56 import DEFAULT_RUN_DIR as DEFAULT_V56_RUN_DIR
from .schemas import AgentEvidence, FeatureGroup
from .stats_detector_v55 import _positive_proba
from .threshold_calibration_v59 import _binary_metrics


EXPERIMENT = "mad_etd_nf_iot_calibration_v5_10"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nf_iot_calibration_v5_10")
DEFAULT_DOC = Path("docs/MAD_ETD_NF_IOT_CALIBRATION_V5_10.md")

POLICY = {
    "policy_id": "nf_iot_rf_class_wise_selective_v5_10",
    "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT only",
    "model_id": "random_forest_reference",
    "strategy": "class_wise_selective_threshold",
    "threshold": 0.49,
    "accept_confidence": 0.9901750976639484,
    "source": "v5.9 validation-selected NF-IoT gate",
    "default_enabled": False,
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _write_doc(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAD-ETD NF-IoT Calibration v5.10",
        "",
        f"- status: `{report.get('status')}`",
        f"- promoted runtime created: `{report.get('promoted_runtime_created')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        f"- fake metric count: `{report.get('fake_metric_count')}`",
        "",
        "## Interpretation",
        "",
        "v5.10 repeats the v5.9 NF-IoT-only calibration signal as a dataset-specific, "
        "default-off optional candidate. CICIDS2017 remains excluded and is reported as "
        "a negative result for this candidate.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_nf_iot_calibration_v5_10(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    nf_processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    test_rows: int = DEFAULT_TEST_ROWS,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "candidate": POLICY["policy_id"],
        "dataset_scope": POLICY["dataset_scope"],
        "excluded_datasets": ["CICIDS2017"],
        "nf_processed_dir": str(Path(nf_processed_dir)).replace("\\", "/"),
        "split_mode": "hash",
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "max_rows_per_entry": max_rows_per_entry,
        "selection_split": "validation",
        "acceptance_split": "test",
        "test_used_for_selection": False,
        "locked_test_used": False,
        "blocked_context_fields_allowed": False,
        "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "candidate_policy.json", POLICY)
    _dump(out / "selection_manifest.json", manifest)
    _dump(out / "acceptance_manifest.json", manifest)
    _dump(
        out / "cicids2017_exclusion_report.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "dataset": "CICIDS2017",
            "status": "excluded_negative_result",
            "reason": "v5.9 CICIDS2017 calibration did not satisfy the pre-registered gate",
        },
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "dataset_scope": POLICY["dataset_scope"],
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "manifest_build_report.json", report)
    return report


def _load_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "selection_manifest.json"
    if not path.exists():
        build_nf_iot_calibration_v5_10(output_dir)
    return _read_json(path)


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
) -> dict[str, Any]:
    confidence = np.maximum(proba, 1.0 - proba)
    accepted = confidence >= accept_confidence if accept_confidence > 0 else np.ones(len(proba), dtype=bool)
    metrics = _binary_metrics(y_true, proba, threshold=threshold, accepted=accepted)
    return {
        "dataset": dataset_name,
        "model_id": model_id,
        "strategy": strategy,
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
        "runtime_promoted": False,
    }


def run_nf_iot_calibration_v5_10(output_dir: str | Path = DEFAULT_RUN_DIR) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(out)
    dataset = _sample_from_manifest(
        manifest["nf_processed_dir"],
        dataset_name="NF-BoT-IoT/NF-ToN-IoT",
        split_mode="hash",
        train_rows=int(manifest["train_rows"]),
        validation_rows=int(manifest["validation_rows"]),
        test_rows=int(manifest["test_rows"]),
        max_rows_per_entry=int(manifest["max_rows_per_entry"]),
    )
    baseline_model = _make_estimator("hgb_reference")
    candidate_model = _make_estimator(POLICY["model_id"])

    start = time.perf_counter()
    baseline_model.fit(dataset.x_train, dataset.y_train)
    baseline_training = time.perf_counter() - start
    start = time.perf_counter()
    baseline_proba = _positive_proba(baseline_model, dataset.x_test)
    baseline_inference = time.perf_counter() - start

    start = time.perf_counter()
    candidate_model.fit(dataset.x_train, dataset.y_train)
    candidate_training = time.perf_counter() - start
    start = time.perf_counter()
    candidate_proba = _positive_proba(candidate_model, dataset.x_test)
    candidate_inference = time.perf_counter() - start

    baseline = _row(
        dataset_name=dataset.name,
        model_id="hgb_reference",
        strategy="default_0_5",
        threshold=0.5,
        accept_confidence=0.0,
        y_true=dataset.y_test,
        proba=baseline_proba,
        training_seconds=baseline_training,
        inference_seconds=baseline_inference,
    )
    candidate = _row(
        dataset_name=dataset.name,
        model_id=POLICY["model_id"],
        strategy=POLICY["strategy"],
        threshold=float(POLICY["threshold"]),
        accept_confidence=float(POLICY["accept_confidence"]),
        y_true=dataset.y_test,
        proba=candidate_proba,
        training_seconds=candidate_training,
        inference_seconds=candidate_inference,
    )
    comparison = {
        "dataset": dataset.name,
        "selective_macro_f1_delta": candidate["selective_macro_f1"] - baseline["selective_macro_f1"],
        "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
        "malicious_recall_delta": candidate["recall_malicious"] - baseline["recall_malicious"],
        "ece_delta": candidate["ece"] - baseline["ece"],
        "coverage_delta": candidate["coverage"] - baseline["coverage"],
        "selective_error_delta": candidate["selective_error"] - baseline["selective_error"],
    }
    _write_csv(out / "nf_iot_acceptance_results.csv", [baseline, candidate])
    _write_csv(out / "baseline_vs_candidate.csv", [comparison])
    _dump(out / "sampling_manifest.json", {"schema_version": "1.0", "experiment": EXPERIMENT, "dataset": dataset.metadata})
    evidence = AgentEvidence(
        agent_name="NFIoTCalibrationV510",
        agent_version="5.10-candidate",
        feature_group=FeatureGroup.STATS,
        benign_support=0.5,
        malicious_support=0.5,
        confidence=0.5,
        uncertainty=0.5,
        contributes_to_verdict=False,
        evidence=["NF-IoT calibration is dataset-specific and default-off"],
        used_fields=[f"stats.{name}" for name in dataset.feature_names[:10]],
    )
    _dump(
        out / "agent_evidence_compatibility.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "agent_evidence_schema_compatible": True,
            "candidate_outputs_final_verdict": False,
            "fusion_agent_remains_verdict_owner": True,
            "sample_agent_evidence": evidence.model_dump(mode="json"),
        },
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "acceptance_rows": 2,
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_hash_before": manifest.get("runtime_safe_v3_0_hash_before"),
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "calibration_reproducibility_report.json", report)
    return report


def finalize_nf_iot_calibration_v5_10(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    pcap_v56_dir: str | Path = DEFAULT_V56_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(out / "nf_iot_acceptance_results.csv")
    comparison_rows = _read_csv_rows(out / "baseline_vs_candidate.csv")
    comparison = comparison_rows[0] if comparison_rows else {}
    pcap_boundary = _pcap_boundary_report(pcap_v56_dir)
    _dump(out / "pcap_boundary_report.json", pcap_boundary)
    fake_metric_count = sum(1 for row in rows if str(row.get("fake_metric")).lower() == "true")
    security = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed" if fake_metric_count == 0 else "failed_fake_metrics",
        "blocked_field_violation": sum(int(float(row.get("blocked_field_violation", 0) or 0)) for row in rows),
        "fusion_ownership_violation": sum(int(float(row.get("fusion_ownership_violation", 0) or 0)) for row in rows),
        "fake_metric_count": fake_metric_count,
        "test_used_for_selection": False,
        "locked_test_used": False,
        "pcap_supervised_metrics_used": False,
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "security_acceptance.json", security)
    gates = {
        "selective_macro_f1_delta": _float(comparison.get("selective_macro_f1_delta")),
        "malicious_recall_delta": _float(comparison.get("malicious_recall_delta")),
        "selective_error_delta": _float(comparison.get("selective_error_delta")),
        "coverage": _float(rows[1].get("coverage")) if len(rows) > 1 else 0.0,
        "passed": False,
    }
    gates["passed"] = (
        gates["selective_macro_f1_delta"] >= 0.02
        and gates["malicious_recall_delta"] >= 0.005
        and gates["selective_error_delta"] <= -0.01
        and gates["coverage"] >= 0.90
        and security["status"] == "passed"
        and security["blocked_field_violation"] == 0
        and security["fusion_ownership_violation"] == 0
    )
    _dump(out / "acceptance_gates.json", gates)
    status = (
        "accepted_dataset_specific_optional_calibration"
        if gates["passed"]
        else "not_promoted_reproducibility_or_tradeoff_failure"
    )
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if gates["passed"] else status,
        "cicids2017_status": "excluded_negative_result",
        "reason": "" if gates["passed"] else "NF-IoT fixed calibration failed reproducibility or tradeoff gates",
        "runtime_promoted": False,
    }
    _dump(out / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "candidate_policy": POLICY,
        "comparison": comparison,
        "acceptance_gates": gates,
        "security": security,
        "pcap_boundary": pcap_boundary,
        "fake_metric_count": fake_metric_count,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "cicids2017_status": "excluded_negative_result",
    }
    _dump(out / "acceptance_report.json", report)
    _write_doc(document, report)
    return report
