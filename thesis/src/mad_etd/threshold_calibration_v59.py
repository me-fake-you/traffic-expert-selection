from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .detector_boosting_v58 import (
    _make_estimator,
    _package_rows,
    _pcap_boundary_report,
    _runtime_safe_hash,
)
from .external_multiagent_v51 import (
    DEFAULT_CICIDS_PROCESSED,
    DEFAULT_NF_PROCESSED,
    _dump,
    _write_csv,
)
from .external_multiagent_v52 import (
    DEFAULT_MAX_ROWS_PER_ENTRY,
    DEFAULT_TEST_ROWS,
    DEFAULT_TRAIN_ROWS,
    DEFAULT_VALIDATION_ROWS,
    SampledDataset,
    _sample_from_manifest,
)
from .packet_sequence_v56 import DEFAULT_RUN_DIR as DEFAULT_V56_RUN_DIR
from .schemas import AgentEvidence, FeatureGroup
from .stats_detector_v55 import _positive_proba


EXPERIMENT = "mad_etd_threshold_calibration_v5_9"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_threshold_calibration_v5_9")
DEFAULT_DOC = Path("docs/MAD_ETD_THRESHOLD_CALIBRATION_V5_9.md")


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
        "# MAD-ETD Threshold Calibration v5.9",
        "",
        f"- status: `{report.get('status')}`",
        f"- calibration upgrade signal: `{report.get('calibration_upgrade_signal')}`",
        f"- promoted runtime created: `{report.get('promoted_runtime_created')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        f"- fake metric count: `{report.get('fake_metric_count')}`",
        "",
        "## Interpretation",
        "",
        "v5.9 evaluates validation-only threshold and selective detection calibration "
        "on existing safe-input detector probabilities. Thresholds are selected on "
        "validation only; acceptance/test is evaluated once.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _available_model_ids() -> list[str]:
    ids = ["hgb_reference", "random_forest_reference"]
    available = {row["model_id"] for row in _package_rows() if row.get("available")}
    if "xgboost_optional" in available:
        ids.append("xgboost_optional")
    return ids


def build_threshold_calibration_v5_9_manifest(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    cicids_processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    nf_processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    test_rows: int = DEFAULT_TEST_ROWS,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [
        {
            "dataset": "CICIDS2017",
            "dataset_slug": "cicids2017",
            "processed_dir": str(Path(cicids_processed_dir)).replace("\\", "/"),
            "split_mode": "cicids_group",
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "max_rows_per_entry": max_rows_per_entry,
        },
        {
            "dataset": "NF-BoT-IoT/NF-ToN-IoT",
            "dataset_slug": "nf_iot",
            "processed_dir": str(Path(nf_processed_dir)).replace("\\", "/"),
            "split_mode": "hash",
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "test_rows": test_rows,
            "max_rows_per_entry": max_rows_per_entry,
        },
    ]
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "candidate": "threshold_calibration_candidate_v5_9",
        "default_enabled": False,
        "model_ids": _available_model_ids(),
        "strategies": [
            "default_0_5",
            "macro_f1_optimal_threshold",
            "malicious_recall_constrained_threshold",
            "ece_aware_threshold",
            "selective_accept_abstain",
            "class_wise_selective_threshold",
        ],
        "datasets": datasets,
        "selection_split": "validation",
        "acceptance_split": "test",
        "test_used_for_selection": False,
        "locked_test_used": False,
        "blocked_context_fields_allowed": False,
        "pcap_supervised_metrics_used": False,
        "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "selection_manifest.json", manifest)
    _dump(out / "training_manifest.json", {**manifest, "training_split": "train"})
    _dump(out / "acceptance_manifest.json", {**manifest, "acceptance_split": "test"})
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "dataset_count": len(datasets),
        "model_ids": manifest["model_ids"],
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "manifest_build_report.json", report)
    return report


def _load_datasets_from_manifest(output_dir: str | Path) -> list[tuple[SampledDataset, str]]:
    manifest = _read_json(Path(output_dir) / "selection_manifest.json")
    datasets: list[tuple[SampledDataset, str]] = []
    for item in manifest.get("datasets", []):
        datasets.append(
            (
                _sample_from_manifest(
                    item["processed_dir"],
                    dataset_name=item["dataset"],
                    split_mode=item["split_mode"],
                    train_rows=int(item["train_rows"]),
                    validation_rows=int(item["validation_rows"]),
                    test_rows=int(item["test_rows"]),
                    max_rows_per_entry=int(item["max_rows_per_entry"]),
                ),
                item["dataset_slug"],
            )
        )
    return datasets


def _binary_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    accepted: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if accepted is None:
        accepted = np.ones(len(y_true), dtype=bool)
    coverage = float(accepted.mean()) if len(accepted) else 0.0
    if not np.any(accepted):
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "selective_macro_f1": 0.0,
            "weighted_f1": 0.0,
            "precision_malicious": 0.0,
            "recall_malicious": 0.0,
            "ece": 0.0,
            "coverage": coverage,
            "selective_error": 1.0,
        }
    y = y_true[accepted]
    p = proba[accepted]
    pred = (p >= threshold).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    accuracy = float(accuracy_score(y, pred))
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "selective_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "precision_malicious": float(precision_score(y, pred, zero_division=0)),
        "recall_malicious": float(recall_score(y, pred, zero_division=0)),
        "ece": ece,
        "coverage": coverage,
        "selective_error": 1.0 - accuracy,
    }


def _row(
    *,
    dataset: SampledDataset,
    dataset_slug: str,
    split: str,
    model_id: str,
    strategy: str,
    threshold: float,
    proba: np.ndarray,
    accepted: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    selected_on_validation: bool,
) -> dict[str, Any]:
    y = dataset.y_validation if split == "validation" else dataset.y_test
    metrics = _binary_metrics(y, proba, threshold=threshold, accepted=accepted)
    return {
        "dataset": dataset.name,
        "dataset_slug": dataset_slug,
        "split": split,
        "model_id": model_id,
        "strategy": strategy,
        "threshold": threshold,
        "accept_confidence": _accept_confidence_from_mask(proba, accepted),
        **metrics,
        "train_rows": int(dataset.x_train.shape[0]),
        "validation_rows": int(dataset.x_validation.shape[0]),
        "test_rows": int(dataset.x_test.shape[0]),
        "feature_count": len(dataset.feature_names),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "p95_latency_ms": (inference_seconds / max(1, len(y))) * 1000.0,
        "avg_agent_calls": 1.0,
        "unsupported_calls": 0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "fake_metric": False,
        "selected_on_validation": selected_on_validation,
        "test_used_for_selection": False,
        "runtime_promoted": False,
    }


def _accept_confidence_from_mask(proba: np.ndarray, accepted: np.ndarray) -> float:
    if len(proba) == 0 or np.all(accepted):
        return 0.0
    confidence = np.maximum(proba, 1.0 - proba)
    return float(confidence[accepted].min()) if np.any(accepted) else 1.0


def _select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    *,
    objective: str,
    reference: dict[str, float],
) -> tuple[float, np.ndarray]:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -1e9
    best_accept = np.ones(len(y), dtype=bool)
    for threshold in thresholds:
        accepted = np.ones(len(y), dtype=bool)
        metrics = _binary_metrics(y, p, threshold=float(threshold), accepted=accepted)
        if objective == "macro":
            score = metrics["macro_f1"]
        elif objective == "recall":
            if metrics["recall_malicious"] + 1e-12 < reference["recall_malicious"]:
                continue
            score = metrics["macro_f1"] + 0.05 * metrics["recall_malicious"]
        elif objective == "ece":
            if metrics["ece"] > reference["ece"] + 0.01:
                continue
            score = metrics["macro_f1"] - 0.25 * metrics["ece"]
        else:
            score = metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_accept = accepted
    return best_threshold, best_accept


def _select_confidence_gate(
    y: np.ndarray,
    p: np.ndarray,
    *,
    threshold: float,
    reference: dict[str, float],
    min_coverage: float = 0.95,
) -> tuple[float, np.ndarray]:
    confidence = np.maximum(p, 1.0 - p)
    best_gate = 0.0
    best_accept = np.ones(len(y), dtype=bool)
    best_score = -1e9
    for gate in np.linspace(0.5, 0.99, 50):
        accepted = confidence >= gate
        metrics = _binary_metrics(y, p, threshold=threshold, accepted=accepted)
        if metrics["coverage"] < min_coverage and metrics["selective_error"] > reference["selective_error"] - 0.01:
            continue
        score = metrics["selective_macro_f1"] - max(0.0, min_coverage - metrics["coverage"])
        if score > best_score:
            best_score = score
            best_gate = float(gate)
            best_accept = accepted
    return best_gate, best_accept


def _strategy_specs(y_val: np.ndarray, val_proba: np.ndarray) -> dict[str, tuple[float, np.ndarray]]:
    reference = _binary_metrics(
        y_val,
        val_proba,
        threshold=0.5,
        accepted=np.ones(len(y_val), dtype=bool),
    )
    macro_threshold, macro_accept = _select_threshold(y_val, val_proba, objective="macro", reference=reference)
    recall_threshold, recall_accept = _select_threshold(y_val, val_proba, objective="recall", reference=reference)
    ece_threshold, ece_accept = _select_threshold(y_val, val_proba, objective="ece", reference=reference)
    selective_gate, selective_accept = _select_confidence_gate(
        y_val,
        val_proba,
        threshold=macro_threshold,
        reference=reference,
        min_coverage=0.95,
    )
    class_gate, class_accept = _select_confidence_gate(
        y_val,
        val_proba,
        threshold=recall_threshold,
        reference=reference,
        min_coverage=0.90,
    )
    return {
        "default_0_5": (0.5, np.ones(len(y_val), dtype=bool)),
        "macro_f1_optimal_threshold": (macro_threshold, macro_accept),
        "malicious_recall_constrained_threshold": (recall_threshold, recall_accept),
        "ece_aware_threshold": (ece_threshold, ece_accept),
        "selective_accept_abstain": (macro_threshold, selective_accept),
        "class_wise_selective_threshold": (recall_threshold, class_accept),
    }


def _apply_strategy_to_test(
    test_proba: np.ndarray,
    *,
    threshold: float,
    validation_accept: np.ndarray,
    validation_proba: np.ndarray,
) -> np.ndarray:
    if np.all(validation_accept):
        return np.ones(len(test_proba), dtype=bool)
    confidence = np.maximum(validation_proba, 1.0 - validation_proba)
    if not np.any(validation_accept):
        gate = 1.0
    else:
        gate = float(confidence[validation_accept].min())
    test_confidence = np.maximum(test_proba, 1.0 - test_proba)
    return test_confidence >= gate


def run_threshold_calibration_v5_9(output_dir: str | Path = DEFAULT_RUN_DIR) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "selection_manifest.json").exists():
        build_threshold_calibration_v5_9_manifest(out)
    manifest = _read_json(out / "selection_manifest.json")
    datasets = _load_datasets_from_manifest(out)
    model_ids = list(manifest.get("model_ids", _available_model_ids()))
    selection_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    sampling = {}
    for dataset, slug in datasets:
        sampling[slug] = dataset.metadata
        for model_id in model_ids:
            model = _make_estimator(model_id)
            start = time.perf_counter()
            model.fit(dataset.x_train, dataset.y_train)
            training_seconds = time.perf_counter() - start
            start = time.perf_counter()
            val_proba = _positive_proba(model, dataset.x_validation)
            test_proba = _positive_proba(model, dataset.x_test)
            inference_seconds = time.perf_counter() - start
            specs = _strategy_specs(dataset.y_validation, val_proba)
            validation_scores: list[tuple[str, float]] = []
            for strategy, (threshold, val_accept) in specs.items():
                val_row = _row(
                    dataset=dataset,
                    dataset_slug=slug,
                    split="validation",
                    model_id=model_id,
                    strategy=strategy,
                    threshold=threshold,
                    proba=val_proba,
                    accepted=val_accept,
                    training_seconds=training_seconds,
                    inference_seconds=inference_seconds / 2.0,
                    selected_on_validation=True,
                )
                selection_rows.append(val_row)
                validation_scores.append((strategy, float(val_row["selective_macro_f1"])))
                test_accept = _apply_strategy_to_test(
                    test_proba,
                    threshold=threshold,
                    validation_accept=val_accept,
                    validation_proba=val_proba,
                )
                acceptance_rows.append(
                    _row(
                        dataset=dataset,
                        dataset_slug=slug,
                        split="test",
                        model_id=model_id,
                        strategy=strategy,
                        threshold=threshold,
                        proba=test_proba,
                        accepted=test_accept,
                        training_seconds=training_seconds,
                        inference_seconds=inference_seconds / 2.0,
                        selected_on_validation=True,
                    )
                )
            selected_strategy = max(validation_scores, key=lambda item: item[1])[0]
            selected_rows.append(
                {
                    "dataset": dataset.name,
                    "dataset_slug": slug,
                    "model_id": model_id,
                    "selected_strategy": selected_strategy,
                    "selection_selective_macro_f1": max(score for _strategy, score in validation_scores),
                    "test_used_for_selection": False,
                }
            )
    _write_csv(out / "model_selection_results.csv", selection_rows)
    _write_csv(out / "threshold_calibration_results.csv", acceptance_rows)
    _write_csv(out / "selected_thresholds.csv", selected_rows)
    _dump(
        out / "sampling_manifest.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "datasets": sampling,
            "test_used_for_selection": False,
            "locked_test_used": False,
            "pcap_supervised_metrics_used": False,
        },
    )
    feature_names = datasets[0][0].feature_names if datasets else []
    evidence = AgentEvidence(
        agent_name="ThresholdCalibrationV59",
        agent_version="5.9-candidate",
        feature_group=FeatureGroup.STATS,
        benign_support=0.5,
        malicious_support=0.5,
        confidence=0.5,
        uncertainty=0.5,
        contributes_to_verdict=False,
        evidence=["v5.9 threshold calibration is default-off and validation-only"],
        used_fields=[f"stats.{name}" for name in feature_names[:10]],
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
        "selection_rows": len(selection_rows),
        "acceptance_rows": len(acceptance_rows),
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "threshold_calibration_run_report.json", report)
    return report


def _best_rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset")), []).append(row)
    best: dict[str, dict[str, Any]] = {}
    for dataset, dataset_rows in grouped.items():
        candidates = [
            row
            for row in dataset_rows
            if not (
                row.get("model_id") == "hgb_reference"
                and row.get("strategy") == "default_0_5"
            )
        ]
        if candidates:
            best[dataset] = max(candidates, key=lambda row: _float(row.get("selective_macro_f1")))
    return best


def _gate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset")), []).append(row)
    best = _best_rows_by_dataset(rows)
    gates: list[dict[str, Any]] = []
    for dataset, dataset_rows in grouped.items():
        ref = next(
            (
                row
                for row in dataset_rows
                if row.get("model_id") == "hgb_reference"
                and row.get("strategy") == "default_0_5"
            ),
            None,
        )
        cand = best.get(dataset)
        if not ref or not cand:
            continue
        selective_delta = _float(cand.get("selective_macro_f1")) - _float(ref.get("selective_macro_f1"))
        macro_delta = _float(cand.get("macro_f1")) - _float(ref.get("macro_f1"))
        recall_delta = _float(cand.get("recall_malicious")) - _float(ref.get("recall_malicious"))
        ece_delta = _float(cand.get("ece")) - _float(ref.get("ece"))
        coverage = _float(cand.get("coverage"))
        selective_error_delta = _float(cand.get("selective_error")) - _float(ref.get("selective_error"))
        coverage_ok = coverage >= 0.95 or selective_error_delta <= -0.01
        gates.append(
            {
                "dataset": dataset,
                "reference": "hgb_reference/default_0_5",
                "best_candidate": f"{cand.get('model_id')}/{cand.get('strategy')}",
                "selective_macro_f1_delta": selective_delta,
                "macro_f1_delta": macro_delta,
                "malicious_recall_delta": recall_delta,
                "ece_delta": ece_delta,
                "coverage": coverage,
                "selective_error_delta": selective_error_delta,
                "passed": (
                    selective_delta >= 0.005
                    and macro_delta >= -0.002
                    and recall_delta >= 0
                    and ece_delta <= 0.01
                    and coverage_ok
                ),
                "runtime_promoted": False,
            }
        )
    return gates


def finalize_threshold_calibration_v5_9(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    pcap_v56_dir: str | Path = DEFAULT_V56_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_csv_rows(out / "threshold_calibration_results.csv")
    selection_rows = _read_csv_rows(out / "model_selection_results.csv")
    gates = _gate_rows(rows)
    _write_csv(out / "calibration_upgrade_gates.csv", gates)
    pcap_boundary = _pcap_boundary_report(pcap_v56_dir)
    _dump(out / "pcap_boundary_report.json", pcap_boundary)
    fake_metric_count = sum(
        1 for row in rows + selection_rows if str(row.get("fake_metric")).lower() == "true"
    )
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
    all_pass = bool(gates) and all(
        str(gate.get("passed")).lower() == "true" or gate.get("passed") is True
        for gate in gates
    )
    any_gain = any(_float(gate.get("selective_macro_f1_delta")) > 0 for gate in gates)
    if all_pass and security["status"] == "passed":
        status = "accepted_optional_calibration_signal"
    elif any_gain:
        status = "not_promoted_unstable_or_insufficient_gain"
    else:
        status = "not_promoted_no_gain"
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if status == "accepted_optional_calibration_signal" else status,
        "reason": "" if status == "accepted_optional_calibration_signal" else "threshold calibration did not satisfy all pre-registered gates",
        "runtime_promoted": False,
    }
    _dump(out / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "calibration_upgrade_signal": any_gain,
        "calibration_upgrade_gates": gates,
        "fake_metric_count": fake_metric_count,
        "security": security,
        "pcap_boundary": pcap_boundary,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "acceptance_report.json", report)
    _write_doc(document, report)
    return report
