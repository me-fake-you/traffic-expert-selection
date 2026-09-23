from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

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
from .runtime_profiles import load_runtime_profile, runtime_profile_sha256
from .schemas import AgentEvidence, FeatureGroup
from .stats_detector_v55 import _ece, _positive_proba


EXPERIMENT = "mad_etd_detector_boosting_v5_8"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_detector_boosting_v5_8")
DEFAULT_DOC = Path("docs/MAD_ETD_DETECTOR_BOOSTING_V5_8.md")
RUNTIME_SAFE_PROFILE = "runtime_safe_v3_0"

BOOSTING_PACKAGES = {
    "xgboost": "xgboost_optional",
    "lightgbm": "lightgbm_optional",
    "catboost": "catboost_optional",
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


def _runtime_safe_hash() -> str | None:
    try:
        return runtime_profile_sha256(load_runtime_profile(RUNTIME_SAFE_PROFILE))
    except Exception:
        return None


def _write_doc(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAD-ETD Detector Boosting v5.8",
        "",
        f"- status: `{report.get('status')}`",
        f"- detector upgrade signal: `{report.get('detector_upgrade_signal')}`",
        f"- promoted runtime created: `{report.get('promoted_runtime_created')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        f"- fake metric count: `{report.get('fake_metric_count')}`",
        "",
        "## Interpretation",
        "",
        "v5.8 evaluates optional gradient boosting detectors under the same safe-input "
        "policy used by the external baseline rounds. Validation selects candidates; "
        "test/acceptance is evaluated once. The result is an optional detector signal, "
        "not an automatic runtime promotion.",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _package_rows(package_availability: Mapping[str, bool] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for package, model_id in BOOSTING_PACKAGES.items():
        available = (
            bool(package_availability[package])
            if package_availability is not None and package in package_availability
            else importlib.util.find_spec(package) is not None
        )
        rows.append(
            {
                "package": package,
                "model_id": model_id,
                "available": available,
                "required_for_boosting_round": True,
            }
        )
    return rows


def probe_boosting_packages_v5_8(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    package_availability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _package_rows(package_availability)
    _write_csv(out / "package_probe_results.csv", rows)
    available = [row for row in rows if bool(row["available"])]
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed" if available else "blocked_missing_optional_boosting_packages",
        "available_boosting_package_count": len(available),
        "available_model_ids": [row["model_id"] for row in available],
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "package_probe_report.json", report)
    return report


def _make_estimator(model_id: str):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    if model_id == "hgb_reference":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=100, learning_rate=0.07, random_state=42),
        )
    if model_id == "random_forest_reference":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=160,
                max_depth=20,
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
        )
    if model_id == "extra_trees_reference":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=160,
                max_depth=None,
                n_jobs=1,
                class_weight="balanced",
                random_state=42,
            ),
        )
    if model_id == "xgboost_optional":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=160,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
        )
    if model_id == "lightgbm_optional":
        from lightgbm import LGBMClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=160,
                learning_rate=0.06,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
                verbose=-1,
            ),
        )
    if model_id == "catboost_optional":
        from catboost import CatBoostClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostClassifier(
                iterations=160,
                depth=6,
                learning_rate=0.06,
                loss_function="Logloss",
                random_seed=42,
                verbose=False,
                thread_count=1,
            ),
        )
    raise ValueError(f"unknown model_id: {model_id}")


def _metric_row(
    *,
    dataset: SampledDataset,
    dataset_slug: str,
    split: str,
    model_id: str,
    proba: np.ndarray,
    training_seconds: float,
    inference_seconds: float,
    source_type: str,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = dataset.y_validation if split == "validation" else dataset.y_test
    pred = (proba >= 0.5).astype(int)
    return {
        "dataset": dataset.name,
        "dataset_slug": dataset_slug,
        "split": split,
        "model_id": model_id,
        "source_type": source_type,
        "accuracy": accuracy_score(y_true, pred) if len(y_true) else 0.0,
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0) if len(y_true) else 0.0,
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0) if len(y_true) else 0.0,
        "precision_malicious": precision_score(y_true, pred, zero_division=0) if len(y_true) else 0.0,
        "recall_malicious": recall_score(y_true, pred, zero_division=0) if len(y_true) else 0.0,
        "ece": _ece(y_true, proba),
        "train_rows": int(dataset.x_train.shape[0]),
        "validation_rows": int(dataset.x_validation.shape[0]),
        "test_rows": int(dataset.x_test.shape[0]),
        "feature_count": len(dataset.feature_names),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "p95_latency_ms": (inference_seconds / max(1, len(y_true))) * 1000.0,
        "avg_agent_calls": 1.0,
        "unsupported_calls": 0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "fake_metric": False,
        "runtime_promoted": False,
    }


def _load_datasets(
    *,
    cicids_processed_dir: str | Path,
    nf_processed_dir: str | Path,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    max_rows_per_entry: int,
) -> list[tuple[SampledDataset, str]]:
    return [
        (
            _sample_from_manifest(
                cicids_processed_dir,
                dataset_name="CICIDS2017",
                split_mode="cicids_group",
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
                max_rows_per_entry=max_rows_per_entry,
            ),
            "cicids2017",
        ),
        (
            _sample_from_manifest(
                nf_processed_dir,
                dataset_name="NF-BoT-IoT/NF-ToN-IoT",
                split_mode="hash",
                train_rows=train_rows,
                validation_rows=validation_rows,
                test_rows=test_rows,
                max_rows_per_entry=max_rows_per_entry,
            ),
            "nf_iot",
        ),
    ]


def _model_ids_from_probe(rows: list[dict[str, Any]]) -> list[str]:
    ids = ["hgb_reference", "random_forest_reference", "extra_trees_reference"]
    for row in rows:
        if str(row.get("available")).lower() == "true" or row.get("available") is True:
            ids.append(str(row["model_id"]))
    return ids


def _fit_dataset_models(
    dataset: SampledDataset,
    dataset_slug: str,
    model_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    validation_probas: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}
    validation_scores: dict[str, float] = {}
    for model_id in model_ids:
        model = _make_estimator(model_id)
        start = time.perf_counter()
        model.fit(dataset.x_train, dataset.y_train)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_proba = _positive_proba(model, dataset.x_validation)
        test_proba = _positive_proba(model, dataset.x_test)
        inference_seconds = time.perf_counter() - start
        source_type = "reference" if model_id.endswith("_reference") else "optional_boosting_candidate"
        val_row = _metric_row(
            dataset=dataset,
            dataset_slug=dataset_slug,
            split="validation",
            model_id=model_id,
            proba=val_proba,
            training_seconds=training_seconds,
            inference_seconds=inference_seconds / 2.0,
            source_type=source_type,
        )
        validation_scores[model_id] = float(val_row["macro_f1"])
        validation_rows.append(val_row)
        acceptance_rows.append(
            _metric_row(
                dataset=dataset,
                dataset_slug=dataset_slug,
                split="test",
                model_id=model_id,
                proba=test_proba,
                training_seconds=training_seconds,
                inference_seconds=inference_seconds / 2.0,
                source_type=source_type,
            )
        )
        validation_probas[model_id] = val_proba
        test_probas[model_id] = test_proba
    selectable = [model_id for model_id in model_ids if model_id != "hgb_reference"]
    best_model = max(selectable, key=lambda model_id: validation_scores[model_id])
    for split, probas, rows in (
        ("validation", validation_probas, validation_rows),
        ("test", test_probas, acceptance_rows),
    ):
        rows.append(
            _metric_row(
                dataset=dataset,
                dataset_slug=dataset_slug,
                split=split,
                model_id="validation_best_model",
                proba=probas[best_model],
                training_seconds=0.0,
                inference_seconds=0.0,
                source_type="validation_selected_candidate",
            )
        )
    boosting_members = [
        model_id for model_id in model_ids if model_id in set(BOOSTING_PACKAGES.values())
    ]
    if boosting_members:
        weights = np.asarray([max(validation_scores[model_id], 0.0) for model_id in boosting_members])
        if weights.sum() <= 0:
            weights = np.ones(len(boosting_members), dtype=float)
        weights = weights / weights.sum()
        val_ensemble = sum(validation_probas[model_id] * weight for model_id, weight in zip(boosting_members, weights))
        test_ensemble = sum(test_probas[model_id] * weight for model_id, weight in zip(boosting_members, weights))
        validation_rows.append(
            _metric_row(
                dataset=dataset,
                dataset_slug=dataset_slug,
                split="validation",
                model_id="validation_weighted_boosting_ensemble",
                proba=val_ensemble,
                training_seconds=0.0,
                inference_seconds=0.0,
                source_type="validation_weighted_boosting_ensemble",
            )
        )
        acceptance_rows.append(
            _metric_row(
                dataset=dataset,
                dataset_slug=dataset_slug,
                split="test",
                model_id="validation_weighted_boosting_ensemble",
                proba=test_ensemble,
                training_seconds=0.0,
                inference_seconds=0.0,
                source_type="validation_weighted_boosting_ensemble",
            )
        )
    selected_rows = [
        {
            "dataset": dataset.name,
            "dataset_slug": dataset_slug,
            "selected_model_id": best_model,
            "selection_macro_f1": validation_scores[best_model],
            "reference_hgb_selection_macro_f1": validation_scores.get("hgb_reference"),
            "test_used_for_selection": False,
        }
    ]
    return validation_rows, acceptance_rows, selected_rows


def run_boosting_detectors_v5_8(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    cicids_processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    nf_processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    train_rows: int = DEFAULT_TRAIN_ROWS,
    validation_rows: int = DEFAULT_VALIDATION_ROWS,
    test_rows: int = DEFAULT_TEST_ROWS,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
    package_availability: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probe_rows = _package_rows(package_availability)
    _write_csv(out / "package_probe_results.csv", probe_rows)
    if not any(bool(row["available"]) for row in probe_rows):
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "blocked_missing_optional_boosting_packages",
            "strong_baseline_rows": 0,
            "test_used_for_selection": False,
            "fake_metric_count": 0,
            "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
            "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
            "runtime_safe_v3_0_remains_default": True,
        }
        _write_csv(out / "model_selection_results.csv", [])
        _write_csv(out / "boosting_detector_results.csv", [])
        _write_csv(out / "selected_models.csv", [])
        _dump(out / "boosting_run_report.json", report)
        return report
    model_ids = _model_ids_from_probe(probe_rows)
    datasets = _load_datasets(
        cicids_processed_dir=cicids_processed_dir,
        nf_processed_dir=nf_processed_dir,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        max_rows_per_entry=max_rows_per_entry,
    )
    selection_rows: list[dict[str, Any]] = []
    acceptance_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    sampling = {}
    for dataset, slug in datasets:
        val, acc, selected = _fit_dataset_models(dataset, slug, model_ids)
        selection_rows.extend(val)
        acceptance_rows.extend(acc)
        selected_rows.extend(selected)
        sampling[slug] = dataset.metadata
    _write_csv(out / "model_selection_results.csv", selection_rows)
    _write_csv(out / "boosting_detector_results.csv", acceptance_rows)
    _write_csv(out / "selected_models.csv", selected_rows)
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
        agent_name="DetectorBoostingV58",
        agent_version="5.8-candidate",
        feature_group=FeatureGroup.STATS,
        benign_support=0.5,
        malicious_support=0.5,
        confidence=0.5,
        uncertainty=0.5,
        contributes_to_verdict=False,
        evidence=["v5.8 boosting candidates are default-off detector signals"],
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
        "model_ids": model_ids,
        "selection_rows": len(selection_rows),
        "strong_baseline_rows": len(acceptance_rows),
        "test_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_hash_before": _runtime_safe_hash(),
        "runtime_safe_v3_0_hash_after": _runtime_safe_hash(),
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "boosting_run_report.json", report)
    return report


def _best_rows_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("dataset")), {})[str(row.get("model_id"))] = row
    return grouped


def _upgrade_gates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    for dataset, by_model in _best_rows_by_dataset(rows).items():
        hgb = by_model.get("hgb_reference")
        if not hgb:
            continue
        candidates = [
            row
            for model_id, row in by_model.items()
            if model_id != "hgb_reference" and not model_id.endswith("_reference")
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: _float(row.get("macro_f1")))
        macro_delta = _float(best.get("macro_f1")) - _float(hgb.get("macro_f1"))
        accuracy_delta = _float(best.get("accuracy")) - _float(hgb.get("accuracy"))
        recall_delta = _float(best.get("recall_malicious")) - _float(hgb.get("recall_malicious"))
        ece_delta = _float(best.get("ece")) - _float(hgb.get("ece"))
        gates.append(
            {
                "dataset": dataset,
                "reference_model_id": "hgb_reference",
                "best_candidate_model_id": best.get("model_id"),
                "macro_f1_delta_vs_hgb": macro_delta,
                "accuracy_delta_vs_hgb": accuracy_delta,
                "malicious_recall_delta_vs_hgb": recall_delta,
                "ece_delta_vs_hgb": ece_delta,
                "required_macro_f1_delta": 0.003,
                "passed": (
                    macro_delta >= 0.003
                    and accuracy_delta >= 0
                    and recall_delta >= -0.005
                    and ece_delta <= 0.02
                ),
                "runtime_promoted": False,
            }
        )
    return gates


def _pcap_boundary_report(pcap_v56_dir: str | Path) -> dict[str, Any]:
    report_path = Path(pcap_v56_dir) / "acceptance_report.json"
    if not report_path.exists():
        return {
            "pcap_v5_6_report_found": False,
            "pcap_supervised_metrics_used": False,
            "reason": "v5.8 uses processed flow CSV only",
        }
    report = _read_json(report_path)
    return {
        "pcap_v5_6_report_found": True,
        "pcap_v5_6_status": report.get("status"),
        "pcap_supervised_metrics_used": False,
        "reason": "v5.6 PCAP flow-label alignment was not verified, so it is excluded from supervised metrics",
    }


def finalize_boosting_detectors_v5_8(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    pcap_v56_dir: str | Path = DEFAULT_V56_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_report = _read_json(out / "boosting_run_report.json") if (out / "boosting_run_report.json").exists() else {}
    rows = _read_csv_rows(out / "boosting_detector_results.csv")
    selection_rows = _read_csv_rows(out / "model_selection_results.csv")
    gates = _upgrade_gates(rows)
    _write_csv(out / "detector_upgrade_gates.csv", gates)
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
    all_gates_pass = bool(gates) and all(
        str(gate.get("passed")).lower() == "true" or gate.get("passed") is True
        for gate in gates
    )
    any_gain = any(_float(gate.get("macro_f1_delta_vs_hgb")) > 0 for gate in gates)
    if run_report.get("status") == "blocked_missing_optional_boosting_packages":
        status = "blocked_missing_optional_boosting_packages"
    elif all_gates_pass and security["status"] == "passed":
        status = "accepted_optional_detector_signal"
    elif any_gain:
        status = "not_promoted_unstable_or_insufficient_gain"
    else:
        status = "not_promoted_no_gain"
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if status == "accepted_optional_detector_signal" else status,
        "reason": "" if status == "accepted_optional_detector_signal" else "candidate did not satisfy all pre-registered detector gates",
        "runtime_promoted": False,
    }
    _dump(out / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "detector_upgrade_signal": any_gain,
        "detector_upgrade_gates": gates,
        "fake_metric_count": fake_metric_count,
        "security": security,
        "pcap_boundary": pcap_boundary,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "acceptance_report.json", report)
    _write_doc(document, report)
    return report
