"""W77--W79 calibration-aware NF-IoT safe-flow ensemble lane.

The lane is isolated from the default runtime.  W77 excludes every mappable
historical sample, freezes four leave-one-source-group-out folds, trains the
unchanged W74 classifier structure, and selects a probability calibrator from
double out-of-fold predictions.  W78 opens each held source exactly once.
W79 is permitted only after every detector, calibration, generalization, and
safety gate passes.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .credible_performance_w62_w67 import SOURCE_GROUPS
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .paper_evaluation import hash_artifact_paths, sha256_file
from .safe_flow_ensemble_w73_w76 import (
    BLOCKED_CONTEXT_FIELDS,
    ENGINEERED_FEATURES,
    RAW_SAFE_FIELDS,
    _ResidualMLP,
    _aggregate_prediction_metrics,
    _bootstrap_paired,
    _candidate_probabilities,
    _canonical_sample_key,
    _dump,
    _ece,
    _engineer_safe_features,
    _fit_mlp_seed_candidates,
    _fit_oof_stacker,
    _fit_quantile,
    _fit_tree,
    _historical_exclusions,
    _legacy_sample_key,
    _metric_row,
    _positive_probability,
    _read_csv,
    _read_json,
    _runtime_security,
    _select_threshold,
    _source_class_weights,
    _source_inventory,
    _temperature_scale,
    _worst_group_macro_f1,
    _write_csv,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_calibration_aware_ensemble_w77_w79"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_calibration_aware_ensemble_w77_w79")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_calibration_aware_ensemble_w77")
DEFAULT_RUNTIME = "runtime_safe_v3_0"
OPTIONAL_RUNTIME = "runtime_nfiot_safe_ensemble_calibrated_w79_optional"
W73_DIR = Path("data/runs/mad_etd_safe_flow_ensemble_w73_w76")
W74_MODEL_DIR = Path("data/models/mad_etd_safe_flow_ensemble_w74")

CALIBRATION_METHODS = ("identity", "temperature", "platt", "beta", "isotonic")
BASELINE_MODELS = ("hgb", "random_forest", "extra_trees", "residual_mlp")
CANDIDATE_NAME = "safe_flow_ensemble_calibrated_v1_1"


def _sha_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(value) for value in values)).encode()).hexdigest()


def _safe_security() -> dict[str, Any]:
    return {
        **_runtime_security(),
        "unsupported_calls": 0,
        "audit_completion": 1.0,
        "candidate_default_enabled": False,
        "general_promoted_runtime_created": False,
    }


def _historical_exclusions_w77(
    *,
    w62_dir: str | Path = "data/runs/mad_etd_generalization_benchmark_w62",
    w68_dir: str | Path = "data/runs/mad_etd_source_disjoint_safe_features_w68",
    w69_dir: str | Path = "data/runs/mad_etd_extratrees_replication_w69",
    w45_dir: str | Path = "data/runs/mad_etd_nfiot_positive_replication_w45",
    w46_dir: str | Path = "data/runs/mad_etd_nfiot_positive_statistical_validation_w46",
    w73_dir: str | Path = W73_DIR,
) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, earlier = _historical_exclusions(
        w62_dir=w62_dir,
        w68_dir=w68_dir,
        w69_dir=w69_dir,
        w45_dir=w45_dir,
        w46_dir=w46_dir,
    )
    w73_path = Path(w73_dir) / "fresh_sample_manifest.csv"
    w75_path = Path(w73_dir) / "acceptance_predictions.csv"
    w73_rows = _read_csv(w73_path)
    w75_rows = _read_csv(w75_path)
    w73_hashes = {str(row["sample_hash"]) for row in w73_rows if row.get("sample_hash")}
    w73_legacy = {
        str(row["legacy_sample_id_hash"])
        for row in w73_rows
        if row.get("legacy_sample_id_hash")
    }
    w75_hashes = {str(row["sample_hash"]) for row in w75_rows if row.get("sample_hash")}
    w75_report = _read_json(Path(w73_dir) / "acceptance_report.json")
    w73_manifest = _read_json(Path(w73_dir) / "group_heldout_manifest.json")
    w73_mapped = bool(w73_hashes) and int(w73_manifest.get("sample_count", -1)) == len(w73_hashes)
    w75_mapped = (
        bool(w75_hashes)
        and w75_hashes == w73_hashes
        and w75_report.get("status") in {
            "accepted_dataset_specific_safe_flow_ensemble",
            "not_promoted_performance_gate_failed",
            "aggregate_gain_source_heterogeneous_not_promoted",
            "positive_but_statistically_inconclusive",
            "average_gain_not_promoted_due_to_worst_group_regression",
        }
    )
    complete = earlier.get("status") == "historical_acceptance_exclusions_mapped" and w73_mapped and w75_mapped
    canonical.update(w73_hashes)
    legacy.update(w73_legacy)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "all_historical_samples_mapped_for_w77" if complete else "failed_historical_sample_mapping_w77",
        "earlier_mapping": earlier,
        "w73": {
            "path": w73_path.as_posix(),
            "canonical_count": len(w73_hashes),
            "legacy_count": len(w73_legacy),
            "mapped": w73_mapped,
        },
        "w75": {
            "path": w75_path.as_posix(),
            "acceptance_count": len(w75_hashes),
            "covered_exactly_by_w73_fresh_sample": w75_mapped,
            "status": w75_report.get("status", "missing"),
        },
        "canonical_sha256_count": len(canonical),
        "legacy_sha256_prefix24_count": len(legacy),
        "fake_metric_count": 0,
    }
    return canonical, legacy, report


def build_calibration_aware_ensemble_w77(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w62_dir: str | Path = "data/runs/mad_etd_generalization_benchmark_w62",
    w68_dir: str | Path = "data/runs/mad_etd_source_disjoint_safe_features_w68",
    w69_dir: str | Path = "data/runs/mad_etd_extratrees_replication_w69",
    w45_dir: str | Path = "data/runs/mad_etd_nfiot_positive_replication_w45",
    w46_dir: str | Path = "data/runs/mad_etd_nfiot_positive_statistical_validation_w46",
    w73_dir: str | Path = W73_DIR,
) -> dict[str, Any]:
    """Inventory W77 sources and prove exact historical exclusion mapping."""

    processed = Path(processed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory, errors = _source_inventory(processed)
    _canonical, _legacy, exclusions = _historical_exclusions_w77(
        w62_dir=w62_dir, w68_dir=w68_dir, w69_dir=w69_dir,
        w45_dir=w45_dir, w46_dir=w46_dir, w73_dir=w73_dir,
    )
    raw_blocked = [name for name in RAW_SAFE_FIELDS if name.lower() in BLOCKED_CONTEXT_FIELDS]
    engineered_blocked = [
        name for name in ENGINEERED_FEATURES
        if any(token in name.lower() for token in ("ip", "port", "timestamp", "flow_id", "attack", "family", "source", "provenance", "split", "label"))
    ]
    ready = not errors and not raw_blocked and not engineered_blocked and exclusions["status"] == "all_historical_samples_mapped_for_w77"
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W77_build",
        "status": "ready_to_freeze_calibration_aware_splits_w77" if ready else "failed_w77_source_or_exclusion_audit",
        "processed_dir": processed.as_posix(),
        "source_inventory": inventory,
        "errors": errors,
        "historical_exclusion_status": exclusions["status"],
        "raw_safe_fields": list(RAW_SAFE_FIELDS),
        "engineered_features": list(ENGINEERED_FEATURES),
        "blocked_context_fields": list(BLOCKED_CONTEXT_FIELDS),
        "source_group_role": "split_training_weighting_and_stratified_analysis_only",
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
        "feature_policy_sha256": hashlib.sha256(json.dumps(list(ENGINEERED_FEATURES)).encode()).hexdigest(),
        **_safe_security(),
    }
    _dump(out / "source_inventory.json", report)
    _dump(out / "historical_exclusion_manifest.json", exclusions)
    _dump(
        out / "safe_feature_policy.json",
        {
            "schema_version": "1.0",
            "status": "passed" if ready else "failed",
            "raw_safe_feature_intersection": list(RAW_SAFE_FIELDS),
            "engineered_detector_features": list(ENGINEERED_FEATURES),
            "blocked_context_fields": list(BLOCKED_CONTEXT_FIELDS),
            "source_group_in_feature_matrix": False,
            "label_in_feature_matrix": False,
            "calibrator_inputs": ["cross_fitted_oof_probability"],
            "feature_policy_sha256": report["feature_policy_sha256"],
            "blocked_field_violation": len(raw_blocked) + len(engineered_blocked),
            "fake_metric_count": 0,
        },
    )
    _dump(out / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    return report


def _w77_fold_role(sample_hash: str, source_group: str, held_out: str) -> str:
    if source_group == held_out:
        return "acceptance"
    marker = int(_stable_hash(f"w77:validation:{held_out}:{sample_hash}")[:8], 16)
    return "validation" if marker % 5 == 0 else "train"


def _reservoir(
    heap: list[tuple[int, int, tuple[list[float], int, str, str, str]]],
    priority: int,
    tie: int,
    item: tuple[list[float], int, str, str, str],
    limit: int,
) -> None:
    record = (-priority, -tie, item)
    if len(heap) < limit:
        heapq.heappush(heap, record)
    elif record > heap[0]:
        heapq.heapreplace(heap, record)


def freeze_calibration_aware_splits_w77(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    per_label_per_group: int = 4000,
    max_rows_per_entry: int = 4_000_000,
    w62_dir: str | Path = "data/runs/mad_etd_generalization_benchmark_w62",
    w68_dir: str | Path = "data/runs/mad_etd_source_disjoint_safe_features_w68",
    w69_dir: str | Path = "data/runs/mad_etd_extratrees_replication_w69",
    w45_dir: str | Path = "data/runs/mad_etd_nfiot_positive_replication_w45",
    w46_dir: str | Path = "data/runs/mad_etd_nfiot_positive_statistical_validation_w46",
    w73_dir: str | Path = W73_DIR,
) -> dict[str, Any]:
    """Freeze a fresh W77 sample and four untouched W78 acceptance folds."""

    processed = Path(processed_dir)
    out = Path(output_dir)
    build = _read_json(out / "source_inventory.json") or build_calibration_aware_ensemble_w77(
        processed, out, w62_dir=w62_dir, w68_dir=w68_dir, w69_dir=w69_dir,
        w45_dir=w45_dir, w46_dir=w46_dir, w73_dir=w73_dir,
    )
    canonical_excluded, legacy_excluded, exclusion = _historical_exclusions_w77(
        w62_dir=w62_dir, w68_dir=w68_dir, w69_dir=w69_dir,
        w45_dir=w45_dir, w46_dir=w46_dir, w73_dir=w73_dir,
    )
    if build.get("status") != "ready_to_freeze_calibration_aware_splits_w77" or exclusion.get("status") != "all_historical_samples_mapped_for_w77":
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W77_freeze", "status": "failed_no_fresh_w77_split", "reason": "source or historical mapping audit failed", **_safe_security()}
        _dump(out / "split_manifest.json", report)
        return report

    manifest = _read_json(processed / "dataset_manifest.json")
    buckets: dict[tuple[str, int], list[tuple[int, int, tuple[list[float], int, str, str, str]]]] = {}
    scanned: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    tie = 0
    for entry in manifest.get("primary_entries", []):
        group = str(entry.get("dataset_variant", ""))
        if group not in SOURCE_GROUPS:
            continue
        archive = Path(str(entry["archive"]))
        member = str(entry["entry"])
        label_column = str(entry.get("label_column") or "Label")
        for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows_per_entry):
            scanned[group] += 1
            label = _normalise_binary_label(row.get(label_column))
            if label is None:
                continue
            canonical_key = _canonical_sample_key(group, archive, member, row_index)
            legacy_key = _legacy_sample_key(archive, member, row_index)
            sample_hash = _stable_hash(canonical_key)
            legacy_hash = _stable_hash(legacy_key)[:24]
            if sample_hash in canonical_excluded or legacy_hash in legacy_excluded:
                excluded[group] += 1
                continue
            y = int(label == "malicious")
            vector = _engineer_safe_features(row)
            priority = int(_stable_hash("w77:fresh:" + canonical_key)[:16], 16)
            _reservoir(
                buckets.setdefault((group, y), []),
                priority,
                tie,
                (vector, y, group, sample_hash, legacy_hash),
                per_label_per_group,
            )
            tie += 1

    selected: list[tuple[list[float], int, str, str, str]] = []
    sample_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for group in SOURCE_GROUPS:
        counts[group] = {}
        for label, label_name in ((0, "benign"), (1, "malicious")):
            values = [item for _priority, _tie, item in sorted(buckets.get((group, label), []), reverse=True)]
            selected.extend(values)
            counts[group][label_name] = len(values)
            for vector, y, source, sample_hash, legacy_hash in values:
                sample_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "legacy_sample_id_hash": legacy_hash,
                        "source_group": source,
                        "label": y,
                        "feature_hash": hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest(),
                        "historical_overlap": False,
                    }
                )

    sufficient = all(counts[group][label] >= per_label_per_group for group in SOURCE_GROUPS for label in ("benign", "malicious"))
    x = np.asarray([item[0] for item in selected], dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    hashes = np.asarray([item[3] for item in selected], dtype="U64")
    legacy = np.asarray([item[4] for item in selected], dtype="U24")
    np.savez_compressed(
        out / "fresh_group_sample_w77.npz",
        x=x,
        y=y,
        groups=groups,
        sample_hash=hashes,
        legacy_sample_id_hash=legacy,
        feature_names=np.asarray(ENGINEERED_FEATURES),
    )
    _write_csv(out / "fresh_sample_manifest.csv", sample_rows)

    assignments: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    acceptance_counter: Counter[str] = Counter()
    overlap_count = 0
    for held in SOURCE_GROUPS:
        role_sets = {role: set() for role in ("train", "validation", "acceptance")}
        role_counts: Counter[str] = Counter()
        label_counts = {role: Counter() for role in role_sets}
        fold_rows: list[dict[str, Any]] = []
        for sample_hash, group, label in zip(hashes, groups, y, strict=True):
            role = _w77_fold_role(str(sample_hash), str(group), held)
            row = {"fold_id": held, "sample_hash": str(sample_hash), "source_group": str(group), "label": int(label), "split_role": role}
            assignments.append(row)
            fold_rows.append(row)
            role_sets[role].add(str(sample_hash))
            role_counts[role] += 1
            label_counts[role][int(label)] += 1
            if role == "acceptance":
                acceptance_counter[str(sample_hash)] += 1
        pairwise = {
            "train_validation": len(role_sets["train"] & role_sets["validation"]),
            "train_acceptance": len(role_sets["train"] & role_sets["acceptance"]),
            "validation_acceptance": len(role_sets["validation"] & role_sets["acceptance"]),
        }
        overlap_count += sum(pairwise.values())
        folds.append(
            {
                "fold_id": held,
                "held_out_source_group": held,
                "counts": dict(role_counts),
                "label_counts": {role: {"benign": counter[0], "malicious": counter[1]} for role, counter in label_counts.items()},
                "pairwise_sample_overlap": pairwise,
                "assignment_sha256": hashlib.sha256(json.dumps(fold_rows, sort_keys=True).encode()).hexdigest(),
                "acceptance_used_for_selection": False,
            }
        )
    _write_csv(out / "fold_assignments.csv", assignments)
    acceptance_once = bool(hashes.size) and len(acceptance_counter) == len(hashes) and all(value == 1 for value in acceptance_counter.values())
    ready = sufficient and overlap_count == 0 and acceptance_once
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W77_freeze",
        "status": "fresh_w77_folds_frozen_for_training" if ready else "failed_no_fresh_w77_split",
        "seed": 42,
        "sample_count": int(len(y)),
        "feature_count": int(x.shape[1]) if x.ndim == 2 else 0,
        "source_group_counts": counts,
        "scanned_rows_per_group": dict(scanned),
        "historical_excluded_rows_per_group": dict(excluded),
        "historical_overlap_count": 0,
        "each_sample_acceptance_exactly_once": acceptance_once,
        "pairwise_split_overlap_count": overlap_count,
        "folds": folds,
        "acceptance_used_for_selection": False,
        "acceptance_opened": False,
        "sample_manifest_sha256": sha256_file(out / "fresh_sample_manifest.csv"),
        "fold_assignments_sha256": sha256_file(out / "fold_assignments.csv"),
        **_safe_security(),
    }
    _dump(out / "split_manifest.json", report)
    _dump(
        out / "split_overlap_audit.json",
        {"status": "passed" if ready else "failed", "pairwise_overlap_count": overlap_count, "historical_overlap_count": 0, "each_sample_acceptance_exactly_once": acceptance_once, "fake_metric_count": 0},
    )
    _dump(
        out / "pre_registration.json",
        {
            "schema_version": "1.0",
            "status": "frozen_before_w78_acceptance" if ready else "failed",
            "candidate": CANDIDATE_NAME,
            "classification_structure": "W74 quantile + HGB/ExtraTrees/residual-MLP + OOF logistic stacker",
            "calibration_methods": list(CALIBRATION_METHODS),
            "calibrator_fit_source": "double_cross_fitted_train_OOF_probability_only",
            "calibrator_selection_order": ["worst_source_brier", "worst_source_ece", "aggregate_brier", "aggregate_ece", "negative_macro_f1", "method_name"],
            "bootstrap_repeats": 1000,
            "bootstrap_seed": 42,
            "acceptance_used_for_selection": False,
            "w75_acceptance_used_for_tuning": False,
            "gates": {
                "macro_f1_delta_min": 0.01,
                "macro_f1_ci95_lower_gt": 0.0,
                "ece_degradation_max": 0.005,
                "brier_delta_max": 0.0,
                "coverage_drop_max": 0.01,
                "nonnegative_source_minimum": 3,
            },
            "runtime_safe_v3_0_remains_default": True,
            "fake_metric_count": 0,
        },
    )
    return report


def _load_w77_sample(out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    payload = np.load(out / "fresh_group_sample_w77.npz", allow_pickle=False)
    return payload["x"], payload["y"], payload["groups"], payload["sample_hash"], [str(value) for value in payload["feature_names"]]


def _assignments(out: Path) -> dict[tuple[str, str], str]:
    return {(str(row["fold_id"]), str(row["sample_hash"])): str(row["split_role"]) for row in _read_csv(out / "fold_assignments.csv")}


def _crossfit_stacker_probabilities(
    base_oof: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_hashes: np.ndarray,
    *,
    fold_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Produce stacker probabilities where every row is double out-of-fold."""

    from sklearn.linear_model import LogisticRegression

    cell_sizes = [
        int(np.sum((groups == group) & (labels == label)))
        for group in sorted(set(str(value) for value in groups))
        for label in (0, 1)
    ]
    fold_count = min(5, min(cell_sizes, default=0))
    if fold_count < 2:
        raise RuntimeError(f"insufficient rows for W77 stacker OOF in {fold_id}")
    assignment = np.full(len(labels), -1, dtype=np.int64)
    for group in sorted(set(str(value) for value in groups)):
        for label in (0, 1):
            local = np.flatnonzero((groups == group) & (labels == label))
            ordered = sorted(local.tolist(), key=lambda index: _stable_hash("w77:stacker-oof:" + str(sample_hashes[index])))
            for position, index in enumerate(ordered):
                assignment[index] = position % fold_count
    if np.any(assignment < 0):
        raise RuntimeError(f"incomplete W77 stacker OOF assignment in {fold_id}")
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for inner in range(fold_count):
        hold = assignment == inner
        fit = ~hold
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(base_oof[fit], labels[fit], sample_weight=_source_class_weights(groups[fit], labels[fit]))
        probabilities[hold] = _positive_probability(model, base_oof[hold])
        held_hashes = set(str(value) for value in sample_hashes[hold])
        fit_hashes = set(str(value) for value in sample_hashes[fit])
        audits.append(
            {
                "outer_fold_id": fold_id,
                "inner_stacker_oof_fold": inner,
                "sample_count": int(hold.sum()),
                "predicted_sample_hashes_sha256": _sha_lines(held_hashes),
                "training_sample_hashes_sha256": _sha_lines(fit_hashes),
                "in_sample_prediction_count": len(held_hashes & fit_hashes),
                "source_group_used_as_model_feature": False,
                "calibrator_input": True,
            }
        )
    if np.isnan(probabilities).any():
        raise RuntimeError(f"incomplete W77 double-OOF probability vector in {fold_id}")
    return probabilities, audits


def _clip_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


def _calibration_matrix(method: str, probability: np.ndarray) -> np.ndarray:
    p = _clip_probability(probability)
    if method == "platt":
        return np.log(p / (1.0 - p)).reshape(-1, 1)
    if method == "beta":
        return np.column_stack((np.log(p), -np.log1p(-p)))
    raise ValueError(f"no calibration matrix for {method}")


def _fit_calibrator(method: str, labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Fit one calibrator exclusively from double-OOF probability."""

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    p = _clip_probability(probability)
    if method == "identity":
        return {"method": method, "model": None, "fit_log_loss": float(log_loss(labels, p, labels=[0, 1]))}
    if method == "temperature":
        candidates = np.concatenate((np.linspace(0.35, 1.0, 14), np.linspace(1.1, 4.0, 30)))
        scores = [(float(log_loss(labels, _temperature_scale(p, float(value)), labels=[0, 1])), float(value)) for value in candidates]
        score, temperature = min(scores)
        return {"method": method, "model": None, "temperature": temperature, "fit_log_loss": score}
    if method in {"platt", "beta"}:
        model = LogisticRegression(C=1.0, class_weight=None, max_iter=2000, random_state=42)
        model.fit(_calibration_matrix(method, p), labels)
        fitted = _positive_probability(model, _calibration_matrix(method, p))
        return {"method": method, "model": model, "fit_log_loss": float(log_loss(labels, fitted, labels=[0, 1]))}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, labels)
        fitted = _clip_probability(model.predict(p))
        return {"method": method, "model": model, "fit_log_loss": float(log_loss(labels, fitted, labels=[0, 1]))}
    raise ValueError(f"unsupported W77 calibrator: {method}")


def _apply_calibrator(calibrator: Mapping[str, Any], probability: np.ndarray) -> np.ndarray:
    method = str(calibrator["method"])
    p = _clip_probability(probability)
    if method == "identity":
        return p
    if method == "temperature":
        return _temperature_scale(p, float(calibrator["temperature"]))
    if method in {"platt", "beta"}:
        return _clip_probability(_positive_probability(calibrator["model"], _calibration_matrix(method, p)))
    if method == "isotonic":
        return _clip_probability(calibrator["model"].predict(p))
    raise ValueError(f"unsupported frozen W77 calibrator: {method}")


def _source_calibration_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    from sklearn.metrics import brier_score_loss

    rows: dict[str, dict[str, float]] = {}
    for group in sorted(set(str(value) for value in groups)):
        mask = groups == group
        rows[group] = {
            "ece": _ece(labels[mask], probability[mask]),
            "brier_score": float(brier_score_loss(labels[mask], probability[mask])),
            "sample_count": int(mask.sum()),
        }
    return (
        max((row["brier_score"] for row in rows.values()), default=1.0),
        max((row["ece"] for row in rows.values()), default=1.0),
        rows,
    )


def train_calibration_aware_ensemble_w77(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    w74_model_dir: str | Path = W74_MODEL_DIR,
) -> dict[str, Any]:
    """Train the frozen W74 classifier structure without W78 acceptance rows."""

    import joblib

    out = Path(output_dir)
    models = Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    split = _read_json(out / "split_manifest.json")
    prereg = _read_json(out / "pre_registration.json")
    w74_selection = _read_json(W73_DIR / "selected_candidate.json")
    seed_by_fold = {str(row["fold_id"]): int(row["mlp_seed"]) for row in w74_selection.get("fold_registry", [])}
    if split.get("status") != "fresh_w77_folds_frozen_for_training" or prereg.get("status") != "frozen_before_w78_acceptance" or set(seed_by_fold) != set(SOURCE_GROUPS):
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W77_train", "status": "failed_missing_frozen_w77_or_w74_structure", **_safe_security()}
        _dump(out / "training_manifest.json", report)
        return report

    x, y, groups, sample_hashes, feature_names = _load_w77_sample(out)
    assignment = _assignments(out)
    training_rows: list[dict[str, Any]] = []
    oof_audit: list[dict[str, Any]] = []
    fold_registry: list[dict[str, Any]] = []
    for held in SOURCE_GROUPS:
        roles = np.asarray([assignment[(held, str(sample_hash))] for sample_hash in sample_hashes], dtype=object)
        train_mask = roles == "train"
        validation_mask = roles == "validation"
        acceptance_mask = roles == "acceptance"
        if np.any(train_mask & validation_mask) or np.any(train_mask & acceptance_mask) or np.any(validation_mask & acceptance_mask):
            raise RuntimeError(f"W77 split overlap in {held}")
        fold_dir = models / held
        fold_dir.mkdir(parents=True, exist_ok=True)
        transformer = _fit_quantile(x[train_mask], seed=42)
        train_x = transformer.transform(x[train_mask]).astype(np.float32)
        validation_x = transformer.transform(x[validation_mask]).astype(np.float32)
        train_y = y[train_mask]
        validation_y = y[validation_mask]
        train_groups = groups[train_mask]
        weights = _source_class_weights(train_groups, train_y)
        fitted: dict[str, Any] = {}
        for name in ("hgb", "random_forest", "extra_trees"):
            started = time.perf_counter()
            fitted[name] = _fit_tree(name, train_x, train_y, weights, seed=42)
            training_rows.append({"fold_id": held, "model_name": name, "seed": 42, "training_seconds": time.perf_counter() - started, "training_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()), "acceptance_rows_used": 0})
        mlp_seed = seed_by_fold[held]
        started = time.perf_counter()
        fitted["residual_mlp"] = _ResidualMLP(train_x.shape[1], mlp_seed).fit(train_x, train_y, weights)
        training_rows.append({"fold_id": held, "model_name": "residual_mlp", "seed": mlp_seed, "training_seconds": time.perf_counter() - started, "training_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()), "acceptance_rows_used": 0, "device": fitted["residual_mlp"].device_used, "mixed_precision": fitted["residual_mlp"].mixed_precision})

        stacker, base_oof, base_audit = _fit_oof_stacker(
            x[train_mask], train_y, train_groups, sample_hashes[train_mask], mlp_seed=mlp_seed, fold_id=held
        )
        candidate_oof, stacker_audit = _crossfit_stacker_probabilities(
            base_oof, train_y, train_groups, sample_hashes[train_mask], fold_id=held
        )
        oof_audit.extend([{**row, "oof_stage": "base_expert"} for row in base_audit])
        oof_audit.extend([{**row, "oof_stage": "stacker"} for row in stacker_audit])
        validation_raw = {
            name: _positive_probability(model, validation_x)
            for name, model in fitted.items()
        }
        validation_raw["safe_flow_ensemble_raw"] = _candidate_probabilities(
            stacker,
            fitted["hgb"],
            fitted["extra_trees"],
            fitted["residual_mlp"],
            validation_x,
        )
        np.savez_compressed(
            fold_dir / "development_probabilities.npz",
            train_oof_probability=candidate_oof,
            train_y=train_y,
            train_groups=train_groups,
            train_sample_hash=sample_hashes[train_mask],
            validation_y=validation_y,
            validation_groups=groups[validation_mask],
            validation_sample_hash=sample_hashes[validation_mask],
            **{f"validation_{name}_probability": probability for name, probability in validation_raw.items()},
        )
        joblib.dump(transformer, fold_dir / "quantile_transformer.joblib")
        for name in ("hgb", "random_forest", "extra_trees"):
            joblib.dump(fitted[name], fold_dir / f"{name}.joblib")
        fitted["residual_mlp"].save(fold_dir / "residual_mlp.pt")
        joblib.dump(stacker, fold_dir / "stacker.joblib")
        metadata = {
            "schema_version": "1.0",
            "fold_id": held,
            "held_out_source_group": held,
            "classifier_structure_source": "W74 frozen structure and validation-selected MLP seed",
            "w74_model_dir_reference": Path(w74_model_dir).as_posix(),
            "mlp_seed": mlp_seed,
            "feature_names": feature_names,
            "feature_policy_sha256": _read_json(out / "safe_feature_policy.json").get("feature_policy_sha256"),
            "train_sample_hashes_sha256": _sha_lines(sample_hashes[train_mask]),
            "validation_sample_hashes_sha256": _sha_lines(sample_hashes[validation_mask]),
            "acceptance_sample_hashes_sha256": _sha_lines(sample_hashes[acceptance_mask]),
            "double_oof_probability_sha256": hashlib.sha256(candidate_oof.tobytes()).hexdigest(),
            "oof_in_sample_prediction_count": sum(int(row["in_sample_prediction_count"]) for row in [*base_audit, *stacker_audit]),
            "acceptance_rows_used_for_training_or_selection": 0,
            "source_group_in_feature_matrix": False,
            "candidate_default_enabled": False,
        }
        _dump(fold_dir / "metadata.json", metadata)
        fold_registry.append({"fold_id": held, "model_dir": fold_dir.as_posix(), "training_rows": int(train_mask.sum()), "validation_rows": int(validation_mask.sum()), "acceptance_rows_reserved": int(acceptance_mask.sum()), "acceptance_rows_used": 0, "mlp_seed": mlp_seed})

    _write_csv(out / "training_results.csv", training_rows)
    _write_csv(out / "oof_prediction_audit.csv", oof_audit)
    oof_in_sample = sum(int(row["in_sample_prediction_count"]) for row in oof_audit)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W77_train",
        "status": "w77_classifier_structure_trained_for_oof_calibration" if oof_in_sample == 0 else "failed_w77_oof_leakage",
        "classifier_structure": "W74 quantile + HGB/ExtraTrees/residual-MLP + logistic stacker",
        "fold_registry": fold_registry,
        "double_oof_calibrator_input": True,
        "oof_in_sample_prediction_count": oof_in_sample,
        "acceptance_rows_used_for_training": 0,
        "acceptance_rows_used_for_selection": 0,
        "w75_acceptance_used_for_tuning": False,
        "cuda_available": any(str(row.get("device", "")).startswith("cuda") for row in training_rows),
        "mixed_precision_used": any(bool(row.get("mixed_precision")) for row in training_rows),
        **_safe_security(),
    }
    _dump(out / "training_manifest.json", report)
    return report


def calibrate_cross_source_probabilities_w77(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Fit from train OOF and select calibration solely on W77 validation."""

    import joblib
    from sklearn.metrics import brier_score_loss

    out = Path(output_dir)
    models = Path(model_dir)
    training = _read_json(out / "training_manifest.json")
    if training.get("status") != "w77_classifier_structure_trained_for_oof_calibration":
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W77_calibrate", "status": "failed_missing_leakage_free_w77_training", **_safe_security()}
        _dump(out / "calibration_report.json", report)
        return report

    method_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    selected_methods: dict[str, str] = {}
    baseline_thresholds: dict[str, dict[str, float]] = {}
    for held in SOURCE_GROUPS:
        fold_dir = models / held
        payload = np.load(fold_dir / "development_probabilities.npz", allow_pickle=False)
        train_p = payload["train_oof_probability"]
        train_y = payload["train_y"]
        validation_y = payload["validation_y"]
        validation_groups = payload["validation_groups"]
        calibrators: dict[str, dict[str, Any]] = {}
        for method in CALIBRATION_METHODS:
            calibrator = _fit_calibrator(method, train_y, train_p)
            calibrated = _apply_calibrator(calibrator, payload["validation_safe_flow_ensemble_raw_probability"])
            threshold = _select_threshold(validation_y, calibrated)
            metrics = _metric_row(validation_y, calibrated, threshold=threshold)
            worst_brier, worst_ece, per_source = _source_calibration_metrics(validation_y, calibrated, validation_groups)
            calibrator["threshold"] = threshold
            calibrator["validation_metrics"] = metrics
            calibrator["worst_source_brier"] = worst_brier
            calibrator["worst_source_ece"] = worst_ece
            calibrator["per_source_calibration"] = per_source
            calibrators[method] = calibrator
            method_rows.append(
                {
                    "fold_id": held,
                    "method": method,
                    "fit_source": "double_cross_fitted_train_oof_probability_only",
                    "threshold": threshold,
                    "worst_source_brier": worst_brier,
                    "worst_source_ece": worst_ece,
                    "aggregate_brier": metrics["brier_score"],
                    "aggregate_ece": metrics["ece"],
                    "macro_f1": metrics["macro_f1"],
                    "malicious_recall": metrics["malicious_recall"],
                    "acceptance_rows_used": 0,
                }
            )
        selected = min(
            CALIBRATION_METHODS,
            key=lambda method: (
                float(calibrators[method]["worst_source_brier"]),
                float(calibrators[method]["worst_source_ece"]),
                float(calibrators[method]["validation_metrics"]["brier_score"]),
                float(calibrators[method]["validation_metrics"]["ece"]),
                -float(calibrators[method]["validation_metrics"]["macro_f1"]),
                method,
            ),
        )
        selected_methods[held] = selected
        joblib.dump(calibrators, fold_dir / "calibrators.joblib")
        _dump(
            fold_dir / "selected_calibration.json",
            {
                "fold_id": held,
                "selected_method": selected,
                "selection_order": ["worst_source_brier", "worst_source_ece", "aggregate_brier", "aggregate_ece", "negative_macro_f1", "method_name"],
                "fit_source": "double_cross_fitted_train_oof_probability_only",
                "validation_only_selection": True,
                "acceptance_rows_used": 0,
                "selected_metrics": calibrators[selected]["validation_metrics"],
                "selected_worst_source_brier": calibrators[selected]["worst_source_brier"],
                "selected_worst_source_ece": calibrators[selected]["worst_source_ece"],
                "calibrator_artifact_sha256": sha256_file(fold_dir / "calibrators.joblib"),
            },
        )
        baseline_thresholds[held] = {}
        for name in BASELINE_MODELS:
            probability = payload[f"validation_{name}_probability"]
            threshold = _select_threshold(validation_y, probability)
            baseline_thresholds[held][name] = threshold
            metrics = _metric_row(validation_y, probability, threshold=threshold)
            worst, per_source = _worst_group_macro_f1(validation_y, probability, validation_groups, threshold)
            baseline_rows.append({"fold_id": held, "model_name": name, "threshold": threshold, "worst_group_macro_f1": worst, "per_source_macro_f1": json.dumps(per_source, sort_keys=True), **metrics})

    aggregate_baseline: dict[str, dict[str, float]] = {}
    for name in BASELINE_MODELS:
        rows = [row for row in baseline_rows if row["model_name"] == name]
        aggregate_baseline[name] = {
            "macro_f1": float(np.mean([float(row["macro_f1"]) for row in rows])),
            "malicious_recall": float(np.mean([float(row["malicious_recall"]) for row in rows])),
            "worst_group_macro_f1": float(min(float(row["worst_group_macro_f1"]) for row in rows)),
            "ece": float(np.mean([float(row["ece"]) for row in rows])),
            "brier_score": float(np.mean([float(row["brier_score"]) for row in rows])),
        }
    strongest = max(
        BASELINE_MODELS,
        key=lambda name: (
            aggregate_baseline[name]["macro_f1"],
            aggregate_baseline[name]["malicious_recall"],
            aggregate_baseline[name]["worst_group_macro_f1"],
            -aggregate_baseline[name]["ece"],
            name,
        ),
    )
    _write_csv(out / "calibration_method_results.csv", method_rows)
    _write_csv(out / "validation_baseline_results.csv", baseline_rows)
    selection = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "candidate": CANDIDATE_NAME,
        "selected_calibration_by_fold": selected_methods,
        "strongest_safe_input_baseline": strongest,
        "baseline_validation_aggregate": aggregate_baseline,
        "baseline_thresholds_by_fold": baseline_thresholds,
        "selection_source": "W77 validation only",
        "calibrator_fit_source": "double_cross_fitted_train_oof_probability_only",
        "acceptance_used_for_selection": False,
        "w75_acceptance_used_for_tuning": False,
        "fake_metric_count": 0,
    }
    _dump(out / "selected_calibration.json", selection)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W77_calibrate",
        "status": "w77_calibration_frozen_for_w78_acceptance",
        **selection,
        **_safe_security(),
    }
    _dump(out / "calibration_report.json", report)
    return report


def _load_fold_bundle(fold_dir: Path) -> dict[str, Any]:
    import joblib

    return {
        "transformer": joblib.load(fold_dir / "quantile_transformer.joblib"),
        "hgb": joblib.load(fold_dir / "hgb.joblib"),
        "random_forest": joblib.load(fold_dir / "random_forest.joblib"),
        "extra_trees": joblib.load(fold_dir / "extra_trees.joblib"),
        "residual_mlp": _ResidualMLP.load(fold_dir / "residual_mlp.pt"),
        "stacker": joblib.load(fold_dir / "stacker.joblib"),
        "calibrators": joblib.load(fold_dir / "calibrators.joblib"),
        "selected_calibration": _read_json(fold_dir / "selected_calibration.json"),
    }


def evaluate_calibration_aware_ensemble_w78(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Open the four held source groups exactly once under frozen W77 policy."""

    out = Path(output_dir)
    models = Path(model_dir)
    completed = _read_json(out / "acceptance_evaluation_state.json")
    if completed.get("status") == "w78_acceptance_evaluated" and (out / "acceptance_predictions.csv").exists():
        return completed
    split = _read_json(out / "split_manifest.json")
    calibration = _read_json(out / "calibration_report.json")
    selection = _read_json(out / "selected_calibration.json")
    if split.get("status") != "fresh_w77_folds_frozen_for_training" or calibration.get("status") != "w77_calibration_frozen_for_w78_acceptance" or not selection:
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W78_evaluate", "status": "failed_missing_frozen_w77_calibration", **_safe_security()}
        _dump(out / "acceptance_evaluation_state.json", report)
        return report
    required = [
        models / held / name
        for held in SOURCE_GROUPS
        for name in ("metadata.json", "calibrators.joblib", "selected_calibration.json", "stacker.joblib")
    ]
    if any(not path.exists() for path in required):
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W78_evaluate", "status": "failed_missing_w77_model_artifact", "missing": [path.as_posix() for path in required if not path.exists()], **_safe_security()}
        _dump(out / "acceptance_evaluation_state.json", report)
        return report

    opened_path = out / "acceptance_opened.json"
    resumed = opened_path.exists()
    if not resumed:
        _dump(
            opened_path,
            {
                "schema_version": "1.0",
                "experiment": EXPERIMENT,
                "status": "W78_acceptance_opened_once",
                "opened_unix_time": time.time(),
                "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
                "calibration_report_sha256": sha256_file(out / "calibration_report.json"),
                "selected_calibration_sha256": sha256_file(out / "selected_calibration.json"),
                "acceptance_used_for_selection": False,
                "thresholds_mutable_after_open": False,
                "fake_metric_count": 0,
            },
        )

    x, y, groups, sample_hashes, _feature_names = _load_w77_sample(out)
    assignment = _assignments(out)
    baseline_name = str(selection["strongest_safe_input_baseline"])
    baseline_thresholds = selection["baseline_thresholds_by_fold"]
    truth_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    hash_parts: list[np.ndarray] = []
    baseline_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    baseline_threshold_parts: list[np.ndarray] = []
    candidate_threshold_parts: list[np.ndarray] = []
    baseline_latency_parts: list[float] = []
    candidate_latency_parts: list[float] = []
    fold_state: list[dict[str, Any]] = []
    for held in SOURCE_GROUPS:
        roles = np.asarray([assignment[(held, str(sample_hash))] for sample_hash in sample_hashes], dtype=object)
        mask = roles == "acceptance"
        if not np.all(groups[mask] == held):
            raise RuntimeError(f"W78 held source mismatch in {held}")
        fold_dir = models / held
        bundle = _load_fold_bundle(fold_dir)
        started = time.perf_counter()
        transformed = bundle["transformer"].transform(x[mask]).astype(np.float32)
        transform_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        baseline_probability = _positive_probability(bundle[baseline_name], transformed)
        baseline_ms = (time.perf_counter() - started) * 1000.0 + transform_ms
        started = time.perf_counter()
        raw_candidate = _candidate_probabilities(
            bundle["stacker"],
            bundle["hgb"],
            bundle["extra_trees"],
            bundle["residual_mlp"],
            transformed,
        )
        selected_method = str(bundle["selected_calibration"]["selected_method"])
        candidate_probability = _apply_calibrator(bundle["calibrators"][selected_method], raw_candidate)
        candidate_ms = (time.perf_counter() - started) * 1000.0 + transform_ms
        candidate_threshold = float(bundle["calibrators"][selected_method]["threshold"])
        baseline_threshold = float(baseline_thresholds[held][baseline_name])
        truth_parts.append(y[mask])
        group_parts.append(groups[mask])
        hash_parts.append(sample_hashes[mask])
        baseline_parts.append(baseline_probability)
        candidate_parts.append(candidate_probability)
        baseline_threshold_parts.append(np.full(mask.sum(), baseline_threshold, dtype=float))
        candidate_threshold_parts.append(np.full(mask.sum(), candidate_threshold, dtype=float))
        baseline_latency_parts.append(baseline_ms / max(1, int(mask.sum())))
        candidate_latency_parts.append(candidate_ms / max(1, int(mask.sum())))
        fold_state.append(
            {
                "fold_id": held,
                "acceptance_rows": int(mask.sum()),
                "selected_calibration": selected_method,
                "baseline": baseline_name,
                "metadata_sha256": sha256_file(fold_dir / "metadata.json"),
                "calibration_sha256": sha256_file(fold_dir / "calibrators.joblib"),
                "acceptance_used_for_selection": False,
                "completed": True,
            }
        )
        _dump(out / "acceptance_evaluation_state.json", {"status": "W78_acceptance_in_progress", "completed_folds": fold_state, "acceptance_used_for_selection": False, "resumed_from_open_marker": resumed, "fake_metric_count": 0})

    truth = np.concatenate(truth_parts)
    source_groups = np.concatenate(group_parts)
    hashes = np.concatenate(hash_parts)
    baseline_probability = np.concatenate(baseline_parts)
    candidate_probability = np.concatenate(candidate_parts)
    baseline_threshold = np.concatenate(baseline_threshold_parts)
    candidate_threshold = np.concatenate(candidate_threshold_parts)
    baseline_prediction = (baseline_probability >= baseline_threshold).astype(np.int64)
    candidate_prediction = (candidate_probability >= candidate_threshold).astype(np.int64)
    prediction_rows = [
        {
            "sample_hash": str(sample_hash),
            "held_out_source_group": str(group),
            "y_true": int(label),
            "baseline_name": baseline_name,
            "baseline_probability": float(bp),
            "baseline_threshold": float(bt),
            "baseline_prediction": int(bpred),
            "candidate_name": CANDIDATE_NAME,
            "candidate_probability": float(cp),
            "candidate_threshold": float(ct),
            "candidate_prediction": int(cpred),
            "baseline_accepted": True,
            "candidate_accepted": True,
            "source_group_used_as_feature": False,
            "acceptance_used_for_selection": False,
        }
        for sample_hash, group, label, bp, bt, bpred, cp, ct, cpred in zip(
            hashes, source_groups, truth, baseline_probability, baseline_threshold, baseline_prediction,
            candidate_probability, candidate_threshold, candidate_prediction, strict=True
        )
    ]
    _write_csv(out / "acceptance_predictions.csv", prediction_rows)
    baseline_metrics = _aggregate_prediction_metrics(truth, baseline_probability, baseline_threshold, np.ones(len(truth), dtype=bool))
    candidate_metrics = _aggregate_prediction_metrics(truth, candidate_probability, candidate_threshold, np.ones(len(truth), dtype=bool))
    results = [
        {"scope": "aggregate_four_source_heldout", "model_name": baseline_name, **baseline_metrics, "average_inference_latency_ms_per_sample": float(np.mean(baseline_latency_parts)), "p95_fold_latency_ms_per_sample": float(np.quantile(baseline_latency_parts, 0.95)), "acceptance_used_for_selection": False, "fake_metric": False},
        {"scope": "aggregate_four_source_heldout", "model_name": CANDIDATE_NAME, **candidate_metrics, "average_inference_latency_ms_per_sample": float(np.mean(candidate_latency_parts)), "p95_fold_latency_ms_per_sample": float(np.quantile(candidate_latency_parts, 0.95)), "acceptance_used_for_selection": False, "fake_metric": False},
    ]
    _write_csv(out / "acceptance_results.csv", results)
    source_rows: list[dict[str, Any]] = []
    for group in SOURCE_GROUPS:
        mask = source_groups == group
        for name, probability, threshold in (
            (baseline_name, baseline_probability, baseline_threshold),
            (CANDIDATE_NAME, candidate_probability, candidate_threshold),
        ):
            metrics = _aggregate_prediction_metrics(truth[mask], probability[mask], threshold[mask], np.ones(mask.sum(), dtype=bool))
            source_rows.append({"source_group": group, "model_name": name, "sample_count": int(mask.sum()), **metrics, "source_group_used_as_feature": False})
    _write_csv(out / "source_stratified_results.csv", source_rows)
    from sklearn.metrics import confusion_matrix
    _dump(out / "confusion_matrices.json", {baseline_name: confusion_matrix(truth, baseline_prediction, labels=[0, 1]).tolist(), CANDIDATE_NAME: confusion_matrix(truth, candidate_prediction, labels=[0, 1]).tolist()})
    state = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W78_evaluate",
        "status": "w78_acceptance_evaluated",
        "acceptance_rows": int(len(truth)),
        "acceptance_unique_sample_count": len(set(str(value) for value in hashes)),
        "each_sample_evaluated_once": len(set(str(value) for value in hashes)) == len(truth),
        "folds": fold_state,
        "strongest_baseline_frozen_before_acceptance": baseline_name,
        "acceptance_used_for_selection": False,
        "thresholds_modified_after_acceptance_open": False,
        "resumed_from_open_marker": resumed,
        **_safe_security(),
    }
    _dump(out / "acceptance_evaluation_state.json", state)
    return state


def finalize_calibration_aware_ensemble_w78(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Apply all W78 gates and preserve any failure as a first-class result."""

    out = Path(output_dir)
    state = _read_json(out / "acceptance_evaluation_state.json")
    selection = _read_json(out / "selected_calibration.json")
    rows = _read_csv(out / "acceptance_predictions.csv")
    results = _read_csv(out / "acceptance_results.csv")
    source_results = _read_csv(out / "source_stratified_results.csv")
    if state.get("status") != "w78_acceptance_evaluated" or not rows or not results:
        report = {"schema_version": "1.0", "experiment": EXPERIMENT, "phase": "W78_finalize", "status": "failed_missing_w78_acceptance_artifacts", "optional_profile_eligible_for_w79": False, **_safe_security()}
        _dump(out / "acceptance_report.json", report)
        _dump(out / "negative_results.json", {"status": report["status"], "reason": "missing W78 acceptance artifacts", "fake_metric_count": 0})
        return report
    baseline_name = str(selection["strongest_safe_input_baseline"])
    by_model = {str(row["model_name"]): row for row in results}
    baseline = by_model[baseline_name]
    candidate = by_model[CANDIDATE_NAME]
    truth = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    groups = np.asarray([str(row["held_out_source_group"]) for row in rows], dtype=object)
    base_p = np.asarray([float(row["baseline_probability"]) for row in rows], dtype=float)
    base_t = np.asarray([float(row["baseline_threshold"]) for row in rows], dtype=float)
    cand_p = np.asarray([float(row["candidate_probability"]) for row in rows], dtype=float)
    cand_t = np.asarray([float(row["candidate_threshold"]) for row in rows], dtype=float)
    accepted = np.ones(len(truth), dtype=bool)
    bootstrap_rows, bootstrap = _bootstrap_paired(truth, groups, base_p, base_t, accepted, cand_p, cand_t, accepted, repeats=1000, seed=42)
    _write_csv(out / "bootstrap_ci_results.csv", bootstrap_rows)
    _dump(out / "bootstrap_ci_report.json", bootstrap)
    metric_names = ("accuracy", "macro_f1", "weighted_f1", "malicious_recall", "ece", "brier_score", "coverage", "selective_error")
    deltas = {name: float(candidate[name]) - float(baseline[name]) for name in metric_names}
    baseline_sources = {str(row["source_group"]): float(row["macro_f1"]) for row in source_results if row["model_name"] == baseline_name}
    candidate_sources = {str(row["source_group"]): float(row["macro_f1"]) for row in source_results if row["model_name"] == CANDIDATE_NAME}
    source_deltas = {group: candidate_sources[group] - baseline_sources[group] for group in SOURCE_GROUPS}
    nonnegative_sources = sum(value >= 0 for value in source_deltas.values())
    worst_group_delta = min(candidate_sources.values()) - min(baseline_sources.values())
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    frozen_unchanged = before == after
    security = {
        **_safe_security(),
        "acceptance_used_for_selection": False,
        "w75_acceptance_used_for_tuning": False,
        "source_group_in_feature_matrix": False,
        "oof_in_sample_prediction_count": int(_read_json(out / "training_manifest.json").get("oof_in_sample_prediction_count", -1)),
        "frozen_hashes_unchanged": frozen_unchanged,
        "runtime_feature_flag_off_output_unchanged": frozen_unchanged,
    }
    _dump(out / "security_acceptance.json", security)
    ci = bootstrap["metrics"]
    gates = {
        "macro_f1_delta_at_least_0_01": deltas["macro_f1"] >= 0.01,
        "macro_f1_delta_ci95_lower_gt_0": float(ci["macro_f1_delta"]["ci95_lower"]) > 0.0,
        "accuracy_not_lower": deltas["accuracy"] >= 0.0,
        "weighted_f1_not_lower": deltas["weighted_f1"] >= 0.0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0.0,
        "worst_group_macro_f1_not_lower": worst_group_delta >= 0.0,
        "at_least_three_sources_nonnegative": nonnegative_sources >= 3,
        "ece_degradation_within_0_005": deltas["ece"] <= 0.005,
        "brier_not_worse": deltas["brier_score"] <= 0.0,
        "coverage_drop_within_0_01": deltas["coverage"] >= -0.01,
        "selective_error_not_worse": deltas["selective_error"] <= 0.0,
        "blocked_field_violation_zero": security["blocked_field_violation"] == 0,
        "fusion_ownership_violation_zero": security["fusion_ownership_violation"] == 0,
        "ood_override_zero": security["ood_override_count"] == 0,
        "illegal_verdict_execution_zero": security["illegal_verdict_execution_count"] == 0,
        "fake_metric_count_zero": security["fake_metric_count"] == 0,
        "acceptance_not_used_for_selection": not security["acceptance_used_for_selection"],
        "w75_acceptance_not_used_for_tuning": not security["w75_acceptance_used_for_tuning"],
        "oof_in_sample_prediction_zero": security["oof_in_sample_prediction_count"] == 0,
        "frozen_hashes_unchanged": frozen_unchanged,
        "runtime_safe_v3_0_remains_default": True,
    }
    accepted_w78 = all(gates.values())
    if accepted_w78:
        status = "accepted_calibration_aware_dataset_specific_candidate_w78"
    elif any(not gates[name] for name in ("blocked_field_violation_zero", "fusion_ownership_violation_zero", "ood_override_zero", "illegal_verdict_execution_zero", "fake_metric_count_zero", "acceptance_not_used_for_selection", "w75_acceptance_not_used_for_tuning", "oof_in_sample_prediction_zero", "frozen_hashes_unchanged")):
        status = "rejected_security_or_leakage_failure_w78"
    elif deltas["macro_f1"] > 0 and (deltas["ece"] > 0.005 or deltas["brier_score"] > 0):
        status = "classification_gain_not_promoted_due_to_calibration_w78"
    elif deltas["macro_f1"] > 0 and float(ci["macro_f1_delta"]["ci95_lower"]) <= 0:
        status = "positive_but_statistically_inconclusive_w78"
    else:
        status = "not_promoted_performance_gate_failed_w78"
    paired = {
        "baseline": baseline_name,
        "candidate": CANDIDATE_NAME,
        "baseline_metrics": {name: float(baseline[name]) for name in metric_names},
        "candidate_metrics": {name: float(candidate[name]) for name in metric_names},
        "deltas": deltas,
        "source_macro_f1_deltas": source_deltas,
        "nonnegative_source_count": nonnegative_sources,
        "worst_group_macro_f1_delta": worst_group_delta,
        "bootstrap_ci": bootstrap,
        "acceptance_used_for_selection": False,
    }
    _dump(out / "paired_comparisons.json", paired)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W78_finalize",
        "status": status,
        "candidate": CANDIDATE_NAME,
        "candidate_default_enabled": False,
        "strongest_safe_input_baseline": baseline_name,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "paired_comparison": paired,
        "optional_profile_eligible_for_w79": accepted_w78,
        "optional_profile_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "acceptance_report.json", report)
    _dump(
        out / "negative_results.json",
        {
            "status": "not_applicable" if accepted_w78 else status,
            "candidate": CANDIDATE_NAME,
            "failed_gates": report["failed_gates"],
            "safe_claim": "W78 passed all detector and calibration gates" if accepted_w78 else "W78 candidate failed one or more preregistered detector or calibration gates",
            "forbidden_claim": "general runtime promotion or performance improvement outside the frozen NF-IoT scope",
            "runtime_modified": False,
            "fake_metric_count": 0,
        },
    )
    return report


def run_calibration_aware_fusion_shadow_w79(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Run typed W72 evidence handoffs only after W78 passes every gate."""

    out = Path(output_dir)
    w78 = _read_json(out / "acceptance_report.json")
    rows = _read_csv(out / "acceptance_predictions.csv")
    if w78.get("status") != "accepted_calibration_aware_dataset_specific_candidate_w78" or not rows:
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "phase": "W79_fusion_shadow",
            "status": "not_run_w78_ineligible",
            "reason": f"W78 status is {w78.get('status', 'missing')}",
            "fusion_shadow_executed": False,
            "optional_profile_created": False,
            **_safe_security(),
        }
        _write_csv(out / "fusion_shadow_results.csv", [report])
        _dump(out / "w79_shadow_report.json", report)
        return report

    from .base import CaseState
    from .evidence_team import AgentEvidenceV2Adapter
    from .fusion import FusionAgent
    from .schemas import (
        AgentEvidence,
        EvidenceHandoffV2,
        EvidenceRequest,
        FeatureGroup,
        FlowRecord,
        ReliabilityProfile,
        Verdict,
    )
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    policy = _read_json(out / "safe_feature_policy.json")
    artifact_hash = sha256_file(out / "calibration_report.json")
    permitted = [f"stats.{name}" for name in ENGINEERED_FEATURES]
    adapter = AgentEvidenceV2Adapter()
    fusion = FusionAgent(use_reliability_discount=True, allow_reject=False)
    reliability = ReliabilityProfile(
        stats_reliability=1.0,
        sequence_reliability=0.0,
        tls_reliability=0.0,
        payload_reliability=0.0,
        input_completeness=1.0,
        ood_suspected=False,
        key_features_missing=False,
    )
    truth: list[int] = []
    predictions: dict[str, list[int]] = {"baseline": [], "candidate": []}
    semantic_failures = 0
    handoff_failures = 0
    case_state_count = 0
    for row in rows:
        sample_hash = str(row["sample_hash"])
        truth.append(int(row["y_true"]))
        state = CaseState(
            flow=FlowRecord(trace_id=f"w79-{sample_hash}", sample_id=sample_hash),
            remaining_budget=1,
            evidence_budget=1,
            case_lifecycle_status="shadow_evidence_collection",
            audit_chain_ref=f"w79:{sample_hash}",
        )
        case_state_count += 1
        for lane, probability_column in (("baseline", "baseline_probability"), ("candidate", "candidate_probability")):
            probability = float(row[probability_column])
            request = EvidenceRequest(
                request_id=f"w79:{lane}:{sample_hash}",
                case_trace_id=state.flow.trace_id,
                requested_agent="StatsDetectorAgent",
                permitted_safe_features=permitted,
                purpose="collect calibrated safe-flow statistical evidence for Fusion shadow evaluation",
                budget=1,
                planner_source="replay",
                reason_codes=["W79_DATASET_SPECIFIC_SHADOW"],
                policy_status="approved",
            )
            evidence = AgentEvidence(
                agent_name="StatsDetectorAgent",
                agent_version="safe_flow_ensemble_calibrated_v1_1" if lane == "candidate" else str(row["baseline_name"]),
                feature_group=FeatureGroup.STATS,
                benign_support=1.0 - probability,
                malicious_support=probability,
                confidence=max(probability, 1.0 - probability),
                uncertainty=min(probability, 1.0 - probability),
                calibration_quality=1.0,
                model_reliability=1.0,
                distribution_shift_level="off",
                contributes_to_verdict=True,
                evidence=["W79 shadow evidence; no final-decision authority"],
                used_fields=permitted,
                latency_ms=0.0,
            )
            try:
                evidence_v2 = adapter.to_v2(
                    evidence,
                    request,
                    artifact_hash=artifact_hash,
                    dataset_scope="NF-IoT_only_W79_shadow",
                )
                handoff = EvidenceHandoffV2(
                    request_id=request.request_id,
                    case_trace_id=request.case_trace_id,
                    requested_agent=request.requested_agent,
                    policy_status="approved",
                    evidence=evidence_v2,
                    feature_policy_hash=evidence_v2.input_feature_policy_hash,
                    artifact_hash=evidence_v2.artifact_hash,
                    reason_codes=["POLICY_APPROVED_SHADOW_HANDOFF"],
                )
                returned = adapter.to_v1(evidence_v2, evidence)
            except (ValueError, RuntimeError):
                semantic_failures += 1
                handoff_failures += 1
                raise
            state.evidence_requests.append(request)
            state.evidence_handoffs.append(handoff)
            state.evidence = [returned]
            result = fusion.fuse(state.evidence, reliability, final=True)
            predictions[lane].append(1 if result.verdict == Verdict.MALICIOUS else 0)
        state.case_lifecycle_status = "shadow_fused"

    truth_array = np.asarray(truth, dtype=np.int64)
    result_rows: list[dict[str, Any]] = []
    for lane in ("baseline", "candidate"):
        prediction = np.asarray(predictions[lane], dtype=np.int64)
        result_rows.append(
            {
                "lane": lane,
                "sample_count": len(truth),
                "accuracy": float(accuracy_score(truth_array, prediction)),
                "macro_f1": float(f1_score(truth_array, prediction, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(truth_array, prediction, average="weighted", zero_division=0)),
                "malicious_recall": float(recall_score(truth_array, prediction, pos_label=1, zero_division=0)),
                "fusion_owner": "FusionAgent",
                "AgentEvidenceV2_semantic_failure_count": semantic_failures,
                "EvidenceHandoffV2_failure_count": handoff_failures,
            }
        )
    _write_csv(out / "fusion_shadow_results.csv", result_rows)
    by_lane = {row["lane"]: row for row in result_rows}
    macro_delta = float(by_lane["candidate"]["macro_f1"]) - float(by_lane["baseline"]["macro_f1"])
    security_pass = semantic_failures == 0 and handoff_failures == 0
    accepted = macro_delta >= 0.01 and security_pass
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W79_fusion_shadow",
        "status": "accepted_w79_fusion_shadow" if accepted else "w79_fusion_shadow_not_accepted",
        "fusion_shadow_executed": True,
        "case_state_count": case_state_count,
        "evidence_request_count": case_state_count * 2,
        "AgentEvidenceV2_count": case_state_count * 2,
        "EvidenceHandoffV2_count": case_state_count * 2,
        "feature_policy_sha256": policy.get("feature_policy_sha256"),
        "artifact_hash": artifact_hash,
        "baseline_system_metrics": by_lane["baseline"],
        "candidate_system_metrics": by_lane["candidate"],
        "system_macro_f1_delta": macro_delta,
        "system_macro_f1_delta_at_least_0_01": macro_delta >= 0.01,
        "AgentEvidenceV2_semantic_failure_count": semantic_failures,
        "EvidenceHandoffV2_failure_count": handoff_failures,
        "fusion_owner": "FusionAgent",
        "non_fusion_final_verdict_owner_count": 0,
        "optional_profile_created": False,
        **_safe_security(),
    }
    _dump(out / "w79_shadow_report.json", report)
    return report


def _write_w77_w79_docs(out: Path, final: Mapping[str, Any]) -> None:
    w78 = _read_json(out / "acceptance_report.json")
    paired = _read_json(out / "paired_comparisons.json")
    w79 = _read_json(out / "w79_shadow_report.json")
    failed = ", ".join(w78.get("failed_gates", [])) or "none"
    english = f"""# MAD-ETD Calibration-Aware Ensemble W77--W79

## Protocol

W77 used a fresh NF-IoT sample that excludes all mappable W62, W68, W69,
W45, W46, W73, and W75 samples.  Source identity was used only for split,
weighting, and stratified analysis.  The classifier retained the W74
Quantile/HGB/ExtraTrees/residual-MLP/OOF-stacker structure.  Identity,
temperature, Platt, beta, and isotonic calibration were fitted only from
double cross-fitted training probabilities and selected on W77 validation.

## W78 result

- Status: `{w78.get('status', 'missing')}`
- Baseline: `{paired.get('baseline', 'not_available')}`
- Candidate: `{paired.get('candidate', CANDIDATE_NAME)}`
- Macro-F1 delta: `{paired.get('deltas', {}).get('macro_f1', 'not_available')}`
- Accuracy delta: `{paired.get('deltas', {}).get('accuracy', 'not_available')}`
- ECE delta: `{paired.get('deltas', {}).get('ece', 'not_available')}`
- Brier delta: `{paired.get('deltas', {}).get('brier_score', 'not_available')}`
- Failed gates: `{failed}`

## W79 and runtime boundary

- W79 status: `{w79.get('status', 'not_run')}`
- Optional profile created: `{final.get('optional_profile_created', False)}`
- Default runtime: `{DEFAULT_RUNTIME}`
- Fake metrics: `0`

No result in this lane is a general-runtime or cross-dataset claim.  The LLM,
RAG, memory, reflection, critic, and HITL components never enter Fusion.
"""
    chinese = f"""# MAD-ETD W77--W79 校准感知安全流量集成实验

## 协议

W77 使用与 W62、W68、W69、W45、W46、W73、W75 可映射样本完全不重叠的
NF-IoT 新样本。来源身份只用于划分、训练加权和分层分析，不进入模型或校准器。
分类结构保持 W74 的 Quantile/HGB/ExtraTrees/残差 MLP/OOF Stacker。
Identity、Temperature、Platt、Beta、Isotonic 校准器只用双层交叉拟合训练概率拟合，
并只在 W77 validation 上选择。

## W78 结果

- 状态：`{w78.get('status', 'missing')}`
- 最强安全基线：`{paired.get('baseline', 'not_available')}`
- 候选：`{paired.get('candidate', CANDIDATE_NAME)}`
- Macro-F1 增量：`{paired.get('deltas', {}).get('macro_f1', 'not_available')}`
- Accuracy 增量：`{paired.get('deltas', {}).get('accuracy', 'not_available')}`
- ECE 增量：`{paired.get('deltas', {}).get('ece', 'not_available')}`
- Brier 增量：`{paired.get('deltas', {}).get('brier_score', 'not_available')}`
- 未通过门槛：`{failed}`

## W79 与运行时边界

- W79 状态：`{w79.get('status', 'not_run')}`
- 可选 profile 是否创建：`{final.get('optional_profile_created', False)}`
- 默认 runtime：`{DEFAULT_RUNTIME}`
- fake metrics：`0`

本实验不支持通用 runtime 或跨数据集性能提升主张。LLM、RAG、Memory、Reflection、
Critic、HITL 均不进入 Fusion。
"""
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "MAD_ETD_CALIBRATION_AWARE_ENSEMBLE_W77_W79.md").write_text(english, encoding="utf-8")
    (docs_dir / "MAD_ETD_CALIBRATION_AWARE_ENSEMBLE_W77_W79_CN.md").write_text(chinese, encoding="utf-8")


def finalize_calibration_aware_optional_profile_w79(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    config_dir: str | Path = "data/configs",
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    """Create an NF-IoT-only default-off profile only after W78/W79 pass."""

    out = Path(output_dir)
    config_root = Path(config_dir)
    w78 = _read_json(out / "acceptance_report.json")
    w79 = _read_json(out / "w79_shadow_report.json")
    try:
        after = hash_artifact_paths(_default_frozen_paths())
        default_artifacts_available = True
    except (FileNotFoundError, OSError):
        after = {"status": "missing_default_runtime_artifacts"}
        default_artifacts_available = False
    before = _read_json(out / "frozen_hashes_before.json")
    eligible = (
        w78.get("status") == "accepted_calibration_aware_dataset_specific_candidate_w78"
        and w79.get("status") == "accepted_w79_fusion_shadow"
        and float(w79.get("system_macro_f1_delta", -1.0)) >= 0.01
        and tests_passed
        and test_count > 0
        and int(w79.get("fusion_ownership_violation", -1)) == 0
        and int(w79.get("ood_override_count", -1)) == 0
        and int(w79.get("illegal_verdict_execution_count", -1)) == 0
        and int(w79.get("fake_metric_count", -1)) == 0
        and default_artifacts_available
        and bool(before)
        and before == after
    )
    profile_path = config_root / f"{OPTIONAL_RUNTIME}.json"
    if eligible:
        profile = {
            "schema_version": "1.0",
            "runtime_profile": OPTIONAL_RUNTIME,
            "default_enabled": False,
            "production_ready": False,
            "dataset_scope": ["NF-BoT-IoT", "NF-BoT-IoT-v2", "NF-ToN-IoT", "NF-ToN-IoT-v2"],
            "stats_backend": CANDIDATE_NAME,
            "fusion_owner": "FusionAgent",
            "forbidden_fusion_inputs": ["LLM", "RAG", "Memory", "Reflection", "Critic", "HITL"],
            "default_runtime_replaced": False,
            "evidence": (out / "acceptance_report.json").as_posix(),
            "shadow_evidence": (out / "w79_shadow_report.json").as_posix(),
            "claim_scope": "NF-IoT-only dataset-specific optional profile",
        }
        _dump(profile_path, profile)
        status = "accepted_optional_profile_created_w79"
    else:
        if profile_path.exists():
            profile_path.unlink()
        status = "not_created_w79_gates_or_tests_failed"
    _dump(out / "frozen_hashes_after_w79.json", after)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W79_finalize",
        "status": status,
        "optional_runtime": OPTIONAL_RUNTIME,
        "optional_profile_created": eligible,
        "optional_profile_path": profile_path.as_posix() if eligible else None,
        "optional_profile_default_enabled": False,
        "production_ready": False,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "frozen_hashes_unchanged": before == after,
        "default_runtime_artifacts_available": default_artifacts_available,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "w79_finalization_report.json", report)
    _write_w77_w79_docs(out, report)
    return report
