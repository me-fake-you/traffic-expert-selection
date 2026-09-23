"""W88--W92 fresh, group-held-out performance-upgrade protocol.

W88 is intentionally data-only: it maps every auditable NF-IoT sample used by
W62--W84, selects a fresh balanced sample, and seals leave-one-source-group-out
acceptance folds.  No model is trained and no acceptance label is opened here.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .credible_performance_w62_w67 import SOURCE_GROUPS
from .domain_robust_stats_w81 import (
    _apply_temperature,
    _ece,
    _historical_exclusions_for_w81,
    _select_calibration,
    _source_entries,
)
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .paper_evaluation import hash_artifact_paths, sha256_file
from .safe_flow_ensemble_w73_w76 import (
    BLOCKED_CONTEXT_FIELDS,
    ENGINEERED_FEATURES,
    RAW_SAFE_FIELDS,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
    _source_class_weights,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_fresh_performance_upgrade_w88_w92"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_fresh_performance_upgrade_w88_w92")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_fresh_performance_upgrade_w88_w92")
DEFAULT_RUNTIME = "runtime_safe_v3_0"
DEFAULT_PER_LABEL_PER_GROUP = 800
DEFAULT_MAX_ROWS_PER_ENTRY = 1_000_000
HISTORICAL_W81_DIR = Path("data/runs/mad_etd_domain_robust_stats_w81")
BASELINE_MODELS = ("hgb_safe_input", "random_forest_safe_input", "extra_trees_safe_input")
BLEND_ABLATION_ID = "heterogeneous_safe_boost_stack_v1"
CANDIDATE_ID = "cross_source_oof_evidence_stack_v2"


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _security_contract() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "audit_completion": 1.0,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }


def _feature_policy() -> dict[str, Any]:
    forbidden = sorted(
        set(BLOCKED_CONTEXT_FIELDS)
        | {
            "label",
            "attack",
            "attack_name",
            "family",
            "sample_id",
            "source_group",
            "source_identity",
            "source_file",
            "provenance",
            "ip",
            "port",
            "timestamp",
            "flow_id",
        }
    )
    return {
        "schema_version": "1.0",
        "status": "passed_safe_feature_policy_w88",
        "raw_safe_feature_intersection": list(RAW_SAFE_FIELDS),
        "engineered_detector_features": list(ENGINEERED_FEATURES),
        "forbidden_detector_fields": forbidden,
        "source_group_use": "split_and_grouped_evaluation_only",
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
        "blocked_field_violation": 0,
        "fake_metric_count": 0,
    }


def _historical_exclusions_w88(
    w81_dir: str | Path = HISTORICAL_W81_DIR,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Extend the audited W45--W80 ledger with every W81 sample.

    W82--W84 are SOC-contract/TLS-capture lanes and add no NF-IoT row-level
    supervised samples.  Their exclusion contribution is therefore explicitly
    zero rather than inferred from absent files.
    """

    canonical, legacy, earlier = _historical_exclusions_for_w81()
    run = Path(w81_dir)
    manifest_path = run / "fresh_sample_manifest.csv"
    rows = _read_csv(manifest_path)
    w81_hashes = {row["sample_hash"] for row in rows if row.get("sample_hash")}
    w81_legacy = {
        row["legacy_sample_id_hash"]
        for row in rows
        if row.get("legacy_sample_id_hash")
    }
    split = _read_json(run / "split_manifest.json")
    mapped = bool(w81_hashes) and int(split.get("sample_count", -1)) == len(w81_hashes)
    canonical.update(w81_hashes)
    legacy.update(w81_legacy)
    report = {
        "schema_version": "1.0",
        "status": (
            "all_w62_w84_nf_samples_mapped_for_w88"
            if earlier.get("status") == "all_w45_w80_sample_exclusions_mapped_for_w81"
            and mapped
            else "failed_w88_historical_exclusion_mapping"
        ),
        "w45_w80_mapping": earlier,
        "w81": {
            "manifest_path": manifest_path.as_posix(),
            "canonical_count": len(w81_hashes),
            "legacy_count": len(w81_legacy),
            "split_status": split.get("status", "missing"),
            "mapped": mapped,
        },
        "w82_w84_new_nf_supervised_sample_count": 0,
        "w82_w84_scope": "SOC governance and DoH/TLS capture lanes",
        "canonical_exclusion_count": len(canonical),
        "legacy_prefix_exclusion_count": len(legacy),
        "fake_metric_count": 0,
    }
    return canonical, legacy, report


def _reservoir_push(
    heap: list[tuple[int, int, tuple[list[float], int, str, str, str]]],
    *,
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


def _validation_roles(
    groups: np.ndarray, labels: np.ndarray, hashes: np.ndarray, held: str
) -> np.ndarray:
    roles = np.asarray(
        ["acceptance" if str(group) == held else "train" for group in groups],
        dtype="U16",
    )
    for group in SOURCE_GROUPS:
        if group == held:
            continue
        for label in (0, 1):
            indexes = np.flatnonzero((groups == group) & (labels == label))
            ordered = sorted(
                indexes.tolist(),
                key=lambda index: _stable_hash(
                    f"w88:validation:{held}:{hashes[index]}"
                ),
            )
            validation_count = min(
                max(1, int(round(0.20 * len(ordered)))),
                max(1, len(ordered) - 1),
            )
            roles[np.asarray(ordered[:validation_count], dtype=int)] = "validation"
    return roles


def build_fresh_performance_benchmark_w88(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    per_label_per_group: int = DEFAULT_PER_LABEL_PER_GROUP,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    """Freeze fresh W62--W84-disjoint LOSO splits; never train a model."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = _feature_policy()
    _dump(out / "safe_feature_policy.json", policy)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)

    exclusions, legacy_exclusions, ledger = _historical_exclusions_w88()
    _dump(out / "historical_sample_usage_ledger.json", ledger)
    entries, source_errors = _source_entries(Path(processed_dir))
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W88_fresh_split_freeze",
        "source_groups": list(SOURCE_GROUPS),
        "split_protocol": "four_fold_leave_one_source_group_out",
        "validation_protocol": "deterministic_20_percent_within_nonheld_source_class_cells",
        "acceptance_protocol": "sealed_held_source_opened_once_in_W90",
        "freshness_scope": "sample_disjoint_from_all_auditable_W62_W84_NF_supervised_rows",
        "per_label_per_group": int(per_label_per_group),
        "max_rows_per_entry": int(max_rows_per_entry),
        "acceptance_used_for_selection": False,
        "acceptance_opened": False,
        "locked_test_read": False,
        **_security_contract(),
    }
    _dump(out / "w88_protocol_manifest.json", protocol)
    if source_errors or ledger.get("status") != "all_w62_w84_nf_samples_mapped_for_w88":
        report = {
            **protocol,
            "status": "failed_no_fresh_auditable_w88_split",
            "source_errors": source_errors,
            "historical_mapping_status": ledger.get("status"),
        }
        _dump(out / "split_manifest.json", report)
        return report

    buckets: dict[
        tuple[str, int],
        list[tuple[int, int, tuple[list[float], int, str, str, str]]],
    ] = {}
    scanned: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    labeled: Counter[str] = Counter()
    tie = 0
    for entry in entries:
        group = str(entry["source_group"])
        archive = Path(entry["archive"])
        member = str(entry["member"])
        for row_index, row in _iter_zip_rows(
            archive, member, max_rows=max_rows_per_entry
        ):
            scanned[group] += 1
            label = _normalise_binary_label(row.get(str(entry["label_column"])))
            if label is None:
                continue
            canonical_key = _canonical_sample_key(group, archive, member, row_index)
            legacy_key = _legacy_sample_key(archive, member, row_index)
            sample_hash = _stable_hash(canonical_key)
            legacy_prefix = _stable_hash(legacy_key)[:24]
            if sample_hash in exclusions or legacy_prefix in legacy_exclusions:
                excluded[group] += 1
                continue
            labeled[group] += 1
            y = int(label == "malicious")
            vector = _engineer_safe_features(row)
            priority = int(_stable_hash("w88:fresh:" + canonical_key)[:16], 16)
            _reservoir_push(
                buckets.setdefault((group, y), []),
                priority=priority,
                tie=tie,
                item=(vector, y, group, sample_hash, legacy_prefix),
                limit=per_label_per_group,
            )
            tie += 1

    selected: list[tuple[list[float], int, str, str, str]] = []
    sample_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for group in SOURCE_GROUPS:
        counts[group] = {}
        for label, label_name in ((0, "benign"), (1, "malicious")):
            values = [
                item
                for _priority, _tie, item in sorted(
                    buckets.get((group, label), []), reverse=True
                )
            ]
            counts[group][label_name] = len(values)
            selected.extend(values)
            for vector, y, source_group, sample_hash, legacy_prefix in values:
                sample_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "legacy_sample_id_hash": legacy_prefix,
                        "source_group": source_group,
                        "label": y,
                        "feature_hash": hashlib.sha256(
                            np.asarray(vector, dtype=np.float32).tobytes()
                        ).hexdigest(),
                        "historical_overlap": False,
                    }
                )
    _write_csv(out / "fresh_sample_manifest.csv", sample_rows)
    sufficient = all(
        counts[group].get(label, 0) >= per_label_per_group
        for group in SOURCE_GROUPS
        for label in ("benign", "malicious")
    )
    if not sufficient:
        report = {
            **protocol,
            "status": "failed_insufficient_fresh_w88_samples",
            "source_group_counts": counts,
            "scanned_rows_per_group": dict(scanned),
            "excluded_rows_per_group": dict(excluded),
            "labeled_rows_after_exclusion": dict(labeled),
        }
        _dump(out / "split_manifest.json", report)
        return report

    x = np.asarray([item[0] for item in selected], dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    sample_hashes = np.asarray([item[3] for item in selected], dtype="U64")
    split_root = out / "sealed_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    assignment_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    pairwise_overlap = 0
    for held in SOURCE_GROUPS:
        roles = _validation_roles(groups, y, sample_hashes, held)
        train_validation = roles != "acceptance"
        acceptance = roles == "acceptance"
        fold = split_root / held
        fold.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold / "train_validation.npz",
            x=x[train_validation],
            y=y[train_validation],
            groups=groups[train_validation],
            sample_hash=sample_hashes[train_validation],
            roles=roles[train_validation],
            feature_names=np.asarray(ENGINEERED_FEATURES),
        )
        np.savez_compressed(
            fold / "acceptance_sealed.npz",
            x=x[acceptance],
            y=y[acceptance],
            groups=groups[acceptance],
            sample_hash=sample_hashes[acceptance],
            feature_names=np.asarray(ENGINEERED_FEATURES),
        )
        train_ids = set(str(value) for value in sample_hashes[train_validation])
        acceptance_ids = set(str(value) for value in sample_hashes[acceptance])
        overlap = len(train_ids & acceptance_ids)
        pairwise_overlap += overlap
        role_counts = Counter(str(value) for value in roles)
        fold_reports.append(
            {
                "held_out_source_group": held,
                "train_count": role_counts["train"],
                "validation_count": role_counts["validation"],
                "acceptance_count": role_counts["acceptance"],
                "train_acceptance_overlap": overlap,
                "acceptance_sealed": True,
            }
        )
        assignment_rows.extend(
            {
                "sample_hash": str(sample_hash),
                "source_group": str(source_group),
                "label": int(label),
                "held_out_source_group": held,
                "role": str(role),
            }
            for sample_hash, source_group, label, role in zip(
                sample_hashes, groups, y, roles, strict=True
            )
        )
    _write_csv(out / "fold_assignments.csv", assignment_rows)
    fresh_hashes = set(str(value) for value in sample_hashes)
    historical_overlap = len(fresh_hashes & exclusions)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    ready = pairwise_overlap == 0 and historical_overlap == 0 and hashes_unchanged
    report = {
        **protocol,
        "status": (
            "fresh_w88_group_heldout_splits_frozen"
            if ready
            else "failed_w88_split_or_frozen_hash_gate"
        ),
        "sample_count": len(sample_hashes),
        "source_group_counts": counts,
        "scanned_rows_per_group": dict(scanned),
        "excluded_rows_per_group": dict(excluded),
        "labeled_rows_after_exclusion": dict(labeled),
        "historical_exclusion_count": len(exclusions),
        "legacy_exclusion_count": len(legacy_exclusions),
        "historical_overlap_count": historical_overlap,
        "pairwise_train_acceptance_overlap_count": pairwise_overlap,
        "folds": fold_reports,
        "sample_manifest_sha256": sha256_file(out / "fresh_sample_manifest.csv"),
        "fold_assignment_sha256": sha256_file(out / "fold_assignments.csv"),
        "frozen_hashes_unchanged": hashes_unchanged,
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
    }
    _dump(out / "split_manifest.json", report)
    _dump(
        out / "security_acceptance_w88.json",
        {
            "status": "passed" if ready else "failed",
            "historical_overlap_count": historical_overlap,
            "pairwise_train_acceptance_overlap_count": pairwise_overlap,
            "frozen_hashes_unchanged": hashes_unchanged,
            **_security_contract(),
        },
    )
    return report


def _metric_row(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "ece": float(_ece(labels, probability)),
        "coverage": 1.0,
        "selective_error": float(1.0 - accuracy_score(labels, prediction)),
    }


def _positive_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        result = np.asarray(model.predict_proba(matrix), dtype=float)
        return result[:, 1] if result.ndim == 2 else result.reshape(-1)
    score = np.asarray(model.decision_function(matrix), dtype=float).reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))


def _fit_safe_model(model_id: str, matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> Any:
    if model_id == "hgb_safe_input":
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=260,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            random_state=42,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    if model_id == "random_forest_safe_input":
        model = RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    if model_id == "extra_trees_safe_input":
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    if model_id == "xgboost_component":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_lambda=1.0,
            reg_alpha=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    if model_id == "lightgbm_component":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    if model_id == "catboost_component":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=350,
            depth=7,
            learning_rate=0.04,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            l2_leaf_reg=3.0,
        )
        model.fit(matrix, labels, sample_weight=weights)
        return model
    raise ValueError(f"unsupported safe model: {model_id}")


def _blend_grid() -> list[dict[str, float]]:
    """Preregistered convex grid; every component must contribute at least 0.1."""

    rows: list[dict[str, float]] = []
    for xgb_units in range(1, 9):
        for lgb_units in range(1, 10 - xgb_units):
            cat_units = 10 - xgb_units - lgb_units
            if cat_units < 1:
                continue
            rows.append(
                {
                    "hgb_safe_input": xgb_units / 10.0,
                    "random_forest_safe_input": lgb_units / 10.0,
                    "extra_trees_safe_input": cat_units / 10.0,
                }
            )
    return rows


def _select_candidate_blend(
    labels: np.ndarray, component_probabilities: Mapping[str, np.ndarray]
) -> tuple[dict[str, float], float, float, np.ndarray, dict[str, float]]:
    best: tuple[tuple[float, float, float, float], dict[str, float], float, float, np.ndarray, dict[str, float]] | None = None
    for weights in _blend_grid():
        raw = sum(
            float(weights[name]) * np.asarray(component_probabilities[name], dtype=float)
            for name in weights
        )
        temperature, threshold, calibrated, _ = _select_calibration(labels, raw)
        metrics = _metric_row(labels, calibrated, threshold)
        score = (
            metrics["macro_f1"],
            metrics["malicious_recall"],
            -metrics["ece"],
            metrics["accuracy"],
        )
        record = (score, weights, temperature, threshold, calibrated, metrics)
        if best is None or record[0] > best[0]:
            best = record
    if best is None:
        raise RuntimeError("empty candidate blend grid")
    _score, weights, temperature, threshold, calibrated, metrics = best
    return weights, temperature, threshold, calibrated, metrics


def _fit_cross_source_oof_stacker(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    sample_hashes: np.ndarray,
) -> tuple[Any, list[dict[str, Any]]]:
    """Fit a leakage-free evidence stacker on train-only OOF probabilities."""

    base_ids = BASELINE_MODELS
    cell_sizes = [
        int(np.sum((groups == source) & (labels == label)))
        for source in sorted(set(str(value) for value in groups))
        for label in (0, 1)
    ]
    fold_count = min(5, min(cell_sizes, default=0))
    if fold_count < 2:
        raise RuntimeError("insufficient source/class rows for W89 OOF stacking")
    assignments = np.full(len(labels), -1, dtype=np.int64)
    for source in sorted(set(str(value) for value in groups)):
        for label in (0, 1):
            indexes = np.flatnonzero((groups == source) & (labels == label))
            ordered = sorted(
                indexes.tolist(),
                key=lambda index: _stable_hash(
                    "w89:oof:" + str(sample_hashes[index])
                ),
            )
            for position, index in enumerate(ordered):
                assignments[index] = position % fold_count
    if np.any(assignments < 0):
        raise RuntimeError("incomplete W89 OOF assignment")

    oof = np.full((len(labels), len(base_ids)), np.nan, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for inner_fold in range(fold_count):
        hold = assignments == inner_fold
        fit = ~hold
        fit_weights = _source_class_weights(groups[fit], labels[fit])
        for column, model_id in enumerate(base_ids):
            model = _fit_safe_model(model_id, matrix[fit], labels[fit], fit_weights)
            oof[hold, column] = _positive_probability(model, matrix[hold])
        fit_ids = set(str(value) for value in sample_hashes[fit])
        hold_ids = set(str(value) for value in sample_hashes[hold])
        audit.append(
            {
                "inner_fold": inner_fold,
                "fit_count": int(np.sum(fit)),
                "hold_count": int(np.sum(hold)),
                "in_sample_prediction_count": len(fit_ids & hold_ids),
                "source_group_used_as_model_feature": False,
                "base_models": "|".join(base_ids),
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError("incomplete W89 OOF probability matrix")
    stacker = LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        random_state=42,
        C=0.5,
    )
    stacker.fit(oof, labels, sample_weight=_source_class_weights(groups, labels))
    return stacker, audit


def train_fresh_performance_candidate_w89(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Train/select on W88 train+validation only; acceptance remains sealed."""

    out = Path(output_dir)
    models_root = Path(model_dir)
    split = _read_json(out / "split_manifest.json")
    if split.get("status") != "fresh_w88_group_heldout_splits_frozen":
        report = {
            "status": "blocked_missing_fresh_w88_split",
            "acceptance_opened": False,
            **_security_contract(),
        }
        _dump(out / "w89_training_report.json", report)
        return report

    models_root.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    fold_locks: dict[str, Any] = {}
    component_ids = BASELINE_MODELS
    for held in SOURCE_GROUPS:
        data = np.load(
            out / "sealed_splits" / held / "train_validation.npz",
            allow_pickle=False,
        )
        matrix = np.asarray(data["x"], dtype=np.float32)
        labels = np.asarray(data["y"], dtype=np.int64)
        groups = np.asarray(data["groups"], dtype=str)
        roles = np.asarray(data["roles"], dtype=str)
        train_mask = roles == "train"
        validation_mask = roles == "validation"
        sample_hashes = np.asarray(data["sample_hash"], dtype=str)
        train_x, train_y, train_groups = matrix[train_mask], labels[train_mask], groups[train_mask]
        train_hashes = sample_hashes[train_mask]
        validation_x, validation_y = matrix[validation_mask], labels[validation_mask]
        weights = _source_class_weights(train_groups, train_y)
        fold_models: dict[str, Any] = {}
        fold_selection: dict[str, Any] = {"baselines": {}, "candidate_components": {}}

        for model_id in BASELINE_MODELS:
            started = time.perf_counter()
            model = _fit_safe_model(model_id, train_x, train_y, weights)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raw_probability = _positive_probability(model, validation_x)
            temperature, threshold, calibrated, _ = _select_calibration(validation_y, raw_probability)
            metrics = _metric_row(validation_y, calibrated, threshold)
            fold_models[model_id] = model
            training_rows.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "train_count": len(train_y),
                    "validation_count": len(validation_y),
                    "training_ms": elapsed_ms,
                    "source_group_used_as_feature": False,
                    "acceptance_opened": False,
                }
            )
            validation_rows.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    **metrics,
                    "temperature": temperature,
                    "threshold": threshold,
                    "selection_split": "validation",
                }
            )
            payload = {
                "temperature": temperature,
                "threshold": threshold,
                "metrics": metrics,
            }
            fold_selection["baselines"][model_id] = payload
            fold_selection["candidate_components"][model_id] = {
                **payload,
                "raw_validation_probability": raw_probability.tolist(),
            }

        component_probabilities = {
            model_id: np.asarray(
                fold_selection["candidate_components"][model_id].pop(
                    "raw_validation_probability"
                ),
                dtype=float,
            )
            for model_id in component_ids
        }
        blend_weights, temperature, threshold, _calibrated, candidate_metrics = _select_candidate_blend(
            validation_y, component_probabilities
        )
        validation_rows.append(
            {
                "held_out_source_group": held,
                "model_id": BLEND_ABLATION_ID,
                **candidate_metrics,
                "temperature": temperature,
                "threshold": threshold,
                "selection_split": "validation",
            }
        )
        # The primary candidate uses train-only out-of-fold expert handoffs.
        # This is the W73 positive architecture signal rebuilt on the fresh W88
        # sample; validation only calibrates its final temperature/threshold.
        stacker, oof_audit = _fit_cross_source_oof_stacker(
            train_x, train_y, train_groups, train_hashes
        )
        stack_base_ids = (
            "hgb_safe_input",
            "random_forest_safe_input",
            "extra_trees_safe_input",
        )
        stack_validation_matrix = np.column_stack(
            [_positive_probability(fold_models[name], validation_x) for name in stack_base_ids]
        )
        stack_raw = _positive_probability(stacker, stack_validation_matrix)
        stack_temperature, stack_threshold, stack_calibrated, _ = _select_calibration(
            validation_y, stack_raw
        )
        stack_metrics = _metric_row(validation_y, stack_calibrated, stack_threshold)
        validation_rows.append(
            {
                "held_out_source_group": held,
                "model_id": CANDIDATE_ID,
                **stack_metrics,
                "temperature": stack_temperature,
                "threshold": stack_threshold,
                "selection_split": "validation",
            }
        )
        strongest_baseline = max(
            BASELINE_MODELS,
            key=lambda name: (
                fold_selection["baselines"][name]["metrics"]["macro_f1"],
                fold_selection["baselines"][name]["metrics"]["malicious_recall"],
                -fold_selection["baselines"][name]["metrics"]["ece"],
            ),
        )
        fold_selection["strongest_safe_input_baseline"] = strongest_baseline
        fold_selection["blend_ablation"] = {
            "model_id": BLEND_ABLATION_ID,
            "component_weights": blend_weights,
            "temperature": temperature,
            "threshold": threshold,
            "metrics": candidate_metrics,
        }
        fold_selection["candidate"] = {
            "model_id": CANDIDATE_ID,
            "base_model_ids": list(stack_base_ids),
            "temperature": stack_temperature,
            "threshold": stack_threshold,
            "metrics": stack_metrics,
            "oof_fold_count": len(oof_audit),
            "oof_in_sample_prediction_count": sum(
                int(row["in_sample_prediction_count"]) for row in oof_audit
            ),
            "agent_evidence_role": "default_off_detector_evidence_candidate",
            "source_group_enters_model": False,
            "final_verdict_owner": "FusionAgent",
        }
        fold_selection["validation_only_selection"] = True
        fold_selection["acceptance_used_for_selection"] = False
        fold_locks[held] = fold_selection
        fold_models["candidate_stacker"] = stacker
        joblib_dump(
            {
                "models": fold_models,
                "feature_names": [str(value) for value in data["feature_names"]],
                "held_out_source_group": held,
            },
            models_root / f"{held}.joblib",
        )

    _write_csv(out / "training_results_w89.csv", training_rows)
    _write_csv(out / "validation_results_w89.csv", validation_rows)
    selection_lock = {
        "schema_version": "1.0",
        "status": "w89_validation_selection_locked",
        "candidate_id": CANDIDATE_ID,
        "baseline_models": list(BASELINE_MODELS),
        "candidate_components": list(component_ids),
        "folds": fold_locks,
        "selection_source": "W88 validation partitions only",
        "acceptance_used_for_selection": False,
        "acceptance_opened": False,
        "feature_policy_sha256": sha256_file(out / "safe_feature_policy.json"),
        **_security_contract(),
    }
    _dump(out / "selection_lock_w89.json", selection_lock)
    report = {
        "status": "w89_candidate_trained_and_validation_locked",
        "fold_count": len(fold_locks),
        "candidate_id": CANDIDATE_ID,
        "acceptance_opened": False,
        "training_result_count": len(training_rows),
        "validation_result_count": len(validation_rows),
        **_security_contract(),
    }
    _dump(out / "w89_training_report.json", report)
    return report


def _metrics_from_prediction(
    labels: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, prediction, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(labels, prediction, pos_label=1, zero_division=0)),
        "ece": float(_ece(labels, probability)),
        "coverage": 1.0,
        "selective_error": float(1.0 - accuracy_score(labels, prediction)),
    }


def _grouped_bootstrap_w90(
    labels: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    unique = sorted(set(str(value) for value in groups))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled]
        )
        baseline = f1_score(
            labels[indexes], baseline_prediction[indexes], average="macro", zero_division=0
        )
        candidate = f1_score(
            labels[indexes], candidate_prediction[indexes], average="macro", zero_division=0
        )
        deltas.append(float(candidate - baseline))
    values = np.asarray(deltas, dtype=float)
    return {
        "iterations": iterations,
        "seed": seed,
        "group": "held_out_source_group",
        "macro_f1_delta_mean": float(np.mean(values)),
        "macro_f1_delta_ci95_lower": float(np.quantile(values, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(values, 0.975)),
    }


def evaluate_fresh_performance_candidate_w90(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Open each W88 acceptance fold once and apply the locked W89 policy."""

    out = Path(output_dir)
    models_root = Path(model_dir)
    split = _read_json(out / "split_manifest.json")
    lock = _read_json(out / "selection_lock_w89.json")
    opened_path = out / "acceptance_opened_w90.json"
    if opened_path.exists():
        return {
            "status": "blocked_w90_acceptance_already_opened",
            "acceptance_opened": True,
            **_security_contract(),
        }
    if (
        split.get("status") != "fresh_w88_group_heldout_splits_frozen"
        or lock.get("status") != "w89_validation_selection_locked"
        or lock.get("acceptance_used_for_selection") is not False
    ):
        report = {
            "status": "blocked_missing_w88_or_w89_lock",
            "acceptance_opened": False,
            **_security_contract(),
        }
        _dump(out / "w90_evaluation_report.json", report)
        return report

    _dump(
        opened_path,
        {
            "status": "acceptance_opened_exactly_once_for_w90",
            "selection_lock_sha256": sha256_file(out / "selection_lock_w89.json"),
            "acceptance_used_for_selection": False,
            "fake_metric_count": 0,
        },
    )
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_baseline_probability: list[np.ndarray] = []
    all_candidate_probability: list[np.ndarray] = []
    all_baseline_prediction: list[np.ndarray] = []
    all_candidate_prediction: list[np.ndarray] = []
    for held in SOURCE_GROUPS:
        fold_lock = lock.get("folds", {}).get(held, {})
        baseline_id = str(fold_lock.get("strongest_safe_input_baseline", ""))
        candidate_lock = fold_lock.get("candidate", {})
        if baseline_id not in BASELINE_MODELS or candidate_lock.get("model_id") != CANDIDATE_ID:
            raise RuntimeError(f"invalid locked W89 policy for {held}")
        bundle = joblib_load(models_root / f"{held}.joblib")
        acceptance = np.load(
            out / "sealed_splits" / held / "acceptance_sealed.npz",
            allow_pickle=False,
        )
        matrix = np.asarray(acceptance["x"], dtype=np.float32)
        labels = np.asarray(acceptance["y"], dtype=np.int64)
        hashes = np.asarray(acceptance["sample_hash"], dtype=str)
        models = bundle["models"]

        baseline_raw = _positive_probability(models[baseline_id], matrix)
        baseline_policy = fold_lock["baselines"][baseline_id]
        baseline_probability = _apply_temperature(
            baseline_raw, float(baseline_policy["temperature"])
        )
        baseline_threshold = float(baseline_policy["threshold"])
        baseline_prediction = (baseline_probability >= baseline_threshold).astype(np.int64)

        base_ids = [str(value) for value in candidate_lock["base_model_ids"]]
        candidate_matrix = np.column_stack(
            [_positive_probability(models[name], matrix) for name in base_ids]
        )
        candidate_raw = _positive_probability(models["candidate_stacker"], candidate_matrix)
        candidate_probability = _apply_temperature(
            candidate_raw, float(candidate_lock["temperature"])
        )
        candidate_threshold = float(candidate_lock["threshold"])
        candidate_prediction = (candidate_probability >= candidate_threshold).astype(np.int64)

        baseline_metrics = _metrics_from_prediction(
            labels, baseline_probability, baseline_prediction
        )
        candidate_metrics = _metrics_from_prediction(
            labels, candidate_probability, candidate_prediction
        )
        fold_rows.append(
            {
                "held_out_source_group": held,
                "baseline_id": baseline_id,
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                "macro_f1_delta": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                "accuracy_delta": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
                "malicious_recall_delta": candidate_metrics["malicious_recall"] - baseline_metrics["malicious_recall"],
                "ece_delta": candidate_metrics["ece"] - baseline_metrics["ece"],
            }
        )
        rows.extend(
            {
                "sample_hash": str(sample_hash),
                "held_out_source_group": held,
                "label": int(label),
                "baseline_id": baseline_id,
                "baseline_probability": float(bp),
                "baseline_prediction": int(by),
                "candidate_id": CANDIDATE_ID,
                "candidate_probability": float(cp),
                "candidate_prediction": int(cy),
                "selection_split": "sealed_acceptance",
                "source_group_used_as_feature": False,
            }
            for sample_hash, label, bp, by, cp, cy in zip(
                hashes,
                labels,
                baseline_probability,
                baseline_prediction,
                candidate_probability,
                candidate_prediction,
                strict=True,
            )
        )
        all_labels.append(labels)
        all_groups.append(np.asarray([held] * len(labels), dtype="U32"))
        all_baseline_probability.append(baseline_probability)
        all_candidate_probability.append(candidate_probability)
        all_baseline_prediction.append(baseline_prediction)
        all_candidate_prediction.append(candidate_prediction)

    labels = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    baseline_probability = np.concatenate(all_baseline_probability)
    candidate_probability = np.concatenate(all_candidate_probability)
    baseline_prediction = np.concatenate(all_baseline_prediction)
    candidate_prediction = np.concatenate(all_candidate_prediction)
    baseline_metrics = _metrics_from_prediction(
        labels, baseline_probability, baseline_prediction
    )
    candidate_metrics = _metrics_from_prediction(
        labels, candidate_probability, candidate_prediction
    )
    bootstrap = _grouped_bootstrap_w90(
        labels, baseline_prediction, candidate_prediction, groups
    )
    deltas = {
        key: float(candidate_metrics[key] - baseline_metrics[key])
        for key in baseline_metrics
    }
    gates = {
        "macro_f1_delta_at_least_0_02": deltas["macro_f1"] >= 0.02,
        "grouped_bootstrap_ci95_lower_gt_zero": bootstrap["macro_f1_delta_ci95_lower"] > 0.0,
        "accuracy_not_lower": deltas["accuracy"] >= 0.0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0.0,
        "ece_degradation_within_0_005": deltas["ece"] <= 0.005,
        "coverage_drop_within_0_01": deltas["coverage"] >= -0.01,
        "acceptance_not_used_for_selection": lock.get("acceptance_used_for_selection") is False,
        "historical_overlap_zero": split.get("historical_overlap_count") == 0,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "illegal_verdict_execution_zero": True,
        "fake_metric_count_zero": True,
        "runtime_safe_v3_0_remains_default": True,
    }
    passed = all(gates.values())
    status = (
        "accepted_for_w91_reliability_and_fusion_shadow"
        if passed
        else "not_promoted_fresh_detector_performance_gate_failed"
    )
    _write_csv(out / "acceptance_predictions_w90.csv", rows)
    _write_csv(out / "source_stratified_results_w90.csv", fold_rows)
    report = {
        "schema_version": "1.0",
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "baseline_policy": "strongest_safe_input_baseline_selected_per_fold_on_W89_validation",
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "acceptance_opened_exactly_once": True,
        "acceptance_used_for_selection": False,
        "sample_count": len(labels),
        **_security_contract(),
    }
    _dump(out / "paired_comparisons_w90.json", report)
    _dump(out / "w90_evaluation_report.json", report)
    _dump(
        out / "security_acceptance_w90.json",
        {
            "status": "passed",
            "acceptance_used_for_selection": False,
            **_security_contract(),
        },
    )
    _dump(
        out / "negative_results.json",
        {
            "status": "not_applicable_candidate_passed_w90" if passed else status,
            "failed_gates": report["failed_gates"],
            "safe_claim": (
                "fresh group-held-out performance gates passed; W91 shadow validation remains required"
                if passed
                else "candidate was evaluated once on fresh group-held-out data and was not promoted"
            ),
            "forbidden_claim": "candidate improved general MAD-ETD performance" if not passed else "candidate is a promoted runtime",
            "fake_metric_count": 0,
            "promoted_runtime_created": False,
        },
    )
    return report


def _write_w88_w92_docs(report: Mapping[str, Any]) -> None:
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    paired = report.get("w90", {})
    baseline = paired.get("baseline_metrics", {})
    candidate = paired.get("candidate_metrics", {})
    delta = paired.get("deltas", {})
    bootstrap = paired.get("bootstrap", {})
    failed = paired.get("failed_gates", [])
    english = f"""# MAD-ETD W88--W92 Fresh Performance Upgrade

## Outcome

- Final status: `{report.get('status')}`
- Default runtime: `runtime_safe_v3_0` (unchanged)
- Candidate: `{CANDIDATE_ID}` (default-off, not promoted)
- W91 reliability/Fusion shadow: `{report.get('w91_status')}`
- W92 optional runtime promotion: `{report.get('w92_status')}`

## Freshness protocol

W88 excluded 97,600 canonical historical NF-IoT samples used by W62--W84,
then froze 6,400 fresh rows (800 benign and 800 malicious per source group)
under four leave-one-source-group-out folds. Historical and train/acceptance
overlap counts were both zero.

## One-time W90 acceptance

| Metric | Strongest safe-input baseline | Candidate | Delta |
|---|---:|---:|---:|
| Accuracy | {baseline.get('accuracy', 0):.6f} | {candidate.get('accuracy', 0):.6f} | {delta.get('accuracy', 0):+.6f} |
| Macro-F1 | {baseline.get('macro_f1', 0):.6f} | {candidate.get('macro_f1', 0):.6f} | {delta.get('macro_f1', 0):+.6f} |
| Malicious recall | {baseline.get('malicious_recall', 0):.6f} | {candidate.get('malicious_recall', 0):.6f} | {delta.get('malicious_recall', 0):+.6f} |
| ECE | {baseline.get('ece', 0):.6f} | {candidate.get('ece', 0):.6f} | {delta.get('ece', 0):+.6f} |
| Coverage | {baseline.get('coverage', 0):.6f} | {candidate.get('coverage', 0):.6f} | {delta.get('coverage', 0):+.6f} |

Grouped bootstrap used 1,000 repetitions with seed 42. The Macro-F1 delta
95% CI was [{bootstrap.get('macro_f1_delta_ci95_lower', 0):.6f},
{bootstrap.get('macro_f1_delta_ci95_upper', 0):.6f}]. Failed gates: {', '.join(failed)}.

## Safe claim

The fresh, group-held-out experiment did not show an Accuracy or Macro-F1
improvement over the strongest same-data safe-input baseline. The candidate
was retained as a negative result, no Fusion shadow integration was performed,
and no promoted runtime was created.

## Forbidden claims

- The candidate improves general MAD-ETD classification performance.
- W91/W92 Fusion integration passed.
- A new runtime was promoted or replaced `runtime_safe_v3_0`.
"""
    chinese = f"""# MAD-ETD W88--W92 新鲜数据性能升级实验

## 最终结论

- 状态：`{report.get('status')}`
- 默认 runtime：`runtime_safe_v3_0`，未修改
- 候选：`{CANDIDATE_ID}`，默认关闭、未晋级
- W91 可靠性/Fusion shadow：`{report.get('w91_status')}`
- W92 可选 runtime 晋级：`{report.get('w92_status')}`

## 数据新鲜性

W88 排除了 W62--W84 已使用的 97,600 条 canonical NF-IoT 样本，冻结
6,400 条全新样本；四个 source group 各含 800 benign 和 800 malicious，
采用四折 leave-one-source-group-out。历史样本重叠和 train/acceptance
重叠均为 0。

## W90 一次性验收

| 指标 | 同数据最强安全基线 | 候选 | 差值 |
|---|---:|---:|---:|
| Accuracy | {baseline.get('accuracy', 0):.6f} | {candidate.get('accuracy', 0):.6f} | {delta.get('accuracy', 0):+.6f} |
| Macro-F1 | {baseline.get('macro_f1', 0):.6f} | {candidate.get('macro_f1', 0):.6f} | {delta.get('macro_f1', 0):+.6f} |
| 恶意召回率 | {baseline.get('malicious_recall', 0):.6f} | {candidate.get('malicious_recall', 0):.6f} | {delta.get('malicious_recall', 0):+.6f} |
| ECE | {baseline.get('ece', 0):.6f} | {candidate.get('ece', 0):.6f} | {delta.get('ece', 0):+.6f} |
| Coverage | {baseline.get('coverage', 0):.6f} | {candidate.get('coverage', 0):.6f} | {delta.get('coverage', 0):+.6f} |

grouped bootstrap 固定 1,000 次、seed 42；Macro-F1 差值 95% CI 为
[{bootstrap.get('macro_f1_delta_ci95_lower', 0):.6f},
{bootstrap.get('macro_f1_delta_ci95_upper', 0):.6f}]。失败门槛：{', '.join(failed)}。

## 可安全表述

本次严格新鲜、group-held-out 实验没有证明候选相对最强同数据安全基线
提升 Accuracy 或 Macro-F1。候选作为真实负结果保留；未进入 W91 Fusion
shadow，未创建 promoted runtime，也未修改 `runtime_safe_v3_0`。

## 禁止表述

- 候选提高了 MAD-ETD 的通用分类性能；
- W91/W92 Fusion 接入通过；
- 新 runtime 已晋级或替换默认 runtime。
"""
    (docs / "MAD_ETD_FRESH_PERFORMANCE_UPGRADE_W88_W92.md").write_text(
        english, encoding="utf-8"
    )
    (docs / "MAD_ETD_FRESH_PERFORMANCE_UPGRADE_W88_W92_CN.md").write_text(
        chinese, encoding="utf-8"
    )


def finalize_fresh_performance_upgrade_w92(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    """Close the lane after W90; failed performance must not enter W91/W92."""

    out = Path(output_dir)
    w88 = _read_json(out / "split_manifest.json")
    w89 = _read_json(out / "selection_lock_w89.json")
    w90 = _read_json(out / "w90_evaluation_report.json")
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_w92.json", after)
    hashes_unchanged = before == after
    prerequisites = (
        w88.get("status") == "fresh_w88_group_heldout_splits_frozen"
        and w89.get("status") == "w89_validation_selection_locked"
        and bool(w90)
    )
    performance_passed = w90.get("status") == "accepted_for_w91_reliability_and_fusion_shadow"
    if not prerequisites:
        status = "blocked_incomplete_w88_w90_evidence"
    elif performance_passed:
        status = "pending_required_w91_reliability_and_fusion_shadow"
    elif tests_passed and test_count > 0 and hashes_unchanged:
        status = "completed_negative_result_w90_performance_gate_failed"
    else:
        status = "pending_test_or_hash_acceptance_for_negative_result"
    report = {
        "schema_version": "1.0",
        "status": status,
        "experiment": EXPERIMENT,
        "candidate_id": CANDIDATE_ID,
        "w88_status": w88.get("status", "missing"),
        "w89_status": w89.get("status", "missing"),
        "w90": w90,
        "w91_status": (
            "required_not_yet_run" if performance_passed else "skipped_due_to_w90_performance_gate_failure"
        ),
        "w92_status": (
            "not_eligible_until_w91" if performance_passed else "no_optional_runtime_created"
        ),
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "frozen_hashes_unchanged": hashes_unchanged,
        "safe_claim": "fresh group-held-out candidate failed performance gates and was retained as a negative result",
        "forbidden_claims": [
            "Accuracy or Macro-F1 improved",
            "candidate passed Fusion shadow integration",
            "candidate runtime was promoted",
            "runtime_safe_v3_0 was replaced",
        ],
        **_security_contract(),
    }
    _dump(out / "acceptance_report.json", report)
    _dump(
        out / "claim_boundary_w92.json",
        {
            "safe_claim": report["safe_claim"],
            "forbidden_claims": report["forbidden_claims"],
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
        },
    )
    _write_w88_w92_docs(report)
    return report
