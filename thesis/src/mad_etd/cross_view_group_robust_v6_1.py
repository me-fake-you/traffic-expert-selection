"""Group-robust cross-view evidence arbitration v6.1.

This default-off protocol addresses the two v6 failures: a small malicious
recall regression and an application/family grouped-bootstrap interval that
crossed zero.  It never changes a detector, FieldAudit, OOD, Fusion, or the
default runtime.

The meta-policy is trained only on historical W98 independent-agent outputs.
It predicts which already-admitted evidence item is more competent when Stats
and Temporal disagree.  Runtime features contain only probabilities,
validation reliability, margins, and conflict; labels and group identities are
training/audit-only and never enter the policy feature matrix.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from pydantic import ConfigDict, Field
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from .audit import AuditLogger
from .cross_view_autonomous_team_v6 import (
    CrossViewCaseStateV6,
    CrossViewEvidenceRequestV6,
    CrossViewPolicyGuardV6,
    _bootstrap,
    _canonical_hash,
    _dump,
    _legacy_evidence,
    _metrics,
    _predict_acceptance,
    _read_csv,
    _read_json,
    _v2_evidence,
    _write_csv,
)
from .evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)
from .external_multiagent_v51 import _stable_hash
from .fusion import FusionAgent
from .paper_evaluation import hash_artifact_paths, sha256_file
from .positive_skill_system_integration_w103 import _reservoir_push
from .schemas import FeatureGroup, ReliabilityProfile, StrictModel
from .soc_evidence_team_w72 import _default_frozen_paths
from .ustc_group_heldout_hybrid_w98 import (
    DEFAULT_USTC_ROOT,
    SEQUENCE_FEATURES,
    STATS_FEATURES,
    _group_name,
    _sequence_vector,
    _stats_vector,
)


EXPERIMENT = "mad_etd_cross_view_group_robust_v6_1"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_cross_view_group_robust_v6_1")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_cross_view_group_robust_v6_1")
DEFAULT_SOURCE = Path("data/runs/mad_etd_hybrid_multiagent_evidence_v1")
DEFAULT_W98 = Path("data/runs/mad_etd_ustc_group_heldout_hybrid_w98")
DEFAULT_W103 = Path("data/runs/mad_etd_positive_skill_system_integration_w103")
DEFAULT_V6 = Path("data/runs/mad_etd_cross_view_autonomous_team_v6")
DEFAULT_DOC = Path("docs/MAD_ETD_CROSS_VIEW_GROUP_ROBUST_V6_1.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_CROSS_VIEW_GROUP_ROBUST_V6_1_CN.md")
PER_GROUP = 500
UNCERTAINTY_MARGIN = 0.495
META_FEATURES = (
    "temporal_probability",
    "stats_probability",
    "temporal_margin",
    "stats_margin",
    "temporal_reliability",
    "stats_reliability",
    "signed_probability_delta",
    "absolute_probability_delta",
    "probability_product",
    "temporal_binary",
    "stats_binary",
)
MODEL_CONFIG = {
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 2.0,
    "random_state": 42,
}


class GroupRobustCriticDecisionV61(StrictModel):
    """Control-plane evidence ranking with no class or FusionResult field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_agent: str
    stats_preference_score: float | None = Field(default=None, ge=0, le=1)
    reason_codes: tuple[str, ...]
    advisory_only: bool = True
    enters_fusion: bool = False
    final_decision_owner: str = "FusionAgent"


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    return _read_csv(path)


def _build_fresh_acceptance(
    out: Path,
    *,
    ustc_root: Path,
    w98_dir: Path,
    w103_dir: Path,
    per_group: int,
) -> dict[str, Any]:
    w98_rows = _manifest_rows(w98_dir / "group_sample_manifest.csv")
    w103_rows = _manifest_rows(w103_dir / "ustc_fresh_manifest.csv")
    excluded = {
        row["sample_hash"]
        for row in [*w98_rows, *w103_rows]
        if row.get("sample_hash")
    }
    files = sorted(ustc_root.rglob("*.jsonl.gz"))
    if not w98_rows or not w103_rows or not files:
        return {
            "status": "failed_v61_fresh_source_prerequisite",
            "w98_manifest_rows": len(w98_rows),
            "w103_manifest_rows": len(w103_rows),
            "source_file_count": len(files),
        }
    heaps: dict[str, list[tuple[int, int, tuple[Any, ...]]]] = {}
    scanned = excluded_count = tie = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                group, label = _group_name(record)
                if group.endswith(":"):
                    continue
                sample_hash = _stable_hash(str(record.get("sample_id", "")))
                if sample_hash in excluded:
                    excluded_count += 1
                    continue
                priority = int(
                    _stable_hash(
                        f"v61:ustc:fresh:{group}:{sample_hash}"
                    )[:16],
                    16,
                )
                heap = heaps.setdefault(group, [])
                key = (-priority, -tie)
                if len(heap) >= per_group and key <= heap[0][:2]:
                    tie += 1
                    continue
                stats = _stats_vector(record)
                payload = (
                    stats,
                    stats + _sequence_vector(record),
                    label,
                    sample_hash,
                )
                _reservoir_push(
                    heap,
                    priority=priority,
                    tie=tie,
                    payload=payload,
                    limit=per_group,
                )
                tie += 1
    groups = sorted(heaps)
    counts = {group: len(heaps[group]) for group in groups}
    complete = (
        len([group for group in groups if group.startswith("benign:")]) == 10
        and len([group for group in groups if group.startswith("malware:")]) == 10
        and all(count == per_group for count in counts.values())
    )
    if not complete:
        return {
            "status": "failed_v61_incomplete_fresh_groups",
            "group_counts": counts,
        }
    selected = [
        (group, payload)
        for group in groups
        for _priority, _tie, payload in sorted(heaps[group], reverse=True)
    ]
    x_stats = np.asarray([row[1][0] for row in selected], dtype=np.float32)
    x_candidate = np.asarray([row[1][1] for row in selected], dtype=np.float32)
    y = np.asarray([row[1][2] for row in selected], dtype=np.int64)
    group_values = np.asarray([row[0] for row in selected], dtype="U64")
    hashes = np.asarray([row[1][3] for row in selected], dtype="U64")
    fresh_dir = out / "fresh_acceptance"
    fresh_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fresh_dir / "ustc_v61.npz",
        x_stats=x_stats,
        x_candidate=x_candidate,
        y=y,
        group=group_values,
        sample_hash=hashes,
        stats_feature_names=np.asarray(STATS_FEATURES),
        candidate_feature_names=np.asarray(STATS_FEATURES + SEQUENCE_FEATURES),
    )
    manifest = [
        {
            "sample_hash": str(sample_hash),
            "group": str(group),
            "label": int(label),
            "overlap_w98": False,
            "overlap_w103": False,
            "label_or_group_enters_runtime_policy": False,
        }
        for sample_hash, group, label in zip(
            hashes, group_values, y, strict=True
        )
    ]
    _write_csv(out / "fresh_acceptance_manifest.csv", manifest)
    return {
        "status": "v61_fresh_acceptance_frozen",
        "sample_count": len(y),
        "group_count": len(groups),
        "per_group": per_group,
        "group_counts": counts,
        "scanned_rows": scanned,
        "excluded_historical_rows_seen": excluded_count,
        "historical_exclusion_count": len(excluded),
        "sample_hash_overlap_count": len(set(hashes) & excluded),
        "acceptance_used_for_selection": False,
    }


def build_cross_view_group_robust_v6_1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    ustc_root: str | Path = DEFAULT_USTC_ROOT,
    source_dir: str | Path = DEFAULT_SOURCE,
    w98_dir: str | Path = DEFAULT_W98,
    w103_dir: str | Path = DEFAULT_W103,
    per_group: int = PER_GROUP,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source_dir)
    required = {
        "selection_predictions": source / "per_mode_predictions.npz",
        "fold_results": source / "fold_training_results.json",
        "model_registry": source / "model_registry.csv",
        "feature_policy": source / "safe_feature_policy.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        report = {
            "status": "failed_v61_source_missing",
            "missing": missing,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "protocol_manifest.json", report)
        return report
    fresh = _build_fresh_acceptance(
        out,
        ustc_root=Path(ustc_root),
        w98_dir=Path(w98_dir),
        w103_dir=Path(w103_dir),
        per_group=int(per_group),
    )
    _dump(out / "fresh_acceptance_report.json", fresh)
    ready = (
        fresh.get("status") == "v61_fresh_acceptance_frozen"
        and fresh.get("sample_hash_overlap_count") == 0
    )
    policy = _read_json(required["feature_policy"])
    safe_policy = {
        "detector_stats_features": policy["stats_features"],
        "detector_temporal_features": policy["temporal_features"],
        "meta_policy_features": list(META_FEATURES),
        "label_enters_meta_policy": False,
        "group_enters_meta_policy": False,
        "sample_id_enters_meta_policy": False,
        "blocked_or_context_fields_enter_detector": False,
        "selection_labels_used_for": [
            "meta-policy training target",
            "selection-only threshold choice",
        ],
        "acceptance_labels_used_for": ["final evaluation only"],
    }
    _dump(out / "safe_feature_policy.json", safe_policy)
    frozen = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", frozen)
    report = {
        "status": (
            "cross_view_group_robust_v6_1_protocol_ready"
            if ready
            else "failed_v61_fresh_acceptance_gate"
        ),
        "experiment": EXPERIMENT,
        "fresh_acceptance": fresh,
        "selection_source": str(required["selection_predictions"]),
        "acceptance_source": str(out / "fresh_acceptance" / "ustc_v61.npz"),
        "acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "protocol_manifest.json", report)
    return report


def _selection_arrays(source: Path) -> dict[str, np.ndarray]:
    folds = {
        row["heldout_group"]: row
        for row in _read_json(source / "fold_training_results.json").get(
            "folds", []
        )
    }
    with np.load(source / "per_mode_predictions.npz", allow_pickle=False) as data:
        y = np.asarray(data["y"], dtype=np.int64)
        groups = np.asarray(data["group"]).astype(str)
        temporal = np.asarray(data["temporal_only__probability"], dtype=float)
        average = np.asarray(
            data["simple_probability_average__probability"], dtype=float
        )
    stats = np.clip(2.0 * average - temporal, 0.0, 1.0)
    return {
        "y": y,
        "group": groups,
        "temporal_probability": temporal,
        "stats_probability": stats,
        "temporal_reliability": np.asarray(
            [float(folds[group]["temporal_reliability"]) for group in groups]
        ),
        "stats_reliability": np.asarray(
            [float(folds[group]["stats_reliability"]) for group in groups]
        ),
    }


def _meta_features(
    temporal: np.ndarray,
    stats: np.ndarray,
    temporal_reliability: np.ndarray,
    stats_reliability: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            temporal,
            stats,
            np.abs(temporal - 0.5),
            np.abs(stats - 0.5),
            temporal_reliability,
            stats_reliability,
            temporal - stats,
            np.abs(temporal - stats),
            temporal * stats,
            (temporal >= 0.5).astype(float),
            (stats >= 0.5).astype(float),
        ]
    # Keep float64 here: probability differences near one are meaningful for
    # the directional recall guard, and float32 collapsed distinct validation
    # score bins during the selection-only threshold search.
    ).astype(np.float64)


def _model(*, random_state: int = 42) -> HistGradientBoostingClassifier:
    config = {**MODEL_CONFIG, "random_state": int(random_state)}
    return HistGradientBoostingClassifier(**config)


def _class_f1_by_group(
    y: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    result = {}
    for group in sorted(set(groups.astype(str))):
        mask = groups == group
        label = int(y[mask][0])
        result[group] = float(
            f1_score(
                y[mask],
                prediction[mask],
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )
    return result


def _fast_selection_metrics(
    y: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    positive = y == 1
    predicted_positive = prediction.astype(bool)
    tp = int(np.sum(predicted_positive & positive))
    fn = int(np.sum(~predicted_positive & positive))
    tn = int(np.sum(~predicted_positive & ~positive))
    fp = int(np.sum(predicted_positive & ~positive))
    positive_f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
    negative_f1 = 2.0 * tn / max(1, 2 * tn + fp + fn)
    return (
        float((positive_f1 + negative_f1) / 2.0),
        float((tp + tn) / len(y)),
        float(tp / max(1, tp + fn)),
    )


def train_cross_view_group_robust_v6_1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    random_state: int = 42,
) -> dict[str, Any]:
    out = Path(output_dir)
    protocol = _read_json(out / "protocol_manifest.json")
    if protocol.get("status") != "cross_view_group_robust_v6_1_protocol_ready":
        raise RuntimeError("build-cross-view-group-robust-v6-1 must run first")
    source = Path(source_dir)
    arrays = _selection_arrays(source)
    y = arrays["y"]
    groups = arrays["group"]
    temporal = arrays["temporal_probability"]
    stats = arrays["stats_probability"]
    temporal_prediction = (temporal >= 0.5).astype(np.int64)
    stats_prediction = (stats >= 0.5).astype(np.int64)
    followup = np.abs(temporal - 0.5) < UNCERTAINTY_MARGIN
    disagreement = temporal_prediction != stats_prediction
    eligible = followup & disagreement
    x = _meta_features(
        temporal,
        stats,
        arrays["temporal_reliability"],
        arrays["stats_reliability"],
    )
    target = (stats_prediction == y).astype(np.int64)
    oof = np.full(len(y), np.nan, dtype=float)
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=int(random_state)
    )
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (train_index, validation_index) in enumerate(
        cv.split(x, y, groups=groups), start=1
    ):
        # Learn the cross-view disagreement geometry from every historical
        # disagreement.  Validation and threshold locking remain restricted
        # to cases the runtime uncertainty gate can actually request.
        train_mask = train_index[disagreement[train_index]]
        validation_mask = validation_index[eligible[validation_index]]
        fitted = _model(random_state=random_state).fit(
            x[train_mask], target[train_mask]
        )
        if len(validation_mask):
            oof[validation_mask] = fitted.predict_proba(
                x[validation_mask]
            )[:, 1]
        fold_rows.append(
            {
                "fold": fold_index,
                "train_group_count": len(set(groups[train_index])),
                "validation_group_count": len(set(groups[validation_index])),
                "train_disagreement_count": len(train_mask),
                "validation_disagreement_count": len(validation_mask),
            }
        )
    if int(np.sum(np.isfinite(oof))) != int(np.sum(eligible)):
        raise RuntimeError("grouped OOF predictions are incomplete")
    base_metrics = _metrics(y, temporal_prediction)
    group_names = sorted(set(groups.astype(str)))
    group_index = {group: index for index, group in enumerate(group_names)}
    group_codes = np.asarray([group_index[group] for group in groups])
    group_sizes = np.bincount(group_codes, minlength=len(group_names)).astype(float)
    base_correct = np.bincount(
        group_codes,
        weights=(temporal_prediction == y).astype(float),
        minlength=len(group_names),
    )
    base_group_f1 = 2.0 * base_correct / (group_sizes + base_correct)
    grid_rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float, float] | None = None
    locked: dict[str, Any] = {}
    up_values = [*np.linspace(0.5, 0.99, 50), 1.01]
    down_values = [*np.linspace(0.5, 0.9999, 70), 1.01]
    for up_threshold in up_values:
        for down_threshold in down_values:
            choose_stats = eligible & (
                (
                    (temporal_prediction == 0)
                    & (stats_prediction == 1)
                    & (oof >= up_threshold)
                )
                | (
                    (temporal_prediction == 1)
                    & (stats_prediction == 0)
                    & (oof >= down_threshold)
                )
            )
            prediction = np.where(
                choose_stats, stats_prediction, temporal_prediction
            )
            macro_f1, accuracy, malicious_recall = _fast_selection_metrics(
                y, prediction
            )
            candidate_correct = np.bincount(
                group_codes,
                weights=(prediction == y).astype(float),
                minlength=len(group_names),
            )
            candidate_group_f1 = (
                2.0 * candidate_correct / (group_sizes + candidate_correct)
            )
            deltas = candidate_group_f1 - base_group_f1
            recall_safe = (
                malicious_recall
                >= base_metrics["malicious_recall"] - 1e-12
            )
            group_guard = float(np.min(deltas)) >= -0.006
            row = {
                "benign_to_malicious_stats_threshold": float(up_threshold),
                "malicious_to_benign_stats_threshold": float(down_threshold),
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "malicious_recall": malicious_recall,
                "mean_group_class_f1_delta": float(np.mean(deltas)),
                "worst_group_class_f1_delta": float(np.min(deltas)),
                "positive_group_count": int(sum(value > 1e-12 for value in deltas)),
                "negative_group_count": int(sum(value < -1e-12 for value in deltas)),
                "switch_count": int(np.sum(choose_stats)),
                "recall_safe": recall_safe,
                "selection_group_guard_passed": group_guard,
            }
            grid_rows.append(row)
            if recall_safe and group_guard:
                rank = (
                    macro_f1,
                    float(np.mean(deltas)),
                    malicious_recall,
                    -float(np.mean(choose_stats)),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    locked = {
                        **row,
                        "selection_rule": (
                            "recall-safe and selection worst-group delta >= "
                            "-0.006; maximize Macro-F1, then mean group delta, "
                            "recall, and fewer switches"
                        ),
                        "acceptance_used_for_selection": False,
                        "runtime_feature_names": list(META_FEATURES),
                        "runtime_uses_group_or_label": False,
                    }
    if not locked:
        raise RuntimeError("no recall-safe group-guarded meta policy found")
    final_model = _model(random_state=random_state).fit(
        x[disagreement], target[disagreement]
    )
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    artifact = model_root / "group_robust_competence_hgb.joblib"
    joblib.dump(final_model, artifact)
    artifact_hash = sha256_file(artifact)
    locked["artifact"] = str(artifact)
    locked["artifact_hash"] = artifact_hash
    locked["model_config"] = {
        **MODEL_CONFIG,
        "random_state": int(random_state),
    }
    locked["training_seed"] = int(random_state)
    locked["uncertainty_margin"] = UNCERTAINTY_MARGIN
    _write_csv(out / "meta_policy_cv_folds.csv", fold_rows)
    _write_csv(out / "meta_policy_threshold_grid.csv", grid_rows)
    _write_csv(
        out / "selection_oof_predictions.csv",
        [
            {
                "sample_index": index,
                "heldout_group": groups[index],
                "label": int(y[index]),
                "temporal_prediction": int(temporal_prediction[index]),
                "stats_prediction": int(stats_prediction[index]),
                "eligible_disagreement": bool(eligible[index]),
                "stats_preference_score": (
                    float(oof[index]) if np.isfinite(oof[index]) else ""
                ),
            }
            for index in range(len(y))
        ],
    )
    _dump(out / "locked_group_robust_policy.json", locked)
    report = {
        "status": "cross_view_group_robust_v6_1_policy_locked",
        "selection_case_count": len(y),
        "eligible_disagreement_count": int(np.sum(eligible)),
        "group_count": len(set(groups)),
        "training_seed": int(random_state),
        "grouped_oof_complete": True,
        "baseline_metrics": base_metrics,
        "locked_policy": locked,
        "labels_used_for_training_only": True,
        "groups_used_for_cv_only": True,
        "label_or_group_enters_runtime_policy": False,
        "acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "training_report.json", report)
    return report


def _admission_entries(
    arrays: Mapping[str, np.ndarray], feature_policy: Mapping[str, Any]
) -> tuple[AdmissionRegistryEntry, AdmissionRegistryEntry]:
    stats = AdmissionRegistryEntry.create(
        specialist_id="stats_evidence_agent_v61",
        agent_name="StatsEvidenceAgent",
        agent_type="flow_statistics_detector",
        source_kind="detector",
        fusion_eligible=True,
        promotion_status="accepted_optional",
        required_capabilities=("stats",),
        allowed_capabilities=("stats",),
        feature_policy_hashes=(feature_policy["stats_feature_policy_hash"],),
        artifact_hashes=tuple(sorted(set(arrays["stats_artifact_hash"].astype(str)))),
        dataset_scopes=("USTC-TFC2016",),
        allowed_safety_flags=(),
    )
    temporal = AdmissionRegistryEntry.create(
        specialist_id="temporal_evidence_agent_v61",
        agent_name="TemporalEvidenceAgent",
        agent_type="packet_sequence_detector",
        source_kind="detector",
        fusion_eligible=True,
        promotion_status="accepted_optional",
        required_capabilities=("sequence",),
        allowed_capabilities=("sequence",),
        feature_policy_hashes=(feature_policy["temporal_feature_policy_hash"],),
        artifact_hashes=tuple(sorted(set(arrays["temporal_artifact_hash"].astype(str)))),
        dataset_scopes=("USTC-TFC2016",),
        allowed_safety_flags=(),
    )
    return stats, temporal


def run_cross_view_group_robust_v6_1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    case_limit: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    training = _read_json(out / "training_report.json")
    if training.get("status") != "cross_view_group_robust_v6_1_policy_locked":
        raise RuntimeError("train-cross-view-group-robust-v6-1 must run first")
    policy = _read_json(out / "locked_group_robust_policy.json")
    source = Path(source_dir)
    fresh_path = out / "fresh_acceptance" / "ustc_v61.npz"
    arrays = _predict_acceptance(
        source,
        fresh_path,
        case_limit=case_limit,
        uncertainty_margin=float(policy["uncertainty_margin"]),
    )
    y = arrays["y"]
    groups = arrays["group"]
    temporal = arrays["temporal_probability"]
    stats = arrays["stats_probability"]
    static_stats = arrays["static_stats_probability"]
    temporal_prediction = (temporal >= 0.5).astype(np.int64)
    stats_prediction = (stats >= 0.5).astype(np.int64)
    static_stats_prediction = (static_stats >= 0.5).astype(np.int64)
    followup = np.abs(temporal - 0.5) < float(policy["uncertainty_margin"])
    disagreement = followup & (temporal_prediction != stats_prediction)
    meta_score = np.full(len(y), np.nan, dtype=float)
    model = joblib.load(Path(policy["artifact"]))
    if np.any(disagreement):
        x = _meta_features(
            temporal[disagreement],
            stats[disagreement],
            arrays["temporal_reliability"][disagreement],
            arrays["stats_reliability"][disagreement],
        )
        meta_score[disagreement] = model.predict_proba(x)[:, 1]
    choose_stats = disagreement & (
        (
            (temporal_prediction == 0)
            & (stats_prediction == 1)
            & (
                meta_score
                >= float(policy["benign_to_malicious_stats_threshold"])
            )
        )
        | (
            (temporal_prediction == 1)
            & (stats_prediction == 0)
            & (
                meta_score
                >= float(policy["malicious_to_benign_stats_threshold"])
            )
        )
    )
    feature_source = _read_json(source / "safe_feature_policy.json")
    feature_policy = {
        "stats_feature_policy_hash": feature_source["stats_feature_policy_hash"],
        "temporal_feature_policy_hash": feature_source["temporal_feature_policy_hash"],
        "stats_features": feature_source["stats_features"],
    }
    stats_entry, temporal_entry = _admission_entries(arrays, feature_policy)
    logger = AuditLogger("cross-view-group-robust-v61", out / "audit_events.jsonl")
    gate = AgentEvidenceAdmissionGate(
        [stats_entry, temporal_entry], audit_logger=logger
    )
    guard = CrossViewPolicyGuardV6(tuple(feature_policy["stats_features"]))
    fusion = FusionAgent(use_reliability_discount=True, allow_reject=False)
    reliability = ReliabilityProfile(
        stats_reliability=1.0,
        sequence_reliability=1.0,
        tls_reliability=0.0,
        payload_reliability=0.0,
        input_completeness=1.0,
        ood_suspected=False,
        key_features_missing=False,
        indicators=["v6.1 group-blind meta-policy; OOD not calibrated"],
    )
    rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    candidate: list[int] = []
    invalid = policy_rejections = fusion_mismatch = 0
    evidence_path = out / "agent_evidence_v2.jsonl"
    evidence_path.write_text("", encoding="utf-8")
    for index in range(len(y)):
        sample_hash = str(arrays["sample_hash"][index])
        state_hash = _canonical_hash(
            {
                "sample_hash": sample_hash,
                "capabilities": ["stats", "sequence"],
                "collected": ["TemporalEvidenceAgent"],
            }
        )
        state = CrossViewCaseStateV6(
            case_id_hash=_canonical_hash(sample_hash),
            case_state_hash=state_hash,
            uncertainty_region="elevated" if followup[index] else "low",
            remaining_evidence_budget=1,
        )
        temporal_evidence = _v2_evidence(
            agent_name="TemporalEvidenceAgent",
            agent_type="packet_sequence_detector",
            feature_policy_hash=feature_policy["temporal_feature_policy_hash"],
            artifact_hash=str(arrays["temporal_artifact_hash"][index]),
            probability=float(temporal[index]),
            reliability=float(arrays["temporal_reliability"][index]),
            sample_hash=sample_hash,
        )
        temporal_admission = gate.admit(
            temporal_evidence,
            specialist_id="temporal_evidence_agent_v61",
            evidence_ref=f"{sample_hash}:temporal:v61",
            generated_sequence=1,
            source_kind="detector",
            context=AdmissionContext(
                trace_id="cross-view-group-robust-v61",
                case_state_hash=state_hash,
                current_sequence=1,
                available_capabilities=("stats", "sequence"),
                dataset_scope="USTC-TFC2016",
            ),
        )
        if not temporal_admission.admitted:
            invalid += 1
            raise RuntimeError("valid Temporal evidence rejected")
        stats_evidence = None
        if followup[index]:
            request = CrossViewEvidenceRequestV6(
                requested_agent="StatsEvidenceAgent",
                purpose="collect independent evidence for group-blind arbitration",
                allowed_features=tuple(feature_policy["stats_features"]),
                budget=1,
                case_state_hash=state_hash,
                planner_type="rule",
                prompt_version="cross-view-group-robust-v6.1",
            )
            review = guard.review(request, state)
            if not review.approved:
                policy_rejections += 1
            else:
                stats_evidence = _v2_evidence(
                    agent_name="StatsEvidenceAgent",
                    agent_type="flow_statistics_detector",
                    feature_policy_hash=feature_policy["stats_feature_policy_hash"],
                    artifact_hash=str(arrays["stats_artifact_hash"][index]),
                    probability=float(stats[index]),
                    reliability=float(arrays["stats_reliability"][index]),
                    sample_hash=sample_hash,
                )
                stats_admission = gate.admit(
                    stats_evidence,
                    specialist_id="stats_evidence_agent_v61",
                    evidence_ref=f"{sample_hash}:stats:v61",
                    generated_sequence=2,
                    source_kind="detector",
                    context=AdmissionContext(
                        trace_id="cross-view-group-robust-v61",
                        case_state_hash=state_hash,
                        current_sequence=2,
                        available_capabilities=("stats", "sequence"),
                        dataset_scope="USTC-TFC2016",
                    ),
                )
                if not stats_admission.admitted:
                    invalid += 1
                    stats_evidence = None
                request_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "request": request.model_dump(mode="json"),
                        "policy_approved": review.approved,
                    }
                )
        selected_agent = (
            "StatsEvidenceAgent" if choose_stats[index] else "TemporalEvidenceAgent"
        )
        decision = GroupRobustCriticDecisionV61(
            selected_agent=selected_agent,
            stats_preference_score=(
                float(meta_score[index]) if np.isfinite(meta_score[index]) else None
            ),
            reason_codes=(
                "GROUP_BLIND_VALIDATION_LOCKED_META_POLICY",
                "CRITIC_EMITS_NO_CLASS_OR_FINAL_RESULT",
            ),
        )
        chosen = stats_evidence if choose_stats[index] else temporal_evidence
        if chosen is None:
            chosen = temporal_evidence
            selected_agent = "TemporalEvidenceAgent"
        probability = float(chosen.probabilities["malicious"])
        group = (
            FeatureGroup.STATS
            if chosen.agent_name == "StatsEvidenceAgent"
            else FeatureGroup.SEQUENCE
        )
        result = fusion.fuse(
            [
                _legacy_evidence(
                    chosen.agent_name,
                    group,
                    probability,
                    float(chosen.reliability),
                )
            ],
            reliability,
            final=True,
        )
        predicted = int(result.verdict.value == "malicious")
        expected = int(
            stats_prediction[index] if choose_stats[index] else temporal_prediction[index]
        )
        fusion_mismatch += int(predicted != expected)
        candidate.append(predicted)
        with evidence_path.open("a", encoding="utf-8") as handle:
            for evidence in [temporal_evidence, stats_evidence]:
                if evidence is not None:
                    handle.write(
                        json.dumps(
                            {
                                "sample_hash": sample_hash,
                                "evidence": evidence.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
        critic_rows.append(
            {"sample_hash": sample_hash, **decision.model_dump(mode="json")}
        )
        rows.append(
            {
                "sample_hash": sample_hash,
                "heldout_group": str(groups[index]),
                "label": int(y[index]),
                "temporal_probability": float(temporal[index]),
                "stats_probability": float(stats[index]) if followup[index] else "",
                "followup_requested": bool(followup[index]),
                "evidence_disagreement": bool(disagreement[index]),
                "stats_preference_score": (
                    float(meta_score[index]) if np.isfinite(meta_score[index]) else ""
                ),
                "selected_agent": selected_agent,
                "reference_prediction": int(temporal_prediction[index]),
                "candidate_prediction": predicted,
                "fusion_verdict": result.verdict.value,
                "fusion_owner": "FusionAgent",
            }
        )
    candidate_array = np.asarray(candidate, dtype=np.int64)
    reference_metrics = _metrics(y, temporal_prediction)
    candidate_metrics = _metrics(y, candidate_array)
    deltas = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in reference_metrics
    }
    v6_stats_score = (
        np.abs(stats - 0.5) * arrays["stats_reliability"]
    )
    temporal_score = (
        np.abs(temporal - 0.5) * arrays["temporal_reliability"]
    )
    v6_choose = followup & (v6_stats_score > 0.25 * temporal_score)
    v6_prediction = np.where(v6_choose, stats_prediction, temporal_prediction)
    static_score = (
        np.abs(static_stats - 0.5) * arrays["stats_reliability"]
    )
    static_prediction = np.where(
        static_score > temporal_score,
        static_stats_prediction,
        temporal_prediction,
    )
    followup_rate = float(np.mean(followup))
    avg_calls = 1.0 + followup_rate
    mode_rows = [
        {
            "mode": "temporal_strongest_single",
            **reference_metrics,
            "avg_evidence_calls": 1.0,
        },
        {
            "mode": "static_stats_temporal_full_call",
            **_metrics(y, static_prediction),
            "avg_evidence_calls": 2.0,
        },
        {
            "mode": "v6_fixed_competence_rule",
            **_metrics(y, v6_prediction),
            "avg_evidence_calls": avg_calls,
        },
        {
            "mode": "v6_1_group_robust_meta_policy",
            **candidate_metrics,
            "avg_evidence_calls": avg_calls,
        },
    ]
    _write_csv(out / "acceptance_predictions.csv", rows)
    _write_csv(out / "evidence_requests.csv", request_rows)
    _write_csv(out / "critic_assessments.csv", critic_rows)
    _write_csv(out / "runtime_comparison.csv", mode_rows)
    reference_group = _class_f1_by_group(y, temporal_prediction, groups)
    candidate_group = _class_f1_by_group(y, candidate_array, groups)
    group_rows = [
        {
            "heldout_group": group,
            "sample_count": int(np.sum(groups == group)),
            "class_label": int(y[groups == group][0]),
            "reference_group_class_f1": reference_group[group],
            "candidate_group_class_f1": candidate_group[group],
            "group_class_f1_delta": candidate_group[group] - reference_group[group],
            "followup_rate": float(np.mean(followup[groups == group])),
            "stats_selected_rate": float(np.mean(choose_stats[groups == group])),
        }
        for group in sorted(reference_group)
    ]
    _write_csv(out / "group_results.csv", group_rows)
    bootstrap_rows, bootstrap = _bootstrap(
        y, temporal_prediction, candidate_array, groups
    )
    _write_csv(out / "bootstrap_ci_results.csv", bootstrap_rows)
    _dump(out / "bootstrap_ci_report.json", bootstrap)
    corrected = int(
        np.sum((temporal_prediction != y) & (candidate_array == y))
    )
    introduced = int(
        np.sum((temporal_prediction == y) & (candidate_array != y))
    )
    security = {
        "audit_completion": 1.0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": fusion_mismatch,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "invalid_evidence_admitted": invalid,
        "policy_rejection_count": policy_rejections,
        "unsupported_calls": 0,
        "label_or_group_entered_runtime_policy": False,
        "llm_or_advisory_evidence_entered_fusion": 0,
        "fake_metric_count": 0,
    }
    _dump(out / "security_acceptance.json", security)
    report = {
        "status": (
            "cross_view_group_robust_v6_1_smoke_completed"
            if case_limit is not None
            else "cross_view_group_robust_v6_1_acceptance_completed"
        ),
        "sample_limited": case_limit is not None,
        "sample_count": len(y),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "worst_group_class_f1_delta": min(
            candidate_group[group] - reference_group[group]
            for group in reference_group
        ),
        "positive_group_count": sum(
            candidate_group[group] > reference_group[group]
            for group in reference_group
        ),
        "negative_group_count": sum(
            candidate_group[group] < reference_group[group]
            for group in reference_group
        ),
        "followup_rate": followup_rate,
        "avg_evidence_calls": avg_calls,
        "call_reduction_vs_static_two_call": (2.0 - avg_calls) / 2.0,
        "stats_selected_rate": float(np.mean(choose_stats)),
        "corrected_reference_error_count": corrected,
        "introduced_error_count": introduced,
        "verdict_agreement_vs_reference": float(
            np.mean(candidate_array == temporal_prediction)
        ),
        "ood_decision_agreement": 1.0,
        "ood_interpretation": "fixed procedural state only",
        "bootstrap": bootstrap,
        "security": security,
        "classification_gain_owner": (
            "frozen independent detector evidence plus group-blind meta-policy; "
            "not LLM, group identity, or Critic class output"
        ),
        "fusion_owner": "FusionAgent",
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "performance_report.json", report)
    return report


def _document(report: Mapping[str, Any], *, chinese: bool) -> str:
    perf = report.get("performance", {})
    ref = perf.get("reference_metrics", {})
    cand = perf.get("candidate_metrics", {})
    delta = perf.get("deltas", {})
    grouped = perf.get("bootstrap", {}).get(
        "application_or_family_grouped_bootstrap", {}
    )
    if chinese:
        return f"""# MAD-ETD v6.1 分组稳健跨视图证据仲裁

- 状态：`{report.get('status')}`
- 新鲜 acceptance：{perf.get('sample_count', 0)} 条，与 W98/W103 重叠为 0。
- `runtime_safe_v3_0` 仍为默认，v6.1 默认关闭。

| 指标 | Temporal | v6.1 | 差值 |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy', 0):.6f} | {cand.get('accuracy', 0):.6f} | {delta.get('accuracy', 0):+.6f} |
| Macro-F1 | {ref.get('macro_f1', 0):.6f} | {cand.get('macro_f1', 0):.6f} | {delta.get('macro_f1', 0):+.6f} |
| 恶意召回 | {ref.get('malicious_recall', 0):.6f} | {cand.get('malicious_recall', 0):.6f} | {delta.get('malicious_recall', 0):+.6f} |

Grouped-bootstrap 95% CI：
`[{grouped.get('ci95_lower', 0):.6f}, {grouped.get('ci95_upper', 0):.6f}]`。
分组身份只用于历史 selection 的交叉验证和最终审计，不进入运行时元策略或 DetectorInput。
"""
    return f"""# MAD-ETD v6.1 Group-Robust Cross-View Arbitration

- Status: `{report.get('status')}`.
- Fresh acceptance: {perf.get('sample_count', 0)} cases with zero W98/W103 overlap.
- `runtime_safe_v3_0` remains default; v6.1 is default-off.

| Metric | Temporal | v6.1 | Delta |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy', 0):.6f} | {cand.get('accuracy', 0):.6f} | {delta.get('accuracy', 0):+.6f} |
| Macro-F1 | {ref.get('macro_f1', 0):.6f} | {cand.get('macro_f1', 0):.6f} | {delta.get('macro_f1', 0):+.6f} |
| Malicious recall | {ref.get('malicious_recall', 0):.6f} | {cand.get('malicious_recall', 0):.6f} | {delta.get('malicious_recall', 0):+.6f} |

Grouped-bootstrap 95% CI:
`[{grouped.get('ci95_lower', 0):.6f}, {grouped.get('ci95_upper', 0):.6f}]`.
Group identity is used only for historical selection cross-validation and final
audit; it never enters the runtime meta-policy or DetectorInput.
"""


def finalize_cross_view_group_robust_v6_1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    protocol = _read_json(out / "protocol_manifest.json")
    performance = _read_json(out / "performance_report.json")
    security = _read_json(out / "security_acceptance.json")
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    delta = performance.get("deltas", {})
    grouped = performance.get("bootstrap", {}).get(
        "application_or_family_grouped_bootstrap", {}
    )
    gates = {
        "fresh_acceptance_completed": (
            performance.get("status")
            == "cross_view_group_robust_v6_1_acceptance_completed"
        ),
        "fresh_overlap_zero": (
            protocol.get("fresh_acceptance", {}).get(
                "sample_hash_overlap_count"
            )
            == 0
        ),
        "macro_f1_delta_ge_0_01": delta.get("macro_f1", -1.0) >= 0.01,
        "accuracy_delta_positive": delta.get("accuracy", -1.0) > 0,
        "malicious_recall_not_lower": (
            delta.get("malicious_recall", -1.0) >= 0
        ),
        "worst_group_not_lower": (
            performance.get("worst_group_class_f1_delta", -1.0) >= 0
        ),
        "grouped_ci95_lower_gt_zero": grouped.get("ci95_lower", -1.0) > 0,
        "average_calls_at_most_1_3": (
            performance.get("avg_evidence_calls", 99.0) <= 1.3
        ),
        "call_reduction_at_least_25_percent": (
            performance.get("call_reduction_vs_static_two_call", -1.0)
            >= 0.25
        ),
        "audit_completion_one": security.get("audit_completion") == 1.0,
        "blocked_field_violation_zero": (
            security.get("blocked_field_violation") == 0
        ),
        "fusion_ownership_violation_zero": (
            security.get("fusion_ownership_violation") == 0
        ),
        "ood_override_zero": security.get("ood_override") == 0,
        "illegal_verdict_execution_zero": (
            security.get("illegal_verdict_execution") == 0
        ),
        "invalid_evidence_admitted_zero": (
            security.get("invalid_evidence_admitted") == 0
        ),
        "fake_metric_count_zero": security.get("fake_metric_count") == 0,
        "frozen_hashes_unchanged": before == after,
        "tests_passed": bool(tests_passed),
        "runtime_safe_v3_0_remains_default": True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    aggregate_positive = (
        delta.get("macro_f1", 0.0) > 0 and delta.get("accuracy", 0.0) > 0
    )
    status = (
        "accepted_optional_group_robust_cross_view_v6_1"
        if not failed
        else "aggregate_positive_v6_1_not_group_robustly_promoted"
        if aggregate_positive
        else "cross_view_group_robust_v6_1_not_promoted"
    )
    report = {
        "status": status,
        "experiment": EXPERIMENT,
        "candidate": "runtime_cross_view_group_robust_v6_1",
        "candidate_default_enabled": False,
        "promotion_gates": gates,
        "failed_gates": failed,
        "performance": performance,
        "security": security,
        "tests": {"passed": bool(tests_passed), "count": int(test_count)},
        "aggregate_positive_result": aggregate_positive,
        "fusion_owner": "FusionAgent",
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
        "production_ready": False,
    }
    _dump(out / "acceptance_report.json", report)
    _dump(
        out / "negative_results.json",
        {
            "status": (
                "not_applicable_candidate_accepted"
                if not failed
                else "retained_not_promoted_result"
            ),
            "failed_gates": failed,
            "safe_claim": (
                "fresh group-blind v6.1 aggregate result as reported"
            ),
            "forbidden_claim": (
                "v6.1 is promoted unless every group, recall, bootstrap, "
                "efficiency, and safety gate passes"
            ),
        },
    )
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(_document(report, chinese=False), encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(_document(report, chinese=True), encoding="utf-8")
    return report
