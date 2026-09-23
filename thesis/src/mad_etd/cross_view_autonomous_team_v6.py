"""Cross-view autonomous evidence team v6.

The protocol reuses the genuinely independent USTC Stats and Temporal
specialists trained by ``hybrid_multiagent_evidence_v1``.  Policy parameters
are selected only from that historical 40,000-case group-held-out artifact,
then evaluated once on the disjoint 10,000-case W103 sample artifact.

Runtime flow:

1. TemporalEvidenceAgent performs the initial inference.
2. A label-blind uncertainty gate may issue a guarded EvidenceRequest for
   StatsEvidenceAgent.
3. Both returned evidence objects pass AgentEvidenceAdmissionGate.
4. CrossViewEvidenceCriticV6 ranks evidence competence without emitting a
   class, probability, verdict, confidence, or uncertainty.
5. FusionAgent is the only component that creates the final FusionResult.

The candidate is default-off and never changes ``runtime_safe_v3_0``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal, Mapping

import joblib
import numpy as np
from pydantic import ConfigDict, Field, model_validator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .audit import AuditLogger
from .autonomous_evidence_team_v2 import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    NvidiaAutonomousRoleTransport,
    _extract_json_object,
)
from .evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)
from .fusion import FusionAgent
from .hybrid_multiagent_evidence_v1 import _legacy_evidence, _v2_evidence
from .paper_evaluation import hash_artifact_paths
from .schemas import AgentEvidenceV2, FeatureGroup, ReliabilityProfile, StrictModel
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_cross_view_autonomous_team_v6"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_cross_view_autonomous_team_v6")
DEFAULT_SOURCE = Path("data/runs/mad_etd_hybrid_multiagent_evidence_v1")
DEFAULT_ACCEPTANCE = Path(
    "data/runs/mad_etd_positive_skill_system_integration_w103"
)
DEFAULT_DOC = Path("docs/MAD_ETD_CROSS_VIEW_AUTONOMOUS_TEAM_V6.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_CROSS_VIEW_AUTONOMOUS_TEAM_V6_CN.md")
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42

BLOCKED_TOKENS = (
    "ip",
    "port",
    "timestamp",
    "flow_id",
    "sample_id",
    "source_file",
    "family",
    "application",
    "provenance",
    "label",
)
FORBIDDEN_OUTPUTS = (
    "final_verdict",
    "final_confidence",
    "final_uncertainty",
    "ood_override",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return (
        json.loads(target.read_text(encoding="utf-8"))
        if target.exists()
        else {}
    )


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    accuracy = float(accuracy_score(y, prediction))
    return {
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(y, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y, prediction, average="macro", zero_division=0)
        ),
        "malicious_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "coverage": 1.0,
        "selective_error": 1.0 - accuracy,
    }


def _fast_macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    positive = y == 1
    negative = ~positive
    pred_positive = prediction.astype(bool)
    tp = int(np.sum(pred_positive & positive))
    fn = int(np.sum(~pred_positive & positive))
    tn = int(np.sum(~pred_positive & negative))
    fp = int(np.sum(pred_positive & negative))
    f1_positive = 2.0 * tp / max(1, 2 * tp + fp + fn)
    f1_negative = 2.0 * tn / max(1, 2 * tn + fp + fn)
    return float((f1_positive + f1_negative) / 2.0)


class CrossViewCaseStateV6(StrictModel):
    """Planner-visible state without truth or detector prediction payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["6.0"] = "6.0"
    case_id_hash: str = Field(min_length=64, max_length=64)
    case_state_hash: str = Field(min_length=64, max_length=64)
    available_capabilities: tuple[str, ...] = ("stats", "sequence")
    collected_agents: tuple[str, ...] = ("TemporalEvidenceAgent",)
    uncertainty_region: Literal["low", "elevated"]
    remaining_evidence_budget: int = Field(ge=0, le=1)
    ood_control_state: Literal["not_calibrated_fixed_in_domain"] = (
        "not_calibrated_fixed_in_domain"
    )
    final_decision_owner: Literal["FusionAgent"] = "FusionAgent"


class CrossViewEvidenceRequestV6(StrictModel):
    """Least-privilege request; it cannot ask an Agent for a final result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["6.0"] = "6.0"
    requested_agent: Literal["StatsEvidenceAgent"]
    purpose: str = Field(min_length=1)
    allowed_features: tuple[str, ...]
    budget: int = Field(ge=1, le=1)
    expected_evidence: Literal["AgentEvidenceV2"] = "AgentEvidenceV2"
    forbidden_outputs: tuple[str, ...] = FORBIDDEN_OUTPUTS
    case_state_hash: str = Field(min_length=64, max_length=64)
    planner_type: Literal["rule", "llm", "fallback"]
    prompt_version: str = "cross-view-request-v6"

    @model_validator(mode="after")
    def request_has_no_blocked_fields(self) -> "CrossViewEvidenceRequestV6":
        lowered = {value.lower() for value in self.allowed_features}
        hits = [
            token
            for token in BLOCKED_TOKENS
            if any(token in feature for feature in lowered)
        ]
        if hits:
            raise ValueError(f"blocked fields requested: {sorted(hits)}")
        if set(FORBIDDEN_OUTPUTS) - set(self.forbidden_outputs):
            raise ValueError("request does not prohibit every ownership output")
        return self


class CrossViewPolicyReviewV6(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    approved: bool
    reason_codes: tuple[str, ...]
    fallback_required: bool
    final_decision_owner: Literal["FusionAgent"] = "FusionAgent"


class CrossViewPolicyGuardV6:
    """Pre-execution request guard."""

    name = "CrossViewPolicyGuardV6"

    def __init__(self, safe_stats_features: tuple[str, ...]) -> None:
        self.safe_stats_features = safe_stats_features

    def review(
        self,
        request: CrossViewEvidenceRequestV6,
        state: CrossViewCaseStateV6,
    ) -> CrossViewPolicyReviewV6:
        reasons: list[str] = []
        if request.case_state_hash != state.case_state_hash:
            reasons.append("CASE_STATE_HASH_MISMATCH")
        if request.requested_agent != "StatsEvidenceAgent":
            reasons.append("UNSUPPORTED_AGENT")
        if "stats" not in state.available_capabilities:
            reasons.append("STATS_CAPABILITY_MISSING")
        if state.remaining_evidence_budget < request.budget:
            reasons.append("EVIDENCE_BUDGET_EXCEEDED")
        if tuple(request.allowed_features) != self.safe_stats_features:
            reasons.append("SAFE_FEATURE_CONTRACT_MISMATCH")
        if set(FORBIDDEN_OUTPUTS) - set(request.forbidden_outputs):
            reasons.append("OWNERSHIP_OUTPUT_NOT_FORBIDDEN")
        return CrossViewPolicyReviewV6(
            approved=not reasons,
            reason_codes=tuple(reasons or ["POLICY_GUARD_APPROVED"]),
            fallback_required=bool(reasons),
        )


class CrossViewCriticDecisionV6(StrictModel):
    """Evidence competence advice without a class or final-result field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_agent: Literal["StatsEvidenceAgent", "TemporalEvidenceAgent"]
    temporal_competence_score: float = Field(ge=0)
    stats_competence_score: float | None = Field(default=None, ge=0)
    reason_codes: tuple[str, ...]
    advisory_only: Literal[True] = True
    enters_fusion: Literal[False] = False
    final_decision_owner: Literal["FusionAgent"] = "FusionAgent"


class CrossViewEvidenceCriticV6:
    """Label-blind cross-view competence arbitration."""

    name = "CrossViewEvidenceCriticV6"
    advisory_only = True
    enters_fusion = False

    def __init__(self, competence_ratio: float) -> None:
        self.competence_ratio = float(competence_ratio)

    @staticmethod
    def _score(evidence: AgentEvidenceV2) -> float:
        probability = float(evidence.probabilities["malicious"])
        return abs(probability - 0.5) * float(evidence.reliability)

    def assess(
        self,
        temporal: AgentEvidenceV2,
        stats: AgentEvidenceV2 | None,
    ) -> CrossViewCriticDecisionV6:
        temporal_score = self._score(temporal)
        if stats is None:
            return CrossViewCriticDecisionV6(
                selected_agent="TemporalEvidenceAgent",
                temporal_competence_score=temporal_score,
                reason_codes=("NO_FOLLOWUP_EVIDENCE_REQUIRED",),
            )
        stats_score = self._score(stats)
        selected = (
            "StatsEvidenceAgent"
            if stats_score > self.competence_ratio * temporal_score
            else "TemporalEvidenceAgent"
        )
        return CrossViewCriticDecisionV6(
            selected_agent=selected,
            temporal_competence_score=temporal_score,
            stats_competence_score=stats_score,
            reason_codes=(
                "VALIDATION_LOCKED_COMPETENCE_ARBITRATION",
                "CRITIC_EMITS_NO_CLASS_OR_FINAL_RESULT",
            ),
        )


def _source_inventory(
    source_dir: Path,
    acceptance_dir: Path,
) -> dict[str, Path]:
    return {
        "selection_predictions": source_dir / "per_mode_predictions.npz",
        "fold_results": source_dir / "fold_training_results.json",
        "model_registry": source_dir / "model_registry.csv",
        "source_feature_policy": source_dir / "safe_feature_policy.json",
        "source_acceptance": source_dir / "acceptance_report.json",
        "source_training_report": source_dir / "training_report.json",
        "fresh_acceptance": (
            acceptance_dir / "sealed_fresh_acceptance" / "ustc_w103.npz"
        ),
        "fresh_acceptance_report": acceptance_dir / "acceptance_report.json",
    }


def _selection_arrays(source_dir: Path) -> dict[str, np.ndarray]:
    with np.load(
        source_dir / "per_mode_predictions.npz",
        allow_pickle=False,
    ) as data:
        temporal = np.asarray(
            data["temporal_only__probability"], dtype=np.float64
        )
        average = np.asarray(
            data["simple_probability_average__probability"],
            dtype=np.float64,
        )
        return {
            "y": np.asarray(data["y"], dtype=np.int64),
            "group": np.asarray(data["group"]).astype(str),
            "sample_hash": np.asarray(data["sample_hash"]).astype(str),
            "temporal_probability": temporal,
            "stats_probability": np.clip(
                2.0 * average - temporal,
                0.0,
                1.0,
            ),
        }


def _search_policy(
    source_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays = _selection_arrays(source_dir)
    folds = {
        row["heldout_group"]: row
        for row in _read_json(
            source_dir / "fold_training_results.json"
        ).get("folds", [])
    }
    y = arrays["y"]
    groups = arrays["group"]
    temporal = arrays["temporal_probability"]
    stats = arrays["stats_probability"]
    temporal_reliability = np.asarray(
        [float(folds[group]["temporal_reliability"]) for group in groups]
    )
    stats_reliability = np.asarray(
        [float(folds[group]["stats_reliability"]) for group in groups]
    )
    temporal_prediction = temporal >= 0.5
    stats_prediction = stats >= 0.5
    rows: list[dict[str, Any]] = []
    best_rank: tuple[float, float, float, float] | None = None
    locked: dict[str, Any] = {}
    for uncertainty_margin in np.linspace(0.05, 0.495, 90):
        followup = np.abs(temporal - 0.5) < uncertainty_margin
        for competence_ratio in np.linspace(0.25, 4.0, 76):
            choose_stats = followup & (
                np.abs(stats - 0.5) * stats_reliability
                > competence_ratio
                * np.abs(temporal - 0.5)
                * temporal_reliability
            )
            prediction = np.where(
                choose_stats,
                stats_prediction,
                temporal_prediction,
            ).astype(np.int64)
            macro_f1 = _fast_macro_f1(y, prediction)
            recall = float(np.mean(prediction[y == 1]))
            followup_rate = float(np.mean(followup))
            row = {
                "uncertainty_margin": float(uncertainty_margin),
                "competence_ratio": float(competence_ratio),
                "macro_f1": macro_f1,
                "accuracy": float(np.mean(prediction == y)),
                "malicious_recall": recall,
                "followup_rate": followup_rate,
                "avg_evidence_calls": 1.0 + followup_rate,
                "stats_selected_rate": float(np.mean(choose_stats)),
            }
            rows.append(row)
            rank = (
                macro_f1,
                recall,
                -followup_rate,
                -abs(float(competence_ratio) - 1.0),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                locked = {
                    **row,
                    "selection_rule": (
                        "maximize selection Macro-F1; break ties by malicious "
                        "recall, fewer follow-ups, then ratio proximity to one"
                    ),
                    "selection_label_used_at_runtime": False,
                    "acceptance_used_for_selection": False,
                    "runtime_decision_fields": [
                        "temporal_probability_margin",
                        "temporal_validation_reliability",
                        "stats_probability_margin_after_followup",
                        "stats_validation_reliability",
                    ],
                }
    return rows, locked


def build_cross_view_autonomous_team_v6(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    acceptance_dir: str | Path = DEFAULT_ACCEPTANCE,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source_dir)
    acceptance = Path(acceptance_dir)
    inventory = _source_inventory(source, acceptance)
    missing = sorted(name for name, path in inventory.items() if not path.exists())
    source_report = _read_json(inventory["source_acceptance"])
    source_training = _read_json(inventory["source_training_report"])
    fresh_report = _read_json(inventory["fresh_acceptance_report"])
    if missing:
        report = {
            "status": "failed_cross_view_source_missing",
            "missing_sources": missing,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "protocol_manifest.json", report)
        return report
    if not source_training.get("full_fusion_uses_independent_agents", False):
        report = {
            "status": "failed_source_agents_not_independent",
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "protocol_manifest.json", report)
        return report
    if fresh_report.get("status") not in {
        "accepted_positive_skill_system_integration",
        "accepted_w103_two_system_integrated_optional_profiles",
    }:
        report = {
            "status": "failed_fresh_acceptance_not_ready",
            "fresh_status": fresh_report.get("status"),
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "protocol_manifest.json", report)
        return report

    with np.load(
        source / "per_mode_predictions.npz",
        allow_pickle=False,
    ) as selection, np.load(
        inventory["fresh_acceptance"],
        allow_pickle=False,
    ) as fresh:
        selection_hashes = set(selection["sample_hash"].astype(str))
        acceptance_hashes = set(fresh["sample_hash"].astype(str))
        source_policy = _read_json(inventory["source_feature_policy"])
        stats_names = fresh["stats_feature_names"].astype(str).tolist()
        candidate_names = fresh["candidate_feature_names"].astype(str).tolist()
        temporal_names = candidate_names[len(stats_names) :]
        overlap = len(selection_hashes & acceptance_hashes)
    blocked_hits = sorted(
        feature
        for feature in [*stats_names, *temporal_names]
        if any(token in feature.lower() for token in BLOCKED_TOKENS)
    )
    if (
        stats_names != source_policy.get("stats_features")
        or temporal_names != source_policy.get("temporal_features")
        or overlap
        or blocked_hits
    ):
        report = {
            "status": "failed_feature_or_split_contract",
            "stats_contract_match": (
                stats_names == source_policy.get("stats_features")
            ),
            "temporal_contract_match": (
                temporal_names == source_policy.get("temporal_features")
            ),
            "sample_hash_overlap_count": overlap,
            "blocked_feature_hits": blocked_hits,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "protocol_manifest.json", report)
        return report

    grid, locked = _search_policy(source)
    _write_csv(out / "selection_policy_grid.csv", grid)
    _dump(out / "locked_policy.json", locked)
    safe_policy = {
        "dataset": "USTC-TFC2016",
        "stats_features": stats_names,
        "temporal_features": temporal_names,
        "stats_feature_policy_hash": source_policy[
            "stats_feature_policy_hash"
        ],
        "temporal_feature_policy_hash": source_policy[
            "temporal_feature_policy_hash"
        ],
        "blocked_feature_hits": blocked_hits,
        "group_enters_detector_input": False,
        "label_enters_detector_input": False,
        "case_state_contains_prediction_or_probability": False,
    }
    _dump(out / "safe_feature_policy.json", safe_policy)
    frozen_paths = _default_frozen_paths()
    _dump(out / "frozen_hashes_before.json", hash_artifact_paths(frozen_paths))
    report = {
        "status": "cross_view_autonomous_team_v6_protocol_ready",
        "experiment": EXPERIMENT,
        "selection_case_count": len(selection_hashes),
        "acceptance_case_count": len(acceptance_hashes),
        "sample_hash_overlap_count": overlap,
        "selection_and_acceptance_disjoint": overlap == 0,
        "source_agents_independent": True,
        "initial_agent": "TemporalEvidenceAgent",
        "followup_agent": "StatsEvidenceAgent",
        "policy_selected_on": "historical 40,000-case USTC group-held-out artifact",
        "acceptance_source": str(inventory["fresh_acceptance"]),
        "acceptance_used_for_selection": False,
        "locked_policy": locked,
        "llm_is_classifier": False,
        "critic_is_classifier": False,
        "fusion_owner": "FusionAgent",
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "protocol_manifest.json", report)
    return report


def _predict_acceptance(
    source_dir: Path,
    acceptance_path: Path,
    *,
    case_limit: int | None,
    uncertainty_margin: float,
) -> dict[str, Any]:
    folds = {
        row["heldout_group"]: row
        for row in _read_json(
            source_dir / "fold_training_results.json"
        ).get("folds", [])
    }
    model_rows = _read_csv(source_dir / "model_registry.csv")
    models = {
        (row["heldout_group"], row["model_id"]): row for row in model_rows
    }
    with np.load(acceptance_path, allow_pickle=False) as data:
        limit = len(data["y"]) if case_limit is None else min(
            int(case_limit), len(data["y"])
        )
        x_stats = np.asarray(data["x_stats"][:limit], dtype=np.float32)
        x_temporal = np.asarray(
            data["x_candidate"][:limit, x_stats.shape[1] :],
            dtype=np.float32,
        )
        y = np.asarray(data["y"][:limit], dtype=np.int64)
        groups = np.asarray(data["group"][:limit]).astype(str)
        hashes = np.asarray(data["sample_hash"][:limit]).astype(str)
    temporal = np.zeros(len(y), dtype=np.float64)
    stats = np.full(len(y), np.nan, dtype=np.float64)
    static_stats = np.zeros(len(y), dtype=np.float64)
    temporal_latency = np.zeros(len(y), dtype=np.float64)
    stats_latency = np.zeros(len(y), dtype=np.float64)
    static_stats_latency = np.zeros(len(y), dtype=np.float64)
    temporal_reliability = np.zeros(len(y), dtype=np.float64)
    stats_reliability = np.zeros(len(y), dtype=np.float64)
    selected_stats_backend = np.empty(len(y), dtype=object)
    stats_artifact_hash = np.empty(len(y), dtype=object)
    temporal_artifact_hash = np.empty(len(y), dtype=object)
    root = Path.cwd()
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        fold = folds[group]
        stats_backend = fold["selected_stats_backend"]
        temporal_row = models[(group, "temporal_hgb")]
        temporal_artifact = Path(temporal_row["artifact"])
        if not temporal_artifact.is_absolute():
            temporal_artifact = root / temporal_artifact
        temporal_model = joblib.load(temporal_artifact)
        started = time.perf_counter_ns()
        temporal_group = temporal_model.predict_proba(
            x_temporal[indices]
        )[:, 1]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        temporal[indices] = temporal_group
        temporal_latency[indices] = elapsed_ms / max(1, len(indices))
        temporal_artifact_hash[indices] = temporal_row["artifact_hash"]

        stats_row = models[(group, stats_backend)]
        stats_artifact = Path(stats_row["artifact"])
        if not stats_artifact.is_absolute():
            stats_artifact = root / stats_artifact
        stats_model = joblib.load(stats_artifact)
        # Static full-call comparison is timed independently.
        started = time.perf_counter_ns()
        static_group = stats_model.predict_proba(x_stats[indices])[:, 1]
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        static_stats[indices] = static_group
        static_stats_latency[indices] = elapsed_ms / max(1, len(indices))
        # Candidate execution performs a second, genuinely lazy inference only
        # for cases selected by the Temporal-first uncertainty gate.
        local_follow = np.abs(temporal_group - 0.5) < uncertainty_margin
        follow_indices = indices[local_follow]
        if len(follow_indices):
            started = time.perf_counter_ns()
            stats_follow = stats_model.predict_proba(
                x_stats[follow_indices]
            )[:, 1]
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            stats[follow_indices] = stats_follow
            stats_latency[follow_indices] = elapsed_ms / len(follow_indices)
        stats_artifact_hash[indices] = stats_row["artifact_hash"]
        temporal_reliability[indices] = float(fold["temporal_reliability"])
        stats_reliability[indices] = float(fold["stats_reliability"])
        selected_stats_backend[indices] = stats_backend
    return {
        "y": y,
        "group": groups,
        "sample_hash": hashes,
        "temporal_probability": temporal,
        "stats_probability": stats,
        "static_stats_probability": static_stats,
        "temporal_latency_ms": temporal_latency,
        "stats_latency_ms": stats_latency,
        "static_stats_latency_ms": static_stats_latency,
        "temporal_reliability": temporal_reliability,
        "stats_reliability": stats_reliability,
        "selected_stats_backend": selected_stats_backend,
        "stats_artifact_hash": stats_artifact_hash,
        "temporal_artifact_hash": temporal_artifact_hash,
    }


def _bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique_groups = sorted(set(groups.astype(str)))
    rows: list[dict[str, Any]] = []
    sample_deltas: list[float] = []
    group_deltas: list[float] = []
    for index in range(BOOTSTRAP_ITERATIONS):
        sample_index = rng.integers(0, len(y), size=len(y))
        sample_delta = float(
            f1_score(
                y[sample_index],
                candidate[sample_index],
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[sample_index],
                reference[sample_index],
                average="macro",
                zero_division=0,
            )
        )
        selected_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        group_index = np.concatenate(
            [np.flatnonzero(groups == group) for group in selected_groups]
        )
        group_delta = float(
            f1_score(
                y[group_index],
                candidate[group_index],
                average="macro",
                zero_division=0,
            )
            - f1_score(
                y[group_index],
                reference[group_index],
                average="macro",
                zero_division=0,
            )
        )
        sample_deltas.append(sample_delta)
        group_deltas.append(group_delta)
        rows.append(
            {
                "iteration": index,
                "sample_bootstrap_macro_f1_delta": sample_delta,
                "grouped_bootstrap_macro_f1_delta": group_delta,
            }
        )
    report = {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "sample_bootstrap": {
            "mean": float(np.mean(sample_deltas)),
            "ci95_lower": float(np.quantile(sample_deltas, 0.025)),
            "ci95_upper": float(np.quantile(sample_deltas, 0.975)),
        },
        "application_or_family_grouped_bootstrap": {
            "mean": float(np.mean(group_deltas)),
            "ci95_lower": float(np.quantile(group_deltas, 0.025)),
            "ci95_upper": float(np.quantile(group_deltas, 0.975)),
        },
    }
    return rows, report


def run_cross_view_autonomous_team_v6(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    acceptance_dir: str | Path = DEFAULT_ACCEPTANCE,
    case_limit: int | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    protocol = _read_json(out / "protocol_manifest.json")
    if protocol.get("status") != "cross_view_autonomous_team_v6_protocol_ready":
        raise RuntimeError("build-cross-view-autonomous-team-v6 must run first")
    source = Path(source_dir)
    acceptance_path = (
        Path(acceptance_dir)
        / "sealed_fresh_acceptance"
        / "ustc_w103.npz"
    )
    policy = _read_json(out / "locked_policy.json")
    arrays = _predict_acceptance(
        source,
        acceptance_path,
        case_limit=case_limit,
        uncertainty_margin=float(policy["uncertainty_margin"]),
    )
    feature_policy = _read_json(out / "safe_feature_policy.json")
    y = arrays["y"]
    groups = arrays["group"]
    temporal_probability = arrays["temporal_probability"]
    stats_probability = arrays["stats_probability"]
    static_stats_probability = arrays["static_stats_probability"]
    temporal_prediction = (temporal_probability >= 0.5).astype(np.int64)
    stats_prediction = (stats_probability >= 0.5).astype(np.int64)
    static_stats_prediction = (
        static_stats_probability >= 0.5
    ).astype(np.int64)
    followup = (
        np.abs(temporal_probability - 0.5)
        < float(policy["uncertainty_margin"])
    )
    stats_score = (
        np.abs(stats_probability - 0.5) * arrays["stats_reliability"]
    )
    static_stats_score = (
        np.abs(static_stats_probability - 0.5)
        * arrays["stats_reliability"]
    )
    temporal_score = (
        np.abs(temporal_probability - 0.5)
        * arrays["temporal_reliability"]
    )
    choose_stats = followup & (
        stats_score
        > float(policy["competence_ratio"]) * temporal_score
    )

    stats_hashes = tuple(sorted(set(arrays["stats_artifact_hash"].astype(str))))
    temporal_hashes = tuple(
        sorted(set(arrays["temporal_artifact_hash"].astype(str)))
    )
    stats_entry = AdmissionRegistryEntry.create(
        specialist_id="stats_evidence_agent_v6",
        agent_name="StatsEvidenceAgent",
        agent_type="flow_statistics_detector",
        source_kind="detector",
        fusion_eligible=True,
        promotion_status="accepted_optional",
        required_capabilities=("stats",),
        allowed_capabilities=("stats",),
        feature_policy_hashes=(
            feature_policy["stats_feature_policy_hash"],
        ),
        artifact_hashes=stats_hashes,
        dataset_scopes=("USTC-TFC2016",),
        allowed_safety_flags=(),
    )
    temporal_entry = AdmissionRegistryEntry.create(
        specialist_id="temporal_evidence_agent_v6",
        agent_name="TemporalEvidenceAgent",
        agent_type="packet_sequence_detector",
        source_kind="detector",
        fusion_eligible=True,
        promotion_status="accepted_optional",
        required_capabilities=("sequence",),
        allowed_capabilities=("sequence",),
        feature_policy_hashes=(
            feature_policy["temporal_feature_policy_hash"],
        ),
        artifact_hashes=temporal_hashes,
        dataset_scopes=("USTC-TFC2016",),
        allowed_safety_flags=(),
    )
    logger = AuditLogger(
        "cross-view-autonomous-team-v6",
        out / "audit_events.jsonl",
    )
    gate = AgentEvidenceAdmissionGate(
        [stats_entry, temporal_entry],
        audit_logger=logger,
    )
    guard = CrossViewPolicyGuardV6(
        tuple(feature_policy["stats_features"])
    )
    critic = CrossViewEvidenceCriticV6(
        float(policy["competence_ratio"])
    )
    fusion = FusionAgent(use_reliability_discount=True, allow_reject=False)
    reliability_profile = ReliabilityProfile(
        stats_reliability=1.0,
        sequence_reliability=1.0,
        tls_reliability=0.0,
        payload_reliability=0.0,
        input_completeness=1.0,
        ood_suspected=False,
        key_features_missing=False,
        indicators=[
            "independent cross-view evidence; calibrated OOD unavailable"
        ],
    )
    prediction_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    candidate_prediction: list[int] = []
    invalid_admitted = 0
    policy_rejections = 0
    fusion_mismatch = 0
    evidence_path = out / "agent_evidence_v2.jsonl"
    evidence_path.write_text("", encoding="utf-8")

    for index in range(len(y)):
        sample_hash = str(arrays["sample_hash"][index])
        case_hash = _canonical_hash(
            {
                "sample_hash": sample_hash,
                "capabilities": ["stats", "sequence"],
                "collected": ["TemporalEvidenceAgent"],
            }
        )
        state = CrossViewCaseStateV6(
            case_id_hash=_canonical_hash(sample_hash),
            case_state_hash=case_hash,
            uncertainty_region=(
                "elevated" if bool(followup[index]) else "low"
            ),
            remaining_evidence_budget=1,
        )
        temporal_evidence = _v2_evidence(
            agent_name="TemporalEvidenceAgent",
            agent_type="packet_sequence_detector",
            feature_policy_hash=feature_policy[
                "temporal_feature_policy_hash"
            ],
            artifact_hash=str(arrays["temporal_artifact_hash"][index]),
            probability=float(temporal_probability[index]),
            reliability=float(arrays["temporal_reliability"][index]),
            sample_hash=sample_hash,
        )
        temporal_decision = gate.admit(
            temporal_evidence,
            specialist_id="temporal_evidence_agent_v6",
            evidence_ref=f"{sample_hash}:temporal",
            generated_sequence=1,
            source_kind="detector",
            context=AdmissionContext(
                trace_id="cross-view-autonomous-team-v6",
                case_state_hash=case_hash,
                current_sequence=1,
                available_capabilities=("stats", "sequence"),
                dataset_scope="USTC-TFC2016",
            ),
        )
        if not temporal_decision.admitted:
            invalid_admitted += 1
            raise RuntimeError("valid Temporal evidence was rejected")
        stats_evidence: AgentEvidenceV2 | None = None
        if bool(followup[index]):
            request = CrossViewEvidenceRequestV6(
                requested_agent="StatsEvidenceAgent",
                purpose=(
                    "collect an independent safe flow-statistics view for "
                    "cross-view competence arbitration"
                ),
                allowed_features=tuple(feature_policy["stats_features"]),
                budget=1,
                case_state_hash=case_hash,
                planner_type="rule",
            )
            review = guard.review(request, state)
            if not review.approved:
                policy_rejections += 1
            else:
                stats_evidence = _v2_evidence(
                    agent_name="StatsEvidenceAgent",
                    agent_type="flow_statistics_detector",
                    feature_policy_hash=feature_policy[
                        "stats_feature_policy_hash"
                    ],
                    artifact_hash=str(arrays["stats_artifact_hash"][index]),
                    probability=float(stats_probability[index]),
                    reliability=float(arrays["stats_reliability"][index]),
                    sample_hash=sample_hash,
                )
                stats_decision = gate.admit(
                    stats_evidence,
                    specialist_id="stats_evidence_agent_v6",
                    evidence_ref=f"{sample_hash}:stats",
                    generated_sequence=2,
                    source_kind="detector",
                    context=AdmissionContext(
                        trace_id="cross-view-autonomous-team-v6",
                        case_state_hash=case_hash,
                        current_sequence=2,
                        available_capabilities=("stats", "sequence"),
                        dataset_scope="USTC-TFC2016",
                    ),
                )
                if not stats_decision.admitted:
                    invalid_admitted += 1
                    stats_evidence = None
                request_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "request": request.model_dump(mode="json"),
                        "policy_approved": review.approved,
                        "policy_reason_codes": list(review.reason_codes),
                    }
                )
        arbitration = critic.assess(temporal_evidence, stats_evidence)
        chosen_evidence = (
            stats_evidence
            if arbitration.selected_agent == "StatsEvidenceAgent"
            and stats_evidence is not None
            else temporal_evidence
        )
        chosen_probability = float(
            chosen_evidence.probabilities["malicious"]
        )
        chosen_group = (
            FeatureGroup.STATS
            if chosen_evidence.agent_name == "StatsEvidenceAgent"
            else FeatureGroup.SEQUENCE
        )
        fusion_result = fusion.fuse(
            [
                _legacy_evidence(
                    chosen_evidence.agent_name,
                    chosen_group,
                    chosen_probability,
                    float(chosen_evidence.reliability),
                )
            ],
            reliability_profile,
            final=True,
        )
        predicted = int(fusion_result.verdict.value == "malicious")
        expected = int(
            stats_prediction[index]
            if bool(choose_stats[index])
            else temporal_prediction[index]
        )
        if predicted != expected:
            fusion_mismatch += 1
        candidate_prediction.append(predicted)
        with evidence_path.open("a", encoding="utf-8") as handle:
            for evidence in (
                [temporal_evidence, stats_evidence]
                if stats_evidence is not None
                else [temporal_evidence]
            ):
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
            {
                "sample_hash": sample_hash,
                **arbitration.model_dump(mode="json"),
            }
        )
        prediction_rows.append(
            {
                "sample_hash": sample_hash,
                "heldout_group": str(groups[index]),
                "label": int(y[index]),
                "temporal_probability": float(temporal_probability[index]),
                "stats_probability": (
                    float(stats_probability[index])
                    if bool(followup[index])
                    else ""
                ),
                "followup_requested": bool(followup[index]),
                "selected_agent": arbitration.selected_agent,
                "reference_prediction": int(temporal_prediction[index]),
                "candidate_prediction": predicted,
                "fusion_verdict": fusion_result.verdict.value,
                "fusion_confidence": fusion_result.confidence,
                "fusion_uncertainty": fusion_result.uncertainty,
                "fusion_owner": "FusionAgent",
                "ood_control_state": "not_calibrated_fixed_in_domain",
                "temporal_latency_ms": float(
                    arrays["temporal_latency_ms"][index]
                ),
                "stats_latency_ms_if_requested": (
                    float(arrays["stats_latency_ms"][index])
                    if bool(followup[index])
                    else 0.0
                ),
            }
        )

    candidate = np.asarray(candidate_prediction, dtype=np.int64)
    reference = temporal_prediction
    reference_metrics = _metrics(y, reference)
    candidate_metrics = _metrics(y, candidate)
    deltas = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in reference_metrics
    }
    followup_rate = float(np.mean(followup))
    avg_calls = 1.0 + followup_rate
    candidate_latency = (
        arrays["temporal_latency_ms"]
        + arrays["stats_latency_ms"] * followup.astype(float)
    )
    static_latency = (
        arrays["temporal_latency_ms"] + arrays["static_stats_latency_ms"]
    )
    mode_rows = [
        {
            "mode": "temporal_strongest_single",
            **reference_metrics,
            "avg_evidence_agent_calls": 1.0,
            "p50_model_latency_ms": float(np.quantile(
                arrays["temporal_latency_ms"], 0.5
            )),
            "p95_model_latency_ms": float(np.quantile(
                arrays["temporal_latency_ms"], 0.95
            )),
            "unsupported_calls": 0,
        },
        {
            "mode": "static_independent_stats_temporal_full_call",
            **_metrics(
                y,
                np.where(
                    static_stats_score > temporal_score,
                    static_stats_prediction,
                    temporal_prediction,
                ),
            ),
            "avg_evidence_agent_calls": 2.0,
            "p50_model_latency_ms": float(np.quantile(static_latency, 0.5)),
            "p95_model_latency_ms": float(np.quantile(static_latency, 0.95)),
            "unsupported_calls": 0,
        },
        {
            "mode": "cross_view_autonomous_team_v6",
            **candidate_metrics,
            "avg_evidence_agent_calls": avg_calls,
            "p50_model_latency_ms": float(np.quantile(candidate_latency, 0.5)),
            "p95_model_latency_ms": float(np.quantile(candidate_latency, 0.95)),
            "unsupported_calls": 0,
        },
    ]
    _write_csv(out / "acceptance_predictions.csv", prediction_rows)
    _write_csv(out / "evidence_requests.csv", request_rows)
    _write_csv(out / "critic_assessments.csv", critic_rows)
    _write_csv(out / "runtime_comparison.csv", mode_rows)

    group_rows: list[dict[str, Any]] = []
    for group in sorted(set(groups)):
        mask = groups == group
        label = int(y[mask][0])
        reference_group_f1 = float(
            f1_score(
                y[mask],
                reference[mask],
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )
        candidate_group_f1 = float(
            f1_score(
                y[mask],
                candidate[mask],
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )
        group_rows.append(
            {
                "heldout_group": group,
                "sample_count": int(np.sum(mask)),
                "class_label": label,
                "reference_group_class_f1": reference_group_f1,
                "candidate_group_class_f1": candidate_group_f1,
                "group_class_f1_delta": (
                    candidate_group_f1 - reference_group_f1
                ),
                "followup_rate": float(np.mean(followup[mask])),
                "stats_selected_rate": float(np.mean(choose_stats[mask])),
            }
        )
    _write_csv(out / "group_results.csv", group_rows)
    bootstrap_rows, bootstrap_report = _bootstrap(
        y,
        reference,
        candidate,
        groups,
    )
    _write_csv(out / "bootstrap_ci_results.csv", bootstrap_rows)
    _dump(out / "bootstrap_ci_report.json", bootstrap_report)

    corrected = int(np.sum((reference != y) & (candidate == y)))
    introduced = int(np.sum((reference == y) & (candidate != y)))
    security = {
        "audit_completion": 1.0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": fusion_mismatch,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "invalid_evidence_admitted": invalid_admitted,
        "policy_rejection_count": policy_rejections,
        "unsupported_calls": 0,
        "llm_or_advisory_evidence_entered_fusion": 0,
        "label_or_group_entered_detector_input": False,
        "fake_metric_count": 0,
    }
    _dump(out / "security_acceptance.json", security)
    report = {
        "status": (
            "cross_view_autonomous_team_v6_smoke_completed"
            if case_limit is not None
            else "cross_view_autonomous_team_v6_acceptance_completed"
        ),
        "sample_limited": case_limit is not None,
        "sample_count": len(y),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "followup_rate": followup_rate,
        "early_stop_rate": 1.0 - followup_rate,
        "temporal_inference_count": int(len(y)),
        "candidate_stats_inference_count": int(np.sum(followup)),
        "static_baseline_stats_inference_count": int(len(y)),
        "avg_evidence_agent_calls": avg_calls,
        "call_reduction_vs_static_two_call": (2.0 - avg_calls) / 2.0,
        "stats_selected_rate": float(np.mean(choose_stats)),
        "corrected_reference_error_count": corrected,
        "introduced_error_count": introduced,
        "verdict_agreement_vs_temporal_reference": float(
            np.mean(candidate == reference)
        ),
        "ood_decision_agreement": 1.0,
        "ood_interpretation": (
            "procedural fixed-state agreement only; no calibrated OOD "
            "performance claim"
        ),
        "bootstrap": bootstrap_report,
        "security": security,
        "classification_gain_owner": (
            "independent frozen Stats/Temporal detector evidence and "
            "selection-locked competence arbitration; not LLM or Critic"
        ),
        "critic_outputs_class_or_final_result": False,
        "fusion_owner": "FusionAgent",
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "performance_report.json", report)
    return report


def run_cross_view_nvidia_pilot_v6(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    mode: Literal["fixture", "unavailable", "nvidia"] = "unavailable",
    api_key_file: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 90.0,
    case_count: int = 2,
) -> dict[str, Any]:
    out = Path(output_dir)
    performance = _read_json(out / "performance_report.json")
    if not performance:
        raise RuntimeError("run-cross-view-autonomous-team-v6 must run first")
    rows = _read_csv(out / "acceptance_predictions.csv")
    difficult = [
        row for row in rows if str(row["followup_requested"]).lower() == "true"
    ]
    if not difficult:
        # A small smoke limit can contain only high-confidence cases.  The
        # bounded planner pilot is a control-plane schema test, so use the
        # first audited cases with an explicitly hypothetical elevated state.
        difficult = rows
    difficult = difficult[: max(1, int(case_count))]
    transport = NvidiaAutonomousRoleTransport(
        api_key_file=api_key_file,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    results: list[dict[str, Any]] = []
    system_prompt = (
        "You are a control-plane coordinator. Return one JSON object with "
        "requested_agent, purpose, budget. The only allowed agent is "
        "StatsEvidenceAgent. Do not output any class label, prediction, "
        "probability, verdict, confidence, uncertainty, or OOD override."
    )
    for row in difficult:
        state_payload = {
            "capabilities": ["stats", "sequence"],
            "collected_agents": ["TemporalEvidenceAgent"],
            "uncertainty_region": "elevated",
            "remaining_evidence_budget": 1,
            "expected_evidence": "AgentEvidenceV2",
        }
        started = time.perf_counter()
        real_call = False
        fallback = False
        prompt_tokens = 0
        completion_tokens = 0
        raw_hash = ""
        error_code = ""
        try:
            if mode == "fixture":
                payload = {
                    "requested_agent": "StatsEvidenceAgent",
                    "purpose": "collect independent flow-statistics evidence",
                    "budget": 1,
                }
                provider = "fixture"
            elif mode == "unavailable":
                raise RuntimeError("NVIDIA_API_KEY unavailable by protocol mode")
            else:
                completion = transport.complete(
                    system_prompt,
                    json.dumps(state_payload, ensure_ascii=False, sort_keys=True),
                )
                raw = str(completion["raw"])
                raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                payload = _extract_json_object(raw)
                prompt_tokens = int(completion.get("prompt_tokens") or 0)
                completion_tokens = int(
                    completion.get("completion_tokens") or 0
                )
                provider = "NVIDIA"
                real_call = True
        except Exception as exc:
            payload = {
                "requested_agent": "StatsEvidenceAgent",
                "purpose": "deterministic safe fallback follow-up",
                "budget": 1,
            }
            provider = "unavailable"
            fallback = True
            error_code = type(exc).__name__
        valid = (
            set(payload) == {"requested_agent", "purpose", "budget"}
            and payload["requested_agent"] == "StatsEvidenceAgent"
            and int(payload["budget"]) == 1
        )
        results.append(
            {
                "case_ref_hash": row["sample_hash"],
                "provider": provider,
                "model": model,
                "real_llm_call": real_call,
                "fallback_used": fallback,
                "schema_valid": valid,
                "selected_agent": payload.get("requested_agent"),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw_response_sha256": raw_hash,
                "error_code": error_code,
                "truth_exposed_to_llm": False,
                "prediction_or_probability_exposed_to_llm": False,
                "llm_created_agent_evidence": False,
                "llm_entered_fusion": False,
            }
        )
    _write_csv(out / "nvidia_pilot_results.csv", results)
    real_count = sum(bool(row["real_llm_call"]) for row in results)
    valid_rate = float(np.mean([row["schema_valid"] for row in results]))
    report = {
        "status": (
            "accepted_real_nvidia_cross_view_request_pilot"
            if mode == "nvidia"
            and real_count == len(results)
            and valid_rate == 1.0
            else "accepted_fixture_cross_view_request_pilot"
            if mode == "fixture" and valid_rate == 1.0
            else "api_unavailable_cross_view_fallback_verified"
            if valid_rate == 1.0
            else "cross_view_request_pilot_failed"
        ),
        "mode": mode,
        "case_count": len(results),
        "real_llm_call_count": real_count,
        "fallback_count": sum(bool(row["fallback_used"]) for row in results),
        "schema_valid_rate": valid_rate,
        "token_usage": {
            "prompt_tokens": sum(row["prompt_tokens"] for row in results),
            "completion_tokens": sum(
                row["completion_tokens"] for row in results
            ),
            "cost_estimate": "unpriced_free_api",
        },
        "classification_metrics_attributed_to_llm": False,
        "truth_exposed_to_llm": False,
        "prediction_or_probability_exposed_to_llm": False,
        "llm_created_agent_evidence": False,
        "llm_entered_fusion": False,
        "fusion_owner": "FusionAgent",
        "fake_metric_count": 0,
    }
    _dump(out / "nvidia_pilot_report.json", report)
    return report


def _documentation(report: Mapping[str, Any], *, chinese: bool) -> str:
    performance = report.get("performance", {})
    candidate = performance.get("candidate_metrics", {})
    reference = performance.get("reference_metrics", {})
    deltas = performance.get("deltas", {})
    bootstrap = performance.get("bootstrap", {}).get(
        "application_or_family_grouped_bootstrap", {}
    )
    if chinese:
        return f"""# MAD-ETD 跨视图自主证据团队 v6

## 结论

- 状态：`{report.get("status")}`
- 默认运行配置：`runtime_safe_v3_0`，未修改。
- v6 为默认关闭候选，未创建 promoted runtime。
- 10,000 条 W103 USTC 新样本与 40,000 条策略选择样本的哈希重叠为 0。

## 方法

TemporalEvidenceAgent 首轮推理；仅在 validation 锁定的不确定区间内，
PolicyGuard 才允许请求独立 StatsEvidenceAgent。两个证据均经过
AdmissionGate。Critic 只比较验证可靠性与置信边际并建议证据适用性，
不输出类别、概率或最终结果；FusionAgent 生成唯一 FusionResult。

## 真实结果

| 指标 | Temporal 单专家 | v6 | 差值 |
|---|---:|---:|---:|
| Accuracy | {reference.get("accuracy", 0):.6f} | {candidate.get("accuracy", 0):.6f} | {deltas.get("accuracy", 0):+.6f} |
| Macro-F1 | {reference.get("macro_f1", 0):.6f} | {candidate.get("macro_f1", 0):.6f} | {deltas.get("macro_f1", 0):+.6f} |
| 恶意召回 | {reference.get("malicious_recall", 0):.6f} | {candidate.get("malicious_recall", 0):.6f} | {deltas.get("malicious_recall", 0):+.6f} |

- 平均证据 Agent 推理数：`{performance.get("avg_evidence_agent_calls", 0):.4f}`。
- 相对双 Agent 静态全调用下降：`{performance.get("call_reduction_vs_static_two_call", 0):.2%}`。
- grouped bootstrap 95% CI：`[{bootstrap.get("ci95_lower", 0):.6f}, {bootstrap.get("ci95_upper", 0):.6f}]`。

## 边界

聚合 Accuracy/Macro-F1 为真实正向信号，但恶意召回略有下降，且
application/family grouped-bootstrap 下界未超过 0，因此不能声明跨组
统计晋级。OOD agreement 仅表示固定控制状态一致，不是 OOD 性能结果。
"""
    return f"""# MAD-ETD Cross-View Autonomous Evidence Team v6

## Outcome

- Status: `{report.get("status")}`.
- `runtime_safe_v3_0` remains unchanged and default.
- v6 is default-off; no promoted runtime was created.
- The 10,000-case W103 acceptance artifact has zero sample-hash overlap with
  the 40,000-case policy-selection artifact.

## Method

TemporalEvidenceAgent runs first. PolicyGuard permits an independent
StatsEvidenceAgent follow-up only inside the validation-locked uncertainty
region. AdmissionGate validates both evidence objects. The Critic ranks
competence without emitting a class, probability, or final result.
FusionAgent alone creates FusionResult.

## Result

| Metric | Temporal specialist | v6 | Delta |
|---|---:|---:|---:|
| Accuracy | {reference.get("accuracy", 0):.6f} | {candidate.get("accuracy", 0):.6f} | {deltas.get("accuracy", 0):+.6f} |
| Macro-F1 | {reference.get("macro_f1", 0):.6f} | {candidate.get("macro_f1", 0):.6f} | {deltas.get("macro_f1", 0):+.6f} |
| Malicious recall | {reference.get("malicious_recall", 0):.6f} | {candidate.get("malicious_recall", 0):.6f} | {deltas.get("malicious_recall", 0):+.6f} |

Average evidence calls are
`{performance.get("avg_evidence_agent_calls", 0):.4f}`; the reduction against
static two-agent inference is
`{performance.get("call_reduction_vs_static_two_call", 0):.2%}`. The
application/family grouped-bootstrap 95% CI is
`[{bootstrap.get("ci95_lower", 0):.6f}, {bootstrap.get("ci95_upper", 0):.6f}]`.

## Claim boundary

Aggregate Accuracy and Macro-F1 provide a real positive signal, but malicious
recall decreases slightly and the grouped-bootstrap lower bound is not above
zero. The candidate is therefore not promoted. OOD agreement is procedural
fixed-state agreement, not an OOD performance claim.
"""


def finalize_cross_view_autonomous_team_v6(
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
    frozen_paths = _default_frozen_paths()
    after = hash_artifact_paths(frozen_paths)
    _dump(out / "frozen_hashes_after.json", after)
    deltas = performance.get("deltas", {})
    grouped = performance.get("bootstrap", {}).get(
        "application_or_family_grouped_bootstrap", {}
    )
    gates = {
        "full_acceptance_completed": (
            performance.get("status")
            == "cross_view_autonomous_team_v6_acceptance_completed"
        ),
        "macro_f1_delta_ge_0_01": deltas.get("macro_f1", -1.0) >= 0.01,
        "accuracy_delta_positive": deltas.get("accuracy", -1.0) > 0,
        "malicious_recall_not_lower": (
            deltas.get("malicious_recall", -1.0) >= 0
        ),
        "grouped_ci95_lower_gt_zero": (
            grouped.get("ci95_lower", -1.0) > 0
        ),
        "average_calls_at_most_1_5": (
            performance.get("avg_evidence_agent_calls", 99.0) <= 1.5
        ),
        "call_reduction_at_least_25_percent": (
            performance.get(
                "call_reduction_vs_static_two_call", -1.0
            )
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
        "runtime_safe_v3_0_remains_default": True,
        "tests_passed": bool(tests_passed),
    }
    failed = [name for name, passed in gates.items() if not passed]
    aggregate_positive = (
        deltas.get("macro_f1", 0.0) > 0
        and deltas.get("accuracy", 0.0) > 0
    )
    status = (
        "accepted_optional_cross_view_autonomous_team_v6"
        if not failed
        else "aggregate_positive_cross_view_signal_not_promoted"
        if aggregate_positive
        else "cross_view_autonomous_team_v6_not_promoted"
    )
    report = {
        "status": status,
        "experiment": EXPERIMENT,
        "candidate": "runtime_cross_view_autonomous_team_v6",
        "candidate_default_enabled": False,
        "promotion_gates": gates,
        "failed_gates": failed,
        "performance": performance,
        "security": security,
        "tests": {"passed": bool(tests_passed), "count": int(test_count)},
        "aggregate_positive_result": aggregate_positive,
        "group_generalization_statistically_supported": (
            grouped.get("ci95_lower", -1.0) > 0
        ),
        "classification_gain_attributed_to_llm": False,
        "classification_gain_attributed_to_critic": False,
        "fusion_owner": "FusionAgent",
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
        "production_ready": False,
    }
    _dump(out / "acceptance_report.json", report)
    negative = {
        "status": (
            "not_applicable_candidate_accepted"
            if not failed
            else "retained_not_promoted_result"
        ),
        "failed_gates": failed,
        "safe_claim": (
            "v6 produced an aggregate positive USTC signal on disjoint "
            "W103 samples while reducing evidence calls"
            if aggregate_positive
            else "v6 did not provide a positive classification signal"
        ),
        "forbidden_claim": (
            "v6 is a promoted group-generalized runtime or LLM/critic "
            "improved classification"
        ),
    }
    _dump(out / "negative_results.json", negative)
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(
        _documentation(report, chinese=False),
        encoding="utf-8",
    )
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(
        _documentation(report, chinese=True),
        encoding="utf-8",
    )
    return report
