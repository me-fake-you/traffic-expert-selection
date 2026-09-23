"""Default-off SOC evidence-team compatibility contracts.

This module deliberately sits beside the existing coordinator/executor path.
It records a typed delegation and a reviewed handoff without changing the
behaviour of ``runtime_safe_v3_0`` or giving any component other than
``FusionAgent`` a final-decision capability.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

from .base import CaseState
from .field_contract import FIELD_CONTRACT
from .schemas import (
    AgentEvidence,
    AgentEvidenceV2,
    CoordinatorAction,
    EvidenceHandoff,
    EvidenceHandoffV2,
    EvidenceRequest,
    ExecutionPlan,
    PlanAction,
)


SOC_VERDICT_AGENT_ALLOWLIST = frozenset(
    {"StatsDetectorAgent", "TemporalBehaviorAgent", "TLSProtocolAgent"}
)
SOC_ADVISORY_AGENT_NAMES = frozenset(
    {
        "LLMPlanner",
        "KnowledgeRetrievalService",
        "CaseMemory",
        "MemoryHintGenerator",
        "SelfReflectionService",
        "AuditCritic",
        "HITL",
    }
)


class ExecutionPlanProducer(Protocol):
    def plan(self, state: CaseState) -> ExecutionPlan: ...


@dataclass(frozen=True, slots=True)
class EvidenceRequestReview:
    request: EvidenceRequest
    approved: bool
    reason_codes: tuple[str, ...]


def _request_id(plan: ExecutionPlan, trace_id: str, step_id: str) -> str:
    canonical = f"soc-evidence-request-v1:{plan.plan_id}:{trace_id}:{step_id}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _safe_consumed_fields(state: CaseState, requested_agent: str) -> list[str]:
    profile = state.detector_capabilities.get(requested_agent)
    if profile is None or profile.status != "available":
        return []
    # Capability profiles are calculated from FieldAudit-controlled input.
    # A request records only actual fields the specialist declares it will
    # consume, not context/provenance or planner-provided free text.
    candidates = set(profile.available_fields or profile.consumed_fields)
    candidates.update(
        path
        for path, contract in FIELD_CONTRACT.items()
        if requested_agent in contract.get("consumers", [])
    )
    if state.field_audit is not None:
        candidates &= _field_audited_safe_paths(state, requested_agent)
    return sorted(candidates)


def _field_audited_safe_paths(
    state: CaseState,
    requested_agent: str,
) -> set[str]:
    """Include allowed values plus contract-safe empty schema slots.

    FieldAudit intentionally omits empty values from ``allowed_fields``. A
    specialist can still consume the empty/default value of an explicitly
    DETECTION_ALLOWED contract field (for example ``sequence.bursts=[]``).
    This helper never admits context, blocked, label, provenance, or unknown
    fields.
    """
    if state.field_audit is None:
        return set()
    safe = set(state.field_audit.allowed_fields)
    ignored = set(state.field_audit.ignored_empty_fields)
    for path in ignored:
        contract = FIELD_CONTRACT.get(path, {})
        if (
            contract.get("role") == "DETECTION_ALLOWED"
            and requested_agent in contract.get("consumers", [])
        ):
            safe.add(path)
    return safe


class EvidenceRequestPlanner:
    """Project a typed execution plan into specialist evidence delegations."""

    name = "EvidenceRequestPlanner"
    version = "1.0"

    def __init__(self, plan_producer: ExecutionPlanProducer) -> None:
        self.plan_producer = plan_producer

    def plan(self, state: CaseState) -> tuple[ExecutionPlan, list[EvidenceRequest]]:
        execution_plan = self.plan_producer.plan(state)
        requests = self.from_execution_plan(execution_plan, state)
        state.evidence_requests.extend(requests)
        return execution_plan, requests

    @staticmethod
    def from_execution_plan(
        execution_plan: ExecutionPlan,
        state: CaseState,
    ) -> list[EvidenceRequest]:
        requests: list[EvidenceRequest] = []
        source = execution_plan.source
        for step in execution_plan.steps:
            if step.action != PlanAction.RUN_AGENT or step.agent is None:
                continue
            fields = _safe_consumed_fields(state, step.agent)
            # An unavailable or unspecified capability does not yield a
            # delegation. PlanPolicyGuard remains the execution authority.
            if not fields:
                continue
            requests.append(
                EvidenceRequest(
                    request_id=_request_id(
                        execution_plan,
                        state.flow.trace_id,
                        step.step_id,
                    ),
                    case_trace_id=state.flow.trace_id,
                    requested_agent=step.agent,
                    permitted_safe_features=fields,
                    purpose=(
                        "Collect specialist AgentEvidence for: "
                        + ", ".join(execution_plan.reason_codes or ["ROUTING"])
                    ),
                    budget=step.budget_cost,
                    allowed_feature_policy_hash=_canonical_sha256(
                        sorted(fields)
                    ),
                    required_capabilities=sorted(fields),
                    expected_evidence_schema="AgentEvidenceV2",
                    audit_context={
                        "case_lifecycle_status": state.case_lifecycle_status,
                        "policy_version": EvidenceRequestGuard.version,
                        "request_origin": source,
                        "audit_chain_ref": state.audit_chain_ref or "unassigned",
                    },
                    planner_source=source,
                    reason_codes=list(execution_plan.reason_codes),
                )
            )
        return requests


class EvidenceRequestGuard:
    """Fail-closed approval of a planned specialist delegation."""

    name = "EvidenceRequestGuard"
    version = "1.1"

    def __init__(
        self,
        *,
        allowed_agents: set[str] | frozenset[str] | None = None,
    ) -> None:
        self.allowed_agents = frozenset(
            allowed_agents or SOC_VERDICT_AGENT_ALLOWLIST
        )

    def review(
        self,
        request: EvidenceRequest,
        state: CaseState,
    ) -> EvidenceRequestReview:
        profile = state.detector_capabilities.get(request.requested_agent)
        if state.field_audit is None:
            return self._reject(request, "MISSING_FIELD_AUDIT")
        if request.requested_agent not in self.allowed_agents:
            return self._reject(request, "AGENT_NOT_IN_SOC_EVIDENCE_ALLOWLIST")
        if request.requested_agent in SOC_ADVISORY_AGENT_NAMES:
            return self._reject(request, "ADVISORY_AGENT_CANNOT_PRODUCE_FUSION_EVIDENCE")
        if profile is None or profile.status != "available":
            return self._reject(request, "AGENT_CAPABILITY_UNAVAILABLE")
        if request.budget > state.remaining_budget:
            return self._reject(request, "EVIDENCE_REQUEST_OVER_BUDGET")
        allowed = _field_audited_safe_paths(state, request.requested_agent)
        if not set(request.permitted_safe_features).issubset(allowed):
            return self._reject(request, "PERMITTED_FEATURE_NOT_FIELD_AUDITED")
        capability_fields = set(profile.consumed_fields)
        capability_fields.update(
            path
            for path, contract in FIELD_CONTRACT.items()
            if request.requested_agent in contract.get("consumers", [])
        )
        if not set(request.permitted_safe_features).issubset(capability_fields):
            return self._reject(request, "PERMITTED_FEATURE_NOT_CAPABILITY_APPROVED")
        expected_policy_hash = _canonical_sha256(
            sorted(request.permitted_safe_features)
        )
        if (
            request.allowed_feature_policy_hash
            and request.allowed_feature_policy_hash != expected_policy_hash
        ):
            return self._reject(request, "FEATURE_POLICY_HASH_MISMATCH")
        if request.required_capabilities and not set(
            request.required_capabilities
        ).issubset(capability_fields):
            return self._reject(request, "REQUIRED_CAPABILITY_NOT_APPROVED")
        approved = request.model_copy(update={"policy_status": "approved"})
        return EvidenceRequestReview(
            request=approved,
            approved=True,
            reason_codes=("EVIDENCE_REQUEST_APPROVED",),
        )

    @staticmethod
    def _reject(
        request: EvidenceRequest,
        reason: str,
    ) -> EvidenceRequestReview:
        rejected = request.model_copy(update={"policy_status": "rejected"})
        return EvidenceRequestReview(
            request=rejected,
            approved=False,
            reason_codes=(reason,),
        )


def handoff_agent_evidence(
    review: EvidenceRequestReview,
    evidence: AgentEvidence | None,
) -> EvidenceHandoff:
    """Return evidence only for a PolicyGuard-approved delegation."""
    request = review.request
    if not review.approved:
        return EvidenceHandoff(
            request_id=request.request_id,
            case_trace_id=request.case_trace_id,
            requested_agent=request.requested_agent,
            policy_status="rejected",
            evidence=None,
            reason_codes=list(review.reason_codes),
        )
    return EvidenceHandoff(
        request_id=request.request_id,
        case_trace_id=request.case_trace_id,
        requested_agent=request.requested_agent,
        policy_status="approved",
        evidence=evidence,
        reason_codes=list(review.reason_codes),
    )


def fusion_eligible_evidence(handoff: EvidenceHandoff) -> AgentEvidence | None:
    """Expose only approved AgentEvidence to an optional Fusion adapter."""
    if handoff.policy_status != "approved":
        return None
    return handoff.evidence


def is_terminal_plan(plan: ExecutionPlan) -> bool:
    """Small audit helper: no EvidenceRequest can represent a fusion action."""
    return any(
        step.action in {PlanAction.FUSE, PlanAction.STOP}
        for step in plan.steps
    )


def _canonical_sha256(payload: object) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("semantic hashes require finite numeric values")
            return numeric
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return hashlib.sha256(
        json.dumps(
            normalize(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def agent_evidence_v1_semantic_payload(evidence: AgentEvidence) -> dict[str, object]:
    """Fields that can affect Fusion or its OOD signature."""
    return {
        "agent_name": evidence.agent_name,
        "agent_version": evidence.agent_version,
        "feature_group": evidence.feature_group.value,
        "benign_support": evidence.benign_support,
        "malicious_support": evidence.malicious_support,
        "confidence": evidence.confidence,
        "uncertainty": evidence.uncertainty,
        "calibration_quality": evidence.calibration_quality,
        "distribution_shift_score": evidence.distribution_shift_score,
        "distribution_shift_raw_score": evidence.distribution_shift_raw_score,
        "distribution_shift_level": evidence.distribution_shift_level,
        "model_reliability": evidence.model_reliability,
        "abstained": evidence.abstained,
        "contributes_to_verdict": evidence.contributes_to_verdict,
        "used_fields": sorted(evidence.used_fields),
    }


def agent_evidence_v2_semantic_payload(evidence: AgentEvidenceV2) -> dict[str, object]:
    probabilities = evidence.probabilities
    return {
        "agent_name": evidence.agent_name,
        "agent_version": evidence.agent_type.split(":", 1)[-1],
        "feature_group": evidence.agent_type.split(":", 1)[0],
        "benign_support": probabilities.get("benign", 0.0),
        "malicious_support": probabilities.get("malicious", 0.0),
        "confidence": evidence.confidence,
        "uncertainty": evidence.uncertainty,
        "calibration_quality": evidence.calibration_quality,
        "distribution_shift_score": evidence.distribution_shift_score,
        "distribution_shift_raw_score": evidence.distribution_shift_raw_score,
        "distribution_shift_level": evidence.distribution_shift_level,
        "model_reliability": evidence.reliability,
        "abstained": evidence.applicability == "abstained",
        "contributes_to_verdict": evidence.contributes_to_verdict,
        "used_fields": sorted(evidence.supported_capabilities),
    }


class AgentEvidenceV2Adapter:
    """Immutable, lossless v1-to-v2 governance adapter.

    The adapter performs no inference and returns the original v1 semantic
    object on the Fusion side only after an exact semantic-hash check.
    """

    name = "AgentEvidenceV2Adapter"
    version = "1.0"

    @staticmethod
    def feature_policy_hash(request: EvidenceRequest) -> str:
        return request.allowed_feature_policy_hash or _canonical_sha256(
            sorted(request.permitted_safe_features)
        )

    def to_v2(
        self,
        evidence: AgentEvidence,
        request: EvidenceRequest,
        *,
        artifact_hash: str,
        dataset_scope: str = "USTC_validation_shadow_only",
    ) -> AgentEvidenceV2:
        if request.policy_status != "approved":
            raise ValueError("AgentEvidenceV2Adapter requires an approved request")
        if evidence.agent_name != request.requested_agent:
            raise ValueError("evidence agent does not match EvidenceRequest")
        if not set(evidence.used_fields).issubset(
            set(request.permitted_safe_features)
        ):
            excess = sorted(
                set(evidence.used_fields) - set(request.permitted_safe_features)
            )
            raise ValueError(
                "evidence used fields exceed the approved request: "
                + ", ".join(excess)
            )
        prediction = (
            "abstain"
            if evidence.abstained
            else "malicious"
            if evidence.malicious_support > evidence.benign_support
            else "benign"
        )
        semantic_hash = _canonical_sha256(
            agent_evidence_v1_semantic_payload(evidence)
        )
        v2 = AgentEvidenceV2(
            agent_name=evidence.agent_name,
            agent_type=f"{evidence.feature_group.value}:{evidence.agent_version}",
            input_feature_policy_hash=self.feature_policy_hash(request),
            artifact_hash=artifact_hash,
            prediction=prediction,
            probabilities={
                "benign": evidence.benign_support,
                "malicious": evidence.malicious_support,
            },
            confidence=evidence.confidence,
            uncertainty=evidence.uncertainty,
            reliability=evidence.model_reliability,
            calibration_quality=evidence.calibration_quality,
            distribution_shift_score=evidence.distribution_shift_score,
            distribution_shift_raw_score=evidence.distribution_shift_raw_score,
            distribution_shift_level=evidence.distribution_shift_level,
            contributes_to_verdict=evidence.contributes_to_verdict,
            applicability="abstained" if evidence.abstained else "applicable",
            reason_codes=list(evidence.evidence),
            calibration_status=(
                "calibrated" if evidence.calibration_quality > 0 else "unknown"
            ),
            latency_ms=evidence.latency_ms,
            cost=0.0,
            supported_capabilities=sorted(evidence.used_fields),
            safety_flags=(
                ["OOD_" + evidence.distribution_shift_level.upper()]
                if evidence.distribution_shift_level != "off"
                else []
            ),
            dataset_scope=dataset_scope,
            promotion_status="shadow_only_not_promoted",
            source_evidence_sha256=semantic_hash,
        )
        v2_payload = agent_evidence_v2_semantic_payload(v2)
        if _canonical_sha256(v2_payload) != semantic_hash:
            v1_payload = agent_evidence_v1_semantic_payload(evidence)
            differing = sorted(
                key
                for key in set(v1_payload) | set(v2_payload)
                if v1_payload.get(key) != v2_payload.get(key)
            )
            raise RuntimeError(
                "AgentEvidence v1/v2 semantic hash mismatch: "
                + ", ".join(differing)
            )
        return v2

    @staticmethod
    def to_v1(
        evidence_v2: AgentEvidenceV2,
        original: AgentEvidence,
    ) -> AgentEvidence:
        original_hash = _canonical_sha256(
            agent_evidence_v1_semantic_payload(original)
        )
        v2_hash = _canonical_sha256(agent_evidence_v2_semantic_payload(evidence_v2))
        if evidence_v2.source_evidence_sha256 != original_hash or v2_hash != original_hash:
            raise RuntimeError("AgentEvidenceV2 cannot be losslessly returned to Fusion")
        return original.model_copy(deep=True)


def handoff_agent_evidence_v2(
    review: EvidenceRequestReview,
    evidence: AgentEvidenceV2 | None,
) -> EvidenceHandoffV2:
    request = review.request
    if not review.approved:
        return EvidenceHandoffV2(
            request_id=request.request_id,
            case_trace_id=request.case_trace_id,
            requested_agent=request.requested_agent,
            policy_status="rejected",
            reason_codes=list(review.reason_codes),
        )
    if evidence is None:
        raise ValueError("approved v2 handoff requires evidence")
    return EvidenceHandoffV2(
        request_id=request.request_id,
        case_trace_id=request.case_trace_id,
        requested_agent=request.requested_agent,
        policy_status="approved",
        evidence=evidence,
        feature_policy_hash=evidence.input_feature_policy_hash,
        artifact_hash=evidence.artifact_hash,
        reason_codes=list(review.reason_codes),
    )


def fusion_eligible_evidence_v2(
    handoff: EvidenceHandoffV2,
) -> AgentEvidenceV2 | None:
    if handoff.policy_status != "approved":
        return None
    return handoff.evidence


class SOCEvidenceBusShadowV1:
    """Default-off request/review/handoff bus with no decision authority."""

    name = "SOCEvidenceBusShadowV1"
    version = "1.0"
    default_enabled = False

    def __init__(self, *, allowed_agents: set[str] | None = None) -> None:
        self.guard = EvidenceRequestGuard(allowed_agents=allowed_agents)
        self.adapter = AgentEvidenceV2Adapter()

    def review(self, request: EvidenceRequest, state: CaseState) -> EvidenceRequestReview:
        return self.guard.review(request, state)

    def handoff(
        self,
        review: EvidenceRequestReview,
        evidence: AgentEvidence,
        *,
        artifact_hash: str,
    ) -> tuple[EvidenceHandoffV2, AgentEvidence | None]:
        if not review.approved:
            return handoff_agent_evidence_v2(review, None), None
        v2 = self.adapter.to_v2(
            evidence,
            review.request,
            artifact_hash=artifact_hash,
        )
        handoff = handoff_agent_evidence_v2(review, v2)
        eligible = fusion_eligible_evidence_v2(handoff)
        if eligible is None:
            return handoff, None
        return handoff, self.adapter.to_v1(eligible, evidence)
