from __future__ import annotations

import pytest
from pydantic import ValidationError

from mad_etd.base import CaseState
from mad_etd.evidence_team import (
    EvidenceRequestGuard,
    EvidenceRequestPlanner,
    fusion_eligible_evidence,
    handoff_agent_evidence,
)
from mad_etd.schemas import (
    AgentEvidence,
    AgentEvidenceV2,
    DetectorCapabilityProfile,
    EvidenceRequest,
    ExecutionPlan,
    FeatureGroup,
    FieldAuditResult,
    FieldDecision,
    FieldRole,
    FlowRecord,
    PlanAction,
    PlanStep,
)


def _state() -> CaseState:
    state = CaseState(
        flow=FlowRecord(trace_id="soc-case", sample_id="soc-case"),
        remaining_budget=2,
    )
    state.field_audit = FieldAuditResult(
        decisions=[
            FieldDecision(
                path="tls.record_lengths",
                role=FieldRole.DETECTION_ALLOWED,
                risk_score=0.0,
            )
        ],
        allowed_fields=["tls.record_lengths"],
        blocked_fields=[],
        context_only_fields=[],
        leakage_risk=0.0,
    )
    state.detector_capabilities["TLSProtocolAgent"] = DetectorCapabilityProfile(
        agent_name="TLSProtocolAgent",
        backend="rule",
        status="available",
        consumed_fields=["tls.record_lengths"],
    )
    return state


def _evidence() -> AgentEvidence:
    return AgentEvidence(
        agent_name="TLSProtocolAgent",
        feature_group=FeatureGroup.TLS,
        benign_support=0.2,
        malicious_support=0.6,
        confidence=0.8,
        uncertainty=0.2,
        used_fields=["tls.record_lengths"],
    )


def test_evidence_request_only_allows_audited_safe_features_and_prohibits_verdicts() -> None:
    with pytest.raises(ValidationError, match="blocked EvidenceRequest feature"):
        EvidenceRequest(
            request_id="blocked",
            case_trace_id="case",
            requested_agent="TLSProtocolAgent",
            permitted_safe_features=["tls.sni"],
            purpose="collect evidence",
            budget=1,
        )
    with pytest.raises(ValidationError, match="must prohibit"):
        EvidenceRequest(
            request_id="unsafe-output",
            case_trace_id="case",
            requested_agent="TLSProtocolAgent",
            permitted_safe_features=["tls.record_lengths"],
            purpose="collect evidence",
            budget=1,
            prohibited_outputs=["final_verdict"],
        )


def test_planner_projects_only_run_agent_steps_to_evidence_requests() -> None:
    state = _state()
    plan = ExecutionPlan(
        plan_id="plan",
        steps=[
            PlanStep(
                step_id="request-tls",
                action=PlanAction.RUN_AGENT,
                agent="TLSProtocolAgent",
                budget_cost=1,
            ),
            PlanStep(step_id="fuse", action=PlanAction.FUSE),
        ],
        reason_codes=["UNCERTAINTY_REQUIRES_PROTOCOL_VIEW"],
        budget=1,
    )
    requests = EvidenceRequestPlanner.from_execution_plan(plan, state)
    assert len(requests) == 1
    assert requests[0].requested_agent == "TLSProtocolAgent"
    assert requests[0].expected_output == "AgentEvidence"
    assert requests[0].policy_status == "proposed"


def test_only_policy_approved_matching_agent_evidence_is_fusion_eligible() -> None:
    state = _state()
    request = EvidenceRequest(
        request_id="request",
        case_trace_id=state.flow.trace_id,
        requested_agent="TLSProtocolAgent",
        permitted_safe_features=["tls.record_lengths"],
        purpose="collect protocol evidence",
        budget=1,
    )
    review = EvidenceRequestGuard().review(request, state)
    assert review.approved is True
    handoff = handoff_agent_evidence(review, _evidence())
    assert fusion_eligible_evidence(handoff) == _evidence()

    rejected = EvidenceRequestGuard().review(
        request.model_copy(update={"budget": 3}), state
    )
    blocked_handoff = handoff_agent_evidence(rejected, _evidence())
    assert fusion_eligible_evidence(blocked_handoff) is None


def test_agent_evidence_v2_is_evidence_not_a_final_verdict_schema() -> None:
    evidence = AgentEvidenceV2(
        agent_name="TLSProtocolAgent",
        agent_type="tls_protocol",
        input_feature_policy_hash="field-contract",
        artifact_hash="model-artifact",
        prediction="malicious",
        probabilities={"benign": 0.1, "malicious": 0.9},
        confidence=0.9,
        uncertainty=0.1,
        reliability=0.8,
        applicability="applicable",
    )
    assert evidence.prediction == "malicious"
    with pytest.raises(ValidationError):
        AgentEvidenceV2(
            **evidence.model_dump(),
            final_verdict="malicious",
        )
