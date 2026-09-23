from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mad_etd.autonomous_evidence_team_v2 import (
    DEFAULT_MODEL,
    AutonomousAgentRoleProfile,
    AutonomousEvidenceTeamHarness,
    AutonomousRoleDecision,
    build_autonomous_evidence_performance_manifest_v2,
    build_autonomous_evidence_team_v2,
    default_role_profiles,
    finalize_autonomous_evidence_team_v2,
    run_autonomous_positive_skill_bridge_v2,
    run_autonomous_evidence_team_v2_smoke,
)
from mad_etd.traffic_skill_registry_v1 import CaseStateSkillV2


def _state() -> CaseStateSkillV2:
    return CaseStateSkillV2(
        case_id="case-1",
        field_audit_summary={
            "status": "passed",
            "blocked_field_violation": 0,
        },
        safe_feature_policy_hash="a" * 64,
        available_views=("stats", "sequence", "tls_records"),
        capability_profile={
            "stats": "supported",
            "sequence": "supported",
            "tls_records": "supported",
            "safe_reference_distribution": "supported",
            "ordered_groups": "supported",
            "case_memory": "supported",
            "audit_chain": "supported",
            "completed_case_snapshot": "supported",
            "frozen_perturbation": "supported",
            "validated_evidence_refs": "supported",
            "fusion_snapshot": "supported",
            "prefix_available": "supported",
            "short_flow": "supported",
            "explicit_capture_truth": "supported",
            "explicit_tool_truth": "supported",
            "feature_forensics_registry": "supported",
        },
        collected_evidence_summary=(),
        evidence_conflict={"present": True},
        uncertainty_state={"level": "medium"},
        OOD_state={"state": "warning"},
        missing_views=(),
        short_flow_state={"is_short": True},
        remaining_agent_budget=3,
        remaining_latency_budget=1000.0,
        previous_requests=(),
        previous_failures=(),
        escalation_state={"status": "none"},
        audit_chain_hash="b" * 64,
    )


def test_all_requested_llm_roles_are_registered_and_bounded() -> None:
    profiles = default_role_profiles()
    assert len(profiles) == 9
    assert {item.role_id for item in profiles} == {
        "coordinator",
        "stats_investigator",
        "temporal_investigator",
        "tls_investigator",
        "ood_investigator",
        "evidence_critic",
        "reflection",
        "memory_rag",
        "reporter_hitl",
    }
    assert all(item.llm_enabled for item in profiles)
    assert all(item.model == DEFAULT_MODEL for item in profiles)
    assert all(not item.can_emit_agent_evidence for item in profiles)
    assert all(not item.can_write_fusion for item in profiles)
    assert all(not item.can_override_ood for item in profiles)
    assert all(item.final_decision_owner == "FusionAgent" for item in profiles)


def test_llm_role_cannot_claim_fusion_or_evidence_authority() -> None:
    with pytest.raises(ValidationError):
        AutonomousAgentRoleProfile(
            role_id="unsafe",
            display_name="unsafe",
            allowed_skills=("GeneralFlowMalwareSkill",),
            allowed_actions=("request_skill",),
            advisory_only=False,
            can_request_evidence=True,
            can_write_fusion=True,
        )
    with pytest.raises(ValidationError):
        AutonomousAgentRoleProfile(
            role_id="unsafe",
            display_name="unsafe",
            allowed_skills=("GeneralFlowMalwareSkill",),
            allowed_actions=("request_skill",),
            advisory_only=False,
            can_request_evidence=True,
            can_emit_agent_evidence=True,
        )


def test_decision_schema_rejects_final_output_and_direct_label() -> None:
    state = _state()
    base = {
        "schema_version": "2.0",
        "role_id": "coordinator",
        "case_state_hash": state.state_hash,
        "action": "request_skill",
        "requested_skill": "GeneralFlowMalwareSkill",
        "purpose": "collect one specialist result",
        "concise_rationale": "the current evidence is incomplete",
        "expected_output": "AgentEvidenceV2",
        "source": "fixture",
        "prompt_version": "mad_etd_autonomous_role_v2.0",
    }
    with pytest.raises(ValidationError):
        AutonomousRoleDecision.model_validate(
            {**base, "final_verdict": "malicious"}
        )
    with pytest.raises(ValidationError):
        AutonomousRoleDecision.model_validate(
            {**base, "concise_rationale": "the final class is malicious"}
        )


def test_case_state_rejects_truth_and_identity() -> None:
    payload = _state().model_dump(mode="python")
    payload["field_audit_summary"]["source_file"] = "secret.csv"
    with pytest.raises(ValidationError):
        CaseStateSkillV2.model_validate(payload)


def test_each_role_fixture_creates_policy_reviewed_request() -> None:
    harness = AutonomousEvidenceTeamHarness()
    state = _state()
    for role_id in sorted(harness.profiles):
        result = harness.execute_role(
            role_id=role_id,
            state=state,
            mode="fixture",
        )
        assert result.status == "fixture_success"
        assert result.evidence_request is not None
        assert result.evidence_request.requested_skill in (
            harness.profiles[role_id].allowed_skills
        )
        assert result.evidence_request.expected_evidence_type == (
            "AgentEvidenceV2"
        )
        assert result.decision.model_dump().keys().isdisjoint(
            {
                "final_verdict",
                "final_confidence",
                "final_uncertainty",
                "probabilities",
            }
        )


def test_illegal_llm_output_is_rejected_and_fallback_is_explicit() -> None:
    harness = AutonomousEvidenceTeamHarness()
    state = _state()
    raw = json.dumps(
        {
            "schema_version": "2.0",
            "role_id": "coordinator",
            "case_state_hash": state.state_hash,
            "action": "request_skill",
            "requested_skill": "GeneralFlowMalwareSkill",
            "purpose": "make a final decision",
            "concise_rationale": "malicious",
            "expected_output": "AgentEvidenceV2",
            "source": "fixture",
            "prompt_version": "mad_etd_autonomous_role_v2.0",
            "final_verdict": "malicious",
        }
    )
    result = harness.execute_role(
        role_id="coordinator",
        state=state,
        mode="fixture",
        fixture_raw=raw,
    )
    assert result.status == "llm_rejected"
    assert result.fallback_used is True
    assert result.decision.source == "fallback"
    assert result.real_llm_call is False


def test_api_unavailable_fallback_never_masquerades_as_llm() -> None:
    harness = AutonomousEvidenceTeamHarness()
    result = harness.execute_role(
        role_id="coordinator",
        state=_state(),
        mode="unavailable",
    )
    assert result.status == "api_unavailable"
    assert result.provider == "unavailable"
    assert result.real_llm_call is False
    assert result.fallback_used is True
    assert result.decision.source == "fallback"


def test_policy_review_rejects_skill_when_required_capability_is_missing() -> None:
    harness = AutonomousEvidenceTeamHarness()
    state_payload = _state().model_dump(mode="python")
    state_payload["capability_profile"].pop("explicit_capture_truth")
    state = CaseStateSkillV2.model_validate(state_payload)
    raw = json.dumps(
        {
            "schema_version": "2.0",
            "role_id": "tls_investigator",
            "case_state_hash": state.state_hash,
            "action": "request_skill",
            "requested_skill": "TLSRecordSkill",
            "purpose": "collect one policy-compliant specialist result",
            "concise_rationale": "the TLS view requires specialist analysis",
            "expected_output": "AgentEvidenceV2",
            "source": "fixture",
            "prompt_version": "mad_etd_autonomous_role_v2.0",
        }
    )
    result = harness.execute_role(
        role_id="tls_investigator",
        state=state,
        mode="fixture",
        fixture_raw=raw,
    )
    assert result.status == "policy_rejected"
    assert result.fallback_used is True
    assert any(
        reason.startswith("required_capability_missing")
        for reason in result.policy_reason_codes
    )
    assert result.decision.requested_skill != "TLSRecordSkill"
    assert result.decision.requested_skill in {
        "DoHCovertChannelSkill",
        "ProtocolInterpretationSkill",
    }


def test_build_smoke_finalize_artifacts_are_default_off(tmp_path: Path) -> None:
    output = tmp_path / "autonomous"
    build = build_autonomous_evidence_team_v2(output)
    assert build["role_count"] == 9
    smoke = run_autonomous_evidence_team_v2_smoke(output, mode="fixture")
    assert smoke["role_count"] == 9
    assert smoke["llm_agent_evidence_execution_count"] == 0
    report = finalize_autonomous_evidence_team_v2(
        output,
        tests_passed=True,
        test_count=8,
    )
    assert report["status"] == "accepted_default_off_autonomous_control_plane"
    assert report["runtime_safe_v3_0_remains_default"] is True
    assert report["candidate_default_enabled"] is False
    assert report["promoted_runtime_created"] is False
    assert report["performance_metrics_generated"] is False
    assert report["fake_metric_count"] == 0
    assert (output / "role_registry.json").exists()
    assert (output / "role_smoke_results.csv").exists()
    assert (output / "security_acceptance.json").exists()
    assert (output / "acceptance_report.json").exists()


def test_performance_manifest_freezes_disjoint_groups_and_strong_gate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "performance"
    report = build_autonomous_evidence_performance_manifest_v2(output)
    assert report["status"] == (
        "performance_protocol_frozen_ready_for_selection_training"
    )
    assert report["selection_group_count"] == 10
    assert report["acceptance_group_count"] == 10
    assert report["selection_acceptance_overlap"] == 0
    assert report["acceptance_used_for_selection"] is False
    assert report["performance_metrics_generated"] is False
    manifest = json.loads(
        (output / "performance_manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["selection_groups"]).isdisjoint(
        manifest["acceptance_groups"]
    )
    assert manifest["strong_positive_acceptance_gates"][
        "macro_f1_delta_ge"
    ] == 0.01
    assert "group" in manifest["forbidden_runtime_features"]
    assert manifest["runtime_safe_v3_0_remains_default"] is True


def test_accepted_nfiot_skill_is_selected_without_expanding_scope(
    tmp_path: Path,
) -> None:
    bridge = run_autonomous_positive_skill_bridge_v2(
        tmp_path / "bridge",
        mode="fixture",
    )
    assert bridge["status"] == "accepted_autonomous_iot_skill_bridge"
    assert bridge["selected_skill"] == "IoTBotnetSkill"
    assert bridge["dataset_scope"] == "NF-BoT-IoT-v2"
    assert bridge["classification_metrics_recomputed"] is False
    assert bridge["source_result_reused_with_provenance"] is True
    assert bridge["llm_is_classifier"] is False
    assert bridge["llm_enters_fusion"] is False
    assert bridge["final_decision_owner"] == "FusionAgent"
    assert bridge["fake_metric_count"] == 0
    assert bridge["runtime_safe_v3_0_remains_default"] is True
