from __future__ import annotations

import hashlib

from mad_etd.audit import AuditLogger
from mad_etd.evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)


POLICY_HASH = hashlib.sha256(b"safe-policy").hexdigest()
ARTIFACT_HASH = hashlib.sha256(b"frozen-artifact").hexdigest()
SOURCE_HASH = hashlib.sha256(b"source-evidence").hexdigest()
CASE_HASH = hashlib.sha256(b"case-state").hexdigest()


def _registry() -> list[AdmissionRegistryEntry]:
    return [
        AdmissionRegistryEntry.create(
            specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
            agent_name="StatsDetectorAgent",
            agent_type="stats:frozen",
            source_kind="detector",
            fusion_eligible=True,
            promotion_status="accepted_default",
            required_capabilities=("stats.safe",),
            allowed_capabilities=("stats.safe",),
            feature_policy_hashes=(POLICY_HASH,),
            artifact_hashes=(ARTIFACT_HASH,),
            dataset_scopes=("frozen_acceptance",),
        )
    ]


def _context(sequence: int = 4) -> AdmissionContext:
    return AdmissionContext(
        trace_id="trace-1",
        case_state_hash=CASE_HASH,
        current_sequence=sequence,
        available_capabilities=("stats.safe",),
        dataset_scope="frozen_acceptance",
    )


def _evidence() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "agent_name": "StatsDetectorAgent",
        "agent_type": "stats:frozen",
        "input_feature_policy_hash": POLICY_HASH,
        "artifact_hash": ARTIFACT_HASH,
        "prediction": "benign",
        "probabilities": {"benign": 0.8, "malicious": 0.2},
        "confidence": 0.8,
        "uncertainty": 0.2,
        "reliability": 0.9,
        "applicability": "applicable",
        "supported_capabilities": ["stats.safe"],
        "safety_flags": [],
        "dataset_scope": "frozen_acceptance",
        "promotion_status": "accepted_default",
        "source_evidence_sha256": SOURCE_HASH,
    }


def _gate() -> tuple[AgentEvidenceAdmissionGate, AuditLogger]:
    logger = AuditLogger("trace-1")
    return AgentEvidenceAdmissionGate(_registry(), audit_logger=logger), logger


def test_admission_accepts_registered_fresh_evidence_and_logs_it():
    gate, logger = _gate()
    decision = gate.admit(
        _evidence(),
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-1",
        generated_sequence=3,
        source_kind="detector",
        context=_context(),
    )

    assert decision.admitted is True
    assert decision.fusion_eligible is True
    assert decision.evidence_semantic_hash
    assert logger.events[-1].event_type == "EVIDENCE_ADMITTED"


def test_admission_rejects_replay_and_stale_evidence():
    gate, logger = _gate()
    first = gate.admit(
        _evidence(),
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-replay",
        generated_sequence=3,
        source_kind="detector",
        context=_context(),
    )
    replay = gate.admit(
        _evidence(),
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-replay",
        generated_sequence=3,
        source_kind="detector",
        context=_context(),
    )
    stale = gate.admit(
        _evidence(),
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-stale",
        generated_sequence=0,
        source_kind="detector",
        context=_context(),
    )

    assert first.admitted is True
    assert replay.admitted is False
    assert "EVIDENCE_REPLAY_REJECTED" in replay.reason_codes
    assert stale.admitted is False
    assert "STALE_EVIDENCE_REJECTED" in stale.reason_codes
    assert len(logger.events) == 3


def test_admission_rejects_advisory_and_illegal_final_output():
    gate, _ = _gate()
    payload = _evidence()
    payload["final_verdict"] = "malicious"
    decision = gate.admit(
        payload,
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-llm",
        generated_sequence=4,
        source_kind="llm",
        context=_context(),
    )

    assert decision.admitted is False
    assert "FORBIDDEN_ADVISORY_SOURCE" in decision.reason_codes
    assert "ILLEGAL_FINAL_OR_OOD_OUTPUT" in decision.reason_codes
    assert "AGENT_EVIDENCE_SCHEMA_INVALID" in decision.reason_codes


def test_admission_rejects_hash_capability_and_safety_mismatches():
    gate, _ = _gate()
    payload = _evidence()
    payload["input_feature_policy_hash"] = hashlib.sha256(b"wrong").hexdigest()
    payload["supported_capabilities"] = ["stats.IP"]
    payload["safety_flags"] = ["blocked_field_access_attempt"]
    decision = gate.admit(
        payload,
        specialist_id="StatsDetectorAgent.runtime_safe_v3_0",
        evidence_ref="ev-bad",
        generated_sequence=4,
        source_kind="detector",
        context=_context(),
    )

    assert decision.admitted is False
    assert "FEATURE_POLICY_HASH_MISMATCH" in decision.reason_codes
    assert "EVIDENCE_CAPABILITY_NOT_REGISTERED" in decision.reason_codes
    assert "EVIDENCE_CAPABILITY_NOT_AVAILABLE" in decision.reason_codes
    assert "EVIDENCE_SAFETY_FLAG_REJECTED" in decision.reason_codes

