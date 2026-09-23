"""Fail-closed AgentEvidenceV2 admission before the Fusion input boundary.

This module is deliberately default-off.  It validates evidence identity,
capability, freshness, provenance hashes, and safety metadata, but it never
calls Fusion and never emits a final verdict, confidence, or uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationError, model_validator

from .audit import AuditLogger
from .schemas import AgentEvidenceV2, StrictModel


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RAW_OUTPUT_KEYS = frozenset(
    {
        "verdict",
        "final_verdict",
        "final_confidence",
        "final_uncertainty",
        "fusion_result",
        "ood_override",
    }
)
_FORBIDDEN_FUSION_SOURCES = frozenset(
    {"llm", "rag", "memory", "reflection", "critic", "hitl", "human_feedback"}
)
_ACCEPTED_PROMOTION_STATES = frozenset({"accepted_default", "accepted_optional"})


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AdmissionRegistryEntry(StrictModel):
    """Immutable Fusion-admission contract for one evidence specialist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    specialist_id: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    agent_type: str = Field(min_length=1)
    source_kind: str = "detector"
    fusion_eligible: bool = True
    promotion_status: str = "accepted_default"
    required_capabilities: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    feature_policy_hashes: tuple[str, ...] = ()
    artifact_hashes: tuple[str, ...] = ()
    dataset_scopes: tuple[str, ...] = ()
    allowed_safety_flags: tuple[str, ...] = ()
    registry_entry_hash: str = Field(min_length=64, max_length=64)

    @classmethod
    def create(cls, **payload: Any) -> "AdmissionRegistryEntry":
        candidate = {
            "schema_version": "1.0",
            **payload,
            "registry_entry_hash": "0" * 64,
        }
        normalized = cls.model_construct(**candidate).model_dump(mode="json")
        normalized["registry_entry_hash"] = _canonical_sha256(
            {key: value for key, value in normalized.items() if key != "registry_entry_hash"}
        )
        return cls.model_validate(normalized)

    @model_validator(mode="after")
    def contract_is_safe(self) -> "AdmissionRegistryEntry":
        payload = self.model_dump(mode="json")
        expected = _canonical_sha256(
            {key: value for key, value in payload.items() if key != "registry_entry_hash"}
        )
        if self.registry_entry_hash != expected:
            raise ValueError("AdmissionRegistryEntry hash mismatch")
        if self.fusion_eligible and self.promotion_status not in _ACCEPTED_PROMOTION_STATES:
            raise ValueError("only accepted specialists can be Fusion eligible")
        if self.source_kind.lower() in _FORBIDDEN_FUSION_SOURCES and self.fusion_eligible:
            raise ValueError("advisory source cannot be Fusion eligible")
        return self


class AdmissionContext(StrictModel):
    """Case-local information required for admission, without labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    trace_id: str = Field(min_length=1)
    case_state_hash: str = Field(min_length=64, max_length=64)
    current_sequence: int = Field(ge=0)
    available_capabilities: tuple[str, ...] = ()
    dataset_scope: str = "unspecified"


class AdmissionDecision(StrictModel):
    """Audit-friendly gate result; it is not a FusionResult."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    evidence_ref: str = Field(min_length=1)
    specialist_id: str
    admitted: bool
    fusion_eligible: bool
    reason_codes: tuple[str, ...]
    fallback_required: bool
    audit_sequence_no: int = Field(ge=1)
    evidence_semantic_hash: str | None = None


class AgentEvidenceAdmissionGate:
    """Default-off, fail-closed boundary immediately before Fusion input."""

    name = "AgentEvidenceAdmissionGate"
    default_enabled = False
    final_decision_owner = "FusionAgent"

    def __init__(
        self,
        registry: Sequence[AdmissionRegistryEntry],
        *,
        audit_logger: AuditLogger,
        max_sequence_age: int = 2,
    ) -> None:
        self._registry = {entry.specialist_id: entry for entry in registry}
        self.audit_logger = audit_logger
        self.max_sequence_age = max_sequence_age
        self._reviewed_refs: set[str] = set()

    @property
    def registry(self) -> tuple[AdmissionRegistryEntry, ...]:
        return tuple(self._registry.values())

    def admit(
        self,
        raw_evidence: Mapping[str, Any] | AgentEvidenceV2,
        *,
        specialist_id: str,
        evidence_ref: str,
        generated_sequence: int,
        source_kind: str,
        context: AdmissionContext,
    ) -> AdmissionDecision:
        reasons: list[str] = []
        raw: dict[str, Any]
        if isinstance(raw_evidence, AgentEvidenceV2):
            raw = raw_evidence.model_dump(mode="json")
        else:
            raw = dict(raw_evidence)

        illegal_keys = sorted(set(raw) & _FORBIDDEN_RAW_OUTPUT_KEYS)
        if illegal_keys:
            reasons.append("ILLEGAL_FINAL_OR_OOD_OUTPUT")

        entry = self._registry.get(specialist_id)
        if entry is None:
            reasons.append("SPECIALIST_NOT_REGISTERED")
        else:
            if not entry.fusion_eligible:
                reasons.append("SPECIALIST_NOT_FUSION_ELIGIBLE")
            if entry.promotion_status not in _ACCEPTED_PROMOTION_STATES:
                reasons.append("SPECIALIST_NOT_PROMOTED")
            if source_kind.lower() != entry.source_kind.lower():
                reasons.append("SOURCE_KIND_MISMATCH")
            if entry.source_kind.lower() in _FORBIDDEN_FUSION_SOURCES:
                reasons.append("FORBIDDEN_ADVISORY_SOURCE")
            missing = set(entry.required_capabilities) - set(context.available_capabilities)
            if missing:
                reasons.append("CASE_CAPABILITY_MISMATCH")

        if source_kind.lower() in _FORBIDDEN_FUSION_SOURCES:
            reasons.append("FORBIDDEN_ADVISORY_SOURCE")
        if evidence_ref in self._reviewed_refs:
            reasons.append("EVIDENCE_REPLAY_REJECTED")
        if generated_sequence > context.current_sequence:
            reasons.append("EVIDENCE_FROM_FUTURE_SEQUENCE")
        elif context.current_sequence - generated_sequence > self.max_sequence_age:
            reasons.append("STALE_EVIDENCE_REJECTED")

        evidence: AgentEvidenceV2 | None = None
        try:
            evidence = AgentEvidenceV2.model_validate(raw)
        except ValidationError:
            reasons.append("AGENT_EVIDENCE_SCHEMA_INVALID")

        if evidence is not None and entry is not None:
            if evidence.agent_name != entry.agent_name:
                reasons.append("AGENT_IDENTITY_MISMATCH")
            if evidence.agent_type != entry.agent_type:
                reasons.append("AGENT_TYPE_MISMATCH")
            if evidence.input_feature_policy_hash not in entry.feature_policy_hashes:
                reasons.append("FEATURE_POLICY_HASH_MISMATCH")
            if evidence.artifact_hash not in entry.artifact_hashes:
                reasons.append("ARTIFACT_HASH_MISMATCH")
            if not _SHA256_RE.fullmatch(evidence.source_evidence_sha256):
                reasons.append("SOURCE_EVIDENCE_HASH_INVALID")
            if evidence.applicability != "applicable" or evidence.unsupported_reason:
                reasons.append("EVIDENCE_UNSUPPORTED_OR_ABSTAINED")
            if not evidence.contributes_to_verdict:
                reasons.append("EVIDENCE_NOT_VERDICT_STAGE")
            if entry.dataset_scopes and context.dataset_scope not in entry.dataset_scopes:
                reasons.append("DATASET_SCOPE_MISMATCH")
            if entry.dataset_scopes and evidence.dataset_scope not in entry.dataset_scopes:
                reasons.append("EVIDENCE_SCOPE_NOT_REGISTERED")
            if not set(evidence.supported_capabilities).issubset(
                set(entry.allowed_capabilities)
            ):
                reasons.append("EVIDENCE_CAPABILITY_NOT_REGISTERED")
            if not set(evidence.supported_capabilities).issubset(
                set(context.available_capabilities)
            ):
                reasons.append("EVIDENCE_CAPABILITY_NOT_AVAILABLE")
            if not set(evidence.safety_flags).issubset(
                set(entry.allowed_safety_flags)
            ):
                reasons.append("EVIDENCE_SAFETY_FLAG_REJECTED")
            if {"benign", "malicious"} - set(evidence.probabilities):
                reasons.append("BINARY_PROBABILITY_KEYS_MISSING")
            elif sum(
                evidence.probabilities[key] for key in ("benign", "malicious")
            ) > 1.000001:
                reasons.append("BINARY_PROBABILITY_MASS_INVALID")

        self._reviewed_refs.add(evidence_ref)
        unique_reasons = tuple(dict.fromkeys(reasons))
        admitted = not unique_reasons
        semantic_hash = (
            _canonical_sha256(evidence.model_dump(mode="json"))
            if admitted and evidence is not None
            else None
        )
        event = self.audit_logger.log(
            actor=self.name,
            event_type="EVIDENCE_ADMITTED" if admitted else "EVIDENCE_REJECTED",
            input_summary={
                "specialist_id": specialist_id,
                "evidence_ref_hash": _canonical_sha256(evidence_ref),
                "source_kind": source_kind,
                "case_state_hash": context.case_state_hash,
            },
            output_summary={
                "admitted": admitted,
                "fusion_eligible": admitted,
                "reason_codes": list(unique_reasons)
                if unique_reasons
                else ["EVIDENCE_ADMISSION_ACCEPTED"],
                "evidence_semantic_hash": semantic_hash,
            },
            reason="fail_closed_fusion_input_admission",
        )
        return AdmissionDecision(
            evidence_ref=evidence_ref,
            specialist_id=specialist_id,
            admitted=admitted,
            fusion_eligible=admitted,
            reason_codes=unique_reasons
            if unique_reasons
            else ("EVIDENCE_ADMISSION_ACCEPTED",),
            fallback_required=not admitted,
            audit_sequence_no=event.sequence_no,
            evidence_semantic_hash=semantic_hash,
        )

