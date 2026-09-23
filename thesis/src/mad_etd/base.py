from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    AgentEvidence,
    AgentEvidenceV2,
    ComplianceReport,
    CoordinatorDecision,
    DetectorCapabilityProfile,
    DeliberationReport,
    DetectorInput,
    EvidenceHandoff,
    EvidenceHandoffV2,
    EvidenceRequest,
    FieldAuditResult,
    FlowRecord,
    FutureAnalysisArtifacts,
    FusionResult,
    MemoryHint,
    ReflectionReport,
    ReliabilityProfile,
    ViewAvailabilityProfile,
)


@dataclass(slots=True)
class CaseState:
    flow: FlowRecord
    safe_flow: FlowRecord | None = None
    field_audit: FieldAuditResult | None = None
    reliability: ReliabilityProfile | None = None
    view_availability: ViewAvailabilityProfile | None = None
    detector_capabilities: dict[str, DetectorCapabilityProfile] = field(
        default_factory=dict
    )
    # Default runtime does not consume these compatibility records. They are
    # retained for the SOC evidence-team protocol and its audit trail.
    evidence_requests: list[EvidenceRequest] = field(default_factory=list)
    evidence_handoffs: list[EvidenceHandoff | EvidenceHandoffV2] = field(
        default_factory=list
    )
    evidence_v2: list[AgentEvidenceV2] = field(default_factory=list)
    ood_state: dict[str, Any] = field(default_factory=dict)
    conflict_state: dict[str, Any] = field(default_factory=dict)
    missing_views: list[str] = field(default_factory=list)
    escalation_state: dict[str, Any] = field(default_factory=dict)
    case_lifecycle_status: str = "created"
    audit_chain_ref: str | None = None
    audit_event_ids: list[str] = field(default_factory=list)
    evidence_budget: int | None = None
    advisory_context_refs: list[str] = field(default_factory=list)
    evidence: list[AgentEvidence] = field(default_factory=list)
    enrichment_results: list[AgentEvidence] = field(default_factory=list)
    decisions: list[CoordinatorDecision] = field(default_factory=list)
    memory_hint: MemoryHint | None = None
    deliberations: list[DeliberationReport] = field(default_factory=list)
    reflection: ReflectionReport | None = None
    compliance: ComplianceReport | None = None
    future_artifacts: FutureAnalysisArtifacts = field(
        default_factory=FutureAnalysisArtifacts
    )
    interim_fusion: FusionResult | None = None
    called_agents: set[str] = field(default_factory=set)
    round_no: int = 0
    remaining_budget: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)


class DetectorAgent(ABC):
    name: str
    version: str = "1.0"

    @abstractmethod
    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        raise NotImplementedError


class Coordinator(ABC):
    @abstractmethod
    def decide(self, state: CaseState) -> CoordinatorDecision:
        raise NotImplementedError
