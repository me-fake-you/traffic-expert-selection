from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(StrEnum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    UNKNOWN = "unknown"


class FieldRole(StrEnum):
    DETECTION_ALLOWED = "DETECTION_ALLOWED"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    BLOCKED = "BLOCKED"
    LABEL_ONLY = "LABEL_ONLY"
    PROVENANCE = "PROVENANCE"
    UNKNOWN = "UNKNOWN"


class FeatureGroup(StrEnum):
    STATS = "stats"
    SEQUENCE = "sequence"
    TLS = "tls"
    PAYLOAD = "payload"
    CONTEXT = "context"


class SequenceFeatures(StrictModel):
    packet_lengths: list[int] = Field(default_factory=list)
    directions: list[Literal[-1, 1]] = Field(default_factory=list)
    iats: list[float] = Field(default_factory=list)
    bursts: list[float] = Field(default_factory=list)
    original_packet_count: int | None = Field(default=None, ge=0)
    truncated: bool = False

    @field_validator("packet_lengths")
    @classmethod
    def non_negative_lengths(cls, value: list[int]) -> list[int]:
        if any(item < 0 for item in value):
            raise ValueError("packet_lengths must be non-negative")
        return value

    @field_validator("iats", "bursts")
    @classmethod
    def non_negative_times(cls, value: list[float]) -> list[float]:
        if any(item < 0 for item in value):
            raise ValueError("time values must be non-negative")
        return value


class TLSRecordSequence(StrictModel):
    """Direction is encoded by the sign of each TLS record length."""

    record_lengths: list[int] = Field(default_factory=list)
    original_record_count: int | None = Field(default=None, ge=0)
    truncated: bool = False

    @field_validator("record_lengths")
    @classmethod
    def non_zero_signed_lengths(cls, value: list[int]) -> list[int]:
        if any(item == 0 for item in value):
            raise ValueError("TLS record lengths must be signed and non-zero")
        return value


class FlowRecord(StrictModel):
    schema_version: str = "1.1"
    trace_id: str
    sample_id: str
    stats: dict[str, float] = Field(default_factory=dict)
    sequence: SequenceFeatures = Field(default_factory=SequenceFeatures)
    tls: dict[str, Any] = Field(default_factory=dict)
    payload_tokens: list[int] | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    labels: dict[str, Any] = Field(default_factory=dict)

    @field_validator("stats")
    @classmethod
    def finite_stats(cls, value: dict[str, float]) -> dict[str, float]:
        for key, item in value.items():
            if not isinstance(item, (int, float)):
                raise ValueError(f"stats.{key} must be numeric")
            if item != item or item in (float("inf"), float("-inf")):
                raise ValueError(f"stats.{key} must be finite")
        return {key: float(item) for key, item in value.items()}


class DetectorInput(StrictModel):
    """The only data object visible to specialist detection agents."""

    stats: dict[str, float] = Field(default_factory=dict)
    sequence: SequenceFeatures = Field(default_factory=SequenceFeatures)
    tls: dict[str, Any] = Field(default_factory=dict)
    payload_tokens: list[int] | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    def visible_field_paths(self) -> list[str]:
        paths: list[str] = []
        for group in (
            "stats",
            "sequence",
            "tls",
            "context",
        ):
            value = getattr(self, group)
            if isinstance(value, BaseModel):
                value = value.model_dump()
            if isinstance(value, dict):
                for key, item in value.items():
                    if item not in (None, {}, []):
                        paths.append(f"{group}.{key}")
        if self.payload_tokens:
            paths.append("payload_tokens")
        return sorted(paths)


class FieldDecision(StrictModel):
    path: str
    role: FieldRole
    risk_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class FieldAuditResult(StrictModel):
    decisions: list[FieldDecision]
    allowed_fields: list[str]
    blocked_fields: list[str]
    context_only_fields: list[str]
    unknown_fields: list[str] = Field(default_factory=list)
    ignored_empty_fields: list[str] = Field(default_factory=list)
    leakage_risk: float = Field(ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


class ReliabilityProfile(StrictModel):
    stats_reliability: float = Field(ge=0, le=1)
    sequence_reliability: float = Field(ge=0, le=1)
    tls_reliability: float = Field(ge=0, le=1)
    payload_reliability: float = Field(ge=0, le=1)
    input_completeness: float = Field(ge=0, le=1)
    ood_suspected: bool = False
    key_features_missing: bool = False
    indicators: list[str] = Field(default_factory=list)

    def for_group(self, group: FeatureGroup) -> float:
        mapping = {
            FeatureGroup.STATS: self.stats_reliability,
            FeatureGroup.SEQUENCE: self.sequence_reliability,
            FeatureGroup.TLS: self.tls_reliability,
            FeatureGroup.PAYLOAD: self.payload_reliability,
            FeatureGroup.CONTEXT: self.input_completeness,
        }
        return mapping[group]


class ViewAvailabilityProfile(StrictModel):
    """Contract-governed detector view availability used for routing only."""

    schema_version: str = "1.0"
    routing_policy: Literal[
        "legacy", "contract_v2_9", "capability_v3_0"
    ] = "legacy"
    stats: bool = False
    sequence: bool = False
    tls: bool = False
    payload: bool = False
    available_fields: dict[str, list[str]] = Field(default_factory=dict)


class DetectorCapabilityProfile(StrictModel):
    """Backend-specific capability decision used for guarded routing."""

    schema_version: str = "1.0"
    agent_name: str
    backend: str
    status: Literal["available", "missing_view", "unsupported_schema"]
    consumed_fields: list[str] = Field(default_factory=list)
    available_fields: list[str] = Field(default_factory=list)
    observed_fields: list[str] = Field(default_factory=list)
    unsupported_fields: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class AgentEvidence(StrictModel):
    agent_name: str
    agent_version: str = "1.0"
    feature_group: FeatureGroup
    benign_support: float = Field(ge=0, le=1)
    malicious_support: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    calibration_quality: float = Field(default=0.8, ge=0, le=1)
    distribution_shift_score: float = Field(default=0, ge=0, le=1)
    distribution_shift_raw_score: float | None = None
    distribution_shift_level: Literal[
        "off", "in_domain", "warning", "hard"
    ] = "off"
    model_reliability: float = Field(default=1, ge=0, le=1)
    abstained: bool = False
    contributes_to_verdict: bool = True
    evidence: list[str] = Field(default_factory=list)
    used_fields: list[str] = Field(default_factory=list)
    family_scores: dict[str, float] = Field(default_factory=dict)
    intent: str = "unknown"
    latency_ms: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def support_mass_is_valid(self) -> "AgentEvidence":
        if self.benign_support + self.malicious_support > 1.000001:
            raise ValueError("benign_support + malicious_support cannot exceed 1")
        return self


_EVIDENCE_REQUEST_SAFE_PREFIXES = (
    "stats.",
    "sequence.",
    "tls.",
    "payload_tokens",
)
_EVIDENCE_REQUEST_BLOCKED_TOKENS = frozenset(
    {
        "ip",
        "port",
        "timestamp",
        "flow_id",
        "flowid",
        "source_file",
        "sample_id",
        "attack",
        "family",
        "provenance",
        "sni",
        "ja3",
        "ja4",
        "resolver",
        "browser",
        "tool",
        "capture",
        "split",
        "label",
        "dataset",
        "source_group",
        "source_identity",
    }
)
_REQUIRED_EVIDENCE_REQUEST_PROHIBITIONS = frozenset(
    {"final_verdict", "final_confidence", "final_uncertainty"}
)
_REQUIRED_EVIDENCE_REQUEST_V2_PROHIBITIONS = frozenset(
    {
        "final_verdict",
        "final_confidence",
        "final_uncertainty",
        "ood_override",
        "raw_blocked_field_access",
    }
)
_EVIDENCE_REQUEST_AUDIT_CONTEXT_KEYS = frozenset(
    {
        "case_lifecycle_status",
        "policy_version",
        "request_origin",
        "audit_chain_ref",
    }
)
_EVIDENCE_REQUEST_FORBIDDEN_PURPOSE_TOKENS = (
    "final verdict",
    "final_verdict",
    "final confidence",
    "final_confidence",
    "final uncertainty",
    "final_uncertainty",
    "override ood",
    "ood override",
    "fusionresult",
    "fusion_result",
)


class EvidenceRequest(StrictModel):
    """A minimal, policy-reviewable delegation contract for one specialist.

    It is deliberately not a verdict-bearing object.  A planner can request
    evidence from a compatible specialist, but cannot ask that specialist to
    emit a final verdict, final confidence, or final uncertainty.
    """

    schema_version: str = "1.0"
    request_id: str = Field(min_length=1)
    case_trace_id: str = Field(min_length=1)
    requested_agent: str = Field(min_length=1)
    permitted_safe_features: list[str] = Field(min_length=1)
    purpose: str = Field(min_length=1)
    budget: int = Field(ge=0)
    allowed_feature_policy_hash: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    timeout_ms: int = Field(default=5000, ge=0)
    prohibited_outputs: list[str] = Field(
        default_factory=lambda: sorted(_REQUIRED_EVIDENCE_REQUEST_PROHIBITIONS)
    )
    forbidden_outputs: list[str] = Field(
        default_factory=lambda: sorted(
            _REQUIRED_EVIDENCE_REQUEST_V2_PROHIBITIONS
        )
    )
    expected_output: Literal["AgentEvidence"] = "AgentEvidence"
    expected_evidence_schema: Literal["AgentEvidence", "AgentEvidenceV2"] = (
        "AgentEvidence"
    )
    audit_context: dict[str, str] = Field(default_factory=dict)
    planner_source: Literal["rule", "llm", "fallback", "replay"] = "rule"
    reason_codes: list[str] = Field(default_factory=list)
    policy_status: Literal["proposed", "approved", "rejected"] = "proposed"

    @model_validator(mode="after")
    def only_delegates_safe_evidence(self) -> "EvidenceRequest":
        for path in self.permitted_safe_features:
            normalized = path.lower().replace("-", "_")
            if not normalized.startswith(_EVIDENCE_REQUEST_SAFE_PREFIXES):
                raise ValueError(f"unsafe EvidenceRequest feature path: {path}")
            if any(token in normalized for token in _EVIDENCE_REQUEST_BLOCKED_TOKENS):
                raise ValueError(f"blocked EvidenceRequest feature path: {path}")
        missing = _REQUIRED_EVIDENCE_REQUEST_PROHIBITIONS - set(
            self.prohibited_outputs
        )
        if missing:
            raise ValueError(
                "EvidenceRequest must prohibit final verdict/confidence/"
                f"uncertainty outputs: {sorted(missing)}"
            )
        missing_v2 = _REQUIRED_EVIDENCE_REQUEST_V2_PROHIBITIONS - set(
            self.forbidden_outputs
        )
        if missing_v2:
            raise ValueError(
                "EvidenceRequest forbidden_outputs must retain final-decision, "
                "OOD-override, and blocked-field prohibitions: "
                f"{sorted(missing_v2)}"
            )
        unknown_audit_keys = set(self.audit_context) - set(
            _EVIDENCE_REQUEST_AUDIT_CONTEXT_KEYS
        )
        if unknown_audit_keys:
            raise ValueError(
                "EvidenceRequest audit_context contains unapproved keys: "
                f"{sorted(unknown_audit_keys)}"
            )
        normalized_purpose = self.purpose.lower()
        if any(
            token in normalized_purpose
            for token in _EVIDENCE_REQUEST_FORBIDDEN_PURPOSE_TOKENS
        ):
            raise ValueError(
                "EvidenceRequest purpose cannot delegate final-decision or "
                "OOD-override authority"
            )
        return self


class EvidenceHandoff(StrictModel):
    """A policy-reviewed return path from a specialist to the evidence bus."""

    schema_version: str = "1.0"
    request_id: str = Field(min_length=1)
    case_trace_id: str = Field(min_length=1)
    requested_agent: str = Field(min_length=1)
    policy_status: Literal["approved", "rejected"]
    evidence: AgentEvidence | None = None
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_requires_a_matching_approved_request(self) -> "EvidenceHandoff":
        if self.policy_status == "approved":
            if self.evidence is None:
                raise ValueError("approved EvidenceHandoff requires AgentEvidence")
            if self.evidence.agent_name != self.requested_agent:
                raise ValueError("EvidenceHandoff agent does not match request")
        elif self.evidence is not None:
            raise ValueError("rejected EvidenceHandoff cannot carry AgentEvidence")
        return self


class AgentEvidenceV2(StrictModel):
    """Governance schema for a future evidence bus; not a FusionResult."""

    schema_version: str = "2.0"
    agent_name: str = Field(min_length=1)
    agent_type: str = Field(min_length=1)
    input_feature_policy_hash: str = Field(min_length=1)
    artifact_hash: str = Field(min_length=1)
    prediction: Literal["benign", "malicious", "abstain"]
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)
    calibration_quality: float = Field(default=0.8, ge=0, le=1)
    distribution_shift_score: float = Field(default=0, ge=0, le=1)
    distribution_shift_raw_score: float | None = None
    distribution_shift_level: Literal[
        "off", "in_domain", "warning", "hard"
    ] = "off"
    contributes_to_verdict: bool = True
    applicability: Literal["applicable", "unsupported", "abstained"]
    unsupported_reason: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    calibration_status: str = "unknown"
    latency_ms: float = Field(default=0, ge=0)
    cost: float = Field(default=0, ge=0)
    supported_capabilities: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    dataset_scope: str = "unspecified"
    promotion_status: str = "not_promoted"
    source_evidence_sha256: str = Field(default="", min_length=0)

    @model_validator(mode="after")
    def probability_and_applicability_contract(self) -> "AgentEvidenceV2":
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("AgentEvidenceV2 probabilities must be in [0, 1]")
        if self.applicability == "unsupported" and not self.unsupported_reason:
            raise ValueError("unsupported AgentEvidenceV2 requires a reason")
        return self


class EvidenceHandoffV2(StrictModel):
    """Policy-reviewed AgentEvidenceV2 transport used by the shadow bus."""

    schema_version: str = "2.0"
    request_id: str = Field(min_length=1)
    case_trace_id: str = Field(min_length=1)
    requested_agent: str = Field(min_length=1)
    policy_status: Literal["approved", "rejected"]
    evidence: AgentEvidenceV2 | None = None
    feature_policy_hash: str = ""
    artifact_hash: str = ""
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def approved_evidence_matches_request(self) -> "EvidenceHandoffV2":
        if self.policy_status == "approved":
            if self.evidence is None:
                raise ValueError(
                    "approved EvidenceHandoffV2 requires AgentEvidenceV2"
                )
            if self.evidence.agent_name != self.requested_agent:
                raise ValueError("EvidenceHandoffV2 agent does not match request")
            if self.feature_policy_hash != self.evidence.input_feature_policy_hash:
                raise ValueError("EvidenceHandoffV2 feature policy hash mismatch")
            if self.artifact_hash != self.evidence.artifact_hash:
                raise ValueError("EvidenceHandoffV2 artifact hash mismatch")
        elif self.evidence is not None:
            raise ValueError(
                "rejected EvidenceHandoffV2 cannot carry AgentEvidenceV2"
            )
        return self


class CoordinatorAction(StrEnum):
    DISPATCH = "DISPATCH"
    STOP_AND_FUSE = "STOP_AND_FUSE"
    ABSTAIN_AND_REPORT = "ABSTAIN_AND_REPORT"


class CoordinatorDecision(StrictModel):
    action: CoordinatorAction
    agents: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    rationale: str = ""
    remaining_budget: int = Field(ge=0)
    source: Literal["rule", "nvidia", "fallback", "replay"] = "rule"
    raw_response: str | None = None
    planner_metadata: dict[str, Any] = Field(default_factory=dict)
    policy_adjustments: list[str] = Field(default_factory=list)
    policy_status: Literal[
        "not_evaluated",
        "accepted",
        "adjusted",
        "fallback",
        "forced_stop",
        "not_applicable",
    ] = "not_evaluated"
    proposed_action: CoordinatorAction | None = None
    proposed_agents: list[str] = Field(default_factory=list)


class LLMPlannerResponse(StrictModel):
    action: CoordinatorAction
    agents: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    rationale: str = ""


class FusionResult(StrictModel):
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    conflict_score: float = Field(ge=0, le=1)
    benign_support: float = Field(ge=0, le=1)
    malicious_support: float = Field(ge=0, le=1)
    distribution_shift_score: float = Field(default=0, ge=0, le=1)
    severity: Literal["none", "low", "medium", "high", "critical"]
    need_escalation: bool
    reasons: list[str] = Field(default_factory=list)


class AttributionResult(StrictModel):
    family: str = "unknown"
    family_confidence: float = Field(default=0, ge=0, le=1)
    intent: str = "unknown"
    intent_confidence: float = Field(default=0, ge=0, le=1)


class AuditEvent(StrictModel):
    schema_version: str = "1.0"
    sequence_no: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    actor: str
    event_type: str
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    duration_ms: float = Field(default=0, ge=0)


class KnowledgeAgentSummary(StrictModel):
    agent_name: str
    feature_group: FeatureGroup
    abstained: bool
    confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    model_reliability: float = Field(ge=0, le=1)
    distribution_shift_level: Literal[
        "off", "in_domain", "warning", "hard"
    ]
    evidence: list[str] = Field(default_factory=list)


class KnowledgeFieldSummary(StrictModel):
    path: str
    role: FieldRole
    reasons: list[str] = Field(default_factory=list)


class KnowledgeQueryContext(StrictModel):
    schema_version: str = "1.0"
    trace_id: str
    fusion_snapshot: FusionResult
    fusion_sha256: str
    agents: list[KnowledgeAgentSummary] = Field(default_factory=list)
    reliability: ReliabilityProfile | None = None
    field_audit: list[KnowledgeFieldSummary] = Field(default_factory=list)
    leakage_risk: float = Field(default=0, ge=0, le=1)
    coordinator_reason_codes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    query_topics: list[str] = Field(default_factory=list)


class KnowledgeReference(StrictModel):
    reference_id: str
    document_id: str
    source_path: str
    heading: str
    chunk_id: str
    content_sha256: str
    retrieval_score: float = Field(ge=0, le=1)
    excerpt: str


class KnowledgeExplanation(StrictModel):
    statement: str
    epistemic_status: Literal[
        "background", "possible_interpretation", "policy_explanation"
    ]
    reference_ids: list[str] = Field(min_length=1)


class KnowledgeSupportResult(StrictModel):
    schema_version: str = "1.0"
    status: Literal[
        "success", "no_match", "failed", "disabled", "policy_blocked"
    ]
    explanatory_only: Literal[True] = True
    retrieval_id: str
    knowledge_base_version: str = ""
    knowledge_base_sha256: str = ""
    retriever_name: str = "local_tfidf"
    retriever_version: str = "1.0"
    retriever_config: dict[str, Any] = Field(default_factory=dict)
    query_topics: list[str] = Field(default_factory=list)
    knowledge_support: list[KnowledgeExplanation] = Field(default_factory=list)
    possible_intent_explanations: list[KnowledgeExplanation] = Field(
        default_factory=list
    )
    response_rationales: list[KnowledgeExplanation] = Field(
        default_factory=list
    )
    policy_explanations: list[KnowledgeExplanation] = Field(
        default_factory=list
    )
    references: list[KnowledgeReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0, ge=0)


class DetectionReport(StrictModel):
    schema_version: str = "1.1"
    trace_id: str
    sample_id: str
    verdict: Verdict
    confidence: float
    uncertainty: float
    conflict_score: float
    benign_support: float
    malicious_support: float
    distribution_shift_score: float = 0
    severity: str
    need_escalation: bool
    traffic_type: str = "encrypted_or_unknown"
    intent: str = "unknown"
    intent_confidence: float = 0
    family: str = "unknown"
    family_confidence: float = 0
    participating_agents: list[str]
    agent_results: list[AgentEvidence]
    enrichment_results: list[AgentEvidence] = Field(default_factory=list)
    main_evidence: list[str]
    blocked_fields: list[str]
    downgraded_feature_groups: dict[str, float]
    leakage_or_pseudo_feature_risks: list[str]
    coordinator_decisions: list[CoordinatorDecision]
    recommended_actions: list[str]
    knowledge_support: KnowledgeSupportResult | None = None
    audit_event_count: int


class DatasetFieldProfile(StrictModel):
    field: str
    role: FieldRole
    unique_ratio: float = Field(ge=0, le=1)
    label_purity: float | None = Field(default=None, ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class DatasetAuditReport(StrictModel):
    schema_version: str = "1.0"
    row_count: int
    label_field: str
    fields: list[DatasetFieldProfile]
    duplicate_row_ratio: float = Field(ge=0, le=1)
    high_risk_fields: list[str]


class FutureFeatureFlags(StrictModel):
    """Opt-in research extensions; every flag is disabled by default."""

    schema_version: str = "1.0"
    memory: bool = False
    planner_executor: bool = False
    deliberation: bool = False
    reflection: bool = False
    audit_critic: bool = False
    learning_queue: bool = False
    evidence_utility_v2: bool = False
    evidence_utility_v2_1: bool = False
    shadow_mode: bool = True


class CaseMemoryRecord(StrictModel):
    schema_version: str = "1.0"
    case_id: str
    source_trace_id: str
    source_report_sha256: str
    source_dataset: str
    case_status: Literal[
        "suspicious",
        "unknown",
        "human_confirmed",
        "false_positive",
        "false_negative",
    ]
    behavior_summary: dict[str, float] = Field(default_factory=dict)
    participating_agents: list[str] = Field(default_factory=list)
    human_outcome: str | None = None
    feedback_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = Field(default=1, ge=1)


class HumanFeedbackRecord(StrictModel):
    schema_version: str = "1.0"
    feedback_id: str
    case_id: str
    reviewer: str
    review_type: Literal[
        "confirm",
        "false_positive",
        "false_negative",
        "needs_more_evidence",
    ]
    confirmed_outcome: str
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supersedes: str | None = None


class MemoryHint(StrictModel):
    schema_version: str = "1.0"
    hint_id: str
    status: Literal["success", "no_match", "failed", "disabled"]
    dispatch_hint: list[str] = Field(default_factory=list)
    explanation_hint: list[str] = Field(default_factory=list)
    retrieval_hint: list[str] = Field(default_factory=list)
    similar_case_refs: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def scores_align_with_references(self) -> "MemoryHint":
        if len(self.similar_case_refs) != len(self.retrieval_scores):
            raise ValueError("similar_case_refs and retrieval_scores must align")
        if any(score < 0 or score > 1 for score in self.retrieval_scores):
            raise ValueError("retrieval scores must be between zero and one")
        return self


class PlanAction(StrEnum):
    RUN_AGENT = "RUN_AGENT"
    FUSE = "FUSE"
    STOP = "STOP"


class PlanStep(StrictModel):
    step_id: str
    action: PlanAction
    agent: str | None = None
    condition: str = "always"
    dependencies: list[str] = Field(default_factory=list)
    budget_cost: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def agent_required_for_run(self) -> "PlanStep":
        if self.action == PlanAction.RUN_AGENT and not self.agent:
            raise ValueError("RUN_AGENT plan steps require an agent")
        if self.action != PlanAction.RUN_AGENT and self.agent is not None:
            raise ValueError("only RUN_AGENT plan steps may name an agent")
        return self


class ExecutionPlan(StrictModel):
    schema_version: str = "1.0"
    plan_id: str
    steps: list[PlanStep] = Field(min_length=1)
    reason_codes: list[str] = Field(default_factory=list)
    budget: int = Field(ge=0)
    termination_condition: str = "stop_or_budget_exhausted"
    source: Literal["rule", "llm", "fallback", "replay"] = "rule"
    planner_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def step_ids_are_unique(self) -> "ExecutionPlan":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique")
        known: set[str] = set()
        for step in self.steps:
            if not set(step.dependencies).issubset(known):
                raise ValueError("plan dependencies must refer to earlier steps")
            known.add(step.step_id)
        if sum(step.budget_cost for step in self.steps) > self.budget:
            raise ValueError("plan exceeds declared budget")
        return self


class EvidenceUtilityEstimate(StrictModel):
    """Advisory estimate for whether another detector call is worthwhile."""

    schema_version: str = "1.0"
    agent: str
    median_utility: float
    lower_utility: float
    expected_uncertainty_reduction: float = 0.0
    expected_conflict_reduction: float = 0.0
    estimated_cost: float = Field(default=1.0, ge=0)
    should_dispatch: bool
    feature_names: list[str] = Field(default_factory=list)
    policy_version: str = "evidence-utility-v2"


class ConformalAssessment(StrictModel):
    """Detector-side reliability annotation; not standalone Fusion evidence."""

    schema_version: str = "1.0"
    benign_p_value: float = Field(ge=0, le=1)
    malicious_p_value: float = Field(ge=0, le=1)
    prediction_set: list[Literal["benign", "malicious"]] = Field(
        default_factory=list
    )
    shift_score: float = Field(ge=0, le=1)
    level: Literal["in_domain", "warning", "hard"]
    calibration_scope: str = "within_dataset_only"


class ExecutorResult(StrictModel):
    schema_version: str = "1.0"
    plan_id: str
    executed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    budget_used: int = Field(default=0, ge=0)
    evidence_refs: list[str] = Field(default_factory=list)


class DeliberationReport(StrictModel):
    schema_version: str = "1.0"
    status: Literal["not_triggered", "completed", "failed", "disabled"]
    conflict_sources: list[str] = Field(default_factory=list)
    reliability_context: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    recommended_dispatch: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ReflectionReport(StrictModel):
    schema_version: str = "1.0"
    status: Literal["completed", "not_needed", "failed", "disabled"]
    failure_modes: list[str] = Field(default_factory=list)
    least_reliable_view: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    future_collection_hint: list[str] = Field(default_factory=list)
    epistemic_status: Literal["meta_analysis_only"] = "meta_analysis_only"


class AuditFinding(StrictModel):
    schema_version: str = "1.0"
    finding_id: str
    rule_id: str
    severity: Literal["info", "low", "medium", "high", "critical"]
    subject_event_refs: list[int] = Field(default_factory=list)
    description: str
    remediation: str = ""


class ComplianceReport(StrictModel):
    schema_version: str = "1.0"
    trace_id: str
    findings: list[AuditFinding] = Field(default_factory=list)
    policy_version: str = "future-contract-v1"
    audit_coverage: float = Field(ge=0, le=1)
    status: Literal["compliant", "non_compliant", "incomplete"]


class TrainingCandidate(StrictModel):
    schema_version: str = "1.0"
    candidate_id: str
    case_ref: str
    feedback_ref: str
    eligibility_reasons: list[str] = Field(default_factory=list)
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FutureAnalysisArtifacts(StrictModel):
    schema_version: str = "1.0"
    memory_hint: MemoryHint | None = None
    execution_plans: list[ExecutionPlan] = Field(default_factory=list)
    executor_results: list[ExecutorResult] = Field(default_factory=list)
    deliberations: list[DeliberationReport] = Field(default_factory=list)
    utility_estimates: list[EvidenceUtilityEstimate] = Field(default_factory=list)
    reflection: ReflectionReport | None = None
    compliance: ComplianceReport | None = None
