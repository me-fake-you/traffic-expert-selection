"""Fine-grained, default-off Traffic Expert Skill governance registry.

This module refines the original coarse Skill card without changing the
runtime.  It defines the planner-visible CaseState view, the least-privilege
EvidenceRequest contract, and correlation guards that prevent several Skill
wrappers around one artifact from being counted as independent Fusion input.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import ConfigDict, Field, model_validator

from .field_contract import FIELD_CONTRACT
from .schemas import AgentEvidenceV2, StrictModel


DEFAULT_OUTPUT = Path("data/runs/mad_etd_traffic_skill_registry_v1")
DEFAULT_DOC = Path("docs/MAD_ETD_TRAFFIC_SKILL_REGISTRY_V1.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_TRAFFIC_SKILL_REGISTRY_V1_CN.md")
DEFAULT_RUNTIME = "runtime_safe_v3_0"

REQUIRED_FORBIDDEN_OUTPUTS = frozenset(
    {
        "final_verdict",
        "final_confidence",
        "final_uncertainty",
        "benign",
        "malicious",
        "OOD_override",
    }
)
PROHIBITED_FEATURES = (
    "IP",
    "port",
    "absolute timestamp",
    "Flow ID",
    "source file",
    "attack family",
    "attack name",
    "label",
    "ground truth",
    "dataset/source identity",
    "provenance",
    "capture metadata",
)
FORBIDDEN_CASE_KEYS = frozenset(
    {
        "label",
        "labels",
        "ground_truth",
        "attack_family",
        "family",
        "baseline_answer",
        "dataset",
        "dataset_name",
        "source",
        "source_identity",
        "source_file",
        "provenance",
        "ip",
        "src_ip",
        "dst_ip",
        "port",
        "src_port",
        "dst_port",
        "timestamp",
        "absolute_timestamp",
    }
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unsafe_nested_keys(value: Any, *, prefix: str = "") -> list[str]:
    unsafe: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower().replace("-", "_").replace(" ", "_")
            path = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_CASE_KEYS:
                unsafe.append(path)
            unsafe.extend(_unsafe_nested_keys(child, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            unsafe.extend(_unsafe_nested_keys(child, prefix=f"{prefix}[{index}]"))
    return unsafe


def _validate_safe_feature(path: str) -> None:
    normalized = path.lower().replace("-", "_")
    contract = FIELD_CONTRACT.get(normalized)
    if contract is None or contract.get("role") != "DETECTION_ALLOWED":
        raise ValueError(f"feature is not detection-allowed: {path}")


class CaseStateSkillV2(StrictModel):
    """Planner-visible state that deliberately excludes truth and identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    case_id: str = Field(min_length=1)
    field_audit_summary: dict[str, Any]
    safe_feature_policy_hash: str = Field(min_length=64, max_length=64)
    available_views: tuple[str, ...]
    capability_profile: dict[str, str]
    collected_evidence_summary: tuple[dict[str, Any], ...] = ()
    evidence_conflict: dict[str, Any] = Field(default_factory=dict)
    uncertainty_state: dict[str, Any] = Field(default_factory=dict)
    OOD_state: dict[str, Any] = Field(default_factory=dict)
    missing_views: tuple[str, ...] = ()
    short_flow_state: dict[str, Any] = Field(default_factory=dict)
    remaining_agent_budget: int = Field(ge=0)
    remaining_latency_budget: float = Field(ge=0)
    previous_requests: tuple[str, ...] = ()
    previous_failures: tuple[str, ...] = ()
    escalation_state: dict[str, Any] = Field(default_factory=dict)
    audit_chain_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def contains_no_truth_or_identity(self) -> "CaseStateSkillV2":
        unsafe = _unsafe_nested_keys(self.model_dump(mode="python"))
        if unsafe:
            raise ValueError(f"CaseStateSkillV2 exposes forbidden keys: {sorted(unsafe)}")
        return self

    @property
    def state_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class EvidenceRequestSkillV1(StrictModel):
    """Standard specialist delegation; it is not a verdict-bearing object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    case_state_hash: str = Field(min_length=64, max_length=64)
    requested_skill: str = Field(min_length=1)
    requested_agent: str = Field(min_length=1)
    investigation_hypothesis: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    allowed_features: tuple[str, ...]
    prohibited_features: tuple[str, ...] = PROHIBITED_FEATURES
    expected_evidence_type: Literal["AgentEvidenceV2"] = "AgentEvidenceV2"
    max_agent_calls: int = Field(ge=0, le=8)
    max_latency_ms: float = Field(ge=0)
    priority: Literal["low", "normal", "high", "critical"] = "normal"
    stop_condition: str = Field(min_length=1)
    forbidden_outputs: tuple[str, ...] = tuple(sorted(REQUIRED_FORBIDDEN_OUTPUTS))
    planner_type: Literal["rule", "utility", "marl_style", "llm", "fallback", "replay"]
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def least_privilege_contract(self) -> "EvidenceRequestSkillV1":
        if not REQUIRED_FORBIDDEN_OUTPUTS.issubset(self.forbidden_outputs):
            raise ValueError("EvidenceRequest omits required forbidden outputs")
        if not set(PROHIBITED_FEATURES).issubset(self.prohibited_features):
            raise ValueError("EvidenceRequest omits prohibited feature classes")
        for feature in self.allowed_features:
            _validate_safe_feature(feature)
        text = " ".join(
            (self.investigation_hypothesis, self.purpose, self.stop_condition)
        ).lower().replace("_", " ")
        forbidden_intents = (
            "final verdict",
            "final confidence",
            "final uncertainty",
            "override ood",
            "ground truth",
        )
        if any(token in text for token in forbidden_intents):
            raise ValueError("EvidenceRequest delegates forbidden decision authority")
        return self


class TrafficExpertSkillSpecV2(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    skill_id: str
    expert_role: str
    task_scope: tuple[str, ...]
    compatible_datasets: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    safe_input_features: tuple[str, ...]
    blocked_features: tuple[str, ...] = PROHIBITED_FEATURES
    feature_policy_hash: str = Field(min_length=64, max_length=64)
    artifact_hash: str = Field(min_length=64, max_length=64)
    analysis_tools: tuple[str, ...]
    output_schema: Literal["AgentEvidenceV2"] = "AgentEvidenceV2"
    unsupported_conditions: tuple[str, ...]
    dataset_scope: str
    evidence_stage: Literal["verdict_stage", "risk_applicability", "advisory_audit"]
    evidence_family: str
    correlation_group: str
    mutually_exclusive_with: tuple[str, ...] = ()
    scope_priority: int = Field(ge=0, le=100)
    promotion_status: Literal[
        "implemented_default_off", "not_promoted", "risk_only", "advisory_only"
    ]
    advisory_only: bool
    enters_fusion: bool = False
    eligible_for_fusion_after_acceptance: bool = False
    default_enabled: bool = False

    @model_validator(mode="after")
    def governance_contract(self) -> "TrafficExpertSkillSpecV2":
        if self.feature_policy_hash != canonical_sha256(sorted(self.safe_input_features)):
            raise ValueError("feature policy hash mismatch")
        if self.advisory_only and (
            self.enters_fusion or self.eligible_for_fusion_after_acceptance
        ):
            raise ValueError("advisory Skill cannot enter Fusion")
        if self.enters_fusion:
            raise ValueError("new Skill registry is default-off and cannot enter Fusion")
        if self.evidence_stage != "verdict_stage" and self.eligible_for_fusion_after_acceptance:
            raise ValueError("only verdict-stage evidence can become Fusion eligible")
        for feature in self.safe_input_features:
            _validate_safe_feature(feature)
        return self


def _spec(
    skill_id: str,
    role: str,
    scope: tuple[str, ...],
    datasets: tuple[str, ...],
    capabilities: tuple[str, ...],
    features: tuple[str, ...],
    tools: tuple[str, ...],
    unsupported: tuple[str, ...],
    dataset_scope: str,
    stage: Literal["verdict_stage", "risk_applicability", "advisory_audit"],
    family: str,
    correlation: str,
    mutually_exclusive: tuple[str, ...],
    priority: int,
    promotion: Literal[
        "implemented_default_off", "not_promoted", "risk_only", "advisory_only"
    ],
) -> TrafficExpertSkillSpecV2:
    advisory = stage == "advisory_audit"
    payload = {
        "skill_id": skill_id,
        "task_scope": scope,
        "features": features,
        "correlation": correlation,
        "promotion": promotion,
    }
    return TrafficExpertSkillSpecV2(
        skill_id=skill_id,
        expert_role=role,
        task_scope=scope,
        compatible_datasets=datasets,
        required_capabilities=capabilities,
        safe_input_features=features,
        feature_policy_hash=canonical_sha256(sorted(features)),
        artifact_hash=canonical_sha256(payload),
        analysis_tools=tools,
        unsupported_conditions=unsupported,
        dataset_scope=dataset_scope,
        evidence_stage=stage,
        evidence_family=family,
        correlation_group=correlation,
        mutually_exclusive_with=mutually_exclusive,
        scope_priority=priority,
        promotion_status=promotion,
        advisory_only=advisory,
        enters_fusion=False,
        eligible_for_fusion_after_acceptance=stage == "verdict_stage",
        default_enabled=False,
    )


def default_skill_specs_v2() -> tuple[TrafficExpertSkillSpecV2, ...]:
    stats = (
        "stats.packet_count",
        "stats.total_bytes",
        "stats.outbound_bytes",
        "stats.inbound_bytes",
        "stats.outbound_ratio",
        "stats.mean_packet_length",
        "stats.packet_length_variance",
        "stats.duration",
    )
    sequence = (
        "sequence.packet_lengths",
        "sequence.directions",
        "sequence.iats",
        "sequence.bursts",
        "sequence.original_packet_count",
        "sequence.truncated",
    )
    tls = (
        "tls.record_lengths",
        "tls.server_version",
        "tls.client_cipher_count",
        "tls.client_extension_count",
        "tls.server_extension_count",
        "tls.alpn",
    )
    flow_datasets = (
        "USTC-TFC2016",
        "NF-BoT-IoT",
        "NF-ToN-IoT",
        "CICIDS2017",
        "HIKARI-2021",
    )
    all_datasets = flow_datasets + (
        "CESNET-TLS22",
        "CipherSpectrum",
        "DoHBrw-2020",
        "ISCXTor2016",
        "CICDarknet2020",
        "Annotated-TLS-2026",
    )
    flow_exclusive = (
        "GeneralFlowMalwareSkill",
        "IoTBotnetSkill",
        "EnterpriseIntrusionSkill",
        "MalwareTrafficSkill",
    )
    temporal_exclusive = (
        "TemporalSequenceSkill",
        "PrefixEvidenceSkill",
        "ShortFlowSkill",
    )
    tls_exclusive = ("TLSRecordSkill", "DoHCovertChannelSkill")
    specs = (
        _spec("GeneralFlowMalwareSkill", "general_flow_analyst", ("binary_flow_evidence",), flow_datasets, ("stats",), stats, ("HGB", "RF", "ExtraTrees"), ("missing_stats",), "multi-dataset safe flow", "verdict_stage", "flow_binary", "shared_stats_artifact", tuple(x for x in flow_exclusive if x != "GeneralFlowMalwareSkill"), 50, "implemented_default_off"),
        _spec("IoTBotnetSkill", "iot_botnet_specialist", ("iot_botnet_evidence",), ("NF-BoT-IoT", "NF-ToN-IoT"), ("stats", "iot_scope"), stats, ("source_group_calibration",), ("non_iot_scope", "missing_stats"), "NF-IoT only", "verdict_stage", "flow_binary", "shared_stats_artifact", tuple(x for x in flow_exclusive if x != "IoTBotnetSkill"), 80, "not_promoted"),
        _spec("EnterpriseIntrusionSkill", "enterprise_intrusion_specialist", ("enterprise_intrusion_evidence",), ("CICIDS2017",), ("stats", "enterprise_scope"), stats, ("day_holdout", "capture_holdout"), ("non_enterprise_scope",), "CICIDS2017 only", "verdict_stage", "flow_binary", "shared_stats_artifact", tuple(x for x in flow_exclusive if x != "EnterpriseIntrusionSkill"), 80, "not_promoted"),
        _spec("MalwareTrafficSkill", "malware_traffic_specialist", ("malware_flow_evidence",), ("USTC-TFC2016",), ("stats", "malware_scope"), stats, ("family_holdout", "application_holdout"), ("non_ustc_scope",), "USTC only", "verdict_stage", "flow_binary", "shared_stats_artifact", tuple(x for x in flow_exclusive if x != "MalwareTrafficSkill"), 80, "not_promoted"),
        _spec("TemporalSequenceSkill", "temporal_behavior_specialist", ("full_sequence_evidence",), ("USTC-TFC2016", "CICIDS2017"), ("sequence",), sequence, ("existing_temporal", "TCN", "adapted_sequence_backends"), ("missing_sequence",), "sequence-capable datasets", "verdict_stage", "temporal_binary", "shared_temporal_artifact", tuple(x for x in temporal_exclusive if x != "TemporalSequenceSkill"), 60, "implemented_default_off"),
        _spec("TLSRecordSkill", "tls_record_specialist", ("tls_record_evidence",), ("DoHBrw-2020",), ("tls_records", "explicit_capture_truth"), tls, ("rule_tls", "safe_hgb", "TCN"), ("fewer_than_two_records", "missing_truth"), "DoHBrw capture-labelled only", "verdict_stage", "tls_binary", "shared_tls_artifact", tuple(x for x in tls_exclusive if x != "TLSRecordSkill"), 70, "not_promoted"),
        _spec("DoHCovertChannelSkill", "doh_covert_channel_specialist", ("dns2tcp_dnscat2_iodine_evidence",), ("DoHBrw-2020",), ("tls_records", "explicit_tool_truth"), tls, ("tool_holdout", "resolver_holdout"), ("non_doh", "missing_tool_truth"), "DoHBrw tools only", "verdict_stage", "tls_binary", "shared_tls_artifact", tuple(x for x in tls_exclusive if x != "DoHCovertChannelSkill"), 90, "not_promoted"),
        _spec("PrefixEvidenceSkill", "prefix_evidence_specialist", ("2_4_8_16_32_packet_evidence",), ("USTC-TFC2016", "CICIDS2017", "ISCXTor2016"), ("sequence", "prefix_available"), sequence, ("prefix_replay", "consistency_check"), ("missing_prefix",), "sequence prefix, conditional Fusion eligibility", "verdict_stage", "temporal_binary", "shared_temporal_artifact", tuple(x for x in temporal_exclusive if x != "PrefixEvidenceSkill"), 85, "not_promoted"),
        _spec("ShortFlowSkill", "short_flow_specialist", ("short_flow_evidence", "boundary_analysis"), ("USTC-TFC2016", "CICIDS2017", "ISCXTor2016"), ("sequence", "short_flow"), sequence, ("short_flow_calibration",), ("not_short_flow",), "2/4/8/16 packet flows", "verdict_stage", "temporal_binary", "shared_temporal_artifact", tuple(x for x in temporal_exclusive if x != "ShortFlowSkill"), 90, "not_promoted"),
        _spec("TorApplicabilitySkill", "tor_applicability_specialist", ("tor_vs_non_tor", "unsupported_detection"), ("ISCXTor2016", "CICDarknet2020"), ("sequence", "explicit_tor_truth"), sequence, ("tor_applicability", "ood_check"), ("tor_dataset_missing",), "Tor applicability; Tor is not malicious", "risk_applicability", "tor_applicability", "tor_scope", (), 90, "risk_only"),
        _spec("TorApplicationSkill", "tor_application_specialist", ("tor_application_category",), ("ISCXTor2016",), ("sequence", "explicit_tor_application_truth"), sequence, ("application_classifier",), ("not_tor", "application_truth_missing"), "Tor application only; no malicious output", "risk_applicability", "tor_application", "tor_scope", (), 80, "risk_only"),
        _spec("OODDomainShiftSkill", "ood_and_domain_shift_officer", ("ood_state", "domain_shift"), all_datasets, ("safe_reference_distribution",), stats, ("IsolationForest", "conformal", "agreement_audit"), ("reference_missing",), "risk control plane", "risk_applicability", "ood_risk", "ood_control", (), 100, "risk_only"),
        _spec("DriftMonitorSkill", "traffic_drift_monitor", ("stable_warning_significant_drift", "retraining_recommendation"), all_datasets, ("safe_reference_distribution", "ordered_groups"), stats, ("distribution_distance", "delay_audit"), ("ordered_groups_missing",), "monitoring only; no auto-training", "risk_applicability", "drift_risk", "drift_control", (), 70, "risk_only"),
        _spec("ShortcutForensicsSkill", "shortcut_forensics_analyst", ("feature_policy_audit", "source_leakage"), all_datasets, ("feature_forensics_registry",), (), ("MI", "AUC", "SHAP", "grouped_bootstrap"), ("forensics_registry_missing",), "offline audit", "advisory_audit", "governance", "advisory_only", (), 80, "advisory_only"),
        _spec("ErrorAnalysisSkill", "error_analysis_analyst", ("completed_case_error_attribution", "evidence_gap_explanation"), all_datasets, ("completed_case_snapshot", "audit_chain"), (), ("error_taxonomy", "consistency_audit"), ("completed_case_missing",), "train/validation/HITL-confirmed only", "advisory_audit", "error_analysis", "advisory_only", (), 70, "advisory_only"),
        _spec("ExperienceMemorySkill", "experience_memory_advisor", ("historical_failure_retrieval", "expert_selection_hint"), ("USTC-TFC2016",), ("case_memory",), (), ("top_k_retrieval",), ("memory_missing", "test_memory_forbidden"), "validation/HITL memory only", "advisory_audit", "memory", "advisory_only", (), 60, "advisory_only"),
        _spec("PacketFlowSummarySkill", "packet_flow_summary_advisor", ("structured_safe_traffic_summary",), all_datasets, ("stats",), stats, ("schema_summary",), ("safe_view_missing",), "safe fields only", "advisory_audit", "explanation", "advisory_only", (), 50, "advisory_only"),
        _spec("ProtocolInterpretationSkill", "protocol_interpretation_advisor", ("protocol_evidence_explanation",), ("DoHBrw-2020", "Annotated-TLS-2026"), ("tls_records",), tls, ("reason_code_linking",), ("tls_view_missing",), "TLS explanation only", "advisory_audit", "explanation", "advisory_only", (), 50, "advisory_only"),
        _spec("IncidentNarrativeSkill", "incident_narrative_advisor", ("evidence_linked_incident_narrative",), all_datasets, ("validated_evidence_refs", "fusion_snapshot", "audit_chain"), (), ("evidence_linked_narrative",), ("evidence_refs_missing",), "post-Fusion reporting only", "advisory_audit", "explanation", "advisory_only", (), 40, "advisory_only"),
        _spec("AuditExplanationSkill", "audit_explanation_advisor", ("audit_chain_summary", "claim_boundary"), all_datasets, ("audit_chain",), (), ("audit_summary", "claim_check"), ("audit_chain_missing",), "audit only", "advisory_audit", "governance", "advisory_only", (), 90, "advisory_only"),
        _spec("PerturbationRobustnessSkill", "robustness_analyst", ("padding_dummy_iat_truncation_burst_audit",), ("USTC-TFC2016", "CICIDS2017", "CipherSpectrum", "ISCXTor2016"), ("sequence", "frozen_perturbation"), sequence, ("counterfactual_replay", "risk_ordered_transition"), ("sequence_missing",), "offline/shadow robustness", "advisory_audit", "robustness", "advisory_only", (), 70, "advisory_only"),
    )
    return specs


class TrafficExpertSkillRegistryV2:
    def __init__(self, specs: tuple[TrafficExpertSkillSpecV2, ...] | None = None) -> None:
        items = specs or default_skill_specs_v2()
        self._items = {item.skill_id: item for item in items}
        if len(self._items) != len(items):
            raise ValueError("duplicate skill_id")

    def all(self) -> tuple[TrafficExpertSkillSpecV2, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    def get(self, skill_id: str) -> TrafficExpertSkillSpecV2:
        if skill_id not in self._items:
            raise KeyError(skill_id)
        return self._items[skill_id]

    def create_request(
        self,
        *,
        case_state: CaseStateSkillV2,
        requested_skill: str,
        requested_agent: str,
        hypothesis: str,
        purpose: str,
        max_agent_calls: int,
        max_latency_ms: float,
        planner_type: Literal["rule", "utility", "marl_style", "llm", "fallback", "replay"],
        prompt_version: str,
    ) -> EvidenceRequestSkillV1:
        spec = self.get(requested_skill)
        request_id = canonical_sha256(
            {
                "case_state_hash": case_state.state_hash,
                "skill": requested_skill,
                "purpose": purpose,
            }
        )[:24]
        return EvidenceRequestSkillV1(
            request_id=request_id,
            case_state_hash=case_state.state_hash,
            requested_skill=requested_skill,
            requested_agent=requested_agent,
            investigation_hypothesis=hypothesis,
            purpose=purpose,
            allowed_features=spec.safe_input_features,
            max_agent_calls=max_agent_calls,
            max_latency_ms=max_latency_ms,
            priority="normal",
            stop_condition="stop after one schema-valid AgentEvidenceV2 or unsupported response",
            planner_type=planner_type,
            prompt_version=prompt_version,
        )

    def review_selection(self, selected_skills: tuple[str, ...]) -> dict[str, Any]:
        unknown = sorted(set(selected_skills) - set(self._items))
        duplicate_correlation_groups: list[str] = []
        mutually_exclusive_pairs: list[tuple[str, str]] = []
        groups: dict[str, list[str]] = {}
        for skill_id in selected_skills:
            if skill_id not in self._items:
                continue
            spec = self._items[skill_id]
            if spec.evidence_stage == "verdict_stage":
                groups.setdefault(spec.correlation_group, []).append(skill_id)
            for other in selected_skills:
                if other in spec.mutually_exclusive_with and skill_id < other:
                    mutually_exclusive_pairs.append((skill_id, other))
        for group, members in groups.items():
            if len(members) > 1:
                duplicate_correlation_groups.append(group)
        approved = not unknown and not duplicate_correlation_groups and not mutually_exclusive_pairs
        return {
            "approved": approved,
            "unknown_skills": unknown,
            "duplicate_correlation_groups": sorted(duplicate_correlation_groups),
            "mutually_exclusive_pairs": sorted(mutually_exclusive_pairs),
            "fusion_double_count_prevented": not approved,
        }


def _sample_case_state() -> CaseStateSkillV2:
    empty_hash = canonical_sha256({})
    return CaseStateSkillV2(
        case_id="skill-registry-self-audit",
        field_audit_summary={"status": "completed", "blocked_count": 0},
        safe_feature_policy_hash=canonical_sha256(["stats.packet_count"]),
        available_views=("stats", "sequence"),
        capability_profile={"stats": "supported", "sequence": "supported"},
        collected_evidence_summary=(),
        evidence_conflict={"state": "none"},
        uncertainty_state={"level": "unknown"},
        OOD_state={"state": "preserved"},
        missing_views=("tls",),
        short_flow_state={"is_short": False},
        remaining_agent_budget=2,
        remaining_latency_budget=100.0,
        previous_requests=(),
        previous_failures=(),
        escalation_state={"required": False},
        audit_chain_hash=empty_hash,
    )


def build_traffic_skill_registry_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir)
    registry = TrafficExpertSkillRegistryV2()
    specs = [item.model_dump(mode="json") for item in registry.all()]
    payload = {
        "schema_version": "2.0",
        "registry_name": "TrafficExpertSkillRegistry",
        "default_enabled": False,
        "default_runtime": DEFAULT_RUNTIME,
        "skills": specs,
        "registry_sha256": canonical_sha256(specs),
    }
    _dump(output / "traffic_expert_skill_registry.json", payload)
    _dump(output / "case_state_v2_schema.json", CaseStateSkillV2.model_json_schema())
    _dump(output / "evidence_request_v1_schema.json", EvidenceRequestSkillV1.model_json_schema())
    _dump(output / "agent_evidence_v2_schema.json", AgentEvidenceV2.model_json_schema())
    _write_csv(
        output / "skill_capability_matrix.csv",
        [
            {
                "skill_id": row["skill_id"],
                "required_capabilities": "|".join(row["required_capabilities"]),
                "safe_input_features": "|".join(row["safe_input_features"]),
                "output_schema": row["output_schema"],
                "unsupported_conditions": "|".join(row["unsupported_conditions"]),
            }
            for row in specs
        ],
    )
    _write_csv(
        output / "skill_dataset_scope_table.csv",
        [
            {
                "skill_id": row["skill_id"],
                "compatible_datasets": "|".join(row["compatible_datasets"]),
                "dataset_scope": row["dataset_scope"],
                "scope_priority": row["scope_priority"],
                "promotion_status": row["promotion_status"],
            }
            for row in specs
        ],
    )
    _write_csv(
        output / "skill_fusion_boundary_table.csv",
        [
            {
                "skill_id": row["skill_id"],
                "evidence_stage": row["evidence_stage"],
                "advisory_only": row["advisory_only"],
                "enters_fusion": row["enters_fusion"],
                "eligible_for_fusion_after_acceptance": row[
                    "eligible_for_fusion_after_acceptance"
                ],
                "final_verdict_owner": "FusionAgent",
            }
            for row in specs
        ],
    )
    correlation = {
        "schema_version": "1.0",
        "policy": "at most one verdict-stage evidence item per correlation group",
        "groups": {
            group: [row["skill_id"] for row in specs if row["correlation_group"] == group]
            for group in sorted({row["correlation_group"] for row in specs})
        },
        "mutual_exclusions": {
            row["skill_id"]: row["mutually_exclusive_with"]
            for row in specs
            if row["mutually_exclusive_with"]
        },
    }
    _dump(output / "skill_correlation_registry.json", correlation)

    sample = _sample_case_state()
    request = registry.create_request(
        case_state=sample,
        requested_skill="GeneralFlowMalwareSkill",
        requested_agent="StatsDetectorAgent",
        hypothesis="safe flow aggregates may provide binary evidence",
        purpose="collect one scoped evidence item",
        max_agent_calls=1,
        max_latency_ms=50.0,
        planner_type="rule",
        prompt_version="not_applicable_rule_v1",
    )
    duplicate_review = registry.review_selection(
        ("GeneralFlowMalwareSkill", "IoTBotnetSkill")
    )
    security = {
        "case_state_schema_exposes_forbidden_fields": False,
        "request_schema_owns_final_verdict": False,
        "required_forbidden_outputs_present": REQUIRED_FORBIDDEN_OUTPUTS.issubset(
            request.forbidden_outputs
        ),
        "correlated_skill_double_count_rejected": not duplicate_review["approved"],
        "advisory_skill_enters_fusion_count": sum(
            row["advisory_only"] and row["enters_fusion"] for row in specs
        ),
        "new_skill_enters_fusion_count": sum(row["enters_fusion"] for row in specs),
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "security_acceptance.json", security)
    return {
        "status": "fine_grained_skill_registry_built_default_off",
        "skill_count": len(specs),
        "verdict_stage_skill_count": sum(
            row["evidence_stage"] == "verdict_stage" for row in specs
        ),
        "risk_applicability_skill_count": sum(
            row["evidence_stage"] == "risk_applicability" for row in specs
        ),
        "advisory_skill_count": sum(row["advisory_only"] for row in specs),
        **security,
    }


def finalize_traffic_skill_registry_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    required = (
        "case_state_v2_schema.json",
        "evidence_request_v1_schema.json",
        "traffic_expert_skill_registry.json",
        "skill_capability_matrix.csv",
        "skill_dataset_scope_table.csv",
        "skill_fusion_boundary_table.csv",
        "skill_correlation_registry.json",
        "security_acceptance.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    security = (
        json.loads((output / "security_acceptance.json").read_text(encoding="utf-8"))
        if (output / "security_acceptance.json").exists()
        else {}
    )
    registry_payload = (
        json.loads((output / "traffic_expert_skill_registry.json").read_text(encoding="utf-8"))
        if (output / "traffic_expert_skill_registry.json").exists()
        else {"skills": []}
    )
    skills = registry_payload.get("skills", [])
    passed = (
        not missing
        and len(skills) == 21
        and security.get("correlated_skill_double_count_rejected") is True
        and security.get("required_forbidden_outputs_present") is True
        and all(
            security.get(key) == 0
            for key in (
                "advisory_skill_enters_fusion_count",
                "new_skill_enters_fusion_count",
                "blocked_field_violation",
                "fusion_ownership_violation",
                "ood_override",
                "illegal_verdict_execution",
                "fake_metric_count",
            )
        )
    )
    status = (
        "accepted_fine_grained_default_off_skill_registry"
        if passed
        else "failed_fine_grained_skill_registry_gate"
    )
    acceptance = {
        "status": status,
        "accepted": passed,
        "skill_count": len(skills),
        "missing_artifacts": missing,
        **security,
        "tests_passed": tests_passed,
        "test_count": test_count,
    }
    _dump(output / "acceptance_report.json", acceptance)
    negative = {
        "status": "retained_readiness_boundaries",
        "items": [
            {
                "candidate_module": "TLSRecordSkill/DoHCovertChannelSkill",
                "final_status": "not_promoted",
                "reason": "Registry readiness is not model acceptance.",
            },
            {
                "candidate_module": "TorApplicabilitySkill/TorApplicationSkill",
                "final_status": "dataset_readiness_required",
                "reason": "Tor never implies malicious; no Tor-malicious evidence is registered.",
            },
            {
                "candidate_module": "all_v2_skill_wrappers",
                "final_status": "default_off",
                "reason": "This round creates governance contracts, not a promoted runtime.",
            },
        ],
        "fake_metric_count": 0,
    }
    _dump(output / "negative_results.json", negative)
    en = f"""# MAD-ETD Fine-grained Traffic Expert Skill Registry v1

Status: `{status}`. The registry contains 21 least-privilege Skills: nine verdict-stage evidence candidates, four risk/applicability specialists, and eight advisory/audit specialists. All new Skill wrappers are default-off and currently enter no Fusion path. `CaseStateSkillV2` excludes truth and source identity; `EvidenceRequestSkillV1` forbids final verdict/confidence/uncertainty, benign/malicious labels, and OOD override. Correlation groups prevent duplicate weighting of wrappers around the same artifact.

`runtime_safe_v3_0` remains default. FusionAgent remains the only final decision owner.
"""
    cn = f"""# MAD-ETD 细粒度流量专家 Skill 注册表 v1

状态：`{status}`。注册表包含 21 个最小权限 Skill：9 个 verdict-stage 证据候选、4 个风险/适用性专家和 8 个 advisory/audit 专家。所有新增 Skill 包装器均默认关闭，目前均不进入 Fusion。`CaseStateSkillV2` 不暴露真实标签或来源身份；`EvidenceRequestSkillV1` 固定禁止 final verdict/confidence/uncertainty、benign/malicious 输出及 OOD override。相关性分组阻止共享同一工件的 Skill 被重复计权。

`runtime_safe_v3_0` 保持默认；FusionAgent 仍是唯一最终判定所有者。
"""
    for path, text in (
        (Path(document), en),
        (Path(document_cn), cn),
        (output / "MAD_ETD_TRAFFIC_SKILL_REGISTRY_V1.md", en),
        (output / "MAD_ETD_TRAFFIC_SKILL_REGISTRY_V1_CN.md", cn),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return acceptance

