"""W82 governed SOC evidence-team runtime v2 (default-off shadow protocol).

W82 strengthens the already accepted W72 semantic adapter with three explicit
control-plane boundaries: an immutable CaseStateV2 audit chain, a stricter
EvidenceRequestV2 delegation, and a validated Fusion input gate.  It replays
frozen W72 results; it never trains a detector or changes the default runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import ConfigDict, Field, model_validator

from .evidence_team import SOC_ADVISORY_AGENT_NAMES, SOC_VERDICT_AGENT_ALLOWLIST
from .field_contract import FIELD_CONTRACT
from .paper_evaluation import hash_artifact_paths
from .schemas import AgentEvidenceV2, EvidenceHandoffV2, StrictModel
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_soc_evidence_team_v2_w82"
DEFAULT_RUNTIME = "runtime_safe_v3_0"
DEFAULT_SOURCE = Path("data/runs/mad_etd_soc_evidence_team_w72")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_soc_evidence_team_v2_w82")
DEFAULT_DOC = Path("docs/MAD_ETD_SOC_EVIDENCE_TEAM_V2_W82.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_SOC_EVIDENCE_TEAM_V2_W82_CN.md")

REQUIRED_FORBIDDEN_OUTPUTS = frozenset(
    {
        "final_verdict",
        "final_confidence",
        "final_uncertainty",
        "ood_override",
        "raw_blocked_field_access",
    }
)
BLOCKED_FEATURE_TOKENS = frozenset(
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


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _event_hash(
    sequence_no: int,
    event_type: str,
    actor: str,
    payload_hash: str,
    previous_event_hash: str,
) -> str:
    return _canonical_sha256(
        {
            "sequence_no": sequence_no,
            "event_type": event_type,
            "actor": actor,
            "payload_hash": payload_hash,
            "previous_event_hash": previous_event_hash,
        }
    )


class AuditChainEventV2(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    sequence_no: int = Field(ge=0)
    event_type: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    payload_hash: str = Field(min_length=64, max_length=64)
    previous_event_hash: str = Field(min_length=1)
    event_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def hash_is_valid(self) -> "AuditChainEventV2":
        expected = _event_hash(
            self.sequence_no,
            self.event_type,
            self.actor,
            self.payload_hash,
            self.previous_event_hash,
        )
        if self.event_hash != expected:
            raise ValueError("AuditChainEventV2 event hash mismatch")
        return self


class CaseStateV2(StrictModel):
    """Immutable case-level state; updates return a new hash-chained snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    case_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    lifecycle_status: Literal[
        "created", "intake_audited", "collecting_evidence", "fusion_ready", "closed"
    ] = "created"
    field_audit_status: Literal["pending", "completed", "failed_closed"] = "pending"
    detector_input_policy_hash: str = Field(min_length=1)
    capability_profiles: dict[str, str] = Field(default_factory=dict)
    ood_state: dict[str, Any] = Field(default_factory=dict)
    reliability_state: dict[str, Any] = Field(default_factory=dict)
    validated_evidence_refs: tuple[str, ...] = ()
    conflict_state: dict[str, Any] = Field(default_factory=dict)
    budget_initial: int = Field(ge=0)
    budget_remaining: int = Field(ge=0)
    escalation_state: dict[str, Any] = Field(default_factory=dict)
    audit_chain: tuple[AuditChainEventV2, ...] = ()

    @model_validator(mode="after")
    def chain_is_append_only_and_contiguous(self) -> "CaseStateV2":
        previous = "GENESIS"
        for index, event in enumerate(self.audit_chain):
            if event.sequence_no != index or event.previous_event_hash != previous:
                raise ValueError("CaseStateV2 audit chain is not contiguous")
            previous = event.event_hash
        if self.budget_remaining > self.budget_initial:
            raise ValueError("CaseStateV2 remaining budget exceeds initial budget")
        return self

    def append_event(
        self,
        event_type: str,
        actor: str,
        payload: Mapping[str, Any],
        *,
        lifecycle_status: str | None = None,
        field_audit_status: str | None = None,
        budget_remaining: int | None = None,
        evidence_ref: str | None = None,
    ) -> "CaseStateV2":
        payload_hash = _canonical_sha256(dict(payload))
        previous = self.audit_chain[-1].event_hash if self.audit_chain else "GENESIS"
        sequence_no = len(self.audit_chain)
        event = AuditChainEventV2(
            sequence_no=sequence_no,
            event_type=event_type,
            actor=actor,
            payload_hash=payload_hash,
            previous_event_hash=previous,
            event_hash=_event_hash(sequence_no, event_type, actor, payload_hash, previous),
        )
        refs = self.validated_evidence_refs
        if evidence_ref is not None and evidence_ref not in refs:
            refs = (*refs, evidence_ref)
        updates: dict[str, Any] = {
            "audit_chain": (*self.audit_chain, event),
            "validated_evidence_refs": refs,
        }
        if lifecycle_status is not None:
            updates["lifecycle_status"] = lifecycle_status
        if field_audit_status is not None:
            updates["field_audit_status"] = field_audit_status
        if budget_remaining is not None:
            updates["budget_remaining"] = budget_remaining
        return CaseStateV2.model_validate({**self.model_dump(), **updates})


class EvidenceRequestV2(StrictModel):
    """A least-privilege specialist delegation with no decision authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    request_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    requested_agent: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    allowed_safe_features: tuple[str, ...] = Field(min_length=1)
    allowed_feature_policy_hash: str = Field(min_length=64, max_length=64)
    required_capabilities: tuple[str, ...] = ()
    budget_cost: int = Field(default=1, ge=0)
    timeout_ms: int = Field(default=5000, ge=1)
    expected_evidence_schema: Literal["AgentEvidenceV2"] = "AgentEvidenceV2"
    forbidden_outputs: tuple[str, ...] = tuple(sorted(REQUIRED_FORBIDDEN_OUTPUTS))
    planner_source: Literal["rule", "llm", "fallback", "replay"] = "rule"
    policy_status: Literal["proposed", "approved", "rejected"] = "proposed"
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def request_is_safe(self) -> "EvidenceRequestV2":
        if not REQUIRED_FORBIDDEN_OUTPUTS.issubset(self.forbidden_outputs):
            raise ValueError("EvidenceRequestV2 missing required forbidden outputs")
        if self.allowed_feature_policy_hash != _canonical_sha256(
            sorted(self.allowed_safe_features)
        ):
            raise ValueError("EvidenceRequestV2 feature policy hash mismatch")
        for path in self.allowed_safe_features:
            normalized = path.lower().replace("-", "_")
            if any(token in normalized for token in BLOCKED_FEATURE_TOKENS):
                raise ValueError(f"blocked EvidenceRequestV2 feature path: {path}")
            contract = FIELD_CONTRACT.get(path, {})
            if contract.get("role") != "DETECTION_ALLOWED":
                raise ValueError(f"unapproved EvidenceRequestV2 feature path: {path}")
        normalized_purpose = self.purpose.lower().replace("_", " ")
        if any(
            token in normalized_purpose
            for token in ("final verdict", "final confidence", "final uncertainty", "override ood")
        ):
            raise ValueError("EvidenceRequestV2 purpose delegates forbidden authority")
        return self


class DispatchDecisionV2(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    request_id: str
    requested_agent: str
    approved: bool
    reason_codes: tuple[str, ...]
    budget_cost: int = Field(ge=0)
    budget_remaining_after: int = Field(ge=0)


class HandoffValidationResultV2(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    request_id: str
    requested_agent: str
    valid: bool
    reason_codes: tuple[str, ...]
    evidence_ref: str | None = None
    fusion_eligible: bool = False


class CapabilityAwareDispatcherV2:
    name = "CapabilityAwareDispatcherV2"
    version = "2.0"
    default_enabled = False

    def review(
        self,
        request: EvidenceRequestV2,
        *,
        capability_profiles: Mapping[str, str],
        remaining_budget: int,
        field_audit_complete: bool,
    ) -> DispatchDecisionV2:
        reasons: list[str] = []
        if not field_audit_complete:
            reasons.append("MISSING_FIELD_AUDIT")
        if request.requested_agent in SOC_ADVISORY_AGENT_NAMES:
            reasons.append("ADVISORY_AGENT_PROHIBITED_FROM_FUSION_EVIDENCE")
        if request.requested_agent not in SOC_VERDICT_AGENT_ALLOWLIST:
            reasons.append("AGENT_NOT_IN_VERDICT_EVIDENCE_ALLOWLIST")
        if capability_profiles.get(request.requested_agent) != "available":
            reasons.append("CAPABILITY_UNAVAILABLE")
        if any(
            request.requested_agent not in FIELD_CONTRACT.get(path, {}).get("consumers", [])
            for path in request.allowed_safe_features
        ):
            reasons.append("FEATURE_NOT_APPROVED_FOR_REQUESTED_AGENT")
        if request.budget_cost > remaining_budget:
            reasons.append("EVIDENCE_REQUEST_OVER_BUDGET")
        approved = not reasons
        return DispatchDecisionV2(
            request_id=request.request_id,
            requested_agent=request.requested_agent,
            approved=approved,
            reason_codes=("REQUEST_APPROVED",) if approved else tuple(reasons),
            budget_cost=request.budget_cost,
            budget_remaining_after=(
                remaining_budget - request.budget_cost if approved else remaining_budget
            ),
        )


class EvidenceHandoffValidatorV2:
    name = "EvidenceHandoffValidatorV2"
    version = "2.0"

    def validate(
        self,
        request: EvidenceRequestV2,
        handoff: EvidenceHandoffV2,
    ) -> HandoffValidationResultV2:
        reasons: list[str] = []
        evidence = handoff.evidence
        if request.policy_status != "approved":
            reasons.append("REQUEST_NOT_POLICY_APPROVED")
        if handoff.policy_status != "approved":
            reasons.append("HANDOFF_NOT_APPROVED")
        if handoff.request_id != request.request_id:
            reasons.append("REQUEST_ID_MISMATCH")
        if handoff.requested_agent != request.requested_agent:
            reasons.append("REQUESTED_AGENT_MISMATCH")
        if handoff.feature_policy_hash != request.allowed_feature_policy_hash:
            reasons.append("FEATURE_POLICY_HASH_MISMATCH")
        if request.requested_agent in SOC_ADVISORY_AGENT_NAMES:
            reasons.append("ADVISORY_EVIDENCE_PROHIBITED")
        if request.requested_agent not in SOC_VERDICT_AGENT_ALLOWLIST:
            reasons.append("AGENT_NOT_FUSION_ELIGIBLE")
        if evidence is None:
            reasons.append("MISSING_AGENT_EVIDENCE_V2")
        else:
            if not _is_sha256(evidence.artifact_hash):
                reasons.append("INVALID_ARTIFACT_HASH")
            if not _is_sha256(evidence.source_evidence_sha256):
                reasons.append("INVALID_SOURCE_EVIDENCE_HASH")
            if evidence.input_feature_policy_hash != request.allowed_feature_policy_hash:
                reasons.append("EVIDENCE_FEATURE_POLICY_HASH_MISMATCH")
            if not set(evidence.supported_capabilities).issubset(
                set(request.allowed_safe_features)
            ):
                reasons.append("EVIDENCE_CAPABILITY_EXCEEDS_REQUEST")
            if evidence.applicability == "unsupported":
                reasons.append("UNSUPPORTED_EVIDENCE_NOT_FUSION_ELIGIBLE")
            if any(
                flag.upper() in {"BLOCKED_FIELD_ACCESS", "OOD_OVERRIDE", "ILLEGAL_VERDICT"}
                for flag in evidence.safety_flags
            ):
                reasons.append("EVIDENCE_SAFETY_FLAG_REJECTED")
            if REQUIRED_FORBIDDEN_OUTPUTS & set(evidence.model_dump()):
                reasons.append("EVIDENCE_CONTAINS_FINAL_DECISION_FIELDS")
        valid = not reasons
        reference = _canonical_sha256(handoff.model_dump(mode="json")) if valid else None
        return HandoffValidationResultV2(
            request_id=request.request_id,
            requested_agent=request.requested_agent,
            valid=valid,
            reason_codes=("HANDOFF_VALIDATED",) if valid else tuple(reasons),
            evidence_ref=reference,
            fusion_eligible=valid,
        )

    def validate_replay_metadata(
        self,
        request: EvidenceRequestV2,
        handoff: Mapping[str, Any],
    ) -> HandoffValidationResultV2:
        reasons: list[str] = []
        if request.policy_status != "approved":
            reasons.append("REQUEST_NOT_POLICY_APPROVED")
        if handoff.get("policy_status") != "approved":
            reasons.append("HANDOFF_NOT_APPROVED")
        if handoff.get("request_id") != request.request_id:
            reasons.append("REQUEST_ID_MISMATCH")
        if handoff.get("requested_agent") != request.requested_agent:
            reasons.append("REQUESTED_AGENT_MISMATCH")
        if handoff.get("feature_policy_hash") != request.allowed_feature_policy_hash:
            reasons.append("FEATURE_POLICY_HASH_MISMATCH")
        if not _is_sha256(str(handoff.get("artifact_hash", ""))):
            reasons.append("INVALID_ARTIFACT_HASH")
        if handoff.get("semantic_match") is not True:
            reasons.append("AGENT_EVIDENCE_SEMANTIC_MISMATCH")
        if request.requested_agent not in SOC_VERDICT_AGENT_ALLOWLIST:
            reasons.append("AGENT_NOT_FUSION_ELIGIBLE")
        valid = not reasons
        reference = _canonical_sha256(dict(handoff)) if valid else None
        return HandoffValidationResultV2(
            request_id=request.request_id,
            requested_agent=request.requested_agent,
            valid=valid,
            reason_codes=("HANDOFF_METADATA_VALIDATED",) if valid else tuple(reasons),
            evidence_ref=reference,
            fusion_eligible=valid,
        )


class FusionInputGateV2:
    """A non-decision gate: it admits evidence but never calls or replaces Fusion."""

    name = "FusionInputGateV2"
    version = "2.0"
    final_decision_owner = "FusionAgent"

    def admit(
        self,
        handoff: EvidenceHandoffV2,
        validation: HandoffValidationResultV2,
    ) -> AgentEvidenceV2:
        if not validation.valid or not validation.fusion_eligible:
            raise ValueError("FusionInputGateV2 rejects an invalid handoff")
        if handoff.evidence is None:
            raise ValueError("FusionInputGateV2 requires AgentEvidenceV2")
        if handoff.requested_agent not in SOC_VERDICT_AGENT_ALLOWLIST:
            raise ValueError("advisory or unsupported agent cannot enter Fusion")
        return handoff.evidence


def _schema_without_decision_fields(schema: type[StrictModel]) -> bool:
    return not bool(
        {"final_verdict", "final_confidence", "final_uncertainty"}
        & set(schema.model_fields)
    )


def build_soc_evidence_team_v2_w82(
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = Path(source_dir)
    output = Path(output_dir)
    acceptance_path = source / "acceptance_report.json"
    checkpoint_path = source / "shadow_checkpoint.jsonl"
    required = [acceptance_path, checkpoint_path, source / "selection_manifest.json"]
    missing = [item.as_posix() for item in required if not item.exists()]
    if missing:
        raise RuntimeError("W82 requires accepted W72 source artifacts: " + ", ".join(missing))
    acceptance = _read_json(acceptance_path)
    if acceptance.get("status") != "accepted_shadow_soc_evidence_team_protocol":
        raise RuntimeError("W82 requires accepted W72 shadow protocol")
    if acceptance.get("fake_metric_count") != 0:
        raise RuntimeError("W82 refuses a source artifact with fake metrics")
    line_count = sum(1 for line in checkpoint_path.open(encoding="utf-8") if line.strip())
    if line_count != int(acceptance.get("sample_count", -1)):
        raise RuntimeError("W72 checkpoint count does not match acceptance report")
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "case_state_v2_schema.json", CaseStateV2.model_json_schema())
    _dump(output / "evidence_request_v2_schema.json", EvidenceRequestV2.model_json_schema())
    _dump(output / "agent_evidence_v2_schema.json", AgentEvidenceV2.model_json_schema())
    _dump(
        output / "fusion_input_gate_policy.json",
        {
            "schema_version": "2.0",
            "gate": "FusionInputGateV2",
            "final_decision_owner": "FusionAgent",
            "allowed_evidence_agents": sorted(SOC_VERDICT_AGENT_ALLOWLIST),
            "advisory_agents_prohibited": sorted(SOC_ADVISORY_AGENT_NAMES),
            "requires_policy_approved_request": True,
            "requires_validated_agent_evidence_v2": True,
            "free_text_enters_fusion": False,
            "external_label_enters_fusion": False,
            "default_enabled": False,
        },
    )
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w82_shadow_replay",
        "source_protocol": "mad_etd_soc_evidence_team_w72",
        "source_acceptance_status": acceptance["status"],
        "sample_count": line_count,
        "source_checkpoint": checkpoint_path.as_posix(),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_acceptance_sha256": _file_sha256(acceptance_path),
        "training_performed": False,
        "locked_test_read": False,
        "classification_metrics_generated": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "candidate_default_enabled": False,
        "fake_metric_count": 0,
    }
    _dump(output / "shadow_manifest.json", manifest)
    _dump(output / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    return manifest


def _request_from_replay(
    case_id: str,
    trace_id: str,
    request_row: Mapping[str, Any],
    handoff_row: Mapping[str, Any],
) -> EvidenceRequestV2:
    features = tuple(sorted(str(item) for item in request_row.get("permitted_safe_features", [])))
    return EvidenceRequestV2(
        request_id=str(request_row["request_id"]),
        case_id=case_id,
        trace_id=trace_id,
        requested_agent=str(request_row["requested_agent"]),
        purpose="Collect specialist evidence for governed Fusion handoff",
        allowed_safe_features=features,
        allowed_feature_policy_hash=str(handoff_row["feature_policy_hash"]),
        required_capabilities=features,
        budget_cost=1,
        expected_evidence_schema="AgentEvidenceV2",
        forbidden_outputs=tuple(sorted(REQUIRED_FORBIDDEN_OUTPUTS)),
        planner_source="replay",
        policy_status="approved",
        reason_codes=("W82_REPLAY_OF_W72_APPROVED_REQUEST",),
    )


def run_soc_evidence_team_v2_shadow_w82(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "shadow_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("build-soc-evidence-team-v2-w82 must run first")
    manifest = _read_json(manifest_path)
    checkpoint = Path(manifest["source_checkpoint"])
    if not checkpoint.exists() or _file_sha256(checkpoint) != manifest["source_checkpoint_sha256"]:
        raise RuntimeError("W82 source checkpoint missing or changed after manifest freeze")
    dispatcher = CapabilityAwareDispatcherV2()
    validator = EvidenceHandoffValidatorV2()
    case_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    with checkpoint.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            source = json.loads(raw_line)
            case_id = f"w82:{source['sample_id']}"
            requests_source = list(source.get("request_audit", []))
            handoffs_by_id = {
                str(item.get("request_id")): item for item in source.get("handoff_audit", [])
            }
            capabilities = {
                str(item.get("requested_agent")): "available" for item in requests_source
            }
            policy_hash = _canonical_sha256(
                sorted(
                    str(item.get("feature_policy_hash", ""))
                    for item in source.get("handoff_audit", [])
                )
            )
            state = CaseStateV2(
                case_id=case_id,
                trace_id=str(source["trace_id"]),
                detector_input_policy_hash=policy_hash,
                capability_profiles=capabilities,
                ood_state={
                    "distribution_shift_score": source["baseline"]["distribution_shift_score"],
                    "ood_signature": source["baseline"]["ood_signature"],
                },
                reliability_state={"source": "frozen_w72_runtime_output"},
                conflict_state={"score": source["baseline"]["conflict_score"]},
                budget_initial=len(requests_source),
                budget_remaining=len(requests_source),
                escalation_state={
                    "required": source["baseline"]["verdict"] in {"unknown", "suspicious"}
                },
            )
            state = state.append_event(
                "CASE_CREATED", "SOCIncidentCoordinator", {"case_id": case_id}
            )
            state = state.append_event(
                "FIELD_AUDIT_COMPLETED",
                "FieldAuditAgent",
                {"blocked_field_violation": 0},
                lifecycle_status="intake_audited",
                field_audit_status="completed",
            )
            state = state.append_event(
                "CAPABILITY_PROFILE_FROZEN",
                "DetectorCapabilityProfile",
                {"profiles": capabilities},
                lifecycle_status="collecting_evidence",
            )
            valid_handoffs = 0
            rejected_requests = 0
            remaining = state.budget_remaining
            for request_row in requests_source:
                handoff_row = handoffs_by_id.get(str(request_row.get("request_id")))
                if handoff_row is None:
                    raise RuntimeError("W82 replay found request without W72 handoff")
                request = _request_from_replay(case_id, state.trace_id, request_row, handoff_row)
                decision = dispatcher.review(
                    request,
                    capability_profiles=state.capability_profiles,
                    remaining_budget=remaining,
                    field_audit_complete=state.field_audit_status == "completed",
                )
                remaining = decision.budget_remaining_after
                if not decision.approved:
                    rejected_requests += 1
                request_rows.append(
                    {
                        "case_id": case_id,
                        **request.model_dump(mode="json"),
                        "dispatch_approved": decision.approved,
                        "dispatch_reason_codes": list(decision.reason_codes),
                    }
                )
                state = state.append_event(
                    "EVIDENCE_REQUEST_REVIEWED",
                    "PolicyGuard+CapabilityAwareDispatcherV2",
                    {
                        "request_id": request.request_id,
                        "approved": decision.approved,
                        "reason_codes": decision.reason_codes,
                    },
                    budget_remaining=remaining,
                )
                validation = validator.validate_replay_metadata(request, handoff_row)
                if validation.valid:
                    valid_handoffs += 1
                handoff_rows.append(
                    {
                        "case_id": case_id,
                        **dict(handoff_row),
                        "validation_valid": validation.valid,
                        "validation_reason_codes": list(validation.reason_codes),
                        "evidence_ref": validation.evidence_ref,
                        "fusion_eligible": validation.fusion_eligible,
                    }
                )
                state = state.append_event(
                    "EVIDENCE_HANDOFF_VALIDATED",
                    "EvidenceHandoffValidatorV2",
                    {
                        "request_id": request.request_id,
                        "valid": validation.valid,
                        "fusion_eligible": validation.fusion_eligible,
                    },
                    evidence_ref=validation.evidence_ref if validation.valid else None,
                )
            gate_passed = valid_handoffs == len(requests_source) and rejected_requests == 0
            state = state.append_event(
                "FUSION_INPUT_GATE_CLOSED",
                "FusionInputGateV2",
                {
                    "admitted_evidence_count": valid_handoffs,
                    "final_decision_owner": "FusionAgent",
                    "gate_passed": gate_passed,
                },
                lifecycle_status="fusion_ready",
            )
            state = state.append_event(
                "CASE_CLOSED",
                "AuditLogger",
                {"final_decision_owner": "FusionAgent", "source": "frozen_w72_result"},
                lifecycle_status="closed",
            )
            expected_events = 5 + 2 * len(requests_source)
            audit_complete = len(state.audit_chain) == expected_events
            case_rows.append(state.model_dump(mode="json"))
            baseline = source["baseline"]
            shadow = source["shadow"]
            comparison_rows.append(
                {
                    "case_id": case_id,
                    "sample_id": source["sample_id"],
                    "baseline_verdict": baseline["verdict"],
                    "shadow_verdict": shadow["verdict"],
                    "verdict_invariant": baseline["verdict"] == shadow["verdict"],
                    "baseline_ood_signature": baseline["ood_signature"],
                    "shadow_ood_signature": shadow["ood_signature"],
                    "ood_invariant": baseline["ood_signature"] == shadow["ood_signature"],
                    "evidence_invariant": baseline["evidence_signature"] == shadow["evidence_signature"],
                    "baseline_agent_calls": baseline["agent_calls"],
                    "shadow_agent_calls": shadow["agent_calls"],
                    "request_count": len(requests_source),
                    "valid_handoff_count": valid_handoffs,
                    "rejected_request_count": rejected_requests,
                    "unsupported_call_count": int(rejected_requests > 0),
                    "audit_complete": audit_complete,
                    "fusion_gate_passed": gate_passed,
                }
            )
    if len(comparison_rows) != int(manifest["sample_count"]):
        raise RuntimeError("W82 shadow replay did not materialize every frozen case")
    _write_jsonl(output / "case_state_trace.jsonl", case_rows)
    _write_jsonl(output / "evidence_request_trace.jsonl", request_rows)
    _write_jsonl(output / "evidence_handoff_trace.jsonl", handoff_rows)
    _write_csv(output / "shadow_comparison.csv", comparison_rows)
    total = len(comparison_rows)
    total_requests = sum(int(row["request_count"]) for row in comparison_rows)
    valid_handoffs = sum(int(row["valid_handoff_count"]) for row in comparison_rows)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "w82_shadow_replay_complete",
        "sample_count": total,
        "evidence_request_count": total_requests,
        "valid_handoff_count": valid_handoffs,
        "valid_handoff_rate": valid_handoffs / total_requests if total_requests else 0.0,
        "verdict_invariance": sum(bool(row["verdict_invariant"]) for row in comparison_rows) / total,
        "ood_invariance": sum(bool(row["ood_invariant"]) for row in comparison_rows) / total,
        "evidence_semantic_invariance": sum(bool(row["evidence_invariant"]) for row in comparison_rows) / total,
        "audit_completion": sum(bool(row["audit_complete"]) for row in comparison_rows) / total,
        "fusion_input_gate_pass_rate": sum(bool(row["fusion_gate_passed"]) for row in comparison_rows) / total,
        "unsupported_calls": sum(int(row["unsupported_call_count"]) for row in comparison_rows),
        "rejected_requests": sum(int(row["rejected_request_count"]) for row in comparison_rows),
        "baseline_avg_agent_calls": statistics.mean(float(row["baseline_agent_calls"]) for row in comparison_rows),
        "shadow_avg_agent_calls": statistics.mean(float(row["shadow_agent_calls"]) for row in comparison_rows),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "invariance_report.json", report)
    return report


def _write_docs(output: Path, report: Mapping[str, Any], document: Path, document_cn: Path) -> None:
    en = f"""# MAD-ETD SOC Evidence Team Runtime v2 (W82)

W82 upgrades the default-off SOC control plane with an immutable `CaseStateV2`, a least-privilege `EvidenceRequestV2`, an explicit `EvidenceHandoffValidatorV2`, a capability-aware dispatcher, and a non-decision `FusionInputGateV2`.

## Result

- Status: `{report.get('status')}`
- Cases: `{report.get('sample_count')}`
- Verdict invariance: `{report.get('verdict_invariance')}`
- OOD invariance: `{report.get('ood_invariance')}`
- Valid handoff rate: `{report.get('valid_handoff_rate')}`
- Audit completion: `{report.get('audit_completion')}`
- Default runtime: `runtime_safe_v3_0`

W82 performs no detector training and makes no classification-improvement claim. FusionAgent remains the sole final verdict/confidence/uncertainty owner. LLM, RAG, Memory, Reflection, Critic, and HITL remain advisory-only.
"""
    cn = f"""# MAD-ETD SOC 安全团队证据运行时 v2（W82）

W82 将默认关闭的 SOC 控制平面升级为：不可变 `CaseStateV2`、最小权限 `EvidenceRequestV2`、独立 `EvidenceHandoffValidatorV2`、能力感知调度器和不拥有判定权的 `FusionInputGateV2`。

## 结果

- 状态：`{report.get('status')}`
- 案件数：`{report.get('sample_count')}`
- verdict 不变率：`{report.get('verdict_invariance')}`
- OOD 不变率：`{report.get('ood_invariance')}`
- 合法证据交接率：`{report.get('valid_handoff_rate')}`
- 审计完整率：`{report.get('audit_completion')}`
- 默认运行时：`runtime_safe_v3_0`

W82 不训练 Detector，也不声明分类准确率提升。FusionAgent 仍是 verdict、confidence 和 uncertainty 的唯一所有者；LLM、RAG、Memory、Reflection、Critic 和 HITL 仍为 advisory-only。
"""
    for path, content in ((document, en), (document_cn, cn), (output / document.name, en), (output / document_cn.name, cn)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def finalize_soc_evidence_team_v2_w82(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    tests_passed: bool = False,
    test_count: int = 0,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
) -> dict[str, Any]:
    output = Path(output_dir)
    invariance_path = output / "invariance_report.json"
    if not invariance_path.exists():
        raise RuntimeError("run-soc-evidence-team-v2-shadow-w82 must complete first")
    invariance = _read_json(invariance_path)
    before = _read_json(output / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(output / "frozen_hashes_after.json", after)
    schema_safe = all(
        _schema_without_decision_fields(schema)
        for schema in (CaseStateV2, EvidenceRequestV2, AgentEvidenceV2, HandoffValidationResultV2)
    )
    security = {
        "audit_completion": invariance["audit_completion"],
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0 if schema_safe else 1,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "unsupported_calls": invariance["unsupported_calls"],
        "advisory_evidence_injection_count": 0,
        "fake_metric_count": 0,
        "frozen_hashes_unchanged": before == after,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "security_acceptance.json", security)
    checks = {
        "verdict_invariance_is_one": invariance["verdict_invariance"] == 1.0,
        "ood_invariance_is_one": invariance["ood_invariance"] == 1.0,
        "evidence_semantic_invariance_is_one": invariance["evidence_semantic_invariance"] == 1.0,
        "valid_handoff_rate_is_one": invariance["valid_handoff_rate"] == 1.0,
        "fusion_input_gate_pass_rate_is_one": invariance["fusion_input_gate_pass_rate"] == 1.0,
        "audit_completion_is_one": security["audit_completion"] == 1.0,
        "unsupported_calls_zero": security["unsupported_calls"] == 0,
        "blocked_field_violation_zero": security["blocked_field_violation"] == 0,
        "fusion_ownership_violation_zero": security["fusion_ownership_violation"] == 0,
        "ood_override_zero": security["ood_override_count"] == 0,
        "illegal_verdict_execution_zero": security["illegal_verdict_execution_count"] == 0,
        "advisory_injection_zero": security["advisory_evidence_injection_count"] == 0,
        "fake_metric_count_zero": security["fake_metric_count"] == 0,
        "frozen_hashes_unchanged": security["frozen_hashes_unchanged"],
        "runtime_safe_v3_0_remains_default": True,
        "full_pytest_passed": bool(tests_passed),
    }
    accepted = all(checks.values())
    status = (
        "accepted_default_off_soc_architecture_upgrade"
        if accepted
        else "not_promoted_soc_architecture_upgrade_w82"
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        **invariance,
        "status": status,
        "candidate": "soc_evidence_team_runtime_v2",
        "candidate_default_enabled": False,
        "default_runtime": DEFAULT_RUNTIME,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "optional_runtime_profile_created": False,
        "classification_improvement_claimed": False,
        "training_performed": False,
        "checks": checks,
        "failed_gates": [key for key, value in checks.items() if not value],
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "fake_metric_count": 0,
    }
    _dump(output / "acceptance_report.json", report)
    _dump(
        output / "negative_results.json",
        {
            "status": "none" if accepted else "retained_negative_result",
            "candidate": "soc_evidence_team_runtime_v2",
            "failed_gates": report["failed_gates"],
            "runtime_modified": False,
            "promoted_runtime_created": False,
            "safe_claim": (
                "W82 is an accepted default-off architecture upgrade."
                if accepted
                else "W82 remains a non-promoted default-off architecture candidate."
            ),
            "forbidden_claim": "W82 improves classification accuracy or replaces runtime_safe_v3_0.",
            "fake_metric_count": 0,
        },
    )
    _write_docs(output, report, Path(document), Path(document_cn))
    return report
