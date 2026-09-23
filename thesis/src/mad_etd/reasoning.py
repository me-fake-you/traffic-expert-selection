from __future__ import annotations

import hashlib

from .audit import AuditLogger
from .base import CaseState
from .schemas import (
    AuditFinding,
    ComplianceReport,
    DeliberationReport,
    FusionResult,
    ReflectionReport,
)


class ConflictDeliberationService:
    name = "ConflictDeliberation"
    version = "1.0"

    def analyze(
        self,
        state: CaseState,
        fusion: FusionResult,
        *,
        available_agents: set[str],
        audit: AuditLogger | None = None,
    ) -> DeliberationReport:
        conflict_sources: list[str] = []
        usable = [item for item in state.evidence if not item.abstained]
        for index, left in enumerate(usable):
            for right in usable[index + 1 :]:
                if abs(left.malicious_support - right.malicious_support) >= 0.5:
                    conflict_sources.append(
                        f"{left.agent_name} disagrees with {right.agent_name}"
                    )
        triggered = bool(conflict_sources) or fusion.conflict_score >= 0.3
        recommended: list[str] = []
        missing: list[str] = []
        flow = state.safe_flow or state.flow
        tls_available = (
            state.view_availability.tls
            if state.view_availability is not None
            else bool(flow.tls)
        )
        if triggered:
            if tls_available and "TLSProtocolAgent" not in state.called_agents:
                recommended.append("TLSProtocolAgent")
            elif not tls_available:
                missing.append("TLS protocol metadata")
            if (
                fusion.malicious_support >= 0.6
                and "FamilyAttributionAgent" not in state.called_agents
            ):
                recommended.append("FamilyAttributionAgent")
        recommended = [
            item for item in recommended if item in available_agents
        ]
        report = DeliberationReport(
            status="completed" if triggered else "not_triggered",
            conflict_sources=conflict_sources,
            reliability_context=list(
                state.reliability.indicators if state.reliability else []
            ),
            missing_evidence=missing,
            recommended_dispatch=recommended,
            limitations=[
                "Deliberation is meta-analysis and does not create AgentEvidence.",
                "Every recommended dispatch must pass PolicyGuard.",
            ],
        )
        if audit:
            audit.log(
                self.name,
                "DELIBERATION_COMPLETED"
                if triggered
                else "DELIBERATION_NOT_TRIGGERED",
                input_summary={
                    "fusion_conflict_score": fusion.conflict_score,
                    "evidence_agents": [item.agent_name for item in state.evidence],
                },
                output_summary=report.model_dump(mode="json"),
                reason="Conflict analysis produced guarded routing advice only.",
            )
        return report


class SelfReflectionService:
    name = "SelfReflection"
    version = "1.0"

    def reflect(
        self,
        state: CaseState,
        fusion: FusionResult,
        *,
        audit: AuditLogger | None = None,
    ) -> ReflectionReport:
        failure_modes: list[str] = []
        if fusion.verdict.value == "unknown":
            failure_modes.append("final verdict remained unknown")
        if fusion.uncertainty >= 0.5:
            failure_modes.append("high residual uncertainty")
        if fusion.conflict_score >= 0.3:
            failure_modes.append("material detector conflict")
        if fusion.distribution_shift_score >= 0.95:
            failure_modes.append("severe model-distribution shift")
        reliabilities = (
            {
                "stats": state.reliability.stats_reliability,
                "sequence": state.reliability.sequence_reliability,
                "tls": state.reliability.tls_reliability,
                "payload": state.reliability.payload_reliability,
            }
            if state.reliability
            else {}
        )
        least_reliable = (
            min(reliabilities, key=reliabilities.get) if reliabilities else None
        )
        report = ReflectionReport(
            status="completed" if failure_modes else "not_needed",
            failure_modes=failure_modes,
            least_reliable_view=least_reliable,
            missing_information=[
                indicator
                for indicator in (
                    state.reliability.indicators if state.reliability else []
                )
                if "missing" in indicator.lower()
            ],
            future_collection_hint=(
                [f"Prioritize higher-quality {least_reliable} evidence."]
                if failure_modes and least_reliable
                else []
            ),
        )
        if audit:
            audit.log(
                self.name,
                "REFLECTION_GENERATED",
                input_summary={"fusion_snapshot": fusion.model_dump(mode="json")},
                output_summary=report.model_dump(mode="json"),
                reason="Reflection is post-decision meta-analysis only.",
            )
        return report


class AuditCritic:
    name = "AuditCritic"
    version = "1.0"

    REQUIRED_EVENTS = {
        "FLOW_VALIDATED",
        "FIELD_POLICY_APPLIED",
        "RELIABILITY_ASSESSED",
        "FINAL_FUSION",
    }
    FORBIDDEN_DECISION_KEYS = {
        "verdict",
        "confidence",
        "uncertainty",
        "benign_support",
        "malicious_support",
        "family",
    }

    def inspect(
        self,
        trace_id: str,
        audit: AuditLogger,
    ) -> ComplianceReport:
        findings: list[AuditFinding] = []
        event_types = {event.event_type for event in audit.events}
        missing = sorted(self.REQUIRED_EVENTS - event_types)
        if missing:
            findings.append(
                self._finding(
                    trace_id,
                    "AUDIT_MISSING_REQUIRED_EVENTS",
                    "high",
                    [],
                    f"Missing required audit events: {missing}",
                    "Restore the mandatory execution and audit stages.",
                )
            )
        for event in audit.events:
            blocked = event.input_summary.get("blocked_field_intersection", [])
            if blocked:
                findings.append(
                    self._finding(
                        trace_id,
                        "BLOCKED_FIELD_VISIBLE",
                        "critical",
                        [event.sequence_no],
                        f"Blocked fields reached a detector: {blocked}",
                        "Reject the execution and repair FieldAudit enforcement.",
                    )
                )
            if event.actor in {
                "CaseMemory",
                "ConflictDeliberation",
                "SelfReflection",
                "KnowledgeRetrievalService",
            }:
                keys = self._nested_keys(event.output_summary)
                forbidden = sorted(keys & self.FORBIDDEN_DECISION_KEYS)
                if forbidden:
                    findings.append(
                        self._finding(
                            trace_id,
                            "ADVISORY_COMPONENT_DECISION_FIELD",
                            "critical",
                            [event.sequence_no],
                            f"Advisory component emitted decision fields: {forbidden}",
                            "Remove Fusion-owned fields from the advisory schema.",
                        )
                    )
        coverage = len(event_types & self.REQUIRED_EVENTS) / len(
            self.REQUIRED_EVENTS
        )
        return ComplianceReport(
            trace_id=trace_id,
            findings=findings,
            audit_coverage=coverage,
            status=(
                "non_compliant"
                if findings
                else "compliant"
                if coverage == 1
                else "incomplete"
            ),
        )

    @staticmethod
    def _nested_keys(value) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key
                for item in value.values()
                for key in AuditCritic._nested_keys(item)
            }
        if isinstance(value, list):
            return {
                key for item in value for key in AuditCritic._nested_keys(item)
            }
        return set()

    @staticmethod
    def _finding(
        trace_id: str,
        rule_id: str,
        severity: str,
        refs: list[int],
        description: str,
        remediation: str,
    ) -> AuditFinding:
        finding_id = hashlib.sha256(
            f"{trace_id}:{rule_id}:{refs}:{description}".encode("utf-8")
        ).hexdigest()[:24]
        return AuditFinding(
            finding_id=finding_id,
            rule_id=rule_id,
            severity=severity,
            subject_event_refs=refs,
            description=description,
            remediation=remediation,
        )
