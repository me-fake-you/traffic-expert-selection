from __future__ import annotations

from .audit import AuditLogger
from .base import CaseState
from .knowledge import KnowledgeRetrievalService, fusion_sha256
from .schemas import (
    AttributionResult,
    DetectionReport,
    FusionResult,
    KnowledgeAgentSummary,
    KnowledgeFieldSummary,
    KnowledgeQueryContext,
    KnowledgeSupportResult,
    Verdict,
)


class ReporterAgent:
    name = "ReporterAgent"
    version = "1.1"

    def __init__(
        self,
        knowledge_service: KnowledgeRetrievalService | None = None,
    ) -> None:
        self.knowledge_service = knowledge_service

    def build(
        self,
        state: CaseState,
        fusion: FusionResult,
        audit_event_count: int,
        knowledge_support: KnowledgeSupportResult | None = None,
    ) -> DetectionReport:
        family = "unknown"
        family_confidence = 0.0
        intent = "unknown"
        intent_confidence = 0.0
        for item in [*state.enrichment_results, *state.evidence]:
            if item.family_scores:
                candidate, score = max(
                    item.family_scores.items(), key=lambda pair: pair[1]
                )
                if score >= 0.5:
                    family, family_confidence = candidate, score
            if item.intent != "unknown" and item.confidence >= 0.5:
                intent, intent_confidence = item.intent, item.confidence

        reliability = state.reliability
        downgraded = {}
        risks: list[str] = []
        if reliability:
            values = {
                "stats": reliability.stats_reliability,
                "sequence": reliability.sequence_reliability,
                "tls": reliability.tls_reliability,
                "payload": reliability.payload_reliability,
            }
            downgraded = {
                key: value for key, value in values.items() if value < 0.75
            }
            risks.extend(reliability.indicators)
        if state.field_audit:
            risks.extend(state.field_audit.warnings)

        main_evidence = []
        for item in state.evidence:
            main_evidence.extend(
                f"{item.agent_name}: {text}" for text in item.evidence[:3]
            )

        return DetectionReport(
            trace_id=state.flow.trace_id,
            sample_id=state.flow.sample_id,
            verdict=fusion.verdict,
            confidence=fusion.confidence,
            uncertainty=fusion.uncertainty,
            conflict_score=fusion.conflict_score,
            benign_support=fusion.benign_support,
            malicious_support=fusion.malicious_support,
            distribution_shift_score=fusion.distribution_shift_score,
            severity=fusion.severity,
            need_escalation=fusion.need_escalation,
            intent=intent,
            intent_confidence=intent_confidence,
            family=family,
            family_confidence=family_confidence,
            participating_agents=[item.agent_name for item in state.evidence],
            agent_results=state.evidence,
            enrichment_results=state.enrichment_results,
            main_evidence=main_evidence + fusion.reasons,
            blocked_fields=state.field_audit.blocked_fields
            if state.field_audit
            else [],
            downgraded_feature_groups=downgraded,
            leakage_or_pseudo_feature_risks=risks,
            coordinator_decisions=state.decisions,
            recommended_actions=self._actions(fusion.verdict),
            knowledge_support=knowledge_support,
            audit_event_count=audit_event_count,
        )

    def retrieve_knowledge(
        self,
        state: CaseState,
        fusion: FusionResult,
        audit: AuditLogger | None = None,
    ) -> KnowledgeSupportResult | None:
        if self.knowledge_service is None:
            return None
        before = fusion_sha256(fusion)
        context = KnowledgeQueryContext(
            trace_id=state.flow.trace_id,
            fusion_snapshot=fusion.model_copy(deep=True),
            fusion_sha256=before,
            agents=[
                KnowledgeAgentSummary(
                    agent_name=item.agent_name,
                    feature_group=item.feature_group,
                    abstained=item.abstained,
                    confidence=item.confidence,
                    uncertainty=item.uncertainty,
                    model_reliability=item.model_reliability,
                    distribution_shift_level=item.distribution_shift_level,
                    evidence=list(item.evidence),
                )
                for item in state.evidence
            ],
            reliability=(
                state.reliability.model_copy(deep=True)
                if state.reliability
                else None
            ),
            field_audit=[
                KnowledgeFieldSummary(
                    path=item.path,
                    role=item.role,
                    reasons=list(item.reasons),
                )
                for item in (
                    state.field_audit.decisions if state.field_audit else []
                )
            ],
            leakage_risk=(
                state.field_audit.leakage_risk if state.field_audit else 0
            ),
            coordinator_reason_codes=sorted(
                {
                    code
                    for decision in state.decisions
                    for code in decision.reason_codes
                }
            ),
            recommended_actions=self._actions(fusion.verdict),
            query_topics=self._query_topics(state, fusion),
        )
        result = self.knowledge_service.retrieve(context, audit)
        if fusion_sha256(fusion) != before:
            raise RuntimeError("Knowledge retrieval modified immutable FusionResult")
        return result

    @staticmethod
    def _query_topics(
        state: CaseState,
        fusion: FusionResult,
    ) -> list[str]:
        evidence = " ".join(
            text.lower()
            for item in state.evidence
            for text in item.evidence
        )
        topics: list[str] = []
        if any(
            token in evidence
            for token in (
                "periodic",
                "beacon",
                "stable direction",
                "iat stability",
                "regular timing",
            )
        ):
            topics.append("c2_beaconing")
        if any(
            token in evidence
            for token in (
                "outbound ratio",
                "high outbound",
                "upstream",
                "exfiltration",
                "unusual burst",
            )
        ):
            topics.append("possible_exfiltration")
        if any(
            item.feature_group.value == "tls"
            and (
                item.malicious_support > item.benign_support
                or item.abstained
                or item.uncertainty >= 0.5
            )
            for item in state.evidence
        ):
            topics.append("tls_anomaly")
        if fusion.verdict in {Verdict.SUSPICIOUS, Verdict.UNKNOWN}:
            topics.append("suspicious_unknown_policy")
        if fusion.distribution_shift_score >= 0.95 or any(
            item.distribution_shift_level in {"warning", "hard"}
            for item in state.evidence
        ):
            topics.append("ood_reliability_policy")
        topics.append("response_action_rationale")
        return list(dict.fromkeys(topics))

    @staticmethod
    def _actions(verdict: Verdict) -> list[str]:
        return {
            Verdict.BENIGN: ["allow traffic", "retain routine telemetry"],
            Verdict.SUSPICIOUS: [
                "increase monitoring",
                "perform host/time-window correlation",
                "request analyst review",
            ],
            Verdict.MALICIOUS: [
                "isolate candidate endpoint",
                "prepare blocking action",
                "preserve traffic for forensics",
            ],
            Verdict.UNKNOWN: [
                "retain sample",
                "collect additional protocol or correlation evidence",
                "escalate to analyst or stronger model",
            ],
        }[verdict]

    def to_markdown(self, report: DetectionReport) -> str:
        lines = [
            f"# Detection report: {report.trace_id}",
            "",
            f"- Schema version: `{report.schema_version}`",
            f"- Verdict: **{report.verdict.value}**",
            f"- Confidence: `{report.confidence:.3f}`",
            f"- Uncertainty: `{report.uncertainty:.3f}`",
            f"- Distribution shift: `{report.distribution_shift_score:.3f}`",
            f"- Severity: `{report.severity}`",
            f"- Family: `{report.family}` ({report.family_confidence:.3f})",
            f"- Intent: `{report.intent}` ({report.intent_confidence:.3f})",
            "",
            "## Participating agents",
            "",
        ]
        lines.extend(f"- {name}" for name in report.participating_agents)
        if report.enrichment_results:
            lines.extend(["", "## Post-fusion enrichment", ""])
            lines.extend(
                (
                    f"- {item.agent_name}: advisory-only, "
                    f"confidence={item.confidence:.3f}"
                )
                for item in report.enrichment_results
            )
        lines.extend(["", "## Evidence", ""])
        lines.extend(f"- {item}" for item in report.main_evidence)
        lines.extend(["", "## Field and reliability controls", ""])
        lines.append(
            "- Blocked fields: "
            + (", ".join(report.blocked_fields) if report.blocked_fields else "none")
        )
        lines.append(
            "- Downgraded groups: "
            + (
                ", ".join(
                    f"{key}={value:.2f}"
                    for key, value in report.downgraded_feature_groups.items()
                )
                if report.downgraded_feature_groups
                else "none"
            )
        )
        lines.extend(["", "## Recommended actions", ""])
        lines.extend(f"- {action}" for action in report.recommended_actions)
        if report.knowledge_support is not None:
            knowledge = report.knowledge_support
            reference_lookup = {
                item.reference_id: (
                    f"[KB:{item.document_id}#{item.chunk_id}]"
                )
                for item in knowledge.references
            }
            lines.extend(["", "## Knowledge Support", ""])
            lines.append(
                "> [explanatory_only] Retrieved knowledge is background "
                "context, not detection evidence, and cannot modify the verdict."
            )
            if knowledge.status != "success":
                lines.append(f"- Retrieval status: `{knowledge.status}`")
            self._append_explanations(
                lines,
                knowledge.knowledge_support,
                reference_lookup,
            )
            lines.extend(["", "## Possible Intent Explanation", ""])
            self._append_explanations(
                lines,
                knowledge.possible_intent_explanations,
                reference_lookup,
            )
            lines.extend(["", "## Recommended Response Rationale", ""])
            self._append_explanations(
                lines,
                knowledge.response_rationales,
                reference_lookup,
            )
            lines.extend(["", "## Policy Explanation", ""])
            self._append_explanations(
                lines,
                knowledge.policy_explanations,
                reference_lookup,
            )
            lines.extend(["", "## Retrieved References", ""])
            if knowledge.references:
                lines.extend(
                    (
                        f"- `[KB:{reference.document_id}#"
                        f"{reference.chunk_id}]` {reference.heading} "
                        f"(score={reference.retrieval_score:.3f}, "
                        f"sha256={reference.content_sha256[:12]})"
                    )
                    for reference in knowledge.references
                )
            else:
                lines.append("- none")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _append_explanations(lines, explanations, reference_lookup) -> None:
        if not explanations:
            lines.append("- none")
            return
        for explanation in explanations:
            citations = " ".join(
                f"`{reference_lookup.get(reference_id, reference_id)}`"
                for reference_id in explanation.reference_ids
            )
            lines.append(
                f"- [explanatory_only] {explanation.statement} {citations}"
            )
