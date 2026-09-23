from __future__ import annotations

from .base import CaseState, Coordinator
from .schemas import AgentEvidence, CoordinatorAction, CoordinatorDecision


class EvidenceBudgetRouterV11(Coordinator):
    """OOD-aware routing-only efficiency candidate.

    The router uses only capability status, sanitized flow shape, and already
    produced AgentEvidence. It does not emit final verdict fields and does not
    mutate evidence or detector inputs.
    """

    name = "EvidenceBudgetRouterV11"
    version = "1.1"

    def __init__(
        self,
        *,
        suspicious_temporal_min_shift: float = 0.58,
        unknown_temporal_min_shift: float = 0.35,
        unknown_temporal_max_shift: float = 0.85,
        hard_ood_score: float = 0.99,
        max_evidence_calls: int = 2,
    ) -> None:
        self.suspicious_temporal_min_shift = suspicious_temporal_min_shift
        self.unknown_temporal_min_shift = unknown_temporal_min_shift
        self.unknown_temporal_max_shift = unknown_temporal_max_shift
        self.hard_ood_score = hard_ood_score
        self.max_evidence_calls = max_evidence_calls

    @staticmethod
    def _available(state: CaseState, agent: str) -> bool:
        capability = state.detector_capabilities.get(agent)
        return capability is not None and capability.status == "available"

    @staticmethod
    def _stats_evidence(state: CaseState) -> AgentEvidence | None:
        for item in reversed(state.evidence):
            if item.agent_name == "StatsDetectorAgent":
                return item
        return None

    @staticmethod
    def _packet_count(state: CaseState) -> int:
        flow = state.safe_flow or state.flow
        if flow.sequence.original_packet_count is not None:
            return int(flow.sequence.original_packet_count)
        if flow.stats.get("packet_count") is not None:
            return int(max(0, flow.stats["packet_count"]))
        return len(flow.sequence.packet_lengths)

    def _stats_region(self, stats: AgentEvidence) -> str:
        if (
            stats.malicious_support >= stats.benign_support
            and stats.malicious_support >= 0.45
        ):
            return "malicious_leaning"
        if (
            stats.benign_support > stats.malicious_support
            and stats.benign_support >= 0.45
        ):
            return "benign_leaning"
        return "missing_decision"

    def _supplement_reason(self, state: CaseState) -> str | None:
        stats = self._stats_evidence(state)
        fusion = state.interim_fusion
        if stats is None or fusion is None:
            return None
        shift = max(stats.distribution_shift_score, fusion.distribution_shift_score)
        margin = abs(stats.malicious_support - stats.benign_support)
        packet_count = self._packet_count(state)
        if shift >= self.hard_ood_score:
            return "hard_ood_precheck_requires_temporal"
        if packet_count <= 1 and shift >= 0.95:
            return "short_flow_warning_requires_temporal"
        if stats.uncertainty >= 0.9:
            return "high_stats_uncertainty"
        if margin < 0.05:
            return "low_stats_margin"
        region = self._stats_region(stats)
        if region == "malicious_leaning":
            return (
                "malicious_shift_window"
                if shift >= self.suspicious_temporal_min_shift
                else None
            )
        if region == "benign_leaning":
            return (
                "benign_shift_window"
                if self.unknown_temporal_min_shift
                <= shift
                <= self.unknown_temporal_max_shift
                else None
            )
        if fusion.need_escalation and fusion.uncertainty >= 0.5:
            return "missing_decision_requires_temporal"
        return None

    def decide(self, state: CaseState) -> CoordinatorDecision:
        available_budget = state.remaining_budget
        metadata = {
            "router": self.name,
            "version": self.version,
            "hard_ood_score": self.hard_ood_score,
            "suspicious_temporal_min_shift": self.suspicious_temporal_min_shift,
            "unknown_temporal_min_shift": self.unknown_temporal_min_shift,
            "unknown_temporal_max_shift": self.unknown_temporal_max_shift,
        }
        if available_budget <= 0 or len(state.called_agents) >= self.max_evidence_calls:
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["EFFICIENCY_V1_1_BUDGET_EXHAUSTED"],
                rationale="EvidenceBudgetRouterV1.1 reached its evidence-call budget.",
                remaining_budget=available_budget,
                source="rule",
                planner_metadata=metadata,
            )
        if not state.evidence:
            for agent in (
                "StatsDetectorAgent",
                "TemporalBehaviorAgent",
                "TLSProtocolAgent",
            ):
                if self._available(state, agent):
                    return CoordinatorDecision(
                        action=CoordinatorAction.DISPATCH,
                        agents=[agent],
                        reason_codes=[f"EFFICIENCY_V1_1_INITIAL_{agent}"],
                        rationale=(
                            "EvidenceBudgetRouterV1.1 collects one initial "
                            "capability-available evidence view."
                        ),
                        remaining_budget=max(0, available_budget - 1),
                        source="rule",
                        planner_metadata=metadata,
                    )
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["EFFICIENCY_V1_1_NO_AVAILABLE_INITIAL_AGENT"],
                rationale="No capability-available evidence agent exists.",
                remaining_budget=available_budget,
                source="rule",
                planner_metadata=metadata,
            )
        if "StatsDetectorAgent" in state.called_agents:
            reason = self._supplement_reason(state)
            if reason:
                for agent in ("TemporalBehaviorAgent", "TLSProtocolAgent"):
                    if (
                        agent not in state.called_agents
                        and self._available(state, agent)
                        and len(state.called_agents) < self.max_evidence_calls
                        and available_budget > 0
                    ):
                        return CoordinatorDecision(
                            action=CoordinatorAction.DISPATCH,
                            agents=[agent],
                            reason_codes=[f"EFFICIENCY_V1_1_SUPPLEMENT_{agent}", reason],
                            rationale=(
                                "OOD-aware budget routing selected one "
                                "supplemental view for an OOD-sensitive or "
                                "uncertain sample."
                            ),
                            remaining_budget=max(0, available_budget - 1),
                            source="rule",
                            planner_metadata={**metadata, "supplement_reason": reason},
                        )
        return CoordinatorDecision(
            action=CoordinatorAction.STOP_AND_FUSE,
            reason_codes=["EFFICIENCY_V1_1_OOD_STABLE_EARLY_STOP"],
            rationale=(
                "EvidenceBudgetRouterV1.1 early-stops because the fixed "
                "OOD-aware gate did not require another evidence view."
            ),
            remaining_budget=available_budget,
            source="rule",
            planner_metadata=metadata,
        )


class EvidenceBudgetRouterV12(EvidenceBudgetRouterV11):
    """Selection-profiled OOD-aware efficiency candidate.

    V1.2 keeps the v1.1 safety envelope but uses a slightly stricter
    supplemental-view gate. It remains a routing-only planner: it does not
    read labels or provenance, does not mutate DetectorInput/AgentEvidence,
    and never owns final verdict fields.
    """

    name = "EvidenceBudgetRouterV12"
    version = "1.2"

    def __init__(
        self,
        *,
        suspicious_temporal_min_shift: float = 0.60,
        unknown_temporal_min_shift: float = 0.40,
        unknown_temporal_max_shift: float = 0.80,
        hard_ood_score: float = 0.99,
        short_flow_warning_score: float = 0.975,
        high_uncertainty_score: float = 0.95,
        low_margin_threshold: float = 0.03,
        max_evidence_calls: int = 2,
    ) -> None:
        super().__init__(
            suspicious_temporal_min_shift=suspicious_temporal_min_shift,
            unknown_temporal_min_shift=unknown_temporal_min_shift,
            unknown_temporal_max_shift=unknown_temporal_max_shift,
            hard_ood_score=hard_ood_score,
            max_evidence_calls=max_evidence_calls,
        )
        self.short_flow_warning_score = short_flow_warning_score
        self.high_uncertainty_score = high_uncertainty_score
        self.low_margin_threshold = low_margin_threshold

    def _metadata(self) -> dict[str, object]:
        return {
            "router": self.name,
            "version": self.version,
            "hard_ood_score": self.hard_ood_score,
            "short_flow_warning_score": self.short_flow_warning_score,
            "high_uncertainty_score": self.high_uncertainty_score,
            "low_margin_threshold": self.low_margin_threshold,
            "suspicious_temporal_min_shift": self.suspicious_temporal_min_shift,
            "unknown_temporal_min_shift": self.unknown_temporal_min_shift,
            "unknown_temporal_max_shift": self.unknown_temporal_max_shift,
        }

    def _supplement_reason(self, state: CaseState) -> str | None:
        stats = self._stats_evidence(state)
        fusion = state.interim_fusion
        if stats is None or fusion is None:
            return None
        shift = max(stats.distribution_shift_score, fusion.distribution_shift_score)
        margin = abs(stats.malicious_support - stats.benign_support)
        packet_count = self._packet_count(state)
        if shift >= self.hard_ood_score:
            return "hard_ood_precheck_requires_temporal"
        if packet_count <= 1 and shift >= self.short_flow_warning_score:
            return "short_flow_warning_requires_temporal"
        if stats.uncertainty >= self.high_uncertainty_score:
            return "high_stats_uncertainty"
        if margin < self.low_margin_threshold:
            return "low_stats_margin"
        region = self._stats_region(stats)
        if region == "malicious_leaning":
            return (
                "malicious_shift_window"
                if shift >= self.suspicious_temporal_min_shift
                else None
            )
        if region == "benign_leaning":
            return (
                "benign_shift_window"
                if self.unknown_temporal_min_shift
                <= shift
                <= self.unknown_temporal_max_shift
                else None
            )
        if fusion.need_escalation and fusion.uncertainty >= 0.55:
            return "missing_decision_requires_temporal"
        return None

    def decide(self, state: CaseState) -> CoordinatorDecision:
        decision = super().decide(state)
        metadata = self._metadata()
        decision.planner_metadata = {
            **metadata,
            **{
                key: value
                for key, value in decision.planner_metadata.items()
                if key == "supplement_reason"
            },
        }
        decision.reason_codes = [
            item.replace("EFFICIENCY_V1_1", "EFFICIENCY_V1_2")
            for item in decision.reason_codes
        ]
        decision.rationale = decision.rationale.replace(
            "EvidenceBudgetRouterV1.1",
            "EvidenceBudgetRouterV1.2",
        )
        decision.rationale = decision.rationale.replace(
            "OOD-aware budget routing",
            "Selection-profiled OOD-aware budget routing",
        )
        return decision
