from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    AgentEvidence,
    FeatureGroup,
    FusionResult,
    ReliabilityProfile,
    Verdict,
)


@dataclass(slots=True)
class Mass:
    benign: float
    malicious: float
    unknown: float


class FusionAgent:
    name = "FusionAgent"
    version = "1.0"

    def __init__(
        self,
        *,
        use_reliability_discount: bool = True,
        allow_reject: bool = True,
    ) -> None:
        self.use_reliability_discount = use_reliability_discount
        self.allow_reject = allow_reject

    def fuse(
        self,
        evidence: list[AgentEvidence],
        reliability: ReliabilityProfile,
        *,
        final: bool,
    ) -> FusionResult:
        distribution_shift_score = max(
            (item.distribution_shift_score for item in evidence),
            default=0.0,
        )
        hard_primary = {
            item.agent_name
            for item in evidence
            if item.abstained
            and item.distribution_shift_level == "hard"
            and item.agent_name
            in {"StatsDetectorAgent", "TemporalBehaviorAgent"}
        }
        if hard_primary == {
            "StatsDetectorAgent",
            "TemporalBehaviorAgent",
        }:
            return FusionResult(
                verdict=Verdict.UNKNOWN,
                confidence=0,
                uncertainty=1,
                conflict_score=0,
                benign_support=0,
                malicious_support=0,
                distribution_shift_score=distribution_shift_score,
                severity="low",
                need_escalation=True,
                reasons=[
                    "both primary learned detectors abstained on hard "
                    "model-distribution shift"
                ],
            )
        usable = [
            item
            for item in evidence
            if item.contributes_to_verdict and not item.abstained
        ]
        if not usable:
            forced_binary = final and not self.allow_reject
            return FusionResult(
                verdict=Verdict.BENIGN if forced_binary else Verdict.UNKNOWN,
                confidence=0,
                uncertainty=1,
                conflict_score=0,
                benign_support=0,
                malicious_support=0,
                distribution_shift_score=distribution_shift_score,
                severity="none" if forced_binary else "low",
                need_escalation=(
                    False
                    if forced_binary
                    else distribution_shift_score > 0 or not final
                ),
                reasons=[
                    "no detector produced usable evidence",
                    *(
                        ["model-distribution shift caused detector abstention"]
                        if distribution_shift_score >= 0.99
                        else []
                    ),
                    *(
                        ["reject option disabled; forced binary verdict"]
                        if forced_binary
                        else []
                    ),
                ],
            )

        combined = Mass(benign=0, malicious=0, unknown=1)
        cumulative_conflict = 0.0
        reasons: list[str] = []
        for item in usable:
            discounted = self._discount(item, reliability)
            combined, conflict = self._combine_yager(combined, discounted)
            cumulative_conflict = 1 - (1 - cumulative_conflict) * (1 - conflict)

        verdict, severity, need_escalation = self._decide(
            combined,
            cumulative_conflict,
            reliability,
            final=final,
        )
        if final and not self.allow_reject and verdict in {
            Verdict.SUSPICIOUS,
            Verdict.UNKNOWN,
        }:
            verdict = (
                Verdict.MALICIOUS
                if combined.malicious >= combined.benign
                else Verdict.BENIGN
            )
            severity = "high" if verdict == Verdict.MALICIOUS else "none"
            need_escalation = False
            reasons.append("reject option disabled; forced binary verdict")
        if cumulative_conflict >= 0.3:
            reasons.append("detector evidence is materially conflicting")
        if reliability.ood_suspected:
            reasons.append("out-of-distribution indicators present")
        if reliability.key_features_missing:
            reasons.append("key feature groups are missing")
        if distribution_shift_score >= 0.99:
            reasons.append("extreme model-distribution shift detected")
        elif distribution_shift_score >= 0.95:
            reasons.append("model-distribution shift warning detected")
        if combined.unknown >= 0.5:
            reasons.append("discounted evidence leaves high ignorance")
        if verdict == Verdict.MALICIOUS:
            reasons.append("reliability-discounted malicious support exceeded threshold")
        elif verdict == Verdict.BENIGN:
            reasons.append("reliability-discounted benign support exceeded threshold")
        elif verdict == Verdict.SUSPICIOUS:
            reasons.append("risk evidence exists but is insufficient for malicious verdict")

        confidence = (
            combined.malicious
            if verdict == Verdict.MALICIOUS
            else combined.benign
            if verdict == Verdict.BENIGN
            else max(combined.malicious, combined.benign) * (1 - combined.unknown)
        )
        return FusionResult(
            verdict=verdict,
            confidence=max(0.0, min(1.0, confidence)),
            uncertainty=max(0.0, min(1.0, combined.unknown)),
            conflict_score=max(0.0, min(1.0, cumulative_conflict)),
            benign_support=combined.benign,
            malicious_support=combined.malicious,
            distribution_shift_score=distribution_shift_score,
            severity=severity,
            need_escalation=need_escalation,
            reasons=reasons,
        )

    def _discount(
        self, item: AgentEvidence, reliability: ReliabilityProfile
    ) -> Mass:
        feature_reliability = reliability.for_group(item.feature_group)
        source_reliability = (
            feature_reliability
            * item.calibration_quality
            * item.model_reliability
            * reliability.input_completeness
            if self.use_reliability_discount
            else 1.0
        )
        assigned = item.benign_support + item.malicious_support
        benign = item.benign_support * source_reliability
        malicious = item.malicious_support * source_reliability
        unknown = 1 - assigned * source_reliability
        total = benign + malicious + unknown
        return Mass(benign / total, malicious / total, unknown / total)

    @staticmethod
    def _combine_yager(left: Mass, right: Mass) -> tuple[Mass, float]:
        conflict = left.benign * right.malicious + left.malicious * right.benign
        benign = (
            left.benign * right.benign
            + left.benign * right.unknown
            + left.unknown * right.benign
        )
        malicious = (
            left.malicious * right.malicious
            + left.malicious * right.unknown
            + left.unknown * right.malicious
        )
        unknown = left.unknown * right.unknown + conflict
        total = benign + malicious + unknown
        return Mass(benign / total, malicious / total, unknown / total), conflict

    @staticmethod
    def _decide(
        mass: Mass,
        conflict: float,
        reliability: ReliabilityProfile,
        *,
        final: bool,
    ) -> tuple[Verdict, str, bool]:
        if (
            reliability.ood_suspected
            or reliability.key_features_missing
            or mass.unknown >= 0.5
        ):
            if not final:
                return Verdict.UNKNOWN, "low", True
            if mass.malicious >= 0.3:
                return Verdict.SUSPICIOUS, "medium", True
            return Verdict.UNKNOWN, "low", True
        if conflict >= 0.3 and mass.malicious >= 0.2:
            return Verdict.SUSPICIOUS, "medium", True
        if mass.malicious >= 0.8 and mass.unknown <= 0.25:
            return Verdict.MALICIOUS, "high", False
        if mass.benign >= 0.8 and mass.unknown <= 0.25:
            return Verdict.BENIGN, "none", False
        if mass.malicious >= 0.45:
            return Verdict.SUSPICIOUS, "medium", True
        if not final:
            return Verdict.UNKNOWN, "low", True
        if mass.malicious >= 0.25:
            return Verdict.SUSPICIOUS, "medium", True
        return Verdict.UNKNOWN, "low", True
