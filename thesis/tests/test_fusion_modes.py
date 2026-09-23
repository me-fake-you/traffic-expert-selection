from mad_etd.fusion import FusionAgent
from mad_etd.schemas import (
    AgentEvidence,
    FeatureGroup,
    ReliabilityProfile,
    Verdict,
)


def _reliability(value: float) -> ReliabilityProfile:
    return ReliabilityProfile(
        stats_reliability=value,
        sequence_reliability=value,
        tls_reliability=value,
        payload_reliability=value,
        input_completeness=1,
    )


def _evidence(
    name: str,
    benign: float,
    malicious: float,
    group: FeatureGroup = FeatureGroup.STATS,
) -> AgentEvidence:
    return AgentEvidence(
        agent_name=name,
        feature_group=group,
        benign_support=benign,
        malicious_support=malicious,
        confidence=max(benign, malicious),
        uncertainty=1 - benign - malicious,
        calibration_quality=1,
    )


def test_reliability_discount_reduces_support_and_increases_ignorance():
    evidence = [_evidence("stats", 0.05, 0.9)]
    fusion = FusionAgent(use_reliability_discount=True)

    high = fusion.fuse(evidence, _reliability(1.0), final=True)
    low = fusion.fuse(evidence, _reliability(0.2), final=True)

    assert low.malicious_support < high.malicious_support
    assert low.uncertainty > high.uncertainty


def test_disabling_reliability_discount_makes_result_invariant_to_profile():
    evidence = [_evidence("stats", 0.05, 0.9)]
    fusion = FusionAgent(use_reliability_discount=False)

    high = fusion.fuse(evidence, _reliability(1.0), final=True)
    low = fusion.fuse(evidence, _reliability(0.1), final=True)

    assert high.malicious_support == low.malicious_support
    assert high.uncertainty == low.uncertainty


def test_conflicting_agents_do_not_produce_confident_binary_verdict():
    evidence = [
        _evidence("stats", 0.9, 0.05, FeatureGroup.STATS),
        _evidence("temporal", 0.05, 0.9, FeatureGroup.SEQUENCE),
    ]

    result = FusionAgent().fuse(evidence, _reliability(1.0), final=True)

    assert result.verdict in {Verdict.SUSPICIOUS, Verdict.UNKNOWN}
    assert result.conflict_score > 0.3


def test_high_ignorance_is_unknown_not_suspicious():
    evidence = [_evidence("weak", 0.1, 0.1)]

    result = FusionAgent().fuse(evidence, _reliability(0.2), final=True)

    assert result.verdict == Verdict.UNKNOWN
    assert result.uncertainty >= 0.5


def test_no_reject_forces_binary_final_verdict():
    evidence = [_evidence("weak", 0.15, 0.2)]

    result = FusionAgent(allow_reject=False).fuse(
        evidence, _reliability(0.3), final=True
    )

    assert result.verdict in {Verdict.BENIGN, Verdict.MALICIOUS}
    assert result.need_escalation is False
