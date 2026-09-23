from mad_etd.detectors import (
    FamilyAttributionAgent,
    IntentAgent,
    PayloadRepresentationAgent,
    StatsDetectorAgent,
    TLSProtocolAgent,
    TemporalBehaviorAgent,
)
from mad_etd.schemas import AgentEvidence, DetectorInput, SequenceFeatures


def test_stats_detector_output_matches_agent_evidence_schema(benign_detector_input):
    result = StatsDetectorAgent().analyze(benign_detector_input)

    assert AgentEvidence.model_validate(result.model_dump()) == result
    assert result.agent_name == "StatsDetectorAgent"
    assert result.benign_support + result.malicious_support <= 1


def test_temporal_detector_recognizes_periodic_beacon():
    detector_input = DetectorInput(
        sequence=SequenceFeatures(
            packet_lengths=[240, 200] * 6,
            directions=[1] * 10 + [-1, -1],
            iats=[5.0, 5.1, 4.9, 5.0, 5.0, 5.1, 4.9, 5.0, 5.0, 5.1, 4.9],
        )
    )

    result = TemporalBehaviorAgent().analyze(detector_input)

    assert result.malicious_support > result.benign_support
    assert any("periodic" in reason for reason in result.evidence)


def test_tls_detector_abstains_without_metadata():
    result = TLSProtocolAgent().analyze(DetectorInput())

    assert result.abstained is True
    assert result.uncertainty == 1
    assert result.benign_support == result.malicious_support == 0


def test_family_attribution_abstains_and_keeps_unknown_when_insufficient():
    result = FamilyAttributionAgent().analyze(
        DetectorInput(sequence=SequenceFeatures(packet_lengths=[100, 120]))
    )

    assert result.abstained is True
    assert result.family_scores == {}
    assert result.contributes_to_verdict is False


def test_payload_agent_remains_explicit_mvp_placeholder():
    result = PayloadRepresentationAgent().analyze(DetectorInput())

    assert result.abstained is True
    assert "disabled in MVP" in result.evidence[0]


def test_intent_agent_returns_unknown_without_reliable_labels():
    result = IntentAgent().analyze(DetectorInput())

    assert result.abstained is True
    assert result.intent == "unknown"
    assert result.contributes_to_verdict is False
