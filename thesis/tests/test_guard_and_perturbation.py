from mad_etd.guard import PseudoFeatureGuardAgent
from mad_etd.io import load_flow_records
from mad_etd.perturb import apply_perturbation


def test_dummy_packet_perturbation_downgrades_sequence_reliability():
    flow = load_flow_records("examples/sample_flows.jsonl")[1]
    guard = PseudoFeatureGuardAgent()
    baseline = guard.assess(flow)
    perturbed = apply_perturbation(
        flow, "dummy_packet", strength=0.8, seed=11
    )
    assessed = guard.assess(perturbed)

    assert assessed.sequence_reliability < baseline.sequence_reliability
    assert any("dummy insertion" in item for item in assessed.indicators)


def test_perturbation_recomputes_statistics_consistently():
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    perturbed = apply_perturbation(flow, "padding", strength=0.7, seed=2)

    assert perturbed.stats["packet_count"] == len(
        perturbed.sequence.packet_lengths
    )
    assert perturbed.stats["total_bytes"] == sum(
        perturbed.sequence.packet_lengths
    )


def test_padding_perturbation_downgrades_size_reliability():
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    guard = PseudoFeatureGuardAgent()
    baseline = guard.assess(flow)
    perturbed = apply_perturbation(flow, "padding", strength=1.0, seed=4)
    assessed = guard.assess(perturbed)

    assert assessed.sequence_reliability < baseline.sequence_reliability
    assert assessed.stats_reliability < baseline.stats_reliability
    assert any("padding" in item for item in assessed.indicators)


def test_iat_jitter_downgrades_temporal_reliability():
    flow = load_flow_records("examples/sample_flows.jsonl")[1]
    guard = PseudoFeatureGuardAgent()
    baseline = guard.assess(flow)
    perturbed = apply_perturbation(flow, "iat_jitter", strength=1.0, seed=3)
    assessed = guard.assess(perturbed)

    assert assessed.sequence_reliability < baseline.sequence_reliability
    assert any("IAT" in item for item in assessed.indicators)
