import json

import numpy as np
import pytest

from mad_etd.base import CaseState
from mad_etd.engine import build_default_engine
from mad_etd.fusion import FusionAgent
from mad_etd.ood import (
    OODReliabilityGate,
    apply_ood_policy_to_evidence,
    ood_adjustment,
    train_ood_gate,
)
from mad_etd.reporter import ReporterAgent
from mad_etd.schemas import (
    AgentEvidence,
    FeatureGroup,
    FusionResult,
    ReliabilityProfile,
    Verdict,
)
from mad_etd.training import FeatureMatrix


def _evidence(agent, group, score, level):
    return AgentEvidence(
        agent_name=agent,
        feature_group=group,
        benign_support=0.05,
        malicious_support=0.85,
        confidence=0.85,
        uncertainty=0.1,
        calibration_quality=1,
        distribution_shift_score=score,
        distribution_shift_raw_score=0.7,
        distribution_shift_level=level,
    )


def _reliability():
    return ReliabilityProfile(
        stats_reliability=1,
        sequence_reliability=1,
        tls_reliability=1,
        payload_reliability=1,
        input_completeness=1,
    )


def test_ood_reliability_is_monotonic():
    values = [ood_adjustment(score)[1] for score in (0.94, 0.96, 0.98, 0.99)]

    assert values == sorted(values, reverse=True)
    assert values[0] == 1
    assert values[-1] == pytest.approx(0.1)


def test_soft_policy_discounts_without_abstaining():
    original = _evidence(
        "StatsDetectorAgent", FeatureGroup.STATS, 0.98, "warning"
    )

    adjusted = apply_ood_policy_to_evidence(original, "soft")

    assert adjusted.abstained is False
    assert adjusted.model_reliability < 1
    assert adjusted.uncertainty > original.uncertainty
    assert adjusted.malicious_support < original.malicious_support


def test_hybrid_policy_hard_shift_abstains():
    original = _evidence(
        "StatsDetectorAgent", FeatureGroup.STATS, 1.0, "hard"
    )

    adjusted = apply_ood_policy_to_evidence(original, "hybrid")

    assert adjusted.abstained is True
    assert adjusted.uncertainty == 1
    assert adjusted.benign_support == adjusted.malicious_support == 0


def test_two_hard_ood_detectors_force_unknown():
    evidence = [
        apply_ood_policy_to_evidence(
            _evidence("StatsDetectorAgent", FeatureGroup.STATS, 1.0, "hard"),
            "hybrid",
        ),
        apply_ood_policy_to_evidence(
            _evidence(
                "TemporalBehaviorAgent", FeatureGroup.SEQUENCE, 1.0, "hard"
            ),
            "hybrid",
        ),
    ]

    result = FusionAgent().fuse(evidence, _reliability(), final=True)

    assert result.verdict == Verdict.UNKNOWN
    assert result.distribution_shift_score == 1
    assert "distribution shift" in " ".join(result.reasons)


def test_model_reliability_is_used_by_fusion_discount():
    high = _evidence(
        "StatsDetectorAgent", FeatureGroup.STATS, 0.5, "in_domain"
    )
    low = high.model_copy(update={"model_reliability": 0.1})

    high_result = FusionAgent().fuse([high], _reliability(), final=True)
    low_result = FusionAgent().fuse([low], _reliability(), final=True)

    assert low_result.malicious_support < high_result.malicious_support
    assert low_result.uncertainty > high_result.uncertainty


def test_enabled_ood_policy_requires_gate_directory():
    with pytest.raises(ValueError, match="ood_gate_dir"):
        build_default_engine(
            detector_backend="learned",
            model_dir="missing-models",
            ood_policy="hybrid",
        )


def test_reporter_copies_fusion_distribution_shift(benign_flow):
    fusion = FusionResult(
        verdict=Verdict.UNKNOWN,
        confidence=0,
        uncertainty=1,
        conflict_score=0,
        benign_support=0,
        malicious_support=0,
        distribution_shift_score=0.99,
        severity="low",
        need_escalation=True,
    )
    state = CaseState(flow=benign_flow)

    report = ReporterAgent().build(state, fusion, audit_event_count=1)

    assert report.verdict == fusion.verdict
    assert report.distribution_shift_score == fusion.distribution_shift_score


def test_train_ood_gate_writes_versioned_ustc_only_artifacts(tmp_path):
    rng = np.random.default_rng(42)
    train = FeatureMatrix(
        x=rng.normal(size=(80, 11)).astype(np.float32),
        y=np.asarray([0, 1] * 40, dtype=np.int8),
        groups=np.arange(80, dtype=np.uint64),
        sample_ids=[f"train-{index}" for index in range(80)],
    )
    policy = FeatureMatrix(
        x=rng.normal(size=(40, 11)).astype(np.float32),
        y=np.asarray([0, 1] * 20, dtype=np.int8),
        groups=np.arange(40, dtype=np.uint64),
        sample_ids=[f"policy-{index}" for index in range(40)],
    )
    manifest = tmp_path / "split.json"
    manifest.write_text('{"version":"1.0"}', encoding="utf-8")
    output = tmp_path / "stats"

    metadata = train_ood_gate(
        agent="stats",
        train=train,
        policy=policy,
        output_dir=output,
        seed=42,
        split_manifest_path=manifest,
    )

    assert {
        "ood_gate.joblib",
        "metadata.json",
        "score_calibration.npz",
        "calibration_report.json",
    } == {path.name for path in output.iterdir()}
    assert metadata["external_dataset_used"] is False
    assert "cesnet" not in json.dumps(metadata).lower()
    gate = OODReliabilityGate(output, expected_agent="stats")
    assert gate.metadata["warning_percentile"] == 0.95
