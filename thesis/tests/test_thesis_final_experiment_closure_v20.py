from __future__ import annotations

import json
from pathlib import Path

from mad_etd.thesis_final_experiment_closure_v20 import (
    DEFAULT_OUTPUT,
    EXPECTED_RUNTIME_SHA256,
    _average_prediction,
    _normalised_malicious_probability,
    finalize_thesis_final_experiment_closure_v20,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / DEFAULT_OUTPUT


def _read(name: str) -> dict:
    return json.loads((RUN / name).read_text(encoding="utf-8-sig"))


def test_v20_probability_average_uses_only_non_abstained_evidence() -> None:
    row = {
        "clean_stats_evidence": json.dumps(
            {
                "benign_support": 0.1,
                "malicious_support": 0.9,
                "abstained": False,
            }
        ),
        "clean_temporal_evidence": json.dumps(
            {
                "benign_support": 0.99,
                "malicious_support": 0.01,
                "abstained": True,
            }
        ),
    }
    assert _average_prediction(row, "clean_") == "malicious"
    assert (
        _normalised_malicious_probability(
            {
                "benign_support": 0.2,
                "malicious_support": 0.8,
                "abstained": False,
            }
        )
        == 0.8
    )


def test_v20_acceptance_contains_positive_but_scoped_ustc_result() -> None:
    report = _read("acceptance_report.json")
    assert report["status"] == "accepted_v20_positive_thesis_evidence_closure"
    assert report["ustc"]["point_estimate_positive"] is True
    assert report["ustc"]["delta_vs_simple_average"] > 0.02
    assert report["ustc"]["sample_ci_lower_gt_zero"] is True
    assert report["ustc"]["group_ci_lower_gt_zero"] is False


def test_v20_fusion_reduces_harmful_flips_and_retains_coverage_cost() -> None:
    report = _read("acceptance_report.json")
    default = report["fusion_robustness"]["actual_runtime_v3_0"]
    continuous = report["fusion_robustness"][
        "current_stats__v2_6_continuous"
    ]
    assert default["probability_average_harmful_flip_count"] == 167
    assert default["fusion_harmful_flip_count"] == 0
    assert default["fusion_perturbed_coverage"] < 0.1
    assert continuous["probability_average_harmful_flip_count"] == 110
    assert continuous["fusion_harmful_flip_count"] == 0
    assert 0.85 < continuous["fusion_perturbed_coverage"] < 0.9


def test_v20_does_not_reopen_nbaiot_acceptance_or_overstate_external_work() -> None:
    report = _read("acceptance_report.json")
    assert report["n_baiot"]["acceptance_reopened"] is False
    assert (
        report["n_baiot"]["formal_same_acceptance_ordinary_ensemble_available"]
        is False
    )
    assert report["external_multiagent"]["system_count"] == 5
    assert report["external_multiagent"]["faithful_reproduction_count"] == 0
    assert (
        report["external_multiagent"][
            "direct_same_dataset_task_split_comparison_count"
        ]
        == 0
    )


def test_v20_runtime_hash_and_safety_invariants_are_unchanged() -> None:
    report = finalize_thesis_final_experiment_closure_v20(RUN)
    assert report["acceptance_gates_passed"] is True
    assert report["runtime_sha256_before"] == EXPECTED_RUNTIME_SHA256
    assert report["runtime_sha256_after"] == EXPECTED_RUNTIME_SHA256
    assert report["runtime_safe_v3_0_remains_default"] is True
    assert report["blocked_field_violation"] == 0
    assert report["fusion_ownership_violation"] == 0
    assert report["ood_override"] == 0
    assert report["illegal_verdict_execution"] == 0
    assert report["fake_metric_count"] == 0
    assert report["test_or_acceptance_used_for_selection"] is False
    assert report["training_performed"] is False
    assert report["promoted_runtime_created"] is False
