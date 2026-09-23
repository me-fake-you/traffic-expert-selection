import csv
import json

import pytest

from mad_etd.evaluation import (
    EVALUATION_MODES,
    evaluate_records,
    evaluate_to_directory,
)
from mad_etd.evaluation import build_engine_for_mode


@pytest.mark.parametrize("mode", EVALUATION_MODES)
def test_every_evaluation_mode_writes_fixed_artifacts(
    mode, sample_flows, tmp_path, monkeypatch
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    output_dir = tmp_path / mode

    result = evaluate_to_directory(
        sample_flows,
        input_path="examples/sample_flows.jsonl",
        mode=mode,
        output_dir=output_dir,
        nvidia_api_key=None,
    )

    expected = {
        "predictions.csv",
        "metrics.json",
        "run_config.json",
        "audit_summary.json",
        "cost_summary.json",
    }
    assert expected == {path.name for path in output_dir.iterdir() if path.is_file()}
    assert result["metrics"]["mode"] == mode
    for json_name in expected - {"predictions.csv"}:
        payload = json.loads((output_dir / json_name).read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"
    with (output_dir / "predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["schema_version"] == "1.0" for row in rows)


def test_unlabeled_evaluation_skips_supervised_metrics(
    benign_flow, monkeypatch
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    benign_flow.labels = {}
    engine = build_engine_for_mode("rule_coordinator")

    result = evaluate_records(engine, [benign_flow], mode="rule_coordinator")
    metrics = result["metrics"]

    assert metrics["metrics_status"] == "skipped_no_labels"
    assert metrics["accuracy"] is None
    assert metrics["macro_f1"] is None
    assert metrics["PR_AUC"] is None
    assert metrics["Brier_score"] is None
    assert metrics["ECE"] is None


def test_no_reject_mode_only_emits_binary_verdicts(sample_flows, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    result = evaluate_records(
        build_engine_for_mode("no_reject"),
        sample_flows,
        mode="no_reject",
    )

    assert {
        row["verdict"] for row in result["rows"]
    } <= {"benign", "malicious"}
    assert result["metrics"]["coverage"] == 1


def test_fixed_all_agents_calls_every_registered_detector(
    sample_flows, monkeypatch
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    result = evaluate_records(
        build_engine_for_mode("fixed_all_agents"),
        sample_flows,
        mode="fixed_all_agents",
    )

    assert all(row["agent_calls"] == 6 for row in result["rows"])


def test_default_pipeline_has_zero_blocked_field_violations(
    sample_flows, monkeypatch
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    result = evaluate_records(
        build_engine_for_mode("full_pipeline"),
        sample_flows,
        mode="full_pipeline",
    )

    assert result["audit_summary"]["blocked_field_violation_count"] == 0


def test_evaluation_directory_accepts_single_pass_generator(
    sample_flows, tmp_path, monkeypatch
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    result = evaluate_to_directory(
        (record for record in sample_flows),
        input_path="examples/sample_flows.jsonl",
        mode="rule_coordinator",
        output_dir=tmp_path / "streaming",
    )

    assert result["metrics"]["sample_count"] == 2
    assert (tmp_path / "streaming" / "predictions.csv").exists()
