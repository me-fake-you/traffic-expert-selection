import json

from mad_etd.engine import build_default_engine
from mad_etd.io import load_flow_records
from mad_etd.schemas import Verdict


def test_end_to_end_pipeline_produces_report_and_complete_audit(tmp_path):
    flow = load_flow_records("examples/sample_flows.jsonl")[1]
    engine = build_default_engine()

    report, audit = engine.analyze(
        flow, audit_path=tmp_path / "trace.audit.jsonl"
    )

    assert report.verdict in set(Verdict)
    assert "StatsDetectorAgent" in report.participating_agents
    assert "TemporalBehaviorAgent" in report.participating_agents
    assert "labels.binary" in report.blocked_fields
    assert report.audit_event_count == len(audit.events)
    assert audit.events[-1].event_type == "REPORT_GENERATED"

    persisted = [
        json.loads(line)
        for line in (tmp_path / "trace.audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(persisted) == len(audit.events)
    assert [event["sequence_no"] for event in persisted] == list(
        range(1, len(persisted) + 1)
    )


def test_nvidia_mode_safely_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    report, _ = build_default_engine(
        use_nvidia=True, nvidia_api_key=None
    ).analyze(flow)

    assert report.coordinator_decisions
    assert report.coordinator_decisions[0].source == "fallback"
    assert any(
        "NVIDIA_API_KEY" in adjustment
        for adjustment in report.coordinator_decisions[0].policy_adjustments
    )

