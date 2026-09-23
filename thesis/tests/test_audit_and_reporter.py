import json

from mad_etd.audit import AuditLogger
from mad_etd.base import CaseState
from mad_etd.engine import build_default_engine
from mad_etd.reporter import ReporterAgent
from mad_etd.schemas import FusionResult, Verdict


def _malicious_fusion():
    return FusionResult(
        verdict=Verdict.MALICIOUS,
        confidence=0.91,
        uncertainty=0.05,
        conflict_score=0.02,
        benign_support=0.04,
        malicious_support=0.91,
        severity="high",
        need_escalation=False,
        reasons=["test fusion result"],
    )


def test_reporter_copies_fusion_verdict_without_reclassification(benign_flow):
    reporter = ReporterAgent()
    report = reporter.build(
        CaseState(flow=benign_flow),
        _malicious_fusion(),
        audit_event_count=1,
    )

    assert report.verdict == Verdict.MALICIOUS
    assert report.confidence == 0.91
    assert report.malicious_support == 0.91


def test_markdown_is_rendered_from_detection_report(benign_flow):
    reporter = ReporterAgent()
    report = reporter.build(
        CaseState(flow=benign_flow),
        _malicious_fusion(),
        audit_event_count=1,
    )
    markdown = reporter.to_markdown(report)

    assert "- Verdict: **malicious**" in markdown
    assert f"Schema version: `{report.schema_version}`" in markdown


def test_full_audit_chain_contains_all_required_stages(suspicious_flow):
    _, audit = build_default_engine().analyze(suspicious_flow)
    event_types = {event.event_type for event in audit.events}

    assert "FIELD_POLICY_APPLIED" in event_types
    assert "RELIABILITY_ASSESSED" in event_types
    assert "ROUTING_DECISION" in event_types
    assert "POLICY_GUARD_EVALUATION" in event_types
    assert "AGENT_EVIDENCE" in event_types
    assert "FINAL_FUSION" in event_types
    assert "REPORT_GENERATED" in event_types


def test_audit_events_are_versioned_and_sequential(suspicious_flow):
    _, audit = build_default_engine().analyze(suspicious_flow)

    assert all(event.schema_version == "1.0" for event in audit.events)
    assert [event.sequence_no for event in audit.events] == list(
        range(1, len(audit.events) + 1)
    )


def test_audit_logger_starts_clean_when_path_is_reused(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLogger("trace", path)
    first.log("one", "FIRST")
    second = AuditLogger("trace", path)
    second.log("two", "SECOND")
    persisted = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(persisted) == 1
    assert persisted[0]["event_type"] == "SECOND"
