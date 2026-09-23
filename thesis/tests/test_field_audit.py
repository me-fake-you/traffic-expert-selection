from mad_etd.field_audit import FieldAuditAgent
from mad_etd.io import load_flow_records


def test_label_and_identifier_fields_never_reach_detectors():
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    auditor = FieldAuditAgent()
    result = auditor.audit(flow)
    safe = auditor.make_safe_flow(flow, result)

    assert "labels.binary" in result.blocked_fields
    assert "trace_id" in result.blocked_fields
    assert "context.src_ip" in result.context_only_fields
    assert safe.labels == {}
    assert safe.provenance == {}
    assert "src_ip" not in safe.context
    assert safe.stats
    assert safe.sequence.packet_lengths


def test_dataset_profiler_flags_label_proxy():
    rows = [
        {"session_id": f"benign-{index}", "capture": "a", "label": "benign"}
        for index in range(5)
    ] + [
        {"session_id": f"malicious-{index}", "capture": "b", "label": "malicious"}
        for index in range(5)
    ]
    report = FieldAuditAgent().profile_dataset(rows, "label")
    by_name = {profile.field: profile for profile in report.fields}

    assert by_name["capture"].risk_score >= 0.95
    assert by_name["session_id"].risk_score >= 0.8
    assert "capture" in report.high_risk_fields

