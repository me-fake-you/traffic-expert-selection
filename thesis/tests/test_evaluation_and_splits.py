from mad_etd.engine import build_default_engine
from mad_etd.evaluation import evaluate_records
from mad_etd.io import load_flow_records
from mad_etd.splits import build_split_manifest


def test_split_manifest_keeps_capture_group_together():
    records = load_flow_records("examples/sample_flows.jsonl")
    duplicate = records[0].model_copy(deep=True)
    duplicate.trace_id = "copy"
    duplicate.sample_id = "copy"
    records.append(duplicate)

    manifest = build_split_manifest(records, seed=3)
    assigned = manifest["assignments"]
    locations = [
        split for split, sample_ids in assigned.items() if "benign-001" in sample_ids
    ]
    copy_locations = [
        split for split, sample_ids in assigned.items() if "copy" in sample_ids
    ]

    assert locations == copy_locations
    assert len(manifest["manifest_sha256"]) == 64


def test_evaluation_reports_selective_metrics():
    records = load_flow_records("examples/sample_flows.jsonl")
    result = evaluate_records(build_default_engine(), records)

    assert result["metrics"]["sample_count"] == 2
    assert 0 <= result["metrics"]["coverage"] <= 1
    assert result["metrics"]["average_agent_calls"] >= 2
    assert result["audit_summary"]["audit_chain_completion_rate"] == 1


def test_split_manifest_stays_close_to_requested_ratios():
    records = []
    base = load_flow_records("examples/sample_flows.jsonl")[0]
    for index in range(100):
        item = base.model_copy(deep=True)
        item.trace_id = f"trace-{index}"
        item.sample_id = f"sample-{index}"
        item.provenance = {
            "source_file": "capture.pcap",
            "capture_start_epoch": index * 300,
        }
        records.append(item)

    manifest = build_split_manifest(
        records, seed=4, train_ratio=0.7, validation_ratio=0.15
    )
    counts = {key: len(value) for key, value in manifest["assignments"].items()}

    assert counts == {"train": 70, "validation": 15, "test": 15}
