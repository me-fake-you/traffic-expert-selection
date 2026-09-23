import csv
import json

import numpy as np

from mad_etd.generalization import (
    DOMAIN_HOLDOUT_PAIRS,
    CachedAgentFeatures,
    DomainFeatureCache,
    MetricAccumulator,
    build_domain_fold_manifest,
    iter_holdout_records,
    validate_generalization,
)
from mad_etd.io import write_jsonl
from mad_etd.schemas import FlowRecord, SequenceFeatures
from mad_etd.training import FeatureMatrix


def _flow(
    sample_id,
    *,
    label,
    domain,
    source,
    start,
    packet_count=4,
):
    labels = {
        "binary": label,
        "application": domain if label == "benign" else "",
        "family": domain if label == "malicious" else "",
    }
    return FlowRecord(
        trace_id=f"trace-{sample_id}",
        sample_id=sample_id,
        stats={
            "packet_count": packet_count,
            "total_bytes": packet_count * 100,
            "outbound_bytes": packet_count * 50,
            "inbound_bytes": packet_count * 50,
            "packet_length_variance": 0,
            "duration": 1,
        },
        sequence=SequenceFeatures(
            packet_lengths=[100] * packet_count,
            directions=[1, -1] * (packet_count // 2),
            iats=[0.25] * max(0, packet_count - 1),
        ),
        provenance={
            "source_file": source,
            "capture_start_epoch": start,
        },
        labels=labels,
    )


def _fake_cache():
    counts = {
        "benign": {benign: 10 for benign, _ in DOMAIN_HOLDOUT_PAIRS},
        "malicious": {
            malicious: 12 for _, malicious in DOMAIN_HOLDOUT_PAIRS
        },
    }
    eligible = {
        "benign": {name: 5 for name in counts["benign"]},
        "malicious": {name: 6 for name in counts["malicious"]},
    }
    return DomainFeatureCache(
        agents={},
        domain_counts=counts,
        sequence_eligible_counts=eligible,
        sample_count=sum(map(sum, (counts["benign"].values(), counts["malicious"].values()))),
    )


def test_domain_manifest_covers_each_application_and_family_once():
    manifest = build_domain_fold_manifest(_fake_cache(), seed=42, folds=10)

    benign = [
        fold["held_out_benign_application"] for fold in manifest["folds"]
    ]
    malicious = [
        fold["held_out_malware_family"] for fold in manifest["folds"]
    ]
    assert len(benign) == len(set(benign)) == 10
    assert len(malicious) == len(set(malicious)) == 10
    assert len(manifest["manifest_sha256"]) == 64


def test_domain_manifest_is_deterministic():
    first = build_domain_fold_manifest(_fake_cache(), seed=42, folds=10)
    second = build_domain_fold_manifest(_fake_cache(), seed=42, folds=10)

    assert first == second


def test_cached_partitions_exclude_both_holdout_domains():
    domains = np.asarray(
        [
            "FTP",
            "FTP",
            "Gmail",
            "Gmail",
            "Gmail",
            "Nsis-ay",
            "Nsis-ay",
            "Zeus",
            "Zeus",
            "Zeus",
        ],
        dtype=object,
    )
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    buckets = np.asarray([2, 0, 2, 1, 0, 2, 0, 2, 1, 0], dtype=np.int8)
    feature = CachedAgentFeatures(
        matrix=FeatureMatrix(
            x=np.arange(20, dtype=np.float32).reshape(10, 2),
            y=labels,
            groups=np.asarray(
                [1, 1, 2, 2, 2, 3, 3, 4, 4, 4], dtype=np.uint64
            ),
            sample_ids=[
                "ftp-train",
                "ftp-cal",
                "gmail-train",
                "gmail-policy",
                "gmail-cal",
                "nsis-train",
                "nsis-cal",
                "zeus-train",
                "zeus-policy",
                "zeus-cal",
            ],
        ),
        domains=domains,
        buckets=buckets,
        partition_groups=np.asarray(
            [1, 1, 2, 3, 4, 5, 5, 6, 7, 8], dtype=np.uint64
        ),
    )

    train, calibration, policy = feature.partitions(
        held_benign="FTP", held_malicious="Nsis-ay"
    )

    all_ids = set(train.sample_ids + calibration.sample_ids + policy.sample_ids)
    assert not any("ftp" in item or "nsis" in item for item in all_ids)
    assert all_ids == {
        "gmail-train",
        "gmail-policy",
        "gmail-cal",
        "zeus-train",
        "zeus-policy",
        "zeus-cal",
    }


def test_all_captures_for_multi_capture_domain_are_held_out(tmp_path):
    records = [
        _flow(
            "weibo-1",
            label="benign",
            domain="Weibo",
            source="Weibo-1.pcap",
            start=1,
        ),
        _flow(
            "weibo-2",
            label="benign",
            domain="Weibo",
            source="Weibo-2.pcap",
            start=2,
        ),
        _flow(
            "tinba",
            label="malicious",
            domain="Tinba",
            source="Tinba.pcap",
            start=3,
        ),
        _flow(
            "ftp",
            label="benign",
            domain="FTP",
            source="FTP.pcap",
            start=4,
        ),
    ]
    path = tmp_path / "flows.jsonl"
    write_jsonl(records, path)

    selected = list(
        iter_holdout_records(
            path,
            benign_application="Weibo",
            malware_family="Tinba",
        )
    )

    assert {record.sample_id for record in selected} == {
        "weibo-1",
        "weibo-2",
        "tinba",
    }


def test_feature_cache_keeps_same_time_group_in_same_partition(tmp_path):
    records = []
    for index, (label, domain) in enumerate(
        [
            ("benign", "FTP"),
            ("benign", "Gmail"),
            ("malicious", "Zeus"),
            ("malicious", "Neris"),
        ]
    ):
        for copy in range(2):
            records.append(
                _flow(
                    f"{domain}-{copy}",
                    label=label,
                    domain=domain,
                    source=f"{domain}.pcap",
                    start=index * 300 + 1,
                )
            )
    path = tmp_path / "flows.jsonl"
    write_jsonl(records, path)

    cache = DomainFeatureCache.build(path, seed=42)
    buckets = cache.agents["stats"].buckets
    ids = cache.agents["stats"].matrix.sample_ids

    by_prefix = {}
    for sample_id, bucket in zip(ids, buckets, strict=True):
        by_prefix.setdefault(sample_id.rsplit("-", 1)[0], set()).add(int(bucket))
    assert all(len(values) == 1 for values in by_prefix.values())


def test_metric_accumulator_skips_supervised_metrics_without_labels():
    accumulator = MetricAccumulator("ood")
    accumulator.add(
        truth="",
        verdict="unknown",
        risk_score=0.5,
        agent_calls=2,
        latency_ms=1,
    )

    metrics = accumulator.metrics()

    assert metrics["metrics_status"] == "skipped_no_labels"
    assert metrics["accuracy"] is None
    assert metrics["PR_AUC"] is None
    assert metrics["selective_error_rate"] is None
    assert metrics["unknown_rate"] == 1


def test_generalization_entrypoint_writes_fold_and_aggregate_artifacts(tmp_path):
    dataset = tmp_path / "dataset"
    flows = dataset / "flows"
    flows.mkdir(parents=True)
    records = []
    for benign, malicious in DOMAIN_HOLDOUT_PAIRS:
        for group in range(6):
            records.append(
                _flow(
                    f"{benign}-{group}",
                    label="benign",
                    domain=benign,
                    source=f"{benign}.pcap",
                    start=group * 300,
                )
            )
            records.append(
                _flow(
                    f"{malicious}-{group}",
                    label="malicious",
                    domain=malicious,
                    source=f"{malicious}.pcap",
                    start=group * 300,
                )
            )
    write_jsonl(records, flows / "synthetic.jsonl")
    output = tmp_path / "holdout"

    result = validate_generalization(
        dataset,
        output,
        folds=1,
        seed=42,
        cv_splits=2,
        baseline_metrics_path=None,
        cesnet_dir=None,
        full_model_dir=None,
    )

    assert result["fold_count"] == 1
    assert (output / "domain-fold-manifest.json").exists()
    assert (output / "folds" / "fold-00" / "predictions.csv").exists()
    assert (output / "folds" / "fold-00" / "models" / "stats" / "model.joblib").exists()
    assert (output / "aggregate" / "aggregate_metrics.json").exists()
    assert (output / "aggregate" / "per_domain_metrics.csv").exists()
    audit = json.loads(
        (output / "aggregate" / "audit_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["blocked_field_violation_count"] == 0
