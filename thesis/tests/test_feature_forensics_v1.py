from __future__ import annotations

import csv
import json

from mad_etd.feature_forensics_v1 import (
    DATASETS,
    FeatureForensicsRecord,
    _mutation_invariance,
    build_feature_forensics_v1,
    run_feature_forensics_v1,
)


def test_feature_forensics_policy_enum_and_mutation_invariance(tmp_path):
    record = FeatureForensicsRecord(
        feature_name="stats.packet_count",
        dataset_name="USTC-TFC2016",
        semantic_role="flow_statistics",
        shortcut_risk=0.0,
        final_feature_policy="safe_common",
        decision_reason="audited",
    )
    assert record.final_feature_policy == "safe_common"
    mutation = _mutation_invariance()
    assert mutation["mutation_invariance"] is True
    assert mutation["blocked_field_violation"] == 0


def test_feature_forensics_build_covers_eight_datasets_without_training(tmp_path):
    report = build_feature_forensics_v1(tmp_path)
    assert report["dataset_count"] == 8
    assert report["new_detector_trained"] is False
    assert report["runtime_safe_v3_0_remains_default"] is True
    with (tmp_path / "dataset_feature_inventory.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["dataset_name"] for row in rows} == set(DATASETS)
    blocked = json.loads((tmp_path / "blocked_feature_registry.json").read_text(encoding="utf-8"))
    assert "context.src_ip" in blocked["blocked"]


def test_feature_forensics_writes_conditional_and_source_predictability_tables(tmp_path):
    build_feature_forensics_v1(tmp_path)
    run_feature_forensics_v1(tmp_path)
    assert (tmp_path / "safe_conditional_features.csv").exists()
    assert (tmp_path / "source_predictability_results.csv").exists()
