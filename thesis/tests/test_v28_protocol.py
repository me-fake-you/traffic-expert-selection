import csv
import json
from pathlib import Path

import pytest

from mad_etd.field_audit import AuditPolicy, FieldAuditAgent
from mad_etd.field_contract import (
    field_contract_payload,
    field_contract_sha256,
)
from mad_etd.io import load_flow_records, write_jsonl
from mad_etd.schemas import FieldRole
from mad_etd.v28_protocol import (
    V28_DATASET_COUNTS,
    V28_SHADOW_MODES,
    build_v28_conformance_manifest,
    finalize_v28,
    run_v28_runtime_conformance,
)


def _test_sources(tmp_path: Path, record) -> dict[str, dict]:
    sources = {}
    for name in V28_DATASET_COUNTS:
        path = tmp_path / "sources" / f"{name}.jsonl"
        cloned = record.model_copy(
            update={
                "trace_id": f"{name}-trace",
                "sample_id": f"{name}-sample",
            },
            deep=True,
        )
        write_jsonl([cloned], path)
        sources[name] = {
            "dataset": name,
            "input_path": str(path),
            "manifest_path": None,
            "split": (
                "development"
                if name == "cipherspectrum_development"
                else "unlabeled_ood"
                if name == "annotated_tls_ood"
                else "validation"
            ),
            "adapter": "all_records",
            "expected_count": 1,
            "supervised_metrics_allowed": False,
        }
    return sources


def _rule_profile(tmp_path: Path) -> None:
    (tmp_path / "runtime_safe_v2_8.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "runtime_safe_v2_8",
                "description": "test profile",
                "promoted": False,
                "acceptance_allowed": False,
                "field_audit_mode": "strict",
                "field_audit_required": True,
                "coordinator": "rule",
                "detector_backend": "rule",
                "ood_policy": "off",
                "feature_flags": {},
            }
        ),
        encoding="utf-8",
    )


def test_field_contract_distinguishes_empty_unknown_and_context(
    suspicious_flow,
):
    flow = suspicious_flow.model_copy(deep=True)
    flow.stats["new_populated_stat"] = 4
    flow.context["transport"] = "tcp"
    flow.payload_tokens = None
    audit = FieldAuditAgent().audit(flow)
    roles = {item.path: item.role for item in audit.decisions}

    assert roles["stats.new_populated_stat"] == FieldRole.UNKNOWN
    assert roles["context.transport"] == FieldRole.CONTEXT_ONLY
    assert "payload_tokens" in audit.ignored_empty_fields
    assert "payload_tokens" not in audit.unknown_fields
    assert AuditPolicy().mode == "strict"


def test_field_contract_is_deterministic_and_has_consumers():
    payload = field_contract_payload()
    assert payload["contract_version"] == "2.8"
    assert len(field_contract_sha256()) == 64
    assert payload["fields"]["stats.packet_count"]["consumers"] == [
        "StatsDetectorAgent"
    ]
    assert (
        payload["fields"]["tls.known_bad_fingerprint"]["role"]
        == "BLOCKED"
    )


def test_v28_manifest_is_deterministic_and_rejects_locked_test(
    tmp_path: Path,
    suspicious_flow,
    monkeypatch,
):
    monkeypatch.setattr(
        "mad_etd.v28_protocol.V28_DATASET_COUNTS",
        {name: 1 for name in V28_DATASET_COUNTS},
    )
    sources = _test_sources(tmp_path, suspicious_flow)
    first = build_v28_conformance_manifest(
        tmp_path / "first",
        sources=sources,
        shadow_count=2,
    )
    second = build_v28_conformance_manifest(
        tmp_path / "second",
        sources=sources,
        shadow_count=2,
    )

    assert first["datasets"] == second["datasets"]
    assert first["shadow_sample_keys"] == second["shadow_sample_keys"]
    assert first["test_used"] is False
    assert first["cipherspectrum_locked_test_used"] is False

    bad = {name: dict(spec) for name, spec in sources.items()}
    bad["cipherspectrum_validation"]["split"] = "locked_test"
    with pytest.raises(ValueError, match="forbids test/locked_test"):
        build_v28_conformance_manifest(
            tmp_path / "bad",
            sources=bad,
            shadow_count=2,
        )


def test_v28_protocol_is_resumable_and_preserves_invariance(
    tmp_path: Path,
    suspicious_flow,
    monkeypatch,
):
    monkeypatch.setattr(
        "mad_etd.v28_protocol.V28_DATASET_COUNTS",
        {name: 1 for name in V28_DATASET_COUNTS},
    )
    output = tmp_path / "run"
    sources = _test_sources(tmp_path, suspicious_flow)
    build_v28_conformance_manifest(
        output,
        sources=sources,
        shadow_count=2,
    )
    _rule_profile(tmp_path)

    first = run_v28_runtime_conformance(
        output,
        config_dir=tmp_path,
        memory_dir=tmp_path / "memory",
        knowledge_base_dir="knowledge_base",
    )
    second = run_v28_runtime_conformance(
        output,
        config_dir=tmp_path,
        memory_dir=tmp_path / "memory",
        knowledge_base_dir="knowledge_base",
    )

    assert first["strict_legacy_sample_count"] == 6
    assert first["shadow_evaluation_count"] == 2 * len(V28_SHADOW_MODES)
    assert second == first
    with (output / "strict_vs_legacy" / "predictions.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 6

    result = finalize_v28(output, test_count=1)
    assert result["acceptance_status"] == "passed"
    assert result["audit_completion_rate"] == 1.0
    assert result["blocked_field_violation_count"] == 0
    assert result["unknown_field_execution_count"] == 0
    assert result["fusion_ownership_violation_count"] == 0
    assert result["ood_override_count"] == 0
    assert result["illegal_verdict_execution_count"] == 0
    assert all(
        item["verdict_agreement"] == 1.0
        for item in result["strict_legacy"]
    )
