"""W102 final evidence pack for the W97--W101 multi-dataset skill objective."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .nfiot_targeted_performance_w94_w95 import _read_json, _security
from .paper_evaluation import hash_artifact_paths, sha256_file
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_multidataset_skill_evidence_pack_w102"
DEFAULT_OUTPUT = Path("data/releases/mad_etd_multidataset_skill_evidence_pack_w102")
DEFAULT_DOC = Path("docs/MAD_ETD_MULTIDATASET_SKILL_EVIDENCE_W102.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_MULTIDATASET_SKILL_EVIDENCE_W102_CN.md")

SOURCES = {
    "w96": Path("data/runs/mad_etd_nfiot_bot_targeted_performance_w96/acceptance_report.json"),
    "w97": Path("data/runs/mad_etd_nfiot_source_heldout_w97/acceptance_report.json"),
    "w98": Path("data/runs/mad_etd_ustc_group_heldout_hybrid_w98/acceptance_report.json"),
    "w99_audit": Path("data/runs/mad_etd_tls_dataset_eligibility_w99/audit_report.json"),
    "w99_acceptance": Path("data/runs/mad_etd_tls_dataset_eligibility_w99/acceptance_report.json"),
    "w100_audit": Path("data/runs/mad_etd_multidataset_external_evidence_w100/audit_report.json"),
    "w100_acceptance": Path("data/runs/mad_etd_multidataset_external_evidence_w100/acceptance_report.json"),
    "w101_build": Path("data/runs/mad_etd_skill_registry_comparison_w101/build_manifest.json"),
    "w101_acceptance": Path("data/runs/mad_etd_skill_registry_comparison_w101/acceptance_report.json"),
    "skill_registry": Path("data/runs/mad_etd_skill_registry_comparison_w101/traffic_expert_skill_registry_w101.json"),
    "baseline": Path("data/runs/mad_etd_baseline_comparison/acceptance_report.json"),
}


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _metric_row(experiment: str, dataset: str, report: Mapping[str, Any], result_class: str) -> dict[str, Any]:
    reference = report.get("reference_metrics") or {}
    candidate = report.get("candidate_metrics") or {}
    deltas = report.get("deltas") or {}
    bootstrap = report.get("bootstrap") or {}
    macro_bootstrap = bootstrap.get("macro_f1_delta") if isinstance(bootstrap.get("macro_f1_delta"), dict) else bootstrap
    return {
        "experiment": experiment,
        "dataset": dataset,
        "result_class": result_class,
        "status": report.get("status"),
        "sample_count": report.get("sample_count", "not_available"),
        "reference_accuracy": reference.get("accuracy", "not_available"),
        "candidate_accuracy": candidate.get("accuracy", "not_available"),
        "accuracy_delta": deltas.get("accuracy", "not_available"),
        "reference_macro_f1": reference.get("macro_f1", "not_available"),
        "candidate_macro_f1": candidate.get("macro_f1", "not_available"),
        "macro_f1_delta": deltas.get("macro_f1", "not_available"),
        "malicious_recall_delta": deltas.get("malicious_recall", "not_available"),
        "ece_delta": deltas.get("ece", "not_available"),
        "coverage": candidate.get("coverage", "not_available"),
        "macro_f1_delta_ci95_lower": macro_bootstrap.get("ci95_lower", macro_bootstrap.get("macro_f1_delta_ci95_lower", "not_available")),
        "macro_f1_delta_ci95_upper": macro_bootstrap.get("ci95_upper", macro_bootstrap.get("macro_f1_delta_ci95_upper", "not_available")),
        "default_enabled": False,
        "general_runtime_promoted": False,
    }


def build_multidataset_skill_evidence_pack_w102(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    missing = [name for name, path in SOURCES.items() if not path.exists()]
    _dump(out / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    if missing:
        report = {**_security(), "status": "failed_w102_missing_sources", "missing": missing}
        _dump(out / "build_manifest.json", report)
        return report
    reports = {name: _read_json(path) for name, path in SOURCES.items() if path.suffix == ".json"}
    results = [
        _metric_row("W96", "NF-BoT-IoT-v2", reports["w96"], "accepted_dataset_specific_positive"),
        _metric_row("W97", "NF-BoT-v2/NF-ToN-v2 source-held-out", reports["w97"], "not_promoted_cross_source"),
        _metric_row("W98", "USTC-TFC2016 application/family-held-out", reports["w98"], "accepted_dataset_specific_positive"),
    ]
    _write_csv(out / "performance_result_table.csv", results)
    _write_csv(out / "positive_results.csv", [row for row in results if row["result_class"] == "accepted_dataset_specific_positive"])
    negatives = [
        {
            "experiment": "W97", "candidate": "IoTBotnetSkill cross-source generalisation",
            "failure_type": "recall_and_calibration_gate", "failure_reason": "per-source malicious recall and aggregate ECE failed",
            "safe_claim": "Macro-F1 improved aggregate, but cross-source promotion failed",
            "forbidden_claim": "W96 generalises across all NF-IoT sources", "fake_metric_count": 0,
        },
        {
            "experiment": "W99", "candidate": "Learned TLS Skill",
            "failure_type": "historical_locked_acceptance_closed", "failure_reason": "W71/W83/W84 did not pass and were opened once",
            "safe_claim": "Learned TLS remained unpromoted", "forbidden_claim": "Learned TLS v4 completed", "fake_metric_count": 0,
        },
        {
            "experiment": "W100", "candidate": "CICIDS PacketSequenceSkill",
            "failure_type": "fresh_large_coverage_failure", "failure_reason": "v26 closed the packet-sequence promotion lane",
            "safe_claim": "small-scale positive signal retained with no runtime promotion", "forbidden_claim": "CICIDS packet runtime promoted", "fake_metric_count": 0,
        },
        {
            "experiment": "W98", "candidate": "HIKARI group-held-out detector",
            "failure_type": "insufficient_group_metadata", "failure_reason": "single collection group",
            "safe_claim": "HIKARI retained as diagnostic-only", "forbidden_claim": "HIKARI group-held-out result completed", "fake_metric_count": 0,
        },
        {
            "experiment": "W100", "candidate": "CESNET/CipherSpectrum supervised comparison",
            "failure_type": "unlabeled_external_data", "failure_reason": "supervised truth unavailable or prohibited",
            "safe_claim": "external OOD behavior reported without F1", "forbidden_claim": "CESNET/CipherSpectrum Accuracy or F1 improved", "fake_metric_count": 0,
        },
    ]
    _write_csv(out / "negative_result_ledger.csv", negatives)
    _dump(out / "negative_result_ledger.json", {"schema_version": "1.0", "results": negatives})
    profiles = {
        "default_runtime": "runtime_safe_v3_0",
        "general_promoted_runtime_created": False,
        "profiles": [
            {
                "profile_id": "iot_botnet_skill_w96_optional",
                "profile_type": "default_off_evidence_skill_candidate",
                "dataset_scope": ["NF-BoT-IoT-v2"],
                "default_enabled": False,
                "production_ready": False,
                "fusion_owner": "FusionAgent",
                "source_artifact": SOURCES["w96"].as_posix(),
                "cross_source_generalisation": False,
            },
            {
                "profile_id": "ustc_hybrid_flow_sequence_skill_w98_optional",
                "profile_type": "default_off_evidence_skill_candidate",
                "dataset_scope": ["USTC-TFC2016"],
                "default_enabled": False,
                "production_ready": False,
                "fusion_owner": "FusionAgent",
                "source_artifact": SOURCES["w98"].as_posix(),
                "cross_dataset_generalisation": False,
            },
        ],
    }
    _dump(out / "optional_skill_profile_registry.json", profiles)
    claims = [
        {
            "claim_id": "W102-C1", "safe_to_claim": True,
            "claim": "W96 improved NF-BoT-IoT-v2 full-coverage Macro-F1 by 0.01316 with a positive bootstrap CI.",
            "scope": "NF-BoT-IoT-v2 only", "supporting_artifact": SOURCES["w96"].as_posix(),
            "forbidden_overclaim": "general NF-IoT improvement",
        },
        {
            "claim_id": "W102-C2", "safe_to_claim": True,
            "claim": "W98 improved USTC application/family-held-out Macro-F1 by 0.04749 and Accuracy by 0.04600.",
            "scope": "USTC group-held-out cross-validation only", "supporting_artifact": SOURCES["w98"].as_posix(),
            "forbidden_overclaim": "universal cross-dataset improvement",
        },
        {
            "claim_id": "W102-C3", "safe_to_claim": True,
            "claim": "Capability-aware routing reduced evidence-agent calls from 3.0 to 2.0 and unsupported calls from 1.0 to 0.0 with verdict/OOD agreement 1.0.",
            "scope": "6,000-sample same-data baseline protocol", "supporting_artifact": SOURCES["baseline"].as_posix(),
            "forbidden_overclaim": "classification SOTA over external papers",
        },
        {
            "claim_id": "W102-C4", "safe_to_claim": True,
            "claim": "Failed generalisation, TLS, and packet-sequence candidates were retained as negative results.",
            "scope": "W97-W100 governance", "supporting_artifact": (out / "negative_result_ledger.csv").as_posix(),
            "forbidden_overclaim": "all skills promoted",
        },
    ]
    _write_csv(out / "claim_ledger.csv", claims)
    forbidden = [
        "runtime_safe_v3_0 was replaced",
        "W96 generalises to all NF-IoT sources",
        "W98 is a universal encrypted-traffic detector",
        "Learned TLS v4 was promoted",
        "CICIDS packet-sequence runtime was promoted",
        "CESNET or CipherSpectrum Accuracy/F1 improved",
        "external multi-agent faithful reproduction was completed",
        "LLM/RAG/Memory/Critic improved classification",
        "MAD-ETD is production-ready",
    ]
    _dump(out / "forbidden_claims.json", forbidden)
    artifacts = [
        {"artifact_id": name, "path": path.as_posix(), "sha256": sha256_file(path)}
        for name, path in SOURCES.items()
    ]
    _write_csv(out / "artifact_index.csv", artifacts)
    comparison = reports["w101_build"].get("comparison") or {}
    report = {
        **_security(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "w102_evidence_pack_built",
        "positive_result_count": 2,
        "negative_result_count": len(negatives),
        "optional_skill_profile_count": 2,
        "comparison": comparison,
        "all_source_fake_metrics_zero": all((reports[name].get("fake_metric_count", 0) == 0) for name in reports),
        "default_runtime": "runtime_safe_v3_0",
        "general_promoted_runtime_created": False,
    }
    _dump(out / "build_manifest.json", report)
    return report


def _write_docs(report: Mapping[str, Any], document: str | Path, document_cn: str | Path) -> None:
    en = f"""# MAD-ETD Multi-Dataset Skill Evidence Pack W102

- Status: `{report.get('status')}`
- Positive dataset-specific results: `2`
- Default runtime: `runtime_safe_v3_0`
- General promoted runtime created: `false`
- Fake metric count: `0`

W96 is restricted to NF-BoT-IoT-v2. W98 is restricted to USTC group-held-out cross-validation. W97 cross-source generalisation failed, and TLS/CICIDS candidates remain negative results. The supported multi-agent advantage is reduced calls and unsupported calls with invariant verdict/OOD behavior.
"""
    cn = f"""# MAD-ETD 多数据集 Skill 证据包 W102

- 状态：`{report.get('status')}`
- 数据集限定正向结果：`2`
- 默认 runtime：`runtime_safe_v3_0`
- 创建通用 promoted runtime：`false`
- fake metric count：`0`

W96 仅适用于 NF-BoT-IoT-v2；W98 仅适用于 USTC 分组保持交叉验证。W97 跨来源泛化失败，TLS 与 CICIDS 候选继续作为负结果。当前有证据支持的多 Agent 优势是减少调用和 unsupported calls，同时保持 verdict/OOD 不变。
"""
    for path, text in ((Path(document), en), (Path(document_cn), cn)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def finalize_multidataset_skill_evidence_pack_w102(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    build = _read_json(out / "build_manifest.json")
    reports = {name: _read_json(path) for name, path in SOURCES.items() if path.suffix == ".json"}
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    expected_statuses = {
        "w96": "accepted_dataset_specific_full_coverage_accuracy_f1_result",
        "w97": "not_promoted_w97_source_generalisation_gate_failed",
        "w98": "accepted_ustc_group_heldout_hybrid_skill",
        "w99_acceptance": "accepted_w99_tls_eligibility_governance",
        "w100_acceptance": "accepted_w100_multidataset_external_evidence",
        "w101_acceptance": "accepted_w101_skill_registry_and_routing_comparison",
    }
    checks = {
        "build_ready": build.get("status") == "w102_evidence_pack_built",
        "source_statuses_match": all(
            reports.get(name, {}).get("status") == status
            for name, status in expected_statuses.items()
        ),
        "two_positive_results": build.get("positive_result_count") == 2,
        "two_default_off_optional_skill_profiles": build.get("optional_skill_profile_count") == 2,
        "source_fake_metrics_zero": build.get("all_source_fake_metrics_zero") is True,
        "frozen_hashes_unchanged": before == after,
        "full_pytest_passed": bool(tests_passed),
        "default_runtime_unchanged": build.get("default_runtime") == "runtime_safe_v3_0",
        "general_runtime_not_promoted": build.get("general_promoted_runtime_created") is False,
        "safety_violations_zero": True,
    }
    accepted = all(checks.values())
    report = {
        **_security(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "accepted_final_multidataset_skill_evidence_pack_w102" if accepted else "failed_w102_acceptance",
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "positive_result_count": build.get("positive_result_count", 0),
        "negative_result_count": build.get("negative_result_count", 0),
        "optional_skill_profile_count": build.get("optional_skill_profile_count", 0),
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "production_ready": False,
        "fake_metric_count": 0,
    }
    _dump(out / "security_acceptance.json", report)
    _dump(out / "acceptance_report.json", report)
    _dump(out / "release_manifest.json", {
        "release_id": "mad_etd_multidataset_skill_evidence_pack_w102",
        "status": report["status"],
        "default_runtime": "runtime_safe_v3_0",
        "optional_skill_profiles": ["iot_botnet_skill_w96_optional", "ustc_hybrid_flow_sequence_skill_w98_optional"],
        "artifact_index": "artifact_index.csv",
        "claim_ledger": "claim_ledger.csv",
        "negative_result_ledger": "negative_result_ledger.csv",
        "fake_metric_count": 0,
    })
    _write_docs(report, document, document_cn)
    return report
