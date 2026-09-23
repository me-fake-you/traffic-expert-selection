"""W325 publication-grade evidence pack for W322-W324 routing efficiency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gotham_routing_efficiency_w322 import _sha256_file
from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _hash_snapshot_equal,
    _load,
    _safe_hashes,
    _write_csv,
)


EXPERIMENT = "mad_etd_routing_efficiency_evidence_pack_w325"
DEFAULT_W322 = Path(
    "data/runs/mad_etd_gotham_routing_efficiency_w322"
)
DEFAULT_W323 = Path(
    "data/runs/mad_etd_cross_dataset_routing_efficiency_w323"
)
DEFAULT_W324 = Path(
    "data/runs/mad_etd_routing_latency_stat_validation_w324"
)
DEFAULT_OUTPUT = Path(
    "data/releases/mad_etd_routing_efficiency_evidence_pack_w325"
)
DEFAULT_DOC = Path(
    "docs/MAD_ETD_ROUTING_EFFICIENCY_EVIDENCE_PACK_W325.md"
)
DEFAULT_DOC_CN = Path(
    "docs/MAD_ETD_ROUTING_EFFICIENCY_EVIDENCE_PACK_W325_CN.md"
)


def _requirements(
    w322: Path,
    w323: Path,
    w324: Path,
) -> dict[str, Path]:
    return {
        "w322_acceptance": w322 / "acceptance_report.json",
        "w322_result": w322 / "routing_efficiency_report_w322.json",
        "w322_routing_results": w322 / "routing_results.csv",
        "w323_acceptance": w323 / "acceptance_report.json",
        "w323_result": w323 / "cross_dataset_efficiency_report_w323.json",
        "w323_results": w323 / "cross_dataset_routing_results.csv",
        "w324_acceptance": w324 / "acceptance_report.json",
        "w324_result": w324 / "routing_latency_stat_report_w324.json",
        "w324_repeat_results": w324 / "latency_repeat_results.csv",
        "w324_bootstrap": w324 / "latency_bootstrap_ci.csv",
        "w324_source_summary": w324 / "source_latency_summary.csv",
    }


def _source_gate(
    source: Mapping[str, Any],
    repeat_rows: pd.DataFrame,
) -> dict[str, bool]:
    static = repeat_rows[repeat_rows["mode"] == "static_full_call"]
    aware = repeat_rows[
        repeat_rows["mode"] == "capability_aware_mad_etd"
    ]
    return {
        "source_passed_w324": source.get("passed") is True,
        "seven_repetitions_present": source.get("repetition_count") == 7,
        "request_attempts_three_to_one": (
            source.get("static_avg_request_attempts") == 3.0
            and source.get("candidate_avg_request_attempts") == 1.0
        ),
        "executed_evidence_calls_one_to_one": (
            bool(static["avg_executed_evidence_agent_calls"].eq(1.0).all())
            and bool(aware["avg_executed_evidence_agent_calls"].eq(1.0).all())
        ),
        "unsupported_calls_two_to_zero": (
            source.get("static_avg_unsupported_calls") == 2.0
            and source.get("candidate_avg_unsupported_calls") == 0.0
        ),
        "bootstrap_ci_lower_positive": (
            float(source.get("bootstrap_ci_lower_95", 0.0)) > 0.0
        ),
        "verdict_agreement_one": (
            source.get("minimum_verdict_agreement") == 1.0
        ),
        "ood_signature_agreement_one": (
            source.get("minimum_ood_agreement") == 1.0
        ),
        "classification_and_coverage_deltas_zero": all(
            source.get(name) == 0.0
            for name in (
                "accuracy_delta",
                "macro_f1_delta",
                "weighted_f1_delta",
                "malicious_recall_delta",
                "coverage_delta",
            )
        ),
    }


def _main_rows(
    result: Mapping[str, Any],
    repeat_results: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, bool]]]:
    rows: list[dict[str, Any]] = []
    source_gates: dict[str, dict[str, bool]] = {}
    for dataset, source in result["source_reports"].items():
        repeat = repeat_results[
            repeat_results["dataset"].astype(str) == dataset
        ]
        static = repeat[repeat["mode"] == "static_full_call"]
        aware = repeat[
            repeat["mode"] == "capability_aware_mad_etd"
        ]
        gates = _source_gate(source, repeat)
        source_gates[dataset] = gates
        rows.append(
            {
                "dataset": dataset,
                "sample_count": int(source["paired_sample_count"]),
                "repetitions": int(source["repetition_count"]),
                "static_request_attempts": float(
                    source["static_avg_request_attempts"]
                ),
                "capability_aware_request_attempts": float(
                    source["candidate_avg_request_attempts"]
                ),
                "request_attempt_reduction_percent": float(
                    source["request_attempt_reduction"] * 100.0
                ),
                "static_executed_evidence_calls": float(
                    static["avg_executed_evidence_agent_calls"].mean()
                ),
                "capability_aware_executed_evidence_calls": float(
                    aware["avg_executed_evidence_agent_calls"].mean()
                ),
                "static_unsupported_calls": float(
                    source["static_avg_unsupported_calls"]
                ),
                "capability_aware_unsupported_calls": float(
                    source["candidate_avg_unsupported_calls"]
                ),
                "repeated_run_median_p95_reduction_percent": float(
                    source["median_repeat_p95_latency_reduction"] * 100.0
                ),
                "sample_median_bootstrap_point_percent": float(
                    source["point_p95_latency_reduction"] * 100.0
                ),
                "bootstrap_ci_lower_95_percent": float(
                    source["bootstrap_ci_lower_95"] * 100.0
                ),
                "bootstrap_ci_upper_95_percent": float(
                    source["bootstrap_ci_upper_95"] * 100.0
                ),
                "verdict_agreement": float(
                    source["minimum_verdict_agreement"]
                ),
                "fixed_in_domain_ood_signature_agreement": float(
                    source["minimum_ood_agreement"]
                ),
                "accuracy_delta": float(source["accuracy_delta"]),
                "macro_f1_delta": float(source["macro_f1_delta"]),
                "coverage_delta": float(source["coverage_delta"]),
                "audit_completion": float(
                    source["minimum_audit_completion"]
                ),
                "source_gate_passed": all(gates.values()),
            }
        )
    return rows, source_gates


def _gotham_row(result: Mapping[str, Any]) -> dict[str, Any]:
    static = result["routing_results"]["static_full_call"]
    aware = result["routing_results"]["capability_aware_mad_etd"]
    comparison = result["candidate_comparison"]
    return {
        "dataset": "Gotham IoT",
        "evidence_role": "development_context_not_W324_statistical_lane",
        "sample_count": int(result["routing_sample_count"]),
        "static_request_attempts": static[
            "avg_evidence_agent_call_attempts"
        ],
        "capability_aware_request_attempts": aware[
            "avg_evidence_agent_call_attempts"
        ],
        "static_executed_evidence_calls": static[
            "avg_executed_evidence_agent_calls"
        ],
        "capability_aware_executed_evidence_calls": aware[
            "avg_executed_evidence_agent_calls"
        ],
        "static_unsupported_calls": static["avg_unsupported_calls"],
        "capability_aware_unsupported_calls": aware[
            "avg_unsupported_calls"
        ],
        "p95_latency_reduction_percent": (
            comparison["p95_latency_reduction"] * 100.0
        ),
        "verdict_agreement": comparison["verdict_agreement"],
        "fixed_in_domain_ood_signature_agreement": comparison[
            "ood_agreement"
        ],
        "macro_f1_delta": comparison["macro_f1_delta"],
        "audit_completion": aware["audit_completion"],
        "statistical_ci_available": False,
        "claim_scope": (
            "development context only; W324 supplies statistical support"
        ),
    }


def _plot_latency_ci(table: pd.DataFrame, output: Path) -> None:
    labels = table["dataset"].tolist()
    points = table["sample_median_bootstrap_point_percent"].to_numpy(float)
    lower = table["bootstrap_ci_lower_95_percent"].to_numpy(float)
    upper = table["bootstrap_ci_upper_95_percent"].to_numpy(float)
    errors = np.vstack([points - lower, upper - points])
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    y = np.arange(len(labels))
    axis.errorbar(
        points,
        y,
        xerr=errors,
        fmt="o",
        color="#185FA5",
        ecolor="#185FA5",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
    )
    axis.axvline(0.0, color="#606060", linewidth=1.0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("p95 governance-replay latency reduction (%)")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    for index, (point, high) in enumerate(zip(points, upper)):
        axis.text(
            high + 0.25,
            index,
            f"{point:.2f}%",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"figure_p95_latency_bootstrap_w325.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _plot_call_semantics(table: pd.DataFrame, output: Path) -> None:
    static = [
        float(table["static_request_attempts"].mean()),
        float(table["static_executed_evidence_calls"].mean()),
        float(table["static_unsupported_calls"].mean()),
    ]
    aware = [
        float(table["capability_aware_request_attempts"].mean()),
        float(table["capability_aware_executed_evidence_calls"].mean()),
        float(table["capability_aware_unsupported_calls"].mean()),
    ]
    categories = [
        "Request attempts",
        "Executed evidence calls",
        "Unsupported attempts",
    ]
    x = np.arange(len(categories))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 3.25))
    bars_static = axis.bar(
        x - width / 2,
        static,
        width,
        label="Static full-call",
        color="#A7C7E7",
        edgecolor="#3D6D99",
    )
    bars_aware = axis.bar(
        x + width / 2,
        aware,
        width,
        label="Capability-aware MAD-ETD",
        color="#2E8B57",
        edgecolor="#1E5E3B",
    )
    axis.bar_label(bars_static, fmt="%.1f", padding=3, fontsize=9)
    axis.bar_label(bars_aware, fmt="%.1f", padding=3, fontsize=9)
    axis.set_xticks(x, categories)
    axis.set_ylabel("Average calls or attempts per case")
    axis.set_ylim(0.0, 3.5)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"figure_call_semantics_w325.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def _claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "W325-C1",
            "claim": (
                "Capability-aware routing reduced evidence-agent request "
                "attempts from 3.0 to 1.0 on all W324 source lanes."
            ),
            "scope": "USTC-TFC2016, NF-IoT, CICIoT2023 frozen replay",
            "supporting_artifact": "main_efficiency_table_w325.csv",
            "safe_to_claim": True,
            "forbidden_overclaim": (
                "The number of actually executed detector inferences fell "
                "from three to one."
            ),
        },
        {
            "claim_id": "W325-C2",
            "claim": (
                "Unsupported request attempts fell from 2.0 to 0.0 while "
                "executed evidence calls remained 1.0."
            ),
            "scope": "capability profile exposed only Stats evidence",
            "supporting_artifact": "figure_call_semantics_w325.pdf",
            "safe_to_claim": True,
            "forbidden_overclaim": (
                "Three supported evidence agents were executed by static "
                "full-call."
            ),
        },
        {
            "claim_id": "W325-C3",
            "claim": (
                "Paired bootstrap intervals for p95 governance-replay latency "
                "reduction were strictly positive on all three source lanes."
            ),
            "scope": "W324 repeated governance replay, not serving latency",
            "supporting_artifact": "latency_bootstrap_table_w325.csv",
            "safe_to_claim": True,
            "forbidden_overclaim": (
                "The same percentage reduction holds for production "
                "end-to-end inference."
            ),
        },
        {
            "claim_id": "W325-C4",
            "claim": (
                "Routing preserved classification metrics, Fusion verdicts, "
                "and fixed in-domain OOD signatures in the replay."
            ),
            "scope": "identical frozen evidence and Fusion semantics",
            "supporting_artifact": "main_efficiency_table_w325.csv",
            "safe_to_claim": True,
            "forbidden_overclaim": (
                "Routing improved Accuracy, Macro-F1, recall, or external OOD "
                "detection accuracy."
            ),
        },
        {
            "claim_id": "W325-C5",
            "claim": (
                "No blocked-field, Fusion-ownership, OOD-override, illegal-"
                "verdict, or fake-metric violation occurred."
            ),
            "scope": "W322-W324 audited experimental capsules",
            "supporting_artifact": "acceptance_report.json",
            "safe_to_claim": True,
            "forbidden_overclaim": "MAD-ETD is production-ready.",
        },
    ]


def _render_docs(
    report: Mapping[str, Any],
    table: pd.DataFrame,
    document: Path,
    document_cn: Path,
) -> None:
    rows = []
    for item in table.itertuples(index=False):
        rows.append(
            f"| {item.dataset} | {item.sample_count:,} | "
            f"{item.static_request_attempts:.1f}→"
            f"{item.capability_aware_request_attempts:.1f} | "
            f"{item.static_unsupported_calls:.1f}→"
            f"{item.capability_aware_unsupported_calls:.1f} | "
            f"{item.repeated_run_median_p95_reduction_percent:.2f}% | "
            f"[{item.bootstrap_ci_lower_95_percent:.2f}%, "
            f"{item.bootstrap_ci_upper_95_percent:.2f}%] | "
            f"{item.verdict_agreement:.3f}/"
            f"{item.fixed_in_domain_ood_signature_agreement:.3f} |"
        )
    common_table = [
        "| Dataset | Samples | Requests | Unsupported | Median p95 reduction | Bootstrap 95% CI | Verdict/OOD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
    ]
    en = [
        "# MAD-ETD Routing Efficiency Evidence Pack W325",
        "",
        f"- Build status: `{report['status']}`",
        "- Evidence type: fixed-evidence governance-routing replay",
        "- Classification improvement: not claimed",
        "- Runtime promotion: none",
        "",
        "## Main statistical evidence",
        "",
        *common_table,
        "",
        "Static full-call made three request attempts but executed one supported",
        "Stats evidence agent; two requests were rejected as unsupported.",
        "Capability-aware routing made one supported request and also executed",
        "one evidence agent. Therefore the demonstrated saving is in request,",
        "guard, admission, and audit work—not three-to-one detector inference.",
        "",
        "All classification deltas were zero; verdict and fixed in-domain OOD",
        "signature agreement were 1.0. The latency values are governance replay",
        "measurements and are not production serving benchmarks.",
    ]
    cn = [
        "# MAD-ETD 路由效率证据包 W325",
        "",
        f"- 构建状态：`{report['status']}`",
        "- 证据类型：固定 evidence 的治理路由重放",
        "- 分类性能提升：不声明",
        "- Runtime 晋级：无",
        "",
        "## 主要统计证据",
        "",
        *common_table,
        "",
        "Static full-call 发出 3 次请求，但实际只执行 1 个受支持的 Stats evidence",
        "Agent；另外 2 次请求因 capability 不支持而被拒绝。Capability-aware",
        "routing 只发出 1 次受支持请求，也执行 1 个 evidence Agent。因此，本结果",
        "证明的是请求、PolicyGuard、AdmissionGate 与审计工作量的降低，不是",
        "Detector 推理次数从 3 次降为 1 次。",
        "",
        "所有分类指标 delta 均为 0，verdict 与固定 in-domain OOD signature",
        "agreement 均为 1.0。延迟属于治理重放测量，不能写成生产服务延迟。",
    ]
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("\n".join(en) + "\n", encoding="utf-8")
    document_cn.write_text("\n".join(cn) + "\n", encoding="utf-8")


def build_routing_efficiency_evidence_pack_w325(
    w322_dir: str | Path = DEFAULT_W322,
    w323_dir: str | Path = DEFAULT_W323,
    w324_dir: str | Path = DEFAULT_W324,
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
) -> dict[str, Any]:
    w322 = Path(w322_dir)
    w323 = Path(w323_dir)
    w324 = Path(w324_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = _requirements(w322, w323, w324)
    missing = [
        path.as_posix() for path in required.values() if not path.is_file()
    ]
    if missing:
        report = {
            "status": "blocked_w325_missing_source_artifacts",
            "missing_artifacts": missing,
            "fake_metric_count": 0,
            "promoted_runtime_created": False,
            "runtime_safe_v3_0_remains_default": True,
        }
        _dump(output / "evidence_pack_build_report.json", report)
        _dump(
            output / "negative_results.json",
            [
                {
                    "experiment_id": EXPERIMENT,
                    "failure_type": "missing_source_artifacts",
                    "missing_artifacts": missing,
                    "fake_metric_count": 0,
                    "runtime_modified": False,
                }
            ],
        )
        return report
    w322_acceptance = _load(required["w322_acceptance"])
    w322_result = _load(required["w322_result"])
    w323_acceptance = _load(required["w323_acceptance"])
    w323_result = _load(required["w323_result"])
    w324_acceptance = _load(required["w324_acceptance"])
    w324_result = _load(required["w324_result"])
    repeat_results = pd.read_csv(required["w324_repeat_results"])
    main_rows, source_gates = _main_rows(w324_result, repeat_results)
    main_table = pd.DataFrame(main_rows)
    all_source_gates = all(
        all(gates.values()) for gates in source_gates.values()
    )
    source_status_gates = {
        "w322_capsule_passed": w322_acceptance.get("status")
        == "passed_w322_development_positive_efficiency_capsule",
        "w323_capsule_passed": w323_acceptance.get("status")
        == "passed_w323_cross_dataset_positive_efficiency_capsule",
        "w324_capsule_passed": w324_acceptance.get("status")
        == "passed_w324_routing_latency_statistical_capsule",
        "w324_statistical_result_accepted": w324_result.get("status")
        == "accepted_cross_dataset_statistically_supported_efficiency_result",
        "all_three_w324_sources_passed": len(source_gates) == 3
        and all_source_gates,
        "source_fake_metrics_zero": all(
            item.get("fake_metric_count") == 0
            for item in (
                w322_acceptance,
                w323_acceptance,
                w324_acceptance,
            )
        ),
        "source_runtime_default_unchanged": all(
            item.get("runtime_safe_v3_0_remains_default") is True
            for item in (
                w322_acceptance,
                w323_acceptance,
                w324_acceptance,
            )
        ),
        "source_promoted_runtime_not_created": all(
            item.get("promoted_runtime_created") is False
            for item in (
                w322_acceptance,
                w323_acceptance,
                w324_acceptance,
            )
        ),
    }
    ready = all(source_status_gates.values())
    main_table.to_csv(
        output / "main_efficiency_table_w325.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_table = main_table[
        [
            "dataset",
            "sample_count",
            "repetitions",
            "repeated_run_median_p95_reduction_percent",
            "sample_median_bootstrap_point_percent",
            "bootstrap_ci_lower_95_percent",
            "bootstrap_ci_upper_95_percent",
        ]
    ].copy()
    bootstrap_table.to_csv(
        output / "latency_bootstrap_table_w325.csv",
        index=False,
        encoding="utf-8-sig",
    )
    call_table = main_table[
        [
            "dataset",
            "static_request_attempts",
            "capability_aware_request_attempts",
            "static_executed_evidence_calls",
            "capability_aware_executed_evidence_calls",
            "static_unsupported_calls",
            "capability_aware_unsupported_calls",
        ]
    ].copy()
    call_table.to_csv(
        output / "call_semantics_table_w325.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_csv(
        output / "gotham_development_context_table_w325.csv",
        [_gotham_row(w322_result)],
    )
    claims = _claim_rows()
    _write_csv(output / "claim_boundary_table_w325.csv", claims)
    trace_rows = [
        {
            "source_id": source_id,
            "artifact_path": path.as_posix(),
            "sha256": _sha256_file(path),
            "artifact_role": (
                "acceptance boundary"
                if "acceptance" in source_id
                else "numeric source"
            ),
            "paper_reported_external_metric": False,
        }
        for source_id, path in required.items()
    ]
    _write_csv(output / "source_traceability_w325.csv", trace_rows)
    visual_contract = [
        {
            "visual": "figure_p95_latency_bootstrap_w325",
            "responsibility": (
                "show cross-dataset p95 governance-replay latency reduction "
                "and paired bootstrap uncertainty"
            ),
            "claim_supported": "W325-C3",
            "must_not_claim": "production end-to-end latency reduction",
            "source_data": "latency_bootstrap_table_w325.csv",
        },
        {
            "visual": "figure_call_semantics_w325",
            "responsibility": (
                "distinguish request attempts, executed evidence calls, and "
                "unsupported request attempts"
            ),
            "claim_supported": "W325-C1;W325-C2",
            "must_not_claim": "three-to-one executed detector inference",
            "source_data": "call_semantics_table_w325.csv",
        },
    ]
    _write_csv(output / "visual_contract_w325.csv", visual_contract)
    if ready:
        _plot_latency_ci(main_table, output)
        _plot_call_semantics(main_table, output)
    figure_map = [
        {
            "figure": item["visual"],
            "source_data": item["source_data"],
            "source_data_sha256": _sha256_file(output / item["source_data"]),
            "caption_boundary": item["must_not_claim"],
        }
        for item in visual_contract
    ]
    _write_csv(output / "figure_source_map_w325.csv", figure_map)
    _dump(output / "frozen_hashes_before_w325.json", _safe_hashes())
    report = {
        "status": (
            "ready_for_w325_evidence_pack_finalization"
            if ready
            else "blocked_w325_source_or_numeric_consistency_gate"
        ),
        "experiment": EXPERIMENT,
        "source_status_gates": source_status_gates,
        "per_source_numeric_gates": source_gates,
        "source_datasets": main_table["dataset"].tolist(),
        "statistical_source_count": int(len(main_table)),
        "gotham_development_context_included": True,
        "paper_reported_external_metrics_included": False,
        "classification_improvement_claim_supported": False,
        "latency_scope": (
            "governance replay, not production serving or full detector-stack "
            "latency"
        ),
        "request_semantics_explicit": True,
        "executed_evidence_call_semantics_explicit": True,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "evidence_pack_build_report.json", report)
    _render_docs(report, main_table, Path(document), Path(document_cn))
    return report


def _artifact_manifest(output: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_manifest.json", "acceptance_report.json"}
    return [
        {
            "artifact": path.name,
            "relative_path": path.relative_to(output).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def finalize_routing_efficiency_evidence_pack_w325(
    output_dir: str | Path = DEFAULT_OUTPUT,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    build = _load(output / "evidence_pack_build_report.json")
    hashes_after = _safe_hashes()
    _dump(output / "frozen_hashes_after_w325.json", hashes_after)
    hashes_unchanged = _hash_snapshot_equal(
        _load(output / "frozen_hashes_before_w325.json"),
        hashes_after,
    )
    expected_artifacts = [
        "main_efficiency_table_w325.csv",
        "latency_bootstrap_table_w325.csv",
        "call_semantics_table_w325.csv",
        "gotham_development_context_table_w325.csv",
        "claim_boundary_table_w325.csv",
        "source_traceability_w325.csv",
        "visual_contract_w325.csv",
        "figure_source_map_w325.csv",
        "figure_p95_latency_bootstrap_w325.png",
        "figure_p95_latency_bootstrap_w325.pdf",
        "figure_p95_latency_bootstrap_w325.svg",
        "figure_call_semantics_w325.png",
        "figure_call_semantics_w325.pdf",
        "figure_call_semantics_w325.svg",
    ]
    gates = {
        "build_ready": build.get("status")
        == "ready_for_w325_evidence_pack_finalization",
        "all_expected_artifacts_present": all(
            (output / name).is_file() for name in expected_artifacts
        ),
        "paper_reported_external_metrics_excluded": build.get(
            "paper_reported_external_metrics_included"
        )
        is False,
        "classification_improvement_not_claimed": build.get(
            "classification_improvement_claim_supported"
        )
        is False,
        "request_and_execution_semantics_explicit": (
            build.get("request_semantics_explicit") is True
            and build.get("executed_evidence_call_semantics_explicit") is True
        ),
        "frozen_hashes_unchanged": hashes_unchanged,
        "fake_metric_count_zero": build.get("fake_metric_count") == 0,
        "promoted_runtime_not_created": build.get(
            "promoted_runtime_created"
        )
        is False,
        "runtime_safe_v3_0_remains_default": build.get(
            "runtime_safe_v3_0_remains_default"
        )
        is True,
        "tests_passed": bool(tests_passed),
    }
    passed = all(gates.values())
    _dump(output / "artifact_manifest.json", _artifact_manifest(output))
    report = {
        "status": (
            "passed_w325_publication_routing_efficiency_evidence_pack"
            if passed
            else "failed_w325_evidence_or_boundary_gate"
        ),
        "build_report": build,
        "acceptance_gates": gates,
        "safe_claims": [
            (
                "Across three frozen source lanes, capability-aware routing "
                "reduced request attempts from 3.0 to 1.0 and unsupported "
                "attempts from 2.0 to 0.0."
            ),
            (
                "Repeated counterbalanced governance replay produced strictly "
                "positive paired-bootstrap p95 latency-reduction intervals on "
                "all three W324 source lanes."
            ),
            (
                "The replay preserved classification metrics, Fusion verdicts, "
                "and fixed in-domain OOD signatures."
            ),
        ],
        "forbidden_claims": [
            "Capability routing improved Accuracy, Macro-F1, or recall",
            "Executed detector inference calls fell from three to one",
            "W325 measured production end-to-end serving latency",
            "W325 validated external OOD detection accuracy",
            "W325 promoted or replaced runtime_safe_v3_0",
        ],
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "acceptance_report.json", report)
    _dump(
        output / "negative_results.json",
        []
        if passed
        else [
            {
                "experiment_id": EXPERIMENT,
                "failure_type": "evidence_pack_or_claim_boundary_gate_failed",
                "failed_gates": [
                    key for key, value in gates.items() if not value
                ],
                "fake_metric_count": 0,
                "runtime_modified": False,
                "final_status": "not_released",
            }
        ],
    )
    return report
