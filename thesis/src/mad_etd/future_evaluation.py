from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .engine import build_default_engine
from .memory import JsonlCaseMemory
from .schemas import FlowRecord, FutureFeatureFlags


FUTURE_EVALUATION_MODES = {
    "baseline": FutureFeatureFlags(),
    "memory_shadow": FutureFeatureFlags(memory=True),
    "planner_executor": FutureFeatureFlags(planner_executor=True),
    "deliberation_shadow": FutureFeatureFlags(deliberation=True),
    "reflection_critic": FutureFeatureFlags(
        reflection=True,
        audit_critic=True,
    ),
    "full_shadow": FutureFeatureFlags(
        memory=True,
        planner_executor=True,
        deliberation=True,
        reflection=True,
        audit_critic=True,
        shadow_mode=True,
    ),
    "full_active": FutureFeatureFlags(
        memory=True,
        planner_executor=True,
        deliberation=True,
        reflection=True,
        audit_critic=True,
        shadow_mode=False,
    ),
}

FUSION_OWNED_FIELDS = (
    "verdict",
    "confidence",
    "uncertainty",
    "conflict_score",
    "benign_support",
    "malicious_support",
    "distribution_shift_score",
    "severity",
    "need_escalation",
)


def _decision_snapshot(report) -> dict[str, Any]:
    return {
        field: (
            getattr(report, field).value
            if hasattr(getattr(report, field), "value")
            else getattr(report, field)
        )
        for field in FUSION_OWNED_FIELDS
    }


def evaluate_future_framework(
    records: Iterable[FlowRecord],
    output_dir: str | Path,
    *,
    modes: Iterable[str] = FUTURE_EVALUATION_MODES,
    detector_backend: str = "rule",
    model_dir: str | Path | None = None,
    ood_policy: str = "off",
    ood_gate_dir: str | Path | None = None,
    enable_rag: bool = False,
    knowledge_base_dir: str | Path = "knowledge_base",
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        raise ValueError("future framework evaluation requires records")
    selected_modes = list(modes)
    unknown = set(selected_modes) - set(FUTURE_EVALUATION_MODES)
    if unknown:
        raise ValueError(f"unknown future evaluation modes: {sorted(unknown)}")

    baseline_engine = build_default_engine(
        field_audit_mode="legacy",
        max_workers=1,
        detector_backend=detector_backend,
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
        enable_rag=enable_rag,
        knowledge_base_dir=knowledge_base_dir,
    )
    baseline_snapshots: dict[str, dict[str, Any]] = {}
    if "baseline" not in selected_modes:
        for record in rows:
            report, _ = baseline_engine.analyze(record)
            baseline_snapshots[record.sample_id] = _decision_snapshot(report)

    prediction_rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}
    for mode in selected_modes:
        flags = FUTURE_EVALUATION_MODES[mode]
        engine = build_default_engine(
            field_audit_mode="legacy",
            max_workers=1,
            detector_backend=detector_backend,
            model_dir=model_dir,
            ood_policy=ood_policy,
            ood_gate_dir=ood_gate_dir,
            enable_rag=enable_rag,
            knowledge_base_dir=knowledge_base_dir,
            future_flags=flags,
            memory_dir=target / "memory" / mode,
        )
        # Warm model loading and OS caches before latency measurement.
        engine.analyze(rows[0])
        verdicts: Counter[str] = Counter()
        calls: list[int] = []
        latencies: list[float] = []
        invariance: list[bool] = []
        memory_matches = 0
        deliberation_triggers = 0
        compliance_failures = 0
        finding_count = 0
        fallback_count = 0
        for record in rows:
            started = time.perf_counter()
            report, audit = engine.analyze(record)
            latency = (time.perf_counter() - started) * 1000
            snapshot = _decision_snapshot(report)
            if mode == "baseline":
                baseline_snapshots[record.sample_id] = snapshot
                invariant = True
            else:
                invariant = snapshot == baseline_snapshots[record.sample_id]
            artifacts = audit.future_artifacts
            verdicts[report.verdict.value] += 1
            calls.append(len(report.agent_results))
            latencies.append(latency)
            invariance.append(invariant)
            memory_matches += int(
                artifacts.memory_hint is not None
                and artifacts.memory_hint.status == "success"
            )
            deliberation_triggers += sum(
                item.status == "completed" for item in artifacts.deliberations
            )
            if artifacts.compliance is not None:
                compliance_failures += int(
                    artifacts.compliance.status != "compliant"
                )
                finding_count += len(artifacts.compliance.findings)
            fallback_count += sum(
                decision.source == "fallback"
                for decision in report.coordinator_decisions
            )
            prediction_rows.append(
                {
                    "mode": mode,
                    "sample_id": record.sample_id,
                    "verdict": report.verdict.value,
                    "confidence": report.confidence,
                    "uncertainty": report.uncertainty,
                    "agent_calls": calls[-1],
                    "latency_ms": latency,
                    "verdict_invariant": invariant,
                    "memory_status": (
                        artifacts.memory_hint.status
                        if artifacts.memory_hint
                        else "disabled"
                    ),
                    "deliberation_count": len(artifacts.deliberations),
                    "compliance_status": (
                        artifacts.compliance.status
                        if artifacts.compliance
                        else "disabled"
                    ),
                }
            )
        count = len(rows)
        mode_summaries[mode] = {
            "sample_count": count,
            "verdict_rates": {
                verdict: verdicts[verdict] / count
                for verdict in ("benign", "malicious", "suspicious", "unknown")
            },
            "average_agent_calls": fmean(calls),
            "average_latency_ms": fmean(latencies),
            "coverage": (
                verdicts["benign"] + verdicts["malicious"]
            )
            / count,
            "human_review_rate": (
                verdicts["suspicious"] + verdicts["unknown"]
            )
            / count,
            "verdict_invariance_rate": sum(invariance) / count,
            "memory_match_rate": memory_matches / count,
            "deliberation_trigger_rate": deliberation_triggers / count,
            "compliance_failure_rate": compliance_failures / count,
            "audit_finding_count": finding_count,
            "coordinator_fallback_count": fallback_count,
            "feature_flags": flags.model_dump(mode="json"),
        }

    with (target / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(prediction_rows[0]),
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    summary = {
        "schema_version": "1.0",
        "experiment": "future_framework_ablation",
        "metrics_scope": (
            "routing, audit, conflict, latency, and verdict invariance; "
            "not a new classifier training experiment"
        ),
        "modes": mode_summaries,
        "boundaries": {
            "memory_is_agent_evidence": False,
            "llm_is_classifier": False,
            "reflection_modifies_current_verdict": False,
            "critic_outputs_labels": False,
            "automatic_retraining": False,
        },
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate_memory_retrieval(
    reference_records: Iterable[FlowRecord],
    query_records: Iterable[FlowRecord],
    output_dir: str | Path,
    *,
    reference_groups: dict[str, str],
    query_groups: dict[str, str],
) -> dict[str, Any]:
    """Evaluate advisory retrieval with evaluator-only group annotations."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    memory = JsonlCaseMemory(target / "memory")
    engine = build_default_engine(field_audit_mode="legacy", max_workers=1)
    case_groups: dict[str, str] = {}
    stored = 0
    for record in reference_records:
        report, _ = engine.analyze(record)
        if report.verdict.value not in {"suspicious", "unknown"}:
            continue
        case = memory.append_case(
            record,
            report,
            source_dataset="memory_retrieval_reference",
        )
        case_groups[case.case_id] = reference_groups[record.sample_id]
        stored += 1
    rows: list[dict[str, Any]] = []
    matched = 0
    correct = 0
    for record in query_records:
        hint = memory.retrieve(record, top_k=3, min_score=0.0)
        top_case = hint.similar_case_refs[0] if hint.similar_case_refs else None
        predicted_group = case_groups.get(top_case, "")
        expected_group = query_groups[record.sample_id]
        is_match = bool(top_case)
        is_correct = is_match and predicted_group == expected_group
        matched += int(is_match)
        correct += int(is_correct)
        rows.append(
            {
                "sample_id": record.sample_id,
                "expected_group": expected_group,
                "predicted_group": predicted_group,
                "top_case": top_case or "",
                "top_score": (
                    hint.retrieval_scores[0] if hint.retrieval_scores else ""
                ),
                "matched": is_match,
                "top1_correct": is_correct,
            }
        )
    with (target / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    total = len(rows)
    result = {
        "schema_version": "1.0",
        "experiment": "case_memory_retrieval",
        "stored_case_count": stored,
        "query_count": total,
        "match_rate": matched / total if total else 0,
        "top1_group_accuracy": correct / matched if matched else None,
        "evaluator_group_used_for_retrieval": False,
        "locked_benchmark_used": False,
        "verdict_fields_in_memory_hint": False,
        "automatic_retraining": False,
    }
    (target / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def write_future_framework_acceptance(
    experiment_dirs: dict[str, str | Path],
    output_path: str | Path,
    markdown_path: str | Path,
    *,
    test_count: int,
) -> dict[str, Any]:
    experiments: dict[str, Any] = {}
    shadow_invariance: list[float] = []
    compliance_failures = 0
    for name, directory in experiment_dirs.items():
        source = Path(directory) / "summary.json"
        summary = json.loads(source.read_text(encoding="utf-8"))
        experiments[name] = {
            # Keep caller-supplied paths instead of resolving through the
            # process CWD.  On Windows, ``conda run`` may decode a non-ASCII
            # working directory with the active ANSI code page, which would
            # permanently write mojibake into otherwise valid JSON.
            "summary_path": source.as_posix(),
            "sample_count": next(iter(summary["modes"].values()))[
                "sample_count"
            ],
            "modes": summary["modes"],
        }
        for mode, values in summary["modes"].items():
            if values["feature_flags"].get("shadow_mode") and mode != "baseline":
                shadow_invariance.append(values["verdict_invariance_rate"])
            compliance_failures += int(
                values["compliance_failure_rate"]
                * values["sample_count"]
            )
    payload = {
        "schema_version": "1.0",
        "framework": "MAD-ETD future research prototype",
        "acceptance_status": (
            "passed"
            if (
                test_count > 0
                and shadow_invariance
                and min(shadow_invariance) == 1.0
                and compliance_failures == 0
            )
            else "failed"
        ),
        "test_count": test_count,
        "minimum_shadow_verdict_invariance": min(shadow_invariance),
        "compliance_failure_count": compliance_failures,
        "implemented_directions": [
            "CaseMemory and Human Feedback",
            "Planner-Executor Coordinator",
            "Conflict-Aware Deliberation",
            "Self-Reflection",
            "Audit Critic",
            "Human-reviewed Learning Queue",
        ],
        "experiments": experiments,
        "boundaries": {
            "fusion_only_final_verdict": True,
            "memory_is_advisory": True,
            "rag_is_explanatory_only": True,
            "ood_can_be_overridden": False,
            "automatic_retraining": False,
            "new_detector_added": False,
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Future framework final acceptance",
        "",
        f"- Status: `{payload['acceptance_status']}`",
        f"- Full tests: `{test_count} passed`",
        "- Minimum shadow-mode verdict invariance: "
        f"`{payload['minimum_shadow_verdict_invariance']:.6f}`",
        f"- Compliance failures: `{compliance_failures}`",
        "- Automatic retraining: `false`",
        "- New detector added: `false`",
        "",
        "## Experiment inventory",
        "",
        "| Experiment | Samples | Modes |",
        "|---|---:|---:|",
    ]
    for name, values in experiments.items():
        lines.append(
            f"| {name} | {values['sample_count']} | {len(values['modes'])} |"
        )
    lines.extend(
        [
            "",
            "The acceptance covers the guarded research prototype and small "
            "integration runs. It does not claim production readiness or "
            "full-dataset performance improvement.",
            "",
        ]
    )
    markdown = Path(markdown_path)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return payload
