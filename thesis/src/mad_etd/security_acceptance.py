from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import build_default_engine
from .io import load_flow_records, write_report
from .paper_evaluation import hash_artifact_paths
from .reporter import ReporterAgent
from .runtime_profiles import (
    RuntimeProfile,
    load_runtime_profile,
    profile_artifact_paths,
    profile_engine_kwargs,
    runtime_profile_sha256,
)


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
REQUIRED_AUDIT_EVENTS = {
    "FLOW_VALIDATED",
    "FIELD_POLICY_APPLIED",
    "RELIABILITY_ASSESSED",
    "POLICY_GUARD_EVALUATION",
    "AGENT_EVIDENCE",
    "FINAL_FUSION",
    "REPORT_GENERATED",
}


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _snapshot(report) -> dict[str, Any]:
    return {
        field: (
            getattr(report, field).value
            if hasattr(getattr(report, field), "value")
            else getattr(report, field)
        )
        for field in FUSION_OWNED_FIELDS
    }


def _fusion_ownership_violation(report, audit) -> bool:
    events = [
        event for event in audit.events if event.event_type == "FINAL_FUSION"
    ]
    if len(events) != 1:
        return True
    final = events[0].output_summary
    return any(final.get(field) != _snapshot(report)[field] for field in FUSION_OWNED_FIELDS)


def _ood_override(audit, report) -> bool:
    audited = {
        event.output_summary.get("agent_name"): (
            event.output_summary.get("distribution_shift_score"),
            event.output_summary.get("distribution_shift_raw_score"),
            event.output_summary.get("distribution_shift_level"),
            event.output_summary.get("model_reliability"),
            event.output_summary.get("abstained"),
        )
        for event in audit.events
        if event.event_type == "AGENT_EVIDENCE"
        and event.output_summary.get("agent_name")
        in {"StatsDetectorAgent", "TemporalBehaviorAgent"}
    }
    reported = {
        item.agent_name: (
            item.distribution_shift_score,
            item.distribution_shift_raw_score,
            item.distribution_shift_level,
            item.model_reliability,
            item.abstained,
        )
        for item in report.agent_results
        if item.agent_name in {"StatsDetectorAgent", "TemporalBehaviorAgent"}
    }
    return audited != reported


def _paper_v1_profile_check(
    records,
    *,
    config_dir: str | Path,
) -> dict[str, Any]:
    profile = load_runtime_profile(
        "paper_v1_reproduction",
        config_dir=config_dir,
    )
    profiled = build_default_engine(
        **profile_engine_kwargs(profile),
        max_workers=1,
    )
    explicit = build_default_engine(
        detector_backend="learned",
        model_dir="data/models/ustc_tfc2016/v1",
        ood_policy="hybrid",
        ood_gate_dir="data/models/ustc_tfc2016/ood_v1",
        field_audit_mode="legacy",
        max_workers=1,
    )
    matches = 0
    for record in records:
        profiled_report, _ = profiled.analyze(record)
        explicit_report, _ = explicit.analyze(record)
        matches += int(_snapshot(profiled_report) == _snapshot(explicit_report))

    paper_acceptance_path = Path(
        "data/runs/future_framework/paper_v1/acceptance_report.json"
    )
    manifest_matches = False
    if paper_acceptance_path.exists():
        payload = json.loads(paper_acceptance_path.read_text(encoding="utf-8"))
        runs = list((payload.get("runs") or {}).values())
        manifest_matches = bool(runs) and all(
            run.get("detector_backend") == "learned"
            and run.get("ood_policy") == "hybrid"
            for run in runs
        )
    return {
        "profile": profile.name,
        "profile_sha256": runtime_profile_sha256(profile),
        "sample_count": len(records),
        "verdict_invariance_rate": matches / max(len(records), 1),
        "accepted_paper_manifest_matches": manifest_matches,
        "artifact_hashes": hash_artifact_paths(profile_artifact_paths(profile)),
    }


def run_v27_security_acceptance(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    profile_name: str = "runtime_safe_v2_7",
    config_dir: str | Path = "data/configs",
    limit: int = 100,
    document_path: str | Path | None = None,
) -> dict[str, Any]:
    profile = load_runtime_profile(profile_name, config_dir=config_dir)
    if not profile.acceptance_allowed or not profile.promoted:
        raise ValueError("formal acceptance requires a promoted acceptance profile")
    if profile.field_audit_mode != "strict" or not profile.field_audit_required:
        raise ValueError("formal acceptance requires strict FieldAudit")

    records = load_flow_records(input_path)[:limit]
    if not records:
        raise ValueError("security acceptance requires at least one flow")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    artifact_paths = profile_artifact_paths(profile)
    hashes_before = hash_artifact_paths(artifact_paths)
    _dump(target / "frozen_artifact_hashes_before.json", hashes_before)
    engine = build_default_engine(
        **profile_engine_kwargs(profile),
        formal_acceptance=True,
        max_workers=1,
    )
    reporter = ReporterAgent()

    blocked = 0
    unknown_execution = 0
    fusion = 0
    ood = 0
    incomplete_audit = 0
    illegal_verdict_execution = 0
    unknown_audited = 0
    for record in records:
        report, audit = engine.analyze(
            record,
            audit_path=target / "audits" / f"{record.trace_id}.jsonl",
        )
        write_report(
            report,
            target / "reports" / f"{record.trace_id}.json",
            markdown=reporter.to_markdown(report),
        )
        event_types = {event.event_type for event in audit.events}
        incomplete_audit += int(not REQUIRED_AUDIT_EVENTS.issubset(event_types))
        field_event = next(
            event
            for event in audit.events
            if event.event_type == "FIELD_POLICY_APPLIED"
        )
        unknown = set(field_event.output_summary.get("unknown_fields", []))
        visible = set(
            field_event.output_summary.get("detector_visible_fields", [])
        )
        unknown_audited += len(unknown)
        unknown_execution += len(unknown & visible)
        blocked += sum(
            bool(
                event.input_summary.get("blocked_field_intersection", [])
            )
            for event in audit.events
            if event.event_type == "AGENT_EVIDENCE"
        )
        fusion += int(_fusion_ownership_violation(report, audit))
        ood += int(_ood_override(audit, report))
        illegal_verdict_execution += sum(
            bool(
                event.output_summary.get("illegal_verdict_execution_count", 0)
            )
            for event in audit.events
        )

    hashes_after = hash_artifact_paths(artifact_paths)
    _dump(target / "frozen_artifact_hashes_after.json", hashes_after)
    paper_check = _paper_v1_profile_check(records, config_dir=config_dir)
    frozen_unchanged = hashes_before == hashes_after
    locked_leakage = bool(
        profile.selection_uses_test_or_external
        or profile.cipherspectrum_locked_test_used_for_selection
    )
    failures = []
    checks = {
        "strict_field_audit": (
            profile.field_audit_required
            and profile.field_audit_mode == "strict"
        ),
        "unknown_field_execution_is_zero": unknown_execution == 0,
        "blocked_field_violation_is_zero": blocked == 0,
        "fusion_ownership_violation_is_zero": fusion == 0,
        "ood_override_is_zero": ood == 0,
        "illegal_verdict_execution_is_zero": illegal_verdict_execution == 0,
        "audit_completion_is_one": incomplete_audit == 0,
        "frozen_artifacts_unchanged": frozen_unchanged,
        "locked_test_leakage_is_zero": not locked_leakage,
        "profile_is_promoted": profile.promoted,
        "automatic_training_is_false": not profile.automatic_training,
        "automatic_deployment_is_false": not profile.automatic_deployment,
        "paper_v1_reproduction_invariant": (
            paper_check["verdict_invariance_rate"] == 1.0
            and paper_check["accepted_paper_manifest_matches"]
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    manifest = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_7_security_acceptance",
        "runtime_profile": profile.model_dump(mode="json"),
        "runtime_profile_sha256": runtime_profile_sha256(profile),
        "input_path": Path(input_path).as_posix(),
        "sample_count": len(records),
        "selection_performed": False,
        "test_or_external_used_for_selection": False,
        "cipherspectrum_locked_test_used_for_selection": False,
    }
    _dump(target / "run_manifest.json", manifest)
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_7_security_acceptance",
        "acceptance_status": "passed" if not failures else "failed",
        "runtime_profile": profile.name,
        "runtime_profile_sha256": runtime_profile_sha256(profile),
        "sample_count": len(records),
        "failures": failures,
        "checks": checks,
        "unknown_field_audit_count": unknown_audited,
        "unknown_field_execution_count": unknown_execution,
        "blocked_field_violation_count": blocked,
        "fusion_ownership_violation_count": fusion,
        "ood_override_count": ood,
        "illegal_verdict_execution_count": illegal_verdict_execution,
        "audit_completion_rate": 1.0 - incomplete_audit / len(records),
        "frozen_artifacts_unchanged": frozen_unchanged,
        "automatic_training": False,
        "automatic_deployment": False,
        "test_or_external_used_for_selection": False,
        "cipherspectrum_locked_test_used_for_selection": False,
        "paper_v1_reproduction": paper_check,
    }
    _dump(target / "acceptance_report.json", report)

    if document_path is not None:
        document = Path(document_path)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(
            "\n".join(
                [
                    "# MAD-ETD v2.7 Security Acceptance",
                    "",
                    f"- Status: `{report['acceptance_status']}`",
                    f"- Runtime profile: `{profile.name}`",
                    f"- Samples: `{len(records)}`",
                    f"- Audit completion: `{report['audit_completion_rate']}`",
                    f"- Unknown-field executions: `{unknown_execution}`",
                    f"- Blocked-field violations: `{blocked}`",
                    f"- Fusion ownership violations: `{fusion}`",
                    f"- OOD overrides: `{ood}`",
                    f"- Illegal verdict executions: `{illegal_verdict_execution}`",
                    f"- Frozen artifacts unchanged: `{str(frozen_unchanged).lower()}`",
                    "- Automatic training: `false`",
                    "- Automatic deployment: `false`",
                    "",
                    "The accepted runtime uses strict fail-closed FieldAudit. "
                    "The legacy paper_v1 profile remains available only for "
                    "reproduction and is not an acceptance profile.",
                ]
            ),
            encoding="utf-8",
        )
    return report
