from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import threading
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .engine import build_default_engine
from .future_evaluation import FUSION_OWNED_FIELDS
from .memory import JsonlCaseMemory, safe_behavior_summary
from .paper_evaluation import (
    REQUIRED_AUDIT_EVENTS,
    _blocked_violation,
    hash_artifact_paths,
    iter_selection_records,
    sha256_file,
)
from .schemas import DetectionReport, FlowRecord, HumanFeedbackRecord


HITL_OUTCOMES = {"benign", "malicious", "suspicious", "unknown"}
RESPONSE_FIELDS = (
    "reviewer_outcome",
    "needs_more_evidence",
    "hint_relevant",
    "hint_accepted",
    "review_seconds",
    "reviewer_rationale",
)


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _priority(seed: int, *values: str) -> str:
    return hashlib.sha256(
        ":".join((str(seed), *values)).encode("utf-8")
    ).hexdigest()


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fusion_snapshot(report: DetectionReport) -> dict[str, Any]:
    return {
        field: (
            getattr(report, field).value
            if hasattr(getattr(report, field), "value")
            else getattr(report, field)
        )
        for field in FUSION_OWNED_FIELDS
    }


def _fusion_ownership_violation(report: DetectionReport, audit) -> bool:
    final_events = [
        event for event in audit.events if event.event_type == "FINAL_FUSION"
    ]
    if not final_events:
        return True
    output = final_events[-1].output_summary
    snapshot = _fusion_snapshot(report)
    return any(output.get(field) != snapshot[field] for field in FUSION_OWNED_FIELDS)


def _ood_override(audit, report: DetectionReport) -> bool:
    audit_signatures: dict[str, tuple[Any, ...]] = {}
    for event in audit.events:
        if event.event_type != "AGENT_EVIDENCE":
            continue
        output = event.output_summary
        agent = output.get("agent_name")
        if agent not in {"StatsDetectorAgent", "TemporalBehaviorAgent"}:
            continue
        audit_signatures[agent] = (
            output.get("distribution_shift_score"),
            output.get("distribution_shift_raw_score"),
            output.get("distribution_shift_level"),
            output.get("model_reliability"),
            output.get("abstained"),
        )
    report_signatures = {
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
    return audit_signatures != report_signatures


def _artifact_paths(
    model_dir: str | Path,
    ood_gate_dir: str | Path,
) -> dict[str, str | Path]:
    source_dir = Path(__file__).parent
    paths: dict[str, str | Path] = {
        "detector_models": model_dir,
        "ood_gates": ood_gate_dir,
        "fusion_source": source_dir / "fusion.py",
        "rag_source": source_dir / "knowledge.py",
        "reporter_source": source_dir / "reporter.py",
        "memory_source": source_dir / "memory.py",
    }
    knowledge_base = Path("knowledge_base")
    if knowledge_base.exists():
        paths["knowledge_base"] = knowledge_base
    return paths


def _allocate(counts: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0 or target > sum(counts.values()):
        raise ValueError("invalid stratified HITL target")
    total = sum(counts.values())
    exact = {key: target * value / total for key, value in counts.items()}
    allocation = {
        key: min(value, int(exact[key]))
        for key, value in counts.items()
    }
    remaining = target - sum(allocation.values())
    ranked = sorted(
        counts,
        key=lambda key: (
            exact[key] - allocation[key],
            counts[key],
            key,
        ),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in ranked:
            if allocation[key] >= counts[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("could not complete stratified allocation")
    return allocation


def _candidate_stratum(row: dict[str, str]) -> str:
    binary = str(row.get("stratum", "")).split("|", 1)[0] or "unlabeled"
    verdict = str(row.get("verdict", "")).strip().lower() or "unknown"
    return f"{binary}|{verdict}"


def _select_candidates(
    candidate_index_path: str | Path,
    *,
    source_ids: set[str],
    total_count: int,
    phase1_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    with Path(candidate_index_path).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("sample_id") in source_ids
            and str(row.get("verdict", "")).lower()
            in {"suspicious", "unknown"}
        ]
    unique = {row["sample_id"] for row in rows}
    if len(unique) != len(rows):
        raise ValueError("candidate index contains duplicate sample IDs")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_candidate_stratum(row)].append(row)
    allocation = _allocate(
        {key: len(value) for key, value in grouped.items()},
        total_count,
    )
    selected_by_stratum: dict[str, list[dict[str, str]]] = {}
    for stratum, items in grouped.items():
        selected_by_stratum[stratum] = sorted(
            items,
            key=lambda row: _priority(seed, stratum, row["sample_id"]),
        )[: allocation[stratum]]

    interleaved: list[dict[str, str]] = []
    offsets = {key: 0 for key in selected_by_stratum}
    while len(interleaved) < total_count:
        progressed = False
        for stratum in sorted(
            selected_by_stratum,
            key=lambda key: _priority(seed, "stratum-order", key),
        ):
            offset = offsets[stratum]
            items = selected_by_stratum[stratum]
            if offset >= len(items):
                continue
            interleaved.append(items[offset])
            offsets[stratum] += 1
            progressed = True
        if not progressed:
            break
    if len(interleaved) != total_count:
        raise RuntimeError("HITL selection did not reach the requested count")

    phase2_rows = interleaved[phase1_count:]
    condition_by_id: dict[str, str] = {}
    phase2_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in phase2_rows:
        phase2_groups[_candidate_stratum(row)].append(row)
    visible_target = len(phase2_rows) // 2
    for stratum, items in phase2_groups.items():
        ordered = sorted(
            items,
            key=lambda row: _priority(
                seed, "phase2-condition", stratum, row["sample_id"]
            ),
        )
        start_visible = int(
            _priority(seed, "phase2-start", stratum)[-1], 16
        ) % 2 == 0
        for index, row in enumerate(ordered):
            visible = (index % 2 == 0) == start_visible
            condition_by_id[row["sample_id"]] = (
                "hint_visible" if visible else "hint_hidden"
            )
    visible = [
        row
        for row in phase2_rows
        if condition_by_id[row["sample_id"]] == "hint_visible"
    ]
    hidden = [
        row
        for row in phase2_rows
        if condition_by_id[row["sample_id"]] == "hint_hidden"
    ]
    if len(visible) != visible_target:
        source = visible if len(visible) > visible_target else hidden
        replacement = "hint_hidden" if len(visible) > visible_target else "hint_visible"
        difference = abs(len(visible) - visible_target)
        for row in sorted(
            source,
            key=lambda item: _priority(
                seed, "phase2-rebalance", item["sample_id"]
            ),
        )[:difference]:
            condition_by_id[row["sample_id"]] = replacement

    selected: list[dict[str, Any]] = []
    for index, row in enumerate(interleaved):
        phase = 1 if index < phase1_count else 2
        selected.append(
            {
                "sample_id": row["sample_id"],
                "phase": phase,
                "condition": (
                    "hint_hidden"
                    if phase == 1
                    else condition_by_id[row["sample_id"]]
                ),
                "private_stratum": _candidate_stratum(row),
                "selection_priority": _priority(
                    seed, "selected", row["sample_id"]
                ),
            }
        )
    return selected


def _task_from_report(
    report: DetectionReport,
    *,
    task_id: str,
    phase: int,
    condition: str,
    memory_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _fusion_snapshot(report)
    return {
        "task_id": task_id,
        "phase": phase,
        "condition": condition,
        "sample_id": report.sample_id,
        "system_verdict": report.verdict.value,
        "system_confidence": report.confidence,
        "system_uncertainty": report.uncertainty,
        "system_conflict_score": report.conflict_score,
        "distribution_shift_score": report.distribution_shift_score,
        "severity": report.severity,
        "need_escalation": report.need_escalation,
        "participating_agents": report.participating_agents,
        "main_evidence": report.main_evidence,
        "memory_hint": memory_hint,
        "fusion_snapshot_sha256": _json_hash(snapshot),
        "report_sha256": hashlib.sha256(
            report.model_dump_json().encode("utf-8")
        ).hexdigest(),
        **{field: "" for field in RESPONSE_FIELDS},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text)


def _write_responses_csv(
    path: Path,
    tasks: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = dict(task)
        row["participating_agents"] = json.dumps(
            row["participating_agents"], ensure_ascii=False
        )
        row["main_evidence"] = json.dumps(
            row["main_evidence"], ensure_ascii=False
        )
        row["memory_hint"] = json.dumps(
            row["memory_hint"], ensure_ascii=False
        )
        row.update(responses.get(task["task_id"], {}))
        rows.append(row)
    if not rows:
        raise ValueError("cannot write an empty HITL response file")
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, "\ufeff" + stream.getvalue())


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_hitl_pilot(
    input_path: str | Path,
    source_selection_manifest: str | Path,
    candidate_index_path: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    ood_gate_dir: str | Path,
    total_count: int = 200,
    phase1_count: int = 80,
    seed: int = 42,
    max_workers: int = 4,
) -> dict[str, Any]:
    if total_count != 200 or phase1_count != 80:
        raise ValueError("HITL pilot v1 is frozen at 200 total / 80 phase-1")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        Path(source_selection_manifest).read_text(encoding="utf-8")
    )
    source_ids = set(source_manifest["sample_ids"])
    selected = _select_candidates(
        candidate_index_path,
        source_ids=source_ids,
        total_count=total_count,
        phase1_count=phase1_count,
        seed=seed,
    )
    selection_payload = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "source_dataset": "USTC-TFC2016",
        "source_split": "validation",
        "strategy": "frozen_candidate_stratified_hash_selection",
        "seed": seed,
        "sample_count": len(selected),
        "phase1_count": phase1_count,
        "phase2_count": total_count - phase1_count,
        "phase2_hint_visible_count": sum(
            row["condition"] == "hint_visible" for row in selected
        ),
        "phase2_hint_hidden_count": sum(
            row["phase"] == 2 and row["condition"] == "hint_hidden"
            for row in selected
        ),
        "source_selection_manifest": Path(
            source_selection_manifest
        ).as_posix(),
        "source_selection_manifest_sha256": sha256_file(
            source_selection_manifest
        ),
        "candidate_index": Path(candidate_index_path).as_posix(),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "labels_exposed_to_reviewer": False,
        "offline_oracle_memory_reused": False,
        "cipherspectrum_locked_test_used": False,
        "selections": selected,
        "sample_ids": [row["sample_id"] for row in selected],
    }
    _dump(target / "selection_manifest.json", selection_payload)
    artifacts = _artifact_paths(model_dir, ood_gate_dir)
    hashes_before = hash_artifact_paths(artifacts)
    _dump(target / "frozen_artifact_hashes_before.json", hashes_before)

    engine = build_default_engine(
        field_audit_mode="legacy",
        detector_backend="learned",
        model_dir=model_dir,
        ood_policy="hybrid",
        ood_gate_dir=ood_gate_dir,
        max_workers=max_workers,
    )
    reports: dict[str, DetectionReport] = {}
    audits: dict[str, dict[str, Any]] = {}
    for record in iter_selection_records(
        input_path, target / "selection_manifest.json"
    ):
        report, audit = engine.analyze(record)
        if report.verdict.value not in {"suspicious", "unknown"}:
            raise RuntimeError(
                f"frozen HITL candidate changed eligibility: {record.sample_id}"
            )
        reports[record.sample_id] = report
        audits[record.sample_id] = {
            "sample_id": record.sample_id,
            "audit_event_count": len(audit.events),
            "final_fusion_count": sum(
                event.event_type == "FINAL_FUSION" for event in audit.events
            ),
        }
    if len(reports) != total_count:
        raise RuntimeError(f"HITL reports generated {len(reports)}/{total_count}")
    private_dir = target / "private"
    _write_jsonl(
        private_dir / "system_reports.jsonl",
        [
            reports[row["sample_id"]].model_dump(mode="json")
            for row in selected
        ],
    )
    _write_jsonl(
        private_dir / "audit_summaries.jsonl",
        [audits[row["sample_id"]] for row in selected],
    )
    phase1_rows = [row for row in selected if row["phase"] == 1]
    phase1_tasks = [
        _task_from_report(
            reports[row["sample_id"]],
            task_id=f"hitl-p1-{index:03d}",
            phase=1,
            condition="hint_hidden",
        )
        for index, row in enumerate(phase1_rows)
    ]
    _write_jsonl(target / "phase1_tasks.jsonl", phase1_tasks)
    _write_responses_csv(
        target / "phase1_responses.csv", phase1_tasks, {}
    )
    _dump(
        target / "phase1_manifest.json",
        {
            "schema_version": "1.0",
            "experiment": "hitl_pilot_v1",
            "phase": 1,
            "status": "awaiting_human_review",
            "sample_count": len(phase1_tasks),
            "condition_counts": {"hint_hidden": len(phase1_tasks)},
            "labels_exposed_to_reviewer": False,
            "memory_used": False,
            "single_reviewer_pilot": True,
            "automatic_retraining": False,
        },
    )
    hashes_after = hash_artifact_paths(artifacts)
    _dump(target / "frozen_artifact_hashes_after_prepare.json", hashes_after)
    payload = {
        "status": "phase1_ready",
        "sample_count": total_count,
        "phase1_task_count": len(phase1_tasks),
        "phase2_reserved_count": total_count - phase1_count,
        "phase2_hint_visible_count": selection_payload[
            "phase2_hint_visible_count"
        ],
        "phase2_hint_hidden_count": selection_payload[
            "phase2_hint_hidden_count"
        ],
        "frozen_artifacts_unchanged": hashes_before == hashes_after,
        "labels_exposed_to_reviewer": False,
    }
    _dump(target / "prepare_report.json", payload)
    return payload


def prepare_hitl_phase2(
    input_path: str | Path,
    output_dir: str | Path,
    memory_dir: str | Path,
    *,
    phase1_import_report: str | Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    import_report = json.loads(
        Path(phase1_import_report).read_text(encoding="utf-8")
    )
    if (
        import_report.get("status") != "completed"
        or int(import_report.get("phase", 0)) != 1
        or int(import_report.get("review_count", 0)) != 80
    ):
        raise ValueError("phase 2 requires 80 completed phase-1 reviews")
    memory = JsonlCaseMemory(memory_dir)
    if len(list(memory.cases())) != 80 or len(list(memory.feedback())) != 80:
        raise ValueError("phase 2 requires exactly 80 phase-1 memory records")
    selection = json.loads(
        (target / "selection_manifest.json").read_text(encoding="utf-8")
    )
    phase2_rows = [
        row for row in selection["selections"] if int(row["phase"]) == 2
    ]
    if len(phase2_rows) != 120:
        raise ValueError("frozen phase-2 selection must contain 120 samples")
    reports = {
        item["sample_id"]: DetectionReport.model_validate(item)
        for item in _load_jsonl(target / "private" / "system_reports.jsonl")
    }
    phase2_manifest = {
        "schema_version": "1.0",
        "sample_ids": [row["sample_id"] for row in phase2_rows],
    }
    private_manifest = target / "private" / "phase2_selection.json"
    _dump(private_manifest, phase2_manifest)
    records = {
        record.sample_id: record
        for record in iter_selection_records(input_path, private_manifest)
    }
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(phase2_rows):
        record = records[row["sample_id"]]
        condition = row["condition"]
        hint = (
            memory.retrieve(record).model_dump(mode="json")
            if condition == "hint_visible"
            else None
        )
        tasks.append(
            _task_from_report(
                reports[row["sample_id"]],
                task_id=f"hitl-p2-{index:03d}",
                phase=2,
                condition=condition,
                memory_hint=hint,
            )
        )
    if Counter(task["condition"] for task in tasks) != {
        "hint_visible": 60,
        "hint_hidden": 60,
    }:
        raise RuntimeError("phase-2 condition allocation is not 60/60")
    _write_jsonl(target / "phase2_tasks.jsonl", tasks)
    _write_responses_csv(target / "phase2_responses.csv", tasks, {})
    payload = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "phase": 2,
        "status": "awaiting_human_review",
        "sample_count": len(tasks),
        "condition_counts": dict(Counter(task["condition"] for task in tasks)),
        "labels_exposed_to_reviewer": False,
        "memory_used": True,
        "memory_case_count": 80,
        "memory_feedback_count": 80,
        "single_reviewer_pilot": True,
        "automatic_retraining": False,
    }
    _dump(target / "phase2_manifest.json", payload)
    return payload


def validate_hitl_response(
    task: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    outcome = str(response.get("reviewer_outcome", "")).strip().lower()
    if outcome not in HITL_OUTCOMES:
        raise ValueError(f"reviewer_outcome must be one of {sorted(HITL_OUTCOMES)}")
    rationale = str(response.get("reviewer_rationale", "")).strip()
    if not rationale:
        raise ValueError("reviewer_rationale is required")
    try:
        seconds = float(response.get("review_seconds", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("review_seconds must be numeric") from exc
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("review_seconds must be positive")

    def required_bool(name: str) -> bool:
        value = response.get(name)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "1", "yes"}:
            return True
        if str(value).strip().lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"{name} must be boolean")

    normalized: dict[str, Any] = {
        "reviewer_outcome": outcome,
        "needs_more_evidence": required_bool("needs_more_evidence"),
        "review_seconds": round(seconds, 3),
        "reviewer_rationale": rationale,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if task["condition"] == "hint_visible":
        normalized["hint_relevant"] = required_bool("hint_relevant")
        normalized["hint_accepted"] = required_bool("hint_accepted")
    else:
        if response.get("hint_relevant") not in {None, ""}:
            raise ValueError("hint_relevant is forbidden for hidden-hint tasks")
        if response.get("hint_accepted") not in {None, ""}:
            raise ValueError("hint_accepted is forbidden for hidden-hint tasks")
        normalized["hint_relevant"] = ""
        normalized["hint_accepted"] = ""
    return normalized


class HitlResponseStore:
    def __init__(self, root: str | Path, phase: int) -> None:
        if phase not in {1, 2}:
            raise ValueError("HITL phase must be 1 or 2")
        self.root = Path(root)
        self.phase = phase
        self.tasks_path = self.root / f"phase{phase}_tasks.jsonl"
        self.responses_path = self.root / f"phase{phase}_responses.json"
        self.responses_csv_path = self.root / f"phase{phase}_responses.csv"
        self.tasks = _load_jsonl(self.tasks_path)
        self.task_index = {task["task_id"]: task for task in self.tasks}
        if len(self.task_index) != len(self.tasks):
            raise ValueError("HITL tasks contain duplicate task IDs")
        self._lock = threading.Lock()

    def responses(self) -> dict[str, dict[str, Any]]:
        if not self.responses_path.exists():
            return {}
        payload = json.loads(self.responses_path.read_text(encoding="utf-8"))
        return dict(payload.get("responses", {}))

    def public_state(self) -> dict[str, Any]:
        responses = self.responses()
        return {
            "phase": self.phase,
            "tasks": self.tasks,
            "responses": responses,
            "completed_count": len(responses),
            "total_count": len(self.tasks),
            "all_complete": len(responses) == len(self.tasks),
        }

    def save(self, task_id: str, response: dict[str, Any]) -> dict[str, Any]:
        if task_id not in self.task_index:
            raise ValueError("unknown HITL task")
        normalized = validate_hitl_response(
            self.task_index[task_id], response
        )
        with self._lock:
            responses = self.responses()
            responses[task_id] = normalized
            payload = {
                "schema_version": "1.0",
                "experiment": "hitl_pilot_v1",
                "phase": self.phase,
                "responses": responses,
            }
            _atomic_write_text(
                self.responses_path,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
            _write_responses_csv(
                self.responses_csv_path, self.tasks, responses
            )
        return normalized


HITL_REVIEW_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAD-ETD HITL Pilot</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d1321;color:#eef2ff;margin:0}
main{max-width:980px;margin:auto;padding:24px}
.card{background:#172033;border:1px solid #2b3958;border-radius:14px;padding:20px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.metric{background:#101827;padding:10px;border-radius:9px}
button{background:#58a6ff;color:#07111f;border:0;border-radius:8px;padding:10px 16px;font-weight:700;cursor:pointer}
button.secondary{background:#334155;color:#fff}
textarea{width:100%;min-height:110px;box-sizing:border-box;background:#0f172a;color:#fff;border:1px solid #475569;border-radius:8px;padding:10px}
select{background:#0f172a;color:#fff;border:1px solid #475569;border-radius:8px;padding:8px}
label{display:block;margin:10px 0}.muted{color:#a8b3cf}.evidence li{margin:7px 0}
.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.ok{color:#68d391}.warn{color:#f6ad55}
</style>
</head>
<body><main>
<h1>MAD-ETD 人工复核 Pilot</h1>
<div class="nav"><button class="secondary" onclick="move(-1)">上一例</button>
<button onclick="save()">保存并下一例</button><button class="secondary" onclick="move(1)">跳到下一例</button>
<span id="progress"></span><span id="status"></span></div>
<div id="task" class="card"></div>
</main>
<script>
let state, index=0, shownAt=Date.now(), accumulated=0;
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function load(){
 state=await (await fetch('/api/state')).json();
 const first=state.tasks.findIndex(t=>!state.responses[t.task_id]);
 index=first<0?0:first; render();
}
function current(){return state.tasks[index]}
function render(){
 const t=current(), r=state.responses[t.task_id]||{}; shownAt=Date.now(); accumulated=Number(r.review_seconds||0);
 const hint=t.condition==='hint_visible'?`<div class="card"><h3>MemoryHint（仅辅助）</h3><pre>${esc(JSON.stringify(t.memory_hint,null,2))}</pre>
 <label>Hint 是否相关？ <select id="hint_relevant"><option value="">请选择</option><option value="true">是</option><option value="false">否</option></select></label>
 <label>你是否采纳 Hint？ <select id="hint_accepted"><option value="">请选择</option><option value="true">是</option><option value="false">否</option></select></label></div>`:'';
 document.querySelector('#task').innerHTML=`<h2>${esc(t.task_id)} · ${esc(t.condition)}</h2>
 <div class="grid"><div class="metric">系统判定<br><b>${esc(t.system_verdict)}</b></div>
 <div class="metric">置信度<br><b>${Number(t.system_confidence).toFixed(4)}</b></div>
 <div class="metric">不确定度<br><b>${Number(t.system_uncertainty).toFixed(4)}</b></div>
 <div class="metric">冲突分数<br><b>${Number(t.system_conflict_score).toFixed(4)}</b></div>
 <div class="metric">分布漂移<br><b>${Number(t.distribution_shift_score).toFixed(4)}</b></div>
 <div class="metric">严重度<br><b>${esc(t.severity)}</b></div></div>
 <h3>主要证据</h3><ul class="evidence">${t.main_evidence.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>
 ${hint}<label>你的复核结论 <select id="outcome"><option value="">请选择</option>
 <option value="benign">benign</option><option value="malicious">malicious</option>
 <option value="suspicious">suspicious</option><option value="unknown">unknown</option></select></label>
 <label>是否需要更多证据？ <select id="more"><option value="">请选择</option><option value="true">是</option><option value="false">否</option></select></label>
 <label>复核理由<textarea id="rationale"></textarea></label>
 <p class="muted">人工结论只写入 CaseMemory 反馈，不修改当前 FusionResult，也不会自动训练。</p>`;
 setValue('outcome',r.reviewer_outcome); setValue('more',boolText(r.needs_more_evidence));
 setValue('hint_relevant',boolText(r.hint_relevant)); setValue('hint_accepted',boolText(r.hint_accepted));
 if(r.reviewer_rationale)document.querySelector('#rationale').value=r.reviewer_rationale;
 updateProgress();
}
function boolText(v){return typeof v==='boolean'?String(v):''}
function setValue(id,v){const e=document.getElementById(id);if(e&&v!==undefined)e.value=v||''}
function updateProgress(){
 document.querySelector('#progress').textContent=`${index+1}/${state.tasks.length}，已完成 ${Object.keys(state.responses).length}`;
}
function move(delta){index=Math.max(0,Math.min(state.tasks.length-1,index+delta));render()}
async function save(){
 const t=current(), visible=t.condition==='hint_visible';
 const payload={reviewer_outcome:document.querySelector('#outcome').value,
 needs_more_evidence:document.querySelector('#more').value,
 review_seconds:accumulated+(Date.now()-shownAt)/1000,
 reviewer_rationale:document.querySelector('#rationale').value,
 hint_relevant:visible?document.querySelector('#hint_relevant').value:'',
 hint_accepted:visible?document.querySelector('#hint_accepted').value:''};
 const res=await fetch('/api/response',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({task_id:t.task_id,response:payload})});
 const body=await res.json(); const status=document.querySelector('#status');
 if(!res.ok){status.className='warn';status.textContent=body.error;return}
 state.responses[t.task_id]=body.response;status.className='ok';status.textContent='已保存';
 const next=state.tasks.findIndex((x,i)=>i>index&&!state.responses[x.task_id]);
 if(next>=0)index=next;else if(index<state.tasks.length-1)index++;render();
}
load();
</script></body></html>"""


def _handler_factory(store: HitlResponseStore):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                data = HITL_REVIEW_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/state":
                self._json(store.public_state())
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/response":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                saved = store.save(
                    str(payload.get("task_id", "")),
                    dict(payload.get("response", {})),
                )
                self._json({"status": "saved", "response": saved})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve_hitl_review(
    output_dir: str | Path,
    *,
    phase: int,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    if host != "127.0.0.1":
        raise ValueError("HITL review server is restricted to 127.0.0.1")
    store = HitlResponseStore(output_dir, phase)
    server = ThreadingHTTPServer((host, port), _handler_factory(store))
    url = f"http://{host}:{server.server_port}/"
    print(f"MAD-ETD HITL review: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def import_hitl_phase(
    input_path: str | Path,
    output_dir: str | Path,
    memory_dir: str | Path,
    *,
    phase: int,
    reviewer: str,
    model_dir: str | Path,
    ood_gate_dir: str | Path,
    max_workers: int = 4,
) -> dict[str, Any]:
    target = Path(output_dir)
    store = HitlResponseStore(target, phase)
    state = store.public_state()
    expected = 80 if phase == 1 else 120
    if state["completed_count"] != expected:
        raise ValueError(
            f"phase {phase} has {state['completed_count']}/{expected} reviews"
        )
    selection = json.loads(
        (target / "selection_manifest.json").read_text(encoding="utf-8")
    )
    phase_ids = {
        row["sample_id"]
        for row in selection["selections"]
        if int(row["phase"]) == phase
    }
    private_manifest = target / "private" / f"phase{phase}_import_selection.json"
    _dump(
        private_manifest,
        {"schema_version": "1.0", "sample_ids": sorted(phase_ids)},
    )
    records = {
        record.sample_id: record
        for record in iter_selection_records(input_path, private_manifest)
    }
    artifact_paths = _artifact_paths(model_dir, ood_gate_dir)
    hashes_before = hash_artifact_paths(artifact_paths)
    _dump(
        target / f"phase{phase}_frozen_artifact_hashes_before_import.json",
        hashes_before,
    )
    engine = build_default_engine(
        field_audit_mode="legacy",
        detector_backend="learned",
        model_dir=model_dir,
        ood_policy="hybrid",
        ood_gate_dir=ood_gate_dir,
        max_workers=max_workers,
    )
    memory = JsonlCaseMemory(memory_dir)
    times: list[float] = []
    hint_relevance: list[bool] = []
    hint_acceptance: list[bool] = []
    ownership_violations = 0
    responses = state["responses"]
    validated_reports: dict[str, DetectionReport] = {}
    audit_complete_count = 0
    blocked_field_violation_count = 0
    fusion_ownership_violation_count = 0
    ood_override_count = 0
    for task in store.tasks:
        record = records[task["sample_id"]]
        report, audit = engine.analyze(record)
        if _json_hash(_fusion_snapshot(report)) != task["fusion_snapshot_sha256"]:
            ownership_violations += 1
            continue
        event_types = {event.event_type for event in audit.events}
        audit_complete_count += int(REQUIRED_AUDIT_EVENTS <= event_types)
        blocked_field_violation_count += int(_blocked_violation(audit))
        fusion_ownership_violation_count += int(
            _fusion_ownership_violation(report, audit)
        )
        ood_override_count += int(_ood_override(audit, report))
        validated_reports[task["sample_id"]] = report
    if ownership_violations:
        raise RuntimeError(
            f"{ownership_violations} Fusion snapshots changed; feedback not accepted"
        )
    if (
        audit_complete_count != expected
        or blocked_field_violation_count
        or fusion_ownership_violation_count
        or ood_override_count
    ):
        raise RuntimeError(
            "HITL import safety validation failed: "
            f"audit={audit_complete_count}/{expected}, "
            f"blocked={blocked_field_violation_count}, "
            f"fusion={fusion_ownership_violation_count}, "
            f"ood={ood_override_count}"
        )

    existing_cases = {item.case_id: item for item in memory.cases()}
    existing_feedback = {
        item.feedback_id: item for item in memory.feedback()
    }
    referenced_case_ids = {
        item.case_id for item in existing_feedback.values()
    }
    recoverable_orphans = {
        case_id: case
        for case_id, case in existing_cases.items()
        if case_id not in referenced_case_ids
    }
    for task in store.tasks:
        response = responses[task["task_id"]]
        record = records[task["sample_id"]]
        report = validated_reports[task["sample_id"]]
        feedback_id = f"human-{task['task_id']}"
        review_type = (
            "needs_more_evidence"
            if response["needs_more_evidence"]
            else "confirm"
        )
        stored_feedback = existing_feedback.get(feedback_id)
        if stored_feedback is not None:
            if stored_feedback.case_id not in existing_cases:
                raise RuntimeError(
                    f"{feedback_id} refers to a missing CaseMemory case"
                )
            expected_feedback_fields = {
                "reviewer": reviewer,
                "review_type": review_type,
                "confirmed_outcome": response["reviewer_outcome"],
                "reason": response["reviewer_rationale"],
            }
            mismatches = {
                field: getattr(stored_feedback, field)
                for field, expected_value in expected_feedback_fields.items()
                if getattr(stored_feedback, field) != expected_value
            }
            if mismatches:
                raise RuntimeError(
                    f"{feedback_id} conflicts with frozen human response: "
                    f"{mismatches}"
                )
        else:
            behavior_summary = safe_behavior_summary(record)
            matching_orphans = [
                case
                for case in recoverable_orphans.values()
                if case.source_dataset
                == "USTC-TFC2016-validation-human-pilot"
                and case.case_status == report.verdict.value
                and case.behavior_summary == behavior_summary
                and case.participating_agents
                == list(report.participating_agents)
            ]
            if len(matching_orphans) > 1:
                raise RuntimeError(
                    f"ambiguous partial CaseMemory recovery for {feedback_id}"
                )
            if matching_orphans:
                case = matching_orphans[0]
                recoverable_orphans.pop(case.case_id)
            else:
                case = memory.append_case(
                    record,
                    report,
                    source_dataset=(
                        "USTC-TFC2016-validation-human-pilot"
                    ),
                )
                existing_cases[case.case_id] = case
            stored_feedback = memory.record_feedback(
                HumanFeedbackRecord(
                    feedback_id=feedback_id,
                    case_id=case.case_id,
                    reviewer=reviewer,
                    review_type=review_type,
                    confirmed_outcome=response["reviewer_outcome"],
                    reason=response["reviewer_rationale"],
                )
            )
            existing_feedback[feedback_id] = stored_feedback
            referenced_case_ids.add(stored_feedback.case_id)
        times.append(float(response["review_seconds"]))
        if task["condition"] == "hint_visible":
            hint_relevance.append(bool(response["hint_relevant"]))
            hint_acceptance.append(bool(response["hint_accepted"]))
    expected_memory_count = 80 if phase == 1 else 200
    memory_case_count = len(list(memory.cases()))
    memory_feedback_count = len(list(memory.feedback()))
    if (
        memory_case_count != expected_memory_count
        or memory_feedback_count != expected_memory_count
    ):
        raise RuntimeError(
            "dedicated HITL memory has an unexpected record count: "
            f"{memory_case_count} cases / {memory_feedback_count} feedback"
        )
    hashes_after = hash_artifact_paths(artifact_paths)
    _dump(
        target / f"phase{phase}_frozen_artifact_hashes_after_import.json",
        hashes_after,
    )
    payload = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "phase": phase,
        "status": "completed",
        "review_count": len(times),
        "reviewer": reviewer,
        "mean_review_seconds": float(np.mean(times)),
        "median_review_seconds": float(np.median(times)),
        "hint_relevance_rate": (
            float(np.mean(hint_relevance)) if hint_relevance else None
        ),
        "hint_acceptance_rate": (
            float(np.mean(hint_acceptance)) if hint_acceptance else None
        ),
        "fusion_snapshot_violation_count": ownership_violations,
        "verdict_invariance_rate": 1.0,
        "audit_completion_rate": audit_complete_count / expected,
        "blocked_field_violation_count": blocked_field_violation_count,
        "fusion_ownership_violation_count": fusion_ownership_violation_count,
        "ood_override_count": ood_override_count,
        "frozen_artifacts_unchanged": hashes_before == hashes_after,
        "current_verdict_modified": False,
        "automatic_retraining": False,
        "single_reviewer_pilot": True,
        "inter_rater_claim": False,
        "memory_case_count": memory_case_count,
        "memory_feedback_count": memory_feedback_count,
    }
    report_path = target / f"phase{phase}_import_report.json"
    _dump(report_path, payload)
    return payload


def close_hitl_phase1_pilot(
    output_dir: str | Path,
    memory_dir: str | Path,
    *,
    document_path: str | Path | None = None,
) -> dict[str, Any]:
    """Close the real 80-case phase-1 pilot without fabricating phase 2."""

    target = Path(output_dir)
    phase1_report_path = target / "phase1_import_report.json"
    if not phase1_report_path.exists():
        raise ValueError(
            "phase 1 must be safely imported before early termination"
        )

    forbidden_paths = (
        target / "phase2_manifest.json",
        target / "phase2_tasks.jsonl",
        target / "phase2_responses.json",
        target / "phase2_responses.csv",
        target / "phase2_import_report.json",
        target / "private" / "phase2_selection.json",
        target / "aggregate_metrics.json",
        target / "acceptance_report.json",
    )
    existing_forbidden = [
        str(path.relative_to(target))
        for path in forbidden_paths
        if path.exists()
    ]
    if existing_forbidden:
        raise ValueError(
            "phase-1-only closure is incompatible with phase-2 or full "
            f"acceptance artifacts: {existing_forbidden}"
        )

    phase1_report = json.loads(
        phase1_report_path.read_text(encoding="utf-8")
    )
    if (
        phase1_report.get("status") != "completed"
        or int(phase1_report.get("phase", 0)) != 1
        or int(phase1_report.get("review_count", 0)) != 80
    ):
        raise ValueError("phase 1 import report is incomplete")
    required_report_values = {
        "verdict_invariance_rate": 1.0,
        "audit_completion_rate": 1.0,
        "blocked_field_violation_count": 0,
        "fusion_ownership_violation_count": 0,
        "ood_override_count": 0,
        "fusion_snapshot_violation_count": 0,
        "frozen_artifacts_unchanged": True,
        "current_verdict_modified": False,
        "automatic_retraining": False,
        "memory_case_count": 80,
        "memory_feedback_count": 80,
    }
    failed_report_fields = {
        field: phase1_report.get(field)
        for field, expected in required_report_values.items()
        if phase1_report.get(field) != expected
    }
    if failed_report_fields:
        raise ValueError(
            "phase 1 safety acceptance failed: "
            f"{failed_report_fields}"
        )

    selection_path = target / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selections = selection.get("selections", [])
    phase_counts = Counter(int(row["phase"]) for row in selections)
    if len(selections) != 200 or phase_counts != {1: 80, 2: 120}:
        raise ValueError(
            "the original frozen 80+120 selection manifest must be preserved"
        )

    store = HitlResponseStore(target, 1)
    state = store.public_state()
    if state["completed_count"] != 80 or state["total_count"] != 80:
        raise ValueError(
            "phase 1 requires exactly 80 completed real reviews"
        )
    if any(
        task.get("condition") != "hint_hidden"
        or task.get("memory_hint") is not None
        for task in store.tasks
    ):
        raise ValueError("phase 1 must contain only hidden-hint tasks")
    for task in store.tasks:
        validate_hitl_response(
            task,
            state["responses"][task["task_id"]],
        )

    responses_csv_path = target / "phase1_responses.csv"
    with responses_csv_path.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        response_rows = list(csv.DictReader(handle))
    task_ids = {task["task_id"] for task in store.tasks}
    if (
        len(response_rows) != 80
        or {row.get("task_id") for row in response_rows} != task_ids
    ):
        raise ValueError(
            "phase1_responses.csv does not match the 80 frozen tasks"
        )

    memory = JsonlCaseMemory(memory_dir)
    memory_case_count = len(list(memory.cases()))
    memory_feedback_count = len(list(memory.feedback()))
    if memory_case_count != 80 or memory_feedback_count != 80:
        raise ValueError(
            "phase-1 CaseMemory must contain exactly 80 cases and 80 feedback "
            "records before early termination"
        )

    response_sources = (
        target / "phase1_responses.json",
        responses_csv_path,
    )
    frozen_dir = target / "frozen_phase1_responses"
    frozen_files: dict[str, dict[str, Any]] = {}
    for source in response_sources:
        source_content = source.read_bytes()
        source_hash = hashlib.sha256(source_content).hexdigest()
        frozen_path = frozen_dir / source.name
        if frozen_path.exists():
            frozen_hash = sha256_file(frozen_path)
            if frozen_hash != source_hash:
                raise ValueError(
                    f"frozen response changed after closure: {source.name}"
                )
        else:
            _atomic_write_bytes(frozen_path, source_content)
            frozen_hash = sha256_file(frozen_path)
        frozen_files[source.name] = {
            "sha256": source_hash,
            "size_bytes": len(source_content),
            "frozen_copy": str(frozen_path),
            "frozen_copy_sha256": frozen_hash,
        }
    response_freeze = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "phase": 1,
        "real_review_count": 80,
        "files": frozen_files,
    }
    _dump(target / "phase1_response_freeze.json", response_freeze)

    responses = list(state["responses"].values())
    review_times = [
        float(response["review_seconds"]) for response in responses
    ]
    outcome_distribution = dict(
        sorted(
            Counter(
                str(response["reviewer_outcome"]) for response in responses
            ).items()
        )
    )
    metrics = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "protocol_status": "partial_complete",
        "termination_reason": "reviewer_declined_phase2",
        "real_review_count": 80,
        "phase1_review_count": 80,
        "phase2_status": "cancelled",
        "full_200_case_acceptance": False,
        "mean_review_seconds": float(np.mean(review_times)),
        "median_review_seconds": float(np.median(review_times)),
        "p95_review_seconds": float(np.percentile(review_times, 95)),
        "needs_more_evidence_rate": float(
            np.mean(
                [
                    bool(response["needs_more_evidence"])
                    for response in responses
                ]
            )
        ),
        "reviewer_outcome_distribution": outcome_distribution,
        "hint_relevance_rate": "not_applicable",
        "hint_acceptance_rate": "not_applicable",
        "hint_comparison_available": False,
        "verdict_invariance_rate": phase1_report[
            "verdict_invariance_rate"
        ],
        "audit_completion_rate": phase1_report[
            "audit_completion_rate"
        ],
        "blocked_field_violation_count": phase1_report[
            "blocked_field_violation_count"
        ],
        "fusion_ownership_violation_count": phase1_report[
            "fusion_ownership_violation_count"
        ],
        "ood_override_count": phase1_report["ood_override_count"],
        "fusion_snapshot_violation_count": phase1_report[
            "fusion_snapshot_violation_count"
        ],
        "frozen_artifacts_unchanged": phase1_report[
            "frozen_artifacts_unchanged"
        ],
        "current_verdict_modified": False,
        "automatic_retraining": False,
        "memory_case_count": memory_case_count,
        "memory_feedback_count": memory_feedback_count,
        "single_reviewer_pilot": True,
        "inter_rater_claim": False,
        "response_freeze": response_freeze,
    }
    _dump(target / "phase1_pilot_metrics.json", metrics)

    closure = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "protocol_status": "partial_complete",
        "termination_reason": "reviewer_declined_phase2",
        "real_review_count": 80,
        "phase2_status": "cancelled",
        "full_200_case_acceptance": False,
        "paper_treatment": "80-case single-reviewer Phase-1 usability pilot",
        "supported_claims": [
            "Phase-1 workflow usability metrics",
            "Fusion verdict invariance during feedback import",
            "Audit and ownership safety for 80 reviewed cases",
        ],
        "unsupported_claims": [
            "MemoryHint effectiveness",
            "Hint-visible versus hint-hidden comparison",
            "Inter-rater reliability",
            "Population-level human-factors generalization",
            "Completion of the preregistered 200-case HITL protocol",
        ],
        "phase1_metrics": metrics,
    }
    _dump(target / "early_termination_report.json", closure)

    if document_path is not None:
        document = Path(document_path)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(
            "\n".join(
                [
                    "# MAD-ETD HITL Phase-1 Pilot Evaluation",
                    "",
                    "- Protocol status: `partial_complete`",
                    "- Termination reason: `reviewer_declined_phase2`",
                    "- Real single-reviewer cases: `80`",
                    "- Phase 2: `cancelled`",
                    "- Full 200-case acceptance: `false`",
                    f"- Mean review time: `{metrics['mean_review_seconds']:.3f}` seconds",
                    f"- Median review time: `{metrics['median_review_seconds']:.3f}` seconds",
                    f"- P95 review time: `{metrics['p95_review_seconds']:.3f}` seconds",
                    f"- More-evidence request rate: `{metrics['needs_more_evidence_rate']:.4f}`",
                    f"- Reviewer outcomes: `{json.dumps(outcome_distribution, sort_keys=True)}`",
                    "- Hint relevance: `not_applicable`",
                    "- Hint acceptance: `not_applicable`",
                    f"- Verdict invariance: `{metrics['verdict_invariance_rate']:.4f}`",
                    f"- Audit completion: `{metrics['audit_completion_rate']:.4f}`",
                    "- Automatic retraining: `false`",
                    "",
                    "This is an 80-case single-reviewer Phase-1 usability pilot. "
                    "It does not support MemoryHint comparisons, inter-rater "
                    "reliability, population-level claims, or completion of the "
                    "original 200-case protocol.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return closure


def aggregate_hitl_pilot(
    output_dir: str | Path,
    memory_dir: str | Path,
    *,
    document_path: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    reports = []
    responses: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for phase, expected in ((1, 80), (2, 120)):
        report_path = target / f"phase{phase}_import_report.json"
        if not report_path.exists():
            raise ValueError(f"phase {phase} has not been imported")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "completed"
            or int(report.get("review_count", 0)) != expected
        ):
            raise ValueError(f"phase {phase} import is incomplete")
        reports.append(report)
        store = HitlResponseStore(target, phase)
        state = store.public_state()
        if state["completed_count"] != expected:
            raise ValueError(f"phase {phase} responses are incomplete")
        for task in store.tasks:
            responses.append(
                (task, state["responses"][task["task_id"]])
            )
    memory = JsonlCaseMemory(memory_dir)
    memory_case_count = len(list(memory.cases()))
    memory_feedback_count = len(list(memory.feedback()))
    if memory_case_count != 200 or memory_feedback_count != 200:
        raise ValueError("HITL CaseMemory must contain exactly 200 reviewed cases")

    review_times = [float(response["review_seconds"]) for _, response in responses]
    needs_more = [
        bool(response["needs_more_evidence"]) for _, response in responses
    ]
    visible = [
        response
        for task, response in responses
        if task["condition"] == "hint_visible"
    ]
    if len(visible) != 60:
        raise ValueError("HITL phase 2 must contain 60 visible-hint responses")
    aggregate = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "status": "completed",
        "review_count": len(responses),
        "phase1_review_count": reports[0]["review_count"],
        "phase2_review_count": reports[1]["review_count"],
        "mean_review_seconds": float(np.mean(review_times)),
        "median_review_seconds": float(np.median(review_times)),
        "p95_review_seconds": float(np.percentile(review_times, 95)),
        "needs_more_evidence_rate": float(np.mean(needs_more)),
        "hint_visible_review_count": len(visible),
        "hint_relevance_rate": float(
            np.mean([bool(item["hint_relevant"]) for item in visible])
        ),
        "hint_acceptance_rate": float(
            np.mean([bool(item["hint_accepted"]) for item in visible])
        ),
        "verdict_invariance_rate": min(
            float(report["verdict_invariance_rate"]) for report in reports
        ),
        "audit_completion_rate": min(
            float(report["audit_completion_rate"]) for report in reports
        ),
        "blocked_field_violation_count": sum(
            int(report["blocked_field_violation_count"]) for report in reports
        ),
        "fusion_ownership_violation_count": sum(
            int(report["fusion_ownership_violation_count"])
            for report in reports
        ),
        "ood_override_count": sum(
            int(report["ood_override_count"]) for report in reports
        ),
        "fusion_snapshot_violation_count": sum(
            int(report["fusion_snapshot_violation_count"])
            for report in reports
        ),
        "frozen_artifacts_unchanged": all(
            bool(report["frozen_artifacts_unchanged"]) for report in reports
        ),
        "current_verdict_modified": False,
        "automatic_retraining": False,
        "memory_case_count": memory_case_count,
        "memory_feedback_count": memory_feedback_count,
        "single_reviewer_pilot": True,
        "inter_rater_claim": False,
    }
    _dump(target / "aggregate_metrics.json", aggregate)
    failures = []
    if aggregate["review_count"] != 200:
        failures.append("incomplete real human review count")
    if aggregate["verdict_invariance_rate"] != 1.0:
        failures.append("Fusion verdict invariance failed")
    if aggregate["audit_completion_rate"] != 1.0:
        failures.append("audit completion failed")
    for field in (
        "blocked_field_violation_count",
        "fusion_ownership_violation_count",
        "ood_override_count",
        "fusion_snapshot_violation_count",
    ):
        if aggregate[field]:
            failures.append(f"{field} is nonzero")
    if not aggregate["frozen_artifacts_unchanged"]:
        failures.append("frozen artifacts changed")
    acceptance = {
        "schema_version": "1.0",
        "experiment": "hitl_pilot_v1",
        "acceptance_status": "passed" if not failures else "failed",
        "failures": failures,
        **aggregate,
    }
    _dump(target / "acceptance_report.json", acceptance)
    if document_path is not None:
        document = Path(document_path)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(
            "\n".join(
                [
                    "# MAD-ETD HITL Pilot Evaluation",
                    "",
                    f"- Acceptance: `{acceptance['acceptance_status']}`",
                    f"- Real reviews: `{aggregate['review_count']}`",
                    f"- Mean review time: `{aggregate['mean_review_seconds']:.3f}` seconds",
                    f"- Median review time: `{aggregate['median_review_seconds']:.3f}` seconds",
                    f"- More-evidence request rate: `{aggregate['needs_more_evidence_rate']:.4f}`",
                    f"- Hint relevance rate: `{aggregate['hint_relevance_rate']:.4f}`",
                    f"- Hint acceptance rate: `{aggregate['hint_acceptance_rate']:.4f}`",
                    f"- Verdict invariance: `{aggregate['verdict_invariance_rate']:.4f}`",
                    f"- Audit completion: `{aggregate['audit_completion_rate']:.4f}`",
                    "- Automatic retraining: `false`",
                    "- Single-reviewer usability pilot; no inter-rater claim.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return acceptance
