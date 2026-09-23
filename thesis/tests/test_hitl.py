import csv
import json

import pytest

from mad_etd.hitl import (
    HitlResponseStore,
    _select_candidates,
    aggregate_hitl_pilot,
    close_hitl_phase1_pilot,
    prepare_hitl_phase2,
    serve_hitl_review,
    validate_hitl_response,
)


def _task(task_id="hitl-p1-000", condition="hint_hidden"):
    return {
        "task_id": task_id,
        "phase": 1,
        "condition": condition,
        "sample_id": f"sample-{task_id}",
        "system_verdict": "unknown",
        "system_confidence": 0.4,
        "system_uncertainty": 0.6,
        "system_conflict_score": 0.2,
        "distribution_shift_score": 0.7,
        "severity": "medium",
        "need_escalation": True,
        "participating_agents": ["StatsDetectorAgent"],
        "main_evidence": ["high uncertainty"],
        "memory_hint": None,
        "fusion_snapshot_sha256": "a" * 64,
        "report_sha256": "b" * 64,
        "reviewer_outcome": "",
        "needs_more_evidence": "",
        "hint_relevant": "",
        "hint_accepted": "",
        "review_seconds": "",
        "reviewer_rationale": "",
    }


def test_frozen_hitl_selection_is_80_plus_120_and_phase2_is_60_60(tmp_path):
    path = tmp_path / "candidates.csv"
    rows = []
    for index in range(400):
        binary = "benign" if index % 2 == 0 else "malicious"
        verdict = "unknown" if index % 4 < 2 else "suspicious"
        rows.append(
            {
                "sample_id": f"sample-{index:04d}",
                "stratum": f"{binary}|family-{index % 7}|capture",
                "verdict": verdict,
                "priority": str(index),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = _select_candidates(
        path,
        source_ids={row["sample_id"] for row in rows},
        total_count=200,
        phase1_count=80,
        seed=42,
    )

    assert len(selected) == 200
    assert len({row["sample_id"] for row in selected}) == 200
    assert sum(row["phase"] == 1 for row in selected) == 80
    assert sum(row["phase"] == 2 for row in selected) == 120
    assert sum(row["condition"] == "hint_visible" for row in selected) == 60
    assert sum(
        row["phase"] == 2 and row["condition"] == "hint_hidden"
        for row in selected
    ) == 60


def test_response_validation_requires_visible_hint_fields_only():
    hidden = validate_hitl_response(
        _task(),
        {
            "reviewer_outcome": "unknown",
            "needs_more_evidence": True,
            "review_seconds": 12.5,
            "reviewer_rationale": "Evidence is insufficient.",
            "hint_relevant": "",
            "hint_accepted": "",
        },
    )
    assert hidden["hint_relevant"] == ""

    visible_task = _task(condition="hint_visible")
    with pytest.raises(ValueError, match="hint_relevant"):
        validate_hitl_response(
            visible_task,
            {
                "reviewer_outcome": "unknown",
                "needs_more_evidence": True,
                "review_seconds": 12.5,
                "reviewer_rationale": "Evidence is insufficient.",
            },
        )


def test_response_store_persists_atomically_and_exports_csv(tmp_path):
    task = _task()
    (tmp_path / "phase1_tasks.jsonl").write_text(
        json.dumps(task) + "\n", encoding="utf-8"
    )
    store = HitlResponseStore(tmp_path, 1)

    saved = store.save(
        task["task_id"],
        {
            "reviewer_outcome": "suspicious",
            "needs_more_evidence": False,
            "review_seconds": 4.2,
            "reviewer_rationale": "Multiple signals agree.",
            "hint_relevant": "",
            "hint_accepted": "",
        },
    )

    assert saved["reviewer_outcome"] == "suspicious"
    assert store.public_state()["completed_count"] == 1
    assert (tmp_path / "phase1_responses.csv").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_phase2_is_blocked_until_80_real_phase1_reviews(tmp_path):
    report = tmp_path / "phase1_import_report.json"
    report.write_text(
        json.dumps({"status": "completed", "phase": 1, "review_count": 79}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="80 completed"):
        prepare_hitl_phase2(
            "unused.jsonl",
            tmp_path,
            tmp_path / "memory",
            phase1_import_report=report,
        )


def test_review_server_rejects_non_loopback_binding(tmp_path):
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve_hitl_review(
            tmp_path, phase=1, host="0.0.0.0", port=0
        )


def test_hitl_aggregate_requires_both_imported_phases(tmp_path):
    with pytest.raises(ValueError, match="phase 1 has not been imported"):
        aggregate_hitl_pilot(tmp_path, tmp_path / "memory")


def test_hitl_aggregate_reports_real_review_metrics(tmp_path, monkeypatch):
    class FakeMemory:
        def __init__(self, root):
            self.root = root

        def cases(self):
            return iter(range(200))

        def feedback(self):
            return iter(range(200))

    monkeypatch.setattr("mad_etd.hitl.JsonlCaseMemory", FakeMemory)
    for phase, count in ((1, 80), (2, 120)):
        tasks = []
        responses = {}
        for index in range(count):
            condition = (
                "hint_visible"
                if phase == 2 and index < 60
                else "hint_hidden"
            )
            task = _task(f"hitl-p{phase}-{index:03d}", condition)
            task["phase"] = phase
            tasks.append(task)
            responses[task["task_id"]] = {
                "reviewer_outcome": "unknown",
                "needs_more_evidence": index % 2 == 0,
                "hint_relevant": condition == "hint_visible",
                "hint_accepted": condition == "hint_visible",
                "review_seconds": 10.0,
                "reviewer_rationale": "Human-reviewed rationale.",
                "saved_at": "2026-06-20T00:00:00+00:00",
            }
        (tmp_path / f"phase{phase}_tasks.jsonl").write_text(
            "".join(json.dumps(task) + "\n" for task in tasks),
            encoding="utf-8",
        )
        (tmp_path / f"phase{phase}_responses.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "phase": phase,
                    "responses": responses,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / f"phase{phase}_import_report.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "review_count": count,
                    "verdict_invariance_rate": 1.0,
                    "audit_completion_rate": 1.0,
                    "blocked_field_violation_count": 0,
                    "fusion_ownership_violation_count": 0,
                    "ood_override_count": 0,
                    "fusion_snapshot_violation_count": 0,
                    "frozen_artifacts_unchanged": True,
                }
            ),
            encoding="utf-8",
        )

    result = aggregate_hitl_pilot(
        tmp_path,
        tmp_path / "memory",
        document_path=tmp_path / "evaluation.md",
    )

    assert result["acceptance_status"] == "passed"
    assert result["review_count"] == 200
    assert result["needs_more_evidence_rate"] == 0.5
    assert result["hint_relevance_rate"] == 1.0
    assert result["verdict_invariance_rate"] == 1.0
    assert (tmp_path / "evaluation.md").exists()


def _write_phase1_closure_fixture(tmp_path, *, include_report=True):
    tasks = []
    responses = {}
    for index in range(80):
        task = _task(f"hitl-p1-{index:03d}")
        tasks.append(task)
        responses[task["task_id"]] = {
            "reviewer_outcome": (
                "suspicious" if index % 2 == 0 else "unknown"
            ),
            "needs_more_evidence": index % 4 == 0,
            "hint_relevant": "",
            "hint_accepted": "",
            "review_seconds": 10.0 + index,
            "reviewer_rationale": "Real reviewer rationale.",
            "saved_at": "2026-06-20T00:00:00+00:00",
        }
    (tmp_path / "phase1_tasks.jsonl").write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )
    (tmp_path / "phase1_responses.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "phase": 1,
                "responses": responses,
            }
        ),
        encoding="utf-8",
    )
    with (tmp_path / "phase1_responses.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id"])
        writer.writeheader()
        writer.writerows({"task_id": task["task_id"]} for task in tasks)
    (tmp_path / "selection_manifest.json").write_text(
        json.dumps(
            {
                "selections": [
                    {
                        "sample_id": f"sample-{index:03d}",
                        "phase": 1 if index < 80 else 2,
                    }
                    for index in range(200)
                ]
            }
        ),
        encoding="utf-8",
    )
    if include_report:
        (tmp_path / "phase1_import_report.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "phase": 1,
                    "review_count": 80,
                    "reviewer": "primary_reviewer",
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
            ),
            encoding="utf-8",
        )


def test_phase1_only_closure_requires_completed_import(tmp_path):
    _write_phase1_closure_fixture(tmp_path, include_report=False)

    with pytest.raises(ValueError, match="safely imported"):
        close_hitl_phase1_pilot(tmp_path, tmp_path / "memory")


def test_phase1_only_closure_is_idempotent_and_never_claims_full_acceptance(
    tmp_path, monkeypatch
):
    _write_phase1_closure_fixture(tmp_path)

    class FakeMemory:
        def __init__(self, root):
            self.root = root

        def cases(self):
            return iter(range(80))

        def feedback(self):
            return iter(range(80))

    monkeypatch.setattr("mad_etd.hitl.JsonlCaseMemory", FakeMemory)
    document = tmp_path / "phase1.md"

    first = close_hitl_phase1_pilot(
        tmp_path,
        tmp_path / "memory",
        document_path=document,
    )
    second = close_hitl_phase1_pilot(
        tmp_path,
        tmp_path / "memory",
        document_path=document,
    )

    assert first == second
    assert first["protocol_status"] == "partial_complete"
    assert first["phase2_status"] == "cancelled"
    assert first["full_200_case_acceptance"] is False
    assert first["phase1_metrics"]["hint_relevance_rate"] == "not_applicable"
    assert first["phase1_metrics"]["hint_acceptance_rate"] == "not_applicable"
    assert first["phase1_metrics"]["real_review_count"] == 80
    assert not (tmp_path / "acceptance_report.json").exists()
    assert not (tmp_path / "aggregate_metrics.json").exists()
    assert (tmp_path / "phase1_pilot_metrics.json").exists()
    assert (tmp_path / "early_termination_report.json").exists()
    assert (
        tmp_path / "frozen_phase1_responses" / "phase1_responses.json"
    ).exists()
    assert document.exists()


def test_phase1_only_closure_rejects_phase2_or_partial_memory(
    tmp_path, monkeypatch
):
    _write_phase1_closure_fixture(tmp_path)

    class PartialMemory:
        count = 79

        def __init__(self, root):
            self.root = root

        def cases(self):
            return iter(range(self.count))

        def feedback(self):
            return iter(range(self.count))

    monkeypatch.setattr("mad_etd.hitl.JsonlCaseMemory", PartialMemory)
    with pytest.raises(ValueError, match="exactly 80"):
        close_hitl_phase1_pilot(tmp_path, tmp_path / "memory")

    PartialMemory.count = 80
    (tmp_path / "phase2_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="incompatible"):
        close_hitl_phase1_pilot(tmp_path, tmp_path / "memory")
