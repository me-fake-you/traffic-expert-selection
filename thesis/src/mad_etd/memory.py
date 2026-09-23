from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .audit import AuditLogger
from .schemas import (
    CaseMemoryRecord,
    DetectionReport,
    FlowRecord,
    HumanFeedbackRecord,
    MemoryHint,
    TrainingCandidate,
)


LOCKED_MEMORY_DATASETS = {
    "CESNET-TLS22",
    "CipherSpectrum",
    "CipherSpectrum-locked_test",
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_behavior_summary(flow: FlowRecord) -> dict[str, float]:
    """Build a behavior-only summary with no context, labels, or provenance."""

    summary = {
        f"stats.{key}": float(value)
        for key, value in flow.stats.items()
    }
    summary.update(
        {
            "sequence.length": float(len(flow.sequence.packet_lengths)),
            "sequence.original_packet_count": float(
                flow.sequence.original_packet_count
                if flow.sequence.original_packet_count is not None
                else len(flow.sequence.packet_lengths)
            ),
            "sequence.truncated": float(flow.sequence.truncated),
            "sequence.mean_iat": (
                sum(flow.sequence.iats) / len(flow.sequence.iats)
                if flow.sequence.iats
                else 0.0
            ),
        }
    )
    return summary


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    numerator = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(left.get(key, 0.0) ** 2 for key in keys))
    right_norm = math.sqrt(sum(right.get(key, 0.0) ** 2 for key in keys))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


class JsonlCaseMemory:
    """Append-only research memory. It never returns AgentEvidence."""

    name = "CaseMemory"
    version = "1.0"

    def __init__(
        self,
        root: str | Path,
        *,
        locked_datasets: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.case_path = self.root / "cases.jsonl"
        self.feedback_path = self.root / "feedback.jsonl"
        self.candidate_path = self.root / "training_candidates.jsonl"
        self.locked_datasets = locked_datasets or set(LOCKED_MEMORY_DATASETS)
        self._case_cache_signature: tuple[int, int] | None = None
        self._case_cache: list[CaseMemoryRecord] = []
        self._index_fields: list[str] = []
        self._index_matrix = np.empty((0, 0), dtype=np.float64)

    @staticmethod
    def _append(path: Path, payload: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")

    @staticmethod
    def _iter(path: Path, model_type):
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield model_type.model_validate_json(line)

    def cases(self) -> Iterable[CaseMemoryRecord]:
        return self._iter(self.case_path, CaseMemoryRecord) or ()

    def _invalidate_case_index(self) -> None:
        self._case_cache_signature = None
        self._case_cache = []
        self._index_fields = []
        self._index_matrix = np.empty((0, 0), dtype=np.float64)

    def _case_index(
        self,
    ) -> tuple[list[CaseMemoryRecord], list[str], np.ndarray]:
        signature = (
            self.case_path.stat().st_mtime_ns,
            self.case_path.stat().st_size,
        ) if self.case_path.exists() else (0, 0)
        if signature == self._case_cache_signature:
            return self._case_cache, self._index_fields, self._index_matrix
        cases = list(self.cases())
        fields = sorted(
            {
                field
                for case in cases
                for field in case.behavior_summary
            }
        )
        matrix = np.zeros((len(cases), len(fields)), dtype=np.float64)
        positions = {field: index for index, field in enumerate(fields)}
        for row_index, case in enumerate(cases):
            for field, value in case.behavior_summary.items():
                matrix[row_index, positions[field]] = value
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = np.divide(
                matrix,
                norms,
                out=np.zeros_like(matrix),
                where=norms != 0,
            )
        self._case_cache_signature = signature
        self._case_cache = cases
        self._index_fields = fields
        self._index_matrix = matrix
        return cases, fields, matrix

    def feedback(self) -> Iterable[HumanFeedbackRecord]:
        return self._iter(self.feedback_path, HumanFeedbackRecord) or ()

    def candidates(self) -> Iterable[TrainingCandidate]:
        return self._iter(self.candidate_path, TrainingCandidate) or ()

    def append_case(
        self,
        flow: FlowRecord,
        report: DetectionReport,
        *,
        source_dataset: str,
        audit: AuditLogger | None = None,
    ) -> CaseMemoryRecord:
        if source_dataset in self.locked_datasets:
            if audit:
                audit.log(
                    self.name,
                    "MEMORY_WRITE_BLOCKED",
                    input_summary={"source_dataset": source_dataset},
                    reason="Locked benchmarks cannot populate CaseMemory.",
                )
            raise ValueError(
                f"locked benchmark cannot populate CaseMemory: {source_dataset}"
            )
        if report.verdict.value not in {"suspicious", "unknown"}:
            raise ValueError("CaseMemory MVP only stores suspicious or unknown cases")
        report_hash = _sha256(report.model_dump_json())
        case_id = _sha256(f"{report.trace_id}:{report_hash}")[:24]
        existing = next(
            (item for item in self.cases() if item.case_id == case_id),
            None,
        )
        if existing is not None:
            return existing
        record = CaseMemoryRecord(
            case_id=case_id,
            source_trace_id=report.trace_id,
            source_report_sha256=report_hash,
            source_dataset=source_dataset,
            case_status=report.verdict.value,
            behavior_summary=safe_behavior_summary(flow),
            participating_agents=list(report.participating_agents),
        )
        self._append(self.case_path, record.model_dump_json())
        self._invalidate_case_index()
        if audit:
            audit.log(
                self.name,
                "MEMORY_CASE_APPENDED",
                output_summary={
                    "case_id": record.case_id,
                    "source_report_sha256": report_hash,
                    "behavior_fields": sorted(record.behavior_summary),
                },
                reason="An eligible case was appended without modifying its report.",
            )
        return record

    def record_feedback(
        self,
        feedback: HumanFeedbackRecord,
        *,
        audit: AuditLogger | None = None,
    ) -> HumanFeedbackRecord:
        existing_ids = {item.feedback_id for item in self.feedback()}
        if feedback.feedback_id in existing_ids:
            return next(
                item
                for item in self.feedback()
                if item.feedback_id == feedback.feedback_id
            )
        case_ids = {item.case_id for item in self.cases()}
        if feedback.case_id not in case_ids:
            raise ValueError(f"unknown CaseMemory case: {feedback.case_id}")
        if feedback.supersedes and feedback.supersedes not in existing_ids:
            raise ValueError("supersedes must refer to existing feedback")
        self._append(self.feedback_path, feedback.model_dump_json())
        if audit:
            audit.log(
                self.name,
                "HUMAN_FEEDBACK_RECORDED",
                output_summary={
                    "feedback_id": feedback.feedback_id,
                    "case_id": feedback.case_id,
                    "review_type": feedback.review_type,
                    "supersedes": feedback.supersedes,
                },
                reason="Human feedback was appended; no model update was triggered.",
            )
        return feedback

    def queue_candidate(
        self,
        candidate: TrainingCandidate,
        *,
        audit: AuditLogger | None = None,
    ) -> TrainingCandidate:
        feedback_ids = {item.feedback_id for item in self.feedback()}
        if candidate.feedback_ref not in feedback_ids:
            raise ValueError("training candidate requires recorded human feedback")
        existing = next(
            (
                item
                for item in self.candidates()
                if item.candidate_id == candidate.candidate_id
            ),
            None,
        )
        if existing is not None:
            return existing
        self._append(self.candidate_path, candidate.model_dump_json())
        if audit:
            audit.log(
                self.name,
                "TRAINING_CANDIDATE_QUEUED",
                output_summary=candidate.model_dump(mode="json"),
                reason="Candidate queued for human approval; no training was started.",
            )
        return candidate

    def retrieve(
        self,
        flow: FlowRecord,
        *,
        top_k: int = 3,
        min_score: float = 0.1,
        audit: AuditLogger | None = None,
    ) -> MemoryHint:
        query = safe_behavior_summary(flow)
        cases, fields, matrix = self._case_index()
        if cases and fields:
            vector = np.asarray(
                [query.get(field, 0.0) for field in fields],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(vector))
            scores = matrix @ (vector / norm) if norm else np.zeros(len(cases))
            scored = sorted(
                (
                    (max(0.0, min(1.0, float(score))), case)
                    for score, case in zip(scores, cases, strict=True)
                ),
                key=lambda pair: (-pair[0], pair[1].case_id),
            )
        else:
            scored = []
        matches = [
            (score, item)
            for score, item in scored[: max(1, top_k)]
            if score >= min_score
        ]
        if not matches:
            hint = MemoryHint(
                hint_id=_sha256(f"{flow.trace_id}:no-match")[:24],
                status="no_match",
                limitations=["No sufficiently similar reviewed case was found."],
            )
        else:
            agents = sorted(
                {
                    agent
                    for _, item in matches
                    for agent in item.participating_agents
                }
            )
            hint = MemoryHint(
                hint_id=_sha256(
                    f"{flow.trace_id}:"
                    + ",".join(item.case_id for _, item in matches)
                )[:24],
                status="success",
                dispatch_hint=agents,
                explanation_hint=[
                    "Similar historical cases exist; compare evidence patterns only."
                ],
                retrieval_hint=["Review cited cases during human escalation."],
                similar_case_refs=[item.case_id for _, item in matches],
                retrieval_scores=[score for score, _ in matches],
                limitations=[
                    "Historical outcomes are advisory and are not detection evidence."
                ],
            )
        if audit:
            audit.log(
                self.name,
                "MEMORY_RETRIEVED"
                if hint.status == "success"
                else "MEMORY_NO_MATCH",
                input_summary={"query_fields": sorted(query)},
                output_summary=hint.model_dump(mode="json"),
                reason="CaseMemory returned advisory hints only.",
            )
        return hint
