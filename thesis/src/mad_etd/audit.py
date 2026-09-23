from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .schemas import AuditEvent, FutureAnalysisArtifacts


class AuditLogger:
    """Append-only, thread-safe in-memory audit log with optional JSONL persistence."""

    def __init__(self, trace_id: str, output_path: str | Path | None = None) -> None:
        self.trace_id = trace_id
        self.output_path = Path(output_path) if output_path else None
        self._events: list[AuditEvent] = []
        self.future_artifacts = FutureAnalysisArtifacts()
        self._lock = threading.Lock()
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("", encoding="utf-8")

    def log(
        self,
        actor: str,
        event_type: str,
        *,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        reason: str = "",
        duration_ms: float = 0,
    ) -> AuditEvent:
        with self._lock:
            event = AuditEvent(
                sequence_no=len(self._events) + 1,
                trace_id=self.trace_id,
                actor=actor,
                event_type=event_type,
                input_summary=input_summary or {},
                output_summary=output_summary or {},
                reason=reason,
                duration_ms=duration_ms,
            )
            self._events.append(event)
            if self.output_path:
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(event.model_dump_json() + "\n")
            return event

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def export(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(event.model_dump_json() + "\n")

    def as_json(self) -> str:
        return json.dumps(
            [event.model_dump(mode="json") for event in self._events],
            ensure_ascii=False,
            indent=2,
        )
