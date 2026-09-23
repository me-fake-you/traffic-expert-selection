from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import FlowRecord


def build_split_manifest(
    records: Iterable[FlowRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    time_block_seconds: int = 300,
) -> dict[str, Any]:
    if train_ratio <= 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below 1")

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        label = str(record.labels.get("binary", "unknown"))
        source_file = record.provenance.get("source_file")
        capture_start = record.provenance.get("capture_start_epoch")
        session_id = record.context.get("session_id")
        if source_file and isinstance(capture_start, (int, float)):
            time_block = int(float(capture_start) // time_block_seconds)
            group = f"{source_file}::timeblock={time_block}"
        elif session_id:
            group = f"session={session_id}"
        elif source_file:
            group = f"source={source_file}"
        else:
            group = f"sample={record.sample_id}"
        grouped[(label, group)].append(record.sample_id)

    by_label: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (label, group), items in grouped.items():
        by_label[label].append((group, items))

    assignments = {"train": [], "validation": [], "test": []}
    group_assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for label, groups in sorted(by_label.items()):
        rng.shuffle(groups)
        total = sum(len(items) for _, items in groups)
        targets = {
            "train": total * train_ratio,
            "validation": total * validation_ratio,
            "test": total * (1 - train_ratio - validation_ratio),
        }
        counts = {"train": 0, "validation": 0, "test": 0}
        groups.sort(key=lambda item: len(item[1]), reverse=True)

        for group, items in groups:
            order = {"train": 0, "validation": 1, "test": 2}

            def projected_error(split_name: str) -> tuple[float, int]:
                projected = dict(counts)
                projected[split_name] += len(items)
                error = sum(
                    (projected[name] - targets[name]) ** 2
                    for name in ("train", "validation", "test")
                )
                return error, order[split_name]

            split = min(("train", "validation", "test"), key=projected_error)
            counts[split] += len(items)
            group_key = f"{label}:{group}"
            group_assignments[group_key] = split
            assignments[split].extend(items)

    manifest_core = {
        "version": "1.0",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": 1 - train_ratio - validation_ratio,
        },
        "grouping_priority": [
            "provenance.source_file + capture_start_epoch time block",
            "context.session_id",
            "provenance.source_file",
            "sample_id",
        ],
        "time_block_seconds": time_block_seconds,
        "assignments": assignments,
        "group_assignments": group_assignments,
    }
    canonical = json.dumps(
        manifest_core, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    manifest_core["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest_core


def write_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
