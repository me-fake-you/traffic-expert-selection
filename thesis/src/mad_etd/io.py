from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from .schemas import DetectionReport, FlowRecord


def _parse_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] in "[{" or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _unflatten(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path, value in row.items():
        if value is None:
            continue
        cursor = result
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return result


def load_raw_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                _unflatten({key: _parse_cell(value) for key, value in row.items()})
                for row in csv.DictReader(handle)
            ]
    raise ValueError(f"unsupported input format: {source.suffix}")


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def iter_flow_records(path: str | Path) -> Iterator[FlowRecord]:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.rglob("*.jsonl")) + sorted(source.rglob("*.jsonl.gz"))
        for item in files:
            yield from iter_flow_records(item)
        return
    name = source.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        with _open_text(source, "rt") as handle:
            for line in handle:
                if line.strip():
                    yield FlowRecord.model_validate_json(line)
        return
    for row in load_raw_rows(source):
        yield FlowRecord.model_validate(row)


def iter_split_records(
    input_path: str | Path,
    manifest_path: str | Path,
    split: str,
) -> Iterator[FlowRecord]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if "assignments" in manifest:
        selected = set(manifest["assignments"][split])
        for record in iter_flow_records(input_path):
            if record.sample_id in selected:
                yield record
        return
    if "files" in manifest:
        dataset_root = manifest_file.parent.parent
        for relative in manifest["files"][split]:
            yield from iter_flow_records(dataset_root / relative)
        return
    raise ValueError("split manifest must contain assignments or files")


def load_flow_records(path: str | Path) -> list[FlowRecord]:
    return list(iter_flow_records(path))


def write_jsonl(records: Iterable[FlowRecord], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_text(target, "wt") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def write_report(
    report: DetectionReport,
    path: str | Path,
    *,
    markdown: str | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if markdown is not None:
        target.with_suffix(".md").write_text(markdown, encoding="utf-8")
