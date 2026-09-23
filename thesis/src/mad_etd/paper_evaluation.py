from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Iterator

import numpy as np

from .engine import build_default_engine
from .future_evaluation import (
    FUSION_OWNED_FIELDS,
    FUTURE_EVALUATION_MODES,
)
from .io import iter_flow_records
from .memory import JsonlCaseMemory
from .schemas import FlowRecord, HumanFeedbackRecord


PAPER_SHADOW_MODES = {
    "memory_shadow",
    "planner_executor",
    "deliberation_shadow",
    "reflection_critic",
    "full_shadow",
}
REQUIRED_AUDIT_EVENTS = {
    "FLOW_VALIDATED",
    "FIELD_POLICY_APPLIED",
    "RELIABILITY_ASSESSED",
    "FINAL_FUSION",
    "REPORT_GENERATED",
}
PREDICTION_FIELDS = [
    "mode",
    "sample_id",
    "group_key",
    "label_binary",
    "label_family",
    "application",
    "cipher_condition",
    "browser",
    "capture_date",
    "verdict",
    "confidence",
    "uncertainty",
    "conflict_score",
    "benign_support",
    "malicious_support",
    "distribution_shift_score",
    "severity",
    "need_escalation",
    "agent_calls",
    "latency_ms",
    "coverage_indicator",
    "alert_indicator",
    "human_review_indicator",
    "verdict_invariant",
    "uncertainty_delta",
    "conflict_delta",
    "memory_status",
    "memory_top1_ref",
    "memory_top1_relevant",
    "memory_topk_relevant",
    "deliberation_count",
    "supplemental_dispatch",
    "compliance_status",
    "compliance_failure",
    "audit_complete",
    "blocked_field_violation",
    "ood_signature",
    "ood_signature_invariant",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_priority(seed: int, *values: str) -> str:
    payload = ":".join((str(seed), *values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def hash_artifact_paths(paths: dict[str, str | Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if path.is_dir():
            files = sorted(
                item for item in path.rglob("*") if item.is_file()
            )
            hashes = {
                item.relative_to(path).as_posix(): sha256_file(item)
                for item in files
            }
            aggregate = hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode("utf-8")
            ).hexdigest()
            result[name] = {
                "path": path.as_posix(),
                "aggregate_sha256": aggregate,
                "files": hashes,
            }
        else:
            result[name] = {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
            }
    return result


def _allocate_stratified(
    counts: dict[str, int],
    target: int,
) -> dict[str, int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError("target must be between zero and the available count")
    if target == total:
        return dict(counts)
    groups = sorted(counts)
    allocation = {group: 0 for group in groups}
    remaining = target
    if target >= len(groups):
        for group in groups:
            allocation[group] = 1
        remaining -= len(groups)
    capacities = {
        group: counts[group] - allocation[group] for group in groups
    }
    capacity_total = sum(capacities.values())
    if remaining and capacity_total:
        raw = {
            group: remaining * capacities[group] / capacity_total
            for group in groups
        }
        for group in groups:
            addition = min(capacities[group], math.floor(raw[group]))
            allocation[group] += addition
        leftover = target - sum(allocation.values())
        order = sorted(
            groups,
            key=lambda group: (
                -(raw[group] - math.floor(raw[group])),
                group,
            ),
        )
        while leftover:
            progressed = False
            for group in order:
                if allocation[group] < counts[group]:
                    allocation[group] += 1
                    leftover -= 1
                    progressed = True
                    if not leftover:
                        break
            if not progressed:
                raise RuntimeError("could not complete stratified allocation")
    return allocation


def _record_group(record: FlowRecord, dataset: str) -> str:
    normalized = dataset.lower()
    if normalized.startswith("ustc"):
        source = str(record.provenance.get("source_file", "unknown"))
        epoch = float(record.provenance.get("capture_start_epoch", 0.0))
        return f"{source}|time_block={int(epoch // 300)}"
    if normalized.startswith("cesnet"):
        date = str(record.provenance.get("capture_date", "unknown"))
        app = str(record.labels.get("application", "unknown"))
        return f"{date}|{app}"
    if normalized.startswith("cipherspectrum"):
        return str(record.labels.get("application", "unknown"))
    return str(record.provenance.get("source_file", record.sample_id))


def build_split_selection_manifest(
    input_path: str | Path,
    split_manifest_path: str | Path,
    split: str,
    output_path: str | Path,
    *,
    dataset: str,
    seed: int = 42,
) -> dict[str, Any]:
    source = Path(split_manifest_path)
    split_manifest = json.loads(source.read_text(encoding="utf-8"))
    assignments = split_manifest.get("assignments", {})
    if split not in assignments:
        raise ValueError(f"split {split!r} is absent from the manifest")
    selected = list(assignments[split])
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("split manifest contains duplicate sample IDs")
    group_counts: Counter[str] = Counter()
    observed: set[str] = set()
    for record in iter_flow_records(input_path):
        if record.sample_id in selected_set:
            if record.sample_id in observed:
                raise ValueError(f"duplicate input sample: {record.sample_id}")
            observed.add(record.sample_id)
            group_counts[_record_group(record, dataset)] += 1
    missing = selected_set - observed
    if missing:
        raise ValueError(
            f"selection references {len(missing)} missing input samples"
        )
    payload = {
        "schema_version": "1.0",
        "dataset": dataset,
        "strategy": "frozen_split",
        "split": split,
        "seed": seed,
        "sample_count": len(selected),
        "unique_sample_count": len(selected_set),
        "sample_ids": selected,
        "grouping": (
            "source capture + 300-second time block"
            if dataset.lower().startswith("ustc")
            else "labels.application domain"
        ),
        "selected_group_counts": dict(sorted(group_counts.items())),
        "source_split_manifest": source.as_posix(),
        "source_split_manifest_sha256": sha256_file(source),
    }
    _json_dump(output_path, payload)
    return payload


def build_cesnet_stratified_manifest(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_count: int = 50_000,
    seed: int = 42,
) -> dict[str, Any]:
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    observed: set[str] = set()
    for record in iter_flow_records(input_path):
        if record.sample_id in observed:
            raise ValueError(f"duplicate input sample: {record.sample_id}")
        observed.add(record.sample_id)
        group = _record_group(record, "CESNET-TLS22")
        candidates[group].append(
            (_hash_priority(seed, group, record.sample_id), record.sample_id)
        )
    counts = {group: len(rows) for group, rows in candidates.items()}
    allocation = _allocate_stratified(counts, sample_count)
    selected: list[str] = []
    selected_group_counts: dict[str, int] = {}
    for group in sorted(candidates):
        rows = sorted(candidates[group])
        quota = allocation[group]
        selected.extend(sample_id for _, sample_id in rows[:quota])
        selected_group_counts[group] = quota
    selected.sort(key=lambda sample_id: _hash_priority(seed, sample_id))
    payload = {
        "schema_version": "1.0",
        "dataset": "CESNET-TLS22",
        "strategy": "date_application_stratified_hash_sample",
        "seed": seed,
        "population_count": len(observed),
        "sample_count": len(selected),
        "unique_sample_count": len(set(selected)),
        "grouping": "provenance.capture_date + labels.application",
        "population_group_counts": dict(sorted(counts.items())),
        "selected_group_counts": dict(sorted(selected_group_counts.items())),
        "sample_ids": selected,
    }
    _json_dump(output_path, payload)
    return payload


def iter_selection_records(
    input_path: str | Path,
    selection_manifest_path: str | Path,
) -> Iterator[FlowRecord]:
    manifest = json.loads(
        Path(selection_manifest_path).read_text(encoding="utf-8")
    )
    selected = set(manifest["sample_ids"])
    yielded: set[str] = set()
    for record in iter_flow_records(input_path):
        if record.sample_id not in selected:
            continue
        if record.sample_id in yielded:
            raise ValueError(f"duplicate selected sample: {record.sample_id}")
        yielded.add(record.sample_id)
        yield record
    missing = selected - yielded
    if missing:
        raise ValueError(f"{len(missing)} selected samples were not found")


def _decision_snapshot(report) -> dict[str, Any]:
    return {
        field: (
            getattr(report, field).value
            if hasattr(getattr(report, field), "value")
            else getattr(report, field)
        )
        for field in FUSION_OWNED_FIELDS
    }


def _ood_signature(report) -> str:
    rows = []
    for evidence in report.agent_results:
        if evidence.agent_name not in {
            "StatsDetectorAgent",
            "TemporalBehaviorAgent",
        }:
            continue
        rows.append(
            {
                "agent": evidence.agent_name,
                "score": evidence.distribution_shift_score,
                "raw": evidence.distribution_shift_raw_score,
                "level": evidence.distribution_shift_level,
                "reliability": evidence.model_reliability,
                "abstained": evidence.abstained,
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _blocked_violation(audit) -> bool:
    for event in audit.events:
        for summary in (event.input_summary, event.output_summary):
            intersection = summary.get("blocked_field_intersection")
            if intersection:
                return True
    return False


def _row_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else 0.0


def _row_bool(row: dict[str, str], field: str) -> bool:
    return str(row.get(field, "")).lower() in {"1", "true", "yes"}


def _load_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _baseline_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _load_prediction_rows(path):
        result[row["sample_id"]] = {
            "snapshot": {
                "verdict": row["verdict"],
                "confidence": _row_float(row, "confidence"),
                "uncertainty": _row_float(row, "uncertainty"),
                "conflict_score": _row_float(row, "conflict_score"),
                "benign_support": _row_float(row, "benign_support"),
                "malicious_support": _row_float(row, "malicious_support"),
                "distribution_shift_score": _row_float(
                    row, "distribution_shift_score"
                ),
                "severity": row["severity"],
                "need_escalation": _row_bool(row, "need_escalation"),
            },
            "ood_signature": row["ood_signature"],
        }
    return result


def _classification_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    covered = [
        row
        for row in rows
        if row["label_binary"] in {"benign", "malicious"}
        and row["verdict"] in {"benign", "malicious"}
    ]
    if not covered:
        return {"labeled_count": 0, "covered_labeled_count": 0}
    confusion = Counter(
        (row["label_binary"], row["verdict"]) for row in covered
    )
    tp = confusion[("malicious", "malicious")]
    fp = confusion[("benign", "malicious")]
    tn = confusion[("benign", "benign")]
    fn = confusion[("malicious", "benign")]
    precision_m = tp / (tp + fp) if tp + fp else 0.0
    recall_m = tp / (tp + fn) if tp + fn else 0.0
    f1_m = (
        2 * precision_m * recall_m / (precision_m + recall_m)
        if precision_m + recall_m
        else 0.0
    )
    precision_b = tn / (tn + fn) if tn + fn else 0.0
    recall_b = tn / (tn + fp) if tn + fp else 0.0
    f1_b = (
        2 * precision_b * recall_b / (precision_b + recall_b)
        if precision_b + recall_b
        else 0.0
    )
    labeled_count = sum(
        row["label_binary"] in {"benign", "malicious"} for row in rows
    )
    return {
        "labeled_count": labeled_count,
        "covered_labeled_count": len(covered),
        "labeled_coverage": len(covered) / labeled_count if labeled_count else 0,
        "selective_accuracy": (tp + tn) / len(covered),
        "selective_macro_f1": (f1_m + f1_b) / 2,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def _bootstrap_intervals(
    rows: list[dict[str, str]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    metrics = {
        "coverage": "coverage_indicator",
        "alert_rate": "alert_indicator",
        "human_review_rate": "human_review_indicator",
        "average_agent_calls": "agent_calls",
        "average_latency_ms": "latency_ms",
        "verdict_invariance_rate": "verdict_invariant",
        "audit_completion_rate": "audit_complete",
        "blocked_field_violation_rate": "blocked_field_violation",
        "ood_signature_invariance_rate": "ood_signature_invariant",
        "uncertainty_delta": "uncertainty_delta",
        "conflict_delta": "conflict_delta",
    }
    grouped: dict[str, dict[str, tuple[float, int]]] = {}
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row["group_key"]].append(row)
    for group, group_rows in buckets.items():
        grouped[group] = {
            name: (
                sum(_row_float(row, field) for row in group_rows),
                len(group_rows),
            )
            for name, field in metrics.items()
        }
    groups = sorted(grouped)
    rng = random.Random(seed)
    samples = {name: [] for name in metrics}
    if not groups:
        return {}
    for _ in range(resamples):
        selected = [rng.choice(groups) for _ in groups]
        for name in metrics:
            numerator = sum(grouped[group][name][0] for group in selected)
            denominator = sum(grouped[group][name][1] for group in selected)
            samples[name].append(numerator / denominator)
    intervals = {}
    for name, values in samples.items():
        intervals[name] = {
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
        }
    return intervals


def _summarize_mode(
    rows: list[dict[str, str]],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    count = len(rows)
    verdicts = Counter(row["verdict"] for row in rows)
    memory_matches = [
        row for row in rows if row["memory_status"] == "success"
    ]
    top1_scored = [
        row for row in memory_matches if row["memory_top1_relevant"] != ""
    ]
    return {
        "sample_count": count,
        "verdict_rates": {
            verdict: verdicts[verdict] / count
            for verdict in ("benign", "malicious", "suspicious", "unknown")
        },
        "coverage": fmean(
            _row_float(row, "coverage_indicator") for row in rows
        ),
        "alert_rate": fmean(
            _row_float(row, "alert_indicator") for row in rows
        ),
        "human_review_rate": fmean(
            _row_float(row, "human_review_indicator") for row in rows
        ),
        "average_agent_calls": fmean(
            _row_float(row, "agent_calls") for row in rows
        ),
        "latency_ms": {
            "mean": fmean(_row_float(row, "latency_ms") for row in rows),
            "p50": float(
                np.quantile(
                    [_row_float(row, "latency_ms") for row in rows], 0.5
                )
            ),
            "p95": float(
                np.quantile(
                    [_row_float(row, "latency_ms") for row in rows], 0.95
                )
            ),
        },
        "verdict_invariance_rate": fmean(
            _row_float(row, "verdict_invariant") for row in rows
        ),
        "mean_uncertainty_delta": fmean(
            _row_float(row, "uncertainty_delta") for row in rows
        ),
        "mean_conflict_delta": fmean(
            _row_float(row, "conflict_delta") for row in rows
        ),
        "memory_match_rate": len(memory_matches) / count,
        "memory_top1_relevance": (
            fmean(
                _row_float(row, "memory_top1_relevant")
                for row in top1_scored
            )
            if top1_scored
            else None
        ),
        "memory_topk_relevance": (
            fmean(
                _row_float(row, "memory_topk_relevant")
                for row in top1_scored
            )
            if top1_scored
            else None
        ),
        "deliberation_trigger_rate": fmean(
            float(int(row["deliberation_count"]) > 0) for row in rows
        ),
        "supplemental_dispatch_rate": fmean(
            _row_float(row, "supplemental_dispatch") for row in rows
        ),
        "compliance_failure_count": sum(
            _row_bool(row, "compliance_failure") for row in rows
        ),
        "audit_completion_rate": fmean(
            _row_float(row, "audit_complete") for row in rows
        ),
        "blocked_field_violation_count": sum(
            _row_bool(row, "blocked_field_violation") for row in rows
        ),
        "ood_signature_invariance_rate": fmean(
            _row_float(row, "ood_signature_invariant") for row in rows
        ),
        "classification": _classification_metrics(rows),
        "grouped_bootstrap_95ci": _bootstrap_intervals(
            rows,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
    }


def evaluate_future_framework_paper(
    record_factory: Callable[[], Iterable[FlowRecord]],
    output_dir: str | Path,
    *,
    dataset: str,
    expected_sample_count: int,
    modes: Iterable[str] = FUTURE_EVALUATION_MODES,
    detector_backend: str = "learned",
    model_dir: str | Path | None = None,
    ood_policy: str = "hybrid",
    ood_gate_dir: str | Path | None = None,
    memory_dir: str | Path | None = None,
    memory_reference_groups: dict[str, str] | None = None,
    resume: bool = True,
    checkpoint_every: int = 1000,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    max_workers: int = 4,
    frozen_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_modes = list(modes)
    unknown = set(selected_modes) - set(FUTURE_EVALUATION_MODES)
    if unknown:
        raise ValueError(f"unknown future evaluation modes: {sorted(unknown)}")
    if "baseline" not in selected_modes:
        selected_modes.insert(0, "baseline")
    frozen_paths = frozen_paths or {}
    hashes_before = hash_artifact_paths(frozen_paths)
    _json_dump(target / "frozen_artifact_hashes_before.json", hashes_before)
    run_manifest = {
        "schema_version": "1.0",
        "experiment": "future_framework_paper_v1",
        "dataset": dataset,
        "expected_sample_count": expected_sample_count,
        "modes": selected_modes,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "detector_backend": detector_backend,
        "ood_policy": ood_policy,
        "memory_is_advisory": True,
        "automatic_retraining": False,
        "external_llm_used": False,
    }
    _json_dump(target / "run_manifest.json", run_manifest)
    baseline_path = target / "baseline" / "predictions.csv"
    mode_summaries: dict[str, Any] = {}
    for mode in selected_modes:
        mode_dir = target / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        output_path = mode_dir / "predictions.csv"
        existing_rows = _load_prediction_rows(output_path) if resume else []
        completed = {row["sample_id"] for row in existing_rows}
        if not resume and output_path.exists():
            output_path.unlink()
        baseline = _baseline_index(baseline_path) if mode != "baseline" else {}
        if mode != "baseline" and len(baseline) != expected_sample_count:
            raise RuntimeError("baseline must complete before augmented modes")
        flags = FUTURE_EVALUATION_MODES[mode]
        engine = build_default_engine(
            field_audit_mode="legacy",
            max_workers=max_workers,
            detector_backend=detector_backend,
            model_dir=model_dir,
            ood_policy=ood_policy,
            ood_gate_dir=ood_gate_dir,
            future_flags=flags,
            memory_dir=(
                memory_dir
                if flags.memory and memory_dir is not None
                else mode_dir / "empty_memory"
            ),
        )
        handle = output_path.open(
            "a", encoding="utf-8-sig", newline=""
        )
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        if output_path.stat().st_size == 0:
            writer.writeheader()
        processed = len(completed)
        try:
            for record in record_factory():
                if record.sample_id in completed:
                    continue
                started = time.perf_counter()
                report, audit = engine.analyze(record)
                latency = (time.perf_counter() - started) * 1000
                snapshot = _decision_snapshot(report)
                signature = _ood_signature(report)
                baseline_row = baseline.get(record.sample_id)
                invariant = (
                    True
                    if mode == "baseline"
                    else snapshot == baseline_row["snapshot"]
                )
                ood_invariant = (
                    True
                    if mode == "baseline"
                    else signature == baseline_row["ood_signature"]
                )
                artifacts = audit.future_artifacts
                hint = artifacts.memory_hint
                query_group = (
                    f"{record.labels.get('binary', '')}|"
                    f"{record.labels.get('family', '')}"
                )
                refs = hint.similar_case_refs if hint else []
                relevance = (
                    [
                        memory_reference_groups.get(ref) == query_group
                        for ref in refs
                    ]
                    if memory_reference_groups is not None
                    else []
                )
                event_types = {event.event_type for event in audit.events}
                base_snapshot = (
                    baseline_row["snapshot"] if baseline_row else snapshot
                )
                row = {
                    "mode": mode,
                    "sample_id": record.sample_id,
                    "group_key": _record_group(record, dataset),
                    "label_binary": record.labels.get("binary", ""),
                    "label_family": record.labels.get("family", ""),
                    "application": record.labels.get("application", ""),
                    "cipher_condition": record.labels.get(
                        "cipher_condition", ""
                    ),
                    "browser": record.labels.get("browser", ""),
                    "capture_date": record.provenance.get(
                        "capture_date", ""
                    ),
                    **snapshot,
                    "agent_calls": len(report.agent_results),
                    "latency_ms": latency,
                    "coverage_indicator": int(
                        report.verdict.value in {"benign", "malicious"}
                    ),
                    "alert_indicator": int(
                        report.verdict.value in {"malicious", "suspicious"}
                    ),
                    "human_review_indicator": int(
                        report.verdict.value in {"suspicious", "unknown"}
                    ),
                    "verdict_invariant": int(invariant),
                    "uncertainty_delta": (
                        snapshot["uncertainty"]
                        - float(base_snapshot["uncertainty"])
                    ),
                    "conflict_delta": (
                        float(base_snapshot["conflict_score"])
                        - snapshot["conflict_score"]
                    ),
                    "memory_status": hint.status if hint else "disabled",
                    "memory_top1_ref": refs[0] if refs else "",
                    "memory_top1_relevant": (
                        int(relevance[0]) if relevance else ""
                    ),
                    "memory_topk_relevant": (
                        int(any(relevance)) if relevance else ""
                    ),
                    "deliberation_count": len(artifacts.deliberations),
                    "supplemental_dispatch": int(
                        mode == "full_active"
                        and any(
                            item.status == "completed"
                            and item.recommended_dispatch
                            for item in artifacts.deliberations
                        )
                    ),
                    "compliance_status": (
                        artifacts.compliance.status
                        if artifacts.compliance
                        else "disabled"
                    ),
                    "compliance_failure": int(
                        artifacts.compliance is not None
                        and artifacts.compliance.status != "compliant"
                    ),
                    "audit_complete": int(
                        REQUIRED_AUDIT_EVENTS <= event_types
                    ),
                    "blocked_field_violation": int(
                        _blocked_violation(audit)
                    ),
                    "ood_signature": signature,
                    "ood_signature_invariant": int(ood_invariant),
                }
                writer.writerow(row)
                processed += 1
                if processed % checkpoint_every == 0:
                    handle.flush()
                    _json_dump(
                        mode_dir / "checkpoint.json",
                        {
                            "mode": mode,
                            "completed_count": processed,
                            "last_sample_id": record.sample_id,
                        },
                    )
        finally:
            handle.flush()
            handle.close()
        rows = _load_prediction_rows(output_path)
        if len(rows) != expected_sample_count:
            raise RuntimeError(
                f"{mode} completed {len(rows)} of {expected_sample_count}"
            )
        if len({row["sample_id"] for row in rows}) != expected_sample_count:
            raise RuntimeError(f"{mode} predictions contain duplicate IDs")
        summary = _summarize_mode(
            rows,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        summary["feature_flags"] = flags.model_dump(mode="json")
        _json_dump(mode_dir / "summary.json", summary)
        mode_summaries[mode] = summary
    hashes_after = hash_artifact_paths(frozen_paths)
    _json_dump(target / "frozen_artifact_hashes_after.json", hashes_after)
    result = {
        **run_manifest,
        "modes": mode_summaries,
        "frozen_artifacts_unchanged": hashes_before == hashes_after,
        "boundaries": {
            "memory_is_agent_evidence": False,
            "llm_is_classifier": False,
            "reflection_modifies_current_verdict": False,
            "critic_outputs_labels": False,
            "automatic_retraining": False,
        },
    }
    _json_dump(target / "summary.json", result)
    return result


def build_ustc_offline_oracle_memory(
    validation_factory: Callable[[], Iterable[FlowRecord]],
    output_dir: str | Path,
    *,
    max_cases: int = 2000,
    seed: int = 42,
    detector_backend: str = "learned",
    model_dir: str | Path | None = None,
    ood_policy: str = "hybrid",
    ood_gate_dir: str | Path | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    eligible_path = target / "eligible_validation_cases.csv"
    engine = build_default_engine(
        field_audit_mode="legacy",
        max_workers=max_workers,
        detector_backend=detector_backend,
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
    )
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with eligible_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "stratum", "verdict", "priority"],
        )
        writer.writeheader()
        for record in validation_factory():
            report, _ = engine.analyze(record)
            if report.verdict.value not in {"suspicious", "unknown"}:
                continue
            stratum = (
                f"{record.labels.get('binary', '')}|"
                f"{record.labels.get('family', '')}|"
                f"{record.provenance.get('source_file', '')}"
            )
            priority = _hash_priority(seed, stratum, record.sample_id)
            candidates[stratum].append((priority, record.sample_id))
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "stratum": stratum,
                    "verdict": report.verdict.value,
                    "priority": priority,
                }
            )
    counts = {group: len(rows) for group, rows in candidates.items()}
    target_count = min(max_cases, sum(counts.values()))
    allocation = _allocate_stratified(counts, target_count)
    selected = {
        sample_id
        for group, rows in candidates.items()
        for _, sample_id in sorted(rows)[: allocation[group]]
    }
    memory = JsonlCaseMemory(target / "memory")
    reference_groups: dict[str, str] = {}
    selected_rows = []
    for record in validation_factory():
        if record.sample_id not in selected:
            continue
        report, audit = engine.analyze(record)
        case = memory.append_case(
            record,
            report,
            source_dataset="USTC-TFC2016-validation-offline-oracle",
            audit=audit,
        )
        outcome = str(record.labels.get("binary", "unknown"))
        feedback = HumanFeedbackRecord(
            feedback_id=f"offline-{case.case_id}",
            case_id=case.case_id,
            reviewer="offline_oracle_ustc_v1",
            review_type="confirm",
            confirmed_outcome=outcome,
            reason=(
                "Offline simulation using the frozen USTC validation label; "
                "this is not a human-factors result."
            ),
        )
        memory.record_feedback(feedback, audit=audit)
        group = (
            f"{record.labels.get('binary', '')}|"
            f"{record.labels.get('family', '')}"
        )
        reference_groups[case.case_id] = group
        selected_rows.append(
            {
                "sample_id": record.sample_id,
                "case_id": case.case_id,
                "reference_group": group,
            }
        )
    payload = {
        "schema_version": "1.0",
        "experiment": "ustc_offline_oracle_memory_v1",
        "reviewer": "offline_oracle_ustc_v1",
        "human_factors_claim": False,
        "source_split": "validation",
        "test_split_used_for_memory": False,
        "selected_case_count": len(selected_rows),
        "max_cases": max_cases,
        "seed": seed,
        "reference_groups": reference_groups,
        "selected_cases": selected_rows,
    }
    _json_dump(target / "reference_index.json", payload)
    return payload


def aggregate_future_paper_evaluation(
    runs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    expected_counts: dict[str, int],
    locked_test_tuning_used: bool = False,
    test_count: int,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summaries = {
        name: json.loads(
            (Path(path) / "summary.json").read_text(encoding="utf-8")
        )
        for name, path in runs.items()
    }
    comparison_rows = []
    failures: list[str] = []
    for name, summary in summaries.items():
        baseline = summary["modes"]["baseline"]
        if baseline["sample_count"] != expected_counts[name]:
            failures.append(f"{name}: unexpected sample count")
        if not summary["frozen_artifacts_unchanged"]:
            failures.append(f"{name}: frozen artifacts changed")
        for mode, values in summary["modes"].items():
            if values["compliance_failure_count"]:
                failures.append(f"{name}/{mode}: compliance failure")
            if values["blocked_field_violation_count"]:
                failures.append(f"{name}/{mode}: blocked field violation")
            if values["audit_completion_rate"] != 1.0:
                failures.append(f"{name}/{mode}: incomplete audit chain")
            if values["ood_signature_invariance_rate"] != 1.0:
                failures.append(f"{name}/{mode}: OOD signature changed")
            if (
                mode in PAPER_SHADOW_MODES
                and values["verdict_invariance_rate"] != 1.0
            ):
                failures.append(f"{name}/{mode}: shadow verdict changed")
            comparison_rows.append(
                {
                    "dataset": name,
                    "mode": mode,
                    "sample_count": values["sample_count"],
                    "coverage_delta": (
                        values["coverage"] - baseline["coverage"]
                    ),
                    "human_review_rate_delta": (
                        values["human_review_rate"]
                        - baseline["human_review_rate"]
                    ),
                    "average_agent_calls_delta": (
                        values["average_agent_calls"]
                        - baseline["average_agent_calls"]
                    ),
                    "latency_mean_delta_ms": (
                        values["latency_ms"]["mean"]
                        - baseline["latency_ms"]["mean"]
                    ),
                    "uncertainty_delta": values["mean_uncertainty_delta"],
                    "conflict_delta": values["mean_conflict_delta"],
                    "verdict_invariance_rate": values[
                        "verdict_invariance_rate"
                    ],
                }
            )
    if locked_test_tuning_used:
        failures.append("CipherSpectrum locked_test participated in tuning")
    with (target / "paired_comparisons.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(comparison_rows[0]),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    payload = {
        "schema_version": "1.0",
        "experiment": "future_framework_paper_v1",
        "acceptance_status": "passed" if not failures else "failed",
        "test_count": test_count,
        "expected_counts": expected_counts,
        "total_expected_samples": sum(expected_counts.values()),
        "locked_test_tuning_used": locked_test_tuning_used,
        "failures": failures,
        "runs": summaries,
        "claims": {
            "supports": [
                "routing efficiency",
                "conflict-resolution utility",
                "memory retrieval relevance under offline oracle simulation",
                "audit and OOD safety",
                "verdict invariance of explanatory components",
            ],
            "does_not_support": [
                "production readiness",
                "real human-review utility",
                "LLM classification gains",
                "Memory or RAG classification gains",
                "automatic retraining",
            ],
        },
    }
    _json_dump(target / "aggregate_metrics.json", summaries)
    _json_dump(target / "acceptance_report.json", payload)
    lines = [
        "# Future Framework Paper Evaluation v1",
        "",
        f"- Acceptance: `{payload['acceptance_status']}`",
        f"- Tests: `{test_count} passed`",
        f"- Evaluated samples: `{sum(expected_counts.values())}`",
        "- Locked test used for tuning: `false`",
        "",
        "## Runs",
        "",
        "| Dataset | Samples | Modes |",
        "|---|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['modes']['baseline']['sample_count']} | "
            f"{len(summary['modes'])} |"
        )
    lines.extend(
        [
            "",
            "This report evaluates orchestration, memory retrieval, conflict "
            "handling, audit safety, OOD invariance, and operational cost. "
            "It does not treat Memory, RAG, or LLM output as detection evidence.",
            "",
        ]
    )
    (target / "FUTURE_FRAMEWORK_PAPER_EVALUATION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return payload
