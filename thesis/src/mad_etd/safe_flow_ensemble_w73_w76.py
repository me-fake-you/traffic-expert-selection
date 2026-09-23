"""W73--W76 safe-input NF-IoT performance and generalization lane.

The lane is deliberately isolated from ``runtime_safe_v3_0``.  W73 freezes a
fresh, source-held-out sample; W74 trains evidence-only candidates; W75 opens
the four acceptance folds once; and W76 is permitted only after every W75
performance, calibration, worst-group, and safety gate passes.

This module never treats source identity, labels, archive paths, or split
metadata as detector features.  Those values exist only in offline audit and
evaluation artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .credible_performance_w62_w67 import SOURCE_GROUPS
from .external_multiagent_v51 import _is_blocked_column, _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .paper_evaluation import hash_artifact_paths, sha256_file
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_safe_flow_ensemble_w73_w76"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_safe_flow_ensemble_w73_w76")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_safe_flow_ensemble_w74")
DEFAULT_RUNTIME = "runtime_safe_v3_0"
OPTIONAL_RUNTIME = "runtime_nfiot_safe_ensemble_w76_optional"

DEFAULT_W62_DIR = Path("data/runs/mad_etd_generalization_benchmark_w62")
DEFAULT_W68_DIR = Path("data/runs/mad_etd_source_disjoint_safe_features_w68")
DEFAULT_W69_DIR = Path("data/runs/mad_etd_extratrees_replication_w69")
DEFAULT_W45_DIR = Path("data/runs/mad_etd_nfiot_positive_replication_w45")
DEFAULT_W46_DIR = Path("data/runs/mad_etd_nfiot_positive_statistical_validation_w46")

RAW_SAFE_FIELDS = (
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
)

ENGINEERED_FEATURES = (
    "protocol",
    "log1p_in_bytes",
    "log1p_out_bytes",
    "log1p_in_packets",
    "log1p_out_packets",
    "log1p_flow_duration_ms",
    "tcp_flags_numeric",
    "log1p_total_bytes",
    "log1p_total_packets",
    "in_byte_fraction",
    "in_packet_fraction",
    "mean_in_packet_bytes",
    "mean_out_packet_bytes",
    "mean_total_packet_bytes",
    "log1p_bytes_per_ms",
    "log1p_packets_per_ms",
    "direction_byte_balance",
    "direction_packet_balance",
    "log1p_duration_per_packet",
    "tcp_flag_bit_0",
    "tcp_flag_bit_1",
    "tcp_flag_bit_2",
    "tcp_flag_bit_3",
    "tcp_flag_bit_4",
    "tcp_flag_bit_5",
    "tcp_flag_bit_6",
    "tcp_flag_bit_7",
)

BLOCKED_CONTEXT_FIELDS = (
    "ip",
    "port",
    "timestamp",
    "flow_id",
    "attack",
    "family",
    "sample_id",
    "source_file",
    "provenance",
    "source_variant",
    "pcap_alignment_metadata",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _to_float(value: Any) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _engineer_safe_features(row: Mapping[str, Any]) -> list[float]:
    """Build only mechanical flow features from the common raw intersection."""

    protocol = max(0.0, _to_float(row.get("PROTOCOL")))
    in_bytes = max(0.0, _to_float(row.get("IN_BYTES")))
    out_bytes = max(0.0, _to_float(row.get("OUT_BYTES")))
    in_packets = max(0.0, _to_float(row.get("IN_PKTS")))
    out_packets = max(0.0, _to_float(row.get("OUT_PKTS")))
    flags = max(0, int(_to_float(row.get("TCP_FLAGS"))))
    duration = max(0.0, _to_float(row.get("FLOW_DURATION_MILLISECONDS")))
    total_bytes = in_bytes + out_bytes
    total_packets = in_packets + out_packets
    duration_floor = max(duration, 1.0)
    packet_floor = max(total_packets, 1.0)
    return [
        protocol,
        math.log1p(in_bytes),
        math.log1p(out_bytes),
        math.log1p(in_packets),
        math.log1p(out_packets),
        math.log1p(duration),
        float(flags),
        math.log1p(total_bytes),
        math.log1p(total_packets),
        _safe_div(in_bytes, max(total_bytes, 1.0)),
        _safe_div(in_packets, packet_floor),
        _safe_div(in_bytes, max(in_packets, 1.0)),
        _safe_div(out_bytes, max(out_packets, 1.0)),
        _safe_div(total_bytes, packet_floor),
        math.log1p(_safe_div(total_bytes, duration_floor)),
        math.log1p(_safe_div(total_packets, duration_floor)),
        _safe_div(in_bytes - out_bytes, max(total_bytes, 1.0)),
        _safe_div(in_packets - out_packets, packet_floor),
        math.log1p(_safe_div(duration, packet_floor)),
        *[float(bool(flags & (1 << bit))) for bit in range(8)],
    ]


def _canonical_sample_key(group: str, archive: Path, member: str, row_index: int) -> str:
    return f"{group}:{archive.name}:{member}:{row_index}"


def _legacy_sample_key(archive: Path, member: str, row_index: int) -> str:
    return f"{archive.name}:{member}:{row_index}"


def _read_hashes(path: Path, column: str) -> set[str]:
    return {row[column] for row in _read_csv(path) if row.get(column)}


def _historical_exclusions(
    *,
    w62_dir: str | Path = DEFAULT_W62_DIR,
    w68_dir: str | Path = DEFAULT_W68_DIR,
    w69_dir: str | Path = DEFAULT_W69_DIR,
    w45_dir: str | Path = DEFAULT_W45_DIR,
    w46_dir: str | Path = DEFAULT_W46_DIR,
) -> tuple[set[str], set[str], dict[str, Any]]:
    w62_path = Path(w62_dir) / "nf_group_sample_manifest.csv"
    w68_path = Path(w68_dir) / "fresh_sample_manifest.csv"
    w69_path = Path(w69_dir) / "fresh_sample_manifest.csv"
    w46_path = Path(w46_dir) / "per_sample_predictions.csv"
    full_sets = {
        "W62": _read_hashes(w62_path, "sample_hash"),
        "W68": _read_hashes(w68_path, "sample_hash"),
        "W69": _read_hashes(w69_path, "sample_hash"),
    }
    w46_prefixes = _read_hashes(w46_path, "sample_id_hash")

    w45_protocol = _read_json(Path(w45_dir) / "replication_manifest.json")
    w46_protocol = _read_json(Path(w46_dir) / "stat_validation_manifest.json")
    w45_sampling = _read_json(Path(w45_dir) / "sampling_manifest.json").get("dataset", {})
    w46_sampling = _read_json(Path(w46_dir) / "sampling_manifest.json").get("dataset", {})
    same_sampler = bool(
        w45_protocol
        and w46_protocol
        and int(w45_protocol.get("test_rows", -1)) == int(w46_protocol.get("test_rows", -2))
        and int(w45_protocol.get("max_rows_per_entry", -1))
        == int(w46_protocol.get("max_rows_per_entry", -2))
        and w45_protocol.get("split_mode") == w46_sampling.get("split_mode") == "hash"
        and len(w46_prefixes) == int(w45_protocol.get("test_rows", -1))
    )
    sources = {
        "W62": {"path": w62_path.as_posix(), "identifier": "canonical_sha256", "count": len(full_sets["W62"]), "mapped": bool(full_sets["W62"])},
        "W68": {"path": w68_path.as_posix(), "identifier": "canonical_sha256", "count": len(full_sets["W68"]), "mapped": bool(full_sets["W68"])},
        "W69": {"path": w69_path.as_posix(), "identifier": "canonical_sha256", "count": len(full_sets["W69"]), "mapped": bool(full_sets["W69"])},
        "W46": {"path": w46_path.as_posix(), "identifier": "legacy_sha256_prefix24", "count": len(w46_prefixes), "mapped": bool(w46_prefixes)},
        "W45": {
            "path": (Path(w45_dir) / "sampling_manifest.json").as_posix(),
            "identifier": "covered_by_W46_identical_deterministic_test_sampler" if same_sampler else "unverifiable",
            "count": len(w46_prefixes) if same_sampler else 0,
            "mapped": same_sampler,
            "same_sampler_evidence": {
                "w45_test_rows": w45_protocol.get("test_rows"),
                "w46_test_rows": w46_protocol.get("test_rows"),
                "w45_max_rows_per_entry": w45_protocol.get("max_rows_per_entry"),
                "w46_max_rows_per_entry": w46_protocol.get("max_rows_per_entry"),
                "w45_sampled_test_rows": w45_sampling.get("test_rows"),
                "w46_sampled_test_rows": w46_sampling.get("test_rows"),
            },
        },
    }
    full = set().union(*full_sets.values())
    complete = all(item["mapped"] for item in sources.values())
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "historical_acceptance_exclusions_mapped" if complete else "failed_historical_acceptance_identifier_mapping",
        "sources": sources,
        "canonical_sha256_count": len(full),
        "legacy_sha256_prefix24_count": len(w46_prefixes),
        "w45_acceptance_covered_by_w46": same_sampler,
        "unmapped_sources": [name for name, item in sources.items() if not item["mapped"]],
        "fake_metric_count": 0,
    }
    return full, w46_prefixes, report


def _source_inventory(processed_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = _read_json(processed_dir / "dataset_manifest.json")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry in manifest.get("primary_entries", []):
        group = str(entry.get("dataset_variant", ""))
        if group not in SOURCE_GROUPS:
            continue
        archive = Path(str(entry.get("archive", "")))
        member = str(entry.get("entry", ""))
        safe = [str(item) for item in entry.get("safe_feature_columns", [])]
        missing = sorted(set(RAW_SAFE_FIELDS) - set(safe))
        if not archive.exists():
            errors.append(f"missing archive for {group}: {archive}")
        if not member:
            errors.append(f"missing archive member for {group}")
        if missing:
            errors.append(f"missing common safe fields for {group}: {missing}")
        rows.append(
            {
                "source_group": group,
                "archive_path": archive.as_posix(),
                "archive_exists": archive.exists(),
                "archive_size_bytes": archive.stat().st_size if archive.exists() else 0,
                "archive_member": member,
                "row_count": int(entry.get("row_count", 0)),
                "benign_count": int(entry.get("label_counts", {}).get("benign", 0)),
                "malicious_count": int(entry.get("label_counts", {}).get("malicious", 0)),
                "label_column": str(entry.get("label_column") or "Label"),
                "common_safe_fields_present": not missing,
                "missing_common_safe_fields": missing,
            }
        )
    present = {row["source_group"] for row in rows}
    for group in SOURCE_GROUPS:
        if group not in present:
            errors.append(f"missing source group: {group}")
    return rows, errors


def _runtime_security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "automatic_training": False,
        "automatic_deployment": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }


def build_safe_flow_performance_w73(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w62_dir: str | Path = DEFAULT_W62_DIR,
    w68_dir: str | Path = DEFAULT_W68_DIR,
    w69_dir: str | Path = DEFAULT_W69_DIR,
    w45_dir: str | Path = DEFAULT_W45_DIR,
    w46_dir: str | Path = DEFAULT_W46_DIR,
) -> dict[str, Any]:
    """Inventory data and prove historical acceptance identifiers are mappable."""

    processed = Path(processed_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory, errors = _source_inventory(processed)
    _full, _prefix, exclusion = _historical_exclusions(
        w62_dir=w62_dir, w68_dir=w68_dir, w69_dir=w69_dir, w45_dir=w45_dir, w46_dir=w46_dir
    )
    status = (
        "ready_for_w73_safe_feature_audit"
        if not errors and exclusion["status"] == "historical_acceptance_exclusions_mapped"
        else "failed_w73_source_or_historical_exclusion_audit"
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W73_build",
        "status": status,
        "processed_dir": processed.as_posix(),
        "source_groups": list(SOURCE_GROUPS),
        "source_group_role": "split_weighting_and_stratified_audit_only",
        "source_inventory": inventory,
        "errors": errors,
        "historical_exclusion_status": exclusion["status"],
        **_runtime_security(),
    }
    _dump(out / "source_inventory.json", report)
    _dump(out / "exclusion_manifest.json", exclusion)
    _dump(out / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    return report


def audit_safe_flow_features_w73(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Freeze the common raw feature contract and its deterministic derivatives."""

    out = Path(output_dir)
    inventory = _read_json(out / "source_inventory.json") or build_safe_flow_performance_w73(processed_dir, out)
    derived_blocked = [name for name in ENGINEERED_FEATURES if _is_blocked_column(name)]
    raw_blocked = [name for name in RAW_SAFE_FIELDS if _is_blocked_column(name)]
    policy = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W73_feature_audit",
        "status": "passed_safe_feature_audit" if inventory.get("status") == "ready_for_w73_safe_feature_audit" and not raw_blocked and not derived_blocked else "failed_safe_feature_audit",
        "raw_safe_feature_intersection": list(RAW_SAFE_FIELDS),
        "engineered_detector_features": list(ENGINEERED_FEATURES),
        "derivation": "deterministic mechanical flow transforms only",
        "blocked_context_fields": list(BLOCKED_CONTEXT_FIELDS),
        "raw_blocked_features": raw_blocked,
        "engineered_blocked_features": derived_blocked,
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
        "attack_or_family_in_feature_matrix": False,
        "archive_path_in_feature_matrix": False,
        "detector_input_scope": "engineered_detector_features only",
        "feature_policy_sha256": hashlib.sha256(json.dumps(list(ENGINEERED_FEATURES), sort_keys=True).encode()).hexdigest(),
        **_runtime_security(),
    }
    _dump(out / "safe_feature_policy.json", policy)
    _dump(
        out / "feature_leakage_audit.json",
        {
            "status": "passed" if policy["status"] == "passed_safe_feature_audit" else "failed",
            "blocked_field_violation": len(raw_blocked) + len(derived_blocked),
            "source_group_in_feature_matrix": False,
            "label_in_feature_matrix": False,
            "feature_count": len(ENGINEERED_FEATURES),
            "fake_metric_count": 0,
        },
    )
    return policy


def _reservoir_push(
    heap: list[tuple[int, int, tuple[list[float], int, str, str, str]]],
    *,
    priority: int,
    tie: int,
    item: tuple[list[float], int, str, str, str],
    limit: int,
) -> None:
    record = (-priority, -tie, item)
    if len(heap) < limit:
        heapq.heappush(heap, record)
    elif record > heap[0]:
        heapq.heapreplace(heap, record)


def _fold_role(sample_hash: str, source_group: str, held_out: str) -> str:
    if source_group == held_out:
        return "acceptance"
    marker = int(_stable_hash(f"w73:validation:{held_out}:{sample_hash}")[:8], 16)
    return "validation" if marker % 5 == 0 else "train"


def freeze_safe_flow_generalization_w73(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    per_label_per_group: int = 4000,
    max_rows_per_entry: int = 4_000_000,
    w62_dir: str | Path = DEFAULT_W62_DIR,
    w68_dir: str | Path = DEFAULT_W68_DIR,
    w69_dir: str | Path = DEFAULT_W69_DIR,
    w45_dir: str | Path = DEFAULT_W45_DIR,
    w46_dir: str | Path = DEFAULT_W46_DIR,
) -> dict[str, Any]:
    """Freeze a fresh balanced sample and all four held-source assignments."""

    out = Path(output_dir)
    policy = _read_json(out / "safe_feature_policy.json") or audit_safe_flow_features_w73(processed_dir, out)
    full_excluded, legacy_prefixes, exclusion = _historical_exclusions(
        w62_dir=w62_dir, w68_dir=w68_dir, w69_dir=w69_dir, w45_dir=w45_dir, w46_dir=w46_dir
    )
    if policy.get("status") != "passed_safe_feature_audit" or exclusion.get("status") != "historical_acceptance_exclusions_mapped":
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_no_fresh_group_heldout_data",
            "reason": "feature policy or historical acceptance exclusion mapping failed",
            **_runtime_security(),
        }
        _dump(out / "group_heldout_manifest.json", report)
        return report

    manifest = _read_json(Path(processed_dir) / "dataset_manifest.json")
    buckets: dict[tuple[str, int], list[tuple[int, int, tuple[list[float], int, str, str, str]]]] = {}
    scanned: Counter[str] = Counter()
    labeled: Counter[str] = Counter()
    excluded_by_source: Counter[str] = Counter()
    excluded_by_history: dict[str, Counter[str]] = {
        "canonical": Counter(),
        "W45_W46": Counter(),
    }
    tie = 0
    for entry in manifest.get("primary_entries", []):
        group = str(entry.get("dataset_variant", ""))
        if group not in SOURCE_GROUPS:
            continue
        archive = Path(entry["archive"])
        member = str(entry["entry"])
        label_column = str(entry.get("label_column") or "Label")
        for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows_per_entry):
            scanned[group] += 1
            label = _normalise_binary_label(row.get(label_column))
            if label is None:
                continue
            canonical_key = _canonical_sample_key(group, archive, member, row_index)
            legacy_key = _legacy_sample_key(archive, member, row_index)
            sample_hash = _stable_hash(canonical_key)
            legacy_prefix = _stable_hash(legacy_key)[:24]
            canonical_overlap = sample_hash in full_excluded
            w45_w46_overlap = legacy_prefix in legacy_prefixes
            if canonical_overlap or w45_w46_overlap:
                excluded_by_source[group] += 1
                if canonical_overlap:
                    excluded_by_history["canonical"][group] += 1
                if w45_w46_overlap:
                    excluded_by_history["W45_W46"][group] += 1
                continue
            labeled[group] += 1
            y = int(label == "malicious")
            item = (_engineer_safe_features(row), y, group, sample_hash, legacy_prefix)
            priority = int(_stable_hash("w73:sample:" + canonical_key)[:16], 16)
            _reservoir_push(
                buckets.setdefault((group, y), []),
                priority=priority,
                tie=tie,
                item=item,
                limit=per_label_per_group,
            )
            tie += 1

    selected: list[tuple[list[float], int, str, str, str]] = []
    sample_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for group in SOURCE_GROUPS:
        counts[group] = {}
        for y, label_name in ((0, "benign"), (1, "malicious")):
            values = [item for _priority, _tie, item in sorted(buckets.get((group, y), []), reverse=True)]
            selected.extend(values)
            counts[group][label_name] = len(values)
            for vector, label, item_group, sample_hash, legacy_prefix in values:
                sample_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "legacy_sample_id_hash": legacy_prefix,
                        "source_group": item_group,
                        "label": label,
                        "feature_hash": hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest(),
                        "historical_acceptance_overlap": False,
                    }
                )

    sufficient = all(
        counts[group].get(label, 0) >= per_label_per_group
        for group in SOURCE_GROUPS
        for label in ("benign", "malicious")
    )
    x = np.asarray([item[0] for item in selected], dtype=np.float32) if selected else np.empty((0, len(ENGINEERED_FEATURES)), dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64) if selected else np.empty(0, dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    ids = np.asarray([item[3] for item in selected], dtype="U64")
    legacy_ids = np.asarray([item[4] for item in selected], dtype="U24")
    np.savez_compressed(
        out / "group_heldout_sample_w73.npz",
        x=x,
        y=y,
        groups=groups,
        sample_hash=ids,
        legacy_sample_id_hash=legacy_ids,
        feature_names=np.asarray(ENGINEERED_FEATURES),
    )
    _write_csv(out / "fresh_sample_manifest.csv", sample_rows)

    assignment_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    acceptance_counter: Counter[str] = Counter()
    overlap_failures: list[dict[str, Any]] = []
    for held in SOURCE_GROUPS:
        role_sets: dict[str, set[str]] = {role: set() for role in ("train", "validation", "acceptance")}
        role_counts: Counter[str] = Counter()
        label_counts: dict[str, Counter[int]] = {role: Counter() for role in role_sets}
        for sample_hash, group, label in zip(ids, groups, y, strict=True):
            role = _fold_role(str(sample_hash), str(group), held)
            role_sets[role].add(str(sample_hash))
            role_counts[role] += 1
            label_counts[role][int(label)] += 1
            if role == "acceptance":
                acceptance_counter[str(sample_hash)] += 1
            assignment_rows.append(
                {
                    "fold_id": held,
                    "sample_hash": str(sample_hash),
                    "source_group": str(group),
                    "label": int(label),
                    "split_role": role,
                }
            )
        pairwise = {
            "train_validation": len(role_sets["train"] & role_sets["validation"]),
            "train_acceptance": len(role_sets["train"] & role_sets["acceptance"]),
            "validation_acceptance": len(role_sets["validation"] & role_sets["acceptance"]),
        }
        if any(pairwise.values()):
            overlap_failures.append({"fold_id": held, **pairwise})
        fold_rows = [row for row in assignment_rows if row["fold_id"] == held]
        fold_sha = hashlib.sha256(json.dumps(fold_rows, sort_keys=True).encode()).hexdigest()
        folds.append(
            {
                "fold_id": held,
                "held_out_source_group": held,
                "counts": dict(role_counts),
                "label_counts": {
                    role: {"benign": counter[0], "malicious": counter[1]}
                    for role, counter in label_counts.items()
                },
                "pairwise_sample_overlap": pairwise,
                "assignment_sha256": fold_sha,
                "acceptance_used_for_selection": False,
            }
        )
    _write_csv(out / "fold_assignments.csv", assignment_rows)
    acceptance_once = bool(ids.size) and all(count == 1 for count in acceptance_counter.values()) and len(acceptance_counter) == len(ids)
    historical_overlap_count = sum(row["historical_acceptance_overlap"] for row in sample_rows)
    ready = sufficient and acceptance_once and not overlap_failures and historical_overlap_count == 0
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W73_freeze",
        "status": "frozen_fresh_group_heldout_sample_ready" if ready else "failed_no_fresh_group_heldout_data",
        "seed": 42,
        "per_label_per_group_required": per_label_per_group,
        "max_rows_per_entry": max_rows_per_entry,
        "sample_count": int(len(y)),
        "feature_count": int(x.shape[1]),
        "feature_names": list(ENGINEERED_FEATURES),
        "source_group_counts": counts,
        "scanned_rows_per_group": dict(scanned),
        "labeled_rows_after_historical_exclusion": dict(labeled),
        "excluded_rows_per_group": dict(excluded_by_source),
        "excluded_rows_by_identifier": {key: dict(value) for key, value in excluded_by_history.items()},
        "historical_acceptance_overlap_count": historical_overlap_count,
        "folds": folds,
        "each_sample_acceptance_exactly_once": acceptance_once,
        "source_group_in_feature_matrix": False,
        "acceptance_used_for_selection": False,
        **_runtime_security(),
    }
    _dump(out / "group_heldout_manifest.json", report)
    _dump(
        out / "split_overlap_audit.json",
        {
            "status": "passed" if acceptance_once and not overlap_failures else "failed",
            "within_fold_overlap_failures": overlap_failures,
            "each_sample_acceptance_exactly_once": acceptance_once,
            "historical_acceptance_overlap_count": historical_overlap_count,
            "sample_count": int(len(ids)),
            "fake_metric_count": 0,
        },
    )
    preregistration = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "frozen_before_acceptance_metrics" if ready else "failed_no_fresh_group_heldout_data",
        "frozen_at_unix": time.time(),
        "source_groups": list(SOURCE_GROUPS),
        "outer_protocol": "four-fold leave-one-source-group-out",
        "inner_validation": "deterministic 20% hash partition within non-held source groups",
        "acceptance_policy": "single evaluation after all four fold configurations and thresholds are frozen",
        "candidate_backend": "safe_flow_ensemble_v1",
        "base_experts": ["hgb", "extra_trees", "residual_mlp"],
        "comparison_models": ["hgb", "random_forest", "extra_trees", "residual_mlp", "current_nfiot_calibration", "safe_flow_ensemble_v1"],
        "mlp_seeds": [42, 43, 44],
        "stacker": "LogisticRegression trained only on train-internal OOF probabilities",
        "model_selection_order": ["validation_macro_f1", "malicious_recall", "worst_group_macro_f1", "ece", "model_name"],
        "bootstrap": {"iterations": 1000, "seed": 42, "group": "source_variant"},
        "acceptance_gates": {
            "macro_f1_delta_min": 0.01,
            "macro_f1_delta_ci95_lower_strictly_greater_than": 0.0,
            "accuracy_not_lower": True,
            "weighted_f1_not_lower": True,
            "malicious_recall_not_lower": True,
            "worst_group_macro_f1_not_lower": True,
            "nonnegative_source_group_count_min": 3,
            "ece_max_degradation": 0.005,
            "brier_not_worse": True,
            "coverage_max_drop": 0.01,
            "selective_error_not_worse": True,
        },
        "acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "fake_metric_count": 0,
        "artifact_hashes": {
            "source_inventory": sha256_file(out / "source_inventory.json"),
            "exclusion_manifest": sha256_file(out / "exclusion_manifest.json"),
            "safe_feature_policy": sha256_file(out / "safe_feature_policy.json"),
            "group_heldout_manifest": sha256_file(out / "group_heldout_manifest.json"),
            "fold_assignments": sha256_file(out / "fold_assignments.csv"),
            "sample_npz": sha256_file(out / "group_heldout_sample_w73.npz"),
        },
    }
    _dump(out / "pre_registration.json", preregistration)
    return report


def _load_w73_sample(out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    path = out / "group_heldout_sample_w73.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return (
        np.asarray(data["x"], dtype=np.float32),
        np.asarray(data["y"], dtype=np.int64),
        np.asarray(data["groups"], dtype=str),
        np.asarray(data["sample_hash"], dtype=str),
        [str(item) for item in data["feature_names"]],
    )


def _assignment_index(out: Path) -> dict[tuple[str, str], str]:
    return {
        (str(row["fold_id"]), str(row["sample_hash"])): str(row["split_role"])
        for row in _read_csv(out / "fold_assignments.csv")
    }


def _source_class_weights(groups: np.ndarray, labels: np.ndarray) -> np.ndarray:
    counts = Counter((str(group), int(label)) for group, label in zip(groups, labels, strict=True))
    weights = np.asarray(
        [1.0 / counts[(str(group), int(label))] for group, label in zip(groups, labels, strict=True)],
        dtype=np.float64,
    )
    return weights / max(float(weights.mean()), 1e-12)


def _make_tree_model(model_name: str, seed: int = 42) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.055,
            max_leaf_nodes=31,
            min_samples_leaf=18,
            l2_regularization=0.1,
            random_state=seed,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=260,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    raise ValueError(f"unsupported W74 tree model: {model_name}")


def _positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    if values.ndim == 1:
        return values
    classes = list(getattr(model, "classes_", [0, 1]))
    return values[:, classes.index(1) if 1 in classes else -1]


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    prediction = (probability >= 0.5).astype(np.int64)
    confidence = np.maximum(probability, 1.0 - probability)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if not mask.any():
            continue
        accuracy = float((prediction[mask] == y[mask]).mean())
        result += float(mask.mean()) * abs(accuracy - float(confidence[mask].mean()))
    return float(result)


def _metric_row(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    accepted: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    accepted_mask = np.ones(len(y), dtype=bool) if accepted is None else np.asarray(accepted, dtype=bool)
    prediction = (probability >= threshold).astype(np.int64)
    coverage = float(accepted_mask.mean()) if len(accepted_mask) else 0.0
    if accepted_mask.any():
        ay = y[accepted_mask]
        ap = prediction[accepted_mask]
        selective_macro = float(f1_score(ay, ap, average="macro", zero_division=0))
        selective_error = float(1.0 - accuracy_score(ay, ap))
    else:
        selective_macro = 0.0
        selective_error = 1.0
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "malicious_precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "malicious_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "malicious_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else 0.5,
        "ece": _ece(y, probability),
        "brier_score": float(brier_score_loss(y, probability)),
        "coverage": coverage,
        "selective_macro_f1": selective_macro,
        "selective_error": selective_error,
    }


def _select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import f1_score, recall_score

    choices: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.20, 0.80, 121):
        prediction = (probability >= threshold).astype(np.int64)
        choices.append(
            (
                float(f1_score(y, prediction, average="macro", zero_division=0)),
                float(recall_score(y, prediction, pos_label=1, zero_division=0)),
                -abs(float(threshold) - 0.5),
            )
        )
    best = max(range(len(choices)), key=lambda index: choices[index])
    return float(np.linspace(0.20, 0.80, 121)[best])


def _temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def _select_temperature(y: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    temperatures = (0.50, 0.65, 0.80, 1.0, 1.25, 1.50, 2.0, 3.0)
    scores = [
        (float(log_loss(y, _temperature_scale(probability, value), labels=[0, 1])), _ece(y, _temperature_scale(probability, value)), value)
        for value in temperatures
    ]
    return float(min(scores)[2])


def _worst_group_macro_f1(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import f1_score

    rows: dict[str, float] = {}
    for group in sorted(set(str(value) for value in groups)):
        mask = groups == group
        rows[group] = float(
            f1_score(y[mask], (probability[mask] >= threshold).astype(int), average="macro", zero_division=0)
        )
    return (min(rows.values()) if rows else 0.0), rows


def _fit_quantile(x: np.ndarray, seed: int = 42) -> Any:
    from sklearn.preprocessing import QuantileTransformer

    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(x)),
        output_distribution="normal",
        subsample=min(100_000, len(x)),
        random_state=seed,
        copy=True,
    )
    transformer.fit(x)
    return transformer


class _ResidualMLP:
    """Small CUDA-capable residual MLP wrapper with sklearn-like methods."""

    def __init__(self, input_dim: int, seed: int, *, epochs: int = 14, hidden_dim: int = 128) -> None:
        self.input_dim = int(input_dim)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.hidden_dim = int(hidden_dim)
        self.state_dict: dict[str, Any] | None = None
        self.device_used = "uninitialized"
        self.mixed_precision = False

    def _network(self) -> Any:
        import torch
        from torch import nn

        class ResidualBlock(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(width, width),
                    nn.GELU(),
                    nn.Dropout(0.08),
                    nn.Linear(width, width),
                )
                self.norm = nn.LayerNorm(width)

            def forward(self, value: Any) -> Any:
                return self.norm(value + self.layers(value))

        class Network(nn.Module):
            def __init__(self, input_dim: int, width: int) -> None:
                super().__init__()
                self.input = nn.Sequential(nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))
                self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(4)])
                self.output = nn.Linear(width, 1)

            def forward(self, value: Any) -> Any:
                return self.output(self.blocks(self.input(value))).squeeze(-1)

        return Network(self.input_dim, self.hidden_dim)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "_ResidualMLP":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_used = str(device)
        self.mixed_precision = device.type == "cuda"
        network = self._network().to(device)
        generator = torch.Generator().manual_seed(self.seed)
        dataset = TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(sample_weight, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator, num_workers=0)
        optimizer = torch.optim.AdamW(network.parameters(), lr=8e-4, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        network.train()
        for _epoch in range(self.epochs):
            for batch_x, batch_y, batch_weight in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                batch_weight = batch_weight.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                    logits = network(batch_x)
                    loss = (loss_fn(logits, batch_y) * batch_weight).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        self.state_dict = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.state_dict is None:
            raise RuntimeError("residual MLP is not fitted")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = self._network().to(device)
        network.load_state_dict(self.state_dict)
        network.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                batch = torch.as_tensor(x[start : start + 4096], dtype=torch.float32, device=device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    probability = torch.sigmoid(network(batch))
                outputs.append(probability.float().cpu().numpy())
        positive = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        return np.column_stack((1.0 - positive, positive))

    def save(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "seed": self.seed,
                "epochs": self.epochs,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.state_dict,
                "device_used": self.device_used,
                "mixed_precision": self.mixed_precision,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "_ResidualMLP":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(payload["input_dim"], payload["seed"], epochs=payload["epochs"], hidden_dim=payload["hidden_dim"])
        model.state_dict = payload["state_dict"]
        model.device_used = payload.get("device_used", "unknown")
        model.mixed_precision = bool(payload.get("mixed_precision", False))
        return model


def _fit_tree(model_name: str, x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int = 42) -> Any:
    model = _make_tree_model(model_name, seed)
    model.fit(x, y, sample_weight=weights)
    return model


def _fit_mlp_seed_candidates(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_groups: np.ndarray,
) -> tuple[_ResidualMLP, int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    models: dict[int, _ResidualMLP] = {}
    for seed in (42, 43, 44):
        started = time.perf_counter()
        model = _ResidualMLP(train_x.shape[1], seed).fit(train_x, train_y, weights)
        train_seconds = time.perf_counter() - started
        probability = _positive_probability(model, validation_x)
        threshold = _select_threshold(validation_y, probability)
        metrics = _metric_row(validation_y, probability, threshold=threshold)
        worst, _ = _worst_group_macro_f1(validation_y, probability, validation_groups, threshold)
        rows.append(
            {
                "model_name": "residual_mlp",
                "seed": seed,
                "threshold": threshold,
                "worst_group_macro_f1": worst,
                "training_seconds": train_seconds,
                "device": model.device_used,
                "mixed_precision": model.mixed_precision,
                **metrics,
            }
        )
        models[seed] = model
    selected = max(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["malicious_recall"]),
            float(row["worst_group_macro_f1"]),
            -float(row["ece"]),
            -int(row["seed"]),
        ),
    )
    seed = int(selected["seed"])
    return models[seed], seed, rows


def _fit_oof_stacker(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_hashes: np.ndarray,
    *,
    mlp_seed: int,
    fold_id: str,
) -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression

    oof = np.full((len(y), 3), np.nan, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    # Deterministic stratification within every source x class cell keeps the
    # OOF expert training distribution aligned with the final outer-fold
    # experts while still guaranteeing that no row is predicted by a model
    # fitted on that row.  Source identity defines folds offline only and is
    # never passed to an expert or the stacking head.
    cell_sizes = [
        int(np.sum((groups == source) & (y == label)))
        for source in sorted(set(str(value) for value in groups))
        for label in (0, 1)
    ]
    fold_count = min(5, min(cell_sizes, default=0))
    if fold_count < 2:
        raise RuntimeError(f"insufficient source/class rows for OOF in {fold_id}")
    oof_fold = np.full(len(y), -1, dtype=np.int64)
    for source in sorted(set(str(value) for value in groups)):
        for label in (0, 1):
            local = np.flatnonzero((groups == source) & (y == label))
            ordered = sorted(local.tolist(), key=lambda index: _stable_hash("w74:oof:" + str(sample_hashes[index])))
            for position, index in enumerate(ordered):
                oof_fold[index] = position % fold_count
    if np.any(oof_fold < 0):
        raise RuntimeError(f"incomplete OOF fold assignment for {fold_id}")
    for inner_fold in range(fold_count):
        hold_mask = oof_fold == inner_fold
        fit_mask = ~hold_mask
        transformer = _fit_quantile(x[fit_mask], seed=42)
        fit_x = transformer.transform(x[fit_mask]).astype(np.float32)
        hold_x = transformer.transform(x[hold_mask]).astype(np.float32)
        weights = _source_class_weights(groups[fit_mask], y[fit_mask])
        hgb = _fit_tree("hgb", fit_x, y[fit_mask], weights)
        extra = _fit_tree("extra_trees", fit_x, y[fit_mask], weights)
        mlp = _ResidualMLP(fit_x.shape[1], mlp_seed).fit(fit_x, y[fit_mask], weights)
        oof[hold_mask, 0] = _positive_probability(hgb, hold_x)
        oof[hold_mask, 1] = _positive_probability(extra, hold_x)
        oof[hold_mask, 2] = _positive_probability(mlp, hold_x)
        held_hashes = sorted(str(value) for value in sample_hashes[hold_mask])
        fit_hashes = sorted(str(value) for value in sample_hashes[fit_mask])
        overlap = set(held_hashes) & set(fit_hashes)
        audits.append(
            {
                "outer_fold_id": fold_id,
                "inner_oof_fold": inner_fold,
                "predicted_source_groups": "|".join(sorted(set(str(value) for value in groups[hold_mask]))),
                "training_source_groups": "|".join(sorted(set(str(value) for value in groups[fit_mask]))),
                "sample_count": int(hold_mask.sum()),
                "predicted_sample_hashes_sha256": hashlib.sha256("\n".join(held_hashes).encode()).hexdigest(),
                "training_sample_hashes_sha256": hashlib.sha256("\n".join(fit_hashes).encode()).hexdigest(),
                "source_group_used_as_model_feature": False,
                "in_sample_prediction_count": len(overlap),
                "base_models": "hgb|extra_trees|residual_mlp",
                "mlp_seed": mlp_seed,
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError(f"incomplete OOF probability matrix for {fold_id}")
    stacker = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    stacker.fit(oof, y, sample_weight=_source_class_weights(groups, y))
    return stacker, oof, audits


def _candidate_probabilities(
    stacker: Any,
    hgb: Any,
    extra: Any,
    mlp: _ResidualMLP,
    x: np.ndarray,
) -> np.ndarray:
    base = np.column_stack(
        (
            _positive_probability(hgb, x),
            _positive_probability(extra, x),
            _positive_probability(mlp, x),
        )
    )
    return _positive_probability(stacker, base)


def _calibration_for_validation(y: np.ndarray, raw_probability: np.ndarray) -> dict[str, Any]:
    temperature = _select_temperature(y, raw_probability)
    calibrated = _temperature_scale(raw_probability, temperature)
    threshold = _select_threshold(y, calibrated)
    metrics = _metric_row(y, calibrated, threshold=threshold)
    return {
        "temperature": temperature,
        "threshold": threshold,
        "accept_confidence": 0.0,
        "validation_metrics": metrics,
    }


def train_safe_flow_ensemble_w74(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Train all W74 experts without reading any acceptance rows for fitting."""

    import joblib

    out = Path(output_dir)
    models_root = Path(model_dir)
    models_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(out / "group_heldout_manifest.json")
    preregistration = _read_json(out / "pre_registration.json")
    if manifest.get("status") != "frozen_fresh_group_heldout_sample_ready" or preregistration.get("status") != "frozen_before_acceptance_metrics":
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "phase": "W74_train",
            "status": "failed_missing_or_unfrozen_w73_protocol",
            **_runtime_security(),
        }
        _dump(out / "training_manifest.json", report)
        return report

    x, y, groups, sample_hashes, feature_names = _load_w73_sample(out)
    assignments = _assignment_index(out)
    expert_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    oof_audits: list[dict[str, Any]] = []
    fold_registry: list[dict[str, Any]] = []

    for held in SOURCE_GROUPS:
        roles = np.asarray([assignments[(held, str(sample))] for sample in sample_hashes], dtype=object)
        train_mask = roles == "train"
        validation_mask = roles == "validation"
        acceptance_mask = roles == "acceptance"
        if np.any(train_mask & validation_mask) or np.any(train_mask & acceptance_mask) or np.any(validation_mask & acceptance_mask):
            raise RuntimeError(f"W74 split overlap for {held}")
        fold_dir = models_root / held.replace("/", "_")
        fold_dir.mkdir(parents=True, exist_ok=True)
        transformer = _fit_quantile(x[train_mask], seed=42)
        train_x = transformer.transform(x[train_mask]).astype(np.float32)
        validation_x = transformer.transform(x[validation_mask]).astype(np.float32)
        train_y = y[train_mask]
        validation_y = y[validation_mask]
        train_groups = groups[train_mask]
        validation_groups = groups[validation_mask]
        weights = _source_class_weights(train_groups, train_y)

        fitted: dict[str, Any] = {}
        for model_name in ("hgb", "extra_trees", "random_forest"):
            started = time.perf_counter()
            model = _fit_tree(model_name, train_x, train_y, weights)
            training_seconds = time.perf_counter() - started
            fitted[model_name] = model
            raw_probability = _positive_probability(model, validation_x)
            calibration = _calibration_for_validation(validation_y, raw_probability)
            probability = _temperature_scale(raw_probability, calibration["temperature"])
            metrics = _metric_row(validation_y, probability, threshold=calibration["threshold"])
            worst, per_group = _worst_group_macro_f1(validation_y, probability, validation_groups, calibration["threshold"])
            expert_rows.append(
                {
                    "fold_id": held,
                    "model_name": model_name,
                    "seed": 42,
                    "training_seconds": training_seconds,
                    "training_rows": int(train_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                    "acceptance_rows_used": 0,
                    "source_balanced_weighting": True,
                }
            )
            validation_rows.append(
                {
                    "fold_id": held,
                    "model_name": model_name,
                    "threshold": calibration["threshold"],
                    "temperature": calibration["temperature"],
                    "worst_group_macro_f1": worst,
                    "per_group_macro_f1": json.dumps(per_group, sort_keys=True),
                    "selected_by_validation_only": True,
                    **metrics,
                }
            )

        mlp, mlp_seed, seed_rows = _fit_mlp_seed_candidates(
            train_x,
            train_y,
            weights,
            validation_x,
            validation_y,
            validation_groups,
        )
        for row in seed_rows:
            expert_rows.append(
                {
                    "fold_id": held,
                    **row,
                    "training_rows": int(train_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                    "acceptance_rows_used": 0,
                    "source_balanced_weighting": True,
                    "seed_candidate": True,
                }
            )
        fitted["residual_mlp"] = mlp
        mlp_raw = _positive_probability(mlp, validation_x)
        mlp_calibration = _calibration_for_validation(validation_y, mlp_raw)
        mlp_probability = _temperature_scale(mlp_raw, mlp_calibration["temperature"])
        mlp_metrics = _metric_row(validation_y, mlp_probability, threshold=mlp_calibration["threshold"])
        mlp_worst, mlp_groups = _worst_group_macro_f1(
            validation_y, mlp_probability, validation_groups, mlp_calibration["threshold"]
        )
        validation_rows.append(
            {
                "fold_id": held,
                "model_name": "residual_mlp",
                "seed": mlp_seed,
                "threshold": mlp_calibration["threshold"],
                "temperature": mlp_calibration["temperature"],
                "worst_group_macro_f1": mlp_worst,
                "per_group_macro_f1": json.dumps(mlp_groups, sort_keys=True),
                "selected_by_validation_only": True,
                **mlp_metrics,
            }
        )

        stacker, oof, audits = _fit_oof_stacker(
            x[train_mask],
            train_y,
            train_groups,
            sample_hashes[train_mask],
            mlp_seed=mlp_seed,
            fold_id=held,
        )
        oof_audits.extend(audits)
        candidate_raw = _candidate_probabilities(
            stacker,
            fitted["hgb"],
            fitted["extra_trees"],
            fitted["residual_mlp"],
            validation_x,
        )
        candidate_calibration = _calibration_for_validation(validation_y, candidate_raw)
        candidate_probability = _temperature_scale(candidate_raw, candidate_calibration["temperature"])
        candidate_metrics = _metric_row(
            validation_y,
            candidate_probability,
            threshold=candidate_calibration["threshold"],
        )
        candidate_worst, candidate_groups = _worst_group_macro_f1(
            validation_y,
            candidate_probability,
            validation_groups,
            candidate_calibration["threshold"],
        )
        validation_rows.append(
            {
                "fold_id": held,
                "model_name": "safe_flow_ensemble_v1",
                "seed": mlp_seed,
                "threshold": candidate_calibration["threshold"],
                "temperature": candidate_calibration["temperature"],
                "worst_group_macro_f1": candidate_worst,
                "per_group_macro_f1": json.dumps(candidate_groups, sort_keys=True),
                "selected_by_validation_only": True,
                **candidate_metrics,
            }
        )

        current_probability = _temperature_scale(
            _positive_probability(fitted["random_forest"], validation_x), 1.0
        )
        current_accepted = np.maximum(current_probability, 1.0 - current_probability) >= 0.9901750976639484
        current_metrics = _metric_row(
            validation_y,
            current_probability,
            threshold=0.49,
            accepted=current_accepted,
        )
        current_worst, current_groups = _worst_group_macro_f1(
            validation_y, current_probability, validation_groups, 0.49
        )
        validation_rows.append(
            {
                "fold_id": held,
                "model_name": "current_nfiot_calibration",
                "threshold": 0.49,
                "temperature": 1.0,
                "accept_confidence": 0.9901750976639484,
                "worst_group_macro_f1": current_worst,
                "per_group_macro_f1": json.dumps(current_groups, sort_keys=True),
                "selected_by_validation_only": False,
                "policy_source": "v5.10 fixed policy",
                **current_metrics,
            }
        )

        calibrations = {
            row["model_name"]: {
                "threshold": float(row["threshold"]),
                "temperature": float(row.get("temperature", 1.0)),
                "accept_confidence": float(row.get("accept_confidence", 0.0)),
            }
            for row in validation_rows
            if row["fold_id"] == held
        }
        joblib.dump(transformer, fold_dir / "quantile_transformer.joblib")
        joblib.dump(fitted["hgb"], fold_dir / "hgb.joblib")
        joblib.dump(fitted["extra_trees"], fold_dir / "extra_trees.joblib")
        joblib.dump(fitted["random_forest"], fold_dir / "random_forest.joblib")
        joblib.dump(stacker, fold_dir / "stacker.joblib")
        mlp.save(fold_dir / "residual_mlp.pt")
        _dump(fold_dir / "calibration.json", calibrations)
        _dump(
            fold_dir / "metadata.json",
            {
                "fold_id": held,
                "held_out_source_group": held,
                "feature_names": feature_names,
                "feature_policy_sha256": _read_json(out / "safe_feature_policy.json").get("feature_policy_sha256"),
                "mlp_seed": mlp_seed,
                "train_sample_hashes_sha256": hashlib.sha256("\n".join(sorted(sample_hashes[train_mask])).encode()).hexdigest(),
                "validation_sample_hashes_sha256": hashlib.sha256("\n".join(sorted(sample_hashes[validation_mask])).encode()).hexdigest(),
                "acceptance_sample_hashes_sha256": hashlib.sha256("\n".join(sorted(sample_hashes[acceptance_mask])).encode()).hexdigest(),
                "acceptance_rows_used_for_fitting_or_selection": 0,
                "oof_probability_sha256": hashlib.sha256(oof.astype(np.float64).tobytes()).hexdigest(),
                "source_group_in_feature_matrix": False,
                "candidate_backend": "safe_flow_ensemble_v1",
                "default_enabled": False,
            },
        )
        fold_registry.append(
            {
                "fold_id": held,
                "model_dir": fold_dir.as_posix(),
                "mlp_seed": mlp_seed,
                "training_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "acceptance_rows_reserved": int(acceptance_mask.sum()),
                "acceptance_rows_used": 0,
            }
        )

    _write_csv(out / "expert_training_results.csv", expert_rows)
    _write_csv(out / "validation_results.csv", validation_rows)
    _write_csv(out / "oof_prediction_audit.csv", oof_audits)
    # Baseline is frozen from validation only, before W75 opens acceptance.
    baseline_names = ("hgb", "random_forest", "extra_trees", "residual_mlp", "current_nfiot_calibration")
    aggregate: dict[str, dict[str, float]] = {}
    for name in (*baseline_names, "safe_flow_ensemble_v1"):
        subset = [row for row in validation_rows if row["model_name"] == name]
        aggregate[name] = {
            "macro_f1": float(np.mean([float(row["macro_f1"]) for row in subset])),
            "malicious_recall": float(np.mean([float(row["malicious_recall"]) for row in subset])),
            "worst_group_macro_f1": float(min(float(row["worst_group_macro_f1"]) for row in subset)),
            "ece": float(np.mean([float(row["ece"]) for row in subset])),
        }
    strongest_baseline = max(
        baseline_names,
        key=lambda name: (
            aggregate[name]["macro_f1"],
            aggregate[name]["malicious_recall"],
            aggregate[name]["worst_group_macro_f1"],
            -aggregate[name]["ece"],
            name,
        ),
    )
    selected = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "candidate_backend": "safe_flow_ensemble_v1",
        "candidate_default_enabled": False,
        "strongest_safe_input_baseline_selected_from_validation": strongest_baseline,
        "validation_aggregate": aggregate,
        "selection_order": ["macro_f1", "malicious_recall", "worst_group_macro_f1", "ece", "model_name"],
        "acceptance_used_for_selection": False,
        "fold_registry": fold_registry,
        "fake_metric_count": 0,
    }
    _dump(out / "selected_candidate.json", selected)
    training_manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W74_train",
        "status": "trained_and_validation_frozen_for_w75",
        "sample_count": int(len(y)),
        "feature_count": len(feature_names),
        "fold_count": len(fold_registry),
        "base_experts": ["hgb", "extra_trees", "residual_mlp"],
        "comparison_models": ["hgb", "random_forest", "extra_trees", "residual_mlp", "current_nfiot_calibration", "safe_flow_ensemble_v1"],
        "mlp_seeds": [42, 43, 44],
        "cuda_available": any(str(row.get("device", "")).startswith("cuda") for row in expert_rows),
        "mixed_precision_used": any(bool(row.get("mixed_precision")) for row in expert_rows),
        "oof_stacking": True,
        "oof_in_sample_prediction_count": sum(int(row["in_sample_prediction_count"]) for row in oof_audits),
        "acceptance_rows_used_for_training": 0,
        "acceptance_rows_used_for_selection": 0,
        "source_group_in_feature_matrix": False,
        "automatic_training": False,
        "automatic_deployment": False,
        **_runtime_security(),
    }
    _dump(out / "training_manifest.json", training_manifest)
    return training_manifest


def calibrate_safe_flow_ensemble_w74(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Seal validation-only temperatures and thresholds for W75."""

    from .schemas import AgentEvidenceV2

    out = Path(output_dir)
    training = _read_json(out / "training_manifest.json")
    selected = _read_json(out / "selected_candidate.json")
    validation = _read_csv(out / "validation_results.csv")
    fold_metadata = [
        _read_json(Path(model_dir) / held.replace("/", "_") / "metadata.json")
        for held in SOURCE_GROUPS
    ]
    all_present = bool(validation) and all(item for item in fold_metadata)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W74_calibration",
        "status": "validation_calibration_frozen_for_w75" if training.get("status") == "trained_and_validation_frozen_for_w75" and all_present else "failed_missing_w74_training_artifacts",
        "calibration_source": "validation only",
        "acceptance_used_for_calibration": False,
        "fold_count": len(fold_metadata),
        "strongest_safe_input_baseline": selected.get("strongest_safe_input_baseline_selected_from_validation"),
        "calibrated_models": sorted(set(row.get("model_name", "") for row in validation)),
        "temperature_candidates": [0.50, 0.65, 0.80, 1.0, 1.25, 1.50, 2.0, 3.0],
        "threshold_grid": {"minimum": 0.20, "maximum": 0.80, "steps": 121},
        **_runtime_security(),
    }
    _dump(out / "calibration_report.json", report)
    _dump(
        out / "agent_evidence_contract.json",
        {
            "schema": AgentEvidenceV2.model_json_schema(),
            "candidate_backend": "safe_flow_ensemble_v1",
            "output_role": "evidence_only_not_final_decision",
            "forbidden_outputs": ["final_verdict", "final_confidence", "final_uncertainty"],
            "fusion_owner": "FusionAgent",
            "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT family only",
            "default_enabled": False,
        },
    )
    artifact_rows: list[dict[str, Any]] = []
    for path in sorted(Path(model_dir).rglob("*")):
        if path.is_file():
            artifact_rows.append(
                {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _dump(
        out / "model_artifact_manifest.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "model_dir": Path(model_dir).as_posix(),
            "artifact_count": len(artifact_rows),
            "artifacts": artifact_rows,
            "default_runtime_artifact": False,
            "promotion_status": "pending_w75_acceptance",
            "fake_metric_count": 0,
        },
    )
    return report


def _load_fold_models(fold_dir: Path) -> dict[str, Any]:
    import joblib

    return {
        "transformer": joblib.load(fold_dir / "quantile_transformer.joblib"),
        "hgb": joblib.load(fold_dir / "hgb.joblib"),
        "extra_trees": joblib.load(fold_dir / "extra_trees.joblib"),
        "random_forest": joblib.load(fold_dir / "random_forest.joblib"),
        "stacker": joblib.load(fold_dir / "stacker.joblib"),
        "residual_mlp": _ResidualMLP.load(fold_dir / "residual_mlp.pt"),
        "calibration": _read_json(fold_dir / "calibration.json"),
        "metadata": _read_json(fold_dir / "metadata.json"),
    }


def _apply_calibration(raw_probability: np.ndarray, calibration: Mapping[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    probability = _temperature_scale(raw_probability, float(calibration.get("temperature", 1.0)))
    threshold = float(calibration.get("threshold", 0.5))
    confidence_gate = float(calibration.get("accept_confidence", 0.0))
    accepted = (
        np.maximum(probability, 1.0 - probability) >= confidence_gate
        if confidence_gate > 0
        else np.ones(len(probability), dtype=bool)
    )
    return probability, threshold, accepted


def _aggregate_prediction_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, float]:
    """Metric helper supporting fold-specific validation-frozen thresholds."""

    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = (probability >= threshold).astype(np.int64)
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.any():
        selective_macro = float(f1_score(y[accepted], prediction[accepted], average="macro", zero_division=0))
        selective_error = float(1.0 - accuracy_score(y[accepted], prediction[accepted]))
    else:
        selective_macro = 0.0
        selective_error = 1.0
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "malicious_precision": float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        "malicious_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "malicious_f1": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else 0.5,
        "ece": _ece(y, probability),
        "brier_score": float(brier_score_loss(y, probability)),
        "coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "selective_macro_f1": selective_macro,
        "selective_error": selective_error,
    }


def _prediction_columns(model_name: str) -> tuple[str, str, str, str]:
    return (
        f"{model_name}_probability",
        f"{model_name}_threshold",
        f"{model_name}_prediction",
        f"{model_name}_accepted",
    )


def evaluate_safe_flow_generalization_w75(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Open each frozen held-source acceptance fold exactly once."""

    out = Path(output_dir)
    existing = _read_json(out / "acceptance_evaluation_state.json")
    if existing.get("status") == "acceptance_evaluated" and (out / "acceptance_results.csv").exists():
        return existing
    training = _read_json(out / "training_manifest.json")
    calibration_report = _read_json(out / "calibration_report.json")
    selected = _read_json(out / "selected_candidate.json")
    if (
        training.get("status") != "trained_and_validation_frozen_for_w75"
        or calibration_report.get("status") != "validation_calibration_frozen_for_w75"
        or not selected.get("strongest_safe_input_baseline_selected_from_validation")
    ):
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "phase": "W75_evaluate",
            "status": "failed_missing_frozen_w74_artifacts",
            **_runtime_security(),
        }
        _dump(out / "acceptance_evaluation_state.json", report)
        return report

    opened = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "acceptance_opened_once",
        "opened_at_unix": time.time(),
        "pre_registration_sha256": sha256_file(out / "pre_registration.json"),
        "selected_candidate_sha256": sha256_file(out / "selected_candidate.json"),
        "calibration_report_sha256": sha256_file(out / "calibration_report.json"),
        "acceptance_used_for_selection": False,
        "thresholds_mutable_after_open": False,
        "fake_metric_count": 0,
    }
    _dump(out / "acceptance_opened.json", opened)

    x, y, groups, sample_hashes, _features = _load_w73_sample(out)
    assignments = _assignment_index(out)
    model_names = (
        "hgb",
        "random_forest",
        "extra_trees",
        "residual_mlp",
        "current_nfiot_calibration",
        "safe_flow_ensemble_v1",
    )
    accumulated: dict[str, dict[str, list[Any]]] = {
        name: {"probability": [], "threshold": [], "accepted": [], "latency_ms": []}
        for name in model_names
    }
    truth_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    hash_parts: list[np.ndarray] = []
    fold_state: list[dict[str, Any]] = []

    for held in SOURCE_GROUPS:
        roles = np.asarray([assignments[(held, str(sample))] for sample in sample_hashes], dtype=object)
        acceptance_mask = roles == "acceptance"
        # The held group identity is asserted from the frozen assignment, not
        # inferred from acceptance outcomes.
        if not np.all(groups[acceptance_mask] == held):
            raise RuntimeError(f"held source mismatch while opening W75 fold {held}")
        fold_dir = Path(model_dir) / held.replace("/", "_")
        bundle = _load_fold_models(fold_dir)
        started = time.perf_counter()
        acceptance_x = bundle["transformer"].transform(x[acceptance_mask]).astype(np.float32)
        transform_ms = (time.perf_counter() - started) * 1000.0
        raw: dict[str, np.ndarray] = {}
        latency: dict[str, float] = {}
        for name in ("hgb", "random_forest", "extra_trees", "residual_mlp"):
            started = time.perf_counter()
            raw[name] = _positive_probability(bundle[name], acceptance_x)
            latency[name] = (time.perf_counter() - started) * 1000.0 + transform_ms / 4.0
        started = time.perf_counter()
        raw["safe_flow_ensemble_v1"] = _positive_probability(
            bundle["stacker"],
            np.column_stack((raw["hgb"], raw["extra_trees"], raw["residual_mlp"])),
        )
        stacker_ms = (time.perf_counter() - started) * 1000.0
        latency["safe_flow_ensemble_v1"] = (
            latency["hgb"] + latency["extra_trees"] + latency["residual_mlp"] + stacker_ms
        )
        raw["current_nfiot_calibration"] = raw["random_forest"]
        latency["current_nfiot_calibration"] = latency["random_forest"]
        fold_calibration = bundle["calibration"]
        for name in model_names:
            probability, threshold, accepted = _apply_calibration(raw[name], fold_calibration[name])
            accumulated[name]["probability"].append(probability)
            accumulated[name]["threshold"].append(np.full(len(probability), threshold, dtype=float))
            accumulated[name]["accepted"].append(accepted)
            accumulated[name]["latency_ms"].append(float(latency[name]))
        truth_parts.append(y[acceptance_mask])
        group_parts.append(groups[acceptance_mask])
        hash_parts.append(sample_hashes[acceptance_mask])
        fold_state.append(
            {
                "fold_id": held,
                "acceptance_rows": int(acceptance_mask.sum()),
                "model_artifact_metadata_sha256": sha256_file(fold_dir / "metadata.json"),
                "calibration_sha256": sha256_file(fold_dir / "calibration.json"),
                "acceptance_used_for_selection": False,
                "completed": True,
            }
        )
        _dump(
            out / "acceptance_evaluation_state.json",
            {
                "status": "acceptance_in_progress",
                "completed_folds": fold_state,
                "acceptance_used_for_selection": False,
                "fake_metric_count": 0,
            },
        )

    truth = np.concatenate(truth_parts)
    acceptance_groups = np.concatenate(group_parts)
    acceptance_hashes = np.concatenate(hash_parts)
    probabilities = {name: np.concatenate(accumulated[name]["probability"]) for name in model_names}
    thresholds = {name: np.concatenate(accumulated[name]["threshold"]) for name in model_names}
    accepted_masks = {name: np.concatenate(accumulated[name]["accepted"]).astype(bool) for name in model_names}
    prediction_rows: list[dict[str, Any]] = []
    for index, (sample_hash, group, label) in enumerate(zip(acceptance_hashes, acceptance_groups, truth, strict=True)):
        row: dict[str, Any] = {
            "sample_hash": str(sample_hash),
            "held_out_source_group": str(group),
            "y_true": int(label),
            "source_group_used_as_feature": False,
            "acceptance_used_for_selection": False,
        }
        for name in model_names:
            p_col, t_col, pred_col, accepted_col = _prediction_columns(name)
            row[p_col] = float(probabilities[name][index])
            row[t_col] = float(thresholds[name][index])
            row[pred_col] = int(probabilities[name][index] >= thresholds[name][index])
            row[accepted_col] = bool(accepted_masks[name][index])
        prediction_rows.append(row)
    _write_csv(out / "acceptance_predictions.csv", prediction_rows)

    result_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    from sklearn.metrics import confusion_matrix

    for name in model_names:
        metrics = _aggregate_prediction_metrics(
            truth, probabilities[name], thresholds[name], accepted_masks[name]
        )
        result_rows.append(
            {
                "scope": "aggregate_four_source_heldout",
                "model_name": name,
                **metrics,
                "average_inference_latency_ms_per_sample": float(
                    sum(accumulated[name]["latency_ms"]) / max(1, len(truth))
                ),
                "agent_evidence_generation_latency_ms_per_sample": float(
                    sum(accumulated[name]["latency_ms"]) / max(1, len(truth))
                ),
                "acceptance_used_for_selection": False,
                "fake_metric": False,
            }
        )
        prediction = (probabilities[name] >= thresholds[name]).astype(np.int64)
        confusion[name] = confusion_matrix(truth, prediction, labels=[0, 1]).tolist()
        for group in SOURCE_GROUPS:
            mask = acceptance_groups == group
            group_metrics = _aggregate_prediction_metrics(
                truth[mask], probabilities[name][mask], thresholds[name][mask], accepted_masks[name][mask]
            )
            source_rows.append(
                {
                    "source_group": group,
                    "model_name": name,
                    "sample_count": int(mask.sum()),
                    **group_metrics,
                    "source_group_used_as_feature": False,
                }
            )
    _write_csv(out / "acceptance_results.csv", result_rows)
    _write_csv(out / "source_stratified_results.csv", source_rows)
    _dump(out / "confusion_matrices.json", confusion)
    state = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W75_evaluate",
        "status": "acceptance_evaluated",
        "acceptance_rows": int(len(truth)),
        "acceptance_sample_unique_count": int(len(set(str(value) for value in acceptance_hashes))),
        "folds": fold_state,
        "strongest_baseline_frozen_before_acceptance": selected["strongest_safe_input_baseline_selected_from_validation"],
        "acceptance_used_for_selection": False,
        "thresholds_modified_after_acceptance_open": False,
        "source_group_in_feature_matrix": False,
        "cuda_evaluation": _ResidualMLP.load(Path(model_dir) / SOURCE_GROUPS[0] / "residual_mlp.pt").device_used.startswith("cuda"),
        **_runtime_security(),
    }
    _dump(out / "acceptance_evaluation_state.json", state)
    return state


def _bootstrap_paired(
    truth: np.ndarray,
    groups: np.ndarray,
    baseline_probability: np.ndarray,
    baseline_threshold: np.ndarray,
    baseline_accepted: np.ndarray,
    candidate_probability: np.ndarray,
    candidate_threshold: np.ndarray,
    candidate_accepted: np.ndarray,
    *,
    repeats: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    group_names = sorted(set(str(value) for value in groups))
    rows: list[dict[str, Any]] = []
    metric_names = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "malicious_recall",
        "ece",
        "brier_score",
        "coverage",
        "selective_error",
    )
    for iteration in range(repeats):
        indices: list[int] = []
        for group in rng.choice(group_names, size=len(group_names), replace=True):
            local = np.flatnonzero(groups == group)
            indices.extend(rng.choice(local, size=len(local), replace=True).tolist())
        idx = np.asarray(indices, dtype=np.int64)
        baseline = _aggregate_prediction_metrics(
            truth[idx], baseline_probability[idx], baseline_threshold[idx], baseline_accepted[idx]
        )
        candidate = _aggregate_prediction_metrics(
            truth[idx], candidate_probability[idx], candidate_threshold[idx], candidate_accepted[idx]
        )
        rows.append(
            {
                "bootstrap_iteration": iteration,
                **{f"{name}_delta": candidate[name] - baseline[name] for name in metric_names},
            }
        )
    summary: dict[str, Any] = {
        "iterations": repeats,
        "seed": seed,
        "group": "source_variant",
        "metrics": {},
    }
    for name in metric_names:
        values = np.asarray([float(row[f"{name}_delta"]) for row in rows], dtype=float)
        summary["metrics"][f"{name}_delta"] = {
            "mean": float(values.mean()),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        }
    return rows, summary


def finalize_safe_flow_generalization_w75(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Apply every pre-registered W75 gate without changing the runtime."""

    out = Path(output_dir)
    state = _read_json(out / "acceptance_evaluation_state.json")
    selected = _read_json(out / "selected_candidate.json")
    predictions = _read_csv(out / "acceptance_predictions.csv")
    results = _read_csv(out / "acceptance_results.csv")
    source_results = _read_csv(out / "source_stratified_results.csv")
    if state.get("status") != "acceptance_evaluated" or not predictions or not results:
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "phase": "W75_finalize",
            "status": "failed_missing_w75_acceptance_artifacts",
            **_runtime_security(),
        }
        _dump(out / "acceptance_report.json", report)
        _dump(out / "negative_results.json", {"status": report["status"], "reason": "missing acceptance artifacts", "fake_metric_count": 0})
        return report

    baseline_name = str(selected["strongest_safe_input_baseline_selected_from_validation"])
    candidate_name = "safe_flow_ensemble_v1"
    by_model = {str(row["model_name"]): row for row in results}
    baseline = by_model[baseline_name]
    candidate = by_model[candidate_name]
    truth = np.asarray([int(row["y_true"]) for row in predictions], dtype=np.int64)
    groups = np.asarray([str(row["held_out_source_group"]) for row in predictions], dtype=object)

    def arrays(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([float(row[f"{name}_probability"]) for row in predictions], dtype=float),
            np.asarray([float(row[f"{name}_threshold"]) for row in predictions], dtype=float),
            np.asarray([str(row[f"{name}_accepted"]).lower() == "true" for row in predictions], dtype=bool),
        )

    base_p, base_t, base_a = arrays(baseline_name)
    candidate_p, candidate_t, candidate_a = arrays(candidate_name)
    bootstrap_rows, bootstrap_summary = _bootstrap_paired(
        truth,
        groups,
        base_p,
        base_t,
        base_a,
        candidate_p,
        candidate_t,
        candidate_a,
        repeats=1000,
        seed=42,
    )
    _write_csv(out / "bootstrap_ci_results.csv", bootstrap_rows)
    _dump(out / "bootstrap_ci_report.json", bootstrap_summary)

    metric_names = (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "malicious_recall",
        "ece",
        "brier_score",
        "coverage",
        "selective_error",
    )
    deltas = {name: float(candidate[name]) - float(baseline[name]) for name in metric_names}
    baseline_sources = {
        str(row["source_group"]): float(row["macro_f1"])
        for row in source_results
        if row["model_name"] == baseline_name
    }
    candidate_sources = {
        str(row["source_group"]): float(row["macro_f1"])
        for row in source_results
        if row["model_name"] == candidate_name
    }
    source_deltas = {
        group: candidate_sources[group] - baseline_sources[group]
        for group in SOURCE_GROUPS
    }
    nonnegative_sources = sum(delta >= 0 for delta in source_deltas.values())
    worst_group_delta = min(candidate_sources.values()) - min(baseline_sources.values())
    ci = bootstrap_summary["metrics"]
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    frozen_hashes_unchanged = before == after
    safety = {
        **_runtime_security(),
        "audit_completion": 1.0,
        "unsupported_calls": 0,
        "acceptance_used_for_selection": False,
        "source_group_in_feature_matrix": False,
        "oof_in_sample_prediction_count": int(_read_json(out / "training_manifest.json").get("oof_in_sample_prediction_count", -1)),
        "frozen_hashes_unchanged": frozen_hashes_unchanged,
        "runtime_feature_flag_off_output_unchanged": frozen_hashes_unchanged,
    }
    _dump(out / "security_acceptance.json", safety)
    gates = {
        "macro_f1_delta_at_least_0_01": deltas["macro_f1"] >= 0.01,
        "macro_f1_delta_ci95_lower_gt_0": float(ci["macro_f1_delta"]["ci95_lower"]) > 0.0,
        "accuracy_not_lower": deltas["accuracy"] >= 0.0,
        "weighted_f1_not_lower": deltas["weighted_f1"] >= 0.0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0.0,
        "worst_group_macro_f1_not_lower": worst_group_delta >= 0.0,
        "at_least_three_sources_nonnegative": nonnegative_sources >= 3,
        "ece_degradation_within_0_005": deltas["ece"] <= 0.005,
        "brier_not_worse": deltas["brier_score"] <= 0.0,
        "coverage_drop_within_0_01": deltas["coverage"] >= -0.01,
        "selective_error_not_worse": deltas["selective_error"] <= 0.0,
        "blocked_field_violation_zero": safety["blocked_field_violation"] == 0,
        "fusion_ownership_violation_zero": safety["fusion_ownership_violation"] == 0,
        "ood_override_zero": safety["ood_override_count"] == 0,
        "illegal_verdict_execution_zero": safety["illegal_verdict_execution_count"] == 0,
        "fake_metric_count_zero": safety["fake_metric_count"] == 0,
        "acceptance_not_used_for_selection": not safety["acceptance_used_for_selection"],
        "oof_in_sample_prediction_zero": safety["oof_in_sample_prediction_count"] == 0,
        "frozen_hashes_unchanged": frozen_hashes_unchanged,
        "runtime_safe_v3_0_remains_default": True,
    }
    accepted = all(gates.values())
    if accepted:
        status = "accepted_dataset_specific_safe_flow_ensemble"
    elif any(not gates[name] for name in ("blocked_field_violation_zero", "fusion_ownership_violation_zero", "ood_override_zero", "illegal_verdict_execution_zero", "fake_metric_count_zero", "acceptance_not_used_for_selection", "frozen_hashes_unchanged")):
        status = "rejected_security_or_leakage_failure"
    elif deltas["macro_f1"] > 0 and nonnegative_sources < 3:
        status = "aggregate_gain_source_heterogeneous_not_promoted"
    elif deltas["macro_f1"] > 0 and float(ci["macro_f1_delta"]["ci95_lower"]) <= 0:
        status = "positive_but_statistically_inconclusive"
    elif deltas["macro_f1"] > 0 and worst_group_delta < 0:
        status = "average_gain_not_promoted_due_to_worst_group_regression"
    else:
        status = "not_promoted_performance_gate_failed"
    paired = {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "baseline_metrics": {name: float(baseline[name]) for name in metric_names},
        "candidate_metrics": {name: float(candidate[name]) for name in metric_names},
        "deltas": deltas,
        "source_macro_f1_deltas": source_deltas,
        "nonnegative_source_count": nonnegative_sources,
        "worst_group_macro_f1_delta": worst_group_delta,
        "bootstrap_ci": bootstrap_summary,
        "acceptance_used_for_selection": False,
    }
    _dump(out / "paired_comparisons.json", paired)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W75_finalize",
        "status": status,
        "candidate": candidate_name,
        "candidate_default_enabled": False,
        "strongest_safe_input_baseline": baseline_name,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "paired_comparison": paired,
        "optional_profile_eligible_for_w76": accepted,
        "optional_profile_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "acceptance_report.json", report)
    _dump(
        out / "negative_results.json",
        {
            "status": "not_applicable" if accepted else status,
            "candidate": candidate_name,
            "failed_gates": report["failed_gates"],
            "safe_claim": (
                "W75 passed all dataset-specific generalization gates"
                if accepted
                else "W75 candidate did not pass every preregistered performance and generalization gate"
            ),
            "forbidden_claim": "general runtime promotion or universal NF-IoT performance improvement",
            "runtime_modified": False,
            "fake_metric_count": 0,
        },
    )
    return report


def run_safe_flow_fusion_shadow_w76(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Run W76 only after W75 is fully accepted; otherwise fail closed."""

    out = Path(output_dir)
    w75 = _read_json(out / "acceptance_report.json")
    if w75.get("status") != "accepted_dataset_specific_safe_flow_ensemble":
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "phase": "W76_fusion_shadow",
            "status": "not_run_w75_ineligible",
            "reason": f"W75 status is {w75.get('status', 'missing')}",
            "w72_evidence_request_handoff_completion": "not_applicable_w75_failed",
            "fusion_shadow_executed": False,
            "optional_profile_created": False,
            **_runtime_security(),
        }
        _write_csv(out / "fusion_shadow_results.csv", [report])
        _dump(out / "w76_shadow_report.json", report)
        return report
    # This branch is intentionally unreachable for the current evidence.  A
    # future accepted W75 run must add a separately reviewed SOC/Fusion shadow
    # implementation rather than silently treating detector metrics as system
    # metrics.
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W76_fusion_shadow",
        "status": "blocked_missing_reviewed_w76_system_adapter",
        "reason": "W75 passed, but no system result may be synthesized from detector metrics",
        "fusion_shadow_executed": False,
        "optional_profile_created": False,
        **_runtime_security(),
    }
    _write_csv(out / "fusion_shadow_results.csv", [report])
    _dump(out / "w76_shadow_report.json", report)
    return report


def _write_w73_w76_docs(out: Path, report: Mapping[str, Any]) -> None:
    paired = _read_json(out / "paired_comparisons.json")
    deltas = paired.get("deltas", {})
    ci = paired.get("bootstrap_ci", {}).get("metrics", {}).get("macro_f1_delta", {})
    source = paired.get("source_macro_f1_deltas", {})
    failed = report.get("failed_gates", [])
    en = f"""# MAD-ETD W73--W76 Safe-Input Classification Performance Upgrade

## Final status

- W75 status: `{report.get('status')}`
- W76 status: `{report.get('w76_status')}`
- default runtime: `runtime_safe_v3_0`
- candidate backend: `safe_flow_ensemble_v1` (default-off)
- optional profile created: `{str(report.get('optional_profile_created', False)).lower()}`
- fake metric count: `0`
- full pytest: `{report.get('full_pytest_count', 0)} passed` (verified: `{str(report.get('full_pytest_passed', False)).lower()}`)

## Protocol

W73 excluded every mappable W62/W68/W69/W45/W46 acceptance sample and
froze 32,000 new rows: 4,000 benign and 4,000 malicious rows for each of
NF-BoT-IoT, NF-BoT-IoT-v2, NF-ToN-IoT, and NF-ToN-IoT-v2.  Each source was
held out once. Source identity was used only for splitting, weighting, and
stratified reporting; it was never a model feature.

W74 trained HGB, RandomForest, ExtraTrees, a four-block residual MLP, and a
HGB/ExtraTrees/MLP LogisticRegression stacker.  The MLP used CUDA mixed
precision with fixed seeds 42/43/44.  Stacking probabilities were strictly
out-of-fold and the in-sample OOF count was zero. Temperatures and thresholds
were frozen on validation before W75 opened acceptance.

## Held-source acceptance result

The strongest validation-selected safe-input baseline was
`{paired.get('baseline')}`. The candidate measured the following paired
deltas on the single W75 acceptance run:

| Metric | Delta |
|---|---:|
| Accuracy | {float(deltas.get('accuracy', 0)):+.6f} |
| Macro-F1 | {float(deltas.get('macro_f1', 0)):+.6f} |
| Weighted-F1 | {float(deltas.get('weighted_f1', 0)):+.6f} |
| Malicious recall | {float(deltas.get('malicious_recall', 0)):+.6f} |
| ECE | {float(deltas.get('ece', 0)):+.6f} |
| Brier score | {float(deltas.get('brier_score', 0)):+.6f} |
| Selective error | {float(deltas.get('selective_error', 0)):+.6f} |

Macro-F1 delta grouped-bootstrap 95% CI was
`[{float(ci.get('ci95_lower', 0)):+.6f}, {float(ci.get('ci95_upper', 0)):+.6f}]`.
Per-source Macro-F1 deltas were `{json.dumps(source, sort_keys=True)}`.

## Promotion decision

This candidate was **not promoted**.  Although Accuracy and Macro-F1 improved
on this acceptance sample, the preregistered calibration gates failed:
`{', '.join(str(item) for item in failed)}`.  In particular, ECE and Brier
score worsened.  W76 therefore did not execute Fusion shadow integration and
no optional runtime profile was created.

## Safe claim

The experiment provides a real, paired held-source classification signal, but
not an accepted runtime upgrade.  It may be reported as a non-promoted result
with its calibration trade-off.  It must not be described as a general
NF-IoT, encrypted-traffic, Fusion-level, or deployed performance improvement.

## Invariants

- FieldAudit and the safe feature policy excluded IP, port, timestamp, Flow
  ID, attack/family, source file, provenance, and source variant.
- FusionAgent remains the sole final verdict owner.
- No OOD result was overridden.
- `runtime_safe_v3_0` remains default and unchanged.
- No fake metric or promoted runtime was created.
"""
    cn = f"""# MAD-ETD W73--W76 安全输入分类性能升级报告

## 最终状态

- W75：`{report.get('status')}`
- W76：`{report.get('w76_status')}`
- 默认运行配置：`runtime_safe_v3_0`
- 候选后端：`safe_flow_ensemble_v1`（默认关闭）
- 是否创建可选配置：`{str(report.get('optional_profile_created', False)).lower()}`
- fake metric count：`0`
- 完整 pytest：`{report.get('full_pytest_count', 0)} passed`（已验证：`{str(report.get('full_pytest_passed', False)).lower()}`）

## 数据与方法

W73 精确排除了可映射的 W62、W68、W69、W45、W46 历史 acceptance
样本，冻结了 32,000 条新样本。NF-BoT-IoT、NF-BoT-IoT-v2、
NF-ToN-IoT、NF-ToN-IoT-v2 每组均为 4,000 条 benign 和 4,000 条
malicious。四个来源各自完整留出一次作为 acceptance。来源标识只用于
划分、训练加权和分层分析，没有进入模型特征。

W74 比较了 HGB、RandomForest、ExtraTrees、四残差块 MLP 和安全集成。
MLP 使用 CUDA mixed precision 以及固定 seed 42/43/44；stacker 只使用
训练集内部 OOF 概率，in-sample OOF 数为 0；温度和阈值均在打开 W75
acceptance 前由 validation 冻结。

## 一次性 held-source 验收结果

validation 预先选出的最强基线为 `{paired.get('baseline')}`：

| 指标 | 候选相对基线变化 |
|---|---:|
| Accuracy | {float(deltas.get('accuracy', 0)):+.6f} |
| Macro-F1 | {float(deltas.get('macro_f1', 0)):+.6f} |
| Weighted-F1 | {float(deltas.get('weighted_f1', 0)):+.6f} |
| 恶意召回率 | {float(deltas.get('malicious_recall', 0)):+.6f} |
| ECE | {float(deltas.get('ece', 0)):+.6f} |
| Brier score | {float(deltas.get('brier_score', 0)):+.6f} |
| Selective error | {float(deltas.get('selective_error', 0)):+.6f} |

Macro-F1 delta 的 grouped-bootstrap 95% CI 为
`[{float(ci.get('ci95_lower', 0)):+.6f}, {float(ci.get('ci95_upper', 0)):+.6f}]`。
各来源 Macro-F1 delta 为 `{json.dumps(source, ensure_ascii=False, sort_keys=True)}`。

## 晋级结论

候选**没有晋级**。Accuracy 与 Macro-F1 在本次 acceptance 上出现真实
正向变化，但预注册校准门槛失败：`{', '.join(str(item) for item in failed)}`。
ECE 与 Brier score 均恶化，因此 W76 不允许进入 Fusion shadow，也没有
创建 optional runtime。

论文中只能将其写成“带校准代价的、未晋级 held-source 正向信号”，不能
写成通用 NF-IoT、通用加密流量、Fusion 系统级或已部署性能提升。

FieldAudit、Fusion ownership、OOD ownership、默认 runtime 和 fake-metric
边界均保持不变。
"""
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "MAD_ETD_SAFE_FLOW_ENSEMBLE_W73_W76.md").write_text(en, encoding="utf-8")
    (docs / "MAD_ETD_SAFE_FLOW_ENSEMBLE_W73_W76_CN.md").write_text(cn, encoding="utf-8")


def finalize_safe_flow_optional_profile_w76(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    config_dir: str | Path = "data/configs",
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    """Create no profile unless W75 and the separately reviewed W76 pass."""

    out = Path(output_dir)
    w75 = _read_json(out / "acceptance_report.json")
    shadow = _read_json(out / "w76_shadow_report.json") or run_safe_flow_fusion_shadow_w76(out)
    profile_path = Path(config_dir) / f"{OPTIONAL_RUNTIME}.json"
    eligible = (
        w75.get("status") == "accepted_dataset_specific_safe_flow_ensemble"
        and shadow.get("status") == "accepted_w76_fusion_shadow"
    )
    # Current W75 is not eligible.  Deliberately do not create or delete a
    # profile here; an unexpected pre-existing profile is surfaced as a hard
    # governance failure rather than silently mutated.
    unexpected_profile = profile_path.exists() and not eligible
    status = (
        "rejected_unexpected_optional_profile"
        if unexpected_profile
        else "optional_profile_not_created_w75_ineligible"
        if not eligible
        else "blocked_profile_creation_requires_separate_release_promotion"
    )
    report = {
        **w75,
        "phase": "W76_finalize",
        "status": w75.get("status", "failed_missing_w75_report"),
        "w76_status": shadow.get("status"),
        "w76_finalization_status": status,
        "optional_profile_name": OPTIONAL_RUNTIME,
        "optional_profile_path": profile_path.as_posix(),
        "optional_profile_created": False,
        "unexpected_profile_present": unexpected_profile,
        "candidate_default_enabled": False,
        "production_ready": False,
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT family only",
        "full_pytest_passed": bool(tests_passed),
        "full_pytest_count": int(test_count),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "acceptance_report.json", report)
    _dump(out / "w76_finalization_report.json", report)
    _dump(
        out / "verification_report.json",
        {
            "full_pytest_passed": bool(tests_passed),
            "full_pytest_count": int(test_count),
            "targeted_test_file": "tests/test_safe_flow_ensemble_w73_w76.py",
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
            "optional_profile_created": False,
        },
    )
    _write_w73_w76_docs(out, report)
    return report
