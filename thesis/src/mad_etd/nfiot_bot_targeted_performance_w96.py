"""W96 NF-BoT-IoT-v2 full-coverage targeted-feature replication lane.

This lane is disjoint from W45--W95 auditable samples.  It reuses the W94/W95
model protocol but adds a validation-time malicious-recall floor before the
single acceptance opening.  The result remains dataset-specific and default-off.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .domain_robust_stats_w81 import _source_entries
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .nfiot_targeted_performance_w94_w95 import (
    BLOCKED_COLUMNS,
    DEFAULT_PROCESSED,
    SHORTCUT_EXCLUDED,
    _dump,
    _exclusions_through_w93,
    _native_feature_names,
    _native_vector,
    _push,
    _read_csv,
    _read_json,
    _security,
    _write_csv,
    evaluate_nfiot_targeted_models_w95,
    finalize_nfiot_targeted_performance_w95,
    run_targeted_feature_forensics_w94,
    train_nfiot_targeted_models_w95,
)
from .paper_evaluation import hash_artifact_paths
from .safe_flow_ensemble_w73_w76 import (
    ENGINEERED_FEATURES,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_nfiot_bot_targeted_performance_w96"
TARGET_VARIANT = "NF-BoT-IoT-v2"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfiot_bot_targeted_performance_w96")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_nfiot_bot_targeted_performance_w96")
DEFAULT_W93_DIR = Path("data/runs/mad_etd_nfiot_skill_positive_w93")
DEFAULT_W95_DIR = Path("data/runs/mad_etd_nfiot_targeted_performance_w94_w95")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_BOT_TARGETED_PERFORMANCE_W96.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_BOT_TARGETED_PERFORMANCE_W96_CN.md")
DEFAULT_PER_LABEL = 10_000
DEFAULT_MAX_ROWS = 6_000_000
SPLIT_STATUS = "w96_fresh_nf_bot_targeted_splits_frozen"
FEATURE_PACK_STATUS = "w96_nf_bot_targeted_feature_pack_locked"


def _role_w96(sample_hash: str) -> str:
    value = int(_stable_hash(f"w96:role:{sample_hash}")[:16], 16) / float(
        0xFFFFFFFFFFFFFFFF
    )
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "validation"
    return "acceptance"


def _exclusions_through_w95(
    w93_dir: str | Path,
    w95_dir: str | Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, through_w93 = _exclusions_through_w93(w93_dir)
    w95 = Path(w95_dir)
    rows = _read_csv(w95 / "fresh_sample_manifest.csv")
    report = _read_json(w95 / "acceptance_report.json")
    hashes = {row["sample_hash"] for row in rows if row.get("sample_hash")}
    legacy_hashes = {
        row["legacy_sample_id_hash"]
        for row in rows
        if row.get("legacy_sample_id_hash")
    }
    allowed_statuses = {
        "accepted_dataset_specific_full_coverage_accuracy_f1_result",
        "statistically_positive_full_coverage_signal_not_promoted",
        "not_promoted_w95_performance_or_safety_gate_failed",
    }
    mapped = bool(rows) and report.get("status") in allowed_statuses
    canonical.update(hashes)
    legacy.update(legacy_hashes)
    ready = (
        through_w93.get("status")
        == "all_auditable_nf_samples_through_w93_excluded"
        and mapped
    )
    return canonical, legacy, {
        "status": (
            "all_auditable_nf_samples_through_w95_excluded"
            if ready
            else "failed_w96_historical_exclusion_mapping"
        ),
        "through_w93": through_w93,
        "w95_status": report.get("status", "missing"),
        "w95_sample_count": len(hashes),
        "w95_legacy_count": len(legacy_hashes),
        "canonical_exclusion_count": len(canonical),
        "legacy_exclusion_count": len(legacy),
    }


def build_nfiot_bot_targeted_features_w96(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w93_dir: str | Path = DEFAULT_W93_DIR,
    w95_dir: str | Path = DEFAULT_W95_DIR,
    per_label: int = DEFAULT_PER_LABEL,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)
    canonical_exclusions, legacy_exclusions, ledger = _exclusions_through_w95(
        w93_dir, w95_dir
    )
    _dump(out / "historical_exclusion_ledger.json", ledger)

    processed = Path(processed_dir)
    entries, source_errors = _source_entries(processed)
    targets = [entry for entry in entries if entry.get("source_group") == TARGET_VARIANT]
    dataset_manifest = _read_json(processed / "dataset_manifest.json")
    metadata = [
        item
        for item in dataset_manifest.get("primary_entries", [])
        if item.get("dataset_variant") == TARGET_VARIANT
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset_scope": TARGET_VARIANT,
        "split_protocol": "fresh historically-disjoint stratified hash 60/20/20",
        "selection_split": "validation",
        "acceptance_split": "sealed and opened once",
        "performance_mode": "full_coverage_only",
        "validation_recall_floor": "candidate recall must not be below strongest common-safe baseline",
        "per_label": int(per_label),
        "max_rows": int(max_rows),
        "acceptance_used_for_selection": False,
        **_security(),
    }
    _dump(out / "protocol_manifest.json", protocol)
    if (
        source_errors
        or len(targets) != 1
        or len(metadata) != 1
        or ledger.get("status")
        != "all_auditable_nf_samples_through_w95_excluded"
    ):
        report = {
            **protocol,
            "status": "failed_w96_source_or_history_gate",
            "source_errors": source_errors,
            "target_entry_count": len(targets),
            "target_metadata_count": len(metadata),
            "history_status": ledger.get("status"),
        }
        _dump(out / "split_manifest.json", report)
        return report

    entry = {**targets[0], **metadata[0]}
    headers = list(entry["headers"])
    native_names = _native_feature_names(headers)
    blocked_seen = sorted(set(native_names) & BLOCKED_COLUMNS)
    policy = {
        "schema_version": "1.0",
        "dataset_scope": TARGET_VARIANT,
        "common_features": list(ENGINEERED_FEATURES),
        "native_candidate_features": native_names,
        "blocked_columns": sorted(BLOCKED_COLUMNS),
        "pre_registered_shortcut_exclusions": sorted(SHORTCUT_EXCLUDED),
        "label_usage": "split and evaluation only",
        "attack_usage": "diagnostic only",
        "blocked_fields_in_feature_matrix": blocked_seen,
        "blocked_field_violation": len(blocked_seen),
    }
    policy["policy_hash"] = hashlib.sha256(
        json.dumps(policy, sort_keys=True).encode()
    ).hexdigest()
    _dump(out / "safe_feature_policy.json", policy)
    if blocked_seen:
        report = {
            **protocol,
            "status": "failed_w96_blocked_feature_gate",
            "blocked_fields": blocked_seen,
        }
        _dump(out / "split_manifest.json", report)
        return report

    archive = Path(entry["archive"])
    member = str(entry["entry"])
    label_column = str(entry.get("label_column") or "Label")
    attack_column = str(entry.get("attack_column") or "Attack")
    buckets: dict[int, list[Any]] = {0: [], 1: []}
    scanned = excluded = labeled = 0
    tie = 0
    for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows):
        scanned += 1
        label = _normalise_binary_label(row.get(label_column))
        if label is None:
            continue
        canonical_key = _canonical_sample_key(
            TARGET_VARIANT, archive, member, row_index
        )
        legacy_key = _legacy_sample_key(archive, member, row_index)
        sample_hash = _stable_hash(canonical_key)
        legacy_prefix = _stable_hash(legacy_key)[:24]
        if (
            sample_hash in canonical_exclusions
            or legacy_prefix in legacy_exclusions
        ):
            excluded += 1
            continue
        labeled += 1
        y = int(label == "malicious")
        priority = int(_stable_hash("w96:fresh:" + canonical_key)[:16], 16)
        bucket = buckets[y]
        # Reservoir membership depends only on the stable sample key.  Avoid
        # materialising two feature vectors for rows that cannot enter the
        # bounded reservoir; this does not alter selection semantics.
        if len(bucket) >= per_label and (-priority, -tie) <= bucket[0][:2]:
            tie += 1
            continue
        item = (
            _engineer_safe_features(row),
            _native_vector(row, native_names),
            y,
            str(row.get(attack_column, "unknown")),
            sample_hash,
            legacy_prefix,
        )
        _push(buckets[y], priority=priority, tie=tie, item=item, limit=per_label)
        tie += 1

    selected = [
        item
        for label in (0, 1)
        for _priority, _tie, item in sorted(buckets[label], reverse=True)
    ]
    counts = {"benign": len(buckets[0]), "malicious": len(buckets[1])}
    if min(counts.values()) < per_label:
        report = {
            **protocol,
            "status": "failed_insufficient_fresh_w96_samples",
            "counts": counts,
            "scanned_rows": scanned,
            "excluded_rows": excluded,
        }
        _dump(out / "split_manifest.json", report)
        return report

    x_common = np.asarray([item[0] for item in selected], dtype=np.float32)
    x_native = np.asarray([item[1] for item in selected], dtype=np.float32)
    y = np.asarray([item[2] for item in selected], dtype=np.int64)
    attacks = np.asarray([item[3] for item in selected], dtype="U64")
    hashes = np.asarray([item[4] for item in selected], dtype="U64")
    legacy_hashes = np.asarray([item[5] for item in selected], dtype="U24")
    roles = np.asarray([_role_w96(str(value)) for value in hashes], dtype="U16")
    complete = all(
        np.any((roles == role) & (y == label))
        for role in ("train", "validation", "acceptance")
        for label in (0, 1)
    )
    train_validation = roles != "acceptance"
    acceptance = roles == "acceptance"
    sealed = out / "sealed_splits"
    sealed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sealed / "train_validation.npz",
        x_common=x_common[train_validation],
        x_native=x_native[train_validation],
        y=y[train_validation],
        attack=attacks[train_validation],
        sample_hash=hashes[train_validation],
        roles=roles[train_validation],
        common_feature_names=np.asarray(ENGINEERED_FEATURES),
        native_feature_names=np.asarray(native_names),
    )
    np.savez_compressed(
        sealed / "acceptance_sealed.npz",
        x_common=x_common[acceptance],
        x_native=x_native[acceptance],
        y=y[acceptance],
        attack=attacks[acceptance],
        sample_hash=hashes[acceptance],
        common_feature_names=np.asarray(ENGINEERED_FEATURES),
        native_feature_names=np.asarray(native_names),
    )
    _write_csv(
        out / "fresh_sample_manifest.csv",
        [
            {
                "sample_hash": str(sample_hash),
                "legacy_sample_id_hash": str(legacy_hash),
                "label": int(label),
                "attack": str(attack),
                "role": str(role),
                "historical_overlap": False,
            }
            for sample_hash, legacy_hash, label, attack, role in zip(
                hashes, legacy_hashes, y, attacks, roles, strict=True
            )
        ],
    )
    role_ids = {
        role: {str(value) for value in hashes[roles == role]}
        for role in ("train", "validation", "acceptance")
    }
    overlap = sum(
        len(role_ids[left] & role_ids[right])
        for left, right in (
            ("train", "validation"),
            ("train", "acceptance"),
            ("validation", "acceptance"),
        )
    )
    historical_overlap = len({str(value) for value in hashes} & canonical_exclusions)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_split.json", after)
    ready = complete and overlap == 0 and historical_overlap == 0 and before == after
    report = {
        **protocol,
        "status": SPLIT_STATUS if ready else "failed_w96_split_gate",
        "sample_count": len(y),
        "train_count": int(np.sum(roles == "train")),
        "validation_count": int(np.sum(roles == "validation")),
        "acceptance_count": int(np.sum(roles == "acceptance")),
        "counts": counts,
        "scanned_rows": scanned,
        "excluded_rows": excluded,
        "labeled_rows": labeled,
        "common_feature_count": x_common.shape[1],
        "native_feature_count": x_native.shape[1],
        "split_overlap_count": overlap,
        "historical_overlap_count": historical_overlap,
        "frozen_hashes_unchanged": before == after,
        "acceptance_sealed": True,
    }
    _dump(out / "split_manifest.json", report)
    return report


def run_nfiot_bot_targeted_feature_forensics_w96(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    return run_targeted_feature_forensics_w94(
        output_dir,
        experiment=EXPERIMENT,
        ready_split_status=SPLIT_STATUS,
        feature_pack_status=FEATURE_PACK_STATUS,
    )


def train_nfiot_bot_targeted_models_w96(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    return train_nfiot_targeted_models_w95(
        output_dir,
        model_dir,
        experiment=EXPERIMENT,
        require_validation_recall_not_lower=True,
        feature_pack_status=FEATURE_PACK_STATUS,
    )


def evaluate_nfiot_bot_targeted_models_w96(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    return evaluate_nfiot_targeted_models_w95(
        output_dir,
        model_dir,
        experiment=EXPERIMENT,
        target_variant=TARGET_VARIANT,
        acceptance_marker_name="acceptance_opened_w96.json",
        evidence_reason_code="W96_FULL_COVERAGE_SHADOW",
    )


def finalize_nfiot_bot_targeted_performance_w96(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    return finalize_nfiot_targeted_performance_w95(
        output_dir,
        document=document,
        document_cn=document_cn,
        tests_passed=tests_passed,
        test_count=test_count,
        target_variant=TARGET_VARIANT,
    )
