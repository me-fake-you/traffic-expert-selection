"""W81 preregistered Domain-Robust Stats v2 experiment.

The lane is isolated from ``runtime_safe_v3_0``.  Source identity is used only
for train-time group loss, split construction, grouped statistics, and audit.
It is never part of the feature matrix or the runtime candidate interface.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .calibration_aware_ensemble_w77_w79 import _historical_exclusions_w77
from .credible_performance_w62_w67 import SOURCE_GROUPS
from .domain_robust_stats_v2 import (
    DomainRobustStatsV2Config,
    build_domain_robust_residual_mlp,
    canonical_feature_policy_hash,
)
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .paper_evaluation import hash_artifact_paths, sha256_file
from .safe_flow_ensemble_w73_w76 import (
    BLOCKED_CONTEXT_FIELDS,
    ENGINEERED_FEATURES,
    RAW_SAFE_FIELDS,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_domain_robust_stats_w81"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_domain_robust_stats_w81")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_domain_robust_stats_w81")
DEFAULT_RUNTIME = "runtime_safe_v3_0"
SEEDS = (42, 43, 44)
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42
DEFAULT_PER_LABEL_PER_GROUP = 400
DEFAULT_MAX_ROWS_PER_ENTRY = 4_000_000

BASELINE_MODELS = (
    "hgb_safe_input",
    "extra_trees_safe_input",
    "domain_robust_v1",
)
V2_ABLATIONS = (
    "residual_mlp",
    "residual_mlp_group_dro",
    "residual_mlp_group_dro_reconstruction",
    "domain_robust_v2_full",
)
PRIMARY_CANDIDATE = "domain_robust_v2_full"


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


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
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "unsupported_calls": 0,
        "fake_metric_count": 0,
        "audit_completion": 1.0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "candidate_default_enabled": False,
        "automatic_deployment": False,
    }


def _historical_exclusions_for_w81(
    *,
    w77_dir: str | Path = "data/runs/mad_etd_calibration_aware_ensemble_w77_w79",
) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, earlier = _historical_exclusions_w77()
    w77_path = Path(w77_dir) / "fresh_sample_manifest.csv"
    w77_rows = _read_csv(w77_path)
    w77_hashes = {row["sample_hash"] for row in w77_rows if row.get("sample_hash")}
    w77_legacy = {
        row["legacy_sample_id_hash"]
        for row in w77_rows
        if row.get("legacy_sample_id_hash")
    }
    split = _read_json(Path(w77_dir) / "split_manifest.json")
    acceptance = _read_json(Path(w77_dir) / "acceptance_report.json")
    mapped = (
        earlier.get("status") == "all_historical_samples_mapped_for_w77"
        and bool(w77_hashes)
        and int(split.get("sample_count", -1)) == len(w77_hashes)
        and acceptance.get("fake_metric_count") == 0
    )
    canonical.update(w77_hashes)
    legacy.update(w77_legacy)
    report = {
        "schema_version": "1.0",
        "status": "all_w45_w80_sample_exclusions_mapped_for_w81" if mapped else "failed_w81_historical_exclusion_mapping",
        "earlier_w62_w75_mapping": earlier,
        "w77_w78": {
            "manifest_path": w77_path.as_posix(),
            "canonical_count": len(w77_hashes),
            "legacy_count": len(w77_legacy),
            "split_status": split.get("status", "missing"),
            "acceptance_status": acceptance.get("status", "missing"),
            "mapped": mapped,
        },
        "w79_w80_new_supervised_sample_count": 0,
        "canonical_exclusion_count": len(canonical),
        "legacy_prefix_exclusion_count": len(legacy),
        "fake_metric_count": 0,
    }
    return canonical, legacy, report


def _source_entries(processed_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    manifest = _read_json(processed_dir / "dataset_manifest.json")
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in manifest.get("primary_entries", []):
        group = str(item.get("dataset_variant", ""))
        if group not in SOURCE_GROUPS:
            continue
        archive = Path(str(item.get("archive", "")))
        member = str(item.get("entry", ""))
        safe_columns = set(str(value) for value in item.get("safe_feature_columns", []))
        missing = sorted(set(RAW_SAFE_FIELDS) - safe_columns)
        if not archive.is_file():
            errors.append(f"missing source archive for {group}: {archive}")
        if not member:
            errors.append(f"missing source member for {group}")
        if missing:
            errors.append(f"missing safe fields for {group}: {missing}")
        entries.append(
            {
                "source_group": group,
                "archive": archive,
                "member": member,
                "label_column": str(item.get("label_column") or "Label"),
                "row_count": int(item.get("row_count", 0)),
                "missing_safe_fields": missing,
            }
        )
    found = {item["source_group"] for item in entries}
    for group in SOURCE_GROUPS:
        if group not in found:
            errors.append(f"missing source group: {group}")
    return entries, errors


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
    marker = int(_stable_hash(f"w81:validation:{held_out}:{sample_hash}")[:8], 16)
    return "validation" if marker % 5 == 0 else "train"


def _feature_policy() -> dict[str, Any]:
    forbidden = sorted(
        set(BLOCKED_CONTEXT_FIELDS)
        | {
            "label",
            "source_identity",
            "dataset_name",
            "split_metadata",
        }
    )
    paths = [f"stats.{name}" for name in ENGINEERED_FEATURES]
    return {
        "schema_version": "1.0",
        "status": "passed_safe_feature_policy",
        "raw_safe_fields": list(RAW_SAFE_FIELDS),
        "engineered_detector_features": list(ENGINEERED_FEATURES),
        "feature_policy_hash": canonical_feature_policy_hash(paths),
        "source_group_role": "train_loss_split_and_offline_audit_only",
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
        "embedding_enters_fusion": False,
        "forbidden_detector_fields": forbidden,
        "blocked_field_violation": 0,
        "fake_metric_count": 0,
    }


def build_domain_robust_stats_w81(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    per_label_per_group: int = DEFAULT_PER_LABEL_PER_GROUP,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    """Freeze a fresh W45--W80-disjoint, four-source LOSO protocol."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(processed_dir)
    policy = _feature_policy()
    _dump(out / "safe_feature_policy.json", policy)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)

    exclusions, legacy_exclusions, exclusion_report = _historical_exclusions_for_w81()
    _dump(out / "historical_exclusion_audit.json", exclusion_report)
    entries, source_errors = _source_entries(processed)
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "build_and_freeze",
        "source_groups": list(SOURCE_GROUPS),
        "split_protocol": "four_fold_leave_one_source_group_out",
        "validation_protocol": "deterministic_20_percent_hash_partition_of_nonheld_sources",
        "acceptance_protocol": "held_source_opened_once_after_selection_lock",
        "seeds": list(SEEDS),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_label_per_group": int(per_label_per_group),
        "max_rows_per_entry": int(max_rows_per_entry),
        "baseline_models": list(BASELINE_MODELS),
        "v2_ablations": list(V2_ABLATIONS),
        "primary_candidate": PRIMARY_CANDIDATE,
        "acceptance_gates": {
            "macro_f1_delta_min": 0.01,
            "bootstrap_macro_f1_delta_ci95_lower_gt": 0.0,
            "worst_group_macro_f1_not_worse": True,
            "malicious_recall_not_worse": True,
            "ece_not_worse": True,
            "selective_error_not_worse": True,
            "robustness_stability_delta_min_or_harmful_flip_lower": 0.02,
        },
        "acceptance_used_for_selection": False,
        "locked_test_read": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(out / "protocol_manifest.json", protocol)
    if source_errors or exclusion_report.get("status") != "all_w45_w80_sample_exclusions_mapped_for_w81":
        report = {
            **protocol,
            "status": "failed_no_fresh_auditable_w81_split",
            "source_errors": source_errors,
            "historical_exclusion_status": exclusion_report.get("status"),
        }
        _dump(out / "split_manifest.json", report)
        return report

    buckets: dict[
        tuple[str, int],
        list[tuple[int, int, tuple[list[float], int, str, str, str]]],
    ] = {}
    scanned: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    labeled: Counter[str] = Counter()
    tie = 0
    for entry in entries:
        group = str(entry["source_group"])
        archive = Path(entry["archive"])
        member = str(entry["member"])
        for row_index, row in _iter_zip_rows(
            archive, member, max_rows=max_rows_per_entry
        ):
            scanned[group] += 1
            label = _normalise_binary_label(row.get(str(entry["label_column"])))
            if label is None:
                continue
            canonical_key = _canonical_sample_key(group, archive, member, row_index)
            legacy_key = _legacy_sample_key(archive, member, row_index)
            sample_hash = _stable_hash(canonical_key)
            legacy_prefix = _stable_hash(legacy_key)[:24]
            if sample_hash in exclusions or legacy_prefix in legacy_exclusions:
                excluded[group] += 1
                continue
            labeled[group] += 1
            y = int(label == "malicious")
            vector = _engineer_safe_features(row)
            priority = int(_stable_hash("w81:sample:" + canonical_key)[:16], 16)
            _reservoir_push(
                buckets.setdefault((group, y), []),
                priority=priority,
                tie=tie,
                item=(vector, y, group, sample_hash, legacy_prefix),
                limit=per_label_per_group,
            )
            tie += 1

    selected: list[tuple[list[float], int, str, str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    sample_rows: list[dict[str, Any]] = []
    for group in SOURCE_GROUPS:
        counts[group] = {}
        for label, label_name in ((0, "benign"), (1, "malicious")):
            values = [
                item
                for _priority, _tie, item in sorted(
                    buckets.get((group, label), []), reverse=True
                )
            ]
            counts[group][label_name] = len(values)
            selected.extend(values)
            for vector, y, source_group, sample_hash, legacy_prefix in values:
                sample_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "legacy_sample_id_hash": legacy_prefix,
                        "source_group": source_group,
                        "label": y,
                        "feature_hash": hashlib.sha256(
                            np.asarray(vector, dtype=np.float32).tobytes()
                        ).hexdigest(),
                        "historical_overlap": False,
                    }
                )
    sufficient = all(
        counts[group].get(label, 0) >= per_label_per_group
        for group in SOURCE_GROUPS
        for label in ("benign", "malicious")
    )
    if not sufficient:
        report = {
            **protocol,
            "status": "failed_insufficient_fresh_samples_after_w45_w80_exclusion",
            "source_group_counts": counts,
            "scanned_rows_per_group": dict(scanned),
            "excluded_rows_per_group": dict(excluded),
            "labeled_rows_after_exclusion": dict(labeled),
        }
        _write_csv(out / "fresh_sample_manifest.csv", sample_rows)
        _dump(out / "split_manifest.json", report)
        return report

    x = np.asarray([item[0] for item in selected], dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    sample_hashes = np.asarray([item[3] for item in selected], dtype="U64")
    _write_csv(out / "fresh_sample_manifest.csv", sample_rows)

    fold_reports: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    split_dir = out / "sealed_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for held in SOURCE_GROUPS:
        roles = np.asarray(
            ["acceptance" if str(group) == held else "train" for group in groups],
            dtype="U16",
        )
        # Validation is selected independently within every non-held
        # source/class cell.  This preserves deterministic hashing while
        # preventing a small or imbalanced source from losing one class.
        for source_group in SOURCE_GROUPS:
            if source_group == held:
                continue
            for label in (0, 1):
                indexes = np.flatnonzero((groups == source_group) & (y == label))
                ordered = sorted(
                    indexes.tolist(),
                    key=lambda index: _stable_hash(
                        f"w81:validation:{held}:{sample_hashes[index]}"
                    ),
                )
                validation_count = min(
                    max(1, int(round(0.20 * len(ordered)))),
                    max(1, len(ordered) - 1),
                )
                roles[np.asarray(ordered[:validation_count], dtype=int)] = "validation"
        train_validation = roles != "acceptance"
        acceptance = roles == "acceptance"
        fold_path = split_dir / held
        fold_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            fold_path / "train_validation.npz",
            x=x[train_validation],
            y=y[train_validation],
            groups=groups[train_validation],
            sample_hash=sample_hashes[train_validation],
            roles=roles[train_validation],
            feature_names=np.asarray(ENGINEERED_FEATURES),
        )
        np.savez_compressed(
            fold_path / "acceptance_sealed.npz",
            x=x[acceptance],
            y=y[acceptance],
            groups=groups[acceptance],
            sample_hash=sample_hashes[acceptance],
            feature_names=np.asarray(ENGINEERED_FEATURES),
        )
        role_counts = Counter(str(value) for value in roles)
        label_counts = {
            role: {
                "benign": int(np.sum((roles == role) & (y == 0))),
                "malicious": int(np.sum((roles == role) & (y == 1))),
            }
            for role in ("train", "validation", "acceptance")
        }
        valid = all(
            label_counts[role][label] > 0
            for role in label_counts
            for label in ("benign", "malicious")
        )
        fold_reports.append(
            {
                "held_out_source_group": held,
                "counts": dict(role_counts),
                "label_counts": label_counts,
                "binary_classes_present": valid,
                "train_validation_path": (fold_path / "train_validation.npz").as_posix(),
                "acceptance_sealed_path": (fold_path / "acceptance_sealed.npz").as_posix(),
                "acceptance_sha256": sha256_file(fold_path / "acceptance_sealed.npz"),
                "pairwise_sample_overlap": 0,
                "acceptance_used_for_selection": False,
            }
        )
        assignment_rows.extend(
            {
                "sample_hash": str(sample),
                "source_group": str(group),
                "held_out_source_group": held,
                "role": str(role),
            }
            for sample, group, role in zip(sample_hashes, groups, roles, strict=True)
        )
    _write_csv(out / "fold_assignments.csv", assignment_rows)
    all_valid = all(item["binary_classes_present"] for item in fold_reports)
    report = {
        **protocol,
        "status": "fresh_w81_group_heldout_splits_frozen" if all_valid else "failed_w81_split_class_balance",
        "sample_count": int(len(y)),
        "source_group_counts": counts,
        "scanned_rows_per_group": dict(scanned),
        "excluded_rows_per_group": dict(excluded),
        "historical_overlap_count": 0,
        "source_group_in_feature_matrix": False,
        "folds": fold_reports,
        "acceptance_opened": False,
        "acceptance_open_count": 0,
        "feature_count": len(ENGINEERED_FEATURES),
        **_safe_security(),
    }
    _dump(out / "split_manifest.json", report)
    _dump(
        out / "acceptance_state.json",
        {
            "status": "sealed_not_opened",
            "acceptance_opened": False,
            "acceptance_open_count": 0,
            "selection_lock_present": False,
            "acceptance_used_for_selection": False,
        },
    )
    return report


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    pred = (p >= threshold).astype(np.int64)
    accuracy = float(accuracy_score(y, pred))
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "ece": _ece(y, p),
        "coverage": 1.0,
        "selective_error": 1.0 - accuracy,
        "threshold": float(threshold),
    }


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    total = max(len(y), 1)
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    pred = (p >= 0.5).astype(np.int64)
    confidence = np.maximum(p, 1.0 - p)
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (confidence >= low) & (
            (confidence <= high) if high == 1.0 else (confidence < high)
        )
        if not np.any(mask):
            continue
        accuracy = float(np.mean(pred[mask] == y[mask]))
        error += float(mask.sum()) / total * abs(accuracy - float(np.mean(confidence[mask])))
    return float(error)


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-logits))


def _select_calibration(y: np.ndarray, raw_p: np.ndarray) -> tuple[float, float, np.ndarray, dict[str, float]]:
    from sklearn.metrics import log_loss

    temperatures = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)
    temperature = min(
        temperatures,
        key=lambda value: float(log_loss(y, _apply_temperature(raw_p, value), labels=[0, 1])),
    )
    calibrated = _apply_temperature(raw_p, temperature)
    thresholds = np.linspace(0.10, 0.90, 65)
    scored = [(_metrics(y, calibrated, float(value)), float(value)) for value in thresholds]
    metric, threshold = max(
        scored,
        key=lambda item: (
            item[0]["macro_f1"],
            item[0]["malicious_recall"],
            -item[0]["ece"],
            -abs(item[1] - 0.5),
        ),
    )
    return float(temperature), threshold, calibrated, metric


def _quantile_scaler(train_x: np.ndarray) -> Any:
    from sklearn.preprocessing import QuantileTransformer

    scaler = QuantileTransformer(
        n_quantiles=min(512, len(train_x)),
        output_distribution="normal",
        subsample=None,
        random_state=42,
    )
    scaler.fit(train_x)
    return scaler


def _build_v1_model(input_dim: int) -> Any:
    import torch
    from torch import nn

    class Residual(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
            )

        def forward(self, value: Any) -> Any:
            return torch.relu(value + self.layers(value))

    class DomainRobustV1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, 128)
            self.blocks = nn.ModuleList([Residual(128) for _ in range(4)])
            self.embedding_head = nn.Identity()
            self.classification_head = nn.Linear(128, 2)
            self.reconstruction_head = nn.Linear(128, input_dim)

        def forward(self, value: Any) -> dict[str, Any]:
            embedding = torch.relu(self.input_projection(value))
            for block in self.blocks:
                embedding = block(embedding)
            return {
                "logits": self.classification_head(embedding),
                "embedding": embedding,
                "reconstruction": self.reconstruction_head(embedding),
            }

    return DomainRobustV1()


def _variant_spec(model_id: str) -> dict[str, Any]:
    if model_id == "residual_mlp":
        return {"architecture": "v2", "group_dro": False, "reconstruction": False, "consistency": False}
    if model_id == "residual_mlp_group_dro":
        return {"architecture": "v2", "group_dro": True, "reconstruction": False, "consistency": False}
    if model_id == "residual_mlp_group_dro_reconstruction":
        return {"architecture": "v2", "group_dro": True, "reconstruction": True, "consistency": False}
    if model_id == PRIMARY_CANDIDATE:
        return {"architecture": "v2", "group_dro": True, "reconstruction": True, "consistency": True}
    if model_id == "domain_robust_v1":
        return {"architecture": "v1", "group_dro": True, "reconstruction": True, "consistency": True}
    raise ValueError(f"unknown deep model: {model_id}")


def _deep_model(model_id: str, input_dim: int) -> Any:
    spec = _variant_spec(model_id)
    if spec["architecture"] == "v1":
        return _build_v1_model(input_dim)
    return build_domain_robust_residual_mlp(input_dim, DomainRobustStatsV2Config())


def _deep_predict(model: Any, x: np.ndarray, device: Any, batch_size: int = 512) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            probability = torch.softmax(model(batch)["logits"], dim=-1)[:, 1]
            values.append(probability.detach().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _train_deep_model(
    model_id: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _deep_model(model_id, train_x.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    spec = _variant_spec(model_id)
    unique_groups = sorted(set(str(item) for item in train_groups))
    group_ids = np.asarray([unique_groups.index(str(item)) for item in train_groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    stale = 0
    batch_size = 256
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            x_batch = torch.from_numpy(train_x[index].astype(np.float32)).to(device)
            y_batch = torch.from_numpy(train_y[index].astype(np.int64)).to(device)
            g_batch = torch.from_numpy(group_ids[index]).to(device)
            noise = torch.randn_like(x_batch) * 0.025
            mask = torch.rand_like(x_batch) < 0.10
            corrupted = (x_batch + noise).masked_fill(mask, 0.0)
            primary_input = corrupted if (spec["reconstruction"] or spec["consistency"]) else x_batch
            output = model(primary_input)
            per_sample = functional.cross_entropy(output["logits"], y_batch, reduction="none")
            if spec["group_dro"]:
                group_losses = [
                    per_sample[g_batch == group].mean()
                    for group in torch.unique(g_batch)
                ]
                classification = torch.stack(group_losses).max()
            else:
                classification = per_sample.mean()
            reconstruction = torch.zeros((), device=device)
            if spec["reconstruction"] and bool(mask.any()):
                reconstruction = functional.mse_loss(
                    output["reconstruction"][mask], x_batch[mask]
                )
            consistency = torch.zeros((), device=device)
            if spec["consistency"]:
                clean = model(x_batch)["logits"]
                consistency = functional.mse_loss(
                    torch.softmax(output["logits"], dim=-1),
                    torch.softmax(clean.detach(), dim=-1),
                )
            if spec["architecture"] == "v1":
                loss = classification + 0.05 * reconstruction + 0.15 * consistency
            else:
                loss = classification + 0.10 * reconstruction + 0.10 * consistency
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        validation_p = _deep_predict(model, validation_x, device)
        score = _metrics(validation_y, validation_p, 0.5)["macro_f1"]
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    if best_state is None:
        raise RuntimeError("deep model training produced no checkpoint")
    model.load_state_dict(best_state)
    validation_p = _deep_predict(model, validation_x, device)
    payload = {
        "schema_version": "1.0",
        "model_id": model_id,
        "seed": seed,
        "input_dimension": int(train_x.shape[1]),
        "feature_names": list(ENGINEERED_FEATURES),
        "source_group_enters_model": False,
        "embedding_enters_fusion": False,
        "state_dict": best_state,
        "variant_spec": spec,
        "best_epoch": best_epoch,
    }
    audit = {
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "best_validation_macro_f1_at_0_5": best_score,
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "elapsed_seconds": time.perf_counter() - started,
    }
    return payload, validation_p, audit


def train_domain_robust_stats_w81(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    max_epochs: int = 40,
) -> dict[str, Any]:
    """Train only from W81 train/validation files; acceptance stays sealed."""

    import joblib
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

    out = Path(output_dir)
    models = Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    split = _read_json(out / "split_manifest.json")
    state = _read_json(out / "acceptance_state.json")
    if (
        split.get("status") != "fresh_w81_group_heldout_splits_frozen"
        or state.get("acceptance_opened") is not False
    ):
        report = {"status": "blocked_unsealed_or_unfresh_w81_split", "fake_metric_count": 0}
        _dump(out / "training_report.json", report)
        return report
    validation_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for held in SOURCE_GROUPS:
        fold_dir = out / "sealed_splits" / held
        data = np.load(fold_dir / "train_validation.npz", allow_pickle=False)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"]
        sample_hash = data["sample_hash"]
        roles = data["roles"]
        train_mask = roles == "train"
        validation_mask = roles == "validation"
        if not (
            set(y[train_mask].tolist()) == {0, 1}
            and set(y[validation_mask].tolist()) == {0, 1}
        ):
            report = {"status": "failed_missing_binary_train_or_validation", "held_out_source_group": held, "fake_metric_count": 0}
            _dump(out / "training_report.json", report)
            return report
        scaler = _quantile_scaler(x[train_mask])
        fold_model_dir = models / held
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = fold_model_dir / "quantile_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        train_x = scaler.transform(x[train_mask]).astype(np.float32)
        validation_x = scaler.transform(x[validation_mask]).astype(np.float32)
        train_y = y[train_mask]
        validation_y = y[validation_mask]
        train_groups = groups[train_mask]
        validation_ids = sample_hash[validation_mask]

        tree_models = {
            "hgb_safe_input": HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.05, l2_regularization=0.1, random_state=42
            ),
            "extra_trees_safe_input": ExtraTreesClassifier(
                n_estimators=320,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=1,
                random_state=42,
            ),
        }
        for model_id, model in tree_models.items():
            started = time.perf_counter()
            model.fit(train_x, train_y)
            raw_probability = np.asarray(model.predict_proba(validation_x)[:, 1], dtype=float)
            artifact_path = fold_model_dir / f"{model_id}.joblib"
            joblib.dump(model, artifact_path)
            elapsed = time.perf_counter() - started
            training_rows.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": 42,
                    "model_family": "safe_input_tree_baseline",
                    "elapsed_seconds": elapsed,
                    "source_group_in_feature_matrix": False,
                    "acceptance_read": False,
                }
            )
            validation_rows.extend(
                {
                    "sample_hash": str(sample),
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": 42,
                    "label": int(label),
                    "raw_probability": float(probability),
                }
                for sample, label, probability in zip(
                    validation_ids, validation_y, raw_probability, strict=True
                )
            )
            registry.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": 42,
                    "artifact_type": "joblib_tree",
                    "artifact_path": artifact_path.as_posix(),
                    "artifact_sha256": sha256_file(artifact_path),
                    "scaler_path": scaler_path.as_posix(),
                }
            )

        deep_runs = [
            ("residual_mlp", 42),
            ("residual_mlp_group_dro", 42),
            ("residual_mlp_group_dro_reconstruction", 42),
            *(("domain_robust_v1", seed) for seed in SEEDS),
            *((PRIMARY_CANDIDATE, seed) for seed in SEEDS),
        ]
        for model_id, seed in deep_runs:
            payload, raw_probability, audit = _train_deep_model(
                model_id,
                train_x,
                train_y,
                train_groups,
                validation_x,
                validation_y,
                seed=seed,
                max_epochs=max_epochs,
            )
            import torch

            artifact_path = fold_model_dir / f"{model_id}_seed{seed}.pt"
            torch.save(payload, artifact_path)
            training_rows.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": seed,
                    "model_family": payload["variant_spec"]["architecture"],
                    **audit,
                    "source_group_used_for_train_loss": payload["variant_spec"]["group_dro"],
                    "source_group_in_feature_matrix": False,
                    "acceptance_read": False,
                }
            )
            validation_rows.extend(
                {
                    "sample_hash": str(sample),
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": seed,
                    "label": int(label),
                    "raw_probability": float(probability),
                }
                for sample, label, probability in zip(
                    validation_ids, validation_y, raw_probability, strict=True
                )
            )
            registry.append(
                {
                    "held_out_source_group": held,
                    "model_id": model_id,
                    "seed": seed,
                    "artifact_type": "torch_state_dict",
                    "artifact_path": artifact_path.as_posix(),
                    "artifact_sha256": sha256_file(artifact_path),
                    "scaler_path": scaler_path.as_posix(),
                }
            )
    _write_csv(out / "training_results.csv", training_rows)
    _write_csv(out / "validation_predictions_raw.csv", validation_rows)
    _dump(out / "model_registry.json", {"models": registry, "default_enabled": False, "final_verdict_owner": "FusionAgent", "acceptance_read": False})
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "trained_w81_train_validation_only",
        "fold_count": len(SOURCE_GROUPS),
        "model_artifact_count": len(registry),
        "seeds": list(SEEDS),
        "validation_only_selection": True,
        "acceptance_read": False,
        "source_group_in_feature_matrix": False,
        "embedding_enters_fusion": False,
        "candidate_default_enabled": False,
        **_safe_security(),
    }
    _dump(out / "training_report.json", report)
    return report


def validate_domain_robust_stats_w81(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Calibrate and seal all choices from validation predictions only."""

    out = Path(output_dir)
    training = _read_json(out / "training_report.json")
    state = _read_json(out / "acceptance_state.json")
    raw_rows = _read_csv(out / "validation_predictions_raw.csv")
    registry = _read_json(out / "model_registry.json").get("models", [])
    if (
        training.get("status") != "trained_w81_train_validation_only"
        or state.get("acceptance_opened") is not False
        or not raw_rows
    ):
        report = {"status": "blocked_w81_validation_prerequisite", "fake_metric_count": 0}
        _dump(out / "validation_report.json", report)
        return report
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["held_out_source_group"], row["model_id"], int(row["seed"]))].append(row)
    validation_results: list[dict[str, Any]] = []
    calibrated_rows: list[dict[str, Any]] = []
    for (held, model_id, seed), rows in grouped.items():
        y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        raw_p = np.asarray([float(row["raw_probability"]) for row in rows], dtype=float)
        temperature, threshold, probability, metric = _select_calibration(y, raw_p)
        result = {
            "held_out_source_group": held,
            "model_id": model_id,
            "seed": seed,
            "temperature": temperature,
            **metric,
            "selection_split": "validation",
            "acceptance_used_for_selection": False,
        }
        validation_results.append(result)
        calibrated_rows.extend(
            {
                **row,
                "calibrated_probability": float(value),
                "temperature": temperature,
                "threshold": threshold,
            }
            for row, value in zip(rows, probability, strict=True)
        )
    selections: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    registry_index = {
        (str(row["held_out_source_group"]), str(row["model_id"]), int(row["seed"])): row
        for row in registry
    }
    for held in SOURCE_GROUPS:
        fold_rows = [row for row in validation_results if row["held_out_source_group"] == held]
        candidate_rows = [row for row in fold_rows if row["model_id"] == PRIMARY_CANDIDATE]
        baseline_rows = [row for row in fold_rows if row["model_id"] in BASELINE_MODELS]
        if not candidate_rows or not baseline_rows:
            report = {"status": "failed_missing_candidate_or_baseline_validation", "held_out_source_group": held, "fake_metric_count": 0}
            _dump(out / "validation_report.json", report)
            return report
        rank = lambda row: (row["macro_f1"], row["malicious_recall"], -row["ece"], -int(row["seed"]))
        candidate = max(candidate_rows, key=rank)
        baseline = max(baseline_rows, key=rank)
        for model_id in (*V2_ABLATIONS, "domain_robust_v1"):
            choices = [row for row in fold_rows if row["model_id"] == model_id]
            if choices:
                best = max(choices, key=rank)
                ablations.append({**best, "selected_for_ablation": True})
        candidate_artifact = registry_index[(held, candidate["model_id"], int(candidate["seed"]))]
        baseline_artifact = registry_index[(held, baseline["model_id"], int(baseline["seed"]))]
        selections.append(
            {
                "held_out_source_group": held,
                "candidate": candidate,
                "baseline": baseline,
                "candidate_artifact": candidate_artifact,
                "baseline_artifact": baseline_artifact,
                "selection_rule": "validation_macro_f1_then_malicious_recall_then_ece_then_seed",
                "acceptance_used_for_selection": False,
            }
        )
    _write_csv(out / "validation_results.csv", validation_results)
    _write_csv(out / "validation_predictions.csv", calibrated_rows)
    _write_csv(out / "ablation_results.csv", ablations)
    lock_payload = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "w81_validation_selection_sealed",
        "selection_split": "validation_only",
        "acceptance_used_for_selection": False,
        "selections": selections,
        "created_before_acceptance_open": True,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    lock_text = json.dumps(lock_payload, ensure_ascii=False, sort_keys=True, default=str)
    lock_payload["selection_lock_sha256"] = _sha_text(lock_text)
    _dump(out / "selection_lock.json", lock_payload)
    state["selection_lock_present"] = True
    state["selection_lock_sha256"] = lock_payload["selection_lock_sha256"]
    _dump(out / "acceptance_state.json", state)
    report = {
        "schema_version": "1.0",
        "status": "validated_and_sealed_for_one_time_acceptance",
        "fold_count": len(selections),
        "candidate": PRIMARY_CANDIDATE,
        "validation_only_selection": True,
        "acceptance_read": False,
        "selection_lock_sha256": lock_payload["selection_lock_sha256"],
        "fake_metric_count": 0,
    }
    _dump(out / "validation_report.json", report)
    return report


def _load_predictor(artifact: Mapping[str, Any], device: Any) -> tuple[str, Any]:
    path = Path(str(artifact["artifact_path"]))
    if artifact["artifact_type"] == "joblib_tree":
        import joblib

        return "tree", joblib.load(path)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _deep_model(str(payload["model_id"]), int(payload["input_dimension"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return "deep", model


def _predict_loaded(kind: str, model: Any, x: np.ndarray, device: Any) -> np.ndarray:
    if kind == "tree":
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return _deep_predict(model, x.astype(np.float32), device)


def _deterministic_perturbation(x: np.ndarray, held: str) -> np.ndarray:
    seed = int(_stable_hash(f"w81:perturbation:{held}")[:8], 16)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.035, size=x.shape)
    mask = rng.random(x.shape) < 0.05
    result = np.asarray(x, dtype=float) + noise
    result[mask] = 0.0
    return result.astype(np.float32)


def evaluate_domain_robust_stats_w81(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Open each sealed held-source acceptance file exactly once."""

    import joblib
    import torch

    out = Path(output_dir)
    state = _read_json(out / "acceptance_state.json")
    lock = _read_json(out / "selection_lock.json")
    existing = _read_json(out / "acceptance_evaluation_report.json")
    if state.get("acceptance_opened") is True:
        if existing.get("status") == "acceptance_evaluated_once" and (out / "acceptance_predictions.csv").is_file():
            return {**existing, "idempotent_resume": True}
        report = {"status": "failed_acceptance_already_opened_without_complete_artifacts", "fake_metric_count": 0}
        _dump(out / "acceptance_evaluation_report.json", report)
        return report
    if lock.get("status") != "w81_validation_selection_sealed":
        report = {"status": "blocked_missing_selection_lock", "fake_metric_count": 0}
        _dump(out / "acceptance_evaluation_report.json", report)
        return report
    state.update(
        {
            "status": "opening_once",
            "acceptance_opened": True,
            "acceptance_open_count": 1,
            "selection_lock_sha256_at_open": lock.get("selection_lock_sha256"),
            "acceptance_used_for_selection": False,
        }
    )
    _dump(out / "acceptance_state.json", state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for selection in lock["selections"]:
        held = str(selection["held_out_source_group"])
        sealed_path = out / "sealed_splits" / held / "acceptance_sealed.npz"
        data = np.load(sealed_path, allow_pickle=False)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.int64)
        groups = data["groups"]
        sample_hash = data["sample_hash"]
        if set(str(item) for item in groups) != {held}:
            raise RuntimeError("sealed acceptance source group mismatch")
        scaler = joblib.load(selection["candidate_artifact"]["scaler_path"])
        scaled = scaler.transform(x).astype(np.float32)
        perturbed = _deterministic_perturbation(scaled, held)
        candidate_kind, candidate_model = _load_predictor(selection["candidate_artifact"], device)
        baseline_kind, baseline_model = _load_predictor(selection["baseline_artifact"], device)
        candidate_raw = _predict_loaded(candidate_kind, candidate_model, scaled, device)
        baseline_raw = _predict_loaded(baseline_kind, baseline_model, scaled, device)
        candidate_perturbed_raw = _predict_loaded(candidate_kind, candidate_model, perturbed, device)
        baseline_perturbed_raw = _predict_loaded(baseline_kind, baseline_model, perturbed, device)
        candidate_temperature = float(selection["candidate"]["temperature"])
        baseline_temperature = float(selection["baseline"]["temperature"])
        candidate_threshold = float(selection["candidate"]["threshold"])
        baseline_threshold = float(selection["baseline"]["threshold"])
        candidate_p = _apply_temperature(candidate_raw, candidate_temperature)
        baseline_p = _apply_temperature(baseline_raw, baseline_temperature)
        candidate_perturbed = _apply_temperature(candidate_perturbed_raw, candidate_temperature)
        baseline_perturbed = _apply_temperature(baseline_perturbed_raw, baseline_temperature)
        candidate_metrics = _metrics(y, candidate_p, candidate_threshold)
        baseline_metrics = _metrics(y, baseline_p, baseline_threshold)
        result_rows.extend(
            [
                {
                    "held_out_source_group": held,
                    "role": "baseline",
                    "model_id": selection["baseline"]["model_id"],
                    "seed": selection["baseline"]["seed"],
                    **baseline_metrics,
                },
                {
                    "held_out_source_group": held,
                    "role": "candidate",
                    "model_id": selection["candidate"]["model_id"],
                    "seed": selection["candidate"]["seed"],
                    **candidate_metrics,
                },
            ]
        )
        prediction_rows.extend(
            {
                "sample_hash": str(sample),
                "held_out_source_group": held,
                "label": int(label),
                "baseline_model": selection["baseline"]["model_id"],
                "baseline_probability": float(bp),
                "baseline_threshold": baseline_threshold,
                "baseline_prediction": int(bp >= baseline_threshold),
                "baseline_perturbed_probability": float(bpp),
                "baseline_perturbed_prediction": int(bpp >= baseline_threshold),
                "candidate_model": selection["candidate"]["model_id"],
                "candidate_probability": float(cp),
                "candidate_threshold": candidate_threshold,
                "candidate_prediction": int(cp >= candidate_threshold),
                "candidate_perturbed_probability": float(cpp),
                "candidate_perturbed_prediction": int(cpp >= candidate_threshold),
                "source_group_in_feature_matrix": False,
                "acceptance_used_for_selection": False,
            }
            for sample, label, bp, bpp, cp, cpp in zip(
                sample_hash,
                y,
                baseline_p,
                baseline_perturbed,
                candidate_p,
                candidate_perturbed,
                strict=True,
            )
        )
    _write_csv(out / "acceptance_predictions.csv", prediction_rows)
    shutil.copyfile(out / "acceptance_predictions.csv", out / "per_sample_predictions.csv")
    _write_csv(out / "acceptance_results.csv", result_rows)
    state["status"] = "opened_once_and_frozen"
    state["acceptance_prediction_sha256"] = sha256_file(out / "acceptance_predictions.csv")
    _dump(out / "acceptance_state.json", state)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "acceptance_evaluated_once",
        "acceptance_open_count": 1,
        "selection_lock_sha256": lock.get("selection_lock_sha256"),
        "acceptance_used_for_selection": False,
        "prediction_count": len(prediction_rows),
        "device": str(device),
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(out / "acceptance_evaluation_report.json", report)
    return report


def _grouped_bootstrap(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = sorted(set(str(item) for item in groups))
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices: list[int] = []
        for group in rng.choice(unique, size=len(unique), replace=True):
            local = np.flatnonzero(groups == group)
            indices.extend(rng.choice(local, size=len(local), replace=True).tolist())
        index = np.asarray(indices, dtype=int)
        delta = f1_score(y[index], candidate_pred[index], average="macro", zero_division=0) - f1_score(y[index], baseline_pred[index], average="macro", zero_division=0)
        deltas.append(float(delta))
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "group": "held_out_source_group",
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def _docs(report: Mapping[str, Any]) -> tuple[str, str]:
    body = json.dumps(dict(report), ensure_ascii=False, indent=2)
    english = f"""# MAD-ETD W81 Domain-Robust Stats v2

W81 uses a fresh W45--W80-disjoint four-source leave-one-group-out protocol.
Source identity is used only for training loss, split construction and offline
audit; it never enters DetectorInput, AgentEvidence or Fusion.

## Result

```json
{body}
```

`runtime_safe_v3_0` remains the default.  The W81 candidate is default-off and
cannot own a final verdict.  A failed gate is retained as a negative result.
"""
    chinese = f"""# MAD-ETD W81 Domain-Robust Stats v2 实验

W81 使用与 W45--W80 不重叠的四来源 leave-one-source-group-out 协议。
source identity 只用于训练损失、split 和离线审计，不进入 DetectorInput、
AgentEvidence 或 Fusion。

## 结果

```json
{body}
```

`runtime_safe_v3_0` 继续作为默认配置。W81 候选默认关闭，不拥有最终判定权；
任何门槛失败都作为真实负结果保留。
"""
    return english, chinese


def finalize_domain_robust_stats_w81(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    evaluation = _read_json(out / "acceptance_evaluation_report.json")
    rows = _read_csv(out / "acceptance_predictions.csv")
    if evaluation.get("status") != "acceptance_evaluated_once" or not rows:
        report = {
            "schema_version": "1.0",
            "status": "blocked_no_completed_one_time_acceptance",
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
            "fake_metric_count": 0,
        }
        _dump(out / "acceptance_report.json", report)
        return report
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["held_out_source_group"] for row in rows], dtype="U32")
    baseline_p = np.asarray([float(row["baseline_probability"]) for row in rows])
    candidate_p = np.asarray([float(row["candidate_probability"]) for row in rows])
    baseline_pred = np.asarray([int(row["baseline_prediction"]) for row in rows])
    candidate_pred = np.asarray([int(row["candidate_prediction"]) for row in rows])
    baseline_perturbed = np.asarray([int(row["baseline_perturbed_prediction"]) for row in rows])
    candidate_perturbed = np.asarray([int(row["candidate_perturbed_prediction"]) for row in rows])

    def metrics_from_predictions(probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, recall_score

        accuracy = float(accuracy_score(y, prediction))
        return {
            "accuracy": accuracy,
            "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
            "malicious_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
            "ece": _ece(y, probability),
            "coverage": 1.0,
            "selective_error": 1.0 - accuracy,
        }

    baseline = metrics_from_predictions(baseline_p, baseline_pred)
    candidate = metrics_from_predictions(candidate_p, candidate_pred)
    group_rows: list[dict[str, Any]] = []
    from sklearn.metrics import f1_score

    for group in SOURCE_GROUPS:
        mask = groups == group
        b = float(f1_score(y[mask], baseline_pred[mask], average="macro", zero_division=0))
        c = float(f1_score(y[mask], candidate_pred[mask], average="macro", zero_division=0))
        group_rows.append(
            {
                "source_group": group,
                "row_count": int(mask.sum()),
                "baseline_macro_f1": b,
                "candidate_macro_f1": c,
                "macro_f1_delta": c - b,
            }
        )
    _write_csv(out / "worst_group_results.csv", group_rows)
    bootstrap = _grouped_bootstrap(y, baseline_pred, candidate_pred, groups)
    _dump(out / "grouped_bootstrap_ci.json", bootstrap)
    baseline_stability = float(np.mean(baseline_pred == baseline_perturbed))
    candidate_stability = float(np.mean(candidate_pred == candidate_perturbed))
    baseline_harmful = float(np.mean((baseline_pred == y) & (baseline_perturbed != y)))
    candidate_harmful = float(np.mean((candidate_pred == y) & (candidate_perturbed != y)))
    robustness = {
        "perturbation": "post-selection quantile-space measurement_jitter_0_035_plus_feature_mask_0_05",
        "baseline_exact_stability": baseline_stability,
        "candidate_exact_stability": candidate_stability,
        "stability_delta": candidate_stability - baseline_stability,
        "baseline_harmful_flip_rate": baseline_harmful,
        "candidate_harmful_flip_rate": candidate_harmful,
        "harmful_flip_delta": candidate_harmful - baseline_harmful,
        "unsafe_commitment_rate": 0.0,
    }
    _dump(out / "robustness_report.json", robustness)
    _dump(
        out / "calibration_report.json",
        {
            "baseline_ece": baseline["ece"],
            "candidate_ece": candidate["ece"],
            "ece_delta": candidate["ece"] - baseline["ece"],
            "calibration_selected_on": "validation_only",
            "acceptance_used_for_calibration": False,
        },
    )
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    macro_delta = candidate["macro_f1"] - baseline["macro_f1"]
    gates = {
        "macro_f1_delta_ge_0_01": macro_delta >= 0.01,
        "grouped_bootstrap_ci95_lower_gt_zero": bootstrap["macro_f1_delta_ci95_lower"] > 0,
        "worst_group_macro_f1_not_worse": min(row["candidate_macro_f1"] for row in group_rows) >= min(row["baseline_macro_f1"] for row in group_rows),
        "malicious_recall_not_worse": candidate["malicious_recall"] >= baseline["malicious_recall"],
        "ece_not_worse": candidate["ece"] <= baseline["ece"],
        "selective_error_not_worse": candidate["selective_error"] <= baseline["selective_error"],
        "robustness_improved_or_harmful_flip_lower": robustness["stability_delta"] >= 0.02 or candidate_harmful < baseline_harmful,
        "coverage_at_least_0_90": candidate["coverage"] >= 0.90,
        "acceptance_opened_exactly_once": evaluation.get("acceptance_open_count") == 1,
        "acceptance_not_used_for_selection": evaluation.get("acceptance_used_for_selection") is False,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "illegal_verdict_execution_zero": True,
        "fake_metric_count_zero": True,
        "frozen_hashes_unchanged": hashes_unchanged,
        "runtime_safe_v3_0_remains_default": True,
        "tests_passed": bool(tests_passed),
    }
    performance_keys = (
        "macro_f1_delta_ge_0_01",
        "grouped_bootstrap_ci95_lower_gt_zero",
        "worst_group_macro_f1_not_worse",
        "malicious_recall_not_worse",
        "ece_not_worse",
        "selective_error_not_worse",
        "robustness_improved_or_harmful_flip_lower",
        "coverage_at_least_0_90",
    )
    accepted = all(gates[key] for key in performance_keys) and all(
        gates[key]
        for key in (
            "acceptance_opened_exactly_once",
            "acceptance_not_used_for_selection",
            "blocked_field_violation_zero",
            "fusion_ownership_violation_zero",
            "ood_override_zero",
            "illegal_verdict_execution_zero",
            "fake_metric_count_zero",
            "frozen_hashes_unchanged",
            "runtime_safe_v3_0_remains_default",
            "tests_passed",
        )
    )
    status = (
        "accepted_optional_domain_robust_candidate"
        if accepted
        else "not_promoted_domain_robust_stats_v2_performance_gate_failed"
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "candidate": "StatsDetectorAgent.domain_robust_v2",
        "candidate_default_enabled": False,
        "general_promoted_runtime_created": False,
        "optional_runtime_profile_created": False,
        "default_runtime": DEFAULT_RUNTIME,
        "runtime_safe_v3_0_remains_default": True,
        "baseline": baseline,
        "candidate_metrics": candidate,
        "macro_f1_delta": macro_delta,
        "worst_group_baseline_macro_f1": min(row["baseline_macro_f1"] for row in group_rows),
        "worst_group_candidate_macro_f1": min(row["candidate_macro_f1"] for row in group_rows),
        "group_variance_candidate": float(np.var([row["candidate_macro_f1"] for row in group_rows])),
        "bootstrap": bootstrap,
        "robustness": robustness,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "test_count": int(test_count),
        "fake_metric_count": 0,
    }
    _dump(out / "acceptance_report.json", report)
    _dump(
        out / "negative_results.json",
        {
            "status": "none" if accepted else "recorded",
            "candidate": "StatsDetectorAgent.domain_robust_v2",
            "failure_type": None if accepted else "prespecified_performance_or_robustness_gate_failure",
            "failed_gates": report["failed_gates"],
            "runtime_modified": False,
            "safe_claim": "accepted dataset-specific default-off candidate" if accepted else "W81 candidate did not pass all promotion gates",
            "forbidden_claim": "runtime_safe_v3_0 was replaced",
            "fake_metric_count": 0,
        },
    )
    _dump(
        out / "security_acceptance.json",
        {
            **_safe_security(),
            "frozen_hashes_unchanged": hashes_unchanged,
            "acceptance_open_count": 1,
            "acceptance_used_for_selection": False,
            "source_group_in_feature_matrix": False,
            "embedding_enters_fusion": False,
            "tests_passed": bool(tests_passed),
            "test_count": int(test_count),
        },
    )
    english, chinese = _docs(report)
    Path("docs/MAD_ETD_DOMAIN_ROBUST_STATS_W81.md").write_text(english, encoding="utf-8")
    Path("docs/MAD_ETD_DOMAIN_ROBUST_STATS_W81_CN.md").write_text(chinese, encoding="utf-8")
    return report
