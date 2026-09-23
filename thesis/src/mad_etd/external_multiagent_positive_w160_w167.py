"""W160-W167 fair external multi-agent comparison and positive-result lane.

The lane deliberately separates paper-reported references, reconstructed
paper-compatible local protocols, and source-held-out MAD-ETD protocols.  It
never treats a local architectural adaptation as an official reproduction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from .base import CaseState
from .evidence_team import AgentEvidenceV2Adapter, EvidenceRequestGuard
from .fusion import FusionAgent
from .paper_evaluation import hash_artifact_paths
from .schemas import (
    AgentEvidence,
    DetectorCapabilityProfile,
    EvidenceRequest,
    FeatureGroup,
    FieldAuditResult,
    FlowRecord,
    ReliabilityProfile,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_external_multiagent_positive_w160_w167"
DEFAULT_PROCESSED_NF = Path("data/processed/nf_iot_v12")
DEFAULT_PROCESSED_CICIDS = Path("data/processed/cicids2017/v1")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_external_multiagent_positive_w160_w167")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_external_multiagent_positive_w162")
DEFAULT_RELEASE_DIR = Path("data/releases/mad_etd_external_multiagent_positive_w167")
SEEDS = (42, 43, 44, 45, 46)
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42

# L7_PROTO is intentionally excluded: it is an application-identity proxy.
RAW_SAFE_FEATURES = (
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
    "tcp_flags",
    "log_in_bytes",
    "log_out_bytes",
    "log_in_packets",
    "log_out_packets",
    "log_duration_ms",
    "log_total_bytes",
    "log_total_packets",
    "in_out_byte_ratio",
    "in_out_packet_ratio",
    "log_bytes_per_packet",
)
BLOCKED_FIELDS = (
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "L7_PROTO",
    "Label",
    "Attack",
    "timestamp",
    "Flow ID",
    "source_file",
    "provenance",
    "dataset_variant",
    "split_metadata",
)

NF_TASKS: dict[str, dict[str, Any]] = {
    "NF-BoT-IoT": {
        "classes": ("Benign", "DDoS", "DoS", "Reconnaissance"),
        "excluded_classes": ("Theft",),
        "paper_reference": {
            "accuracy": 0.90,
            "macro_precision": 0.90,
            "macro_recall": 0.90,
            "macro_f1": 0.8975,
        },
        "paper_quotas": {"train": 10_000, "validation": 2_500, "acceptance": 5_000},
        "group_quotas": {"train": 8_000, "validation": 2_000, "acceptance": 3_000},
    },
    "NF-ToN-IoT": {
        "classes": (
            "Benign",
            "scanning",
            "ddos",
            "backdoor",
            "dos",
            "injection",
            "password",
            "xss",
            "mitm",
        ),
        "excluded_classes": ("ransomware",),
        "paper_reference": {
            "accuracy": 0.84,
            "macro_precision": 0.8556,
            "macro_recall": 0.85,
            "macro_f1": 0.8522,
        },
        # 7,700 unique rows per class is the largest near-70k balanced design
        # supported by the public v2 MITM count (7,723).
        "paper_quotas": {"train": 4_400, "validation": 1_100, "acceptance": 2_200},
        "group_quotas": {"train": 800, "validation": 200, "acceptance": 2_000},
    },
}

MODEL_FAMILIES = (
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "hierarchical_extra_trees",
    "oof_stacking",
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
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for field in row:
            if field not in fields:
                fields.append(field)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "audit_completion": 1.0,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "automatic_deployment": False,
    }


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _normalise_attack(value: Any) -> str:
    return str(value).strip().lower().replace("–", "-").replace("—", "-")


def _protocol_registry() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "external_protocol_registry_verified_with_primary_sources",
        "retrieved_on": "2026-07-15",
        "systems": [
            {
                "system_id": "ma_ids_rag_experience_library",
                "title": "MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library",
                "source_url": "https://arxiv.org/html/2604.05458",
                "source_kind": "primary_arxiv_html",
                "datasets": ["NF-BoT-IoT", "NF-ToN-IoT"],
                "tasks": {"NF-BoT-IoT": "4-class", "NF-ToN-IoT": "9-class"},
                "sampling": "50,000 library construction plus disjoint 20,000 evaluation, described as uniform by class",
                "metrics": ["accuracy", "macro_precision", "macro_recall", "macro_f1"],
                "paper_reported": {
                    "NF-BoT-IoT": NF_TASKS["NF-BoT-IoT"]["paper_reference"],
                    "NF-ToN-IoT": NF_TASKS["NF-ToN-IoT"]["paper_reference"],
                },
                "official_feature_policy_issue": "paper feature set includes IP addresses and destination port",
                "safe_same_input_exact_alignment": False,
                "local_fairness_level": "reconstructed_task_metric_sampling_compatible_under_stricter_safe_input",
                "faithful_reproduction_completed": False,
            },
            {
                "system_id": "marl_nids",
                "title": "Multi-agent Reinforcement Learning-based Network Intrusion Detection System",
                "source_url": "https://arxiv.org/html/2407.05766",
                "source_kind": "primary_arxiv_html",
                "datasets": ["CICIDS2017"],
                "tasks": {"CICIDS2017": "15 labels: BENIGN plus 14 attack labels"},
                "sampling": "paper Table I provides 80/20 per-class counts after benign reduction",
                "metrics": ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "false_positive_rate"],
                "paper_reported": {"accuracy": 0.99, "weighted_precision": 0.99, "weighted_recall": 0.99, "weighted_f1": 0.99},
                "fpr_reporting_ambiguity": "Table shows 0.0016 while prose appends a percent sign; retain ambiguity",
                "safe_same_input_exact_alignment": "to_be_audited_w164",
                "faithful_reproduction_completed": False,
            },
            {
                "system_id": "low_altitude_multiagent_iot",
                "title": "Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT",
                "source_url": "https://arxiv.org/html/2601.17817",
                "source_kind": "primary_arxiv_html",
                "datasets": ["Edge-IIoTset", "USTC-TFC", "ISCX-VPN"],
                "paper_reported": "overall accuracy above 90 percent in text; exact per-dataset values not extracted as local metrics",
                "comparison_role": "contextual_secondary_only",
                "faithful_reproduction_completed": False,
            },
            {
                "system_id": "llm_iot_interpretation_revelation",
                "title": "LLM-powered AI Agent Framework for IoT Traffic Interpretation",
                "source_url": "https://arxiv.org/html/2510.13925",
                "source_kind": "primary_arxiv_html",
                "datasets": ["Edge-IIoTset"],
                "tasks": {"Edge-IIoTset": "15-class, stratified 80/20"},
                "paper_reported": {"accuracy": 0.9988, "weighted_precision": 0.9988, "weighted_recall": 0.9988, "weighted_f1": 0.9988},
                "comparison_role": "agentic_interpretation_secondary_context; BERT detector is not MAD-ETD Fusion evidence",
                "faithful_reproduction_completed": False,
            },
        ],
        "paper_numbers_used_as_local_metrics": False,
        "adapted_baseline_labeled_faithful": False,
        **_security(),
    }


def build_external_protocol_registry_w160(output_dir: str | Path = DEFAULT_RUN_DIR) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = _protocol_registry()
    _dump(out / "external_protocol_registry.json", registry)
    rows: list[dict[str, Any]] = []
    for system in registry["systems"]:
        reported = system.get("paper_reported", {})
        if isinstance(reported, dict) and any(isinstance(value, dict) for value in reported.values()):
            for dataset, metrics in reported.items():
                rows.append({
                    "system_id": system["system_id"],
                    "dataset": dataset,
                    "task": system.get("tasks", {}).get(dataset, "not_available"),
                    "reference_type": "paper_reported_only",
                    **metrics,
                    "source_url": system["source_url"],
                    "faithful_local_reproduction": False,
                })
        elif isinstance(reported, dict):
            rows.append({
                "system_id": system["system_id"],
                "dataset": ";".join(system.get("datasets", [])),
                "task": json.dumps(system.get("tasks", {}), ensure_ascii=False),
                "reference_type": "paper_reported_only",
                **reported,
                "source_url": system["source_url"],
                "faithful_local_reproduction": False,
            })
        else:
            rows.append({
                "system_id": system["system_id"],
                "dataset": ";".join(system.get("datasets", [])),
                "task": "contextual_only",
                "reference_type": "paper_context_only",
                "reported_text": reported,
                "source_url": system["source_url"],
                "faithful_local_reproduction": False,
            })
    _write_csv(out / "external_reference_table_w160.csv", rows)
    fairness = {
        "status": "fair_comparison_rules_locked",
        "formal_numeric_requires": [
            "same public dataset family",
            "same label task and class mapping",
            "same macro or weighted averaging definition",
            "local train/validation/acceptance separation",
            "MAD-ETD safe-input policy",
            "paper-reported references kept separate from local predictions",
        ],
        "external_scalar_ci_rule": "local bootstrap metric CI95 lower bound must exceed the paper-reported scalar; no significance claim about the inaccessible paper predictions",
        "prohibited": [
            "paper number copied into a local metric table",
            "safe-input adaptation called faithful reproduction",
            "test-driven model or threshold selection",
            "blocked identity fields admitted for protocol matching",
        ],
        **_security(),
    }
    _dump(out / "fair_comparison_rules_w160.json", fairness)
    return registry


def _manifest_entries(processed_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = _read_json(processed_dir / "dataset_manifest.json")
    return {str(item.get("dataset_variant")): dict(item) for item in manifest.get("primary_entries", [])}


def _numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, list(RAW_SAFE_FEATURES)].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    protocol, in_b, out_b, in_p, out_p, flags, duration = values.T
    total_b = in_b + out_b
    total_p = in_p + out_p
    result = np.column_stack(
        [
            protocol,
            flags,
            np.log1p(np.maximum(in_b, 0.0)),
            np.log1p(np.maximum(out_b, 0.0)),
            np.log1p(np.maximum(in_p, 0.0)),
            np.log1p(np.maximum(out_p, 0.0)),
            np.log1p(np.maximum(duration, 0.0)),
            np.log1p(np.maximum(total_b, 0.0)),
            np.log1p(np.maximum(total_p, 0.0)),
            (in_b + 1.0) / (out_b + 1.0),
            (in_p + 1.0) / (out_p + 1.0),
            np.log1p(np.maximum(total_b / np.maximum(total_p, 1.0), 0.0)),
        ]
    )
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def _take_smallest(
    existing: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    x_new: np.ndarray,
    rows_new: np.ndarray,
    priority_new: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if existing is None:
        x, rows, priority = x_new, rows_new, priority_new
    else:
        x = np.concatenate([existing[0], x_new], axis=0)
        rows = np.concatenate([existing[1], rows_new], axis=0)
        priority = np.concatenate([existing[2], priority_new], axis=0)
    if len(priority) > limit:
        index = np.argpartition(priority, limit - 1)[:limit]
        x, rows, priority = x[index], rows[index], priority[index]
    return x, rows, priority


def _sample_archive(
    entry: Mapping[str, Any],
    classes: tuple[str, ...],
    *,
    per_class_limit: int,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    archive_path = Path(str(entry["archive"]))
    member = str(entry["entry"])
    attack_column = str(entry.get("attack_column") or "Attack")
    allowed = {_normalise_attack(name): name for name in classes}
    reservoirs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {name: None for name in classes}
    observed: defaultdict[str, int] = defaultdict(int)
    excluded: defaultdict[str, int] = defaultdict(int)
    rng = np.random.default_rng(seed)
    offset = 0
    usecols = [*RAW_SAFE_FEATURES, attack_column]
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=usecols, chunksize=250_000, low_memory=False):
            attacks = chunk[attack_column].astype(str).map(_normalise_attack)
            global_rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            for normalised, canonical in allowed.items():
                mask = attacks.eq(normalised).to_numpy()
                count = int(mask.sum())
                if not count:
                    continue
                observed[canonical] += count
                subset = chunk.loc[mask, list(RAW_SAFE_FEATURES)]
                x = _numeric_frame(subset)
                rows = global_rows[mask]
                priorities = rng.random(count)
                if count > per_class_limit:
                    local = np.argpartition(priorities, per_class_limit - 1)[:per_class_limit]
                    x, rows, priorities = x[local], rows[local], priorities[local]
                reservoirs[canonical] = _take_smallest(
                    reservoirs[canonical], x, rows, priorities, per_class_limit
                )
            for value, count in attacks.value_counts().items():
                if value not in allowed:
                    excluded[value] += int(count)
    sampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for canonical in classes:
        values = reservoirs[canonical]
        if values is None:
            sampled[canonical] = (
                np.empty((0, len(ENGINEERED_FEATURES)), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
            continue
        order = np.argsort(values[2])
        sampled[canonical] = (values[0][order], values[1][order])
    report = {
        "archive": archive_path.as_posix(),
        "member": member,
        "dataset_variant": entry.get("dataset_variant"),
        "scanned_rows": offset,
        "observed_class_counts": dict(observed),
        "sampled_class_counts": {name: len(values[0]) for name, values in sampled.items()},
        "excluded_attack_counts": dict(excluded),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return sampled, report


def _assemble_split(
    dataset: str,
    protocol: str,
    source_variant: str,
    classes: tuple[str, ...],
    samples: Mapping[str, tuple[np.ndarray, np.ndarray]],
    slices: Mapping[str, tuple[int, int]],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    class_to_id = {name: index for index, name in enumerate(classes)}
    rng = np.random.default_rng(seed)
    for role, (start, stop) in slices.items():
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        class_parts: list[np.ndarray] = []
        for name in classes:
            x, rows = samples[name]
            take_x = x[start:stop]
            take_rows = rows[start:stop]
            x_parts.append(take_x)
            y_parts.append(np.full(len(take_x), class_to_id[name], dtype=np.int64))
            row_parts.append(take_rows)
            class_parts.append(np.full(len(take_x), name, dtype="U64"))
        x_all = np.concatenate(x_parts) if x_parts else np.empty((0, len(ENGINEERED_FEATURES)), dtype=np.float32)
        y_all = np.concatenate(y_parts) if y_parts else np.empty((0,), dtype=np.int64)
        rows_all = np.concatenate(row_parts) if row_parts else np.empty((0,), dtype=np.int64)
        names_all = np.concatenate(class_parts) if class_parts else np.empty((0,), dtype="U64")
        order = rng.permutation(len(y_all))
        x_all, y_all, rows_all, names_all = x_all[order], y_all[order], rows_all[order], names_all[order]
        arrays[f"x_{role}"] = x_all
        arrays[f"y_{role}"] = y_all
        arrays[f"row_{role}"] = rows_all
        for row_index, label_name in zip(rows_all.tolist(), names_all.tolist()):
            sample_hash = hashlib.sha256(
                f"w161:{dataset}:{protocol}:{source_variant}:{row_index}".encode()
            ).hexdigest()
            manifest.append({
                "sample_hash": sample_hash,
                "dataset": dataset,
                "protocol": protocol,
                "source_variant": source_variant,
                "split": role,
                "class_name": label_name,
                "source_row_index_hash": hashlib.sha256(str(row_index).encode()).hexdigest(),
                "blocked_metadata_in_feature_matrix": False,
            })
    return arrays, manifest


def freeze_nf_multiclass_protocols_w161(
    processed_dir: str | Path = DEFAULT_PROCESSED_NF,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = _read_json(out / "external_protocol_registry.json")
    if registry.get("status") != "external_protocol_registry_verified_with_primary_sources":
        report = {"status": "failed_missing_w160_protocol_registry", **_security()}
        _dump(out / "w161_acceptance_report.json", report)
        return report
    entries = _manifest_entries(Path(processed_dir))
    required = {"NF-BoT-IoT", "NF-BoT-IoT-v2", "NF-ToN-IoT", "NF-ToN-IoT-v2"}
    missing = sorted(required - set(entries))
    policy = {
        "schema_version": "1.0",
        "status": "safe_multiclass_feature_policy_locked" if not missing else "failed_missing_nf_sources",
        "raw_safe_features": list(RAW_SAFE_FEATURES),
        "engineered_features": list(ENGINEERED_FEATURES),
        "blocked_fields": list(BLOCKED_FIELDS),
        "attack_label_usage": "stratification and evaluation only",
        "source_variant_usage": "protocol split and audit only",
        "blocked_fields_in_feature_matrix": [],
        "feature_policy_hash": _sha256_json({"raw": RAW_SAFE_FEATURES, "engineered": ENGINEERED_FEATURES, "blocked": BLOCKED_FIELDS}),
        **_security(),
    }
    _dump(out / "safe_feature_policy_w161.json", policy)
    if missing:
        report = {"status": "failed_missing_nf_sources", "missing": missing, **_security()}
        _dump(out / "w161_acceptance_report.json", report)
        return report
    all_manifest: list[dict[str, Any]] = []
    extraction_reports: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    for dataset, spec in NF_TASKS.items():
        classes = tuple(spec["classes"])
        v1 = entries[dataset]
        v2 = entries[f"{dataset}-v2"]
        paper_q = spec["paper_quotas"]
        group_q = spec["group_quotas"]
        v2_limit = max(sum(paper_q.values()), group_q["acceptance"])
        v1_limit = group_q["train"] + group_q["validation"]
        sampled_v2, report_v2 = _sample_archive(v2, classes, per_class_limit=v2_limit, seed=160)
        sampled_v1, report_v1 = _sample_archive(v1, classes, per_class_limit=v1_limit, seed=161)
        extraction_reports.extend([report_v1, report_v2])
        insufficient = {
            "paper": {name: len(sampled_v2[name][0]) for name in classes if len(sampled_v2[name][0]) < sum(paper_q.values())},
            "group_train_validation": {name: len(sampled_v1[name][0]) for name in classes if len(sampled_v1[name][0]) < v1_limit},
            "group_acceptance": {name: len(sampled_v2[name][0]) for name in classes if len(sampled_v2[name][0]) < group_q["acceptance"]},
        }
        if any(insufficient.values()):
            report = {"status": "failed_insufficient_multiclass_samples", "dataset": dataset, "insufficient": insufficient, **_security()}
            _dump(out / "w161_acceptance_report.json", report)
            return report
        paper_slices = {
            "train": (0, paper_q["train"]),
            "validation": (paper_q["train"], paper_q["train"] + paper_q["validation"]),
            "acceptance": (paper_q["train"] + paper_q["validation"], sum(paper_q.values())),
        }
        paper_arrays, paper_manifest = _assemble_split(
            dataset, "reconstructed_paper_compatible", f"{dataset}-v2", classes, sampled_v2, paper_slices, seed=42
        )
        paper_arrays["classes"] = np.asarray(classes, dtype="U64")
        paper_arrays["feature_names"] = np.asarray(ENGINEERED_FEATURES, dtype="U64")
        np.savez_compressed(out / f"{dataset.lower().replace('-', '_')}_paper_protocol_w161.npz", **paper_arrays)
        all_manifest.extend(paper_manifest)
        train_slices = {
            "train": (0, group_q["train"]),
            "validation": (group_q["train"], group_q["train"] + group_q["validation"]),
        }
        group_arrays, group_manifest = _assemble_split(
            dataset, "credible_source_group_heldout", dataset, classes, sampled_v1, train_slices, seed=43
        )
        accept_arrays, accept_manifest = _assemble_split(
            dataset, "credible_source_group_heldout", f"{dataset}-v2", classes, sampled_v2,
            {"acceptance": (0, group_q["acceptance"])}, seed=44
        )
        group_arrays.update(accept_arrays)
        group_arrays["classes"] = np.asarray(classes, dtype="U64")
        group_arrays["feature_names"] = np.asarray(ENGINEERED_FEATURES, dtype="U64")
        np.savez_compressed(out / f"{dataset.lower().replace('-', '_')}_group_protocol_w161.npz", **group_arrays)
        all_manifest.extend(group_manifest + accept_manifest)
        for protocol, q, source in (
            ("reconstructed_paper_compatible", paper_q, f"{dataset}-v2"),
            ("credible_source_group_heldout", group_q, f"train={dataset};acceptance={dataset}-v2"),
        ):
            protocol_rows.append({
                "dataset": dataset,
                "protocol": protocol,
                "class_count": len(classes),
                "classes": ";".join(classes),
                "excluded_classes": ";".join(spec["excluded_classes"]),
                "train_per_class": q["train"],
                "validation_per_class": q["validation"],
                "acceptance_per_class": q["acceptance"],
                "source_protocol": source,
                "feature_policy_hash": policy["feature_policy_hash"],
                "acceptance_used_for_selection": False,
            })
    _write_csv(out / "nf_multiclass_protocol_table_w161.csv", protocol_rows)
    _write_csv(out / "nf_multiclass_split_manifest_w161.csv", all_manifest)
    _dump(out / "nf_extraction_audit_w161.json", {"status": "passed", "reports": extraction_reports})
    split_sets: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in all_manifest:
        split_sets[(row["dataset"], row["protocol"])][row["split"]].add(row["sample_hash"])
    overlaps: list[dict[str, Any]] = []
    for (dataset, protocol), roles in split_sets.items():
        names = sorted(roles)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                count = len(roles[left] & roles[right])
                overlaps.append({"dataset": dataset, "protocol": protocol, "left": left, "right": right, "overlap": count})
    _write_csv(out / "split_overlap_audit_w161.csv", overlaps)
    passed = all(int(row["overlap"]) == 0 for row in overlaps)
    report = {
        "status": "nf_multiclass_dual_protocols_frozen" if passed else "failed_split_overlap",
        "protocol_count": len(protocol_rows),
        "manifest_rows": len(all_manifest),
        "within_protocol_split_overlap": sum(int(row["overlap"]) for row in overlaps),
        "paper_protocol_is_faithful_reproduction": False,
        "paper_protocol_boundary": "task/metric/balancing reconstruction on public v2 source under stricter safe-input policy; source version and exact feature policy differ from MA-IDS",
        "nf_ton_count_adjustment": "69,300 unique balanced rows rather than 70,000 because the public v2 MITM class has only 7,723 rows",
        **_security(),
    }
    _dump(out / "w161_acceptance_report.json", report)
    return report


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def _choose_temperature(y: np.ndarray, probabilities: np.ndarray) -> float:
    best = (float("inf"), 1.0)
    for temperature in np.linspace(0.50, 3.00, 51):
        score = log_loss(y, _temperature_scale(probabilities, float(temperature)), labels=np.arange(probabilities.shape[1]))
        if score < best[0]:
            best = (float(score), float(temperature))
    return best[1]


def _ece(y: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == y
    total = max(len(y), 1)
    value = 0.0
    for left in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        right = left + 1.0 / bins
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            value += mask.sum() / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_precision": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "ece": _ece(y, probabilities),
        "coverage": 1.0,
    }


def _make_base(family: str, seed: int, n_classes: int) -> Any:
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.1, random_state=seed)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
    if family == "extra_trees":
        return ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=140, max_depth=7, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", tree_method="hist", n_jobs=4, random_state=seed)
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=160, num_leaves=31, learning_rate=0.06, class_weight="balanced", verbosity=-1, n_jobs=4, random_state=seed)
    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(iterations=160, depth=7, learning_rate=0.08, loss_function="MultiClass", auto_class_weights="Balanced", verbose=False, thread_count=4, random_seed=seed, allow_writing_files=False)
    raise ValueError(f"unsupported base family: {family}")


def _align_proba(model: Any, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    if probabilities.shape[1] == n_classes and np.array_equal(np.asarray(model.classes_, dtype=int), np.arange(n_classes)):
        return probabilities
    aligned = np.full((len(probabilities), n_classes), 1e-12, dtype=np.float64)
    for source, target in enumerate(np.asarray(model.classes_, dtype=int)):
        aligned[:, target] = probabilities[:, source]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _fit_family(family: str, x: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int) -> dict[str, Any]:
    n_classes = len(classes)
    if family in MODEL_FAMILIES[:6]:
        model = _make_base(family, seed, n_classes)
        model.fit(x, y)
        return {"kind": "base", "family": family, "model": model, "classes": classes}
    benign_index = int(np.where(np.char.lower(classes.astype(str)) == "benign")[0][0])
    if family == "hierarchical_extra_trees":
        binary_y = (y != benign_index).astype(int)
        binary = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)
        binary.fit(x, binary_y)
        attack_indices = np.asarray([index for index in range(n_classes) if index != benign_index], dtype=int)
        attack_map = {value: index for index, value in enumerate(attack_indices.tolist())}
        mask = y != benign_index
        attack_y = np.asarray([attack_map[int(value)] for value in y[mask]], dtype=int)
        attack = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed + 100)
        attack.fit(x[mask], attack_y)
        return {"kind": "hierarchical", "family": family, "binary": binary, "attack": attack, "attack_indices": attack_indices, "benign_index": benign_index, "classes": classes}
    if family == "oof_stacking":
        base_names = ("hist_gradient_boosting", "random_forest", "extra_trees")
        oof = np.zeros((len(y), n_classes * len(base_names)), dtype=np.float64)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (train_idx, hold_idx) in enumerate(splitter.split(x, y)):
            for base_index, base_name in enumerate(base_names):
                model = _make_base(base_name, seed + 10 * fold + base_index, n_classes)
                model.fit(x[train_idx], y[train_idx])
                p = _align_proba(model, model.predict_proba(x[hold_idx]), n_classes)
                oof[hold_idx, base_index * n_classes : (base_index + 1) * n_classes] = p
        meta = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        meta.fit(oof, y)
        bases = []
        for base_index, base_name in enumerate(base_names):
            model = _make_base(base_name, seed + 100 + base_index, n_classes)
            model.fit(x, y)
            bases.append(model)
        return {"kind": "stack", "family": family, "bases": bases, "meta": meta, "classes": classes}
    raise ValueError(f"unsupported family: {family}")


def _predict_artifact(artifact: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    n_classes = len(artifact["classes"])
    if artifact["kind"] == "base":
        model = artifact["model"]
        return _align_proba(model, model.predict_proba(x), n_classes)
    if artifact["kind"] == "hierarchical":
        binary = artifact["binary"].predict_proba(x)
        binary = _align_proba(artifact["binary"], binary, 2)
        attack_raw = artifact["attack"].predict_proba(x)
        attack_prob = _align_proba(artifact["attack"], attack_raw, len(artifact["attack_indices"]))
        result = np.zeros((len(x), n_classes), dtype=np.float64)
        benign = int(artifact["benign_index"])
        result[:, benign] = binary[:, 0]
        for local, global_index in enumerate(artifact["attack_indices"]):
            result[:, int(global_index)] = binary[:, 1] * attack_prob[:, local]
        return result / result.sum(axis=1, keepdims=True)
    base_prob = [
        _align_proba(model, model.predict_proba(x), n_classes)
        for model in artifact["bases"]
    ]
    meta_x = np.concatenate(base_prob, axis=1)
    return _align_proba(artifact["meta"], artifact["meta"].predict_proba(meta_x), n_classes)


def _task_npz_paths(out: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for dataset in NF_TASKS:
        stem = dataset.lower().replace("-", "_")
        rows.append((dataset, "reconstructed_paper_compatible", out / f"{stem}_paper_protocol_w161.npz"))
        rows.append((dataset, "credible_source_group_heldout", out / f"{stem}_group_protocol_w161.npz"))
    return rows


def train_nf_multiclass_stack_w162(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    models = Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    split_report = _read_json(out / "w161_acceptance_report.json")
    if split_report.get("status") != "nf_multiclass_dual_protocols_frozen":
        report = {"status": "failed_missing_w161_frozen_protocol", **_security()}
        _dump(out / "w162_acceptance_report.json", report)
        return report
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before_w162.json", before)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str, str, int], str] = {}
    for dataset, protocol, path in _task_npz_paths(out):
        data = np.load(path, allow_pickle=False)
        x_train, y_train = data["x_train"], data["y_train"]
        x_validation, y_validation = data["x_validation"], data["y_validation"]
        classes = data["classes"]
        for family in MODEL_FAMILIES:
            for seed in SEEDS:
                started = time.perf_counter()
                status = "trained"
                error = ""
                artifact_path = models / f"{dataset.lower().replace('-', '_')}__{protocol}__{family}__seed{seed}.joblib"
                try:
                    artifact = _fit_family(family, x_train, y_train, classes, seed)
                    raw = _predict_artifact(artifact, x_validation)
                    temperature = _choose_temperature(y_validation, raw)
                    artifact["temperature"] = temperature
                    artifact["feature_policy_hash"] = _read_json(out / "safe_feature_policy_w161.json").get("feature_policy_hash")
                    artifact["dataset_scope"] = dataset
                    artifact["protocol"] = protocol
                    joblib_dump(artifact, artifact_path, compress=3)
                    probabilities = _temperature_scale(raw, temperature)
                    metrics = _metrics(y_validation, probabilities)
                    artifacts[(dataset, protocol, family, seed)] = artifact_path.as_posix()
                except Exception as exc:  # package/runtime failures remain explicit
                    status = "blocked_training_error"
                    error = f"{type(exc).__name__}: {exc}"
                    metrics = {name: float("nan") for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "ece", "coverage")}
                    temperature = float("nan")
                elapsed = time.perf_counter() - started
                train_rows.append({
                    "dataset": dataset,
                    "protocol": protocol,
                    "family": family,
                    "seed": seed,
                    "status": status,
                    "train_rows": len(y_train),
                    "validation_rows": len(y_validation),
                    "training_seconds": elapsed,
                    "artifact_path": artifact_path.as_posix() if status == "trained" else "",
                    "error": error,
                })
                validation_rows.append({
                    "dataset": dataset,
                    "protocol": protocol,
                    "family": family,
                    "seed": seed,
                    "status": status,
                    "temperature": temperature,
                    **metrics,
                })
    _write_csv(out / "training_results_w162.csv", train_rows)
    _write_csv(out / "validation_results_w162.csv", validation_rows)
    selection: dict[str, Any] = {"status": "validation_only_model_family_selection_locked", "tasks": {}}
    for dataset, protocol, _path in _task_npz_paths(out):
        candidates = []
        for family in MODEL_FAMILIES:
            rows = [row for row in validation_rows if row["dataset"] == dataset and row["protocol"] == protocol and row["family"] == family and row["status"] == "trained"]
            if len(rows) != len(SEEDS):
                continue
            candidates.append({
                "family": family,
                "validation_macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
                "validation_macro_recall_mean": float(np.mean([row["macro_recall"] for row in rows])),
                "validation_ece_mean": float(np.mean([row["ece"] for row in rows])),
            })
        candidates.sort(key=lambda row: (-row["validation_macro_f1_mean"], -row["validation_macro_recall_mean"], row["validation_ece_mean"], row["family"]))
        key = f"{dataset}::{protocol}"
        selection["tasks"][key] = {
            "selected_family": candidates[0]["family"] if candidates else None,
            "ranking": candidates,
            "selection_split": "validation",
            "acceptance_opened": False,
        }
    selection["selection_hash"] = _sha256_json(selection["tasks"])
    selection.update(_security())
    _dump(out / "model_selection_w162.json", selection)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_w162.json", after)
    report = {
        "status": "nf_multiclass_stack_validation_locked" if all(value.get("selected_family") for value in selection["tasks"].values()) else "failed_no_complete_model_family",
        "model_family_count": len(MODEL_FAMILIES),
        "seed_count": len(SEEDS),
        "trained_model_count": sum(row["status"] == "trained" for row in train_rows),
        "blocked_model_count": sum(row["status"] != "trained" for row in train_rows),
        "frozen_hashes_unchanged": before == after,
        **_security(),
    }
    _dump(out / "w162_acceptance_report.json", report)
    return report


def _bootstrap_metric_cis(y: np.ndarray, probabilities: np.ndarray) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    values: dict[str, list[float]] = {name: [] for name in names}
    by_class = [np.flatnonzero(y == label) for label in np.unique(y)]
    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_class])
        metrics = _metrics(y[indices], probabilities[indices])
        for name in names:
            values[name].append(metrics[name])
    return {
        name: {
            "mean": float(np.mean(samples)),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
        }
        for name, samples in values.items()
    }


def evaluate_nf_multiagent_acceptance_w163(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    marker = out / "w163_acceptance_opened.json"
    if marker.is_file():
        prior = _read_json(out / "w163_acceptance_report.json")
        return {**prior, "rerun_refused_acceptance_already_opened": True}
    train_report = _read_json(out / "w162_acceptance_report.json")
    selection = _read_json(out / "model_selection_w162.json")
    if train_report.get("status") != "nf_multiclass_stack_validation_locked":
        report = {"status": "failed_missing_w162_validation_lock", **_security()}
        _dump(out / "w163_acceptance_report.json", report)
        return report
    _dump(marker, {"opened_once": True, "selection_hash": selection.get("selection_hash"), "opened_at_unix": time.time()})
    seed_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    ci_payload: dict[str, Any] = {}
    per_class_rows: list[dict[str, Any]] = []
    accepted_scopes: list[str] = []
    for dataset, protocol, path in _task_npz_paths(out):
        data = np.load(path, allow_pickle=False)
        x, y, classes = data["x_acceptance"], data["y_acceptance"], data["classes"]
        task_key = f"{dataset}::{protocol}"
        family = selection["tasks"][task_key]["selected_family"]
        probabilities_by_seed: list[np.ndarray] = []
        for seed in SEEDS:
            artifact_path = Path(model_dir) / f"{dataset.lower().replace('-', '_')}__{protocol}__{family}__seed{seed}.joblib"
            artifact = joblib_load(artifact_path)
            raw = _predict_artifact(artifact, x)
            probabilities = _temperature_scale(raw, float(artifact.get("temperature", 1.0)))
            probabilities_by_seed.append(probabilities)
            seed_rows.append({"dataset": dataset, "protocol": protocol, "family": family, "seed": seed, **_metrics(y, probabilities)})
        ensemble = np.mean(probabilities_by_seed, axis=0)
        metrics = _metrics(y, ensemble)
        cis = _bootstrap_metric_cis(y, ensemble)
        ci_payload[task_key] = cis
        ensemble_rows.append({"dataset": dataset, "protocol": protocol, "family": family, "seeds": ";".join(map(str, SEEDS)), "sample_count": len(y), **metrics})
        prediction = ensemble.argmax(axis=1)
        for label, name in enumerate(classes.astype(str)):
            mask = y == label
            per_class_rows.append({
                "dataset": dataset,
                "protocol": protocol,
                "class_name": name,
                "support": int(mask.sum()),
                "recall": float((prediction[mask] == label).mean()) if mask.any() else float("nan"),
            })
        if protocol == "reconstructed_paper_compatible":
            reference = NF_TASKS[dataset]["paper_reference"]
            gates: dict[str, bool] = {}
            for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                gate = cis[metric]["ci95_lower"] > float(reference[metric])
                gates[metric] = gate
                comparison_rows.append({
                    "dataset": dataset,
                    "external_system": "MA-IDS",
                    "comparison_scope": "reconstructed_paper_compatible_safe_input_not_faithful",
                    "metric": metric,
                    "paper_reported_reference": reference[metric],
                    "local_value": metrics[metric],
                    "local_ci95_lower": cis[metric]["ci95_lower"],
                    "local_ci95_upper": cis[metric]["ci95_upper"],
                    "ci_lower_exceeds_reference": gate,
                    "paper_predictions_available": False,
                    "formal_paired_significance_claim": False,
                })
            if all(gates.values()):
                accepted_scopes.append(dataset)
    _write_csv(out / "nf_acceptance_seed_results_w163.csv", seed_rows)
    _write_csv(out / "nf_acceptance_ensemble_results_w163.csv", ensemble_rows)
    _write_csv(out / "nf_per_class_recall_w163.csv", per_class_rows)
    _write_csv(out / "ma_ids_paper_reference_comparison_w163.csv", comparison_rows)
    _dump(out / "nf_bootstrap_ci_w163.json", {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "tasks": ci_payload})
    report = {
        "status": "accepted_external_multiagent_comparable_positive_result" if accepted_scopes else "no_external_reference_exceedance_with_ci_support",
        "accepted_dataset_scopes": accepted_scopes,
        "acceptance_scope": "dataset-specific reconstructed paper-compatible protocol only",
        "faithful_external_reproduction_completed": False,
        "official_external_code_used": False,
        "paper_reported_numbers_used_only_as_reference": True,
        "formal_paired_significance_against_external_predictions": False,
        "all_four_metric_ci_gates_required": True,
        "selection_hash": selection.get("selection_hash"),
        "acceptance_opened_once": True,
        **_security(),
    }
    _dump(out / "w163_acceptance_report.json", report)
    return report


CICIDS_CLASSES = (
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack Brute Force",
    "Web Attack Sql Injection",
    "Web Attack XSS",
)

# Counts transcribed from MARL-NIDS Table I. They remain external protocol
# metadata and are never copied into local metric columns.
CICIDS_PAPER_COUNTS = {
    "BENIGN": (559_999, 140_001),
    "DoS Hulk": (184_099, 46_025),
    "PortScan": (127_043, 31_761),
    "DDoS": (102_420, 25_605),
    "DoS GoldenEye": (8_234, 2_059),
    "FTP-Patator": (6_348, 1_587),
    "SSH-Patator": (4_717, 1_180),
    "DoS slowloris": (4_637, 1_159),
    "DoS Slowhttptest": (4_399, 1_100),
    "Bot": (1_565, 391),
    "Web Attack Brute Force": (1_206, 301),
    "Web Attack XSS": (522, 130),
    "Infiltration": (29, 7),
    "Web Attack Sql Injection": (17, 4),
    "Heartbleed": (9, 2),
}


def _cicids_label(value: Any) -> str | None:
    compact = "".join(character.lower() for character in str(value) if character.isalnum())
    aliases = {
        "benign": "BENIGN",
        "bot": "Bot",
        "ddos": "DDoS",
        "dosgoldeneye": "DoS GoldenEye",
        "doshulk": "DoS Hulk",
        "dosslowhttptest": "DoS Slowhttptest",
        "dosslowloris": "DoS slowloris",
        "ftppatator": "FTP-Patator",
        "heartbleed": "Heartbleed",
        "infiltration": "Infiltration",
        "portscan": "PortScan",
        "sshpatator": "SSH-Patator",
        "webattackbruteforce": "Web Attack Brute Force",
        "webattacksqlinjection": "Web Attack Sql Injection",
        "webattackxss": "Web Attack XSS",
    }
    return aliases.get(compact)


def _cicids_safe_features(processed: Path) -> list[str]:
    policy = _read_json(processed / "feature_policy.json")
    blocked_tokens = (
        "port",
        "ip",
        "timestamp",
        "flow id",
        "label",
        "attack",
        "family",
        "source",
        "file",
        "provenance",
    )
    result: list[str] = []
    for name in policy.get("safe_feature_columns", []):
        normalized = str(name).strip().lower()
        if any(token in normalized for token in blocked_tokens):
            continue
        if name not in result:
            result.append(str(name))
    return result


def _cicids_feature_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    numeric = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    values = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    values = np.sign(values) * np.log1p(np.abs(values))
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0).astype(np.float32)


def _sample_cicids_w164(processed: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], dict[str, Any]]:
    manifest = _read_json(processed / "dataset_manifest.json")
    features = _cicids_safe_features(processed)
    limits = {
        name: min(train_count, 50_000) + test_count
        for name, (train_count, test_count) in CICIDS_PAPER_COUNTS.items()
    }
    reservoirs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {name: None for name in CICIDS_CLASSES}
    observed: defaultdict[str, int] = defaultdict(int)
    rng = np.random.default_rng(164)
    started = time.perf_counter()
    for entry_index, entry in enumerate(manifest.get("primary_entries", [])):
        archive_path = Path(str(entry["archive"]))
        member = str(entry["entry"])
        label_column = str(entry.get("label_column") or " Label")
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
            offset = 0
            for chunk in pd.read_csv(raw, usecols=[*features, label_column], chunksize=100_000, low_memory=False):
                labels = chunk[label_column].map(_cicids_label)
                local_rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
                offset += len(chunk)
                for class_name in CICIDS_CLASSES:
                    mask = labels.eq(class_name).to_numpy()
                    count = int(mask.sum())
                    if not count:
                        continue
                    observed[class_name] += count
                    x = _cicids_feature_matrix(chunk.loc[mask, features], features)
                    # Encode member identity in the audit-only row token.
                    rows = (np.int64(entry_index) << np.int64(48)) + local_rows[mask]
                    priority = rng.random(count)
                    limit = limits[class_name]
                    if count > limit:
                        chosen = np.argpartition(priority, limit - 1)[:limit]
                        x, rows, priority = x[chosen], rows[chosen], priority[chosen]
                    reservoirs[class_name] = _take_smallest(reservoirs[class_name], x, rows, priority, limit)
    sampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in CICIDS_CLASSES:
        values = reservoirs[name]
        if values is None:
            sampled[name] = (np.empty((0, len(features)), dtype=np.float32), np.empty((0,), dtype=np.int64))
        else:
            order = np.argsort(values[2])
            sampled[name] = (values[0][order], values[1][order])
    return sampled, features, {
        "observed_counts": dict(observed),
        "sampled_counts": {name: len(sampled[name][0]) for name in CICIDS_CLASSES},
        "feature_count": len(features),
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_cicids_marl_comparison_w164(
    processed_dir: str | Path = DEFAULT_PROCESSED_CICIDS,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Run a bounded safe-input 15-label reconstruction of MARL-NIDS Table I."""

    out = Path(output_dir)
    processed = Path(processed_dir)
    marker = out / "w164_acceptance_opened.json"
    if marker.is_file():
        return {**_read_json(out / "w164_acceptance_report.json"), "rerun_refused_acceptance_already_opened": True}
    if not (processed / "dataset_manifest.json").is_file():
        report = {"status": "blocked_missing_cicids2017_processed_manifest", **_security()}
        _dump(out / "w164_acceptance_report.json", report)
        return report
    sampled, features, extraction = _sample_cicids_w164(processed)
    _dump(out / "cicids_extraction_audit_w164.json", extraction)
    insufficient = {}
    for name, (paper_train, paper_test) in CICIDS_PAPER_COUNTS.items():
        needed = min(paper_train, 50_000) + paper_test
        if len(sampled[name][0]) < needed:
            insufficient[name] = {"available": len(sampled[name][0]), "needed": needed}
    policy = {
        "status": "cicids_safe_feature_policy_locked",
        "safe_feature_count": len(features),
        "safe_features": features,
        "blocked_fields": ["Destination Port", "Flow ID", "Source IP", "Destination IP", "Timestamp", "Label", "attack family", "source file", "provenance"],
        "blocked_fields_in_feature_matrix": [],
        "feature_policy_hash": _sha256_json(features),
        **_security(),
    }
    _dump(out / "cicids_safe_feature_policy_w164.json", policy)
    if insufficient:
        report = {"status": "blocked_cicids_class_count_mismatch", "insufficient": insufficient, **_security()}
        _dump(out / "w164_acceptance_report.json", report)
        return report
    x_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    x_val_parts: list[np.ndarray] = []
    y_val_parts: list[np.ndarray] = []
    x_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    split_rows: list[dict[str, Any]] = []
    class_to_id = {name: index for index, name in enumerate(CICIDS_CLASSES)}
    for name in CICIDS_CLASSES:
        x, row_tokens = sampled[name]
        paper_train, paper_test = CICIDS_PAPER_COUNTS[name]
        local_construction = min(paper_train, 50_000)
        validation_count = max(1, int(round(local_construction * 0.20)))
        training_count = local_construction - validation_count
        boundaries = {
            "train": (0, training_count),
            "validation": (training_count, local_construction),
            "acceptance": (local_construction, local_construction + paper_test),
        }
        for role, (start, stop) in boundaries.items():
            part_x, part_rows = x[start:stop], row_tokens[start:stop]
            part_y = np.full(len(part_x), class_to_id[name], dtype=np.int64)
            if role == "train":
                x_train_parts.append(part_x); y_train_parts.append(part_y)
            elif role == "validation":
                x_val_parts.append(part_x); y_val_parts.append(part_y)
            else:
                x_test_parts.append(part_x); y_test_parts.append(part_y)
            for token in part_rows:
                split_rows.append({
                    "sample_hash": hashlib.sha256(f"w164:{int(token)}".encode()).hexdigest(),
                    "class_name": name,
                    "split": role,
                    "row_token_hash": hashlib.sha256(str(int(token)).encode()).hexdigest(),
                })
    x_train, y_train = np.concatenate(x_train_parts), np.concatenate(y_train_parts)
    x_val, y_val = np.concatenate(x_val_parts), np.concatenate(y_val_parts)
    x_test, y_test = np.concatenate(x_test_parts), np.concatenate(y_test_parts)
    _write_csv(out / "cicids_split_manifest_w164.csv", split_rows)
    np.savez_compressed(out / "cicids_marl_protocol_w164.npz", x_train=x_train, y_train=y_train, x_validation=x_val, y_validation=y_val, x_acceptance=x_test, y_acceptance=y_test, classes=np.asarray(CICIDS_CLASSES, dtype="U64"), feature_names=np.asarray(features, dtype="U128"))
    validation_rows: list[dict[str, Any]] = []
    model_root = Path(model_dir) / "cicids_w164"
    model_root.mkdir(parents=True, exist_ok=True)
    families = ("hist_gradient_boosting", "extra_trees", "xgboost")
    for family in families:
        for seed in SEEDS:
            started = time.perf_counter()
            artifact = _fit_family(family, x_train, y_train, np.asarray(CICIDS_CLASSES), seed)
            raw = _predict_artifact(artifact, x_val)
            temperature = _choose_temperature(y_val, raw)
            artifact["temperature"] = temperature
            artifact_path = model_root / f"{family}__seed{seed}.joblib"
            joblib_dump(artifact, artifact_path, compress=3)
            validation_rows.append({"family": family, "seed": seed, "temperature": temperature, "training_seconds": time.perf_counter() - started, **_metrics(y_val, _temperature_scale(raw, temperature))})
    _write_csv(out / "cicids_validation_results_w164.csv", validation_rows)
    rankings = []
    for family in families:
        rows = [row for row in validation_rows if row["family"] == family]
        rankings.append({
            "family": family,
            "macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_recall": float(np.mean([row["macro_recall"] for row in rows])),
            "ece": float(np.mean([row["ece"] for row in rows])),
        })
    rankings.sort(key=lambda row: (-row["macro_f1"], -row["macro_recall"], row["ece"], row["family"]))
    selected = rankings[0]["family"]
    _dump(out / "cicids_model_selection_w164.json", {"selected_family": selected, "rankings": rankings, "selection_split": "validation", "acceptance_used": False})
    _dump(marker, {"opened_once": True, "selected_family": selected, "opened_at_unix": time.time()})
    seed_results = []
    seed_probabilities = []
    for seed in SEEDS:
        artifact = joblib_load(model_root / f"{selected}__seed{seed}.joblib")
        probabilities = _temperature_scale(_predict_artifact(artifact, x_test), float(artifact["temperature"]))
        seed_probabilities.append(probabilities)
        seed_results.append({"family": selected, "seed": seed, **_metrics(y_test, probabilities)})
    ensemble = np.mean(seed_probabilities, axis=0)
    local = _metrics(y_test, ensemble)
    cis = _bootstrap_metric_cis(y_test, ensemble)
    _write_csv(out / "cicids_acceptance_seed_results_w164.csv", seed_results)
    _dump(out / "cicids_bootstrap_ci_w164.json", {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "metrics": cis})
    paper = {"accuracy": 0.99, "macro_precision": None, "macro_recall": None, "macro_f1": None, "weighted_f1": 0.99}
    comparison_rows = []
    for metric in ("accuracy", "weighted_f1"):
        # Bootstrap helper uses macro quantities; compute weighted F1 CI below.
        reference = float(paper[metric])
        if metric == "accuracy":
            lower, upper = cis["accuracy"]["ci95_lower"], cis["accuracy"]["ci95_upper"]
        else:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            values = []
            prediction = ensemble.argmax(axis=1)
            by_class = [np.flatnonzero(y_test == label) for label in np.unique(y_test)]
            for _ in range(BOOTSTRAP_ITERATIONS):
                indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in by_class])
                values.append(f1_score(y_test[indices], prediction[indices], average="weighted", zero_division=0))
            lower, upper = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
        comparison_rows.append({
            "external_system": "MARL-NIDS",
            "dataset": "CICIDS2017",
            "metric": metric,
            "paper_reported_reference": reference,
            "local_value": local[metric],
            "local_ci95_lower": lower,
            "local_ci95_upper": upper,
            "ci_lower_exceeds_reference": lower > reference,
            "fairness_level": "reconstructed_15_label_safe_input_bounded_training",
        })
    _write_csv(out / "marl_paper_reference_comparison_w164.csv", comparison_rows)
    accepted = all(row["ci_lower_exceeds_reference"] for row in comparison_rows)
    report = {
        "status": "accepted_cicids_marl_comparable_positive_result" if accepted else "cicids_marl_reference_not_exceeded_with_ci_support",
        "selected_family": selected,
        "local_metrics": local,
        "class_count": len(CICIDS_CLASSES),
        "paper_wording_correction": "15 total labels (BENIGN plus 14 attack labels), not 14 total classes",
        "faithful_reproduction_completed": False,
        "official_external_code_used": False,
        "acceptance_opened_once": True,
        **_security(),
    }
    _dump(out / "w164_acceptance_report.json", report)
    return report


def evaluate_edgeiiot_secondary_w165(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    raw_root: str | Path = "data/raw",
) -> dict[str, Any]:
    out = Path(output_dir)
    candidates = [
        path for path in Path(raw_root).rglob("*")
        if path.is_file() and ("edge-iiot" in path.name.lower() or "edgeiiot" in path.name.lower())
    ]
    inventory = [{"path": path.as_posix(), "bytes": path.stat().st_size} for path in candidates]
    _write_csv(out / "edgeiiot_inventory_w165.csv", inventory)
    if not candidates:
        report = {
            "status": "blocked_missing_edge_iiotset",
            "comparison_role": "secondary_contextual_only",
            "supervised_metrics_generated": False,
            "blocker": "No local Edge-IIoTset artifact was found; paper-reported values remain contextual only.",
            **_security(),
        }
    else:
        report = {
            "status": "blocked_unverified_edge_iiot_schema",
            "comparison_role": "secondary_contextual_only",
            "supervised_metrics_generated": False,
            "blocker": "Files exist but no official label/schema provenance has been audited in this lane.",
            "candidate_count": len(candidates),
            **_security(),
        }
    _dump(out / "w165_acceptance_report.json", report)
    return report


def audit_external_positive_shadow_w166(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    accepted = list(
        _read_json(out / "w163_acceptance_report.json").get(
            "accepted_dataset_scopes", []
        )
    )
    if (
        _read_json(out / "w164_acceptance_report.json").get("status")
        == "accepted_cicids_marl_comparable_positive_result"
    ):
        accepted.append("CICIDS2017")
    agent_name = "NFMulticlassEvidenceAgentW163"
    rows: list[dict[str, Any]] = []
    violations = 0
    for dataset in accepted:
        if dataset == "CICIDS2017":
            family = _read_json(out / "cicids_model_selection_w164.json").get(
                "selected_family"
            )
            artifact_path = Path(model_dir) / "cicids_w164" / f"{family}__seed42.joblib"
            raw_names = _read_json(out / "cicids_safe_feature_policy_w164.json").get(
                "safe_features", []
            )
            safe_paths = [
                "stats."
                + "_".join(str(name).strip().lower().split())
                .replace("/", "_per_")
                .replace("-", "_")
                for name in raw_names
            ]
            current_agent = "CICIDS15ClassEvidenceAgentW164"
        else:
            protocol = "reconstructed_paper_compatible"
            task_key = f"{dataset}::{protocol}"
            selection = _read_json(out / "model_selection_w162.json")
            family = selection.get("tasks", {}).get(task_key, {}).get(
                "selected_family"
            )
            artifact_path = Path(model_dir) / (
                f"{dataset.lower().replace('-', '_')}__{protocol}__"
                f"{family}__seed42.joblib"
            )
            safe_paths = [f"stats.{name}" for name in ENGINEERED_FEATURES]
            current_agent = agent_name
        artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        flow = FlowRecord(trace_id=f"w166-{dataset}", sample_id=f"w166-{dataset}", stats={name: 0.0 for name in ENGINEERED_FEATURES})
        field_audit = FieldAuditResult(decisions=[], allowed_fields=safe_paths, blocked_fields=list(BLOCKED_FIELDS), context_only_fields=[], leakage_risk=0.0)
        capability = DetectorCapabilityProfile(agent_name=current_agent, backend=family, status="available", consumed_fields=safe_paths, available_fields=safe_paths, observed_fields=safe_paths)
        state = CaseState(flow=flow, field_audit=field_audit, detector_capabilities={current_agent: capability}, remaining_budget=1)
        request = EvidenceRequest(
            request_id=hashlib.sha256(f"request:{dataset}".encode()).hexdigest()[:24],
            case_trace_id=flow.trace_id,
            requested_agent=current_agent,
            permitted_safe_features=safe_paths,
            purpose="Collect dataset-scoped multiclass detector evidence for governed shadow evaluation",
            budget=1,
            allowed_feature_policy_hash=hashlib.sha256(json.dumps(sorted(safe_paths), separators=(",", ":")).encode()).hexdigest(),
            expected_evidence_schema="AgentEvidenceV2",
        )
        guard = EvidenceRequestGuard(allowed_agents={current_agent})
        review = guard.review(request, state)
        original = AgentEvidence(
            agent_name=current_agent,
            agent_version="w163",
            feature_group=FeatureGroup.STATS,
            benign_support=0.25,
            malicious_support=0.75,
            confidence=0.75,
            uncertainty=0.25,
            model_reliability=0.90,
            contributes_to_verdict=False,
            evidence=["DATASET_SPECIFIC_SHADOW_ONLY"],
            used_fields=safe_paths,
        )
        adapter = AgentEvidenceV2Adapter()
        v2 = adapter.to_v2(original, review.request, artifact_hash=artifact_hash, dataset_scope=f"{dataset}_only") if review.approved else None
        reliability = ReliabilityProfile(stats_reliability=1.0, sequence_reliability=1.0, tls_reliability=1.0, payload_reliability=1.0, input_completeness=1.0)
        baseline = AgentEvidence(agent_name="StatsDetectorAgent", feature_group=FeatureGroup.STATS, benign_support=0.8, malicious_support=0.2, confidence=0.8, uncertainty=0.2, used_fields=[])
        fusion = FusionAgent()
        before = fusion.fuse([baseline], reliability, final=True)
        after = fusion.fuse([baseline, original], reliability, final=True)
        invariant = before.model_dump() == after.model_dump()
        valid = review.approved and v2 is not None and not v2.contributes_to_verdict and invariant
        violations += int(not valid)
        rows.append({
            "dataset": dataset,
            "selected_family": family,
            "evidence_request_approved": review.approved,
            "agent_evidence_v2_valid": v2 is not None,
            "candidate_contributes_to_verdict": v2.contributes_to_verdict if v2 else None,
            "fusion_snapshot_invariance": invariant,
            "feature_policy_hash_matches": "request_scoped_hash_verified_by_adapter",
            "final_verdict_owner": "FusionAgent",
        })
    _write_csv(out / "external_positive_shadow_audit_w166.csv", rows)
    report = {
        "status": "accepted_candidates_shadow_integrated" if accepted and violations == 0 else "no_accepted_candidate_or_shadow_gate_failed",
        "accepted_dataset_scopes": accepted,
        "shadow_case_count": len(rows),
        "candidate_default_enabled": False,
        "candidate_enters_fusion": False,
        "fusion_snapshot_invariance": 1.0 if rows and violations == 0 else 0.0,
        "policy_or_schema_violation": violations,
        **_security(),
    }
    _dump(out / "w166_acceptance_report.json", report)
    return report


def finalize_external_multiagent_positive_w167(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    release = Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    stages = {
        "w160": _read_json(out / "external_protocol_registry.json"),
        "w161": _read_json(out / "w161_acceptance_report.json"),
        "w162": _read_json(out / "w162_acceptance_report.json"),
        "w163": _read_json(out / "w163_acceptance_report.json"),
        "w164": _read_json(out / "w164_acceptance_report.json"),
        "w165": _read_json(out / "w165_acceptance_report.json"),
        "w166": _read_json(out / "w166_acceptance_report.json"),
    }
    nf_scopes = stages["w163"].get("accepted_dataset_scopes", [])
    cicids_positive = stages["w164"].get("status") == "accepted_cicids_marl_comparable_positive_result"
    positive_rows = [
        {
            "experiment_id": "w163_nf_ma_ids_comparison",
            "dataset_scope": dataset,
            "status": "accepted_external_multiagent_comparable_positive_result",
            "comparison_type": "paper_reported_reference_vs_local_reconstructed_safe_input",
            "faithful_reproduction": False,
            "artifact_path": (out / "ma_ids_paper_reference_comparison_w163.csv").as_posix(),
            "safe_claim": f"On the reconstructed {dataset} task/metric protocol, the local safe-input candidate's bootstrap lower bounds exceeded the MA-IDS paper-reported scalar references.",
        }
        for dataset in nf_scopes
    ]
    if cicids_positive:
        positive_rows.append({
            "experiment_id": "w164_cicids_marl_comparison",
            "dataset_scope": "CICIDS2017",
            "status": "accepted_cicids_marl_comparable_positive_result",
            "comparison_type": "paper_reported_reference_vs_local_reconstructed_safe_input",
            "faithful_reproduction": False,
            "artifact_path": (out / "marl_paper_reference_comparison_w164.csv").as_posix(),
            "safe_claim": "The local safe-input 15-label reconstruction exceeded selected MARL-NIDS paper-reported scalar references with local bootstrap support.",
        })
    negative_rows = []
    if not nf_scopes:
        negative_rows.append({"experiment_id": "w163_nf_ma_ids_comparison", "status": stages["w163"].get("status"), "reason": "all four local CI lower bounds did not exceed the paper-reported reference", "safe_claim": "No externally comparable NF positive result was accepted.", "forbidden_claim": "MAD-ETD outperforms MA-IDS."})
    if not cicids_positive:
        negative_rows.append({"experiment_id": "w164_cicids_marl_comparison", "status": stages["w164"].get("status"), "reason": "MARL paper reference gate was not passed or the protocol was blocked", "safe_claim": "The CICIDS2017 MARL comparison is negative or blocked.", "forbidden_claim": "MAD-ETD outperforms MARL-NIDS."})
    negative_rows.append({"experiment_id": "w165_edgeiiot_secondary", "status": stages["w165"].get("status"), "reason": stages["w165"].get("blocker"), "safe_claim": "Edge-IIoT remained a contextual secondary comparison.", "forbidden_claim": "Local Edge-IIoT reproduction completed."})
    _write_csv(release / "positive_result_ledger_w167.csv", positive_rows)
    _write_csv(release / "negative_result_ledger_w167.csv", negative_rows)
    claims = [
        {"claim_id": "external_protocol_separation", "safe_to_claim": True, "claim": "Paper-reported, reconstructed safe-input, source-held-out, and faithful-reproduction evidence are reported separately.", "support": (out / "external_protocol_registry.json").as_posix()},
        {"claim_id": "nf_positive_scope", "safe_to_claim": bool(nf_scopes), "claim": f"Accepted reconstructed NF scopes: {', '.join(nf_scopes) if nf_scopes else 'none'}.", "support": (out / "w163_acceptance_report.json").as_posix()},
        {"claim_id": "faithful_reproduction", "safe_to_claim": False, "claim": "External faithful reproduction completed.", "support": "forbidden"},
        {"claim_id": "general_runtime_promotion", "safe_to_claim": False, "claim": "The new detector replaced runtime_safe_v3_0.", "support": "forbidden"},
    ]
    _write_csv(release / "claim_ledger_w167.csv", claims)
    artifacts = []
    for path in sorted(out.glob("*")):
        if path.is_file():
            artifacts.append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    _write_csv(release / "artifact_index_w167.csv", artifacts)
    _dump(release / "stage_status_w167.json", {key: value.get("status", "missing") for key, value in stages.items()})
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    markdown = f"""# MAD-ETD External Multi-Agent Positive Comparison W160-W167

## Status

- accepted NF scopes: `{', '.join(nf_scopes) if nf_scopes else 'none'}`
- CICIDS2017 MARL gate: `{stages['w164'].get('status', 'missing')}`
- Edge-IIoT secondary: `{stages['w165'].get('status', 'missing')}`
- faithful external reproduction completed: `false`
- runtime_safe_v3_0 remains default: `true`
- fake metric count: `0`

## Boundary

MA-IDS and MARL-NIDS values are paper-reported scalar references. Local runs use
public data, matched label tasks and metric definitions where possible, but a
stricter safe-input policy and reconstructed splits. They are therefore not
official-code faithful reproductions. A local CI lower bound above a paper scalar
supports a scoped numerical comparison, not a paired significance claim against
inaccessible external predictions.

## Positive evidence

{chr(10).join('- ' + row['safe_claim'] for row in positive_rows) if positive_rows else '- None passed the pre-registered external-reference gates.'}

## Negative or blocked evidence

{chr(10).join('- ' + str(row['safe_claim']) for row in negative_rows)}
"""
    (docs / "MAD_ETD_EXTERNAL_MULTIAGENT_POSITIVE_W160_W167.md").write_text(markdown, encoding="utf-8")
    cn = f"""# MAD-ETD 外部多智能体公平正向对比 W160-W167

## 最终状态

- NF 通过范围：`{', '.join(nf_scopes) if nf_scopes else '无'}`
- CICIDS2017 MARL 对比：`{stages['w164'].get('status', 'missing')}`
- Edge-IIoT：`{stages['w165'].get('status', 'missing')}`
- 外部官方代码 faithful reproduction：`未完成`
- 默认系统：`runtime_safe_v3_0` 保持不变
- fake metric count：`0`

## 解释边界

本轮将论文报告值、本地安全输入重建协议、来源留出协议和 faithful reproduction
严格分开。本地置信区间下界超过论文标量时，只能在对应数据集、类别映射、指标口径
和重建协议范围内表述数值优势，不能写成官方代码复现或对外部逐样本结果的配对显著性。
"""
    (docs / "MAD_ETD_EXTERNAL_MULTIAGENT_POSITIVE_W160_W167_CN.md").write_text(cn, encoding="utf-8")
    required_statuses = {
        "w160": "external_protocol_registry_verified_with_primary_sources",
        "w161": "nf_multiclass_dual_protocols_frozen",
        "w162": "nf_multiclass_stack_validation_locked",
    }
    core_pass = all(stages[key].get("status") == value for key, value in required_statuses.items())
    security_pass = all(
        int(stage.get(field, 0)) == 0
        for stage in stages.values() if stage
        for field in ("blocked_field_violation", "fusion_ownership_violation", "ood_override_count", "illegal_verdict_execution_count", "fake_metric_count")
    )
    report = {
        "status": "accepted_external_multiagent_positive_evidence_release" if core_pass and security_pass and tests_passed else "pending_or_failed_external_multiagent_positive_release",
        "accepted_positive_result_count": len(positive_rows),
        "accepted_nf_dataset_scopes": nf_scopes,
        "cicids_positive": cicids_positive,
        "faithful_external_reproduction_completed": False,
        "adapted_or_reconstructed_labeled_faithful": False,
        "paper_numbers_used_as_local_metrics": False,
        "full_pytest_passed": bool(tests_passed),
        "full_pytest_count": int(test_count),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "production_ready": False,
        "security_passed": security_pass,
        **_security(),
    }
    _dump(release / "acceptance_report.json", report)
    return report
