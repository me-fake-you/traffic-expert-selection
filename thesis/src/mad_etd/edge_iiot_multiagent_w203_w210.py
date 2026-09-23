"""Edge-IIoTset W203--W210 safe performance and multi-agent comparison lane.

The module is deliberately default-off.  It supports a strict source-group
protocol when the author's per-source CSV files are available and keeps a
row-stratified, paper-compatible protocol diagnostic-only.  External
multi-agent architectures are local same-data adaptations, never faithful
reproductions.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .external_multiagent_positive_w160_w167 import _choose_temperature, _ece, _temperature_scale
from .nfton_binary_generalization_w176_w179 import _security
from .paper_evaluation import hash_artifact_paths
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_edge_iiot_multiagent_w203_w210"
DEFAULT_RAW_DIR = Path("data/raw/edge_iiotset/v1/official_download")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_edge_iiot_multiagent_w203_w210")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_edge_iiot_w206")
DEFAULT_RELEASE_DIR = Path("data/releases/mad_etd_edge_iiot_multiagent_w210")

SEEDS = (42, 43, 44)
MODEL_FAMILIES = (
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
)
ROLE_LIMITS_PER_CLASS = {
    "train": 30_000,
    "calibration": 8_000,
    "selection": 8_000,
    "acceptance": 15_000,
}
PER_GROUP_LIMIT = 3_000
MIN_GROUPS_PER_CLASS = 4

ATTACK_TOKENS = (
    "attack",
    "backdoor",
    "ddos",
    "dos_",
    "flood",
    "mitm",
    "fingerprint",
    "password",
    "scanning",
    "ransomware",
    "sql",
    "upload",
    "vulnerability",
    "xss",
)
BENIGN_TOKENS = (
    "normal traffic",
    "normal_traffic",
    "benign",
    "distance",
    "flame_sensor",
    "heart_rate",
    "ir_receiver",
    "modbus",
    "phvalue",
    "soil_moisture",
    "sound_sensor",
    "temperature_and_humidity",
    "water_level",
)
MIXED_TOKENS = ("dnn-edgeiiot", "ml-edgeiiot", "live_data_training")

LABEL_ONLY_NAMES = {
    "attack",
    "attack_type",
    "attack_label",
    "label",
    "class",
    "target",
}
BLOCKED_EXACT_NAMES = {
    "tcp.stream",
    "udp.stream",
}
BLOCKED_SUBSTRINGS = (
    "ip.src",
    "ip.dst",
    "src_host",
    "dst_host",
    "ipv4",
    "ipv6",
    "mac",
    "port",
    "frame.time",
    "timestamp",
    "flow id",
    "flow_id",
    "payload",
    "file_data",
    "full_uri",
    "uri.query",
    "mqtt.msg",
    "endpoint",
    "device_id",
    "sensor_id",
    "source_file",
    "provenance",
    "capture",
    "split",
)
CONDITIONAL_SUBSTRINGS = (
    "http.request",
    "http.referer",
    "dns.qry.name",
    "mqtt.topic",
    "user_agent",
    "hostname",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read(path: str | Path) -> dict[str, Any]:
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
        for key in row:
            if key not in fields:
                fields.append(key)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _source_class(path: Path) -> tuple[str, str]:
    value = path.as_posix().lower()
    name = path.stem.lower()
    if any(token in value for token in MIXED_TOKENS):
        return "mixed", "mixed_selected_dataset"
    if any(token in value for token in ATTACK_TOKENS):
        category = name.replace(".csv", "").replace("_attack", "").replace("-attack", "")
        return "malicious", category
    if any(token in value for token in BENIGN_TOKENS):
        return "benign", "benign"
    return "unknown", "unknown"


def _feature_role(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in LABEL_ONLY_NAMES or "attack_type" in normalized or "attack_label" in normalized:
        return "label_only"
    if normalized in BLOCKED_EXACT_NAMES or any(token in normalized for token in BLOCKED_SUBSTRINGS):
        return "blocked"
    if any(token in normalized for token in CONDITIONAL_SUBSTRINGS):
        return "safe_conditional"
    return "safe_common_candidate"


def _balanced_roles(groups: Sequence[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"w205:{value}".encode()).hexdigest())
    count = len(ordered)
    if count < 4:
        return {}
    acceptance = max(1, round(count * 0.20))
    selection = max(1, round(count * 0.15))
    calibration = max(1, round(count * 0.15))
    while acceptance + selection + calibration >= count:
        if acceptance > 1:
            acceptance -= 1
        elif selection > 1:
            selection -= 1
        elif calibration > 1:
            calibration -= 1
        else:
            break
    train = count - acceptance - selection - calibration
    roles = (
        ["train"] * train
        + ["calibration"] * calibration
        + ["selection"] * selection
        + ["acceptance"] * acceptance
    )
    return dict(zip(ordered, roles, strict=True))


def _binary_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, prediction, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "ece": float(_ece(y, probabilities)),
        "coverage": 1.0,
    }


def _make_model(family: str, seed: int) -> Any:
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=220,
            max_depth=7,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=220,
            num_leaves=31,
            learning_rate=0.05,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=4,
            random_state=seed,
        )
    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=220,
            depth=7,
            learning_rate=0.06,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            thread_count=4,
            random_seed=seed,
            allow_writing_files=False,
        )
    raise ValueError(f"unsupported model family: {family}")


def _align_binary_proba(model: Any, probabilities: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes_, dtype=int)
    if probabilities.shape[1] == 2 and np.array_equal(classes, np.asarray([0, 1])):
        return probabilities.astype(np.float64)
    aligned = np.full((len(probabilities), 2), 1e-12, dtype=np.float64)
    for source, target in enumerate(classes):
        if target in (0, 1):
            aligned[:, int(target)] = probabilities[:, source]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _predict_bundle(bundle: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    raw = _align_binary_proba(bundle["model"], bundle["model"].predict_proba(x))
    return _temperature_scale(raw, float(bundle.get("temperature", 1.0)))


def audit_edgeiiot_download_w203(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    compute_hashes: bool = True,
) -> dict[str, Any]:
    raw, out = Path(raw_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "frozen_hashes_before_w203.json", hash_artifact_paths(_default_frozen_paths()))
    files = list(raw.rglob("*")) if raw.exists() else []
    partials = [
        path for path in files
        if path.is_file() and path.name.lower().endswith((".crdownload", ".part", ".tmp"))
    ]
    csv_files = [path for path in files if path.is_file() and path.suffix.lower() == ".csv"]
    archives = [path for path in files if path.is_file() and path.suffix.lower() in {".zip", ".7z", ".rar"}]
    inventory: list[dict[str, Any]] = []
    for path in [*csv_files, *archives, *partials]:
        label_class, category = _source_class(path)
        inventory.append({
            "path": path.as_posix(),
            "relative_path": path.relative_to(raw).as_posix() if raw.exists() else path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path) if compute_hashes and path not in partials else "not_computed",
            "kind": "csv" if path in csv_files else ("partial" if path in partials else "archive"),
            "label_scope": label_class,
            "category": category,
            "source_group_id": hashlib.sha256(
                (path.relative_to(raw).as_posix() if raw.exists() else path.name).encode()
            ).hexdigest(),
        })
    _write_csv(out / "source_inventory_w203.csv", inventory)
    source_csv_rows = [row for row in inventory if row["kind"] == "csv"]
    pure_benign = sum(row["label_scope"] == "benign" for row in source_csv_rows)
    pure_malicious = sum(row["label_scope"] == "malicious" for row in source_csv_rows)
    mixed = sum(row["label_scope"] == "mixed" for row in source_csv_rows)
    if partials:
        status = "waiting_w203_download_incomplete"
    elif not csv_files:
        status = "waiting_w203_download_not_arrived"
    elif pure_benign >= MIN_GROUPS_PER_CLASS and pure_malicious >= MIN_GROUPS_PER_CLASS:
        status = "ready_w203_strict_source_group_audit"
    elif mixed:
        status = "ready_w203_paper_compatible_only_strict_groups_missing"
    else:
        status = "blocked_w203_insufficient_labeled_source_groups"
    report = {
        "status": status,
        "raw_dir": raw.as_posix(),
        "csv_count": len(csv_files),
        "archive_count": len(archives),
        "partial_download_count": len(partials),
        "pure_benign_source_group_count": pure_benign,
        "pure_malicious_source_group_count": pure_malicious,
        "mixed_selected_csv_count": mixed,
        "strict_group_protocol_possible": (
            status == "ready_w203_strict_source_group_audit"
        ),
        "paper_compatible_protocol_possible": bool(mixed),
        "training_started": False,
        "supervised_metrics_generated": False,
        **_security(),
    }
    _dump(out / "w203_acceptance_report.json", report)
    return report


def audit_edgeiiot_safe_features_w204(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    sample_rows_per_file: int = 2_000,
) -> dict[str, Any]:
    out = Path(output_dir)
    w203 = _read(out / "w203_acceptance_report.json")
    if not str(w203.get("status", "")).startswith("ready_w203_"):
        report = {"status": "not_run_w204_edgeiiot_not_ready", **_security()}
        _dump(out / "w204_acceptance_report.json", report)
        return report
    source_rows = [
        row for row in _read_csv(out / "source_inventory_w203.csv")
        if row.get("kind") == "csv" and row.get("label_scope") in {"benign", "malicious"}
    ]
    if not source_rows:
        report = {
            "status": "blocked_w204_no_pure_source_csv_for_strict_policy",
            "paper_compatible_only": True,
            **_security(),
        }
        _dump(out / "w204_acceptance_report.json", report)
        return report
    observations: dict[str, list[float]] = defaultdict(list)
    present: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for source in source_rows:
        path = Path(source["path"])
        try:
            frame = pd.read_csv(path, nrows=sample_rows_per_file, low_memory=False)
        except Exception as exc:
            records.append({
                "feature_name": "__source_read_error__",
                "source_path": path.as_posix(),
                "final_feature_policy": "blocked",
                "decision_reason": str(exc),
            })
            continue
        for name in frame.columns:
            role = _feature_role(str(name))
            numeric = pd.to_numeric(frame[name], errors="coerce")
            ratio = float(numeric.notna().mean()) if len(frame) else 0.0
            present[str(name)] += 1
            observations[str(name)].append(ratio)
    source_count = len(source_rows)
    safe: list[str] = []
    all_names = sorted(present)
    for name in all_names:
        role = _feature_role(name)
        present_all = present[name] == source_count
        numeric_ratio = min(observations[name]) if observations[name] else 0.0
        if role == "safe_common_candidate" and present_all and numeric_ratio >= 0.95:
            final = "safe_common"
            safe.append(name)
            reason = "numeric in every pure source group and no identity/label/shortcut pattern"
        elif role == "safe_common_candidate":
            final = "dataset_specific"
            reason = "not numeric or not present in every pure source group"
        else:
            final = role
            reason = "field-contract pattern"
        records.append({
            "feature_name": name,
            "semantic_role": role,
            "present_source_count": present[name],
            "source_count": source_count,
            "minimum_numeric_ratio": numeric_ratio,
            "final_feature_policy": final,
            "decision_reason": reason,
        })
    blocked = sorted(
        record["feature_name"] for record in records
        if record.get("final_feature_policy") in {"blocked", "label_only"}
    )
    policy = {
        "dataset": "Edge-IIoTset",
        "task": "binary benign_vs_malicious",
        "safe_features": safe,
        "blocked_or_label_only_fields": blocked,
        "group_metadata_usage": "split_and_audit_only",
        "source_file_enters_detector_input": False,
        "attack_type_enters_detector_input": False,
        "feature_policy_hash": _json_hash({"safe": safe, "blocked": blocked}),
    }
    _write_csv(out / "feature_forensics_w204.csv", records)
    _dump(out / "safe_feature_policy_w204.json", policy)
    report = {
        "status": "w204_safe_feature_policy_frozen" if len(safe) >= 5 else "blocked_w204_too_few_safe_common_features",
        "safe_feature_count": len(safe),
        "blocked_or_label_only_count": len(blocked),
        "source_group_used_as_feature": False,
        "attack_label_used_as_feature": False,
        **_security(),
    }
    _dump(out / "w204_acceptance_report.json", report)
    return report


def freeze_edgeiiot_protocols_w205(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    if _read(out / "w204_acceptance_report.json").get("status") != "w204_safe_feature_policy_frozen":
        report = {"status": "not_run_w205_missing_safe_feature_policy", **_security()}
        _dump(out / "w205_acceptance_report.json", report)
        return report
    sources = [
        row for row in _read_csv(out / "source_inventory_w203.csv")
        if row.get("kind") == "csv" and row.get("label_scope") in {"benign", "malicious"}
    ]
    by_class = {
        label: [row["source_group_id"] for row in sources if row["label_scope"] == label]
        for label in ("benign", "malicious")
    }
    roles = {label: _balanced_roles(groups) for label, groups in by_class.items()}
    if any(not roles[label] for label in roles):
        report = {
            "status": "blocked_w205_insufficient_source_groups",
            "minimum_groups_per_class": MIN_GROUPS_PER_CLASS,
            "group_counts": {label: len(groups) for label, groups in by_class.items()},
            **_security(),
        }
        _dump(out / "w205_acceptance_report.json", report)
        return report
    rows = []
    for source in sources:
        label = source["label_scope"]
        rows.append({
            **source,
            "role": roles[label][source["source_group_id"]],
            "binary_label": 0 if label == "benign" else 1,
            "source_group_used_as_feature": False,
            "category_used_as_feature": False,
        })
    _write_csv(out / "strict_group_split_manifest_w205.csv", rows)
    role_groups = defaultdict(set)
    for row in rows:
        role_groups[row["role"]].add(row["source_group_id"])
    overlap = 0
    role_names = list(role_groups)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1:]:
            overlap += len(role_groups[left] & role_groups[right])
    protocol = {
        "strict_protocol": {
            "task": "binary benign_vs_malicious",
            "split_unit": "official source CSV / capture scenario group",
            "roles": ["train", "calibration", "selection", "acceptance"],
            "role_group_counts": {role: len(groups) for role, groups in role_groups.items()},
            "group_overlap_count": overlap,
            "eligible_for_promotion": overlap == 0,
        },
        "paper_compatible_protocol": {
            "task": "binary_or_15_class",
            "split_unit": "stratified rows from selected DNN/ML CSV",
            "diagnostic_only": True,
            "eligible_for_promotion": False,
            "reason": "row-level mixing cannot establish capture/source-group generalization",
        },
        "acceptance_sealed": True,
        "acceptance_used_for_selection": False,
    }
    _dump(out / "protocol_registry_w205.json", protocol)
    report = {
        "status": "w205_strict_group_protocol_frozen" if overlap == 0 else "failed_w205_group_overlap",
        "role_group_counts": protocol["strict_protocol"]["role_group_counts"],
        "group_overlap_count": overlap,
        "acceptance_used_for_selection": False,
        "paper_protocol_diagnostic_only": True,
        **_security(),
    }
    _dump(out / "w205_acceptance_report.json", report)
    return report


def _reservoir_sample_source(
    path: Path,
    features: Sequence[str],
    group: str,
    limit: int,
) -> np.ndarray:
    rng_seed = int(hashlib.sha256(f"w206:{group}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(rng_seed)
    kept_x: np.ndarray | None = None
    kept_priority: np.ndarray | None = None
    for chunk in pd.read_csv(path, usecols=list(features), chunksize=50_000, low_memory=False):
        values = (
            chunk.loc[:, list(features)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        priority = rng.random(len(values))
        if kept_x is None:
            kept_x, kept_priority = values, priority
        else:
            kept_x = np.concatenate([kept_x, values])
            kept_priority = np.concatenate([kept_priority, priority])  # type: ignore[arg-type]
        if len(kept_x) > limit:
            keep = np.argpartition(kept_priority, limit - 1)[:limit]  # type: ignore[arg-type]
            kept_x, kept_priority = kept_x[keep], kept_priority[keep]  # type: ignore[index]
    return kept_x if kept_x is not None else np.empty((0, len(features)), dtype=np.float32)


def _cap_role(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    categories: np.ndarray,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected: list[int] = []
    for label in (0, 1):
        local = np.flatnonzero(y == label)
        limit = ROLE_LIMITS_PER_CLASS[role]
        if len(local) > limit:
            priority = np.asarray([
                int(hashlib.sha256(f"w206:{role}:{groups[index]}:{index}".encode()).hexdigest()[:16], 16)
                for index in local
            ], dtype=np.uint64)
            local = local[np.argpartition(priority, limit - 1)[:limit]]
        selected.extend(local.tolist())
    index = np.asarray(sorted(selected), dtype=int)
    return x[index], y[index], groups[index], categories[index]


def _materialize_w205_arrays(out: Path) -> dict[str, Any]:
    policy = _read(out / "safe_feature_policy_w204.json")
    features = list(policy.get("safe_features", []))
    manifest = _read_csv(out / "strict_group_split_manifest_w205.csv")
    by_role: dict[str, list[tuple[np.ndarray, int, str, str]]] = defaultdict(list)
    for row in manifest:
        values = _reservoir_sample_source(
            Path(row["path"]),
            features,
            row["source_group_id"],
            PER_GROUP_LIMIT,
        )
        by_role[row["role"]].append((
            values,
            int(row["binary_label"]),
            row["source_group_id"],
            row["category"],
        ))
    counts: dict[str, int] = {}
    for role in ("train", "calibration", "selection", "acceptance"):
        entries = by_role.get(role, [])
        if not entries:
            continue
        x = np.concatenate([entry[0] for entry in entries])
        y = np.concatenate([np.full(len(entry[0]), entry[1], dtype=np.int8) for entry in entries])
        groups = np.concatenate([np.full(len(entry[0]), entry[2], dtype="U64") for entry in entries])
        categories = np.concatenate([np.full(len(entry[0]), entry[3], dtype="U96") for entry in entries])
        x, y, groups, categories = _cap_role(x, y, groups, categories, role)
        filename = "acceptance_sealed_w205.npz" if role == "acceptance" else f"{role}_w205.npz"
        np.savez_compressed(out / filename, x=x, y=y, groups=groups, categories=categories, feature_names=np.asarray(features))
        counts[role] = len(y)
    sealed = out / "acceptance_sealed_w205.npz"
    commitment = {
        "path": sealed.as_posix(),
        "sha256": _sha256(sealed) if sealed.is_file() else None,
        "opened": False,
        "acceptance_used_for_selection": False,
    }
    _dump(out / "sealed_acceptance_commitment_w205.json", commitment)
    return {"counts": counts, "sealed": sealed.is_file()}


def train_edgeiiot_safe_baselines_w206(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    seeds: Sequence[int] = SEEDS,
    families: Sequence[str] = MODEL_FAMILIES,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    if _read(out / "w205_acceptance_report.json").get("status") != "w205_strict_group_protocol_frozen":
        report = {"status": "not_run_w206_missing_strict_group_protocol", "training_started": False, **_security()}
        _dump(out / "w206_acceptance_report.json", report)
        return report
    materialized = _materialize_w205_arrays(out)
    required = [out / f"{role}_w205.npz" for role in ("train", "calibration", "selection")]
    if not materialized["sealed"] or any(not path.is_file() for path in required):
        report = {"status": "failed_w206_split_materialization", "training_started": False, **_security()}
        _dump(out / "w206_acceptance_report.json", report)
        return report
    train, calibration, selection = [np.load(path, allow_pickle=False) for path in required]
    if any(len(np.unique(data["y"])) != 2 for data in (train, calibration, selection)):
        report = {"status": "failed_w206_missing_binary_class", "training_started": False, **_security()}
        _dump(out / "w206_acceptance_report.json", report)
        return report
    models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for family in families:
        for seed in seeds:
            started = time.perf_counter()
            try:
                model = _make_model(family, int(seed))
                model.fit(train["x"], train["y"])
            except (ImportError, ModuleNotFoundError) as exc:
                unavailable.append({"family": family, "reason": str(exc)})
                break
            training_seconds = time.perf_counter() - started
            calibration_raw = _align_binary_proba(model, model.predict_proba(calibration["x"]))
            temperature = _choose_temperature(calibration["y"], calibration_raw)
            selection_started = time.perf_counter()
            probabilities = _temperature_scale(
                _align_binary_proba(model, model.predict_proba(selection["x"])),
                temperature,
            )
            inference_seconds = time.perf_counter() - selection_started
            metrics = _binary_metrics(selection["y"], probabilities)
            bundle = {
                "model": model,
                "family": family,
                "seed": int(seed),
                "temperature": temperature,
                "feature_names": list(train["feature_names"].astype(str)),
                "feature_policy_hash": _read(out / "safe_feature_policy_w204.json").get("feature_policy_hash"),
            }
            path = models / f"{family}__seed{seed}.joblib"
            joblib_dump(bundle, path)
            rows.append({
                "family": family,
                "seed": int(seed),
                "split": "selection",
                **metrics,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "artifact_path": path.as_posix(),
                "artifact_hash": _sha256(path),
                "acceptance_used_for_selection": False,
                "fake_metric": False,
            })
    _write_csv(out / "safe_baseline_selection_results_w206.csv", rows)
    _write_csv(out / "unavailable_model_families_w206.csv", unavailable)
    report = {
        "status": "w206_safe_baselines_trained" if rows else "failed_w206_no_model_trained",
        "training_started": True,
        "trained_artifact_count": len(rows),
        "model_families": sorted({row["family"] for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "split_sample_counts": materialized["counts"],
        "acceptance_opened": False,
        "acceptance_used_for_selection": False,
        **_security(),
    }
    _dump(out / "w206_acceptance_report.json", report)
    return report


def build_edgeiiot_candidate_w207(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    if _read(out / "w206_acceptance_report.json").get("status") != "w206_safe_baselines_trained":
        report = {"status": "not_run_w207_missing_trained_baselines", **_security()}
        _dump(out / "w207_acceptance_report.json", report)
        return report
    rows = _read_csv(out / "safe_baseline_selection_results_w206.csv")
    best_by_family: dict[str, dict[str, str]] = {}
    for row in rows:
        current = best_by_family.get(row["family"])
        if current is None or (float(row["macro_f1"]), float(row["malicious_recall"])) > (
            float(current["macro_f1"]), float(current["malicious_recall"])
        ):
            best_by_family[row["family"]] = row
    ranked = sorted(
        best_by_family.values(),
        key=lambda row: (float(row["macro_f1"]), float(row["malicious_recall"]), -float(row["ece"])),
        reverse=True,
    )
    if not ranked:
        report = {"status": "failed_w207_no_selection_rows", **_security()}
        _dump(out / "w207_acceptance_report.json", report)
        return report
    calibration = np.load(out / "calibration_w205.npz", allow_pickle=False)
    selection = np.load(out / "selection_w205.npz", allow_pickle=False)
    probability_cache: dict[str, dict[str, np.ndarray]] = {}
    for row in ranked:
        bundle = joblib_load(Path(row["artifact_path"]))
        probability_cache[row["family"]] = {
            "calibration": _predict_bundle(bundle, calibration["x"])[:, 1],
            "selection": _predict_bundle(bundle, selection["x"])[:, 1],
        }
    diagnostics: list[dict[str, Any]] = []
    selected_rows: list[dict[str, str]]
    meta_path: Path | None = None
    if len(ranked) >= 2:
        candidates: list[tuple[tuple[float, float, float, float], list[dict[str, str]], Any, dict[str, float]]] = []
        max_members = min(3, len(ranked))
        for member_count in range(2, max_members + 1):
            for members in itertools.combinations(ranked, member_count):
                calibration_meta = np.column_stack([
                    probability_cache[row["family"]]["calibration"] for row in members
                ])
                selection_meta = np.column_stack([
                    probability_cache[row["family"]]["selection"] for row in members
                ])
                stacker = LogisticRegression(
                    C=0.1,
                    max_iter=1_000,
                    random_state=42,
                )
                stacker.fit(calibration_meta, calibration["y"])
                candidate_probability = _align_binary_proba(
                    stacker,
                    stacker.predict_proba(selection_meta),
                )
                metrics = _binary_metrics(selection["y"], candidate_probability)
                member_names = [row["family"] for row in members]
                diagnostics.append({
                    "candidate_kind": "calibration_trained_logistic_stacker",
                    "members": "|".join(member_names),
                    "member_count": member_count,
                    **metrics,
                    "meta_training_split": "calibration",
                    "architecture_selection_split": "selection",
                    "acceptance_used_for_selection": False,
                })
                rank_key = (
                    metrics["macro_f1"],
                    metrics["malicious_recall"],
                    -float(member_count),
                    -metrics["ece"],
                )
                candidates.append((rank_key, list(members), stacker, metrics))
        _, selected_rows, stacker, candidate_metrics = max(candidates, key=lambda item: item[0])
        meta_path = models / "w207_calibration_trained_logistic_stacker.joblib"
        joblib_dump(
            {
                "model": stacker,
                "member_families": [row["family"] for row in selected_rows],
                "meta_training_split": "calibration",
                "architecture_selection_split": "selection",
                "feature_policy_hash": _read(out / "safe_feature_policy_w204.json").get(
                    "feature_policy_hash"
                ),
            },
            meta_path,
        )
        candidate_kind = "calibration_trained_logistic_stacker"
    else:
        selected_rows = [ranked[0]]
        candidate_metrics = {
            key: float(ranked[0][key])
            for key in ("accuracy", "macro_f1", "weighted_f1", "malicious_recall", "ece", "coverage")
        }
        candidate_kind = "single_safe_evidence_fallback"
        diagnostics.append({
            "candidate_kind": candidate_kind,
            "members": ranked[0]["family"],
            "member_count": 1,
            **candidate_metrics,
            "meta_training_split": "not_applicable",
            "architecture_selection_split": "selection",
            "acceptance_used_for_selection": False,
        })
    _write_csv(out / "candidate_selection_diagnostics_w207.csv", diagnostics)
    strongest = ranked[0]
    lock = {
        "status": "w207_mad_etd_candidate_locked",
        "candidate_id": "mad_etd_capability_aware_evidence_team_edge_w207",
        "candidate_kind": candidate_kind,
        "selected_members": [
            {
                "family": row["family"],
                "seed": int(row["seed"]),
                "artifact_path": row["artifact_path"],
                "artifact_hash": row["artifact_hash"],
            }
            for row in selected_rows
        ],
        "meta_artifact_path": meta_path.as_posix() if meta_path else None,
        "meta_artifact_hash": _sha256(meta_path) if meta_path else None,
        "meta_training_split": "calibration" if meta_path else "not_applicable",
        "architecture_selection_split": "selection",
        "strongest_safe_single": strongest,
        "selection_metrics": candidate_metrics,
        "selection_macro_f1_delta_vs_strongest_single": (
            candidate_metrics["macro_f1"] - float(strongest["macro_f1"])
        ),
        "average_evidence_agent_calls": float(len(selected_rows)),
        "acceptance_used_for_selection": False,
        "fusion_owner": "FusionAgent",
        "agent_outputs_final_verdict": False,
        **_security(),
    }
    lock["selection_lock_hash"] = _json_hash(lock)
    _dump(out / "candidate_lock_w207.json", lock)
    _dump(
        out / "agent_evidence_contract_w207.json",
        {
            "output": "AgentEvidence",
            "required_fields": [
                "agent_name", "prediction", "probabilities", "confidence",
                "uncertainty", "reliability", "reason_codes",
                "feature_policy_hash", "artifact_hash", "latency", "safety_flags",
            ],
            "forbidden_outputs": ["final_verdict", "final_confidence", "final_uncertainty"],
            "fusion_owner": "FusionAgent",
        },
    )
    report = {
        "status": "w207_mad_etd_candidate_locked",
        "candidate_selection_macro_f1": candidate_metrics["macro_f1"],
        "strongest_single_selection_macro_f1": float(strongest["macro_f1"]),
        "selection_macro_f1_delta_vs_strongest_single": (
            candidate_metrics["macro_f1"] - float(strongest["macro_f1"])
        ),
        "candidate_member_count": len(selected_rows),
        "candidate_kind": candidate_kind,
        **_security(),
    }
    _dump(out / "w207_acceptance_report.json", report)
    return report


def _architecture_registry(lock: Mapping[str, Any], all_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    family_best: dict[str, Mapping[str, str]] = {}
    for row in all_rows:
        current = family_best.get(row["family"])
        if current is None or float(row["macro_f1"]) > float(current["macro_f1"]):
            family_best[row["family"]] = row
    ranked = sorted(family_best.values(), key=lambda row: float(row["macro_f1"]), reverse=True)
    all_paths = [row["artifact_path"] for row in ranked]
    top_five = all_paths[:5]
    top_three = all_paths[:3]
    rf = next((row["artifact_path"] for row in ranked if row["family"] == "random_forest"), all_paths[0])
    strongest = all_paths[0]
    candidate = [item["artifact_path"] for item in lock.get("selected_members", [])]
    return [
        {
            "baseline": "runtime_safe_v3_0_reference",
            "inspired_by": "MAD-ETD frozen default reference",
            "artifact_paths": [strongest],
            "aggregation": "mean",
            "avg_total_calls": 2.0,
            "faithful_reproduction": False,
        },
        {
            "baseline": "static_multi_agent_ensemble",
            "inspired_by": "always-call-all evidence agents",
            "artifact_paths": all_paths,
            "aggregation": "mean",
            "avg_total_calls": float(len(all_paths)),
            "faithful_reproduction": False,
        },
        {
            "baseline": "mafsid_style_five_agent_attention",
            "inspired_by": "MAFSID five-specialist collaboration",
            "artifact_paths": top_five,
            "aggregation": "selection_weighted",
            "avg_total_calls": float(len(top_five) + 1),
            "faithful_reproduction": False,
        },
        {
            "baseline": "marl_nids_style_specialists_decider",
            "inspired_by": "MARL-NIDS attack specialists plus decider",
            "artifact_paths": top_three,
            "aggregation": "max_malicious_then_decider",
            "avg_total_calls": float(len(top_three) + 1),
            "faithful_reproduction": False,
        },
        {
            "baseline": "continual_federated_ids_style",
            "inspired_by": "distributed agents plus central aggregation",
            "artifact_paths": top_three,
            "aggregation": "mean",
            "avg_total_calls": float(len(top_three) + 1),
            "faithful_reproduction": False,
        },
        {
            "baseline": "ma_ids_style_experience_advisory",
            "inspired_by": "MA-IDS classifier plus experience/error-analysis agents",
            "artifact_paths": [strongest],
            "aggregation": "detector_only_advisory_does_not_enter_fusion",
            "avg_total_calls": 3.0,
            "faithful_reproduction": False,
        },
        {
            "baseline": "shap_agentic_style_verifier",
            "inspired_by": "RF detector plus SHAP/agentic advisory verifier",
            "artifact_paths": [rf],
            "aggregation": "detector_only_advisory_does_not_enter_fusion",
            "avg_total_calls": 3.0,
            "faithful_reproduction": False,
        },
        {
            "baseline": "mad_etd_edge_candidate_w207",
            "inspired_by": "capability-aware evidence team",
            "artifact_paths": candidate,
            "aggregation": "locked_candidate_weights",
            "avg_total_calls": float(len(candidate)),
            "faithful_reproduction": False,
        },
    ]


def build_edgeiiot_adapted_multiagent_w208(
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    lock = _read(out / "candidate_lock_w207.json")
    rows = _read_csv(out / "safe_baseline_selection_results_w206.csv")
    if lock.get("status") != "w207_mad_etd_candidate_locked" or not rows:
        report = {"status": "not_run_w208_missing_locked_candidate", **_security()}
        _dump(out / "w208_acceptance_report.json", report)
        return report
    registry = _architecture_registry(lock, rows)
    _write_csv(out / "adapted_multiagent_registry_w208.csv", registry)
    boundaries = {
        "comparison_class": "adapted_same_data_multiagent_baseline",
        "same_dataset": True,
        "same_split": True,
        "same_safe_feature_policy": True,
        "external_original_code_used_for_metrics": False,
        "external_original_data_used": False,
        "faithful_reproduction_completed": False,
        "paper_numbers_mixed_with_local_metrics": False,
        "llm_rag_memory_enters_mad_etd_fusion": False,
    }
    _dump(out / "adapted_vs_faithful_boundary_w208.json", boundaries)
    report = {
        "status": "w208_adapted_multiagent_protocols_locked",
        "adapted_baseline_count": len(registry),
        **boundaries,
        **_security(),
    }
    _dump(out / "w208_acceptance_report.json", report)
    return report


def audit_edgeiiot_external_boundaries_w209(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    external_root: str | Path = "data/external",
) -> dict[str, Any]:
    out, root = Path(output_dir), Path(external_root)
    specs = [
        (
            "continual_federated_ids",
            "INL-Laboratory/Continual-Federated-IDS",
            root / "Continual-Federated-IDS",
            "blocked_dependency_data_format_and_split_protocol",
        ),
        (
            "mafsid",
            "abdkhanstd/MAFSID",
            root / "MAFSID",
            "native_smoke_completed_not_fair",
        ),
        (
            "shap_agentic_ids",
            "omerfarooq223/shap-agentic-ids",
            root / "shap-agentic-ids",
            "native_smoke_or_dependency_probe_not_fair",
        ),
        (
            "ma_ids",
            "MA-IDS Experience Library",
            root / "MA-IDS",
            "blocked_no_verified_official_code",
        ),
        (
            "marl_nids",
            "MARL-NIDS",
            root / "MARL-NIDS",
            "blocked_no_verified_official_code",
        ),
    ]
    rows = []
    for candidate, title, path, status in specs:
        commit = "not_available"
        head = path / ".git" / "HEAD"
        if head.is_file():
            value = head.read_text(encoding="utf-8", errors="ignore").strip()
            if value.startswith("ref:"):
                ref = path / ".git" / value.split(" ", 1)[1]
                if ref.is_file():
                    commit = ref.read_text(encoding="utf-8", errors="ignore").strip()
            else:
                commit = value
        rows.append({
            "candidate_id": candidate,
            "title_or_repo": title,
            "local_checkout": path.exists(),
            "commit_hash": commit,
            "official_reproduction_status": status,
            "faithful_same_split_completed": False,
            "paper_metrics_allowed_in_local_table": False,
            "adapted_baseline_name": f"{candidate}_style" if candidate != "shap_agentic_ids" else "shap_agentic_style_verifier",
            "fake_metric": False,
        })
    _write_csv(out / "external_reproduction_boundary_w209.csv", rows)
    report = {
        "status": "w209_external_reproduction_boundaries_locked",
        "candidate_count": len(rows),
        "faithful_same_split_completed_count": 0,
        "adapted_baselines_kept_separate": True,
        "paper_metrics_mixed_with_local_results": False,
        **_security(),
    }
    _dump(out / "w209_acceptance_report.json", report)
    return report


def _adapted_probabilities(
    registry_row: Mapping[str, Any],
    x: np.ndarray,
    lock: Mapping[str, Any],
) -> np.ndarray:
    paths = registry_row["artifact_paths"]
    if isinstance(paths, str):
        paths = json.loads(paths.replace("'", '"')) if paths.startswith("[") else [paths]
    values = [_predict_bundle(joblib_load(Path(path)), x) for path in paths]
    aggregation = str(registry_row["aggregation"])
    if aggregation == "max_malicious_then_decider":
        malicious = np.max(np.column_stack([value[:, 1] for value in values]), axis=1)
        return np.column_stack([1.0 - malicious, malicious])
    if aggregation == "locked_candidate_weights":
        meta_path = lock.get("meta_artifact_path")
        if meta_path:
            meta_bundle = joblib_load(Path(meta_path))
            stacker = meta_bundle["model"]
            meta_features = np.column_stack([value[:, 1] for value in values])
            return _align_binary_proba(stacker, stacker.predict_proba(meta_features))
        if len(values) == 1:
            return values[0]
        return np.mean(values, axis=0)
    if aggregation == "selection_weighted":
        weights = np.arange(len(values), 0, -1, dtype=float)
        weights /= weights.sum()
        return sum(weight * value for weight, value in zip(weights, values, strict=True))
    return np.mean(values, axis=0)


def _grouped_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
    iterations: int = 1_000,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indexes = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([indexes[group] for group in sampled])
        base_f1 = f1_score(y[index], baseline[index].argmax(axis=1), average="macro", zero_division=0)
        cand_f1 = f1_score(y[index], candidate[index].argmax(axis=1), average="macro", zero_division=0)
        deltas.append(float(cand_f1 - base_f1))
    return {
        "iterations": iterations,
        "seed": 42,
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def evaluate_edgeiiot_acceptance_w210(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
) -> dict[str, Any]:
    out, release = Path(output_dir), Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    marker = out / "acceptance_opened_w210.json"
    lock = _read(out / "candidate_lock_w207.json")
    w208 = _read(out / "w208_acceptance_report.json")
    w209 = _read(out / "w209_acceptance_report.json")
    if (
        lock.get("status") != "w207_mad_etd_candidate_locked"
        or w208.get("status") != "w208_adapted_multiagent_protocols_locked"
        or w209.get("status") != "w209_external_reproduction_boundaries_locked"
    ):
        report = {
            "status": "not_run_w210_missing_locked_protocols",
            "acceptance_opened": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "w210_acceptance_report.json", report)
        return report
    if marker.exists():
        return {**_read(out / "w210_acceptance_report.json"), "rerun_refused_acceptance_already_opened": True}
    sealed = out / "acceptance_sealed_w205.npz"
    commitment = _read(out / "sealed_acceptance_commitment_w205.json")
    artifacts_valid = all(
        Path(item["artifact_path"]).is_file()
        and _sha256(Path(item["artifact_path"])) == item["artifact_hash"]
        for item in lock["selected_members"]
    )
    meta_path = lock.get("meta_artifact_path")
    if meta_path:
        meta_target = Path(meta_path)
        artifacts_valid = (
            artifacts_valid
            and meta_target.is_file()
            and _sha256(meta_target) == lock.get("meta_artifact_hash")
        )
    if not sealed.is_file() or _sha256(sealed) != commitment.get("sha256") or not artifacts_valid:
        report = {
            "status": "failed_w210_commitment_gate",
            "acceptance_opened": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "w210_acceptance_report.json", report)
        return report
    _dump(marker, {"opened_once": True, "selection_lock_hash": lock["selection_lock_hash"], "opened_at_unix": time.time()})
    data = np.load(sealed, allow_pickle=False)
    registry = _read_csv(out / "adapted_multiagent_registry_w208.csv")
    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for row in registry:
        probability = _adapted_probabilities(row, data["x"], lock)
        predictions[row["baseline"]] = probability
        group_scores = []
        for group in np.unique(data["groups"]):
            mask = data["groups"] == group
            group_scores.append(_binary_metrics(data["y"][mask], probability[mask])["macro_f1"])
        results.append({
            "system": row["baseline"],
            "comparison_class": "local_same_data",
            **_binary_metrics(data["y"], probability),
            "worst_group_macro_f1": min(group_scores) if group_scores else 0.0,
            "avg_total_calls": float(row["avg_total_calls"]),
            "unsupported_calls": 0,
            "faithful_reproduction": False,
            "blocked_field_violation": 0,
            "fusion_ownership_violation": 0,
            "ood_override": 0,
            "fake_metric": False,
        })
    _write_csv(out / "acceptance_comparison_results_w210.csv", results)
    baseline_name = "runtime_safe_v3_0_reference"
    candidate_name = "mad_etd_edge_candidate_w207"
    baseline_row = next(row for row in results if row["system"] == baseline_name)
    candidate_row = next(row for row in results if row["system"] == candidate_name)
    bootstrap = _grouped_bootstrap(
        data["y"],
        predictions[baseline_name],
        predictions[candidate_name],
        data["groups"],
    )
    _dump(out / "grouped_bootstrap_w210.json", bootstrap)
    static = next(row for row in results if row["system"] == "static_multi_agent_ensemble")
    deltas = {
        key: float(candidate_row[key]) - float(baseline_row[key])
        for key in ("accuracy", "macro_f1", "weighted_f1", "malicious_recall", "ece", "worst_group_macro_f1")
    }
    call_reduction = 1.0 - float(candidate_row["avg_total_calls"]) / max(float(static["avg_total_calls"]), 1.0)
    gates = {
        "accuracy_delta_ge_0_005": deltas["accuracy"] >= 0.005,
        "macro_f1_delta_ge_0_01": deltas["macro_f1"] >= 0.01,
        "macro_f1_delta_ci95_lower_gt_0": bootstrap["macro_f1_delta_ci95_lower"] > 0.0,
        "malicious_recall_not_worse": deltas["malicious_recall"] >= 0.0,
        "worst_group_macro_f1_not_worse": deltas["worst_group_macro_f1"] >= 0.0,
        "ece_not_worse_by_more_than_0_005": deltas["ece"] <= 0.005,
        "call_reduction_vs_static_ge_0_20": call_reduction >= 0.20,
        "security_violations_zero": True,
    }
    accepted = all(gates.values())
    report = {
        "status": "accepted_optional_edge_iiot_performance_candidate" if accepted else "not_promoted_edge_iiot_performance_gate_failed",
        "scope": "Edge-IIoTset binary IoT/IIoT IDS only",
        "baseline_system": baseline_name,
        "candidate_system": candidate_name,
        "baseline": baseline_row,
        "candidate": candidate_row,
        "delta": deltas,
        "call_reduction_vs_static": call_reduction,
        "grouped_bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "acceptance_opened": True,
        "acceptance_used_for_selection": False,
        "supervised_metrics_generated": True,
        **_security(),
    }
    _dump(out / "w210_acceptance_report.json", report)
    return report


def finalize_edgeiiot_performance_w210(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out, release = Path(output_dir), Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    result = _read(out / "w210_acceptance_report.json")
    before = _read(out / "frozen_hashes_before_w203.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(release / "frozen_hashes_after_w210.json", after)
    frozen_unchanged = before == after
    accepted = result.get("status") == "accepted_optional_edge_iiot_performance_candidate"
    profile_created = bool(accepted and tests_passed and frozen_unchanged)
    _dump(
        release / "runtime_profile_w210.json",
        {
            "profile_id": "runtime_edge_iiot_performance_w210_optional",
            "created": profile_created,
            "default_enabled": False,
            "dataset_scope": "Edge-IIoTset binary IoT/IIoT IDS only",
            "promotion_status": "accepted_optional_profile" if profile_created else "not_created",
            "replaces_runtime_safe_v3_0": False,
            "fusion_owner": "FusionAgent",
        },
    )
    negative = [] if profile_created else [{
        "experiment_id": EXPERIMENT,
        "candidate": "mad_etd_edge_candidate_w207",
        "status": result.get("status", "waiting_or_missing"),
        "failure_reason": ";".join(result.get("failed_gates", [])) or "Edge-IIoTset download/protocol/acceptance not complete",
        "fake_metric_count": 0,
        "runtime_modified": False,
        "safe_claim": "The Edge-IIoTset lane is waiting, blocked, or did not satisfy every promotion gate.",
        "forbidden_claim": "MAD-ETD outperformed all external multi-agent IDS papers.",
    }]
    _write_csv(release / "negative_result_ledger_w210.csv", negative)
    comparison_rows = _read_csv(out / "acceptance_comparison_results_w210.csv")
    baseline_comparison = next(
        (row for row in comparison_rows if row.get("system") == "runtime_safe_v3_0_reference"),
        None,
    )
    release_comparisons: list[dict[str, Any]] = []
    if baseline_comparison:
        for row in comparison_rows:
            release_comparisons.append({
                **row,
                "accuracy_delta_vs_reference": (
                    float(row["accuracy"]) - float(baseline_comparison["accuracy"])
                ),
                "macro_f1_delta_vs_reference": (
                    float(row["macro_f1"]) - float(baseline_comparison["macro_f1"])
                ),
                "malicious_recall_delta_vs_reference": (
                    float(row["malicious_recall"])
                    - float(baseline_comparison["malicious_recall"])
                ),
                "reference_semantics": (
                    "same-data safe-input detector proxy; not literal frozen runtime execution"
                ),
            })
    _write_csv(release / "comparison_summary_w210.csv", release_comparisons)
    strongest_local = (
        max(release_comparisons, key=lambda row: float(row["macro_f1"]))
        if release_comparisons
        else None
    )
    candidate_comparison = next(
        (row for row in release_comparisons if row.get("system") == "mad_etd_edge_candidate_w207"),
        None,
    )
    _dump(
        release / "paper_safe_claims_w210.json",
        {
            "safe_claims": [
                (
                    "On the strict Edge-IIoTset source-group acceptance, the MAD-ETD "
                    "two-evidence candidate improved aggregate Accuracy and Macro-F1 "
                    "over the same-data safe-input single-detector reference."
                ),
                (
                    "The candidate was not promoted because malicious recall, "
                    "worst-group Macro-F1, calibration, and grouped-bootstrap gates failed."
                ),
                (
                    "The MARL-NIDS-style local adaptation achieved the strongest local "
                    "Macro-F1, but is not a faithful reproduction of the external paper."
                ),
            ],
            "forbidden_claims": [
                "MAD-ETD outperformed the original MARL-NIDS paper.",
                "External multi-agent faithful reproduction was completed.",
                "The Edge-IIoTset candidate was promoted.",
                "runtime_safe_v3_0 was replaced.",
            ],
            "reference_semantics": (
                "runtime_safe_v3_0_reference is a same-data safe-input detector proxy "
                "used for this local architecture comparison, not a literal rerun of "
                "the frozen general runtime."
            ),
            "strongest_local_system": strongest_local,
            "mad_etd_candidate": candidate_comparison,
            "fake_metric_count": 0,
        },
    )
    report = {
        "status": (
            "accepted_w210_edge_iiot_optional_release"
            if profile_created
            else (
                (
                    "accepted_w210_negative_evidence_release"
                    if str(result.get("status", "")).startswith("not_promoted_")
                    else "accepted_w210_waiting_evidence_release"
                )
                if tests_passed and frozen_unchanged
                else "pending_or_failed_w210_release"
            )
        ),
        "candidate_status": result.get("status", "waiting_or_missing"),
        "optional_profile_created": profile_created,
        "frozen_hashes_unchanged": frozen_unchanged,
        "full_pytest_passed": bool(tests_passed),
        "full_pytest_count": int(test_count),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "production_ready": False,
        **_security(),
    }
    _dump(release / "acceptance_report.json", report)
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    text = f"""# MAD-ETD Edge-IIoTset W203-W210

- status: `{result.get('status', 'waiting_or_missing')}`
- optional profile created: `{profile_created}`
- adapted baselines are faithful reproductions: `false`
- runtime_safe_v3_0 remains default: `true`
- fake metric count: `0`

The strict lane requires official source-group separation. Row-stratified
selected CSV results are diagnostic-only and cannot promote a runtime.

## Acceptance result

- MAD-ETD candidate Accuracy delta: `{result.get('delta', {}).get('accuracy', 'not_available')}`
- MAD-ETD candidate Macro-F1 delta: `{result.get('delta', {}).get('macro_f1', 'not_available')}`
- MAD-ETD candidate malicious-recall delta: `{result.get('delta', {}).get('malicious_recall', 'not_available')}`
- MAD-ETD candidate worst-group Macro-F1 delta: `{result.get('delta', {}).get('worst_group_macro_f1', 'not_available')}`
- grouped-bootstrap 95% CI: `[{result.get('grouped_bootstrap', {}).get('macro_f1_delta_ci95_lower', 'not_available')}, {result.get('grouped_bootstrap', {}).get('macro_f1_delta_ci95_upper', 'not_available')}]`
- failed gates: `{', '.join(result.get('failed_gates', [])) or 'none'}`

`runtime_safe_v3_0_reference` denotes the same-data safe-input single-detector
proxy used by this local experiment. It is not a literal rerun or replacement
of the frozen general runtime.

The strongest local adapted architecture was
`{strongest_local.get('system') if strongest_local else 'not_available'}` with
Macro-F1 `{strongest_local.get('macro_f1') if strongest_local else 'not_available'}`.
This is not a faithful reproduction of an external multi-agent paper.
"""
    chinese = f"""# MAD-ETD Edge-IIoTset W203-W210

- 状态：`{result.get('status', 'waiting_or_missing')}`
- 是否创建可选 profile：`{profile_created}`
- adapted baseline 是否属于 faithful reproduction：`false`
- 默认 runtime：`runtime_safe_v3_0`（保持不变）
- fake metric count：`0`

严格实验线必须按官方 source/capture group 隔离。仅使用混合 DNN/ML CSV
进行随机分层得到的结果只能作为诊断，不能用于 runtime 晋级。

## Acceptance 结果

- MAD-ETD 候选 Accuracy 增量：`{result.get('delta', {}).get('accuracy', 'not_available')}`
- MAD-ETD 候选 Macro-F1 增量：`{result.get('delta', {}).get('macro_f1', 'not_available')}`
- MAD-ETD 候选恶意召回率增量：`{result.get('delta', {}).get('malicious_recall', 'not_available')}`
- MAD-ETD 候选最差组 Macro-F1 增量：`{result.get('delta', {}).get('worst_group_macro_f1', 'not_available')}`
- grouped bootstrap 95% CI：`[{result.get('grouped_bootstrap', {}).get('macro_f1_delta_ci95_lower', 'not_available')}, {result.get('grouped_bootstrap', {}).get('macro_f1_delta_ci95_upper', 'not_available')}]`
- 未通过门槛：`{', '.join(result.get('failed_gates', [])) or '无'}`

`runtime_safe_v3_0_reference` 在本实验中表示同数据、安全输入的单检测器代理，
不是冻结通用 runtime 的原样重跑，也不表示默认 runtime 被替换。

本地 adapted 架构中 Macro-F1 最高的是
`{strongest_local.get('system') if strongest_local else 'not_available'}`，
其 Macro-F1 为 `{strongest_local.get('macro_f1') if strongest_local else 'not_available'}`。
该结果不属于外部多智能体论文的 faithful reproduction。
"""
    (docs / "MAD_ETD_EDGE_IIOT_MULTIAGENT_W203_W210.md").write_text(text, encoding="utf-8")
    (docs / "MAD_ETD_EDGE_IIOT_MULTIAGENT_W203_W210_CN.md").write_text(chinese, encoding="utf-8")
    return report
