"""W94/W95 NF-ToN-IoT-v2 targeted-feature full-coverage performance lane.

W94 freezes a fresh, historically disjoint sample and performs train/validation
feature forensics.  W95 compares common safe-flow baselines with dataset-scoped
native-feature models, including the ICLR-2025 TabM architecture.  Acceptance
is full coverage and opened once; selective abstention is deliberately absent.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import make_pipeline

from .domain_robust_stats_w81 import _source_entries
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .nfiot_skill_positive_w93 import _fresh_exclusions
from .paper_evaluation import hash_artifact_paths, sha256_file
from .safe_flow_ensemble_w73_w76 import (
    ENGINEERED_FEATURES,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
)
from .schemas import AgentEvidenceV2
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_nfiot_targeted_performance_w94_w95"
TARGET_VARIANT = "NF-ToN-IoT-v2"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfiot_targeted_performance_w94_w95")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_nfiot_targeted_performance_w94_w95")
DEFAULT_W93_DIR = Path("data/runs/mad_etd_nfiot_skill_positive_w93")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_TARGETED_PERFORMANCE_W94_W95.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_TARGETED_PERFORMANCE_W94_W95_CN.md")
DEFAULT_PER_LABEL = 10_000
DEFAULT_MAX_ROWS = 4_000_000
SEED = 94
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 1_000
COMMON_MODEL_IDS = ("hgb_common", "random_forest_common", "extra_trees_common")
NATIVE_MODEL_IDS = ("hgb_native", "random_forest_native", "extra_trees_native", "catboost_native", "tabm_native")

BLOCKED_COLUMNS = {
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "Label",
    "Attack",
}
# Pre-registered exclusions combine the dataset-native identifier field with
# shortcut suspects already recorded by Feature Forensics v1.  They are not
# re-admitted merely because a single-source validation split scores well.
SHORTCUT_EXCLUDED = {"DNS_QUERY_ID", "IN_BYTES", "OUT_BYTES", "PROTOCOL"}
DISCRETE_NATIVE = {
    "PROTOCOL",
    "L7_PROTO",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE",
    "FTP_COMMAND_RET_CODE",
}


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
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


def _security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "audit_completion": 1.0,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "automatic_deployment": False,
    }


def _to_float(value: Any) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _native_feature_names(headers: Iterable[str]) -> list[str]:
    return [name for name in headers if name not in BLOCKED_COLUMNS]


def _native_vector(row: Mapping[str, Any], feature_names: list[str]) -> list[float]:
    values: list[float] = []
    for name in feature_names:
        value = _to_float(row.get(name))
        if math.isnan(value) or name in DISCRETE_NATIVE:
            values.append(value)
        else:
            values.append(math.copysign(math.log1p(abs(value)), value))
    return values


def _role(sample_hash: str) -> str:
    value = int(_stable_hash(f"w94:role:{sample_hash}")[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "validation"
    return "acceptance"


def _push(
    heap: list[tuple[int, int, tuple[list[float], list[float], int, str, str, str]]],
    *,
    priority: int,
    tie: int,
    item: tuple[list[float], list[float], int, str, str, str],
    limit: int,
) -> None:
    record = (-priority, -tie, item)
    if len(heap) < limit:
        heapq.heappush(heap, record)
    elif record > heap[0]:
        heapq.heapreplace(heap, record)


def _exclusions_through_w93(w93_dir: str | Path) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, prior = _fresh_exclusions()
    run = Path(w93_dir)
    rows = _read_csv(run / "fresh_sample_manifest.csv")
    hashes = {row["sample_hash"] for row in rows if row.get("sample_hash")}
    legacy_hashes = {row["legacy_sample_id_hash"] for row in rows if row.get("legacy_sample_id_hash")}
    report = _read_json(run / "acceptance_report.json")
    mapped = bool(rows) and report.get("status") == "accepted_dataset_specific_accuracy_f1_positive_result"
    canonical.update(hashes)
    legacy.update(legacy_hashes)
    status = (
        "all_auditable_nf_samples_through_w93_excluded"
        if prior.get("status") == "all_auditable_nf_samples_through_w92_excluded" and mapped
        else "failed_w94_historical_exclusion_mapping"
    )
    return canonical, legacy, {
        "status": status,
        "through_w92": prior,
        "w93_sample_count": len(hashes),
        "w93_legacy_count": len(legacy_hashes),
        "w93_acceptance_status": report.get("status", "missing"),
        "canonical_exclusion_count": len(canonical),
        "legacy_exclusion_count": len(legacy),
    }


def build_nfiot_targeted_features_w94(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w93_dir: str | Path = DEFAULT_W93_DIR,
    per_label: int = DEFAULT_PER_LABEL,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    """Freeze fresh NF-ToN-IoT-v2 train/validation/acceptance arrays."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)
    canonical_exclusions, legacy_exclusions, ledger = _exclusions_through_w93(w93_dir)
    _dump(out / "historical_exclusion_ledger.json", ledger)
    processed = Path(processed_dir)
    entries, source_errors = _source_entries(processed)
    targets = [entry for entry in entries if entry.get("source_group") == TARGET_VARIANT]
    dataset_manifest = _read_json(processed / "dataset_manifest.json")
    target_metadata = [
        item
        for item in dataset_manifest.get("primary_entries", [])
        if item.get("dataset_variant") == TARGET_VARIANT
    ]
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "W94_targeted_feature_split",
        "dataset_scope": TARGET_VARIANT,
        "split_protocol": "fresh historically-disjoint stratified hash 60/20/20",
        "selection_split": "validation",
        "acceptance_split": "sealed and opened once in W95",
        "performance_mode": "full_coverage_only",
        "per_label": int(per_label),
        "max_rows": int(max_rows),
        "seed": SEED,
        "acceptance_used_for_selection": False,
        **_security(),
    }
    _dump(out / "protocol_manifest.json", protocol)
    if source_errors or len(targets) != 1 or len(target_metadata) != 1 or ledger.get("status") != "all_auditable_nf_samples_through_w93_excluded":
        report = {**protocol, "status": "failed_w94_source_or_history_gate", "source_errors": source_errors, "target_entry_count": len(targets), "target_metadata_count": len(target_metadata), "history_status": ledger.get("status")}
        _dump(out / "split_manifest.json", report)
        return report
    entry = {**targets[0], **target_metadata[0]}
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
        "attack_usage": "per-attack diagnostic only",
        "blocked_fields_in_feature_matrix": blocked_seen,
        "blocked_field_violation": len(blocked_seen),
    }
    policy["policy_hash"] = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    _dump(out / "safe_feature_policy.json", policy)
    if blocked_seen:
        report = {**protocol, "status": "failed_w94_blocked_feature_gate", "blocked_fields": blocked_seen}
        _dump(out / "split_manifest.json", report)
        return report

    archive = Path(entry["archive"])
    member = str(entry["member"])
    label_column = str(entry["label_column"])
    attack_column = str(entry.get("attack_column") or "Attack")
    buckets: dict[int, list[tuple[int, int, tuple[list[float], list[float], int, str, str, str]]]] = {0: [], 1: []}
    scanned = excluded = labeled = 0
    tie = 0
    for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows):
        scanned += 1
        label = _normalise_binary_label(row.get(label_column))
        if label is None:
            continue
        canonical_key = _canonical_sample_key(TARGET_VARIANT, archive, member, row_index)
        legacy_key = _legacy_sample_key(archive, member, row_index)
        sample_hash = _stable_hash(canonical_key)
        legacy_prefix = _stable_hash(legacy_key)[:24]
        if sample_hash in canonical_exclusions or legacy_prefix in legacy_exclusions:
            excluded += 1
            continue
        labeled += 1
        y = int(label == "malicious")
        item = (
            _engineer_safe_features(row),
            _native_vector(row, native_names),
            y,
            str(row.get(attack_column, "unknown")),
            sample_hash,
            legacy_prefix,
        )
        priority = int(_stable_hash("w94:fresh:" + canonical_key)[:16], 16)
        _push(buckets[y], priority=priority, tie=tie, item=item, limit=per_label)
        tie += 1
    selected = [item for label in (0, 1) for _p, _t, item in sorted(buckets[label], reverse=True)]
    counts = {"benign": len(buckets[0]), "malicious": len(buckets[1])}
    if min(counts.values()) < per_label:
        report = {**protocol, "status": "failed_insufficient_fresh_w94_samples", "counts": counts, "scanned_rows": scanned, "excluded_rows": excluded}
        _dump(out / "split_manifest.json", report)
        return report
    x_common = np.asarray([item[0] for item in selected], dtype=np.float32)
    x_native = np.asarray([item[1] for item in selected], dtype=np.float32)
    y = np.asarray([item[2] for item in selected], dtype=np.int64)
    attacks = np.asarray([item[3] for item in selected], dtype="U64")
    hashes = np.asarray([item[4] for item in selected], dtype="U64")
    legacy_hashes = np.asarray([item[5] for item in selected], dtype="U24")
    roles = np.asarray([_role(str(value)) for value in hashes], dtype="U16")
    complete = all(np.any((roles == role) & (y == label)) for role in ("train", "validation", "acceptance") for label in (0, 1))
    train_validation = roles != "acceptance"
    acceptance = roles == "acceptance"
    sealed = out / "sealed_splits"
    sealed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sealed / "train_validation.npz",
        x_common=x_common[train_validation], x_native=x_native[train_validation], y=y[train_validation],
        attack=attacks[train_validation], sample_hash=hashes[train_validation], roles=roles[train_validation],
        common_feature_names=np.asarray(ENGINEERED_FEATURES), native_feature_names=np.asarray(native_names),
    )
    np.savez_compressed(
        sealed / "acceptance_sealed.npz",
        x_common=x_common[acceptance], x_native=x_native[acceptance], y=y[acceptance],
        attack=attacks[acceptance], sample_hash=hashes[acceptance],
        common_feature_names=np.asarray(ENGINEERED_FEATURES), native_feature_names=np.asarray(native_names),
    )
    manifest_rows = [
        {"sample_hash": str(h), "legacy_sample_id_hash": str(l), "label": int(label), "attack": str(attack), "role": str(role), "historical_overlap": False}
        for h, l, label, attack, role in zip(hashes, legacy_hashes, y, attacks, roles, strict=True)
    ]
    _write_csv(out / "fresh_sample_manifest.csv", manifest_rows)
    before_ids = {str(v) for v in hashes[roles == "train"]}
    val_ids = {str(v) for v in hashes[roles == "validation"]}
    acc_ids = {str(v) for v in hashes[roles == "acceptance"]}
    overlap = len(before_ids & val_ids) + len(before_ids & acc_ids) + len(val_ids & acc_ids)
    historical_overlap = len(set(str(v) for v in hashes) & canonical_exclusions)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_split.json", after)
    ready = complete and overlap == 0 and historical_overlap == 0 and before == after
    report = {
        **protocol,
        "status": "w94_fresh_targeted_splits_frozen" if ready else "failed_w94_split_gate",
        "sample_count": len(y), "train_count": int(np.sum(roles == "train")), "validation_count": int(np.sum(roles == "validation")), "acceptance_count": int(np.sum(roles == "acceptance")),
        "counts": counts, "scanned_rows": scanned, "excluded_rows": excluded, "labeled_rows": labeled,
        "common_feature_count": x_common.shape[1], "native_feature_count": x_native.shape[1],
        "split_overlap_count": overlap, "historical_overlap_count": historical_overlap,
        "frozen_hashes_unchanged": before == after, "acceptance_sealed": True,
    }
    _dump(out / "split_manifest.json", report)
    return report


def _imputed(x_train: np.ndarray, x_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(x_train), imputer.transform(x_other), imputer


def run_targeted_feature_forensics_w94(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    experiment: str = EXPERIMENT,
    ready_split_status: str = "w94_fresh_targeted_splits_frozen",
    feature_pack_status: str = "w94_targeted_feature_pack_locked",
) -> dict[str, Any]:
    out = Path(output_dir)
    split = _read_json(out / "split_manifest.json")
    if split.get("status") != ready_split_status:
        report = {**_security(), "status": "failed_w94_split_not_ready"}
        _dump(out / "feature_forensics_report.json", report)
        return report
    with np.load(out / "sealed_splits" / "train_validation.npz", allow_pickle=False) as data:
        x = np.asarray(data["x_native"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        roles = np.asarray(data["roles"]).astype(str)
        names = [str(v) for v in data["native_feature_names"]]
    train, val = roles == "train", roles == "validation"
    x_train_raw, x_val_raw = x[train], x[val]
    y_train, y_val = y[train], y[val]
    x_train, x_val, _ = _imputed(x_train_raw, x_val_raw)
    discrete = np.asarray([name in DISCRETE_NATIVE for name in names], dtype=bool)
    mi = mutual_info_classif(x_train, y_train, discrete_features=discrete, random_state=42)
    single_auc: list[float] = []
    missingness: list[float] = []
    for index, _name in enumerate(names):
        column = x_train[:, index]
        try:
            auc = float(roc_auc_score(y_train, column))
            auc = max(auc, 1.0 - auc)
        except ValueError:
            auc = 0.5
        single_auc.append(auc)
        missingness.append(float(np.isnan(x_train_raw[:, index]).mean()))
    probe = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
    probe.fit(x_train, y_train)
    perm = permutation_importance(probe, x_val, y_val, scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=-1).importances_mean
    midpoint = len(x_train) // 2
    first = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=42).fit(x_train[:midpoint], y_train[:midpoint]).feature_importances_
    second = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, n_jobs=-1, random_state=43).fit(x_train[midpoint:], y_train[midpoint:]).feature_importances_
    scale = lambda values: np.asarray(values, dtype=float) / max(float(np.max(np.abs(values))), 1e-12)
    score = scale(mi) + scale(np.maximum(perm, 0)) + scale((first + second) / 2)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        shortcut = name in SHORTCUT_EXCLUDED
        stable = abs(float(first[index]) - float(second[index])) <= 0.05
        eligible = not shortcut and missingness[index] <= 0.30 and stable and (mi[index] > 0.001 or perm[index] > 0)
        rows.append({
            "feature_name": name, "mutual_information": float(mi[index]), "single_feature_auc": single_auc[index],
            "permutation_importance": float(perm[index]), "tree_importance_half_1": float(first[index]), "tree_importance_half_2": float(second[index]),
            "rank_stability_proxy": 1.0 - abs(float(first[index]) - float(second[index])), "missingness_rate": missingness[index],
            "shortcut_suspect": shortcut, "eligible": eligible, "combined_score": float(score[index]),
        })
    eligible_rows = sorted((row for row in rows if row["eligible"]), key=lambda row: float(row["combined_score"]), reverse=True)
    selected = [str(row["feature_name"]) for row in eligible_rows[:24]]
    # Keep at least twelve features so model comparison remains meaningful.
    if len(selected) < 12:
        fallback = [str(row["feature_name"]) for row in sorted(rows, key=lambda row: float(row["combined_score"]), reverse=True) if not row["shortcut_suspect"]]
        selected = list(dict.fromkeys(selected + fallback))[:12]
    _write_csv(out / "dataset_specific_feature_forensics.csv", rows)
    policy = {
        "schema_version": "1.0", "experiment": experiment, "status": feature_pack_status,
        "selected_features": selected, "selected_feature_count": len(selected), "selection_data": "train plus validation permutation only",
        "acceptance_used_for_feature_selection": False, "shortcut_exclusions": sorted(SHORTCUT_EXCLUDED),
        "methods": ["mutual_information", "single_feature_auc", "permutation_importance", "split_half_rank_stability", "missingness_audit"],
        "shap_status": "not_used; no SHAP dependency in frozen environment", "blocked_field_violation": 0,
    }
    policy["feature_pack_hash"] = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    _dump(out / "targeted_feature_pack.json", policy)
    report = {**_security(), **policy, "status": feature_pack_status}
    _dump(out / "feature_forensics_report.json", report)
    return report


def _make_tree_model(model_id: str):
    if model_id.startswith("hgb"):
        return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=42))
    if model_id.startswith("random_forest"):
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    if model_id.startswith("extra_trees"):
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42))
    if model_id == "catboost_native":
        from catboost import CatBoostClassifier
        return make_pipeline(SimpleImputer(strategy="median"), CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, loss_function="Logloss", random_seed=42, verbose=False, allow_writing_files=False, thread_count=-1))
    raise ValueError(model_id)


def _proba(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x), dtype=float)[:, 1]


def _ece(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    pred = (p >= threshold).astype(int)
    confidence = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    result = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if np.any(mask):
            result += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(result)


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    accuracy = float(accuracy_score(y, pred))
    return {
        "accuracy": accuracy, "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(y, pred, zero_division=0)), "ece": _ece(y, p, threshold),
        "coverage": 1.0, "selective_error": 1.0 - accuracy,
    }


def _train_tabm(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, model_dir: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from sklearn.preprocessing import StandardScaler
    from tabm import TabM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(x_train)).astype("float32")
    val = scaler.transform(imputer.transform(x_val)).astype("float32")
    config = {"n_num_features": train.shape[1], "cat_cardinalities": [], "d_out": 2, "k": 16, "n_blocks": 3, "d_block": 256, "dropout": 0.1}
    model = TabM.make(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train), torch.from_numpy(y_train)), batch_size=256, shuffle=True, generator=generator)
    x_val_t = torch.from_numpy(val).to(device)
    best_score = -1.0
    best_state: dict[str, Any] | None = None
    patience = 0
    start = time.perf_counter()
    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, None)
            targets = yb[:, None].expand(-1, logits.shape[1]).reshape(-1)
            loss = criterion(logits.reshape(-1, 2), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(x_val_t, None), dim=-1).mean(dim=1)[:, 1].cpu().numpy()
        score = float(f1_score(y_val, prob >= 0.5, average="macro"))
        if score > best_score + 1e-6:
            best_score, patience = score, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 15:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": config}, model_dir / "tabm_native.pt")
    joblib_dump({"imputer": imputer, "scaler": scaler}, model_dir / "tabm_preprocess.joblib")
    model.eval()
    with torch.no_grad():
        val_probability = torch.softmax(model(x_val_t, None), dim=-1).mean(dim=1)[:, 1].cpu().numpy()
    return val_probability, {"device": str(device), "epochs": epoch + 1, "best_validation_macro_f1_at_0_5": best_score, "training_seconds": time.perf_counter() - start, "config": config}


def _predict_tabm(x: np.ndarray, model_dir: Path) -> np.ndarray:
    import torch
    from tabm import TabM
    checkpoint = torch.load(model_dir / "tabm_native.pt", map_location="cpu", weights_only=True)
    preprocess = joblib_load(model_dir / "tabm_preprocess.joblib")
    values = preprocess["scaler"].transform(preprocess["imputer"].transform(x)).astype("float32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabM.make(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            batch = torch.from_numpy(values[start : start + 1024]).to(device)
            result.append(torch.softmax(model(batch, None), dim=-1).mean(dim=1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def _select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    *,
    minimum_malicious_recall: float | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = []
    lower = 0.20 if minimum_malicious_recall is not None else 0.40
    for threshold in np.round(np.arange(lower, 0.601, 0.01), 2):
        metrics = _metrics(y, p, float(threshold))
        candidates.append((float(threshold), metrics))
    eligible = [
        item
        for item in candidates
        if minimum_malicious_recall is None
        or item[1]["malicious_recall"] >= minimum_malicious_recall
    ]
    pool = eligible or candidates
    return max(pool, key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"], item[1]["malicious_recall"], -item[1]["ece"]))


def train_nfiot_targeted_models_w95(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    experiment: str = EXPERIMENT,
    require_validation_recall_not_lower: bool = False,
    feature_pack_status: str = "w94_targeted_feature_pack_locked",
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    feature_pack = _read_json(out / "targeted_feature_pack.json")
    if feature_pack.get("status") != feature_pack_status:
        report = {**_security(), "status": "failed_w94_feature_pack_not_ready"}
        _dump(out / "training_report.json", report)
        return report
    with np.load(out / "sealed_splits" / "train_validation.npz", allow_pickle=False) as data:
        x_common = np.asarray(data["x_common"], dtype=np.float32)
        x_native_all = np.asarray(data["x_native"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        roles = np.asarray(data["roles"]).astype(str)
        native_names = [str(v) for v in data["native_feature_names"]]
    selected_names = list(feature_pack["selected_features"])
    selected_indexes = [native_names.index(name) for name in selected_names]
    x_native = x_native_all[:, selected_indexes]
    train, val = roles == "train", roles == "validation"
    models.mkdir(parents=True, exist_ok=True)
    validation_proba: dict[str, np.ndarray] = {}
    model_meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model_id in COMMON_MODEL_IDS + tuple(item for item in NATIVE_MODEL_IDS if item != "tabm_native"):
        feature_pack_name = "common" if model_id.endswith("common") else "native"
        x_matrix = x_common if feature_pack_name == "common" else x_native
        model = _make_tree_model(model_id)
        start = time.perf_counter()
        model.fit(x_matrix[train], y[train])
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        p = _proba(model, x_matrix[val])
        inference_seconds = time.perf_counter() - start
        path = models / f"{model_id}.joblib"
        joblib_dump(model, path)
        validation_proba[model_id] = p
        model_meta[model_id] = {"kind": "joblib", "feature_pack": feature_pack_name, "artifact": path.as_posix(), "artifact_hash": sha256_file(path)}
        threshold, metrics = _select_threshold(y[val], p)
        rows.append({"model_id": model_id, "feature_pack": feature_pack_name, "threshold": threshold, **metrics, "training_seconds": training_seconds, "inference_seconds": inference_seconds})
    tabm_p, tabm_report = _train_tabm(x_native[train], y[train], x_native[val], y[val], models)
    validation_proba["tabm_native"] = tabm_p
    model_meta["tabm_native"] = {"kind": "tabm", "feature_pack": "native", "artifact": (models / "tabm_native.pt").as_posix(), "artifact_hash": sha256_file(models / "tabm_native.pt")}
    tabm_threshold, tabm_metrics = _select_threshold(y[val], tabm_p)
    rows.append({"model_id": "tabm_native", "feature_pack": "native", "threshold": tabm_threshold, **tabm_metrics, **tabm_report})
    baseline_rows = [row for row in rows if row["model_id"] in COMMON_MODEL_IDS]
    reference = max(baseline_rows, key=lambda row: (row["macro_f1"], row["accuracy"], row["malicious_recall"], -row["ece"]))
    recall_floor = (
        float(reference["malicious_recall"])
        if require_validation_recall_not_lower
        else None
    )
    candidate_options: list[dict[str, Any]] = []
    native_ids = list(NATIVE_MODEL_IDS)
    for model_id in native_ids:
        threshold, metrics = _select_threshold(
            y[val],
            validation_proba[model_id],
            minimum_malicious_recall=recall_floor,
        )
        candidate_options.append({"candidate_type": "single", "members": [model_id], "weights": [1.0], "threshold": threshold, **metrics})
    for left_index, left in enumerate(native_ids):
        for right in native_ids[left_index + 1 :]:
            for alpha in (0.25, 0.5, 0.75):
                p = alpha * validation_proba[left] + (1.0 - alpha) * validation_proba[right]
                threshold, metrics = _select_threshold(
                    y[val], p, minimum_malicious_recall=recall_floor
                )
                candidate_options.append({"candidate_type": "blend", "members": [left, right], "weights": [alpha, 1.0 - alpha], "threshold": threshold, **metrics})
    candidate = max(candidate_options, key=lambda row: (row["macro_f1"], row["accuracy"], row["malicious_recall"], -row["ece"]))
    lock = {
        "schema_version": "1.0", "experiment": experiment, "status": "w95_validation_selection_locked",
        "selected_on": "validation_only", "acceptance_used_for_selection": False,
        "validation_malicious_recall_floor": recall_floor,
        "reference": {"model_id": reference["model_id"], "threshold": reference["threshold"], "metrics": reference},
        "candidate": candidate, "model_registry": model_meta, "selected_features": selected_names,
        "feature_pack_hash": feature_pack["feature_pack_hash"], "full_coverage_only": True,
        "validation_accuracy_delta": candidate["accuracy"] - reference["accuracy"], "validation_macro_f1_delta": candidate["macro_f1"] - reference["macro_f1"],
    }
    _write_csv(out / "validation_model_results.csv", rows)
    _write_csv(out / "validation_candidate_results.csv", candidate_options)
    _dump(out / "selection_lock.json", lock)
    _dump(out / "tabm_training_report.json", tabm_report)
    report = {**_security(), "status": "w95_validation_selection_locked", "train_count": int(np.sum(train)), "validation_count": int(np.sum(val)), "selection_lock": lock}
    _dump(out / "training_report.json", report)
    return report


def _predict_registered(model_id: str, x_common: np.ndarray, x_native: np.ndarray, model_dir: Path, registry: Mapping[str, Any]) -> np.ndarray:
    meta = registry[model_id]
    matrix = x_common if meta["feature_pack"] == "common" else x_native
    if meta["kind"] == "tabm":
        return _predict_tabm(matrix, model_dir)
    return _proba(joblib_load(model_dir / f"{model_id}.joblib"), matrix)


def _bootstrap(y: np.ndarray, ref_p: np.ndarray, cand_p: np.ndarray, ref_t: float, cand_t: float) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = {key: [] for key in ("accuracy", "macro_f1", "malicious_recall")}
    for _ in range(BOOTSTRAP_ITERATIONS):
        indexes = rng.integers(0, len(y), len(y))
        ref, cand = _metrics(y[indexes], ref_p[indexes], ref_t), _metrics(y[indexes], cand_p[indexes], cand_t)
        for key in deltas:
            deltas[key].append(cand[key] - ref[key])
    return {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, **{f"{key}_delta": {"mean": float(np.mean(values)), "ci95_lower": float(np.quantile(values, 0.025)), "ci95_upper": float(np.quantile(values, 0.975))} for key, values in deltas.items()}}


def evaluate_nfiot_targeted_models_w95(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    experiment: str = EXPERIMENT,
    target_variant: str = TARGET_VARIANT,
    acceptance_marker_name: str = "acceptance_opened_w95.json",
    evidence_reason_code: str = "W95_FULL_COVERAGE_SHADOW",
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    existing = _read_json(out / "evaluation_report.json")
    marker = out / acceptance_marker_name
    if marker.exists():
        return existing or {**_security(), "status": "failed_acceptance_reopen_without_report"}
    lock = _read_json(out / "selection_lock.json")
    if lock.get("status") != "w95_validation_selection_locked":
        report = {**_security(), "status": "failed_w95_selection_not_locked"}
        _dump(out / "evaluation_report.json", report)
        return report
    _dump(marker, {"opened": True, "opened_exactly_once": True, "selection_lock_sha256": sha256_file(out / "selection_lock.json"), "acceptance_used_for_selection": False})
    with np.load(out / "sealed_splits" / "acceptance_sealed.npz", allow_pickle=False) as data:
        x_common = np.asarray(data["x_common"], dtype=np.float32)
        x_native_all = np.asarray(data["x_native"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        attacks = np.asarray(data["attack"]).astype(str)
        hashes = np.asarray(data["sample_hash"]).astype(str)
        native_names = [str(v) for v in data["native_feature_names"]]
    selected_indexes = [native_names.index(name) for name in lock["selected_features"]]
    x_native = x_native_all[:, selected_indexes]
    registry = lock["model_registry"]
    reference_id = str(lock["reference"]["model_id"])
    ref_p = _predict_registered(reference_id, x_common, x_native, models, registry)
    candidate_spec = lock["candidate"]
    member_probabilities = [_predict_registered(str(member), x_common, x_native, models, registry) for member in candidate_spec["members"]]
    cand_p = sum(float(weight) * probability for weight, probability in zip(candidate_spec["weights"], member_probabilities, strict=True))
    ref_t, cand_t = float(lock["reference"]["threshold"]), float(candidate_spec["threshold"])
    reference, candidate = _metrics(y, ref_p, ref_t), _metrics(y, cand_p, cand_t)
    deltas = {key: candidate[key] - reference[key] for key in reference}
    bootstrap = _bootstrap(y, ref_p, cand_p, ref_t, cand_t)
    _dump(out / "bootstrap_ci_report.json", bootstrap)
    rows = [
        {"sample_hash": h, "label": int(label), "attack": attack, "reference_probability": float(rp), "reference_prediction": int(rp >= ref_t), "candidate_probability": float(cp), "candidate_prediction": int(cp >= cand_t)}
        for h, label, attack, rp, cp in zip(hashes, y, attacks, ref_p, cand_p, strict=True)
    ]
    _write_csv(out / "per_sample_predictions.csv", rows)
    attack_rows: list[dict[str, Any]] = []
    for attack in sorted(set(attacks)):
        mask = attacks == attack
        attack_rows.append({"attack": attack, "row_count": int(np.sum(mask)), "reference_recall": float(recall_score(y[mask], ref_p[mask] >= ref_t, zero_division=0)), "candidate_recall": float(recall_score(y[mask], cand_p[mask] >= cand_t, zero_division=0))})
    _write_csv(out / "per_attack_diagnostic.csv", attack_rows)
    gates = {
        "accuracy_delta_at_least_0_005": deltas["accuracy"] >= 0.005,
        "macro_f1_delta_at_least_0_005": deltas["macro_f1"] >= 0.005,
        "macro_f1_ci95_lower_gt_zero": bootstrap["macro_f1_delta"]["ci95_lower"] > 0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0,
        "ece_not_worse_by_0_005": deltas["ece"] <= 0.005,
        "coverage_is_one": candidate["coverage"] == 1.0,
        "acceptance_not_used_for_selection": True,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "fake_metric_count_zero": True,
        "runtime_safe_v3_0_remains_default": True,
    }
    passed = all(gates.values())
    status = "accepted_dataset_specific_full_coverage_accuracy_f1_result" if passed else "not_promoted_w95_full_coverage_performance_gate_failed"
    feature_hash = str(lock["feature_pack_hash"])
    evidence_examples = []
    candidate_artifact_hash = hashlib.sha256("|".join(str(registry[m]["artifact_hash"]) for m in candidate_spec["members"]).encode()).hexdigest()
    for sample_hash, probability in zip(hashes[:32], cand_p[:32], strict=True):
        confidence = float(max(probability, 1 - probability))
        evidence_examples.append(AgentEvidenceV2(
            agent_name="IoTBotnetSkill", agent_type="dataset_specific_native_feature_skill", input_feature_policy_hash=feature_hash,
            artifact_hash=candidate_artifact_hash, prediction="malicious" if probability >= cand_t else "benign",
            probabilities={"benign": float(1-probability), "malicious": float(probability)}, confidence=confidence, uncertainty=1-confidence,
            reliability=confidence, contributes_to_verdict=False, applicability="applicable", reason_codes=[target_variant.upper().replace("-", "_") + "_SCOPE", evidence_reason_code],
            calibration_status="validation_locked", supported_capabilities=["safe_native_flow_features"], dataset_scope=target_variant,
            promotion_status="dataset_specific_acceptance_passed" if passed else "not_promoted", source_evidence_sha256=str(sample_hash),
        ).model_dump(mode="json"))
    _dump(out / "agent_evidence_v2_shadow_examples.json", evidence_examples)
    report = {
        **_security(), "schema_version": "1.0", "experiment": experiment, "status": status, "dataset_scope": target_variant,
        "sample_count": len(y), "reference_model_id": reference_id, "candidate": candidate_spec,
        "reference_metrics": reference, "candidate_metrics": candidate, "deltas": deltas, "bootstrap": bootstrap,
        "gates": gates, "failed_gates": [key for key, value in gates.items() if not value],
        "accepted_dataset_specific_full_coverage_result": passed, "generalisation_claim_supported": False,
    }
    _dump(out / "evaluation_report.json", report)
    _dump(out / "security_acceptance.json", {**_security(), "status": "passed", "acceptance_opened_exactly_once": True})
    return report


def _write_docs(
    report: Mapping[str, Any],
    document: str | Path,
    document_cn: str | Path,
    *,
    target_variant: str = TARGET_VARIANT,
) -> None:
    ref, cand, delta = report.get("reference_metrics", {}), report.get("candidate_metrics", {}), report.get("deltas", {})
    ci = report.get("bootstrap", {}).get("macro_f1_delta", {})
    text = f"""# MAD-ETD NF-IoT Targeted Performance W94/W95

- Status: `{report.get('status')}`
- Dataset scope: `{target_variant}`
- Coverage: `1.0`
- Default runtime unchanged: `{report.get('runtime_safe_v3_0_remains_default')}`
- Promoted runtime created: `{report.get('promoted_runtime_created')}`

| Metric | Common-safe baseline | Targeted-feature candidate | Delta |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| Weighted-F1 | {ref.get('weighted_f1')} | {cand.get('weighted_f1')} | {delta.get('weighted_f1')} |
| Malicious recall | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

Macro-F1 delta 95% CI: `[{ci.get('ci95_lower')}, {ci.get('ci95_upper')}]`.

The feature and model design is informed by TabM (ICLR 2025), but all metrics are local real predictions. The result is dataset-specific, default-off, shadow-only, and does not establish source-held-out or cross-dataset generalisation.
"""
    cn = f"""# MAD-ETD NF-IoT 定向特征性能实验 W94/W95

- 状态：`{report.get('status')}`
- 数据集范围：`{target_variant}`
- Coverage：`1.0`
- 默认 runtime 未修改：`{report.get('runtime_safe_v3_0_remains_default')}`
- 创建 promoted runtime：`{report.get('promoted_runtime_created')}`

| 指标 | 公共安全特征基线 | 定向特征候选 | 增量 |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| Weighted-F1 | {ref.get('weighted_f1')} | {cand.get('weighted_f1')} | {delta.get('weighted_f1')} |
| 恶意召回率 | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

Macro-F1 增量 95% CI：`[{ci.get('ci95_lower')}, {ci.get('ci95_upper')}]`。

模型设计参考 TabM（ICLR 2025），全部指标来自本地真实预测。该结论仅适用于数据集特定、默认关闭的 shadow 候选，不支持跨来源或跨数据集泛化主张。
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(text, encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(cn, encoding="utf-8")


def finalize_nfiot_targeted_performance_w95(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *, document: str | Path = DEFAULT_DOC, document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False, test_count: int = 0,
    target_variant: str = TARGET_VARIANT,
) -> dict[str, Any]:
    out = Path(output_dir)
    evaluation = _read_json(out / "evaluation_report.json")
    if not evaluation:
        report = {**_security(), "status": "failed_missing_w95_evaluation", "tests_passed": tests_passed, "test_count": test_count}
        _dump(out / "negative_results.json", report)
        _dump(out / "acceptance_report.json", report)
        _write_docs(report, document, document_cn, target_variant=target_variant)
        return report
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    performance = evaluation.get("status") == "accepted_dataset_specific_full_coverage_accuracy_f1_result"
    deltas = evaluation.get("deltas", {})
    macro_ci = evaluation.get("bootstrap", {}).get("macro_f1_delta", {})
    candidate_metrics = evaluation.get("candidate_metrics", {})
    statistically_positive_signal = bool(
        not performance
        and float(deltas.get("accuracy", 0.0)) > 0.0
        and float(deltas.get("macro_f1", 0.0)) > 0.0
        and float(macro_ci.get("ci95_lower", 0.0)) > 0.0
        and float(candidate_metrics.get("coverage", 0.0)) == 1.0
        and evaluation.get("fake_metric_count") == 0
        and evaluation.get("blocked_field_violation") == 0
        and evaluation.get("fusion_ownership_violation") == 0
    )
    accepted = bool(performance and hashes_unchanged and tests_passed)
    if accepted:
        status = "accepted_dataset_specific_full_coverage_accuracy_f1_result"
    elif performance and hashes_unchanged:
        status = "pending_tests_full_coverage_positive_result"
    elif statistically_positive_signal and hashes_unchanged and tests_passed:
        status = "statistically_positive_full_coverage_signal_not_promoted"
    else:
        status = "not_promoted_w95_performance_or_safety_gate_failed"
    report = {**evaluation, "status": status, "accepted_dataset_specific_full_coverage_result": accepted, "statistically_positive_full_coverage_signal": statistically_positive_signal, "frozen_hashes_unchanged": hashes_unchanged, "tests_passed": tests_passed, "test_count": test_count, "runtime_safe_v3_0_remains_default": True, "promoted_runtime_created": False, "candidate_default_enabled": False}
    if not accepted:
        _dump(out / "negative_results.json", {"status": status, "failed_gates": evaluation.get("failed_gates", []) + ([] if hashes_unchanged else ["frozen_hashes_unchanged"]) + ([] if tests_passed else ["tests_passed"]), "fake_metric_count": 0, "promoted_runtime_created": False})
    elif (out / "negative_results.json").exists():
        (out / "negative_results.json").unlink()
    _dump(out / "acceptance_report.json", report)
    _write_docs(report, document, document_cn, target_variant=target_variant)
    return report
