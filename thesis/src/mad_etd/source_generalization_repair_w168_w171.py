"""W168-W171 source-generalization repair for the NF-BoT multiclass lane.

The target-domain development partition is label-blind. Endpoint identities are
used only to create irreversible group hashes and never enter DetectorInput.
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
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .base import CaseState
from .evidence_team import AgentEvidenceV2Adapter, EvidenceRequestGuard
from .external_multiagent_positive_w160_w167 import (
    BLOCKED_FIELDS,
    ENGINEERED_FEATURES,
    NF_TASKS,
    RAW_SAFE_FEATURES,
    SEEDS,
    _choose_temperature,
    _dump,
    _fit_family,
    _manifest_entries,
    _metrics,
    _normalise_attack,
    _numeric_frame,
    _predict_artifact,
    _read_json,
    _security,
    _sha256_json,
    _take_smallest,
    _temperature_scale,
    _write_csv,
)
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


EXPERIMENT = "mad_etd_source_generalization_repair_w168_w171"
DEFAULT_PREVIOUS_DIR = Path("data/runs/mad_etd_external_multiagent_positive_w160_w167")
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_source_generalization_repair_w168_w171")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_source_generalization_repair_w169")
DEFAULT_RELEASE_DIR = Path("data/releases/mad_etd_source_generalization_repair_w171")
DATASET = "NF-BoT-IoT"
CLASSES = tuple(NF_TASKS[DATASET]["classes"])
SOURCE_TRAIN_PER_CLASS = 3_000
SOURCE_VALIDATION_PER_CLASS = 500
TARGET_DEVELOPMENT_PER_CLASS = 5_000
TARGET_ACCEPTANCE_PER_CLASS = 5_000
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 168

METHODS = (
    "baseline_hgb",
    "baseline_oof_stacking",
    "dual_quantile_hgb",
    "dual_quantile_extra_trees",
    "dual_quantile_oof_stacking",
    "coral_target_to_source_hgb",
    "stable_dual_quantile_hgb",
)

# W169.1 is a separately recorded development iteration.  It was declared only
# after W169 failed its label-blind selection gate and before the sealed W170
# acceptance partition was opened.  The gate itself is intentionally unchanged.
W169_1_METHODS = (
    "coral_target_to_source_oof_stacking",
)


def _read_npz_rows(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    data = np.load(path, allow_pickle=False)
    values: set[int] = set()
    for key in ("row_train", "row_validation", "row_acceptance"):
        if key in data:
            values.update(int(item) for item in data[key])
    return values


def _historical_exclusions(previous: Path) -> tuple[set[int], set[int], dict[str, Any]]:
    source = _read_npz_rows(previous / "nf_bot_iot_group_protocol_w161.npz")
    target = _read_npz_rows(previous / "nf_bot_iot_paper_protocol_w161.npz")
    target.update(_read_npz_rows(previous / "nf_bot_iot_group_protocol_w161.npz"))
    prior = _read_json(previous / "w163_acceptance_report.json")
    ready = (
        prior.get("acceptance_opened_once") is True
        and bool(source)
        and bool(target)
    )
    return source, target, {
        "status": "w161_w163_rows_mapped_for_exclusion" if ready else "failed_historical_exclusion_mapping",
        "source_v1_exclusion_count": len(source),
        "target_v2_exclusion_count": len(target),
        "prior_acceptance_status": prior.get("status", "missing"),
        "fake_metric_count": 0,
    }


def _endpoint_group(src: Any, dst: Any) -> str:
    pair = sorted((str(src).strip(), str(dst).strip()))
    return hashlib.sha256(("w168:endpoint:" + "|".join(pair)).encode()).hexdigest()


def _role_from_group(group_hash: str) -> str:
    value = int(hashlib.sha256(("w168:target-role:" + group_hash).encode()).hexdigest()[:16], 16)
    return "target_development" if value % 2 == 0 else "target_acceptance"


def _push(
    buckets: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None],
    key: tuple[str, str],
    x: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
    priorities: np.ndarray,
    limit: int,
) -> None:
    existing = buckets.get(key)
    if existing is None:
        merged_x, merged_rows, merged_groups, merged_p = x, rows, groups, priorities
    else:
        merged_x = np.concatenate([existing[0], x])
        merged_rows = np.concatenate([existing[1], rows])
        merged_groups = np.concatenate([existing[2], groups])
        merged_p = np.concatenate([existing[3], priorities])
    if len(merged_p) > limit:
        keep = np.argpartition(merged_p, limit - 1)[:limit]
        merged_x, merged_rows, merged_groups, merged_p = (
            merged_x[keep], merged_rows[keep], merged_groups[keep], merged_p[keep]
        )
    buckets[key] = merged_x, merged_rows, merged_groups, merged_p


def _extract_fresh(
    entry: Mapping[str, Any],
    exclusions: set[int],
    *,
    target: bool,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    archive_path = Path(str(entry["archive"]))
    member = str(entry["entry"])
    attack_column = str(entry.get("attack_column") or "Attack")
    allowed = {_normalise_attack(name): name for name in CLASSES}
    buckets: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = {}
    observed: defaultdict[str, int] = defaultdict(int)
    excluded_history = 0
    rng = np.random.default_rng(168 if not target else 169)
    offset = 0
    exclusion_array = np.fromiter(exclusions, dtype=np.int64) if exclusions else np.empty((0,), dtype=np.int64)
    usecols = [*RAW_SAFE_FEATURES, attack_column, "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=usecols, chunksize=250_000, low_memory=False):
            attacks = chunk[attack_column].astype(str).map(_normalise_attack)
            rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            history_mask = np.isin(rows, exclusion_array) if len(exclusion_array) else np.zeros(len(rows), dtype=bool)
            excluded_history += int(history_mask.sum())
            for normalised, canonical in allowed.items():
                mask = attacks.eq(normalised).to_numpy() & ~history_mask
                count = int(mask.sum())
                if not count:
                    continue
                observed[canonical] += count
                selected_frame = chunk.loc[mask]
                x = _numeric_frame(selected_frame.loc[:, list(RAW_SAFE_FEATURES)])
                selected_rows = rows[mask]
                groups = np.asarray(
                    [
                        _endpoint_group(src, dst)
                        for src, dst in zip(
                            selected_frame["IPV4_SRC_ADDR"],
                            selected_frame["IPV4_DST_ADDR"],
                        )
                    ],
                    dtype="U64",
                )
                priorities = rng.random(count)
                if target:
                    roles = np.asarray([_role_from_group(value) for value in groups], dtype="U32")
                    role_limits = {
                        "target_development": TARGET_DEVELOPMENT_PER_CLASS,
                        "target_acceptance": TARGET_ACCEPTANCE_PER_CLASS,
                    }
                    for role, limit in role_limits.items():
                        role_mask = roles == role
                        if role_mask.any():
                            _push(
                                buckets, (canonical, role), x[role_mask],
                                selected_rows[role_mask], groups[role_mask],
                                priorities[role_mask], limit,
                            )
                else:
                    _push(
                        buckets, (canonical, "source_pool"), x, selected_rows,
                        groups, priorities,
                        SOURCE_TRAIN_PER_CLASS + SOURCE_VALIDATION_PER_CLASS,
                    )
    materialized: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, values in buckets.items():
        if values is None:
            continue
        order = np.argsort(values[3])
        materialized[key] = values[0][order], values[1][order], values[2][order]
    return materialized, {
        "archive": archive_path.as_posix(),
        "member": member,
        "scanned_rows": offset,
        "historical_rows_excluded": excluded_history,
        "observed_fresh_counts": dict(observed),
        "bucket_counts": {f"{key[0]}::{key[1]}": len(value[0]) for key, value in materialized.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }


def _concat_role(
    buckets: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    role: str,
    *,
    source_pool_slice: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for label, name in enumerate(CLASSES):
        key = (name, role)
        values = buckets[key]
        x, rows, groups = values
        if source_pool_slice is not None:
            start, stop = source_pool_slice
            x, rows, groups = x[start:stop], rows[start:stop], groups[start:stop]
        x_parts.append(x)
        y_parts.append(np.full(len(x), label, dtype=np.int64))
        row_parts.append(rows)
        group_parts.append(groups)
    return (
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(row_parts),
        np.concatenate(group_parts),
    )


def _single_feature_auc(source: np.ndarray, target: np.ndarray) -> float:
    values = np.concatenate([source, target])
    labels = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    if np.all(values == values[0]):
        return 0.5
    auc = float(roc_auc_score(labels, values))
    return max(auc, 1.0 - auc)


def _source_auc(source: np.ndarray, target: np.ndarray, seed: int = 168) -> float:
    rng = np.random.default_rng(seed)
    n = min(8_000, len(source), len(target))
    source_index = rng.choice(len(source), size=n, replace=False)
    target_index = rng.choice(len(target), size=n, replace=False)
    x = np.concatenate([source[source_index], target[target_index]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(y))
    cut = int(len(y) * 0.70)
    train, test = order[:cut], order[cut:]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(x[train], y[train])
    return float(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))


def build_source_generalization_repair_w168(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    previous_dir: str | Path = DEFAULT_PREVIOUS_DIR,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    previous = Path(previous_dir)
    source_exclusions, target_exclusions, history = _historical_exclusions(previous)
    _dump(out / "historical_exclusion_audit_w168.json", history)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before_w168.json", before)
    entries = _manifest_entries(Path(processed_dir))
    required = {DATASET, f"{DATASET}-v2"}
    if history.get("status") != "w161_w163_rows_mapped_for_exclusion" or not required.issubset(entries):
        report = {"status": "failed_w168_history_or_source_gate", "missing_sources": sorted(required - set(entries)), **_security()}
        _dump(out / "w168_acceptance_report.json", report)
        return report
    source_buckets, source_audit = _extract_fresh(entries[DATASET], source_exclusions, target=False)
    target_buckets, target_audit = _extract_fresh(entries[f"{DATASET}-v2"], target_exclusions, target=True)
    _dump(out / "fresh_extraction_audit_w168.json", {"source": source_audit, "target": target_audit})
    missing = []
    for name in CLASSES:
        if len(source_buckets.get((name, "source_pool"), ([], [], []))[0]) < SOURCE_TRAIN_PER_CLASS + SOURCE_VALIDATION_PER_CLASS:
            missing.append(f"source:{name}")
        for role, limit in (("target_development", TARGET_DEVELOPMENT_PER_CLASS), ("target_acceptance", TARGET_ACCEPTANCE_PER_CLASS)):
            if len(target_buckets.get((name, role), ([], [], []))[0]) < limit:
                missing.append(f"{role}:{name}")
    if missing:
        report = {"status": "failed_insufficient_fresh_endpoint_group_samples", "missing": missing, **_security()}
        _dump(out / "w168_acceptance_report.json", report)
        return report
    source_train = _concat_role(source_buckets, "source_pool", source_pool_slice=(0, SOURCE_TRAIN_PER_CLASS))
    source_validation = _concat_role(source_buckets, "source_pool", source_pool_slice=(SOURCE_TRAIN_PER_CLASS, SOURCE_TRAIN_PER_CLASS + SOURCE_VALIDATION_PER_CLASS))
    target_development = _concat_role(target_buckets, "target_development")
    target_acceptance = _concat_role(target_buckets, "target_acceptance")
    dev_groups = set(target_development[3].tolist())
    acceptance_groups = set(target_acceptance[3].tolist())
    group_overlap = len(dev_groups & acceptance_groups)
    np.savez_compressed(out / "source_train_w168.npz", x=source_train[0], y=source_train[1], rows=source_train[2], groups=source_train[3], classes=np.asarray(CLASSES, dtype="U64"), feature_names=np.asarray(ENGINEERED_FEATURES, dtype="U64"))
    np.savez_compressed(out / "source_validation_w168.npz", x=source_validation[0], y=source_validation[1], rows=source_validation[2], groups=source_validation[3])
    # Target-development labels are intentionally omitted from the artifact.
    np.savez_compressed(out / "target_development_unlabeled_w168.npz", x=target_development[0], rows=target_development[2], groups=target_development[3])
    np.savez_compressed(out / "target_acceptance_sealed_w168.npz", x=target_acceptance[0], y=target_acceptance[1], rows=target_acceptance[2], groups=target_acceptance[3])
    sealed_hash = hashlib.sha256((out / "target_acceptance_sealed_w168.npz").read_bytes()).hexdigest()
    _dump(out / "sealed_acceptance_label_commitment_w168.json", {"sha256": sealed_hash, "opened": False, "sample_count": len(target_acceptance[1]), "group_count": len(acceptance_groups)})
    manifest_rows: list[dict[str, Any]] = []
    for split, values, labels_visible in (
        ("source_train", source_train, True),
        ("source_validation", source_validation, True),
        ("target_development_unlabeled", target_development, False),
        ("target_acceptance_sealed", target_acceptance, False),
    ):
        for index, (row, group) in enumerate(zip(values[2], values[3])):
            label_value = CLASSES[int(values[1][index])] if labels_visible else "sealed"
            manifest_rows.append({
                "sample_hash": hashlib.sha256(f"w168:{split}:{int(row)}".encode()).hexdigest(),
                "split": split,
                "class_name": label_value,
                "endpoint_group_hash": group,
                "endpoint_identity_in_feature_matrix": False,
                "source_variant_in_feature_matrix": False,
            })
    _write_csv(out / "fresh_split_manifest_w168.csv", manifest_rows)
    x_source, y_source = source_train[0], source_train[1]
    x_target = target_development[0]
    mi = mutual_info_classif(x_source, y_source, random_state=168)
    records = []
    for index, name in enumerate(ENGINEERED_FEATURES):
        records.append({
            "feature_name": name,
            "source_predictability_auc": _single_feature_auc(x_source[:, index], x_target[:, index]),
            "ks_statistic": float(ks_2samp(x_source[:, index], x_target[:, index]).statistic),
            "wasserstein_distance": float(wasserstein_distance(x_source[:, index], x_target[:, index])),
            "source_train_label_mutual_information": float(mi[index]),
            "target_labels_used": False,
        })
    mi_floor = float(np.quantile(mi, 0.25))
    eligible = [row for row in records if row["source_predictability_auc"] <= 0.75 and row["source_train_label_mutual_information"] >= mi_floor]
    if len(eligible) < 4:
        eligible = sorted(records, key=lambda row: (row["source_predictability_auc"], -row["source_train_label_mutual_information"]))[: max(4, min(8, len(records)))]
    stable_features = [row["feature_name"] for row in sorted(eligible, key=lambda row: (-row["source_train_label_mutual_information"], row["source_predictability_auc"]))]
    _write_csv(out / "source_shift_feature_forensics_w168.csv", records)
    source_auc = _source_auc(x_source, x_target)
    policy = {
        "status": "source_invariant_feature_policy_locked",
        "all_safe_features": list(ENGINEERED_FEATURES),
        "stable_candidate_features": stable_features,
        "stable_feature_indices": [ENGINEERED_FEATURES.index(name) for name in stable_features],
        "blocked_fields": list(BLOCKED_FIELDS),
        "endpoint_usage": "split-only hashed group",
        "target_development_label_usage": "prohibited_and_not_materialized",
        "target_acceptance_label_usage": "sealed_until_w170",
        "feature_policy_hash": _sha256_json({"features": ENGINEERED_FEATURES, "stable": stable_features, "blocked": BLOCKED_FIELDS}),
        **_security(),
    }
    _dump(out / "source_invariant_feature_policy_w168.json", policy)
    nf_ton_blocker = {
        "status": "failed_no_fresh_nf_ton_nine_class_acceptance",
        "reason": "NF-ToN-IoT-v2 contains 7,723 MITM rows and W161 froze 7,700; only 23 unused rows remain, so a fresh credible 9-class acceptance cannot be constructed.",
        "fake_metric_count": 0,
    }
    _dump(out / "nf_ton_fresh_protocol_blocker_w168.json", nf_ton_blocker)
    report = {
        "status": "fresh_nf_bot_source_generalization_protocol_frozen" if group_overlap == 0 else "failed_target_endpoint_group_overlap",
        "source_train_count": len(source_train[1]),
        "source_validation_count": len(source_validation[1]),
        "target_development_unlabeled_count": len(target_development[0]),
        "target_acceptance_sealed_count": len(target_acceptance[0]),
        "target_development_group_count": len(dev_groups),
        "target_acceptance_group_count": len(acceptance_groups),
        "target_group_overlap": group_overlap,
        "baseline_source_predictability_auc": source_auc,
        "stable_feature_count": len(stable_features),
        "target_development_labels_materialized": False,
        "target_acceptance_opened": False,
        "historical_sample_overlap": 0,
        **_security(),
    }
    _dump(out / "w168_acceptance_report.json", report)
    return report


def _fit_quantile(x: np.ndarray, seed: int) -> QuantileTransformer:
    transformer = QuantileTransformer(
        n_quantiles=min(1_000, len(x)),
        output_distribution="normal",
        subsample=min(100_000, len(x)),
        random_state=seed,
    )
    transformer.fit(x)
    return transformer


def _sqrtm_psd(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-6, None)
    diagonal = np.diag(1.0 / np.sqrt(values) if inverse else np.sqrt(values))
    return vectors @ diagonal @ vectors.T


def _coral_target_to_source(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = np.cov(source, rowvar=False) + np.eye(source.shape[1]) * 1e-4
    target_cov = np.cov(target, rowvar=False) + np.eye(target.shape[1]) * 1e-4
    matrix = _sqrtm_psd(target_cov, inverse=True) @ _sqrtm_psd(source_cov)
    return {"source_mean": source_mean, "target_mean": target_mean, "matrix": matrix}


def _apply_representation(artifact: Mapping[str, Any], x: np.ndarray, *, domain: str) -> np.ndarray:
    indices = np.asarray(artifact.get("feature_indices", np.arange(x.shape[1])), dtype=int)
    values = x[:, indices]
    mode = artifact["representation"]
    if mode == "identity":
        return values
    if mode == "dual_quantile":
        transformer = artifact["source_transformer"] if domain == "source" else artifact["target_transformer"]
        return transformer.transform(values).astype(np.float32)
    if mode == "coral_target_to_source":
        if domain == "source":
            return values
        return ((values - artifact["target_mean"]) @ artifact["coral_matrix"] + artifact["source_mean"]).astype(np.float32)
    raise ValueError(f"unknown representation: {mode}")


def _fit_method(
    method: str,
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    classes: np.ndarray,
    stable_indices: list[int],
    seed: int,
) -> dict[str, Any]:
    indices = stable_indices if method == "stable_dual_quantile_hgb" else list(range(x_source.shape[1]))
    source = x_source[:, indices]
    target = x_target[:, indices]
    if method.startswith("baseline_"):
        representation = "identity"
        transformed = source
        metadata: dict[str, Any] = {}
    elif method.startswith("dual_quantile") or method == "stable_dual_quantile_hgb":
        representation = "dual_quantile"
        source_transformer = _fit_quantile(source, seed)
        target_transformer = _fit_quantile(target, seed + 1_000)
        transformed = source_transformer.transform(source).astype(np.float32)
        metadata = {"source_transformer": source_transformer, "target_transformer": target_transformer}
    elif method.startswith("coral_target_to_source_"):
        representation = "coral_target_to_source"
        coral = _coral_target_to_source(source, target)
        transformed = source
        metadata = {"source_mean": coral["source_mean"], "target_mean": coral["target_mean"], "coral_matrix": coral["matrix"]}
    else:
        raise ValueError(method)
    if method.endswith("oof_stacking"):
        family = "oof_stacking"
    elif method.endswith("extra_trees"):
        family = "extra_trees"
    else:
        family = "hist_gradient_boosting"
    model = _fit_family(family, transformed, y_source, classes, seed)
    return {
        "method": method,
        "representation": representation,
        "feature_indices": indices,
        "model": model,
        "classes": classes,
        **metadata,
    }


def _predict_method(artifact: Mapping[str, Any], x: np.ndarray, *, domain: str) -> np.ndarray:
    representation = _apply_representation(artifact, x, domain=domain)
    return _predict_artifact(artifact["model"], representation)


def train_source_invariant_models_w169(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    report168 = _read_json(out / "w168_acceptance_report.json")
    if report168.get("status") != "fresh_nf_bot_source_generalization_protocol_frozen":
        report = {"status": "failed_missing_w168_fresh_protocol", **_security()}
        _dump(out / "w169_acceptance_report.json", report)
        return report
    train = np.load(out / "source_train_w168.npz", allow_pickle=False)
    validation = np.load(out / "source_validation_w168.npz", allow_pickle=False)
    target = np.load(out / "target_development_unlabeled_w168.npz", allow_pickle=False)
    classes = train["classes"]
    policy = _read_json(out / "source_invariant_feature_policy_w168.json")
    stable_indices = [int(value) for value in policy["stable_feature_indices"]]
    models = Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            started = time.perf_counter()
            artifact = _fit_method(method, train["x"], train["y"], target["x"], classes, stable_indices, seed)
            raw_validation = _predict_method(artifact, validation["x"], domain="source")
            temperature = _choose_temperature(validation["y"], raw_validation)
            artifact["temperature"] = temperature
            artifact["feature_policy_hash"] = policy["feature_policy_hash"]
            source_repr = _apply_representation(artifact, train["x"], domain="source")
            target_repr = _apply_representation(artifact, target["x"], domain="target")
            source_auc = _source_auc(source_repr, target_repr, seed)
            probabilities = _temperature_scale(raw_validation, temperature)
            artifact_path = models / f"{method}__seed{seed}.joblib"
            joblib_dump(artifact, artifact_path, compress=3)
            rows.append({
                "method": method,
                "seed": seed,
                "representation": artifact["representation"],
                "feature_count": len(artifact["feature_indices"]),
                "source_predictability_auc": source_auc,
                "temperature": temperature,
                "training_seconds": time.perf_counter() - started,
                "target_development_labels_used": False,
                **_metrics(validation["y"], probabilities),
            })
    _write_csv(out / "source_invariant_validation_results_w169.csv", rows)
    summaries = []
    for method in METHODS:
        values = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "validation_macro_f1_mean": float(np.mean([row["macro_f1"] for row in values])),
            "validation_macro_recall_mean": float(np.mean([row["macro_recall"] for row in values])),
            "validation_ece_mean": float(np.mean([row["ece"] for row in values])),
            "source_predictability_auc_mean": float(np.mean([row["source_predictability_auc"] for row in values])),
        })
    baselines = [row for row in summaries if row["method"].startswith("baseline_")]
    baselines.sort(key=lambda row: (-row["validation_macro_f1_mean"], row["validation_ece_mean"], row["method"]))
    baseline = baselines[0]
    candidates = [
        row for row in summaries
        if not row["method"].startswith("baseline_")
        and row["validation_macro_f1_mean"] >= baseline["validation_macro_f1_mean"] - 0.01
        and row["source_predictability_auc_mean"] <= baseline["source_predictability_auc_mean"] - 0.05
    ]
    candidates.sort(key=lambda row: (row["source_predictability_auc_mean"], -row["validation_macro_f1_mean"], row["validation_ece_mean"], row["method"]))
    selected = candidates[0] if candidates else None
    selection = {
        "status": "source_invariant_candidate_locked" if selected else "no_candidate_passed_label_blind_selection_gate",
        "baseline_reference": baseline,
        "selected_candidate": selected,
        "all_method_summaries": summaries,
        "selection_inputs": ["source validation labels", "unlabeled target development distribution"],
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "selection_hash": _sha256_json({"baseline": baseline, "selected": selected}),
        **_security(),
    }
    _dump(out / "source_invariant_selection_w169.json", selection)
    report = {
        "status": selection["status"],
        "trained_model_count": len(rows),
        "method_count": len(METHODS),
        "seed_count": len(SEEDS),
        "baseline_method": baseline["method"],
        "selected_candidate_method": selected["method"] if selected else None,
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        **_security(),
    }
    _dump(out / "w169_acceptance_report.json", report)
    return report


def train_source_invariant_models_w169_1(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Run the pre-acceptance CORAL + OOF-stacking repair iteration.

    This function never reads target-development labels (the development NPZ
    intentionally has no ``y`` array) and never opens the sealed W170 labels.
    It preserves the original W169 result and writes distinct W169.1 artifacts.
    """

    out = Path(output_dir)
    report168 = _read_json(out / "w168_acceptance_report.json")
    original_selection = _read_json(out / "source_invariant_selection_w169.json")
    if report168.get("status") != "fresh_nf_bot_source_generalization_protocol_frozen":
        report = {"status": "failed_missing_w168_fresh_protocol", **_security()}
        _dump(out / "w169_1_acceptance_report.json", report)
        return report
    if original_selection.get("status") != "no_candidate_passed_label_blind_selection_gate":
        report = {"status": "failed_w169_1_requires_recorded_w169_gate_failure", **_security()}
        _dump(out / "w169_1_acceptance_report.json", report)
        return report
    if (out / "w170_acceptance_opened.json").is_file():
        report = {"status": "failed_w169_1_acceptance_already_opened", **_security()}
        _dump(out / "w169_1_acceptance_report.json", report)
        return report

    train = np.load(out / "source_train_w168.npz", allow_pickle=False)
    validation = np.load(out / "source_validation_w168.npz", allow_pickle=False)
    target = np.load(out / "target_development_unlabeled_w168.npz", allow_pickle=False)
    if "y" in target.files:
        report = {"status": "failed_target_development_labels_materialized", **_security()}
        _dump(out / "w169_1_acceptance_report.json", report)
        return report

    classes = train["classes"]
    policy = _read_json(out / "source_invariant_feature_policy_w168.json")
    stable_indices = [int(value) for value in policy["stable_feature_indices"]]
    models = Path(model_dir)
    models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for method in W169_1_METHODS:
        for seed in SEEDS:
            started = time.perf_counter()
            artifact = _fit_method(method, train["x"], train["y"], target["x"], classes, stable_indices, seed)
            raw_validation = _predict_method(artifact, validation["x"], domain="source")
            temperature = _choose_temperature(validation["y"], raw_validation)
            artifact["temperature"] = temperature
            artifact["feature_policy_hash"] = policy["feature_policy_hash"]
            source_repr = _apply_representation(artifact, train["x"], domain="source")
            target_repr = _apply_representation(artifact, target["x"], domain="target")
            source_auc = _source_auc(source_repr, target_repr, seed)
            probabilities = _temperature_scale(raw_validation, temperature)
            joblib_dump(artifact, models / f"{method}__seed{seed}.joblib", compress=3)
            rows.append({
                "method": method,
                "seed": seed,
                "representation": artifact["representation"],
                "feature_count": len(artifact["feature_indices"]),
                "source_predictability_auc": source_auc,
                "temperature": temperature,
                "training_seconds": time.perf_counter() - started,
                "target_development_labels_used": False,
                **_metrics(validation["y"], probabilities),
            })
    _write_csv(out / "source_invariant_validation_results_w169_1.csv", rows)

    summaries: list[dict[str, Any]] = []
    for method in W169_1_METHODS:
        values = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "validation_macro_f1_mean": float(np.mean([row["macro_f1"] for row in values])),
            "validation_macro_recall_mean": float(np.mean([row["macro_recall"] for row in values])),
            "validation_ece_mean": float(np.mean([row["ece"] for row in values])),
            "source_predictability_auc_mean": float(np.mean([row["source_predictability_auc"] for row in values])),
        })
    baseline = original_selection["baseline_reference"]
    eligible = [
        row for row in summaries
        if row["validation_macro_f1_mean"] >= baseline["validation_macro_f1_mean"] - 0.01
        and row["source_predictability_auc_mean"] <= baseline["source_predictability_auc_mean"] - 0.05
    ]
    eligible.sort(key=lambda row: (row["source_predictability_auc_mean"], -row["validation_macro_f1_mean"], row["validation_ece_mean"], row["method"]))
    selected = eligible[0] if eligible else None
    selection = {
        "status": "source_invariant_candidate_locked" if selected else "no_candidate_passed_label_blind_selection_gate",
        "development_round": "W169.1",
        "pre_acceptance_iteration": True,
        "original_w169_status": original_selection["status"],
        "selection_gate_unchanged": True,
        "baseline_reference": baseline,
        "selected_candidate": selected,
        "all_method_summaries": summaries,
        "selection_inputs": ["source validation labels", "unlabeled target development distribution"],
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
        "selection_hash": _sha256_json({"round": "W169.1", "baseline": baseline, "selected": selected}),
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "automatic_deployment": False,
        **_security(),
    }
    _dump(out / "source_invariant_selection_w169_1.json", selection)
    report = {
        "status": selection["status"],
        "development_round": "W169.1",
        "trained_model_count": len(rows),
        "method_count": len(W169_1_METHODS),
        "seed_count": len(SEEDS),
        "baseline_method": baseline["method"],
        "selected_candidate_method": selected["method"] if selected else None,
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "selection_gate_unchanged": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        **_security(),
    }
    _dump(out / "w169_1_acceptance_report.json", report)
    return report


def _per_class_recall(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    values = []
    for label in range(len(CLASSES)):
        mask = y == label
        values.append(float((prediction[mask] == label).mean()) if mask.any() else float("nan"))
    return np.asarray(values)


def _grouped_bootstrap_delta(
    y: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        base = _metrics(y[indices], baseline_probabilities[indices])
        candidate = _metrics(y[indices], candidate_probabilities[indices])
        macro_deltas.append(candidate["macro_f1"] - base["macro_f1"])
        accuracy_deltas.append(candidate["accuracy"] - base["accuracy"])
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "group_count": len(unique_groups),
        "macro_f1_delta": {
            "mean": float(np.mean(macro_deltas)),
            "ci95_lower": float(np.quantile(macro_deltas, 0.025)),
            "ci95_upper": float(np.quantile(macro_deltas, 0.975)),
        },
        "accuracy_delta": {
            "mean": float(np.mean(accuracy_deltas)),
            "ci95_lower": float(np.quantile(accuracy_deltas, 0.025)),
            "ci95_upper": float(np.quantile(accuracy_deltas, 0.975)),
        },
    }


def evaluate_source_generalization_acceptance_w170(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    marker = out / "w170_acceptance_opened.json"
    if marker.is_file():
        return {**_read_json(out / "w170_acceptance_report.json"), "rerun_refused_acceptance_already_opened": True}
    selection_path = out / "source_invariant_selection_w169_1.json"
    if not selection_path.is_file():
        selection_path = out / "source_invariant_selection_w169.json"
    selection = _read_json(selection_path)
    if selection.get("status") != "source_invariant_candidate_locked":
        report = {"status": "not_run_no_w169_candidate", **_security()}
        _dump(out / "w170_acceptance_report.json", report)
        return report
    baseline_method = selection["baseline_reference"]["method"]
    candidate_method = selection["selected_candidate"]["method"]
    _dump(marker, {"opened_once": True, "selection_hash": selection["selection_hash"], "selection_artifact": selection_path.name, "opened_at_unix": time.time()})
    sealed = np.load(out / "target_acceptance_sealed_w168.npz", allow_pickle=False)
    validation = np.load(out / "source_validation_w168.npz", allow_pickle=False)
    root = Path(model_dir)
    probabilities: dict[str, list[np.ndarray]] = {baseline_method: [], candidate_method: []}
    validation_metrics: dict[str, list[dict[str, float]]] = {baseline_method: [], candidate_method: []}
    for method in (baseline_method, candidate_method):
        for seed in SEEDS:
            artifact = joblib_load(root / f"{method}__seed{seed}.joblib")
            raw = _predict_method(artifact, sealed["x"], domain="target")
            probabilities[method].append(_temperature_scale(raw, float(artifact["temperature"])))
            validation_raw = _predict_method(artifact, validation["x"], domain="source")
            validation_metrics[method].append(_metrics(validation["y"], _temperature_scale(validation_raw, float(artifact["temperature"]))))
    baseline_prob = np.mean(probabilities[baseline_method], axis=0)
    candidate_prob = np.mean(probabilities[candidate_method], axis=0)
    baseline_metrics = _metrics(sealed["y"], baseline_prob)
    candidate_metrics = _metrics(sealed["y"], candidate_prob)
    baseline_prediction = baseline_prob.argmax(axis=1)
    candidate_prediction = candidate_prob.argmax(axis=1)
    baseline_recall = _per_class_recall(sealed["y"], baseline_prediction)
    candidate_recall = _per_class_recall(sealed["y"], candidate_prediction)
    class_rows = [
        {
            "class_name": name,
            "support": int((sealed["y"] == label).sum()),
            "baseline_recall": baseline_recall[label],
            "candidate_recall": candidate_recall[label],
            "recall_delta": candidate_recall[label] - baseline_recall[label],
        }
        for label, name in enumerate(CLASSES)
    ]
    _write_csv(out / "source_generalization_per_class_w170.csv", class_rows)
    bootstrap = _grouped_bootstrap_delta(sealed["y"], baseline_prob, candidate_prob, sealed["groups"])
    _dump(out / "source_generalization_grouped_bootstrap_w170.json", bootstrap)
    validation_base = float(np.mean([row["macro_f1"] for row in validation_metrics[baseline_method]]))
    validation_candidate = float(np.mean([row["macro_f1"] for row in validation_metrics[candidate_method]]))
    comparison = {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": {key: candidate_metrics[key] - baseline_metrics[key] for key in baseline_metrics},
        "source_validation_macro_f1": {"baseline": validation_base, "candidate": validation_candidate, "delta": validation_candidate - validation_base},
        "worst_class_recall": {"baseline": float(np.nanmin(baseline_recall)), "candidate": float(np.nanmin(candidate_recall)), "delta": float(np.nanmin(candidate_recall) - np.nanmin(baseline_recall))},
    }
    _dump(out / "source_generalization_comparison_w170.json", comparison)
    gates = {
        "macro_f1_delta_at_least_0_05": comparison["delta"]["macro_f1"] >= 0.05,
        "macro_f1_delta_ci95_lower_gt_0": bootstrap["macro_f1_delta"]["ci95_lower"] > 0.0,
        "source_validation_macro_f1_drop_le_0_01": validation_candidate >= validation_base - 0.01,
        "worst_class_recall_not_worse": comparison["worst_class_recall"]["candidate"] >= comparison["worst_class_recall"]["baseline"],
        "ece_not_worse_by_more_than_0_01": candidate_metrics["ece"] <= baseline_metrics["ece"] + 0.01,
        "endpoint_group_count_positive": bootstrap["group_count"] > 0,
    }
    accepted = all(gates.values())
    commitment = _read_json(out / "sealed_acceptance_label_commitment_w168.json")
    _dump(out / "sealed_acceptance_label_commitment_w170.json", {**commitment, "opened": True, "opened_once": True, "sha256_verified": commitment.get("sha256") == hashlib.sha256((out / "target_acceptance_sealed_w168.npz").read_bytes()).hexdigest()})
    report = {
        "status": "accepted_optional_source_generalization_candidate" if accepted else "not_promoted_source_generalization_gate_failed",
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "macro_f1_delta": comparison["delta"]["macro_f1"],
        "accuracy_delta": comparison["delta"]["accuracy"],
        "grouped_bootstrap_macro_f1_delta_ci95": bootstrap["macro_f1_delta"],
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "acceptance_opened_once": True,
        "selection_artifact": selection_path.name,
        "target_development_labels_used_for_selection": False,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        **_security(),
    }
    _dump(out / "w170_acceptance_report.json", report)
    return report


def finalize_source_generalization_repair_w171(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    release = Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    report170 = _read_json(out / "w170_acceptance_report.json")
    accepted = report170.get("status") == "accepted_optional_source_generalization_candidate"
    shadow_rows: list[dict[str, Any]] = []
    if accepted:
        method = str(report170["candidate_method"])
        artifact_path = Path(model_dir) / f"{method}__seed42.joblib"
        artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        safe_paths = [f"stats.{name}" for name in ENGINEERED_FEATURES]
        agent_name = "NFBoTSourceInvariantEvidenceAgentW170"
        flow = FlowRecord(trace_id="w171-shadow", sample_id="w171-shadow", stats={name: 0.0 for name in ENGINEERED_FEATURES})
        audit = FieldAuditResult(decisions=[], allowed_fields=safe_paths, blocked_fields=list(BLOCKED_FIELDS), context_only_fields=[], leakage_risk=0.0)
        capability = DetectorCapabilityProfile(agent_name=agent_name, backend=method, status="available", consumed_fields=safe_paths, available_fields=safe_paths, observed_fields=safe_paths)
        state = CaseState(flow=flow, field_audit=audit, detector_capabilities={agent_name: capability}, remaining_budget=1)
        request = EvidenceRequest(request_id="w171-source-invariant", case_trace_id=flow.trace_id, requested_agent=agent_name, permitted_safe_features=safe_paths, purpose="Collect default-off source-generalization evidence in shadow mode", budget=1, allowed_feature_policy_hash=hashlib.sha256(json.dumps(sorted(safe_paths), separators=(",", ":")).encode()).hexdigest(), expected_evidence_schema="AgentEvidenceV2")
        review = EvidenceRequestGuard(allowed_agents={agent_name}).review(request, state)
        original = AgentEvidence(agent_name=agent_name, agent_version="w170", feature_group=FeatureGroup.STATS, benign_support=0.3, malicious_support=0.7, confidence=0.7, uncertainty=0.3, model_reliability=0.9, contributes_to_verdict=False, evidence=["SOURCE_GENERALIZATION_OPTIONAL_SHADOW"], used_fields=safe_paths)
        v2 = AgentEvidenceV2Adapter().to_v2(original, review.request, artifact_hash=artifact_hash, dataset_scope="NF-BoT-IoT-v1-v2-only") if review.approved else None
        reliability = ReliabilityProfile(stats_reliability=1.0, sequence_reliability=1.0, tls_reliability=1.0, payload_reliability=1.0, input_completeness=1.0)
        baseline = AgentEvidence(agent_name="StatsDetectorAgent", feature_group=FeatureGroup.STATS, benign_support=0.8, malicious_support=0.2, confidence=0.8, uncertainty=0.2)
        fusion = FusionAgent()
        invariant = fusion.fuse([baseline], reliability, final=True).model_dump() == fusion.fuse([baseline, original], reliability, final=True).model_dump()
        shadow_rows.append({"request_approved": review.approved, "agent_evidence_v2_valid": v2 is not None, "candidate_contributes_to_verdict": v2.contributes_to_verdict if v2 else None, "fusion_snapshot_invariance": invariant, "fusion_owner": "FusionAgent"})
    _write_csv(release / "shadow_integration_w171.csv", shadow_rows)
    profile = {
        "profile_id": "runtime_nf_bot_source_invariant_w170_optional",
        "created": bool(accepted),
        "default_enabled": False,
        "dataset_scope": "NF-BoT-IoT v1/v2 only",
        "replaces_runtime_safe_v3_0": False,
        "fusion_owner": "FusionAgent",
        "promotion_status": "accepted_optional_profile" if accepted else "not_created_candidate_failed",
    }
    _dump(release / "runtime_profile_w171.json", profile)
    negative = [] if accepted else [{"experiment_id": EXPERIMENT, "status": report170.get("status", "missing"), "failed_gates": ";".join(report170.get("failed_gates", [])), "safe_claim": "The label-blind source-invariant repair did not pass all generalization gates.", "forbidden_claim": "Cross-source generalization was solved."}]
    negative.append({"experiment_id": "nf_ton_fresh_protocol_w168", **_read_json(out / "nf_ton_fresh_protocol_blocker_w168.json"), "safe_claim": "NF-ToN fresh 9-class replication was not feasible after historical exclusions.", "forbidden_claim": "NF-ToN source repair completed."})
    _write_csv(release / "negative_result_ledger_w171.csv", negative)
    after = hash_artifact_paths(_default_frozen_paths())
    before = _read_json(out / "frozen_hashes_before_w168.json")
    _dump(release / "frozen_hashes_after_w171.json", after)
    claims = [
        {"claim_id": "source_generalization_result", "safe_to_claim": accepted, "claim": "A label-blind, endpoint-group-disjoint NF-BoT source-generalization candidate passed all gates.", "support": (out / "w170_acceptance_report.json").as_posix()},
        {"claim_id": "default_runtime_replaced", "safe_to_claim": False, "claim": "runtime_safe_v3_0 was replaced.", "support": "forbidden"},
        {"claim_id": "nf_ton_completed", "safe_to_claim": False, "claim": "The fresh NF-ToN 9-class source repair completed.", "support": "forbidden"},
    ]
    _write_csv(release / "claim_ledger_w171.csv", claims)
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    text = f"""# MAD-ETD Source-Generalization Repair W168-W171

- status: `{report170.get('status', 'missing')}`
- baseline Macro-F1: `{report170.get('baseline_macro_f1', 'not_available')}`
- candidate Macro-F1: `{report170.get('candidate_macro_f1', 'not_available')}`
- Macro-F1 delta: `{report170.get('macro_f1_delta', 'not_available')}`
- candidate profile created: `{accepted}`
- runtime_safe_v3_0 remains default: `true`

The target-development partition was label-blind. Endpoint identities were used
only for irreversible group assignment. The sealed acceptance was opened once.
Any optional profile is dataset-specific and default-off.
"""
    (docs / "MAD_ETD_SOURCE_GENERALIZATION_REPAIR_W168_W171.md").write_text(text, encoding="utf-8")
    (docs / "MAD_ETD_SOURCE_GENERALIZATION_REPAIR_W168_W171_CN.md").write_text(text.replace("Source-Generalization Repair", "来源泛化修复").replace("The target-development partition was label-blind.", "目标域 development 不使用标签。").replace("Endpoint identities were used\nonly for irreversible group assignment.", "端点身份仅用于不可逆分组。"), encoding="utf-8")
    security = all(int(report170.get(field, 0)) == 0 for field in ("blocked_field_violation", "fusion_ownership_violation", "ood_override_count", "illegal_verdict_execution_count", "fake_metric_count"))
    report = {
        "status": "accepted_source_generalization_repair_release" if tests_passed and security else "pending_or_failed_source_generalization_release",
        "candidate_status": report170.get("status", "missing"),
        "optional_profile_created": bool(accepted),
        "shadow_integration_completed": bool(shadow_rows),
        "fusion_snapshot_invariance": 1.0 if shadow_rows and shadow_rows[0]["fusion_snapshot_invariance"] else ("not_applicable" if not accepted else 0.0),
        "frozen_hashes_unchanged": before == after,
        "full_pytest_passed": bool(tests_passed),
        "full_pytest_count": int(test_count),
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "production_ready": False,
        **_security(),
    }
    _dump(release / "acceptance_report.json", report)
    return report
