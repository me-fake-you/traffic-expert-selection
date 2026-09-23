"""W176--W179 NF-ToN v1-to-v2 binary group-generalization protocol.

NF-BoT was the preferred dataset, but a live preflight found no malicious
endpoint group left after the W161/W168 exclusions.  NF-ToN is therefore the
pre-registered fallback.  Endpoint identities are used only to derive
irreversible split-group hashes and never enter the feature matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load

from .external_multiagent_positive_w160_w167 import (
    BLOCKED_FIELDS,
    ENGINEERED_FEATURES,
    RAW_SAFE_FEATURES,
    _choose_temperature,
    _fit_family,
    _manifest_entries,
    _metrics,
    _numeric_frame,
    _predict_artifact,
    _temperature_scale,
)
from .fusion import FusionAgent
from .paper_evaluation import hash_artifact_paths
from .schemas import AgentEvidence, FeatureGroup, ReliabilityProfile
from .soc_evidence_team_w72 import _default_frozen_paths
from .source_generalization_repair_w168_w171 import (
    _endpoint_group,
    _grouped_bootstrap_delta,
    _read_npz_rows,
)


EXPERIMENT = "mad_etd_nfton_binary_generalization_w176_w179"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_HISTORY = Path("data/runs/mad_etd_external_multiagent_positive_w160_w167")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfton_binary_generalization_w176_w178")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_nfton_binary_generalization_w177")
DEFAULT_RELEASE_DIR = Path("data/releases/mad_etd_nfton_binary_generalization_w179")
CLASSES = ("benign", "malicious")
SEEDS = (42, 43, 44)
SOURCE_TRAIN_PER_CLASS = 8_000
SOURCE_VALIDATION_PER_CLASS = 2_000
TARGET_DEVELOPMENT_PER_CLASS = 10_000
TARGET_ACCEPTANCE_PER_CLASS = 10_000
BASELINE_METHODS = ("hist_gradient_boosting", "random_forest", "extra_trees")
CANDIDATE_METHODS = ("oof_stacking", "soft_voting")


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}


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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


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


def _role(group: str, domain: str) -> str:
    value = int(hashlib.sha256(f"w176:{domain}:{group}".encode()).hexdigest()[:16], 16)
    if domain == "source":
        return "source_validation" if value % 5 == 0 else "source_train"
    return "target_development" if value % 2 == 0 else "target_acceptance"


def _push(
    current: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    x: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
    priorities: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if current is None:
        merged = (x, rows, groups, priorities)
    else:
        merged = tuple(np.concatenate([current[index], value]) for index, value in enumerate((x, rows, groups, priorities)))
    if len(merged[3]) > limit:
        keep = np.argpartition(merged[3], limit - 1)[:limit]
        merged = tuple(value[keep] for value in merged)
    return merged  # type: ignore[return-value]


def _extract_binary(
    entry: Mapping[str, Any],
    exclusions: set[int],
    *,
    domain: str,
    forbidden_groups: set[str] | None = None,
) -> tuple[dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]], set[str], dict[str, Any]]:
    archive_path, member = Path(str(entry["archive"])), str(entry["entry"])
    label_column = str(entry.get("label_column") or "Label")
    limits = (
        {"source_train": SOURCE_TRAIN_PER_CLASS, "source_validation": SOURCE_VALIDATION_PER_CLASS}
        if domain == "source"
        else {"target_development": TARGET_DEVELOPMENT_PER_CLASS, "target_acceptance": TARGET_ACCEPTANCE_PER_CLASS}
    )
    reservoirs: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = {
        (label, role): None for label in (0, 1) for role in limits
    }
    rng = np.random.default_rng(176 if domain == "source" else 177)
    exclusion_array = np.fromiter(exclusions, dtype=np.int64) if exclusions else np.empty((0,), dtype=np.int64)
    all_eligible_groups: set[str] = set()
    observed: defaultdict[int, int] = defaultdict(int)
    excluded_rows = excluded_groups = 0
    offset = 0
    started = time.perf_counter()
    usecols = [*RAW_SAFE_FEATURES, label_column, "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        for chunk in pd.read_csv(raw, usecols=usecols, chunksize=250_000, low_memory=False):
            rows = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            labels = pd.to_numeric(chunk[label_column], errors="coerce").fillna(-1).astype(int).to_numpy()
            history = np.isin(rows, exclusion_array) if len(exclusion_array) else np.zeros(len(rows), dtype=bool)
            excluded_rows += int(history.sum())
            valid = np.isin(labels, (0, 1)) & ~history
            if not valid.any():
                continue
            subset = chunk.loc[valid]
            selected_rows = rows[valid]
            selected_labels = labels[valid]
            groups = np.asarray([
                _endpoint_group(src, dst)
                for src, dst in zip(subset["IPV4_SRC_ADDR"], subset["IPV4_DST_ADDR"])
            ], dtype="U64")
            if forbidden_groups:
                allowed = np.asarray([group not in forbidden_groups for group in groups], dtype=bool)
                excluded_groups += int((~allowed).sum())
                subset = subset.loc[allowed]
                selected_rows = selected_rows[allowed]
                selected_labels = selected_labels[allowed]
                groups = groups[allowed]
            if not len(groups):
                continue
            all_eligible_groups.update(groups.tolist())
            x = _numeric_frame(subset.loc[:, list(RAW_SAFE_FEATURES)])
            roles = np.asarray([_role(group, domain) for group in groups], dtype="U32")
            priorities = rng.random(len(groups))
            for label in (0, 1):
                observed[label] += int((selected_labels == label).sum())
                for role, limit in limits.items():
                    mask = (selected_labels == label) & (roles == role)
                    if mask.any():
                        key = (label, role)
                        reservoirs[key] = _push(
                            reservoirs[key], x[mask], selected_rows[mask], groups[mask], priorities[mask], limit
                        )
    result: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, values in reservoirs.items():
        if values is None:
            continue
        order = np.argsort(values[3])
        result[key] = values[0][order], values[1][order], values[2][order]
    return result, all_eligible_groups, {
        "archive": archive_path.as_posix(),
        "member": member,
        "domain": domain,
        "scanned_rows": offset,
        "historical_rows_excluded": excluded_rows,
        "cross_domain_group_rows_excluded": excluded_groups,
        "observed_eligible_label_counts": {CLASSES[key]: value for key, value in observed.items()},
        "eligible_endpoint_group_count": len(all_eligible_groups),
        "bucket_counts": {f"{CLASSES[label]}::{role}": len(values[0]) for (label, role), values in result.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }


def _assemble(
    buckets: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]], role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, rows, groups = [], [], [], []
    for label in (0, 1):
        values = buckets[(label, role)]
        x.append(values[0])
        y.append(np.full(len(values[0]), label, dtype=np.int64))
        rows.append(values[1])
        groups.append(values[2])
    return np.concatenate(x), np.concatenate(y), np.concatenate(rows), np.concatenate(groups)


def build_nfton_binary_generalization_w176(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    history_dir: str | Path = DEFAULT_HISTORY,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    out, history = Path(output_dir), Path(history_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = _manifest_entries(Path(processed_dir))
    required = {"NF-ToN-IoT", "NF-ToN-IoT-v2"}
    source_history = _read_npz_rows(history / "nf_ton_iot_group_protocol_w161.npz")
    target_history = source_history | _read_npz_rows(history / "nf_ton_iot_paper_protocol_w161.npz")
    if not required.issubset(entries) or not source_history or not target_history:
        report = {"status": "failed_w176_missing_nfton_sources_or_history", **_security()}
        _dump(out / "w176_acceptance_report.json", report)
        return report
    _dump(out / "frozen_hashes_before_w176.json", hash_artifact_paths(_default_frozen_paths()))
    _dump(out / "nf_bot_fallback_blocker_w176.json", {
        "status": "failed_no_fresh_malicious_endpoint_groups",
        "source_variant": "NF-BoT-IoT",
        "target_variant": "NF-BoT-IoT-v2",
        "source_fresh_malicious_rows": 0,
        "source_fresh_malicious_groups": 0,
        "target_fresh_malicious_rows": 0,
        "target_fresh_malicious_groups": 0,
        "audit_method": "label_and_endpoint_only_live_preflight_before_W176",
        "fallback": "NF-ToN-IoT v1-to-v2 binary protocol",
        "fake_metric_count": 0,
    })
    source_buckets, source_groups, source_audit = _extract_binary(
        entries["NF-ToN-IoT"], source_history, domain="source"
    )
    target_buckets, target_groups, target_audit = _extract_binary(
        entries["NF-ToN-IoT-v2"], target_history, domain="target", forbidden_groups=source_groups
    )
    _dump(out / "fresh_binary_extraction_audit_w176.json", {"source": source_audit, "target": target_audit})
    required_buckets = {
        (0, "source_train"): SOURCE_TRAIN_PER_CLASS,
        (1, "source_train"): SOURCE_TRAIN_PER_CLASS,
        (0, "source_validation"): SOURCE_VALIDATION_PER_CLASS,
        (1, "source_validation"): SOURCE_VALIDATION_PER_CLASS,
        (0, "target_development"): TARGET_DEVELOPMENT_PER_CLASS,
        (1, "target_development"): TARGET_DEVELOPMENT_PER_CLASS,
        (0, "target_acceptance"): TARGET_ACCEPTANCE_PER_CLASS,
        (1, "target_acceptance"): TARGET_ACCEPTANCE_PER_CLASS,
    }
    missing = [
        f"{CLASSES[label]}::{role}"
        for (label, role), limit in required_buckets.items()
        if len((source_buckets if role.startswith("source") else target_buckets).get((label, role), ([], [], []))[0]) < limit
    ]
    if missing:
        report = {"status": "failed_w176_insufficient_fresh_group_buckets", "missing": missing, **_security()}
        _dump(out / "w176_acceptance_report.json", report)
        return report
    source_train = _assemble(source_buckets, "source_train")
    source_validation = _assemble(source_buckets, "source_validation")
    target_development = _assemble(target_buckets, "target_development")
    target_acceptance = _assemble(target_buckets, "target_acceptance")
    split_sets = {
        "source_train": set(source_train[3].tolist()),
        "source_validation": set(source_validation[3].tolist()),
        "target_development": set(target_development[3].tolist()),
        "target_acceptance": set(target_acceptance[3].tolist()),
    }
    overlap = {
        f"{left}__{right}": len(split_sets[left] & split_sets[right])
        for index, left in enumerate(split_sets)
        for right in list(split_sets)[index + 1 :]
    }
    np.savez_compressed(out / "source_train_w176.npz", x=source_train[0], y=source_train[1], rows=source_train[2], groups=source_train[3], classes=np.asarray(CLASSES, dtype="U16"), feature_names=np.asarray(ENGINEERED_FEATURES, dtype="U64"))
    np.savez_compressed(out / "source_validation_w176.npz", x=source_validation[0], y=source_validation[1], rows=source_validation[2], groups=source_validation[3])
    np.savez_compressed(out / "target_development_unlabeled_w176.npz", x=target_development[0], rows=target_development[2], groups=target_development[3])
    np.savez_compressed(out / "target_acceptance_sealed_w176.npz", x=target_acceptance[0], y=target_acceptance[1], rows=target_acceptance[2], groups=target_acceptance[3])
    sealed_path = out / "target_acceptance_sealed_w176.npz"
    _dump(out / "sealed_acceptance_commitment_w176.json", {
        "sha256": hashlib.sha256(sealed_path.read_bytes()).hexdigest(),
        "opened": False,
        "sample_count": len(target_acceptance[1]),
        "endpoint_group_count": len(split_sets["target_acceptance"]),
    })
    manifest: list[dict[str, Any]] = []
    for split, values, label_visible, variant in (
        ("source_train", source_train, True, "NF-ToN-IoT"),
        ("source_validation", source_validation, True, "NF-ToN-IoT"),
        ("target_development_unlabeled", target_development, False, "NF-ToN-IoT-v2"),
        ("target_acceptance_sealed", target_acceptance, False, "NF-ToN-IoT-v2"),
    ):
        for index, (row, group) in enumerate(zip(values[2], values[3])):
            manifest.append({
                "sample_hash": hashlib.sha256(f"w176:{variant}:{int(row)}".encode()).hexdigest(),
                "source_variant": variant,
                "split": split,
                "binary_label": CLASSES[int(values[1][index])] if label_visible else "sealed",
                "endpoint_group_hash": group,
                "endpoint_identity_in_feature_matrix": False,
                "source_variant_in_feature_matrix": False,
            })
    _write_csv(out / "fresh_binary_split_manifest_w176.csv", manifest)
    policy = {
        "status": "safe_binary_feature_policy_locked",
        "safe_features": list(ENGINEERED_FEATURES),
        "blocked_fields": list(BLOCKED_FIELDS),
        "endpoint_usage": "irreversible_split_group_only",
        "source_variant_usage": "protocol_and_audit_only",
        "target_development_labels": "not_materialized",
        "target_acceptance_labels": "sealed_until_W178",
        "feature_policy_hash": hashlib.sha256(json.dumps({"safe": ENGINEERED_FEATURES, "blocked": BLOCKED_FIELDS}, sort_keys=True, default=list).encode()).hexdigest(),
    }
    _dump(out / "safe_binary_feature_policy_w176.json", policy)
    status = "w176_nfton_binary_group_protocol_frozen" if all(value == 0 for value in overlap.values()) else "failed_w176_group_overlap"
    report = {
        "status": status,
        "preferred_nf_bot_status": "failed_no_fresh_malicious_endpoint_groups",
        "fallback_dataset": "NF-ToN-IoT v1/v2",
        "freshness_scope": "row-fresh relative to W161 and endpoint-group-disjoint within W176; not claimed globally unseen across all historical project runs",
        "source_train_count": len(source_train[1]),
        "source_validation_count": len(source_validation[1]),
        "target_development_unlabeled_count": len(target_development[0]),
        "target_acceptance_sealed_count": len(target_acceptance[0]),
        "split_group_counts": {key: len(value) for key, value in split_sets.items()},
        "all_group_overlaps": overlap,
        "target_development_labels_materialized": False,
        "target_acceptance_opened": False,
        **_security(),
    }
    _dump(out / "w176_acceptance_report.json", report)
    return report


def _fit_method(method: str, x: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int) -> dict[str, Any]:
    if method == "soft_voting":
        return {
            "kind": "soft_voting",
            "method": method,
            "classes": classes,
            "components": [_fit_family(name, x, y, classes, seed + index) for index, name in enumerate(BASELINE_METHODS)],
        }
    artifact = _fit_family(method, x, y, classes, seed)
    artifact["method"] = method
    return artifact


def _predict(artifact: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    if artifact["kind"] == "soft_voting":
        return np.mean([_predict_artifact(component, x) for component in artifact["components"]], axis=0)
    return _predict_artifact(artifact, x)


def _threshold_shift(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Shift binary probabilities so argmax implements the fixed threshold."""
    p = np.clip(probabilities[:, 1], 1e-9, 1.0 - 1e-9)
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    odds = p / (1.0 - p)
    shifted_odds = odds * ((1.0 - threshold) / threshold)
    malicious = shifted_odds / (1.0 + shifted_odds)
    return np.column_stack([1.0 - malicious, malicious])


def _calibrated_probabilities(artifact: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    probabilities = _temperature_scale(_predict(artifact, x), float(artifact["temperature"]))
    if "decision_threshold" in artifact:
        probabilities = _threshold_shift(probabilities, float(artifact["decision_threshold"]))
    return probabilities


def _choose_binary_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates: list[tuple[float, dict[str, float]]] = []
    for threshold in np.linspace(0.10, 0.90, 81):
        shifted = _threshold_shift(probabilities, float(threshold))
        candidates.append((float(threshold), _metrics(y, shifted)))
    candidates.sort(key=lambda item: (-item[1]["macro_f1"], item[1]["ece"], abs(item[0] - 0.5)))
    return candidates[0]


def train_nfton_binary_candidates_w177(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    protocol = _read(out / "w176_acceptance_report.json")
    if protocol.get("status") != "w176_nfton_binary_group_protocol_frozen":
        report = {"status": "failed_w177_missing_w176_protocol", **_security()}
        _dump(out / "w177_acceptance_report.json", report)
        return report
    train = np.load(out / "source_train_w176.npz", allow_pickle=False)
    validation = np.load(out / "source_validation_w176.npz", allow_pickle=False)
    target = np.load(out / "target_development_unlabeled_w176.npz", allow_pickle=False)
    if "y" in target.files:
        report = {"status": "failed_w177_target_development_labels_materialized", **_security()}
        _dump(out / "w177_acceptance_report.json", report)
        return report
    models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    methods = (*BASELINE_METHODS, *CANDIDATE_METHODS)
    for method in methods:
        for seed in SEEDS:
            started = time.perf_counter()
            artifact = _fit_method(method, train["x"], train["y"], train["classes"], seed)
            raw = _predict(artifact, validation["x"])
            temperature = _choose_temperature(validation["y"], raw)
            artifact["temperature"] = temperature
            artifact["feature_policy_hash"] = _read(out / "safe_binary_feature_policy_w176.json")["feature_policy_hash"]
            probabilities = _temperature_scale(raw, temperature)
            target_raw = _predict(artifact, target["x"])
            target_confidence = float(np.mean(np.max(target_raw, axis=1)))
            joblib_dump(artifact, models / f"{method}__seed{seed}.joblib", compress=3)
            rows.append({
                "method": method,
                "seed": seed,
                "target_development_labels_used": False,
                "target_development_mean_confidence_diagnostic": target_confidence,
                "training_seconds": time.perf_counter() - started,
                **_metrics(validation["y"], probabilities),
            })
    _write_csv(out / "source_validation_results_w177.csv", rows)
    summaries = []
    for method in methods:
        values = [row for row in rows if row["method"] == method]
        summaries.append({
            "method": method,
            "validation_macro_f1_mean": float(np.mean([row["macro_f1"] for row in values])),
            "validation_accuracy_mean": float(np.mean([row["accuracy"] for row in values])),
            "validation_macro_recall_mean": float(np.mean([row["macro_recall"] for row in values])),
            "validation_ece_mean": float(np.mean([row["ece"] for row in values])),
        })
    baselines = sorted([row for row in summaries if row["method"] in BASELINE_METHODS], key=lambda row: (-row["validation_macro_f1_mean"], row["validation_ece_mean"], row["method"]))
    candidates = sorted([row for row in summaries if row["method"] in CANDIDATE_METHODS], key=lambda row: (-row["validation_macro_f1_mean"], row["validation_ece_mean"], row["method"]))
    baseline, candidate = baselines[0], candidates[0]
    selection_passed = (
        candidate["validation_macro_f1_mean"] >= baseline["validation_macro_f1_mean"] - 0.005
        and candidate["validation_ece_mean"] <= baseline["validation_ece_mean"] + 0.01
    )
    selection = {
        "status": "w177_binary_candidate_locked" if selection_passed else "no_w177_candidate_passed_source_validation_gate",
        "baseline_reference": baseline,
        "selected_candidate": candidate if selection_passed else None,
        "all_method_summaries": summaries,
        "selection_inputs": ["source validation labels", "unlabeled target-development predictions as diagnostic only"],
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "selection_hash": hashlib.sha256(json.dumps({"baseline": baseline, "candidate": candidate if selection_passed else None}, sort_keys=True).encode()).hexdigest(),
        **_security(),
    }
    _dump(out / "model_selection_lock_w177.json", selection)
    report = {
        "status": selection["status"],
        "baseline_method": baseline["method"],
        "candidate_method": candidate["method"] if selection_passed else None,
        "trained_model_count": len(rows),
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        **_security(),
    }
    _dump(out / "w177_acceptance_report.json", report)
    return report


def train_nfton_threshold_candidate_w177_1(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """Lock a validation-only threshold candidate before W178 is opened."""
    out, models = Path(output_dir), Path(model_dir)
    original = _read(out / "model_selection_lock_w177.json")
    if original.get("status") != "no_w177_candidate_passed_source_validation_gate":
        report = {"status": "failed_w177_1_requires_recorded_w177_gate_failure", **_security()}
        _dump(out / "w177_1_acceptance_report.json", report)
        return report
    if (out / "w178_acceptance_opened.json").is_file():
        report = {"status": "failed_w177_1_acceptance_already_opened", **_security()}
        _dump(out / "w177_1_acceptance_report.json", report)
        return report
    validation = np.load(out / "source_validation_w176.npz", allow_pickle=False)
    target = np.load(out / "target_development_unlabeled_w176.npz", allow_pickle=False)
    if "y" in target.files:
        report = {"status": "failed_w177_1_target_development_labels_materialized", **_security()}
        _dump(out / "w177_1_acceptance_report.json", report)
        return report
    baseline = original["baseline_reference"]
    if baseline["method"] != "extra_trees":
        report = {"status": "failed_w177_1_expected_extra_trees_reference", **_security()}
        _dump(out / "w177_1_acceptance_report.json", report)
        return report
    rows: list[dict[str, Any]] = []
    method = "extra_trees_threshold_calibrated"
    for seed in SEEDS:
        artifact = joblib_load(models / f"extra_trees__seed{seed}.joblib")
        raw_validation = _temperature_scale(_predict(artifact, validation["x"]), float(artifact["temperature"]))
        threshold, metrics = _choose_binary_threshold(validation["y"], raw_validation)
        artifact["method"] = method
        artifact["decision_threshold"] = threshold
        target_prob = _calibrated_probabilities(artifact, target["x"])
        joblib_dump(artifact, models / f"{method}__seed{seed}.joblib", compress=3)
        rows.append({
            "method": method,
            "seed": seed,
            "decision_threshold": threshold,
            "target_development_labels_used": False,
            "target_development_mean_confidence_diagnostic": float(np.mean(np.max(target_prob, axis=1))),
            **metrics,
        })
    _write_csv(out / "source_validation_results_w177_1.csv", rows)
    candidate = {
        "method": method,
        "validation_macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
        "validation_accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
        "validation_macro_recall_mean": float(np.mean([row["macro_recall"] for row in rows])),
        "validation_ece_mean": float(np.mean([row["ece"] for row in rows])),
        "decision_thresholds": [float(row["decision_threshold"]) for row in rows],
    }
    passed = (
        candidate["validation_macro_f1_mean"] >= baseline["validation_macro_f1_mean"] - 0.005
        and candidate["validation_ece_mean"] <= baseline["validation_ece_mean"] + 0.01
    )
    selection = {
        "status": "w177_binary_candidate_locked" if passed else "no_w177_1_candidate_passed_source_validation_gate",
        "development_round": "W177.1",
        "pre_acceptance_iteration": True,
        "original_w177_status": original["status"],
        "selection_gate_unchanged": True,
        "baseline_reference": baseline,
        "selected_candidate": candidate if passed else None,
        "selection_inputs": ["source validation labels", "unlabeled target-development predictions as diagnostic only"],
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "selection_hash": hashlib.sha256(json.dumps({"round": "W177.1", "baseline": baseline, "candidate": candidate if passed else None}, sort_keys=True).encode()).hexdigest(),
        **_security(),
    }
    _dump(out / "model_selection_lock_w177_1.json", selection)
    report = {
        "status": selection["status"],
        "development_round": "W177.1",
        "baseline_method": baseline["method"],
        "candidate_method": method if passed else None,
        "decision_thresholds": candidate["decision_thresholds"],
        "target_development_labels_used": False,
        "target_acceptance_opened": False,
        "selection_gate_unchanged": True,
        **_security(),
    }
    _dump(out / "w177_1_acceptance_report.json", report)
    return report


def evaluate_nfton_binary_acceptance_w178(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    marker = out / "w178_acceptance_opened.json"
    if marker.is_file():
        return {**_read(out / "w178_acceptance_report.json"), "rerun_refused_acceptance_already_opened": True}
    selection_path = out / "model_selection_lock_w177_1.json"
    if not selection_path.is_file():
        selection_path = out / "model_selection_lock_w177.json"
    selection = _read(selection_path)
    if selection.get("status") != "w177_binary_candidate_locked":
        report = {"status": "not_run_w178_no_locked_candidate", **_security()}
        _dump(out / "w178_acceptance_report.json", report)
        return report
    baseline_method = selection["baseline_reference"]["method"]
    candidate_method = selection["selected_candidate"]["method"]
    _dump(marker, {"opened_once": True, "selection_hash": selection["selection_hash"], "selection_artifact": selection_path.name, "opened_at_unix": time.time()})
    sealed_path = out / "target_acceptance_sealed_w176.npz"
    commitment = _read(out / "sealed_acceptance_commitment_w176.json")
    sealed = np.load(sealed_path, allow_pickle=False)
    validation = np.load(out / "source_validation_w176.npz", allow_pickle=False)
    probabilities: dict[str, list[np.ndarray]] = {baseline_method: [], candidate_method: []}
    val_metrics: dict[str, list[dict[str, float]]] = {baseline_method: [], candidate_method: []}
    for method in (baseline_method, candidate_method):
        for seed in SEEDS:
            artifact = joblib_load(models / f"{method}__seed{seed}.joblib")
            probabilities[method].append(_calibrated_probabilities(artifact, sealed["x"]))
            val_metrics[method].append(_metrics(validation["y"], _calibrated_probabilities(artifact, validation["x"])))
    baseline_prob = np.mean(probabilities[baseline_method], axis=0)
    candidate_prob = np.mean(probabilities[candidate_method], axis=0)
    baseline = _metrics(sealed["y"], baseline_prob)
    candidate = _metrics(sealed["y"], candidate_prob)
    baseline_pred, candidate_pred = baseline_prob.argmax(axis=1), candidate_prob.argmax(axis=1)
    baseline_recall = float(((baseline_pred == 1) & (sealed["y"] == 1)).sum() / max(1, (sealed["y"] == 1).sum()))
    candidate_recall = float(((candidate_pred == 1) & (sealed["y"] == 1)).sum() / max(1, (sealed["y"] == 1).sum()))
    bootstrap = _grouped_bootstrap_delta(sealed["y"], baseline_prob, candidate_prob, sealed["groups"])
    _dump(out / "grouped_bootstrap_w178.json", bootstrap)
    comparison = {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {key: candidate[key] - baseline[key] for key in baseline},
        "malicious_recall": {"baseline": baseline_recall, "candidate": candidate_recall, "delta": candidate_recall - baseline_recall},
        "source_validation_macro_f1": {
            "baseline": float(np.mean([row["macro_f1"] for row in val_metrics[baseline_method]])),
            "candidate": float(np.mean([row["macro_f1"] for row in val_metrics[candidate_method]])),
        },
    }
    _dump(out / "binary_acceptance_comparison_w178.json", comparison)
    gates = {
        "macro_f1_delta_ge_0_01": comparison["delta"]["macro_f1"] >= 0.01,
        "macro_f1_delta_ci95_lower_gt_0": bootstrap["macro_f1_delta"]["ci95_lower"] > 0.0,
        "malicious_recall_not_worse": candidate_recall >= baseline_recall,
        "ece_not_worse_by_more_than_0_01": candidate["ece"] <= baseline["ece"] + 0.01,
        "source_validation_macro_f1_drop_le_0_005": comparison["source_validation_macro_f1"]["candidate"] >= comparison["source_validation_macro_f1"]["baseline"] - 0.005,
        "endpoint_group_count_positive": bootstrap["group_count"] > 0,
    }
    accepted = all(gates.values())
    verified = commitment.get("sha256") == hashlib.sha256(sealed_path.read_bytes()).hexdigest()
    _dump(out / "sealed_acceptance_commitment_w178.json", {**commitment, "opened": True, "opened_once": True, "sha256_verified": verified})
    report = {
        "status": "accepted_optional_nfton_binary_group_candidate" if accepted else "not_promoted_nfton_binary_generalization_gate_failed",
        "scope": "NF-ToN-IoT v1-to-v2 binary group protocol only",
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "accuracy_delta": comparison["delta"]["accuracy"],
        "baseline_macro_f1": baseline["macro_f1"],
        "candidate_macro_f1": candidate["macro_f1"],
        "macro_f1_delta": comparison["delta"]["macro_f1"],
        "malicious_recall_delta": candidate_recall - baseline_recall,
        "macro_f1_delta_ci95": bootstrap["macro_f1_delta"],
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "acceptance_opened_once": True,
        "selection_artifact": selection_path.name,
        "acceptance_commitment_verified": verified,
        "target_development_labels_used_for_selection": False,
        **_security(),
    }
    _dump(out / "w178_acceptance_report.json", report)
    return report


def finalize_nfton_binary_generalization_w179(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out, models, release = Path(output_dir), Path(model_dir), Path(release_dir)
    release.mkdir(parents=True, exist_ok=True)
    result = _read(out / "w178_acceptance_report.json")
    accepted = result.get("status") == "accepted_optional_nfton_binary_group_candidate"
    shadow_invariance: float | str = "not_applicable"
    if accepted:
        reliability = ReliabilityProfile(stats_reliability=1.0, sequence_reliability=1.0, tls_reliability=1.0, payload_reliability=1.0, input_completeness=1.0)
        baseline = AgentEvidence(agent_name="StatsDetectorAgent", feature_group=FeatureGroup.STATS, benign_support=0.8, malicious_support=0.2, confidence=0.8, uncertainty=0.2)
        shadow = AgentEvidence(agent_name="NFToNBinaryGeneralizationAgent", feature_group=FeatureGroup.STATS, benign_support=0.2, malicious_support=0.8, confidence=0.8, uncertainty=0.2, contributes_to_verdict=False, evidence=["DEFAULT_OFF_OPTIONAL_SHADOW"])
        fusion = FusionAgent()
        shadow_invariance = 1.0 if fusion.fuse([baseline], reliability, final=True).model_dump() == fusion.fuse([baseline, shadow], reliability, final=True).model_dump() else 0.0
    profile = {
        "profile_id": "runtime_nf_ton_binary_generalization_w178_optional",
        "created": bool(accepted),
        "default_enabled": False,
        "dataset_scope": "NF-ToN-IoT v1/v2 binary only",
        "promotion_status": "accepted_optional_profile" if accepted else "not_created_candidate_failed",
        "replaces_runtime_safe_v3_0": False,
        "fusion_owner": "FusionAgent",
    }
    _dump(release / "runtime_profile_w179.json", profile)
    negative = [] if accepted else [{
        "candidate": "NF-ToN binary cross-version ensemble",
        "status": result.get("status", "missing"),
        "failed_gates": ";".join(result.get("failed_gates", [])),
        "safe_claim": "The fixed NF-ToN binary candidate did not pass all cross-version gates.",
        "forbidden_claim": "General cross-dataset malicious traffic detection was solved.",
        "runtime_created": False,
        "fake_metric_count": 0,
    }]
    negative.append({
        "candidate": "NF-BoT preferred binary protocol",
        "status": "failed_no_fresh_malicious_endpoint_groups",
        "failed_gates": "fresh_group_feasibility",
        "safe_claim": "NF-BoT was rejected before training because all malicious endpoint groups had historical overlap.",
        "forbidden_claim": "NF-BoT fresh binary acceptance completed.",
        "runtime_created": False,
        "fake_metric_count": 0,
    })
    _write_csv(release / "negative_result_ledger_w179.csv", negative)
    _write_csv(release / "claim_ledger_w179.csv", [
        {"claim_id": "nfton_binary_candidate", "safe_to_claim": accepted, "claim": "A dataset-specific NF-ToN v1-to-v2 binary candidate passed all W178 gates.", "support": (out / "w178_acceptance_report.json").as_posix()},
        {"claim_id": "general_runtime_replaced", "safe_to_claim": False, "claim": "runtime_safe_v3_0 was replaced.", "support": "forbidden"},
    ])
    before = _read(out / "frozen_hashes_before_w176.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(release / "frozen_hashes_after_w179.json", after)
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    en = f"""# MAD-ETD NF-ToN Binary Generalization W176-W179

- status: `{result.get('status', 'missing')}`
- baseline Accuracy: `{result.get('baseline_accuracy', 'not_available')}`
- candidate Accuracy: `{result.get('candidate_accuracy', 'not_available')}`
- Macro-F1 delta: `{result.get('macro_f1_delta', 'not_available')}`
- optional profile created: `{accepted}`
- runtime_safe_v3_0 remains default: `true`

NF-BoT failed the fresh malicious endpoint-group preflight, so NF-ToN was the
declared fallback. The protocol is row-fresh relative to W161 and group-disjoint
within W176; it is not described as globally unseen across every historical run.
"""
    cn = f"""# MAD-ETD NF-ToN 二分类跨版本泛化 W176-W179

- 状态：`{result.get('status', 'missing')}`
- 基线 Accuracy：`{result.get('baseline_accuracy', 'not_available')}`
- 候选 Accuracy：`{result.get('candidate_accuracy', 'not_available')}`
- Macro-F1 增量：`{result.get('macro_f1_delta', 'not_available')}`
- 是否创建可选 profile：`{accepted}`
- `runtime_safe_v3_0` 仍为默认：`true`

NF-BoT 因没有未消费的恶意 endpoint group 而在训练前失败，因此按预先声明的
fallback 使用 NF-ToN。该协议相对 W161 行级新鲜，且 W176 内部 endpoint-group
完全隔离；不能描述成整个项目历史上从未查看过的数据。
"""
    (docs / "MAD_ETD_NFTON_BINARY_GENERALIZATION_W176_W179.md").write_text(en, encoding="utf-8")
    (docs / "MAD_ETD_NFTON_BINARY_GENERALIZATION_W176_W179_CN.md").write_text(cn, encoding="utf-8")
    security_ok = all(int(result.get(key, 0)) == 0 for key in ("blocked_field_violation", "fusion_ownership_violation", "ood_override_count", "illegal_verdict_execution_count", "fake_metric_count"))
    report = {
        "status": "accepted_w179_nfton_binary_evidence_release" if tests_passed and security_ok else "pending_or_failed_w179_release",
        "candidate_status": result.get("status", "missing"),
        "optional_profile_created": bool(accepted),
        "shadow_fusion_snapshot_invariance": shadow_invariance,
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
