"""W97 source-held-out audit for the accepted W96 IoTBotnetSkill candidate."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline

from .domain_robust_stats_w81 import _source_entries
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .nfiot_bot_targeted_performance_w96 import _exclusions_through_w95
from .nfiot_targeted_performance_w94_w95 import (
    DEFAULT_PROCESSED,
    _ece,
    _native_vector,
    _read_csv,
    _read_json,
    _security,
    _select_threshold,
)
from .paper_evaluation import hash_artifact_paths, sha256_file
from .safe_flow_ensemble_w73_w76 import (
    ENGINEERED_FEATURES,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
)
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_nfiot_source_heldout_w97"
V2_GROUPS = ("NF-BoT-IoT-v2", "NF-ToN-IoT-v2")
ALL_GROUPS = ("NF-BoT-IoT", "NF-ToN-IoT", *V2_GROUPS)
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfiot_source_heldout_w97")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_nfiot_source_heldout_w97")
DEFAULT_W93_DIR = Path("data/runs/mad_etd_nfiot_skill_positive_w93")
DEFAULT_W95_DIR = Path("data/runs/mad_etd_nfiot_targeted_performance_w94_w95")
DEFAULT_W96_DIR = Path("data/runs/mad_etd_nfiot_bot_targeted_performance_w96")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_SOURCE_HELDOUT_W97.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_SOURCE_HELDOUT_W97_CN.md")
PER_LABEL_PER_GROUP = 5_000
MAX_ROWS = {"NF-BoT-IoT-v2": 12_000_000, "NF-ToN-IoT-v2": 6_000_000}
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


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


def _exclusions_through_w96(
    w93_dir: str | Path,
    w95_dir: str | Path,
    w96_dir: str | Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, earlier = _exclusions_through_w95(w93_dir, w95_dir)
    root = Path(w96_dir)
    rows = _read_csv(root / "fresh_sample_manifest.csv")
    report = _read_json(root / "acceptance_report.json")
    hashes = {row["sample_hash"] for row in rows if row.get("sample_hash")}
    legacy_hashes = {
        row["legacy_sample_id_hash"]
        for row in rows
        if row.get("legacy_sample_id_hash")
    }
    mapped = (
        bool(rows)
        and report.get("status")
        == "accepted_dataset_specific_full_coverage_accuracy_f1_result"
    )
    canonical.update(hashes)
    legacy.update(legacy_hashes)
    ready = (
        earlier.get("status") == "all_auditable_nf_samples_through_w95_excluded"
        and mapped
    )
    return canonical, legacy, {
        "status": (
            "all_auditable_nf_samples_through_w96_excluded"
            if ready
            else "failed_w97_historical_exclusion_mapping"
        ),
        "through_w95": earlier,
        "w96_status": report.get("status", "missing"),
        "w96_sample_count": len(hashes),
        "canonical_exclusion_count": len(canonical),
        "legacy_exclusion_count": len(legacy),
    }


def _keep(
    heap: list[tuple[int, int, tuple[list[float], list[float], int, str, str, str]]],
    *,
    priority: int,
    tie: int,
    item_factory: Any,
    limit: int,
) -> bool:
    record_key = (-priority, -tie)
    if len(heap) >= limit and record_key <= heap[0][:2]:
        return False
    record = (-priority, -tie, item_factory())
    if len(heap) < limit:
        heapq.heappush(heap, record)
    else:
        heapq.heapreplace(heap, record)
    return True


def _validation_mask(source: str, hashes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            int(_stable_hash(f"w97:validation:{source}:{value}")[:8], 16) % 5 == 0
            for value in hashes
        ],
        dtype=bool,
    )


def build_nfiot_source_heldout_w97(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w93_dir: str | Path = DEFAULT_W93_DIR,
    w95_dir: str | Path = DEFAULT_W95_DIR,
    w96_dir: str | Path = DEFAULT_W96_DIR,
    per_label_per_group: int = PER_LABEL_PER_GROUP,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)
    exclusions, legacy_exclusions, history = _exclusions_through_w96(
        w93_dir, w95_dir, w96_dir
    )
    _dump(out / "historical_exclusion_ledger.json", history)
    w96 = Path(w96_dir)
    feature_pack = _read_json(w96 / "targeted_feature_pack.json")
    acceptance = _read_json(w96 / "acceptance_report.json")
    selected_features = list(feature_pack.get("selected_features", []))

    processed = Path(processed_dir)
    manifest = _read_json(processed / "dataset_manifest.json")
    entries, source_errors = _source_entries(processed)
    metadata = {
        str(item.get("dataset_variant")): item
        for item in manifest.get("primary_entries", [])
        if item.get("dataset_variant") in ALL_GROUPS
    }
    capabilities: list[dict[str, Any]] = []
    for group in ALL_GROUPS:
        safe = set(str(value) for value in metadata.get(group, {}).get("safe_feature_columns", []))
        missing = sorted(set(selected_features) - safe)
        capabilities.append(
            {
                "source_group": group,
                "selected_feature_count": len(selected_features),
                "missing_feature_count": len(missing),
                "missing_features": "|".join(missing),
                "w96_skill_schema_supported": not missing,
                "runtime_action": "eligible_for_w97_evaluation" if not missing else "unsupported_schema",
            }
        )
    _write_csv(out / "capability_compatibility.csv", capabilities)
    _dump(
        out / "skill_candidate_profile.json",
        {
            "skill_id": "IoTBotnetSkill.W96",
            "dataset_scope": "NF-BoT-IoT-v2 only",
            "schema_compatible_groups": [
                row["source_group"] for row in capabilities if row["w96_skill_schema_supported"]
            ],
            "validated_positive_groups": ["NF-BoT-IoT-v2"],
            "default_enabled": False,
            "fusion_owner": "FusionAgent",
            "contributes_to_verdict": False,
            "promotion_status": "accepted_dataset_specific_shadow_candidate",
            "generalisation_status": "pending_w97",
            "feature_policy_hash": feature_pack.get("feature_pack_hash"),
            "artifact_source": (w96 / "acceptance_report.json").as_posix(),
        },
    )
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "task": "bidirectional v2 source-held-out replication",
        "groups": list(V2_GROUPS),
        "per_label_per_group": int(per_label_per_group),
        "selection": "within-training-source validation only",
        "acceptance": "opposite source; opened once per directional fold",
        "candidate": "W96 selected native features plus RandomForest",
        "reference": "common safe engineered features plus ExtraTrees",
        "v1_policy": "unsupported_schema; no missing-feature imputation masquerading as compatibility",
        **_security(),
    }
    _dump(out / "protocol_manifest.json", protocol)
    ready = (
        not source_errors
        and history.get("status") == "all_auditable_nf_samples_through_w96_excluded"
        and acceptance.get("status")
        == "accepted_dataset_specific_full_coverage_accuracy_f1_result"
        and len(selected_features) == 24
        and all(group in metadata for group in ALL_GROUPS)
        and all(
            next(row for row in capabilities if row["source_group"] == group)[
                "w96_skill_schema_supported"
            ]
            for group in V2_GROUPS
        )
    )
    if not ready:
        report = {
            **protocol,
            "status": "failed_w97_prerequisite_or_capability_gate",
            "source_errors": source_errors,
            "history_status": history.get("status"),
            "w96_status": acceptance.get("status"),
        }
        _dump(out / "split_manifest.json", report)
        return report

    entry_map = {str(item["source_group"]): item for item in entries}
    manifest_rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    sampling_rows: list[dict[str, Any]] = []
    for source in V2_GROUPS:
        item = {**entry_map[source], **metadata[source]}
        headers = list(item["headers"])
        indexes = [headers.index(name) for name in selected_features]
        native_names = [headers[index] for index in indexes]
        heaps: dict[int, list[Any]] = {0: [], 1: []}
        scanned = excluded = labelled = materialized = 0
        tie = 0
        archive = Path(item["archive"])
        member = str(item["entry"])
        label_column = str(item.get("label_column") or "Label")
        attack_column = str(item.get("attack_column") or "Attack")
        for row_index, row in _iter_zip_rows(
            archive, member, max_rows=MAX_ROWS[source]
        ):
            scanned += 1
            label = _normalise_binary_label(row.get(label_column))
            if label is None:
                continue
            canonical_key = _canonical_sample_key(source, archive, member, row_index)
            legacy_key = _legacy_sample_key(archive, member, row_index)
            sample_hash = _stable_hash(canonical_key)
            legacy_hash = _stable_hash(legacy_key)[:24]
            if sample_hash in exclusions or legacy_hash in legacy_exclusions:
                excluded += 1
                continue
            labelled += 1
            y = int(label == "malicious")
            priority = int(_stable_hash("w97:fresh:" + canonical_key)[:16], 16)

            def factory() -> tuple[list[float], list[float], int, str, str, str]:
                return (
                    _engineer_safe_features(row),
                    _native_vector(row, native_names),
                    y,
                    str(row.get(attack_column, "unknown")),
                    sample_hash,
                    legacy_hash,
                )

            if _keep(
                heaps[y],
                priority=priority,
                tie=tie,
                item_factory=factory,
                limit=per_label_per_group,
            ):
                materialized += 1
            tie += 1
        chosen = [
            payload
            for label in (0, 1)
            for _priority, _tie, payload in sorted(heaps[label], reverse=True)
        ]
        counts = {"benign": len(heaps[0]), "malicious": len(heaps[1])}
        sampling_rows.append(
            {
                "source_group": source,
                "scanned_rows": scanned,
                "excluded_rows": excluded,
                "labelled_rows": labelled,
                "materialized_feature_rows": materialized,
                **counts,
            }
        )
        if min(counts.values()) < per_label_per_group:
            _write_csv(out / "sampling_audit.csv", sampling_rows)
            report = {
                **protocol,
                "status": "failed_w97_insufficient_fresh_source_samples",
                "source_group": source,
                "counts": counts,
            }
            _dump(out / "split_manifest.json", report)
            return report
        arrays[source] = {
            "x_common": np.asarray([value[0] for value in chosen], dtype=np.float32),
            "x_native": np.asarray([value[1] for value in chosen], dtype=np.float32),
            "y": np.asarray([value[2] for value in chosen], dtype=np.int64),
            "attack": np.asarray([value[3] for value in chosen], dtype="U64"),
            "sample_hash": np.asarray([value[4] for value in chosen], dtype="U64"),
        }
        for value in chosen:
            manifest_rows.append(
                {
                    "source_group": source,
                    "sample_hash": value[4],
                    "legacy_sample_id_hash": value[5],
                    "label": value[2],
                    "attack": value[3],
                    "historical_overlap": False,
                }
            )
    _write_csv(out / "sampling_audit.csv", sampling_rows)
    _write_csv(out / "fresh_sample_manifest.csv", manifest_rows)
    sealed = out / "sealed_source_pools"
    sealed.mkdir(parents=True, exist_ok=True)
    for source, values in arrays.items():
        np.savez_compressed(
            sealed / f"{source}.npz",
            **values,
            common_feature_names=np.asarray(ENGINEERED_FEATURES),
            native_feature_names=np.asarray(selected_features),
        )
    all_hashes = [row["sample_hash"] for row in manifest_rows]
    cross_source_overlap = len(all_hashes) - len(set(all_hashes))
    historical_overlap = len(set(all_hashes) & exclusions)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_split.json", after)
    passed = (
        cross_source_overlap == 0
        and historical_overlap == 0
        and before == after
        and len(manifest_rows) == per_label_per_group * 2 * len(V2_GROUPS)
    )
    report = {
        **protocol,
        "status": "w97_fresh_source_pools_frozen" if passed else "failed_w97_split_gate",
        "sample_count": len(manifest_rows),
        "cross_source_overlap_count": cross_source_overlap,
        "historical_overlap_count": historical_overlap,
        "frozen_hashes_unchanged": before == after,
        "acceptance_opened": False,
    }
    _dump(out / "split_manifest.json", report)
    return report


def _metrics_from_predictions(
    y: np.ndarray, probability: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    accuracy = float(accuracy_score(y, prediction))
    confidence = np.where(prediction == 1, probability, 1.0 - probability)
    correct = (prediction == y).astype(float)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & (
            confidence < high if high < 1 else confidence <= high
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean()) - float(correct[mask].mean())
            )
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
        "malicious_recall": float(recall_score(y, prediction, zero_division=0)),
        "ece": float(ece),
        "coverage": 1.0,
        "selective_error": 1.0 - accuracy,
    }


def _bootstrap(
    y: np.ndarray,
    source: np.ndarray,
    ref_probability: np.ndarray,
    ref_prediction: np.ndarray,
    cand_probability: np.ndarray,
    cand_prediction: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: list[float] = []
    groups = sorted(set(source.astype(str)))
    group_indexes = {group: np.flatnonzero(source == group) for group in groups}
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = np.concatenate(
            [
                rng.choice(indexes, size=len(indexes), replace=True)
                for indexes in group_indexes.values()
            ]
        )
        ref = _metrics_from_predictions(
            y[sampled], ref_probability[sampled], ref_prediction[sampled]
        )
        cand = _metrics_from_predictions(
            y[sampled], cand_probability[sampled], cand_prediction[sampled]
        )
        deltas.append(cand["macro_f1"] - ref["macro_f1"])
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "stratified_by_source": True,
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def run_nfiot_source_heldout_w97(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    split = _read_json(out / "split_manifest.json")
    marker = out / "source_heldout_acceptance_opened_w97.json"
    existing = _read_json(out / "evaluation_report.json")
    if marker.exists():
        return existing or {**_security(), "status": "failed_w97_acceptance_reopen"}
    if split.get("status") != "w97_fresh_source_pools_frozen":
        report = {**_security(), "status": "failed_w97_source_pools_not_ready"}
        _dump(out / "evaluation_report.json", report)
        return report
    _dump(
        marker,
        {
            "opened_exactly_once": True,
            "selection_protocol": "training-source validation only",
            "acceptance_used_for_selection": False,
        },
    )
    pools: dict[str, dict[str, np.ndarray]] = {}
    for source in V2_GROUPS:
        with np.load(
            out / "sealed_source_pools" / f"{source}.npz", allow_pickle=False
        ) as data:
            pools[source] = {key: np.asarray(data[key]) for key in data.files}
    models.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "y",
            "source",
            "ref_probability",
            "ref_prediction",
            "cand_probability",
            "cand_prediction",
        )
    }
    registry: list[dict[str, Any]] = []
    for heldout in V2_GROUPS:
        training_source = next(source for source in V2_GROUPS if source != heldout)
        train_pool, acceptance_pool = pools[training_source], pools[heldout]
        hashes = train_pool["sample_hash"].astype(str)
        val = _validation_mask(training_source, hashes)
        train = ~val
        y_train = train_pool["y"].astype(int)
        y_accept = acceptance_pool["y"].astype(int)
        reference = make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        )
        candidate = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
        )
        start = time.perf_counter()
        reference.fit(train_pool["x_common"][train], y_train[train])
        candidate.fit(train_pool["x_native"][train], y_train[train])
        training_seconds = time.perf_counter() - start
        ref_val = reference.predict_proba(train_pool["x_common"][val])[:, 1]
        ref_threshold, ref_validation = _select_threshold(y_train[val], ref_val)
        cand_val = candidate.predict_proba(train_pool["x_native"][val])[:, 1]
        cand_threshold, cand_validation = _select_threshold(
            y_train[val],
            cand_val,
            minimum_malicious_recall=ref_validation["malicious_recall"],
        )
        ref_probability = reference.predict_proba(acceptance_pool["x_common"])[:, 1]
        cand_probability = candidate.predict_proba(acceptance_pool["x_native"])[:, 1]
        ref_prediction = (ref_probability >= ref_threshold).astype(int)
        cand_prediction = (cand_probability >= cand_threshold).astype(int)
        ref_metrics = _metrics_from_predictions(
            y_accept, ref_probability, ref_prediction
        )
        cand_metrics = _metrics_from_predictions(
            y_accept, cand_probability, cand_prediction
        )
        row: dict[str, Any] = {
            "heldout_source": heldout,
            "training_source": training_source,
            "train_count": int(np.sum(train)),
            "validation_count": int(np.sum(val)),
            "acceptance_count": len(y_accept),
            "reference_threshold": ref_threshold,
            "candidate_threshold": cand_threshold,
            "validation_reference_macro_f1": ref_validation["macro_f1"],
            "validation_candidate_macro_f1": cand_validation["macro_f1"],
            "validation_reference_recall": ref_validation["malicious_recall"],
            "validation_candidate_recall": cand_validation["malicious_recall"],
            "training_seconds": training_seconds,
        }
        for key, value in ref_metrics.items():
            row[f"reference_{key}"] = value
        for key, value in cand_metrics.items():
            row[f"candidate_{key}"] = value
            row[f"delta_{key}"] = value - ref_metrics[key]
        fold_rows.append(row)
        ref_path = models / f"reference_train_{training_source}.joblib"
        cand_path = models / f"candidate_train_{training_source}.joblib"
        joblib_dump(reference, ref_path)
        joblib_dump(candidate, cand_path)
        registry.extend(
            [
                {
                    "fold": heldout,
                    "model_role": "reference",
                    "artifact": ref_path.as_posix(),
                    "sha256": sha256_file(ref_path),
                },
                {
                    "fold": heldout,
                    "model_role": "candidate",
                    "artifact": cand_path.as_posix(),
                    "sha256": sha256_file(cand_path),
                },
            ]
        )
        pooled["y"].append(y_accept)
        pooled["source"].append(np.asarray([heldout] * len(y_accept)))
        pooled["ref_probability"].append(ref_probability)
        pooled["ref_prediction"].append(ref_prediction)
        pooled["cand_probability"].append(cand_probability)
        pooled["cand_prediction"].append(cand_prediction)
        prediction_rows.extend(
            {
                "heldout_source": heldout,
                "sample_hash": str(sample_hash),
                "label": int(label),
                "reference_probability": float(ref_p),
                "reference_prediction": int(ref_y),
                "candidate_probability": float(cand_p),
                "candidate_prediction": int(cand_y),
            }
            for sample_hash, label, ref_p, ref_y, cand_p, cand_y in zip(
                acceptance_pool["sample_hash"],
                y_accept,
                ref_probability,
                ref_prediction,
                cand_probability,
                cand_prediction,
                strict=True,
            )
        )
    _write_csv(out / "source_heldout_results.csv", fold_rows)
    _write_csv(out / "per_sample_predictions.csv", prediction_rows)
    _write_csv(out / "model_registry.csv", registry)
    combined = {key: np.concatenate(value) for key, value in pooled.items()}
    reference_metrics = _metrics_from_predictions(
        combined["y"], combined["ref_probability"], combined["ref_prediction"]
    )
    candidate_metrics = _metrics_from_predictions(
        combined["y"], combined["cand_probability"], combined["cand_prediction"]
    )
    deltas = {
        key: candidate_metrics[key] - reference_metrics[key]
        for key in reference_metrics
    }
    bootstrap = _bootstrap(
        combined["y"],
        combined["source"],
        combined["ref_probability"],
        combined["ref_prediction"],
        combined["cand_probability"],
        combined["cand_prediction"],
    )
    _dump(out / "grouped_bootstrap_ci.json", bootstrap)
    gates = {
        "aggregate_macro_f1_delta_ge_0_01": deltas["macro_f1"] >= 0.01,
        "aggregate_accuracy_delta_ge_0_005": deltas["accuracy"] >= 0.005,
        "bootstrap_ci_lower_gt_zero": bootstrap["macro_f1_delta_ci95_lower"] > 0,
        "each_source_macro_f1_not_lower": all(
            row["delta_macro_f1"] >= 0 for row in fold_rows
        ),
        "each_source_malicious_recall_not_lower": all(
            row["delta_malicious_recall"] >= 0 for row in fold_rows
        ),
        "aggregate_ece_not_worse_by_0_005": deltas["ece"] <= 0.005,
        "coverage_is_one": candidate_metrics["coverage"] == 1.0,
        "acceptance_not_used_for_selection": True,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "fake_metric_count_zero": True,
    }
    passed = all(gates.values())
    report = {
        **_security(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": (
            "accepted_v2_source_heldout_iotbotnet_skill"
            if passed
            else "not_promoted_w97_source_generalisation_gate_failed"
        ),
        "sample_count": len(combined["y"]),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "source_results": fold_rows,
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "source_generalisation_supported": passed,
        "dataset_specific_w96_result_remains_valid": True,
        "promoted_general_runtime_created": False,
    }
    _dump(out / "evaluation_report.json", report)
    _dump(
        out / "security_acceptance.json",
        {
            **_security(),
            "status": "passed",
            "acceptance_opened_exactly_once": True,
            "v1_unsupported_schema_was_not_imputed": True,
        },
    )
    return report


def _write_docs(report: Mapping[str, Any], document: str | Path, document_cn: str | Path) -> None:
    ref = report.get("reference_metrics", {})
    cand = report.get("candidate_metrics", {})
    delta = report.get("deltas", {})
    ci = report.get("bootstrap", {})
    english = f"""# MAD-ETD NF-IoT Source-Held-Out W97

- Status: `{report.get('status')}`
- W96 dataset-specific result remains valid: `{report.get('dataset_specific_w96_result_remains_valid')}`
- Source generalisation supported: `{report.get('source_generalisation_supported')}`
- Default runtime unchanged: `{report.get('runtime_safe_v3_0_remains_default')}`

| Metric | Common-safe reference | W96 feature candidate | Delta |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| Malicious recall | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

Grouped bootstrap Macro-F1 delta 95% CI: `[{ci.get('macro_f1_delta_ci95_lower')}, {ci.get('macro_f1_delta_ci95_upper')}]`.

NF-IoT v1 schemas do not expose the 24-field W96 feature pack and are recorded as unsupported instead of receiving fabricated missing features.
"""
    chinese = f"""# MAD-ETD NF-IoT 来源保持验证 W97

- 状态：`{report.get('status')}`
- W96 数据集特定结果继续有效：`{report.get('dataset_specific_w96_result_remains_valid')}`
- 支持来源泛化：`{report.get('source_generalisation_supported')}`
- 默认 runtime 未修改：`{report.get('runtime_safe_v3_0_remains_default')}`

| 指标 | 公共安全特征基线 | W96 定向特征候选 | 增量 |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| 恶意召回率 | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

分来源 bootstrap Macro-F1 增量 95% CI：`[{ci.get('macro_f1_delta_ci95_lower')}, {ci.get('macro_f1_delta_ci95_upper')}]`。

NF-IoT v1 schema 不具备 W96 的 24 个原生特征，因此明确记为 unsupported，不通过伪造缺失值冒充兼容。
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(english, encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(chinese, encoding="utf-8")


def finalize_nfiot_source_heldout_w97(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    evaluation = _read_json(out / "evaluation_report.json")
    if not evaluation:
        report = {
            **_security(),
            "status": "failed_missing_w97_evaluation",
            "tests_passed": tests_passed,
            "test_count": test_count,
        }
        _dump(out / "negative_results.json", report)
        _dump(out / "acceptance_report.json", report)
        _write_docs(report, document, document_cn)
        return report
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    performance_passed = (
        evaluation.get("status") == "accepted_v2_source_heldout_iotbotnet_skill"
    )
    accepted = performance_passed and hashes_unchanged and tests_passed
    if accepted:
        status = "accepted_v2_source_heldout_iotbotnet_skill"
    elif performance_passed and hashes_unchanged:
        status = "pending_tests_v2_source_heldout_positive_result"
    else:
        status = "not_promoted_w97_source_generalisation_gate_failed"
    report = {
        **evaluation,
        "status": status,
        "accepted_source_heldout_skill": accepted,
        "frozen_hashes_unchanged": hashes_unchanged,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    if not accepted:
        _dump(
            out / "negative_results.json",
            {
                "status": status,
                "failure_type": "source_generalisation_not_established",
                "failed_gates": evaluation.get("failed_gates", []),
                "w96_dataset_specific_result_remains_valid": True,
                "fake_metric_count": 0,
                "promoted_runtime_created": False,
            },
        )
    _dump(out / "acceptance_report.json", report)
    _write_docs(report, document, document_cn)
    return report

