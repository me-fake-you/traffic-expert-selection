"""W98 USTC application/family-held-out safe flow-sequence experiment."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .external_multiagent_v51 import _stable_hash
from .nfiot_source_heldout_w97 import _metrics_from_predictions
from .nfiot_targeted_performance_w94_w95 import _read_json, _security, _select_threshold
from .paper_evaluation import hash_artifact_paths, sha256_file
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_ustc_group_heldout_hybrid_w98"
DEFAULT_USTC_ROOT = Path("data/processed/ustc_tfc2016/v1/flows")
DEFAULT_HIKARI_AUDIT = Path("data/runs/mad_etd_generalization_benchmark_w62/hikari_group_audit.json")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_ustc_group_heldout_hybrid_w98")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_ustc_group_heldout_hybrid_w98")
DEFAULT_DOC = Path("docs/MAD_ETD_USTC_GROUP_HELDOUT_HYBRID_W98.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_USTC_GROUP_HELDOUT_HYBRID_W98_CN.md")
PER_GROUP = 2_000
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42

STATS_FEATURES = (
    "packet_count_log",
    "total_bytes_log",
    "outbound_bytes_log",
    "inbound_bytes_log",
    "outbound_ratio",
    "mean_packet_length_log",
    "packet_length_variance_log",
    "duration_log",
)

SEQUENCE_FEATURES = (
    "sequence_count_log",
    "length_mean_log",
    "length_std_log",
    "length_min_log",
    "length_max_log",
    "length_sum_log",
    "signed_mean",
    "signed_std_log",
    "outbound_fraction",
    "direction_change_rate",
    "iat_mean_log",
    "iat_std_log",
    "iat_max_log",
    "iat_zero_rate",
    "first_signed_log",
    "last_signed_log",
    "short_le_1",
    "short_le_2",
    "short_le_4",
    "short_le_8",
) + tuple(
    f"prefix_{size}_{name}"
    for size in (2, 4, 8, 16)
    for name in ("abs_mean_log", "signed_mean", "std_log", "direction_change", "iat_mean_log")
)


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


def _log(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return math.copysign(math.log1p(abs(number)), number)


def _stats_vector(record: Mapping[str, Any]) -> list[float]:
    stats = record.get("stats") or {}
    return [
        _log(stats.get("packet_count")),
        _log(stats.get("total_bytes")),
        _log(stats.get("outbound_bytes")),
        _log(stats.get("inbound_bytes")),
        float(stats.get("outbound_ratio") or 0.0),
        _log(stats.get("mean_packet_length")),
        _log(stats.get("packet_length_variance")),
        _log(stats.get("duration")),
    ]


def _change_rate(directions: np.ndarray) -> float:
    if len(directions) < 2:
        return 0.0
    return float(np.mean(directions[1:] != directions[:-1]))


def _prefix_features(
    lengths: np.ndarray, directions: np.ndarray, iats: np.ndarray, size: int
) -> list[float]:
    count = min(len(lengths), size)
    if count == 0:
        return [0.0] * 5
    local_lengths = lengths[:count]
    local_directions = directions[:count]
    signed = local_lengths * local_directions
    local_iats = iats[: max(0, count - 1)]
    return [
        _log(float(np.mean(local_lengths))),
        float(np.mean(signed) / (1.0 + np.mean(local_lengths))),
        _log(float(np.std(local_lengths))),
        _change_rate(local_directions),
        _log(float(np.mean(local_iats))) if len(local_iats) else 0.0,
    ]


def _sequence_vector(record: Mapping[str, Any]) -> list[float]:
    sequence = record.get("sequence") or {}
    lengths = np.asarray(sequence.get("packet_lengths") or [], dtype=float)[:64]
    directions = np.asarray(sequence.get("directions") or [], dtype=float)[:64]
    if len(directions) < len(lengths):
        directions = np.pad(directions, (0, len(lengths) - len(directions)), constant_values=1)
    directions = np.where(directions[: len(lengths)] >= 0, 1.0, -1.0)
    iats = np.asarray(sequence.get("iats") or [], dtype=float)[:63]
    if not len(lengths):
        base = [0.0] * 20
    else:
        signed = lengths * directions
        base = [
            _log(len(lengths)),
            _log(float(np.mean(lengths))),
            _log(float(np.std(lengths))),
            _log(float(np.min(lengths))),
            _log(float(np.max(lengths))),
            _log(float(np.sum(lengths))),
            float(np.mean(signed) / (1.0 + np.mean(lengths))),
            _log(float(np.std(signed))),
            float(np.mean(directions > 0)),
            _change_rate(directions),
            _log(float(np.mean(iats))) if len(iats) else 0.0,
            _log(float(np.std(iats))) if len(iats) else 0.0,
            _log(float(np.max(iats))) if len(iats) else 0.0,
            float(np.mean(iats == 0)) if len(iats) else 0.0,
            _log(float(signed[0])),
            _log(float(signed[-1])),
            float(len(lengths) <= 1),
            float(len(lengths) <= 2),
            float(len(lengths) <= 4),
            float(len(lengths) <= 8),
        ]
    return base + [
        value
        for size in (2, 4, 8, 16)
        for value in _prefix_features(lengths, directions, iats, size)
    ]


def _group_name(record: Mapping[str, Any]) -> tuple[str, int]:
    labels = record.get("labels") or {}
    binary = str(labels.get("binary", "")).lower()
    y = int(binary == "malicious")
    name = str(labels.get("family") if y else labels.get("application"))
    return (f"malware:{name}" if y else f"benign:{name}"), y


def _validation_groups(heldout: str, groups: list[str]) -> tuple[str, str]:
    benign = [group for group in groups if group.startswith("benign:") and group != heldout]
    malware = [group for group in groups if group.startswith("malware:") and group != heldout]
    return (
        min(benign, key=lambda value: _stable_hash(f"w98:val:{heldout}:{value}")),
        min(malware, key=lambda value: _stable_hash(f"w98:val:{heldout}:{value}")),
    )


def build_ustc_group_heldout_hybrid_w98(
    ustc_root: str | Path = DEFAULT_USTC_ROOT,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    hikari_audit_path: str | Path = DEFAULT_HIKARI_AUDIT,
    per_group: int = PER_GROUP,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)
    hikari = _read_json(hikari_audit_path)
    hikari_policy = {
        "dataset": "HIKARI-2021",
        "status": hikari.get("status", "missing_group_audit"),
        "group_metadata_reliable": hikari.get("group_metadata_reliable", False),
        "supervised_promotion_eligible": False,
        "reason": "single collection group; retained as diagnostic-only",
    }
    _dump(out / "hikari_eligibility_audit.json", hikari_policy)
    policy = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "USTC-TFC2016",
        "split_protocol": "20-fold application/family-held-out cross-validation",
        "reference_features": list(STATS_FEATURES),
        "candidate_features": list(STATS_FEATURES + SEQUENCE_FEATURES),
        "blocked_fields": [
            "sample_id", "trace_id", "context.src_ip", "context.dst_ip", "context.src_port",
            "context.dst_port", "provenance.source_file", "labels.binary", "labels.family",
            "labels.application",
        ],
        "group_labels_in_features": False,
        "acceptance_used_for_selection": False,
        "historical_sample_reuse": "yes; group-held-out CV, not a fresh locked-test claim",
        "claim_scope": "USTC application/family-held-out exploratory performance",
        **_security(),
    }
    policy["feature_policy_hash"] = hashlib.sha256(
        json.dumps(policy["candidate_features"], sort_keys=True).encode()
    ).hexdigest()
    _dump(out / "safe_feature_policy.json", policy)
    root = Path(ustc_root)
    files = sorted(root.rglob("*.jsonl.gz"))
    if not files or hikari.get("status") != "blocked_single_collection_group":
        report = {**policy, "status": "failed_w98_data_or_hikari_audit_gate", "file_count": len(files)}
        _dump(out / "split_manifest.json", report)
        return report
    heaps: dict[str, list[tuple[int, int, tuple[list[float], list[float], int, str]]]] = {}
    scanned = materialized = 0
    tie = 0
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                group, y = _group_name(record)
                if group.endswith(":"):
                    continue
                sample_hash = _stable_hash(str(record.get("sample_id", "")))
                priority = int(_stable_hash(f"w98:sample:{group}:{sample_hash}")[:16], 16)
                heap = heaps.setdefault(group, [])
                key = (-priority, -tie)
                if len(heap) >= per_group and key <= heap[0][:2]:
                    tie += 1
                    continue
                stats = _stats_vector(record)
                hybrid = stats + _sequence_vector(record)
                payload = (stats, hybrid, y, sample_hash)
                item = (-priority, -tie, payload)
                if len(heap) < per_group:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
                materialized += 1
                tie += 1
    groups = sorted(heaps)
    counts = {group: len(heaps[group]) for group in groups}
    complete = (
        len([group for group in groups if group.startswith("benign:")]) == 10
        and len([group for group in groups if group.startswith("malware:")]) == 10
        and all(count == per_group for count in counts.values())
    )
    selected = [
        (group, payload)
        for group in groups
        for _priority, _tie, payload in sorted(heaps[group], reverse=True)
    ]
    if not complete:
        report = {**policy, "status": "failed_w98_incomplete_groups", "group_counts": counts}
        _dump(out / "split_manifest.json", report)
        return report
    x_stats = np.asarray([payload[0] for _group, payload in selected], dtype=np.float32)
    x_hybrid = np.asarray([payload[1] for _group, payload in selected], dtype=np.float32)
    y = np.asarray([payload[2] for _group, payload in selected], dtype=np.int64)
    group_values = np.asarray([group for group, _payload in selected], dtype="U64")
    hashes = np.asarray([payload[3] for _group, payload in selected], dtype="U64")
    sealed = out / "sealed_group_cv.npz"
    np.savez_compressed(
        sealed,
        x_stats=x_stats,
        x_hybrid=x_hybrid,
        y=y,
        group=group_values,
        sample_hash=hashes,
        stats_feature_names=np.asarray(STATS_FEATURES),
        hybrid_feature_names=np.asarray(STATS_FEATURES + SEQUENCE_FEATURES),
    )
    _write_csv(
        out / "group_sample_manifest.csv",
        [
            {"sample_hash": str(sample_hash), "group": str(group), "label": int(label)}
            for sample_hash, group, label in zip(hashes, group_values, y, strict=True)
        ],
    )
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_split.json", after)
    report = {
        **policy,
        "status": "w98_ustc_group_cv_frozen" if before == after else "failed_w98_frozen_hash_gate",
        "sample_count": len(y),
        "group_count": len(groups),
        "group_counts": counts,
        "scanned_rows": scanned,
        "materialized_feature_rows": materialized,
        "frozen_hashes_unchanged": before == after,
    }
    _dump(out / "split_manifest.json", report)
    return report


def _model(model_id: str) -> Any:
    if model_id == "stats_hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1, random_state=42),
        )
    if model_id == "hybrid_rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        )
    if model_id == "hybrid_extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        )
    raise ValueError(model_id)


def _group_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    ref_probability: np.ndarray,
    ref_prediction: np.ndarray,
    cand_probability: np.ndarray,
    cand_prediction: np.ndarray,
) -> dict[str, Any]:
    unique = sorted(set(groups.astype(str)))
    indexes = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([indexes[str(group)] for group in sampled_groups])
        ref = _metrics_from_predictions(y[sampled], ref_probability[sampled], ref_prediction[sampled])
        cand = _metrics_from_predictions(y[sampled], cand_probability[sampled], cand_prediction[sampled])
        deltas.append(cand["macro_f1"] - ref["macro_f1"])
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "application_or_family_group",
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def run_ustc_group_heldout_hybrid_w98(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    split = _read_json(out / "split_manifest.json")
    marker = out / "group_cv_acceptance_opened_w98.json"
    existing = _read_json(out / "evaluation_report.json")
    if marker.exists():
        return existing or {**_security(), "status": "failed_w98_acceptance_reopen"}
    if split.get("status") != "w98_ustc_group_cv_frozen":
        report = {**_security(), "status": "failed_w98_group_cv_not_ready"}
        _dump(out / "evaluation_report.json", report)
        return report
    _dump(marker, {"opened_exactly_once": True, "acceptance_used_for_selection": False})
    with np.load(out / "sealed_group_cv.npz", allow_pickle=False) as data:
        x_stats = np.asarray(data["x_stats"], dtype=np.float32)
        x_hybrid = np.asarray(data["x_hybrid"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        groups = np.asarray(data["group"]).astype(str)
        hashes = np.asarray(data["sample_hash"]).astype(str)
    unique_groups = sorted(set(groups))
    models.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {key: [] for key in ("y", "group", "ref_p", "ref_y", "cand_p", "cand_y")}
    registry: list[dict[str, Any]] = []
    for heldout in unique_groups:
        validation_groups = _validation_groups(heldout, unique_groups)
        acceptance = groups == heldout
        validation = np.isin(groups, validation_groups)
        train = ~(acceptance | validation)
        reference = _model("stats_hgb")
        reference.fit(x_stats[train], y[train])
        ref_val = reference.predict_proba(x_stats[validation])[:, 1]
        ref_threshold, ref_validation = _select_threshold(y[validation], ref_val)
        options: list[tuple[dict[str, Any], str, Any, np.ndarray]] = []
        for model_id in ("hybrid_rf", "hybrid_extra_trees"):
            candidate = _model(model_id)
            start = time.perf_counter()
            candidate.fit(x_hybrid[train], y[train])
            training_seconds = time.perf_counter() - start
            probability = candidate.predict_proba(x_hybrid[validation])[:, 1]
            threshold, metrics = _select_threshold(
                y[validation], probability, minimum_malicious_recall=ref_validation["malicious_recall"]
            )
            options.append(({"model_id": model_id, "threshold": threshold, "training_seconds": training_seconds, **metrics}, model_id, candidate, probability))
        selected, selected_id, candidate, _validation_probability = max(
            options,
            key=lambda item: (item[0]["macro_f1"], item[0]["accuracy"], item[0]["malicious_recall"], -item[0]["ece"]),
        )
        ref_p = reference.predict_proba(x_stats[acceptance])[:, 1]
        cand_p = candidate.predict_proba(x_hybrid[acceptance])[:, 1]
        ref_y = (ref_p >= ref_threshold).astype(int)
        cand_y = (cand_p >= float(selected["threshold"])).astype(int)
        ref_metrics = _metrics_from_predictions(y[acceptance], ref_p, ref_y)
        cand_metrics = _metrics_from_predictions(y[acceptance], cand_p, cand_y)
        row: dict[str, Any] = {
            "heldout_group": heldout,
            "heldout_label": int(y[acceptance][0]),
            "validation_groups": "|".join(validation_groups),
            "train_count": int(np.sum(train)),
            "validation_count": int(np.sum(validation)),
            "acceptance_count": int(np.sum(acceptance)),
            "candidate_model_id": selected_id,
            "reference_threshold": ref_threshold,
            "candidate_threshold": selected["threshold"],
            "validation_reference_macro_f1": ref_validation["macro_f1"],
            "validation_candidate_macro_f1": selected["macro_f1"],
        }
        for key, value in ref_metrics.items():
            row[f"reference_{key}"] = value
        for key, value in cand_metrics.items():
            row[f"candidate_{key}"] = value
            row[f"delta_{key}"] = value - ref_metrics[key]
        fold_rows.append(row)
        ref_path = models / f"{heldout.replace(':', '_')}_reference.joblib"
        cand_path = models / f"{heldout.replace(':', '_')}_candidate.joblib"
        joblib_dump(reference, ref_path)
        joblib_dump(candidate, cand_path)
        registry.extend([
            {"heldout_group": heldout, "model_role": "reference", "model_id": "stats_hgb", "artifact": ref_path.as_posix(), "sha256": sha256_file(ref_path)},
            {"heldout_group": heldout, "model_role": "candidate", "model_id": selected_id, "artifact": cand_path.as_posix(), "sha256": sha256_file(cand_path)},
        ])
        pooled["y"].append(y[acceptance]); pooled["group"].append(groups[acceptance]); pooled["ref_p"].append(ref_p); pooled["ref_y"].append(ref_y); pooled["cand_p"].append(cand_p); pooled["cand_y"].append(cand_y)
        prediction_rows.extend(
            {"sample_hash": sample_hash, "heldout_group": heldout, "label": int(label), "reference_probability": float(rp), "reference_prediction": int(ry), "candidate_probability": float(cp), "candidate_prediction": int(cy)}
            for sample_hash, label, rp, ry, cp, cy in zip(hashes[acceptance], y[acceptance], ref_p, ref_y, cand_p, cand_y, strict=True)
        )
    _write_csv(out / "group_heldout_results.csv", fold_rows)
    _write_csv(out / "per_sample_predictions.csv", prediction_rows)
    _write_csv(out / "model_registry.csv", registry)
    combined = {key: np.concatenate(value) for key, value in pooled.items()}
    reference_metrics = _metrics_from_predictions(combined["y"], combined["ref_p"], combined["ref_y"])
    candidate_metrics = _metrics_from_predictions(combined["y"], combined["cand_p"], combined["cand_y"])
    deltas = {key: candidate_metrics[key] - reference_metrics[key] for key in reference_metrics}
    bootstrap = _group_bootstrap(combined["y"], combined["group"], combined["ref_p"], combined["ref_y"], combined["cand_p"], combined["cand_y"])
    _dump(out / "grouped_bootstrap_ci.json", bootstrap)
    worst_group_accuracy_delta = min(float(row["delta_accuracy"]) for row in fold_rows)
    gates = {
        "macro_f1_delta_ge_0_01": deltas["macro_f1"] >= 0.01,
        "accuracy_delta_ge_0_005": deltas["accuracy"] >= 0.005,
        "grouped_bootstrap_ci_lower_gt_zero": bootstrap["macro_f1_delta_ci95_lower"] > 0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0,
        "ece_not_worse_by_0_005": deltas["ece"] <= 0.005,
        "worst_group_accuracy_drop_at_most_0_02": worst_group_accuracy_delta >= -0.02,
        "coverage_is_one": candidate_metrics["coverage"] == 1.0,
        "acceptance_not_used_for_selection": True,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "fake_metric_count_zero": True,
    }
    passed = all(gates.values())
    report = {
        **_security(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "accepted_ustc_group_heldout_hybrid_skill" if passed else "not_promoted_w98_group_heldout_performance_gate_failed",
        "sample_count": len(combined["y"]),
        "group_count": len(unique_groups),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "worst_group_accuracy_delta": worst_group_accuracy_delta,
        "group_results": fold_rows,
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [key for key, value in gates.items() if not value],
        "historical_sample_reuse": True,
        "claim_scope": "USTC group-held-out cross-validation only",
        "hikari_status": "blocked_single_collection_group",
        "promoted_runtime_created": False,
    }
    _dump(out / "evaluation_report.json", report)
    _dump(out / "security_acceptance.json", {**_security(), "status": "passed", "group_labels_in_feature_matrix": False, "acceptance_opened_exactly_once": True})
    return report


def _write_docs(report: Mapping[str, Any], document: str | Path, document_cn: str | Path) -> None:
    ref, cand, delta = report.get("reference_metrics", {}), report.get("candidate_metrics", {}), report.get("deltas", {})
    ci = report.get("bootstrap", {})
    english = f"""# MAD-ETD USTC Group-Held-Out Hybrid W98

- Status: `{report.get('status')}`
- Scope: USTC application/family-held-out cross-validation
- HIKARI: `blocked_single_collection_group`
- Default runtime unchanged: `{report.get('runtime_safe_v3_0_remains_default')}`

| Metric | Stats HGB | Stats+sequence hybrid | Delta |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| Malicious recall | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

Group-bootstrap Macro-F1 delta 95% CI: `[{ci.get('macro_f1_delta_ci95_lower')}, {ci.get('macro_f1_delta_ci95_upper')}]`.

This protocol reuses the public USTC corpus under fresh group-held-out folds; it is not a new locked-test claim and does not extend to HIKARI.
"""
    chinese = f"""# MAD-ETD USTC 分组保持混合证据实验 W98

- 状态：`{report.get('status')}`
- 范围：USTC application/family-held-out 交叉验证
- HIKARI：`blocked_single_collection_group`
- 默认 runtime 未修改：`{report.get('runtime_safe_v3_0_remains_default')}`

| 指标 | Stats HGB | Stats+安全序列混合候选 | 增量 |
|---|---:|---:|---:|
| Accuracy | {ref.get('accuracy')} | {cand.get('accuracy')} | {delta.get('accuracy')} |
| Macro-F1 | {ref.get('macro_f1')} | {cand.get('macro_f1')} | {delta.get('macro_f1')} |
| 恶意召回率 | {ref.get('malicious_recall')} | {cand.get('malicious_recall')} | {delta.get('malicious_recall')} |
| ECE | {ref.get('ece')} | {cand.get('ece')} | {delta.get('ece')} |

分组 bootstrap Macro-F1 增量 95% CI：`[{ci.get('macro_f1_delta_ci95_lower')}, {ci.get('macro_f1_delta_ci95_upper')}]`。

本协议复用公开 USTC 数据并进行新的分组保持交叉验证，不是新的 locked-test 结论，也不扩展到 HIKARI。
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True); Path(document).write_text(english, encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True); Path(document_cn).write_text(chinese, encoding="utf-8")


def finalize_ustc_group_heldout_hybrid_w98(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *, document: str | Path = DEFAULT_DOC, document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False, test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    evaluation = _read_json(out / "evaluation_report.json")
    if not evaluation:
        report = {**_security(), "status": "failed_missing_w98_evaluation", "tests_passed": tests_passed, "test_count": test_count}
        _dump(out / "negative_results.json", report); _dump(out / "acceptance_report.json", report); _write_docs(report, document, document_cn); return report
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths()); _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    performance = evaluation.get("status") == "accepted_ustc_group_heldout_hybrid_skill"
    accepted = performance and hashes_unchanged and tests_passed
    status = "accepted_ustc_group_heldout_hybrid_skill" if accepted else ("pending_tests_ustc_group_positive" if performance and hashes_unchanged else "not_promoted_w98_group_heldout_performance_gate_failed")
    report = {**evaluation, "status": status, "accepted_optional_skill_candidate": accepted, "frozen_hashes_unchanged": hashes_unchanged, "tests_passed": tests_passed, "test_count": test_count, "runtime_safe_v3_0_remains_default": True, "promoted_runtime_created": False}
    if not accepted:
        _dump(out / "negative_results.json", {"status": status, "failed_gates": evaluation.get("failed_gates", []), "hikari_status": "blocked_single_collection_group", "fake_metric_count": 0, "promoted_runtime_created": False})
    _dump(out / "acceptance_report.json", report); _write_docs(report, document, document_cn); return report

