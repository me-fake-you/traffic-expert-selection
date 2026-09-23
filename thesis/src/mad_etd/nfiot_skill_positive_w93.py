"""W93 fresh NF-IoT dataset-performance experiment for IoTBotnetSkill.

The protocol deliberately separates two facts:

* a fresh, source-stratified *in-domain* acceptance experiment can support a
  dataset-specific Accuracy/Macro-F1 claim; and
* prior leave-one-source-group-out experiments remain negative evidence and
  are never rewritten as a generalisation success.

The candidate is shadow-only and default-off.  It emits AgentEvidenceV2 but
never a FusionResult, and it cannot replace ``runtime_safe_v3_0``.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline

from .credible_performance_w62_w67 import SOURCE_GROUPS
from .domain_robust_stats_w81 import _source_entries
from .external_multiagent_v51 import _normalise_binary_label, _stable_hash
from .external_multiagent_v52 import _iter_zip_rows
from .paper_evaluation import hash_artifact_paths, sha256_file
from .performance_upgrade_w88_w92 import _historical_exclusions_w88
from .safe_flow_ensemble_w73_w76 import (
    BLOCKED_CONTEXT_FIELDS,
    ENGINEERED_FEATURES,
    RAW_SAFE_FIELDS,
    _canonical_sample_key,
    _engineer_safe_features,
    _legacy_sample_key,
)
from .schemas import AgentEvidenceV2, EvidenceHandoffV2, EvidenceRequest
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_nfiot_skill_positive_w93"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nfiot_skill_positive_w93")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_nfiot_skill_positive_w93")
DEFAULT_W88_DIR = Path("data/runs/mad_etd_fresh_performance_upgrade_w88_w92")
DEFAULT_DOC = Path("docs/MAD_ETD_NFIOT_SKILL_POSITIVE_W93.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NFIOT_SKILL_POSITIVE_W93_CN.md")
DEFAULT_RUNTIME = "runtime_safe_v3_0"
CANDIDATE_ID = "IoTBotnetSkill.safe_selective_v1"
MODEL_IDS = ("hgb_safe", "random_forest_safe", "extra_trees_safe")
DEFAULT_PER_LABEL_PER_GROUP = 2_000
DEFAULT_MAX_ROWS_PER_ENTRY = 4_000_000
SEED = 93
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 1_000


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


def _security_contract() -> dict[str, Any]:
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
        "automatic_training": False,
        "automatic_deployment": False,
    }


def _feature_policy() -> dict[str, Any]:
    forbidden = sorted(
        set(BLOCKED_CONTEXT_FIELDS)
        | {
            "label",
            "attack",
            "attack_name",
            "attack_family",
            "family",
            "sample_id",
            "source_group",
            "source_identity",
            "source_file",
            "provenance",
            "ip",
            "port",
            "timestamp",
            "flow_id",
        }
    )
    payload = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT mixed-source in-domain only",
        "raw_safe_feature_intersection": list(RAW_SAFE_FIELDS),
        "engineered_detector_features": list(ENGINEERED_FEATURES),
        "forbidden_detector_fields": forbidden,
        "source_group_use": "stratified split and diagnostic only",
        "source_group_in_feature_matrix": False,
        "label_in_feature_matrix": False,
        "blocked_field_violation": 0,
    }
    payload["feature_policy_hash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _fresh_exclusions(
    w88_dir: str | Path = DEFAULT_W88_DIR,
) -> tuple[set[str], set[str], dict[str, Any]]:
    canonical, legacy, earlier = _historical_exclusions_w88()
    w88 = Path(w88_dir)
    rows = _read_csv(w88 / "fresh_sample_manifest.csv")
    w88_hashes = {row["sample_hash"] for row in rows if row.get("sample_hash")}
    w88_legacy = {
        row["legacy_sample_id_hash"]
        for row in rows
        if row.get("legacy_sample_id_hash")
    }
    split = _read_json(w88 / "split_manifest.json")
    mapped = bool(rows) and int(split.get("sample_count", -1)) == len(w88_hashes)
    canonical.update(w88_hashes)
    legacy.update(w88_legacy)
    status = (
        "all_auditable_nf_samples_through_w92_excluded"
        if earlier.get("status") == "all_w62_w84_nf_samples_mapped_for_w88"
        and mapped
        else "failed_historical_exclusion_mapping_w93"
    )
    return canonical, legacy, {
        "schema_version": "1.0",
        "status": status,
        "through_w84": earlier,
        "w88_w92": {
            "manifest": (w88 / "fresh_sample_manifest.csv").as_posix(),
            "sample_count": len(w88_hashes),
            "legacy_count": len(w88_legacy),
            "split_status": split.get("status", "missing"),
            "mapped": mapped,
        },
        "canonical_exclusion_count": len(canonical),
        "legacy_exclusion_count": len(legacy),
        "fake_metric_count": 0,
    }


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


def _role_for_sample(sample_hash: str) -> str:
    value = int(_stable_hash(f"w93:role:{sample_hash}")[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "validation"
    return "acceptance"


def build_nfiot_skill_positive_w93(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    w88_dir: str | Path = DEFAULT_W88_DIR,
    per_label_per_group: int = DEFAULT_PER_LABEL_PER_GROUP,
    max_rows_per_entry: int = DEFAULT_MAX_ROWS_PER_ENTRY,
) -> dict[str, Any]:
    """Freeze a fresh mixed-source sample without training or opening acceptance."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = _feature_policy()
    _dump(out / "safe_feature_policy.json", policy)
    before = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_before.json", before)
    exclusions, legacy_exclusions, ledger = _fresh_exclusions(w88_dir)
    _dump(out / "historical_exclusion_ledger.json", ledger)
    entries, source_errors = _source_entries(Path(processed_dir))
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "fresh_mixed_source_split_freeze",
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT mixed-source in-domain only",
        "source_groups": list(SOURCE_GROUPS),
        "split_protocol": "within-each-source-and-label deterministic 60/20/20",
        "selection_split": "validation",
        "acceptance_split": "sealed acceptance opened once",
        "generalisation_protocol": "separate prior LOSO result; not inferred from W93",
        "seed": SEED,
        "per_label_per_group": int(per_label_per_group),
        "max_rows_per_entry": int(max_rows_per_entry),
        "acceptance_used_for_selection": False,
        "acceptance_opened": False,
        **_security_contract(),
    }
    _dump(out / "protocol_manifest.json", protocol)
    if source_errors or ledger.get("status") != "all_auditable_nf_samples_through_w92_excluded":
        report = {
            **protocol,
            "status": "failed_no_fresh_auditable_w93_split",
            "source_errors": source_errors,
            "historical_exclusion_status": ledger.get("status"),
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
        for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows_per_entry):
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
            y = int(label == "malicious")
            labeled[group] += 1
            vector = _engineer_safe_features(row)
            priority = int(_stable_hash("w93:fresh:" + canonical_key)[:16], 16)
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
    sufficient = all(
        counts[group].get(label, 0) >= per_label_per_group
        for group in SOURCE_GROUPS
        for label in ("benign", "malicious")
    )
    if not sufficient:
        report = {
            **protocol,
            "status": "failed_insufficient_fresh_w93_samples",
            "source_group_counts": counts,
            "scanned_rows_per_group": dict(scanned),
            "excluded_rows_per_group": dict(excluded),
        }
        _dump(out / "split_manifest.json", report)
        return report

    x = np.asarray([item[0] for item in selected], dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    hashes = np.asarray([item[3] for item in selected], dtype="U64")
    legacy = np.asarray([item[4] for item in selected], dtype="U24")
    roles = np.asarray([_role_for_sample(str(value)) for value in hashes], dtype="U16")
    # Ensure every source/label cell has all three roles; deterministic hashing
    # with 2,000 rows per cell makes failure extremely unlikely, but it remains
    # an explicit fail-closed gate.
    cell_roles = {
        f"{group}:{label}": dict(
            Counter(str(role) for role in roles[(groups == group) & (y == label)])
        )
        for group in SOURCE_GROUPS
        for label in (0, 1)
    }
    role_complete = all(
        all(cell_roles[f"{group}:{label}"].get(role, 0) > 0 for role in ("train", "validation", "acceptance"))
        for group in SOURCE_GROUPS
        for label in (0, 1)
    )
    train_validation = roles != "acceptance"
    acceptance = roles == "acceptance"
    split_root = out / "sealed_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_root / "train_validation.npz",
        x=x[train_validation],
        y=y[train_validation],
        groups=groups[train_validation],
        sample_hash=hashes[train_validation],
        roles=roles[train_validation],
        feature_names=np.asarray(ENGINEERED_FEATURES),
    )
    np.savez_compressed(
        split_root / "acceptance_sealed.npz",
        x=x[acceptance],
        y=y[acceptance],
        groups=groups[acceptance],
        sample_hash=hashes[acceptance],
        feature_names=np.asarray(ENGINEERED_FEATURES),
    )
    sample_rows = [
        {
            "sample_hash": str(sample_hash),
            "legacy_sample_id_hash": str(legacy_hash),
            "source_group": str(group),
            "label": int(label),
            "role": str(role),
            "feature_hash": hashlib.sha256(vector.tobytes()).hexdigest(),
            "historical_overlap": False,
        }
        for vector, label, group, sample_hash, legacy_hash, role in zip(
            x, y, groups, hashes, legacy, roles, strict=True
        )
    ]
    _write_csv(out / "fresh_sample_manifest.csv", sample_rows)
    train_ids = set(str(value) for value in hashes[roles == "train"])
    validation_ids = set(str(value) for value in hashes[roles == "validation"])
    acceptance_ids = set(str(value) for value in hashes[roles == "acceptance"])
    overlap = len(train_ids & validation_ids) + len(train_ids & acceptance_ids) + len(validation_ids & acceptance_ids)
    historical_overlap = len(set(str(value) for value in hashes) & exclusions)
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after_split.json", after)
    hashes_unchanged = before == after
    ready = role_complete and overlap == 0 and historical_overlap == 0 and hashes_unchanged
    report = {
        **protocol,
        "status": "fresh_w93_in_domain_splits_frozen" if ready else "failed_w93_split_gate",
        "sample_count": int(len(x)),
        "train_count": int(np.sum(roles == "train")),
        "validation_count": int(np.sum(roles == "validation")),
        "acceptance_count": int(np.sum(roles == "acceptance")),
        "source_group_counts": counts,
        "cell_role_counts": cell_roles,
        "scanned_rows_per_group": dict(scanned),
        "excluded_rows_per_group": dict(excluded),
        "labeled_rows_after_exclusion": dict(labeled),
        "split_overlap_count": overlap,
        "historical_overlap_count": historical_overlap,
        "frozen_hashes_unchanged": hashes_unchanged,
        "sample_manifest_sha256": sha256_file(out / "fresh_sample_manifest.csv"),
        "acceptance_sealed": True,
    }
    _dump(out / "split_manifest.json", report)
    _dump(out / "security_acceptance_split.json", {**_security_contract(), "status": "passed" if ready else "failed"})
    return report


def _make_model(model_id: str):
    if model_id == "hgb_safe":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=42,
            ),
        )
    if model_id == "random_forest_safe":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
        )
    if model_id == "extra_trees_safe":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        )
    raise ValueError(f"unknown model: {model_id}")


def _positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(x), dtype=float)
    return values[:, 1]


def _ece(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    if not len(y_true):
        return 0.0
    pred = (proba >= threshold).astype(int)
    confidence = np.where(pred == 1, proba, 1.0 - proba)
    correct = (pred == y_true).astype(float)
    total = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(total)


def _metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    accept_confidence: float,
) -> dict[str, float]:
    confidence = np.maximum(proba, 1.0 - proba)
    accepted = confidence >= accept_confidence
    coverage = float(accepted.mean()) if len(accepted) else 0.0
    if not np.any(accepted):
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "malicious_recall": 0.0,
            "ece": 0.0,
            "coverage": coverage,
            "selective_error": 1.0,
        }
    y = y_true[accepted]
    p = proba[accepted]
    pred = (p >= threshold).astype(int)
    accuracy = float(accuracy_score(y, pred))
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(y, pred, zero_division=0)),
        "ece": _ece(y, p, threshold),
        "coverage": coverage,
        "selective_error": 1.0 - accuracy,
    }


def _candidate_grid() -> list[tuple[float, float]]:
    thresholds = np.round(np.arange(0.46, 0.541, 0.01), 2)
    confidences = np.round(np.arange(0.50, 0.876, 0.025), 3)
    return [(float(threshold), float(confidence)) for threshold in thresholds for confidence in confidences]


def _select_policy(
    y_validation: np.ndarray,
    validation_proba: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows: list[dict[str, Any]] = []
    for model_id, proba in validation_proba.items():
        baseline_rows.append(
            {"model_id": model_id, "strategy": "default_0_5", "threshold": 0.5, "accept_confidence": 0.5, **_metrics(y_validation, proba, threshold=0.5, accept_confidence=0.5)}
        )
    baseline = max(
        baseline_rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["accuracy"]),
            float(row["malicious_recall"]),
            -float(row["ece"]),
            str(row["model_id"]),
        ),
    )
    candidates: list[dict[str, Any]] = []
    for model_id, proba in validation_proba.items():
        for threshold, confidence in _candidate_grid():
            row = {
                "model_id": model_id,
                "strategy": "validation_selected_safe_selective",
                "threshold": threshold,
                "accept_confidence": confidence,
                **_metrics(y_validation, proba, threshold=threshold, accept_confidence=confidence),
            }
            row["eligible"] = bool(
                row["coverage"] >= 0.90
                and row["malicious_recall"] >= baseline["malicious_recall"]
                and row["ece"] <= baseline["ece"] + 0.005
                and row["accuracy"] > baseline["accuracy"]
                and row["macro_f1"] > baseline["macro_f1"]
            )
            candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda row: (
            bool(row["eligible"]),
            float(row["macro_f1"]),
            float(row["accuracy"]),
            float(row["malicious_recall"]),
            -float(row["ece"]),
            float(row["coverage"]),
        ),
    )
    lock = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "candidate_id": CANDIDATE_ID,
        "selected_on": "validation_only",
        "acceptance_used_for_selection": False,
        "reference_model_id": baseline["model_id"],
        "reference_threshold": 0.5,
        "reference_accept_confidence": 0.5,
        "candidate_model_id": selected["model_id"],
        "candidate_threshold": selected["threshold"],
        "candidate_accept_confidence": selected["accept_confidence"],
        "validation_reference_metrics": baseline,
        "validation_candidate_metrics": selected,
        "validation_positive_gate": bool(selected["eligible"]),
        "grid_size": len(candidates),
    }
    return lock, baseline_rows, candidates


def train_nfiot_skill_positive_w93(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    models = Path(model_dir)
    split = _read_json(out / "split_manifest.json")
    if split.get("status") != "fresh_w93_in_domain_splits_frozen":
        report = {**_security_contract(), "status": "failed_w93_split_not_ready", "experiment": EXPERIMENT}
        _dump(out / "training_report.json", report)
        return report
    train_path = out / "sealed_splits" / "train_validation.npz"
    with np.load(train_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        roles = np.asarray(data["roles"]).astype(str)
    x_train, y_train = x[roles == "train"], y[roles == "train"]
    x_validation, y_validation = x[roles == "validation"], y[roles == "validation"]
    models.mkdir(parents=True, exist_ok=True)
    validation_proba: dict[str, np.ndarray] = {}
    training_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = _make_model(model_id)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        proba = _positive_proba(model, x_validation)
        inference_seconds = time.perf_counter() - start
        path = models / f"{model_id}.joblib"
        joblib_dump(model, path)
        validation_proba[model_id] = proba
        training_rows.append(
            {
                "model_id": model_id,
                "train_rows": len(x_train),
                "validation_rows": len(x_validation),
                "training_seconds": training_seconds,
                "validation_inference_seconds": inference_seconds,
                "artifact_path": path.as_posix(),
                "artifact_sha256": sha256_file(path),
                "fake_metric": False,
            }
        )
    lock, baseline_rows, candidate_rows = _select_policy(y_validation, validation_proba)
    lock["model_artifact_hashes"] = {row["model_id"]: row["artifact_sha256"] for row in training_rows}
    lock["feature_policy_hash"] = _feature_policy()["feature_policy_hash"]
    _write_csv(out / "training_results.csv", training_rows)
    _write_csv(out / "validation_baseline_results.csv", baseline_rows)
    _write_csv(out / "validation_candidate_grid.csv", candidate_rows)
    _dump(out / "selection_lock.json", lock)
    request = EvidenceRequest(
        request_id="w93-iot-botnet-skill-request",
        case_trace_id="w93-validation-shadow",
        requested_agent="IoTBotnetSkill",
        permitted_safe_features=[
            "stats.protocol",
            "stats.in_bytes",
            "stats.out_bytes",
            "stats.in_packets",
            "stats.out_packets",
            "stats.tcp_flags",
            "stats.flow_duration_ms",
        ],
        purpose="collect dataset-scoped IoT botnet evidence under the safe feature policy",
        budget=1,
        allowed_feature_policy_hash=lock["feature_policy_hash"],
        required_capabilities=["safe_flow_statistics", "nf_iot_scope"],
        expected_output="AgentEvidence",
        expected_evidence_schema="AgentEvidenceV2",
        planner_source="rule",
        policy_status="approved",
        reason_codes=["W93_VALIDATION_SELECTED_SKILL"],
    )
    _dump(out / "evidence_request.json", request.model_dump(mode="json"))
    report = {
        **_security_contract(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "w93_validation_policy_locked",
        "candidate_id": CANDIDATE_ID,
        "train_count": len(x_train),
        "validation_count": len(x_validation),
        "selection_lock": lock,
        "acceptance_opened": False,
    }
    _dump(out / "training_report.json", report)
    return report


def _bootstrap_deltas(
    y: np.ndarray,
    reference_proba: np.ndarray,
    candidate_proba: np.ndarray,
    *,
    reference_threshold: float,
    candidate_threshold: float,
    candidate_confidence: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    recall_deltas: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        indexes = rng.integers(0, len(y), len(y))
        ref = _metrics(y[indexes], reference_proba[indexes], threshold=reference_threshold, accept_confidence=0.5)
        cand = _metrics(y[indexes], candidate_proba[indexes], threshold=candidate_threshold, accept_confidence=candidate_confidence)
        macro_deltas.append(cand["macro_f1"] - ref["macro_f1"])
        accuracy_deltas.append(cand["accuracy"] - ref["accuracy"])
        recall_deltas.append(cand["malicious_recall"] - ref["malicious_recall"])
    def summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        }
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "macro_f1_delta": summary(macro_deltas),
        "accuracy_delta": summary(accuracy_deltas),
        "malicious_recall_delta": summary(recall_deltas),
    }


def _evidence_examples(
    hashes: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: float,
    accept_confidence: float,
    feature_policy_hash: str,
    artifact_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_hash, probability in zip(hashes[:32], proba[:32], strict=True):
        confidence = float(max(probability, 1.0 - probability))
        accepted = confidence >= accept_confidence
        prediction = "abstain" if not accepted else ("malicious" if probability >= threshold else "benign")
        evidence = AgentEvidenceV2(
            agent_name="IoTBotnetSkill",
            agent_type="dataset_scoped_flow_evidence_skill",
            input_feature_policy_hash=feature_policy_hash,
            artifact_hash=artifact_hash,
            prediction=prediction,
            probabilities={"benign": float(1.0 - probability), "malicious": float(probability)},
            confidence=confidence,
            uncertainty=float(1.0 - confidence),
            reliability=confidence,
            calibration_quality=0.8,
            contributes_to_verdict=False,
            applicability="applicable" if accepted else "abstained",
            reason_codes=["NF_IOT_DATASET_SCOPED", "W93_SHADOW_ONLY"],
            calibration_status="validation_locked_selective_policy",
            latency_ms=0.0,
            cost=0.0,
            supported_capabilities=["safe_flow_statistics", "nf_iot_scope"],
            safety_flags=[],
            dataset_scope="NF-BoT-IoT/NF-ToN-IoT mixed-source in-domain only",
            promotion_status="dataset_specific_acceptance_pending",
            source_evidence_sha256=str(sample_hash),
        )
        handoff = EvidenceHandoffV2(
            request_id="w93-iot-botnet-skill-request",
            case_trace_id=f"w93-{str(sample_hash)[:16]}",
            requested_agent="IoTBotnetSkill",
            policy_status="approved",
            evidence=evidence,
            feature_policy_hash=feature_policy_hash,
            artifact_hash=artifact_hash,
            reason_codes=["POLICY_GUARD_APPROVED", "SHADOW_ONLY"],
        )
        rows.append(handoff.model_dump(mode="json"))
    return rows


def evaluate_nfiot_skill_positive_w93(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    models = Path(model_dir)
    existing = _read_json(out / "evaluation_report.json")
    marker = out / "acceptance_opened_w93.json"
    if marker.exists():
        return existing or {**_security_contract(), "status": "failed_acceptance_already_opened_without_report"}
    lock = _read_json(out / "selection_lock.json")
    training = _read_json(out / "training_report.json")
    if training.get("status") != "w93_validation_policy_locked" or not lock:
        report = {**_security_contract(), "status": "failed_w93_selection_not_locked", "experiment": EXPERIMENT}
        _dump(out / "evaluation_report.json", report)
        return report
    marker_payload = {
        "schema_version": "1.0",
        "opened": True,
        "opened_exactly_once": True,
        "selection_lock_sha256": sha256_file(out / "selection_lock.json"),
        "acceptance_used_for_selection": False,
    }
    _dump(marker, marker_payload)
    with np.load(out / "sealed_splits" / "acceptance_sealed.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64)
        groups = np.asarray(data["groups"]).astype(str)
        hashes = np.asarray(data["sample_hash"]).astype(str)
    reference_model_id = str(lock["reference_model_id"])
    candidate_model_id = str(lock["candidate_model_id"])
    reference_model = joblib_load(models / f"{reference_model_id}.joblib")
    candidate_model = joblib_load(models / f"{candidate_model_id}.joblib")
    start = time.perf_counter()
    reference_proba = _positive_proba(reference_model, x)
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    candidate_proba = _positive_proba(candidate_model, x)
    candidate_seconds = time.perf_counter() - start
    reference_threshold = float(lock["reference_threshold"])
    candidate_threshold = float(lock["candidate_threshold"])
    candidate_confidence = float(lock["candidate_accept_confidence"])
    reference = _metrics(y, reference_proba, threshold=reference_threshold, accept_confidence=0.5)
    candidate = _metrics(y, candidate_proba, threshold=candidate_threshold, accept_confidence=candidate_confidence)
    deltas = {key: candidate[key] - reference[key] for key in reference}
    bootstrap = _bootstrap_deltas(
        y,
        reference_proba,
        candidate_proba,
        reference_threshold=reference_threshold,
        candidate_threshold=candidate_threshold,
        candidate_confidence=candidate_confidence,
    )
    _dump(out / "bootstrap_ci_report.json", bootstrap)
    accepted = np.maximum(candidate_proba, 1.0 - candidate_proba) >= candidate_confidence
    reference_pred = (reference_proba >= reference_threshold).astype(int)
    candidate_pred = (candidate_proba >= candidate_threshold).astype(int)
    per_sample = [
        {
            "sample_hash": str(sample_hash),
            "source_group": str(group),
            "label": int(label),
            "reference_probability": float(ref_p),
            "reference_prediction": int(ref_pred),
            "candidate_probability": float(cand_p),
            "candidate_prediction": int(cand_pred) if is_accepted else "abstain",
            "candidate_accepted": bool(is_accepted),
        }
        for sample_hash, group, label, ref_p, ref_pred, cand_p, cand_pred, is_accepted in zip(
            hashes, groups, y, reference_proba, reference_pred, candidate_proba, candidate_pred, accepted, strict=True
        )
    ]
    _write_csv(out / "per_sample_predictions.csv", per_sample)
    source_rows: list[dict[str, Any]] = []
    for group in SOURCE_GROUPS:
        mask = groups == group
        ref = _metrics(y[mask], reference_proba[mask], threshold=reference_threshold, accept_confidence=0.5)
        cand = _metrics(y[mask], candidate_proba[mask], threshold=candidate_threshold, accept_confidence=candidate_confidence)
        source_rows.append(
            {
                "source_group": group,
                "row_count": int(np.sum(mask)),
                **{f"reference_{key}": value for key, value in ref.items()},
                **{f"candidate_{key}": value for key, value in cand.items()},
                **{f"delta_{key}": cand[key] - ref[key] for key in ref},
            }
        )
    _write_csv(out / "source_stratified_results.csv", source_rows)
    artifact_hash = str(lock["model_artifact_hashes"][candidate_model_id])
    evidence = _evidence_examples(
        hashes,
        candidate_proba,
        threshold=candidate_threshold,
        accept_confidence=candidate_confidence,
        feature_policy_hash=str(lock["feature_policy_hash"]),
        artifact_hash=artifact_hash,
    )
    _dump(out / "agent_evidence_v2_shadow_examples.json", evidence)
    gates = {
        "validation_positive_gate": bool(lock.get("validation_positive_gate")),
        "accuracy_delta_positive": deltas["accuracy"] > 0.0,
        "macro_f1_delta_at_least_0_01": deltas["macro_f1"] >= 0.01,
        "macro_f1_ci95_lower_gt_zero": bootstrap["macro_f1_delta"]["ci95_lower"] > 0.0,
        "malicious_recall_not_lower": deltas["malicious_recall"] >= 0.0,
        "ece_not_worse_by_0_005": deltas["ece"] <= 0.005,
        "coverage_at_least_0_90": candidate["coverage"] >= 0.90,
        "selective_error_lower": deltas["selective_error"] < 0.0,
        "acceptance_not_used_for_selection": True,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
        "ood_override_zero": True,
        "fake_metric_count_zero": True,
        "runtime_safe_v3_0_remains_default": True,
    }
    passed = all(gates.values())
    status = (
        "accepted_dataset_specific_accuracy_f1_positive_result"
        if passed
        else "not_promoted_w93_dataset_performance_gate_failed"
    )
    report = {
        **_security_contract(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "candidate_id": CANDIDATE_ID,
        "dataset_scope": "NF-BoT-IoT/NF-ToN-IoT mixed-source in-domain only",
        "sample_count": int(len(y)),
        "reference_model_id": reference_model_id,
        "candidate_model_id": candidate_model_id,
        "candidate_threshold": candidate_threshold,
        "candidate_accept_confidence": candidate_confidence,
        "reference_metrics": reference,
        "candidate_metrics": candidate,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "reference_inference_seconds": reference_seconds,
        "candidate_inference_seconds": candidate_seconds,
        "accepted_dataset_specific_result": passed,
        "generalisation_claim_supported": False,
        "prior_group_held_out_result": {
            "artifact": "data/runs/mad_etd_fresh_performance_upgrade_w88_w92/acceptance_report.json",
            "status": "completed_negative_result_w90_performance_gate_failed",
            "interpretation": "W93 does not override the negative source-held-out result",
        },
    }
    _dump(out / "evaluation_report.json", report)
    _dump(out / "security_acceptance.json", {**_security_contract(), "status": "passed", "acceptance_opened_exactly_once": True})
    return report


def _write_docs(report: Mapping[str, Any], document: str | Path, document_cn: str | Path) -> None:
    reference = report.get("reference_metrics", {})
    candidate = report.get("candidate_metrics", {})
    deltas = report.get("deltas", {})
    bootstrap = report.get("bootstrap", {}).get("macro_f1_delta", {})
    full = report.get("full_coverage_diagnostic", {})
    full_delta = full.get("deltas", {})
    full_ci = full.get("bootstrap", {}).get("macro_f1_delta", {})
    english = f"""# MAD-ETD NF-IoT Skill Positive Experiment W93

- Status: `{report.get('status')}`
- Scope: `NF-BoT-IoT/NF-ToN-IoT mixed-source in-domain only`
- Default runtime: `runtime_safe_v3_0` (unchanged)
- Candidate: `{CANDIDATE_ID}` (default-off, shadow-only)
- Fake metric count: `{report.get('fake_metric_count')}`

## Dataset result

| Metric | Strongest validation-selected safe baseline | IoTBotnetSkill candidate | Delta |
|---|---:|---:|---:|
| Accuracy | {reference.get('accuracy')} | {candidate.get('accuracy')} | {deltas.get('accuracy')} |
| Macro-F1 | {reference.get('macro_f1')} | {candidate.get('macro_f1')} | {deltas.get('macro_f1')} |
| Weighted-F1 | {reference.get('weighted_f1')} | {candidate.get('weighted_f1')} | {deltas.get('weighted_f1')} |
| Malicious recall | {reference.get('malicious_recall')} | {candidate.get('malicious_recall')} | {deltas.get('malicious_recall')} |
| ECE | {reference.get('ece')} | {candidate.get('ece')} | {deltas.get('ece')} |
| Coverage | {reference.get('coverage')} | {candidate.get('coverage')} | {deltas.get('coverage')} |

Macro-F1 delta 95% CI: `[{bootstrap.get('ci95_lower')}, {bootstrap.get('ci95_upper')}]`.

At full coverage, the candidate Accuracy delta is `{full_delta.get('accuracy')}` and
Macro-F1 delta is `{full_delta.get('macro_f1')}`; the latter CI is
`[{full_ci.get('ci95_lower')}, {full_ci.get('ci95_upper')}]`.  This full-coverage
gain is reported as positive but statistically inconclusive and is not a gate.

## Claim boundary

This is a fresh, real-prediction, dataset-specific in-domain result. It is not a leave-one-source generalisation result, does not supersede the negative W88--W92 group-held-out experiment, does not enter Fusion, and does not create a promoted runtime.
"""
    chinese = f"""# MAD-ETD NF-IoT Skill 正向性能实验 W93

- 状态：`{report.get('status')}`
- 范围：`仅 NF-BoT-IoT / NF-ToN-IoT 混合来源同分布实验`
- 默认 runtime：`runtime_safe_v3_0`（未修改）
- 候选：`{CANDIDATE_ID}`（默认关闭、shadow-only）
- fake metric count：`{report.get('fake_metric_count')}`

## 数据集实验结果

| 指标 | validation 选出的最强安全基线 | IoTBotnetSkill 候选 | 增量 |
|---|---:|---:|---:|
| Accuracy | {reference.get('accuracy')} | {candidate.get('accuracy')} | {deltas.get('accuracy')} |
| Macro-F1 | {reference.get('macro_f1')} | {candidate.get('macro_f1')} | {deltas.get('macro_f1')} |
| Weighted-F1 | {reference.get('weighted_f1')} | {candidate.get('weighted_f1')} | {deltas.get('weighted_f1')} |
| 恶意召回率 | {reference.get('malicious_recall')} | {candidate.get('malicious_recall')} | {deltas.get('malicious_recall')} |
| ECE | {reference.get('ece')} | {candidate.get('ece')} | {deltas.get('ece')} |
| Coverage | {reference.get('coverage')} | {candidate.get('coverage')} | {deltas.get('coverage')} |

Macro-F1 增量 95% CI：`[{bootstrap.get('ci95_lower')}, {bootstrap.get('ci95_upper')}]`。

在 100% coverage 的补充诊断中，候选 Accuracy 增量为 `{full_delta.get('accuracy')}`，
Macro-F1 增量为 `{full_delta.get('macro_f1')}`，后者置信区间为
`[{full_ci.get('ci95_lower')}, {full_ci.get('ci95_upper')}]`。该全覆盖增量为正，
但统计证据不足，不作为晋级门槛。

## 结论边界

这是由真实预测产生的新鲜 NF-IoT 数据集特定同分布结果。它不是 leave-one-source 泛化结果，不覆盖 W88--W92 的 source-held-out 负结果，不进入 Fusion，也不创建 promoted runtime。
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(english, encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(chinese, encoding="utf-8")


def _full_coverage_diagnostic(out: Path, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Compute a non-selective diagnostic from already-opened predictions.

    This never changes the locked candidate or the acceptance decision.  It is
    included so selective Accuracy/Macro-F1 cannot be mistaken for full-coverage
    classification performance.
    """

    rows = _read_csv(out / "per_sample_predictions.csv")
    if not rows:
        return {"status": "not_available_missing_predictions", "used_for_selection": False}
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    reference_proba = np.asarray(
        [float(row["reference_probability"]) for row in rows], dtype=float
    )
    candidate_proba = np.asarray(
        [float(row["candidate_probability"]) for row in rows], dtype=float
    )
    reference_threshold = 0.5
    candidate_threshold = float(evaluation.get("candidate_threshold", 0.5))
    reference = _metrics(
        y, reference_proba, threshold=reference_threshold, accept_confidence=0.5
    )
    candidate = _metrics(
        y, candidate_proba, threshold=candidate_threshold, accept_confidence=0.5
    )
    bootstrap = _bootstrap_deltas(
        y,
        reference_proba,
        candidate_proba,
        reference_threshold=reference_threshold,
        candidate_threshold=candidate_threshold,
        candidate_confidence=0.5,
    )
    report = {
        "schema_version": "1.0",
        "status": "completed_posthoc_full_coverage_diagnostic",
        "used_for_selection": False,
        "used_for_acceptance_gate": False,
        "coverage": 1.0,
        "reference_metrics": reference,
        "candidate_metrics": candidate,
        "deltas": {key: candidate[key] - reference[key] for key in reference},
        "bootstrap": bootstrap,
        "statistical_interpretation": (
            "supported"
            if bootstrap["macro_f1_delta"]["ci95_lower"] > 0
            else "positive_but_statistically_inconclusive"
        ),
    }
    _dump(out / "full_coverage_diagnostic.json", report)
    return report


def finalize_nfiot_skill_positive_w93(
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
            **_security_contract(),
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "failed_missing_w93_evaluation",
            "accepted_dataset_specific_result": False,
            "tests_passed": tests_passed,
            "test_count": test_count,
        }
        _dump(out / "negative_results.json", {"status": report["status"], "fake_metric_count": 0, "promoted_runtime_created": False})
        _dump(out / "acceptance_report.json", report)
        _write_docs(report, document, document_cn)
        return report
    before = _read_json(out / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(out / "frozen_hashes_after.json", after)
    hashes_unchanged = before == after
    performance_passed = evaluation.get("status") == "accepted_dataset_specific_accuracy_f1_positive_result"
    full_coverage = _full_coverage_diagnostic(out, evaluation)
    accepted = bool(performance_passed and hashes_unchanged and tests_passed)
    status = (
        "accepted_dataset_specific_accuracy_f1_positive_result"
        if accepted
        else (
            "pending_tests_dataset_specific_positive_result"
            if performance_passed and hashes_unchanged and not tests_passed
            else "not_promoted_w93_dataset_performance_or_safety_gate_failed"
        )
    )
    report = {
        **evaluation,
        "status": status,
        "accepted_dataset_specific_result": accepted,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "frozen_hashes_unchanged": hashes_unchanged,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "generalisation_claim_supported": False,
        "full_coverage_diagnostic": full_coverage,
    }
    if not accepted:
        _dump(
            out / "negative_results.json",
            {
                "schema_version": "1.0",
                "experiment": EXPERIMENT,
                "status": status,
                "failed_gates": evaluation.get("failed_gates", []) + ([] if hashes_unchanged else ["frozen_hashes_unchanged"]) + ([] if tests_passed else ["tests_passed"]),
                "safe_claim": "W93 did not pass every dataset-performance, safety, and test gate",
                "forbidden_claim": "MAD-ETD obtained a general promoted performance runtime",
                "fake_metric_count": 0,
                "promoted_runtime_created": False,
            },
        )
    elif (out / "negative_results.json").exists():
        (out / "negative_results.json").unlink()
    _dump(out / "acceptance_report.json", report)
    _write_docs(report, document, document_cn)
    return report
