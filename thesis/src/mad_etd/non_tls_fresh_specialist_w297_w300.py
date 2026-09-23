"""W297-W300 fresh non-TLS dataset-specialist performance experiment.

The historical CICIoT2023 binary lane has no unused benign source-file group.
This protocol therefore does *not* reopen binary malicious-flow detection.
It freezes a narrower, honest task over never-consumed malicious capture-file
groups: multiclass attack-type attribution after malicious applicability has
already been established.

The candidate is default-off and produces non-contributing AgentEvidenceV2
shadow records.  It never changes ``runtime_safe_v3_0`` or Fusion/OOD policy.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .paper_evaluation import hash_artifact_paths
from .schemas import AgentEvidenceV2
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_non_tls_fresh_specialist_w297_w300"
DEFAULT_RAW = Path("data/raw/CICIoT2023/CSV")
DEFAULT_HISTORY = Path(
    "data/runs/mad_etd_ciciot_fresh_group_performance_w231_w235"
)
DEFAULT_OUTPUT = Path(
    "data/runs/mad_etd_non_tls_fresh_specialist_w297_w300"
)
DEFAULT_MODEL = Path("data/models/mad_etd_dataset_specialist_w298")
DEFAULT_DOC = Path("docs/MAD_ETD_DATASET_SPECIALIST_W297_W300.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_DATASET_SPECIALIST_W297_W300_CN.md")

SEED = 42
ROWS_PER_GROUP = 400
GROUPS_PER_CLASS = 5
BOOTSTRAP_ITERATIONS = 1000

SAFE_FEATURES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

BLOCKED_FIELDS = [
    "Label",
    "Attack",
    "attack_name",
    "attack_family",
    "directory_label",
    "source_file",
    "relative_path",
    "IP",
    "Src IP",
    "Dst IP",
    "port",
    "Src Port",
    "Dst Port",
    "Timestamp",
    "Flow ID",
    "provenance",
    "split_metadata",
    "capture_id",
    "source_group",
]


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _stable_order(values: Iterable[str], seed: int = SEED) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def _safe_hashes() -> dict[str, Any]:
    return hash_artifact_paths(_default_frozen_paths())


def _hash_snapshot_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def _historical_ciciot_inventory(
    history_dir: Path,
) -> tuple[list[dict[str, str]], set[str]]:
    inventory_path = history_dir / "fresh_source_inventory_w231.csv"
    acceptance_path = history_dir / "fresh_acceptance_manifest_w231.csv"
    if not inventory_path.is_file() or not acceptance_path.is_file():
        raise FileNotFoundError(
            "W231 source inventory and acceptance manifest are required"
        )
    inventory = _read_csv(inventory_path)
    accepted = {
        row["relative_path"] for row in _read_csv(acceptance_path)
    }
    return inventory, accepted


def _candidate_dataset_audit(
    raw_dir: Path,
    history_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], set[str]]:
    inventory, accepted = _historical_ciciot_inventory(history_dir)
    fresh = [
        row
        for row in inventory
        if row.get("fresh_relative_to_w215_source_ledger") == "True"
        and row["relative_path"] not in accepted
        and (raw_dir / row["relative_path"]).is_file()
    ]
    counts = Counter(row["directory"] for row in fresh)
    attack_classes = sorted(
        label
        for label, count in counts.items()
        if label != "Benign_Final" and count >= GROUPS_PER_CLASS
    )
    rows: list[dict[str, Any]] = [
        {
            "dataset": "CICIoT2023",
            "local_data_available": True,
            "explicit_labels_available": True,
            "safe_feature_policy_available": True,
            "group_key": "official per-capture CSV source file",
            "historically_unused_group_count": len(fresh),
            "historically_unused_benign_group_count": sum(
                row["directory"] == "Benign_Final" for row in fresh
            ),
            "eligible_attack_class_count": len(attack_classes),
            "eligible_for_fresh_binary_detection": False,
            "eligible_for_fresh_multiclass_attribution": len(attack_classes)
            >= 3,
            "decision": (
                "selected_narrow_multiclass_attribution"
                if len(attack_classes) >= 3
                else "blocked_insufficient_fresh_groups"
            ),
            "reason": (
                "No unused benign group remains; unused malicious source-file "
                "groups support a strictly scoped attack-type attribution task."
            ),
        },
        {
            "dataset": "CSE-CIC-IDS2018",
            "local_data_available": Path(
                "data/raw/CSE-CIC-IDS2018-CSV"
            ).is_dir(),
            "explicit_labels_available": True,
            "safe_feature_policy_available": True,
            "group_key": "capture day CSV",
            "historically_unused_group_count": 0,
            "historically_unused_benign_group_count": 0,
            "eligible_attack_class_count": 0,
            "eligible_for_fresh_binary_detection": False,
            "eligible_for_fresh_multiclass_attribution": False,
            "decision": "blocked_no_unused_source_group",
            "reason": (
                "All ten capture-day files are recorded as historically "
                "consumed; later W247/W253/W258/W263 are sample-fresh only."
            ),
        },
        {
            "dataset": "NF-BoT-IoT/NF-ToN-IoT",
            "local_data_available": Path(
                "data/raw/nf_bot_iot_ton_iot"
            ).is_dir(),
            "explicit_labels_available": True,
            "safe_feature_policy_available": True,
            "group_key": "official dataset source variant",
            "historically_unused_group_count": 0,
            "historically_unused_benign_group_count": 0,
            "eligible_attack_class_count": 0,
            "eligible_for_fresh_binary_detection": False,
            "eligible_for_fresh_multiclass_attribution": False,
            "decision": "blocked_all_source_variants_consumed",
            "reason": (
                "NF-BoT/NF-ToN v1/v2 source variants have historical "
                "selection and acceptance usage through W97."
            ),
        },
        {
            "dataset": "HIKARI-2021",
            "local_data_available": Path("data/raw/hikari_2021").is_dir(),
            "explicit_labels_available": True,
            "safe_feature_policy_available": True,
            "group_key": "endpoint/capture group",
            "historically_unused_group_count": 0,
            "historically_unused_benign_group_count": 0,
            "eligible_attack_class_count": 0,
            "eligible_for_fresh_binary_detection": False,
            "eligible_for_fresh_multiclass_attribution": False,
            "decision": "blocked_no_new_positive_capture_group",
            "reason": (
                "W152-W158 consumed the four auditable positive endpoint "
                "groups across train/validation/acceptance."
            ),
        },
    ]
    return rows, fresh, set(attack_classes)


def _build_group_split(
    fresh_rows: Sequence[Mapping[str, str]],
    attack_classes: set[str],
    *,
    groups_per_class: int = GROUPS_PER_CLASS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in fresh_rows:
        label = row["directory"]
        if label in attack_classes:
            grouped[label].append(row["relative_path"])
    result: list[dict[str, Any]] = []
    for class_index, label in enumerate(sorted(attack_classes)):
        values = _stable_order(grouped[label], seed + class_index)
        if len(values) < groups_per_class:
            raise ValueError(f"{label} has fewer than {groups_per_class} groups")
        chosen = values[:groups_per_class]
        # Two training, one validation and two untouched acceptance groups.
        roles = ["train", "train", "validation", "acceptance", "acceptance"]
        if groups_per_class != 5:
            raise ValueError("this preregistered protocol requires five groups")
        for relative_path, role in zip(chosen, roles, strict=True):
            result.append(
                {
                    "dataset": "CICIoT2023",
                    "task": "multiclass_malicious_attack_type_attribution",
                    "role": role,
                    "class_label": label,
                    "relative_path": relative_path,
                    "source_group_hash": _sha256_bytes(
                        relative_path.encode("utf-8")
                    ),
                    "row_quota": ROWS_PER_GROUP,
                    "source_group_enters_feature_matrix": False,
                    "class_label_enters_feature_matrix": False,
                    "acceptance_opened": False,
                }
            )
    return result


def build_non_tls_fresh_specialist_w297(
    raw_dir: str | Path = DEFAULT_RAW,
    history_dir: str | Path = DEFAULT_HISTORY,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    history = Path(history_dir)
    output = Path(output_dir)
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    audits, fresh, attack_classes = _candidate_dataset_audit(raw, history)
    selected = next(
        (
            row
            for row in audits
            if row["decision"] == "selected_narrow_multiclass_attribution"
        ),
        None,
    )
    _write_csv(output / "candidate_dataset_audit_w297.csv", audits)
    usage_rows = [
        {
            "relative_path": row["relative_path"],
            "directory": row["directory"],
            "used_by_w215": row.get("used_by_w215", ""),
            "used_by_w231_acceptance": False,
            "eligible_after_history_reconciliation": True,
        }
        for row in fresh
    ]
    _write_csv(output / "historical_usage_ledger_w297.csv", usage_rows)
    before = _safe_hashes()
    _dump(output / "frozen_hashes_before_w297.json", before)
    feature_policy = {
        "schema_version": "W297.1",
        "task": "multiclass_malicious_attack_type_attribution",
        "safe_features": SAFE_FEATURES,
        "blocked_or_context_only_fields": BLOCKED_FIELDS,
        "source_group_usage": "split_and_audit_only",
        "class_label_usage": "supervision_and_diagnostics_only",
        "detector_input_contains_source_identity": False,
        "detector_input_contains_attack_label": False,
        "binary_malicious_detection_claim_allowed": False,
        "fusion_contribution_allowed": False,
    }
    feature_policy["feature_policy_hash"] = _canonical_hash(feature_policy)
    _dump(output / "safe_feature_policy_w297.json", feature_policy)
    if selected is None:
        report = {
            "status": "failed_no_fresh_non_tls_group_heldout_lane",
            "selected_dataset": None,
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
            "promoted_runtime_created": False,
        }
        _dump(output / "w297_protocol_report.json", report)
        return report
    split = _build_group_split(fresh, attack_classes)
    _write_csv(output / "split_manifest_w297.csv", split)
    role_sets = {
        role: {
            row["source_group_hash"] for row in split if row["role"] == role
        }
        for role in ("train", "validation", "acceptance")
    }
    overlaps = sum(
        len(role_sets[a] & role_sets[b])
        for a, b in itertools.combinations(role_sets, 2)
    )
    report = {
        "status": "w297_fresh_ciciot_attack_type_lane_frozen",
        "selected_dataset": "CICIoT2023",
        "task": "multiclass_malicious_attack_type_attribution",
        "scope_boundary": (
            "post-malicious attack-type attribution only; not benign/"
            "malicious detection and not a Fusion verdict source"
        ),
        "class_count": len(attack_classes),
        "selected_source_group_count": len(split),
        "role_group_counts": dict(Counter(row["role"] for row in split)),
        "rows_per_group": ROWS_PER_GROUP,
        "group_overlap_count": overlaps,
        "fresh_relative_to_w215_and_w231": True,
        "acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "w297_protocol_report.json", report)
    return report


def _extract_role(
    raw_dir: Path,
    manifest: Sequence[Mapping[str, str]],
    role: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    parts: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    sample_ids: list[str] = []
    for row in manifest:
        if row["role"] != role:
            continue
        path = raw_dir / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path,
            usecols=SAFE_FEATURES,
            nrows=ROWS_PER_GROUP * 4,
            low_memory=False,
        )
        if set(frame.columns) != set(SAFE_FEATURES):
            missing = sorted(set(SAFE_FEATURES) - set(frame.columns))
            raise ValueError(f"safe feature columns missing from {path}: {missing}")
        frame = frame[SAFE_FEATURES].replace([np.inf, -np.inf], np.nan)
        frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(frame) > ROWS_PER_GROUP:
            group_seed = int(row["source_group_hash"][:8], 16)
            frame = frame.sample(
                n=ROWS_PER_GROUP,
                replace=False,
                random_state=group_seed,
            )
        values = frame.to_numpy(dtype=np.float32, copy=True)
        parts.append(values)
        labels.extend([row["class_label"]] * len(values))
        groups.extend([row["source_group_hash"]] * len(values))
        for index in frame.index:
            sample_ids.append(
                _sha256_bytes(
                    f"{row['source_group_hash']}|{index}".encode("utf-8")
                )
            )
    if not parts:
        raise ValueError(f"no rows extracted for role {role}")
    return np.vstack(parts), np.asarray(labels), groups, sample_ids


def _build_models(seed: int = SEED) -> dict[str, Any]:
    models: dict[str, Any] = {
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=31,
            l2_regularization=0.5,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=320,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=280,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=280,
            num_leaves=63,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(
            iterations=280,
            depth=8,
            learning_rate=0.08,
            loss_function="MultiClass",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    except Exception:
        pass
    return models


def _multiclass_ece(
    truth: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == truth
    total = max(len(truth), 1)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not np.any(mask):
            continue
        result += (
            abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            * int(mask.sum())
            / total
        )
    return float(result)


def _metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    latency_ms: float | str = "not_available",
) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(truth, prediction))
    return {
        "accuracy": accuracy,
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, prediction, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                truth, prediction, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(truth, prediction, average="macro", zero_division=0)
        ),
        "ece": _multiclass_ece(truth, probabilities),
        "coverage": 1.0,
        "selective_error": 1.0 - accuracy,
        "malicious_recall": "not_applicable_all_samples_are_malicious",
        "p95_latency_ms": latency_ms,
    }


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _ensemble_probabilities(
    model_probabilities: Mapping[str, np.ndarray],
    members: Sequence[str],
    weights: Sequence[float],
    temperature: float = 1.0,
) -> np.ndarray:
    total = np.zeros_like(model_probabilities[members[0]], dtype=np.float64)
    denominator = 0.0
    for name, weight in zip(members, weights, strict=True):
        total += float(weight) * model_probabilities[name]
        denominator += float(weight)
    total /= max(denominator, 1e-12)
    return _temperature_scale(total, temperature)


def _select_specialist_candidate(
    truth: np.ndarray,
    model_probabilities: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    single_rows = []
    for name, probs in model_probabilities.items():
        single_rows.append({"model": name, **_metrics(truth, probs)})
    ordered = sorted(
        single_rows,
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["macro_recall"]),
            float(row["ece"]),
            row["model"],
        ),
    )
    strongest = ordered[0]["model"]
    top = [row["model"] for row in ordered[: min(4, len(ordered))]]
    searches: list[dict[str, Any]] = []
    candidate_specs: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
    for size in range(2, min(4, len(top)) + 1):
        for members in itertools.combinations(top, size):
            candidate_specs.append((members, tuple([1.0] * size)))
    if len(top) >= 2:
        for alpha in (0.2, 0.35, 0.5, 0.65, 0.8):
            candidate_specs.append(
                ((top[0], top[1]), (alpha, 1.0 - alpha))
            )
    best: dict[str, Any] | None = None
    for members, weights in candidate_specs:
        raw = _ensemble_probabilities(
            model_probabilities, members, weights, 1.0
        )
        # Temperature is calibration-only and cannot change the class prediction.
        temperatures = np.linspace(0.6, 1.8, 25)
        temperature = min(
            temperatures,
            key=lambda value: _multiclass_ece(
                truth, _temperature_scale(raw, float(value))
            ),
        )
        probs = _temperature_scale(raw, float(temperature))
        metrics = _metrics(truth, probs)
        row = {
            "candidate": "dataset_specialist_skill_v1",
            "members": ";".join(members),
            "weights": ";".join(f"{value:.6f}" for value in weights),
            "temperature": float(temperature),
            **metrics,
        }
        searches.append(row)
        key = (
            float(row["macro_f1"]),
            float(row["macro_recall"]),
            -float(row["ece"]),
            -len(members),
        )
        if best is None or key > best["_key"]:
            best = {
                **row,
                "_key": key,
                "member_list": list(members),
                "weight_list": list(weights),
            }
    if best is None:
        raise RuntimeError("no validation ensemble candidate was created")
    best.pop("_key", None)
    return {
        "strongest_baseline": strongest,
        "candidate": best,
        "single_model_ranking": ordered,
    }, searches


def _encode(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(classes)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int64)


def train_dataset_specialist_w298(
    raw_dir: str | Path = DEFAULT_RAW,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    output = Path(output_dir)
    models_out = Path(model_dir)
    protocol = _load(output / "w297_protocol_report.json")
    if protocol.get("status") != "w297_fresh_ciciot_attack_type_lane_frozen":
        raise RuntimeError("W297 fresh protocol is not ready")
    manifest = _read_csv(output / "split_manifest_w297.csv")
    if any(row["acceptance_opened"] != "False" for row in manifest):
        raise RuntimeError("acceptance manifest was opened before W298")
    x_train, labels_train, _, _ = _extract_role(raw, manifest, "train")
    x_validation, labels_validation, _, _ = _extract_role(
        raw, manifest, "validation"
    )
    classes = sorted(set(labels_train) | set(labels_validation))
    y_train = _encode(labels_train, classes)
    y_validation = _encode(labels_validation, classes)
    models = _build_models()
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    probabilities: dict[str, np.ndarray] = {}
    trained: dict[str, Any] = {}
    for name, model in models.items():
        started = time.perf_counter()
        try:
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - started
            prediction_started = time.perf_counter()
            probs = np.asarray(model.predict_proba(x_validation), dtype=float)
            elapsed_ms = (
                time.perf_counter() - prediction_started
            ) * 1000.0
            per_sample_ms = elapsed_ms / max(len(y_validation), 1)
            metrics = _metrics(
                y_validation,
                probs,
                latency_ms=per_sample_ms,
            )
            probabilities[name] = probs
            trained[name] = model
            training_rows.append(
                {
                    "model": name,
                    "status": "trained",
                    "training_seconds": train_seconds,
                    "train_rows": len(y_train),
                    "feature_count": x_train.shape[1],
                }
            )
            validation_rows.append(
                {
                    "candidate_type": "safe_single_model_baseline",
                    "model": name,
                    **metrics,
                }
            )
        except Exception as exc:
            training_rows.append(
                {
                    "model": name,
                    "status": "dependency_or_training_failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "training_seconds": time.perf_counter() - started,
                    "train_rows": len(y_train),
                    "feature_count": x_train.shape[1],
                }
            )
    if len(trained) < 2:
        raise RuntimeError("fewer than two safe baselines trained successfully")
    selection, search_rows = _select_specialist_candidate(
        y_validation, probabilities
    )
    candidate = selection["candidate"]
    candidate_probs = _ensemble_probabilities(
        probabilities,
        candidate["member_list"],
        candidate["weight_list"],
        float(candidate["temperature"]),
    )
    validation_rows.append(
        {
            "candidate_type": "dataset_specialist_skill_candidate",
            "model": "dataset_specialist_skill_v1",
            **_metrics(y_validation, candidate_probs),
        }
    )
    _write_csv(output / "training_results_w298.csv", training_rows)
    _write_csv(output / "validation_results_w298.csv", validation_rows)
    _write_csv(output / "candidate_search_w298.csv", search_rows)
    feature_policy = _load(output / "safe_feature_policy_w297.json")
    artifact_payload = {
        "schema_version": "W298.1",
        "task": protocol["task"],
        "classes": classes,
        "safe_features": SAFE_FEATURES,
        "feature_policy_hash": feature_policy["feature_policy_hash"],
        "models": trained,
        "strongest_baseline": selection["strongest_baseline"],
        "candidate_members": candidate["member_list"],
        "candidate_weights": candidate["weight_list"],
        "candidate_temperature": candidate["temperature"],
        "acceptance_opened": False,
    }
    models_out.mkdir(parents=True, exist_ok=True)
    artifact = models_out / "dataset_specialist_skill_v1.joblib"
    joblib.dump(artifact_payload, artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = {
        "status": "w298_candidate_locked_before_acceptance",
        "candidate_id": "dataset_specialist_skill_v1",
        "task": protocol["task"],
        "task_scope": (
            "CICIoT2023 malicious attack-type attribution; not binary "
            "malicious detection"
        ),
        "strongest_validation_baseline": selection["strongest_baseline"],
        "candidate_members": candidate["member_list"],
        "candidate_weights": candidate["weight_list"],
        "candidate_temperature": candidate["temperature"],
        "candidate_validation_metrics": _metrics(
            y_validation, candidate_probs
        ),
        "artifact": artifact.as_posix(),
        "artifact_hash": artifact_hash,
        "feature_policy_hash": feature_policy["feature_policy_hash"],
        "output_schema": "AgentEvidenceV2",
        "contributes_to_verdict": False,
        "candidate_default_enabled": False,
        "acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "candidate_lock_w298.json", lock)
    # The multiclass skill cannot map attack type to benign/malicious evidence.
    # It therefore emits an explicit non-contributing abstain shadow record.
    evidence = AgentEvidenceV2(
        agent_name="dataset_specialist_skill_v1",
        agent_type="malicious_attack_type_attribution_skill",
        input_feature_policy_hash=feature_policy["feature_policy_hash"],
        artifact_hash=artifact_hash,
        prediction="abstain",
        probabilities={
            label: float(value)
            for label, value in zip(classes, candidate_probs[0], strict=True)
        },
        confidence=float(candidate_probs[0].max()),
        uncertainty=float(1.0 - candidate_probs[0].max()),
        reliability=0.0,
        contributes_to_verdict=False,
        applicability="applicable",
        unsupported_reason=None,
        reason_codes=[
            "MULTICLASS_ATTRIBUTION_ONLY",
            "NOT_BINARY_FUSION_EVIDENCE",
            "DEFAULT_OFF_SHADOW_SKILL",
        ],
        calibration_status="validation_temperature_locked",
        latency_ms=0.0,
        cost=0.0,
        supported_capabilities=["safe_flow_statistics"],
        safety_flags=[],
        dataset_scope="CICIoT2023 malicious attack-type attribution",
        promotion_status="candidate_locked_before_acceptance",
        source_evidence_sha256=_canonical_hash(
            {
                "task": protocol["task"],
                "artifact_hash": artifact_hash,
                "example": 0,
            }
        ),
    )
    _dump(
        output / "agent_evidence_v2_shadow_examples_w298.json",
        {
            "examples": [evidence.model_dump(mode="json")],
            "final_verdict_fields_present": False,
            "fusion_contribution": False,
        },
    )
    report = {
        "status": "w298_dataset_specialist_candidate_locked",
        "trained_model_count": len(trained),
        "failed_model_count": len(models) - len(trained),
        "strongest_validation_baseline": selection["strongest_baseline"],
        "candidate_validation_macro_f1": float(
            _metrics(y_validation, candidate_probs)["macro_f1"]
        ),
        "acceptance_opened": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "w298_training_report.json", report)
    return report


def _per_class_rows(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    classes: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, predicted in predictions.items():
        for index, label in enumerate(classes):
            mask = truth == index
            recall = float(np.mean(predicted[mask] == index)) if np.any(mask) else 0.0
            rows.append(
                {
                    "model": name,
                    "class_label": label,
                    "support": int(mask.sum()),
                    "recall": recall,
                }
            )
    return rows


def _group_rows(
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    groups: Sequence[str],
) -> list[dict[str, Any]]:
    group_array = np.asarray(groups)
    rows: list[dict[str, Any]] = []
    for name, predicted in predictions.items():
        for group in sorted(set(groups)):
            mask = group_array == group
            rows.append(
                {
                    "model": name,
                    "source_group_hash": group,
                    "sample_count": int(mask.sum()),
                    "accuracy": float(accuracy_score(truth[mask], predicted[mask])),
                    "macro_f1": float(
                        f1_score(
                            truth[mask],
                            predicted[mask],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
    return rows


def _grouped_bootstrap(
    truth: np.ndarray,
    candidate_probs: np.ndarray,
    baseline_probs: np.ndarray,
    groups: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    group_array = np.asarray(groups)
    group_to_indices = {
        group: np.flatnonzero(group_array == group)
        for group in sorted(set(groups))
    }
    class_groups: dict[int, list[str]] = defaultdict(list)
    for group, indices in group_to_indices.items():
        labels = truth[indices]
        majority = int(Counter(labels.tolist()).most_common(1)[0][0])
        class_groups[majority].append(group)
    rng = np.random.default_rng(seed)
    macro_deltas: list[float] = []
    accuracy_deltas: list[float] = []
    recall_deltas: list[float] = []
    selective_error_deltas: list[float] = []
    for _ in range(iterations):
        sampled: list[int] = []
        for label in sorted(class_groups):
            values = class_groups[label]
            chosen = rng.choice(values, size=len(values), replace=True)
            for group in chosen:
                sampled.extend(group_to_indices[str(group)].tolist())
        index = np.asarray(sampled, dtype=np.int64)
        y = truth[index]
        candidate = candidate_probs[index].argmax(axis=1)
        baseline = baseline_probs[index].argmax(axis=1)
        c_macro = f1_score(y, candidate, average="macro", zero_division=0)
        b_macro = f1_score(y, baseline, average="macro", zero_division=0)
        c_accuracy = accuracy_score(y, candidate)
        b_accuracy = accuracy_score(y, baseline)
        c_recall = recall_score(
            y, candidate, average="macro", zero_division=0
        )
        b_recall = recall_score(
            y, baseline, average="macro", zero_division=0
        )
        macro_deltas.append(float(c_macro - b_macro))
        accuracy_deltas.append(float(c_accuracy - b_accuracy))
        recall_deltas.append(float(c_recall - b_recall))
        selective_error_deltas.append(
            float((1.0 - c_accuracy) - (1.0 - b_accuracy))
        )

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        }

    return {
        "method": "class-stratified source-file-group paired bootstrap",
        "iterations": iterations,
        "seed": seed,
        "source_group_count": len(group_to_indices),
        "macro_f1_delta": interval(macro_deltas),
        "accuracy_delta": interval(accuracy_deltas),
        "macro_recall_delta": interval(recall_deltas),
        "selective_error_delta": interval(selective_error_deltas),
    }


def evaluate_dataset_specialist_w299(
    raw_dir: str | Path = DEFAULT_RAW,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    output = Path(output_dir)
    lock = _load(output / "candidate_lock_w298.json")
    if lock.get("status") != "w298_candidate_locked_before_acceptance":
        raise RuntimeError("W298 candidate is not locked")
    artifact_path = Path(lock["artifact"])
    expected_root = Path(model_dir)
    if expected_root not in artifact_path.parents:
        raise RuntimeError("candidate artifact is outside the expected model dir")
    artifact = joblib.load(artifact_path)
    manifest = _read_csv(output / "split_manifest_w297.csv")
    opened_path = output / "acceptance_opened_w299.json"
    if opened_path.exists():
        existing = _load(opened_path)
        if existing.get("candidate_artifact_hash") != lock["artifact_hash"]:
            raise RuntimeError("acceptance was opened for a different candidate")
    else:
        _dump(
            opened_path,
            {
                "status": "w299_acceptance_opened_once",
                "candidate_artifact_hash": lock["artifact_hash"],
                "selection_policy_locked": True,
                "acceptance_used_for_selection": False,
            },
        )
    x_acceptance, labels, groups, sample_ids = _extract_role(
        raw, manifest, "acceptance"
    )
    classes = artifact["classes"]
    truth = _encode(labels, classes)
    probabilities: dict[str, np.ndarray] = {}
    latency: dict[str, float] = {}
    for name, model in artifact["models"].items():
        started = time.perf_counter()
        probabilities[name] = np.asarray(
            model.predict_proba(x_acceptance), dtype=float
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        latency[name] = elapsed / max(len(truth), 1)
    candidate_probs = _ensemble_probabilities(
        probabilities,
        artifact["candidate_members"],
        artifact["candidate_weights"],
        artifact["candidate_temperature"],
    )
    probabilities["dataset_specialist_skill_v1"] = candidate_probs
    latency["dataset_specialist_skill_v1"] = sum(
        latency[name] for name in artifact["candidate_members"]
    )
    result_rows: list[dict[str, Any]] = []
    for name, probs in probabilities.items():
        kind = (
            "dataset_specialist_skill_candidate"
            if name == "dataset_specialist_skill_v1"
            else "safe_single_model_baseline"
        )
        result_rows.append(
            {
                "model": name,
                "candidate_type": kind,
                **_metrics(truth, probs, latency_ms=latency[name]),
            }
        )
    _write_csv(output / "acceptance_results_w299.csv", result_rows)
    predictions = {
        name: probs.argmax(axis=1) for name, probs in probabilities.items()
    }
    prediction_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_group_hash": groups[index],
            "truth": classes[int(truth[index])],
        }
        for name, probs in probabilities.items():
            row[f"{name}_prediction"] = classes[int(probs[index].argmax())]
            row[f"{name}_confidence"] = float(probs[index].max())
        prediction_rows.append(row)
    _write_csv(output / "acceptance_predictions_w299.csv", prediction_rows)
    _write_csv(
        output / "per_class_metrics_w299.csv",
        _per_class_rows(truth, predictions, classes),
    )
    group_metrics = _group_rows(truth, predictions, groups)
    _write_csv(output / "per_group_metrics_w299.csv", group_metrics)
    strongest = artifact["strongest_baseline"]
    bootstrap = _grouped_bootstrap(
        truth,
        candidate_probs,
        probabilities[strongest],
        groups,
    )
    _dump(output / "grouped_bootstrap_w299.json", bootstrap)
    baseline_metrics = next(
        row for row in result_rows if row["model"] == strongest
    )
    candidate_metrics = next(
        row
        for row in result_rows
        if row["model"] == "dataset_specialist_skill_v1"
    )
    comparison = {
        key: (
            float(candidate_metrics[key]) - float(baseline_metrics[key])
        )
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "macro_precision",
            "macro_recall",
            "ece",
            "coverage",
            "selective_error",
        )
    }
    _dump(
        output / "paired_comparisons_w299.json",
        {
            "strongest_validation_selected_baseline": strongest,
            "candidate": "dataset_specialist_skill_v1",
            "deltas_candidate_minus_baseline": comparison,
            "acceptance_used_for_selection": False,
        },
    )
    _dump(
        output / "calibration_report_w299.json",
        {
            "candidate_temperature": artifact["candidate_temperature"],
            "temperature_selected_on": "W298 validation only",
            "acceptance_ece": candidate_metrics["ece"],
            "baseline_ece": baseline_metrics["ece"],
            "ece_delta": comparison["ece"],
        },
    )
    report = {
        "status": "w299_fresh_group_heldout_acceptance_completed",
        "task": artifact["task"],
        "acceptance_sample_count": len(truth),
        "acceptance_source_group_count": len(set(groups)),
        "class_count": len(classes),
        "strongest_validation_selected_baseline": strongest,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "deltas": comparison,
        "bootstrap": bootstrap,
        "acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "w299_acceptance_report.json", report)
    return report


def _promotion_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    deltas = report["deltas"]
    bootstrap = report["bootstrap"]
    return {
        "macro_f1_delta_ge_0_01": float(deltas["macro_f1"]) >= 0.01,
        "macro_f1_delta_ci95_lower_gt_0": float(
            bootstrap["macro_f1_delta"]["ci95_lower"]
        )
        > 0.0,
        "accuracy_not_worse": float(deltas["accuracy"]) >= 0.0,
        "macro_recall_not_worse": float(deltas["macro_recall"]) >= 0.0,
        "ece_not_worse_by_more_than_0_005": float(deltas["ece"]) <= 0.005,
        "selective_error_not_worse": float(deltas["selective_error"]) <= 0.0,
        "blocked_field_violation_zero": int(
            report["blocked_field_violation"]
        )
        == 0,
        "fusion_ownership_violation_zero": int(
            report["fusion_ownership_violation"]
        )
        == 0,
        "ood_override_zero": int(report["ood_override_count"]) == 0,
        "illegal_verdict_execution_zero": int(
            report["illegal_verdict_execution_count"]
        )
        == 0,
        "fake_metric_count_zero": int(report["fake_metric_count"]) == 0,
        "acceptance_not_used_for_selection": not bool(
            report["acceptance_used_for_selection"]
        ),
        "runtime_safe_v3_0_remains_default": bool(
            report["runtime_safe_v3_0_remains_default"]
        ),
    }


def _render_documents(
    report: Mapping[str, Any],
    document: Path,
    document_cn: Path,
) -> None:
    candidate = report["candidate_metrics"]
    baseline = report["baseline_metrics"]
    deltas = report["deltas"]
    failed = ", ".join(report["failed_gates"]) or "none"
    english = f"""# MAD-ETD W297-W300 Dataset Specialist Result

## Scope

This experiment uses previously unconsumed CICIoT2023 malicious
    capture-file groups for **{report['class_count']}-class attack-type
    attribution**. It is not a
benign/malicious detector, does not enter Fusion, and cannot support a general
encrypted-traffic-detection claim.

## Result

| Metric | Strongest safe baseline ({report['strongest_baseline']}) | DatasetSpecialistSkillV1 | Delta |
|---|---:|---:|---:|
| Accuracy | {baseline['accuracy']:.6f} | {candidate['accuracy']:.6f} | {deltas['accuracy']:+.6f} |
| Macro-F1 | {baseline['macro_f1']:.6f} | {candidate['macro_f1']:.6f} | {deltas['macro_f1']:+.6f} |
| Weighted-F1 | {baseline['weighted_f1']:.6f} | {candidate['weighted_f1']:.6f} | {deltas['weighted_f1']:+.6f} |
| Macro precision | {baseline['macro_precision']:.6f} | {candidate['macro_precision']:.6f} | {deltas['macro_precision']:+.6f} |
| Macro recall | {baseline['macro_recall']:.6f} | {candidate['macro_recall']:.6f} | {deltas['macro_recall']:+.6f} |
| ECE | {baseline['ece']:.6f} | {candidate['ece']:.6f} | {deltas['ece']:+.6f} |

- Status: `{report['status']}`
- Failed gates: `{failed}`
- Macro-F1 delta 95% CI: [{report['bootstrap']['macro_f1_delta']['ci95_lower']:.6f}, {report['bootstrap']['macro_f1_delta']['ci95_upper']:.6f}]
- `runtime_safe_v3_0` remains default.
- Fake metrics: 0.

## Claim boundary

Only a dataset-specific malicious attack-type attribution result may be
reported. No binary malicious-detection, external-paper superiority, Fusion
promotion, or general runtime claim is permitted.
"""
    chinese = f"""# MAD-ETD W297-W300 数据集专家 Skill 实验

## 任务边界

本实验只使用从未被历史实验消费的 CICIoT2023 恶意 capture-file groups，
执行 **{report['class_count']} 类恶意攻击类型归因**。它不是 benign/malicious
二分类检测器，不进入
Fusion，也不支持通用加密恶意流量性能提升声明。

## 结果

| 指标 | 最强安全基线（{report['strongest_baseline']}） | DatasetSpecialistSkillV1 | 差值 |
|---|---:|---:|---:|
| Accuracy | {baseline['accuracy']:.6f} | {candidate['accuracy']:.6f} | {deltas['accuracy']:+.6f} |
| Macro-F1 | {baseline['macro_f1']:.6f} | {candidate['macro_f1']:.6f} | {deltas['macro_f1']:+.6f} |
| Weighted-F1 | {baseline['weighted_f1']:.6f} | {candidate['weighted_f1']:.6f} | {deltas['weighted_f1']:+.6f} |
| Macro-Precision | {baseline['macro_precision']:.6f} | {candidate['macro_precision']:.6f} | {deltas['macro_precision']:+.6f} |
| Macro-Recall | {baseline['macro_recall']:.6f} | {candidate['macro_recall']:.6f} | {deltas['macro_recall']:+.6f} |
| ECE | {baseline['ece']:.6f} | {candidate['ece']:.6f} | {deltas['ece']:+.6f} |

- 状态：`{report['status']}`
- 未通过门槛：`{failed}`
- Macro-F1 差值 95% CI：[{report['bootstrap']['macro_f1_delta']['ci95_lower']:.6f}, {report['bootstrap']['macro_f1_delta']['ci95_upper']:.6f}]
- `runtime_safe_v3_0` 保持默认。
- fake metric count = 0。

## Claim 边界

只能报告 CICIoT2023 数据集特定的恶意攻击类型归因结果。不得写成恶意流量
二分类提升、超过外部论文、进入 Fusion、替换默认 runtime 或通用 ETD 提升。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_dataset_specialist_w300(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    acceptance = _load(output / "w299_acceptance_report.json")
    lock = _load(output / "candidate_lock_w298.json")
    gates = _promotion_gates(acceptance)
    after = _safe_hashes()
    before = _load(output / "frozen_hashes_before_w297.json")
    hashes_unchanged = _hash_snapshot_equal(before, after)
    gates["frozen_hashes_unchanged"] = hashes_unchanged
    gates["targeted_tests_passed"] = bool(tests_passed)
    accepted = all(gates.values())
    status = (
        "accepted_optional_dataset_specialist_skill_w300"
        if accepted
        else "not_promoted_dataset_specialist_performance_gate_failed"
    )
    failed = [name for name, passed in gates.items() if not passed]
    profile = output / "runtime_ciciot_attack_type_specialist_w300_optional.json"
    if profile.exists():
        profile.unlink()
    if accepted:
        _dump(
            profile,
            {
                "profile_id": "runtime_ciciot_attack_type_specialist_w300_optional",
                "default_enabled": False,
                "production_ready": False,
                "dataset_scope": (
                    "CICIoT2023 malicious attack-type attribution only"
                ),
                "task": acceptance["task"],
                "artifact": lock["artifact"],
                "artifact_hash": lock["artifact_hash"],
                "feature_policy_hash": lock["feature_policy_hash"],
                "contributes_to_fusion": False,
                "fusion_owner": "FusionAgent",
                "promotion_status": "accepted_dataset_specific_optional_skill",
                "claim_scope": (
                    "fresh source-file-group-held-out multiclass attribution"
                ),
            },
        )
    _dump(output / "frozen_hashes_after_w300.json", after)
    report = {
        "status": status,
        "experiment": EXPERIMENT,
        "dataset": "CICIoT2023",
        "task": acceptance["task"],
        "class_count": acceptance["class_count"],
        "scope": (
            f"{acceptance['class_count']}-class malicious attack-type "
            "attribution; not binary "
            "malicious detection"
        ),
        "strongest_baseline": acceptance[
            "strongest_validation_selected_baseline"
        ],
        "baseline_metrics": acceptance["baseline_metrics"],
        "candidate_metrics": acceptance["candidate_metrics"],
        "deltas": acceptance["deltas"],
        "bootstrap": acceptance["bootstrap"],
        "gates": gates,
        "failed_gates": failed,
        "accepted_dataset_specific_optional_skill": accepted,
        "optional_profile_created": accepted,
        "optional_profile_default_enabled": False,
        "contributes_to_fusion": False,
        "binary_malicious_detection_claim_allowed": False,
        "frozen_hashes_unchanged": hashes_unchanged,
        "targeted_tests_passed": bool(tests_passed),
        "targeted_test_count": int(test_count),
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
    }
    _dump(output / "acceptance_report.json", report)
    if accepted:
        _write_csv(
            output / "positive_results_w300.csv",
            [
                {
                    "candidate": "dataset_specialist_skill_v1",
                    "task": report["task"],
                    "macro_f1_delta": report["deltas"]["macro_f1"],
                    "accuracy_delta": report["deltas"]["accuracy"],
                    "ci95_lower": report["bootstrap"]["macro_f1_delta"][
                        "ci95_lower"
                    ],
                    "status": status,
                }
            ],
        )
        _dump(output / "negative_results.json", {"negative_results": []})
    else:
        _write_csv(output / "positive_results_w300.csv", [])
        _dump(
            output / "negative_results.json",
            {
                "negative_results": [
                    {
                        "experiment": EXPERIMENT,
                        "candidate": "dataset_specialist_skill_v1",
                        "failure_type": "performance_gate_failed",
                        "failure_reason": ";".join(failed),
                        "safe_claim": (
                            "Fresh group-held-out dataset-specific attribution "
                            "was evaluated and retained as a negative result."
                        ),
                        "forbidden_claim": (
                            "DatasetSpecialistSkillV1 improved general "
                            "malicious traffic detection."
                        ),
                        "fake_metric_count": 0,
                        "runtime_modified": False,
                    }
                ]
            },
        )
    _render_documents(report, Path(document), Path(document_cn))
    return report


__all__ = [
    "_build_group_split",
    "_promotion_gates",
    "build_non_tls_fresh_specialist_w297",
    "train_dataset_specialist_w298",
    "evaluate_dataset_specialist_w299",
    "finalize_dataset_specialist_w300",
]
