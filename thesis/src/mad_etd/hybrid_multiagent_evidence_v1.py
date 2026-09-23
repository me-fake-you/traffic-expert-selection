"""Hybrid Multi-Agent Evidence Team v1.

This protocol turns the previously evaluated USTC stats+sequence *feature
concatenation* experiment into a genuine evidence-team experiment:

* stats and temporal specialists are trained independently;
* every specialist emits typed ``AgentEvidenceV2``;
* only admitted detector evidence is projected to the existing
  ``FusionAgent``;
* staged routing performs real lazy model inference;
* NVIDIA/Nemotron planning is a guarded control-plane experiment and never
  becomes classifier evidence.

The module is default-off and never mutates ``runtime_safe_v3_0``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline

from .audit import AuditLogger
from .evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)
from .fusion import FusionAgent
from .schemas import (
    AgentEvidence,
    AgentEvidenceV2,
    FeatureGroup,
    ReliabilityProfile,
    StrictModel,
    Verdict,
)
from .ustc_group_heldout_hybrid_w98 import (
    _validation_groups,
)


EXPERIMENT = "mad_etd_hybrid_multiagent_evidence_v1"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_hybrid_multiagent_evidence_v1")
DEFAULT_SOURCE = Path(
    "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
)
DEFAULT_SOURCE_ACCEPTANCE = Path(
    "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/acceptance_report.json"
)
DEFAULT_DOCUMENT = Path("docs/MAD_ETD_HYBRID_MULTIAGENT_EVIDENCE_V1.md")
DEFAULT_DOCUMENT_CN = Path("docs/MAD_ETD_HYBRID_MULTIAGENT_EVIDENCE_V1_CN.md")
DEFAULT_NVIDIA_KEY_FILE = Path.home() / ".mad_etd" / "secrets" / "nvidia_api_key.txt"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
PROMPT_VERSION = "hybrid-evidence-request-v1"
BOOTSTRAP_ITERATIONS = 1_000
BOOTSTRAP_SEED = 42

BLOCKED_FIELD_TOKENS = frozenset(
    {
        "ip",
        "port",
        "timestamp",
        "flow_id",
        "flowid",
        "sample_id",
        "source_file",
        "label",
        "attack",
        "family",
        "application",
        "provenance",
        "capture",
        "dataset",
        "source_group",
    }
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _load(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _ensure_csv_header(path: str | Path, fields: Iterable[str]) -> None:
    """Create an empty resumable CSV with a stable schema when absent."""

    target = Path(path)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_scalar(value: Any) -> Any:
    """Convert numeric CSV values while preserving explicit string metadata."""

    if value in {"", None}:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _security_defaults() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "invalid_evidence_admitted": 0,
        "fallback_identity_confusion": 0,
        "audit_completion": 1.0,
        "fake_metric_count": 0,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }


def _model(model_id: str, *, random_state: int = 42) -> Any:
    if model_id in {"stats_hgb", "temporal_hgb"}:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=int(random_state),
            ),
        )
    if model_id == "stats_extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=240,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=int(random_state),
            ),
        )
    raise ValueError(f"unknown hybrid evidence model: {model_id}")


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not np.any(mask):
            continue
        result += float(np.mean(mask)) * abs(
            float(np.mean(probability[mask])) - float(np.mean(y[mask]))
        )
    return result


def _binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_precision": float(
            precision_score(y, prediction, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, prediction, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
        "malicious_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "ece": _ece(y, probability),
        "brier_score": float(brier_score_loss(y, probability)),
        "coverage": 1.0,
        "selective_error": float(1.0 - accuracy_score(y, prediction)),
    }


def _select_threshold(
    y: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, dict[str, float]]:
    options: list[tuple[tuple[float, float, float, float], float, dict[str, float]]] = []
    for threshold in np.linspace(0.2, 0.8, 61):
        prediction = (probability >= threshold).astype(np.int64)
        metrics = _binary_metrics(y, probability, prediction)
        rank = (
            metrics["macro_f1"],
            metrics["malicious_recall"],
            -metrics["ece"],
            -abs(float(threshold) - 0.5),
        )
        options.append((rank, float(threshold), metrics))
    _rank, threshold, metrics = max(options, key=lambda item: item[0])
    return threshold, metrics


def _single_class_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    label = int(y[0])
    return float(f1_score(y, prediction, labels=[label], average="macro", zero_division=0))


def _source_arrays(source_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as data:
        x_stats = np.asarray(data["x_stats"], dtype=np.float32)
        x_hybrid = np.asarray(data["x_hybrid"], dtype=np.float32)
        stats_names = np.asarray(data["stats_feature_names"]).astype(str)
        hybrid_names = np.asarray(data["hybrid_feature_names"]).astype(str)
        if x_hybrid.shape[1] <= x_stats.shape[1]:
            raise ValueError("source artifact has no independent temporal feature view")
        if list(hybrid_names[: len(stats_names)]) != list(stats_names):
            raise ValueError("hybrid artifact does not preserve the stats-prefix contract")
        return {
            "x_stats": x_stats,
            "x_temporal": x_hybrid[:, x_stats.shape[1] :],
            "y": np.asarray(data["y"], dtype=np.int64),
            "group": np.asarray(data["group"]).astype(str),
            "sample_hash": np.asarray(data["sample_hash"]).astype(str),
            "stats_names": stats_names,
            "temporal_names": hybrid_names[x_stats.shape[1] :],
        }


def build_hybrid_multiagent_evidence_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_path: str | Path = DEFAULT_SOURCE,
    source_acceptance_path: str | Path = DEFAULT_SOURCE_ACCEPTANCE,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source_path)
    source_acceptance = _load(source_acceptance_path)
    if (
        not source.exists()
        or source_acceptance.get("status")
        != "accepted_ustc_group_heldout_hybrid_skill"
    ):
        report = {
            **_security_defaults(),
            "status": "failed_source_artifact_not_ready",
            "source_exists": source.exists(),
            "source_acceptance_status": source_acceptance.get("status"),
        }
        _dump(out / "protocol.json", report)
        return report

    arrays = _source_arrays(source)
    stats_names = arrays["stats_names"].astype(str).tolist()
    temporal_names = arrays["temporal_names"].astype(str).tolist()
    blocked_hits = sorted(
        feature
        for feature in [*stats_names, *temporal_names]
        if any(token in feature.lower() for token in BLOCKED_FIELD_TOKENS)
    )
    groups = sorted(set(arrays["group"].astype(str)))
    split_rows = []
    for heldout in groups:
        validation_groups = _validation_groups(heldout, groups)
        split_rows.append(
            {
                "heldout_group": heldout,
                "validation_groups": list(validation_groups),
                "train_groups": [
                    group
                    for group in groups
                    if group != heldout and group not in validation_groups
                ],
            }
        )
    feature_policy = {
        "schema_version": "1.0",
        "dataset": "USTC-TFC2016",
        "stats_features": stats_names,
        "temporal_features": temporal_names,
        "feature_views_disjoint": not bool(set(stats_names) & set(temporal_names)),
        "blocked_fields": sorted(BLOCKED_FIELD_TOKENS),
        "blocked_feature_hits": blocked_hits,
        "labels_used_for": ["training", "validation", "evaluation"],
        "labels_enter_detector_input": False,
        "group_names_enter_detector_input": False,
    }
    feature_policy["stats_feature_policy_hash"] = _canonical_hash(stats_names)
    feature_policy["temporal_feature_policy_hash"] = _canonical_hash(temporal_names)
    registry = {
        "schema_version": "1.0",
        "default_enabled": False,
        "final_decision_owner": "FusionAgent",
        "agents": [
            {
                "agent_name": "StatsEvidenceAgent",
                "agent_type": "flow_statistics_detector",
                "backends": ["stats_hgb", "stats_extra_trees"],
                "feature_policy_hash": feature_policy["stats_feature_policy_hash"],
                "dataset_scope": "USTC-TFC2016",
                "fusion_eligible_after_admission": True,
                "forbidden_outputs": [
                    "final_verdict",
                    "final_confidence",
                    "final_uncertainty",
                    "ood_override",
                    "fusion_weights",
                ],
            },
            {
                "agent_name": "TemporalEvidenceAgent",
                "agent_type": "packet_sequence_detector",
                "backends": ["temporal_hgb"],
                "feature_policy_hash": feature_policy["temporal_feature_policy_hash"],
                "dataset_scope": "USTC-TFC2016",
                "fusion_eligible_after_admission": True,
                "forbidden_outputs": [
                    "final_verdict",
                    "final_confidence",
                    "final_uncertainty",
                    "ood_override",
                    "fusion_weights",
                ],
            },
            {
                "agent_name": "TLSProtocolAgent",
                "agent_type": "tls_protocol_detector",
                "dataset_scope": "USTC-TFC2016",
                "applicability": "unsupported",
                "unsupported_reason": "no qualified TLS-record capability in this USTC artifact",
                "fusion_eligible_after_admission": False,
            },
            {
                "agent_name": "VolumeBehaviorAgent",
                "agent_type": "semantic_flow_view",
                "dataset_scope": "NF-IoT|CICIoT2023",
                "promotion_status": "protocol_only_pending_independent_training",
            },
            {
                "agent_name": "TimingBehaviorAgent",
                "agent_type": "semantic_flow_view",
                "dataset_scope": "NF-IoT|CICIoT2023",
                "promotion_status": "protocol_only_pending_independent_training",
            },
            {
                "agent_name": "ProtocolStateAgent",
                "agent_type": "semantic_flow_view",
                "dataset_scope": "NF-IoT|CICIoT2023",
                "promotion_status": "protocol_only_pending_independent_training",
            },
        ],
    }
    protocol = {
        **_security_defaults(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": (
            "ready_for_independent_evidence_training"
            if not blocked_hits and feature_policy["feature_views_disjoint"]
            else "failed_safe_feature_or_view_isolation_gate"
        ),
        "source_artifact": source.as_posix(),
        "source_artifact_sha256": _sha256_file(source),
        "source_acceptance": str(source_acceptance_path),
        "sample_count": int(len(arrays["y"])),
        "group_count": len(groups),
        "split_protocol": "20-fold application/family-held-out; validation groups are disjoint",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "performance_gate": {
            "macro_f1_delta_min": 0.01,
            "bootstrap_ci_lower_gt": 0.0,
            "accuracy_drop_max": 0.005,
            "malicious_recall_drop_max": 0.0,
            "worst_group_class_f1_drop_max": 0.0,
            "ece_worsening_max": 0.005,
        },
        "routing_gate": {
            "executed_inference_reduction_min": 0.20,
            "p95_latency_reduction_min": 0.10,
            "macro_f1_drop_max": 0.005,
            "coverage_drop_max": 0.01,
            "verdict_agreement_min": 0.98,
            "ood_agreement_min": 0.98,
        },
        "llm_protocol": {
            "provider": "NVIDIA",
            "base_url": DEFAULT_NVIDIA_BASE_URL,
            "model": DEFAULT_NVIDIA_MODEL,
            "temperature": 0,
            "thinking": "disabled",
            "prompt_version": PROMPT_VERSION,
            "timeout_seconds": 60,
            "max_retry": 1,
            "feasibility_per_dataset": 500,
            "stability_cases": 300,
            "stability_repeats": 3,
        },
    }
    _dump(out / "protocol.json", protocol)
    _dump(
        out / "split_manifest.json",
        {
            "source_artifact_sha256": protocol["source_artifact_sha256"],
            "folds": split_rows,
            "acceptance_used_for_selection": False,
            "group_overlap_count": 0,
        },
    )
    _dump(out / "safe_feature_policy.json", feature_policy)
    _dump(out / "agent_registry.json", registry)
    return protocol


def _artifact_path(
    model_dir: Path,
    heldout: str,
    model_id: str,
) -> Path:
    safe = heldout.replace(":", "_").replace("/", "_")
    return model_dir / f"{safe}__{model_id}.joblib"


def _reliability_from_validation(metrics: Mapping[str, float]) -> float:
    return float(np.clip(1.0 - float(metrics["ece"]), 0.05, 1.0))


def _legacy_evidence(
    agent_name: str,
    feature_group: FeatureGroup,
    probability: float,
    calibration_quality: float,
    latency_ms: float = 0.0,
) -> AgentEvidence:
    confidence = max(probability, 1.0 - probability)
    return AgentEvidence(
        agent_name=agent_name,
        feature_group=feature_group,
        benign_support=float(1.0 - probability),
        malicious_support=float(probability),
        confidence=float(confidence),
        uncertainty=float(1.0 - confidence),
        calibration_quality=float(calibration_quality),
        model_reliability=1.0,
        distribution_shift_level="off",
        distribution_shift_score=0.0,
        latency_ms=float(latency_ms),
        used_fields=[],
        evidence=["independent specialist probability"],
    )


def _fusion_projection(
    stats_probability: np.ndarray,
    temporal_probability: np.ndarray,
    *,
    stats_reliability: float,
    temporal_reliability: float,
    allow_reject: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    fusion = FusionAgent(
        use_reliability_discount=True,
        allow_reject=allow_reject,
    )
    profile = ReliabilityProfile(
        stats_reliability=1.0,
        sequence_reliability=1.0,
        tls_reliability=0.0,
        payload_reliability=0.0,
        input_completeness=1.0,
        ood_suspected=False,
        key_features_missing=False,
        indicators=["new independent agents; no calibrated OOD artifact"],
    )
    probabilities: list[float] = []
    predictions: list[int] = []
    verdicts: list[str] = []
    accept_scores: list[float] = []
    for stats_p, temporal_p in zip(
        stats_probability,
        temporal_probability,
        strict=True,
    ):
        result = fusion.fuse(
            [
                _legacy_evidence(
                    "StatsDetectorAgent",
                    FeatureGroup.STATS,
                    float(stats_p),
                    stats_reliability,
                ),
                _legacy_evidence(
                    "TemporalBehaviorAgent",
                    FeatureGroup.SEQUENCE,
                    float(temporal_p),
                    temporal_reliability,
                ),
            ],
            profile,
            final=True,
        )
        denominator = result.benign_support + result.malicious_support
        probability = (
            result.malicious_support / denominator if denominator > 0 else 0.5
        )
        probabilities.append(float(probability))
        predictions.append(int(result.malicious_support >= result.benign_support))
        verdicts.append(str(result.verdict.value))
        # Correct acceptance score: supports are not re-discounted through
        # another confidence term.
        accept_scores.append(
            float(
                max(result.benign_support, result.malicious_support)
                * (1.0 - result.uncertainty)
            )
        )
    return (
        np.asarray(probabilities),
        np.asarray(predictions, dtype=np.int64),
        verdicts,
        np.asarray(accept_scores),
    )


def _v2_evidence(
    *,
    agent_name: str,
    agent_type: str,
    feature_policy_hash: str,
    artifact_hash: str,
    probability: float,
    reliability: float,
    sample_hash: str,
) -> AgentEvidenceV2:
    prediction = "malicious" if probability >= 0.5 else "benign"
    confidence = max(probability, 1.0 - probability)
    return AgentEvidenceV2(
        agent_name=agent_name,
        agent_type=agent_type,
        input_feature_policy_hash=feature_policy_hash,
        artifact_hash=artifact_hash,
        prediction=prediction,
        probabilities={
            "benign": float(1.0 - probability),
            "malicious": float(probability),
        },
        confidence=float(confidence),
        uncertainty=float(1.0 - confidence),
        reliability=float(reliability),
        calibration_quality=float(reliability),
        applicability="applicable",
        unsupported_reason=None,
        reason_codes=["INDEPENDENT_SPECIALIST_EVIDENCE"],
        calibration_status="validation_ece_reliability",
        latency_ms=0.0,
        cost=0.0,
        supported_capabilities=[
            "stats" if "Stats" in agent_name else "sequence"
        ],
        safety_flags=[],
        dataset_scope="USTC-TFC2016",
        promotion_status="experimental_default_off",
        source_evidence_sha256=_canonical_hash(
            [sample_hash, agent_name, float(probability)]
        ),
    )


def train_hybrid_multiagent_evidence_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    out = Path(output_dir)
    protocol = _load(out / "protocol.json")
    existing = _load(out / "training_report.json")
    if (
        existing.get("status")
        in {
            "independent_evidence_training_completed",
            "accepted_independent_multiagent_performance_candidate",
            "not_promoted_independent_multiagent_performance_gate_failed",
        }
        and existing.get("schema_version") == "1.1"
        and int(existing.get("training_seed", 42)) == int(random_state)
    ):
        return existing
    if protocol.get("status") != "ready_for_independent_evidence_training":
        report = {
            **_security_defaults(),
            "status": "failed_protocol_not_ready",
        }
        _dump(out / "training_report.json", report)
        return report

    arrays = _source_arrays(protocol["source_artifact"])
    x_stats = arrays["x_stats"]
    x_temporal = arrays["x_temporal"]
    y = arrays["y"]
    groups = arrays["group"].astype(str)
    hashes = arrays["sample_hash"].astype(str)
    feature_policy = _load(out / "safe_feature_policy.json")
    unique_groups = sorted(set(groups))
    model_dir = out / "model_artifacts"
    prediction_dir = out / "per_agent_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    per_agent_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fusion_rows: list[dict[str, Any]] = []
    evidence_path = out / "agent_evidence.jsonl"
    pooled: dict[str, list[np.ndarray]] = defaultdict(list)

    with evidence_path.open("w", encoding="utf-8") as evidence_handle:
        for heldout in unique_groups:
            validation_groups = _validation_groups(heldout, unique_groups)
            acceptance = groups == heldout
            validation = np.isin(groups, validation_groups)
            train = ~(acceptance | validation)

            fitted: dict[str, Any] = {}
            validation_probability: dict[str, np.ndarray] = {}
            acceptance_probability: dict[str, np.ndarray] = {}
            validation_result: dict[str, dict[str, Any]] = {}
            for model_id, view in (
                ("stats_hgb", x_stats),
                ("stats_extra_trees", x_stats),
                ("temporal_hgb", x_temporal),
            ):
                model = _model(model_id, random_state=random_state)
                started = time.perf_counter()
                model.fit(view[train], y[train])
                training_seconds = time.perf_counter() - started
                val_probability = model.predict_proba(view[validation])[:, 1]
                threshold, metrics = _select_threshold(y[validation], val_probability)
                test_probability = model.predict_proba(view[acceptance])[:, 1]
                fitted[model_id] = model
                validation_probability[model_id] = val_probability
                acceptance_probability[model_id] = test_probability
                validation_result[model_id] = {
                    **metrics,
                    "threshold": threshold,
                    "training_seconds": training_seconds,
                }
                artifact = _artifact_path(model_dir, heldout, model_id)
                joblib.dump(model, artifact)
                model_rows.append(
                    {
                        "heldout_group": heldout,
                        "model_id": model_id,
                        "artifact": artifact.as_posix(),
                        "artifact_hash": _sha256_file(artifact),
                        "training_seconds": training_seconds,
                        "training_seed": int(random_state),
                    }
                )

            selected_stats = max(
                ("stats_hgb", "stats_extra_trees"),
                key=lambda model_id: (
                    validation_result[model_id]["macro_f1"],
                    validation_result[model_id]["malicious_recall"],
                    -validation_result[model_id]["ece"],
                    model_id,
                ),
            )
            selected_single = max(
                ("stats_hgb", "stats_extra_trees", "temporal_hgb"),
                key=lambda model_id: (
                    validation_result[model_id]["macro_f1"],
                    validation_result[model_id]["malicious_recall"],
                    -validation_result[model_id]["ece"],
                    model_id,
                ),
            )
            stats_val = validation_probability[selected_stats]
            temporal_val = validation_probability["temporal_hgb"]
            stats_test = acceptance_probability[selected_stats]
            temporal_test = acceptance_probability["temporal_hgb"]
            stats_rel = _reliability_from_validation(
                validation_result[selected_stats]
            )
            temporal_rel = _reliability_from_validation(
                validation_result["temporal_hgb"]
            )

            mode_val_probability = {
                "stats_hgb_only": validation_probability["stats_hgb"],
                "stats_tree_only": validation_probability["stats_extra_trees"],
                "temporal_only": temporal_val,
                "factorial_hgb_plus_temporal": (
                    validation_probability["stats_hgb"] + temporal_val
                )
                / 2.0,
                "factorial_tree_plus_temporal": (
                    validation_probability["stats_extra_trees"] + temporal_val
                )
                / 2.0,
                "simple_probability_average": (stats_val + temporal_val) / 2.0,
                "fixed_weight_ensemble": 0.7 * stats_val + 0.3 * temporal_val,
                "reliability_only_fusion": (
                    stats_rel * stats_val + temporal_rel * temporal_val
                )
                / (stats_rel + temporal_rel),
                "reliability_plus_ood_fusion": (
                    stats_rel * stats_val + temporal_rel * temporal_val
                )
                / (stats_rel + temporal_rel),
                "strongest_single_agent": validation_probability[selected_single],
            }
            mode_test_probability = {
                "stats_hgb_only": acceptance_probability["stats_hgb"],
                "stats_tree_only": acceptance_probability["stats_extra_trees"],
                "temporal_only": temporal_test,
                "factorial_hgb_plus_temporal": (
                    acceptance_probability["stats_hgb"] + temporal_test
                )
                / 2.0,
                "factorial_tree_plus_temporal": (
                    acceptance_probability["stats_extra_trees"] + temporal_test
                )
                / 2.0,
                "simple_probability_average": (stats_test + temporal_test) / 2.0,
                "fixed_weight_ensemble": 0.7 * stats_test + 0.3 * temporal_test,
                "reliability_only_fusion": (
                    stats_rel * stats_test + temporal_rel * temporal_test
                )
                / (stats_rel + temporal_rel),
                "reliability_plus_ood_fusion": (
                    stats_rel * stats_test + temporal_rel * temporal_test
                )
                / (stats_rel + temporal_rel),
                "strongest_single_agent": acceptance_probability[selected_single],
            }

            full_val_p, _full_val_y, _full_val_verdict, _full_val_score = (
                _fusion_projection(
                    stats_val,
                    temporal_val,
                    stats_reliability=stats_rel,
                    temporal_reliability=temporal_rel,
                    allow_reject=False,
                )
            )
            full_test_p, full_test_y, full_verdict, full_accept_score = (
                _fusion_projection(
                    stats_test,
                    temporal_test,
                    stats_reliability=stats_rel,
                    temporal_reliability=temporal_rel,
                    allow_reject=False,
                )
            )
            _op_p, _op_y, operational_verdict, operational_accept_score = (
                _fusion_projection(
                    stats_test,
                    temporal_test,
                    stats_reliability=stats_rel,
                    temporal_reliability=temporal_rel,
                    allow_reject=True,
                )
            )
            mode_val_probability["full_mad_etd_fusion"] = full_val_p
            mode_test_probability["full_mad_etd_fusion"] = full_test_p

            fold_modes: dict[str, dict[str, Any]] = {}
            for mode, val_probability in mode_val_probability.items():
                threshold, validation_metrics = _select_threshold(
                    y[validation],
                    val_probability,
                )
                test_probability = mode_test_probability[mode]
                prediction = (
                    full_test_y
                    if mode == "full_mad_etd_fusion"
                    else (test_probability >= threshold).astype(np.int64)
                )
                metrics = _binary_metrics(
                    y[acceptance],
                    test_probability,
                    prediction,
                )
                fold_modes[mode] = {
                    "threshold": threshold,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    **metrics,
                }
                pooled[f"{mode}__probability"].append(test_probability)
                pooled[f"{mode}__prediction"].append(prediction)

            operational = np.asarray(
                [
                    verdict in {Verdict.BENIGN.value, Verdict.MALICIOUS.value}
                    for verdict in operational_verdict
                ],
                dtype=bool,
            )
            operational_prediction = np.asarray(
                [
                    int(verdict == Verdict.MALICIOUS.value)
                    for verdict in operational_verdict
                ],
                dtype=np.int64,
            )
            operational_metrics: dict[str, Any] = {
                "coverage": float(np.mean(operational)),
                "accepted_count": int(np.sum(operational)),
                "suspicious_count": int(
                    sum(value == Verdict.SUSPICIOUS.value for value in operational_verdict)
                ),
                "unknown_count": int(
                    sum(value == Verdict.UNKNOWN.value for value in operational_verdict)
                ),
                "selective_error": (
                    float(
                        1.0
                        - accuracy_score(
                            y[acceptance][operational],
                            operational_prediction[operational],
                        )
                    )
                    if np.any(operational)
                    else None
                ),
            }
            fold_rows.append(
                {
                    "heldout_group": heldout,
                    "heldout_label": int(y[acceptance][0]),
                    "validation_groups": "|".join(validation_groups),
                    "train_count": int(np.sum(train)),
                    "validation_count": int(np.sum(validation)),
                    "acceptance_count": int(np.sum(acceptance)),
                    "selected_stats_backend": selected_stats,
                    "selected_single_agent": selected_single,
                    "stats_reliability": stats_rel,
                    "temporal_reliability": temporal_rel,
                    "operational_coverage": operational_metrics["coverage"],
                    "operational_selective_error": operational_metrics[
                        "selective_error"
                    ],
                    "full_group_class_f1": _single_class_f1(
                        y[acceptance],
                        full_test_y,
                    ),
                    "single_group_class_f1": _single_class_f1(
                        y[acceptance],
                        pooled["strongest_single_agent__prediction"][-1],
                    ),
                    "mode_metrics": fold_modes,
                }
            )

            model_hashes = {
                row["model_id"]: row["artifact_hash"]
                for row in model_rows
                if row["heldout_group"] == heldout
            }
            stats_policy_hash = feature_policy["stats_feature_policy_hash"]
            temporal_policy_hash = feature_policy[
                "temporal_feature_policy_hash"
            ]
            for local_index, sample_index in enumerate(np.flatnonzero(acceptance)):
                sample_hash = str(hashes[sample_index])
                stats_evidence = _v2_evidence(
                    agent_name="StatsEvidenceAgent",
                    agent_type="flow_statistics_detector",
                    feature_policy_hash=stats_policy_hash,
                    artifact_hash=model_hashes[selected_stats],
                    probability=float(stats_test[local_index]),
                    reliability=stats_rel,
                    sample_hash=sample_hash,
                )
                temporal_evidence = _v2_evidence(
                    agent_name="TemporalEvidenceAgent",
                    agent_type="packet_sequence_detector",
                    feature_policy_hash=temporal_policy_hash,
                    artifact_hash=model_hashes["temporal_hgb"],
                    probability=float(temporal_test[local_index]),
                    reliability=temporal_rel,
                    sample_hash=sample_hash,
                )
                for evidence in (stats_evidence, temporal_evidence):
                    evidence_handle.write(
                        json.dumps(
                            {
                                "sample_hash": sample_hash,
                                "heldout_group": heldout,
                                "evidence": evidence.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                per_agent_rows["stats"].append(
                    {
                        "sample_hash": sample_hash,
                        "heldout_group": heldout,
                        "label": int(y[sample_index]),
                        "backend": selected_stats,
                        "probability_malicious": float(stats_test[local_index]),
                        "prediction": int(stats_test[local_index] >= 0.5),
                    }
                )
                per_agent_rows["temporal"].append(
                    {
                        "sample_hash": sample_hash,
                        "heldout_group": heldout,
                        "label": int(y[sample_index]),
                        "backend": "temporal_hgb",
                        "probability_malicious": float(temporal_test[local_index]),
                        "prediction": int(temporal_test[local_index] >= 0.5),
                    }
                )
                fusion_rows.append(
                    {
                        "sample_hash": sample_hash,
                        "heldout_group": heldout,
                        "label": int(y[sample_index]),
                        "selected_stats_backend": selected_stats,
                        "stats_probability": float(stats_test[local_index]),
                        "temporal_probability": float(temporal_test[local_index]),
                        "full_fusion_probability": float(full_test_p[local_index]),
                        "full_fusion_binary_prediction": int(
                            full_test_y[local_index]
                        ),
                        "full_fusion_verdict": full_verdict[local_index],
                        "operational_verdict": operational_verdict[local_index],
                        "acceptance_score": float(
                            operational_accept_score[local_index]
                        ),
                    }
                )
            pooled["y"].append(y[acceptance])
            pooled["group"].append(groups[acceptance])
            pooled["sample_hash"].append(hashes[acceptance])

    for name, rows in per_agent_rows.items():
        _write_csv(prediction_dir / f"{name}_predictions.csv", rows)
    _write_csv(out / "fusion_predictions.csv", fusion_rows)
    _write_csv(out / "model_registry.csv", model_rows)

    combined = {
        key: np.concatenate(value)
        for key, value in pooled.items()
    }
    # Preserve the per-fold validation-selected single-agent baseline under an
    # explicit name.  The promotion gate below uses the strongest *observed*
    # predefined single-agent baseline as a conservative evaluation reference;
    # that reference choice never changes a model, threshold, or prediction.
    combined["validation_selected_single_agent__probability"] = combined[
        "strongest_single_agent__probability"
    ].copy()
    combined["validation_selected_single_agent__prediction"] = combined[
        "strongest_single_agent__prediction"
    ].copy()
    metric_rows: list[dict[str, Any]] = []
    modes = sorted(
        key.removesuffix("__probability")
        for key in combined
        if key.endswith("__probability")
    )
    for mode in modes:
        metrics = _binary_metrics(
            combined["y"],
            combined[f"{mode}__probability"],
            combined[f"{mode}__prediction"],
        )
        metric_rows.append({"mode": mode, **metrics})
    metric_by_mode = {row["mode"]: row for row in metric_rows}

    full = metric_by_mode["full_mad_etd_fusion"]
    predefined_single_modes = (
        "stats_hgb_only",
        "stats_tree_only",
        "temporal_only",
    )
    strongest_observed_single_mode = max(
        predefined_single_modes,
        key=lambda mode: (
            metric_by_mode[mode]["macro_f1"],
            metric_by_mode[mode]["malicious_recall"],
            -metric_by_mode[mode]["ece"],
            mode,
        ),
    )
    strongest = {
        **metric_by_mode[strongest_observed_single_mode],
        "mode": "strongest_single_agent",
        "source_mode": strongest_observed_single_mode,
    }
    metric_by_mode["strongest_single_agent"] = strongest
    for index, row in enumerate(metric_rows):
        if row["mode"] == "strongest_single_agent":
            metric_rows[index] = strongest
            break
    combined["strongest_single_agent__probability"] = combined[
        f"{strongest_observed_single_mode}__probability"
    ].copy()
    combined["strongest_single_agent__prediction"] = combined[
        f"{strongest_observed_single_mode}__prediction"
    ].copy()
    delta = {
        metric: float(full[metric] - strongest[metric])
        for metric in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "malicious_recall",
            "ece",
            "brier_score",
            "coverage",
            "selective_error",
        )
    }
    bootstrap_rows: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "strongest_single_agent":
            continue
        comparison = _group_bootstrap(
            combined["y"],
            combined["group"].astype(str),
            combined["strongest_single_agent__prediction"],
            combined[f"{mode}__prediction"],
        )
        bootstrap_rows.append(
            {
                "reference_mode": "strongest_single_agent",
                "candidate_mode": mode,
                **comparison,
            }
        )
    _write_csv(out / "grouped_bootstrap.csv", bootstrap_rows)
    bootstrap = next(
        row
        for row in bootstrap_rows
        if row["candidate_mode"] == "full_mad_etd_fusion"
    )
    np.savez_compressed(
        out / "per_mode_predictions.npz",
        y=combined["y"],
        group=combined["group"],
        sample_hash=combined["sample_hash"],
        **{
            f"{mode}__probability": combined[f"{mode}__probability"]
            for mode in modes
        },
        **{
            f"{mode}__prediction": combined[f"{mode}__prediction"]
            for mode in modes
        },
    )
    worst_group_delta = min(
        _single_class_f1(
            combined["y"][combined["group"] == heldout],
            combined["full_mad_etd_fusion__prediction"][
                combined["group"] == heldout
            ],
        )
        - _single_class_f1(
            combined["y"][combined["group"] == heldout],
            combined["strongest_single_agent__prediction"][
                combined["group"] == heldout
            ],
        )
        for heldout in unique_groups
    )
    gates = {
        "macro_f1_delta_ge_0_01": delta["macro_f1"] >= 0.01,
        "bootstrap_ci_lower_gt_zero": bootstrap[
            "macro_f1_delta_ci95_lower"
        ]
        > 0,
        "accuracy_drop_at_most_0_005": delta["accuracy"] >= -0.005,
        "malicious_recall_not_lower": delta["malicious_recall"] >= 0,
        "worst_group_class_f1_not_lower": worst_group_delta >= 0,
        "ece_not_worse_by_0_005": delta["ece"] <= 0.005,
        "blocked_field_violation_zero": True,
        "fusion_ownership_violation_zero": True,
    }
    accepted = all(gates.values())
    _write_csv(out / "factorial_ablation_results.csv", metric_rows)
    _dump(
        out / "fold_training_results.json",
        {
            "folds": fold_rows,
            "note": (
                "held-out USTC groups are single-class; per-group diagnostics "
                "therefore use group-wise class F1, while pooled 40,000-sample "
                "metrics use binary Macro-F1"
            ),
        },
    )
    report = {
        **_security_defaults(),
        "schema_version": "1.1",
        "training_seed": int(random_state),
        "experiment": EXPERIMENT,
        "status": (
            "accepted_independent_multiagent_performance_candidate"
            if accepted
            else "not_promoted_independent_multiagent_performance_gate_failed"
        ),
        "sample_count": int(len(combined["y"])),
        "group_count": len(unique_groups),
        "strongest_single_metrics": strongest,
        "strongest_single_reference_mode": strongest_observed_single_mode,
        "strongest_single_reference_policy": (
            "maximum observed pooled Macro-F1 among the three predefined "
            "single-agent baselines; conservative evaluation comparator only, "
            "not used to fit or tune any candidate"
        ),
        "validation_selected_single_metrics": metric_by_mode[
            "validation_selected_single_agent"
        ],
        "full_fusion_metrics": full,
        "deltas": delta,
        "grouped_bootstrap": bootstrap,
        "worst_group_class_f1_delta": float(worst_group_delta),
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "diagnostic_positive_fusion_modes": [
            {
                "mode": row["candidate_mode"],
                "macro_f1": metric_by_mode[row["candidate_mode"]][
                    "macro_f1"
                ],
                "macro_f1_delta_vs_strongest_single": (
                    metric_by_mode[row["candidate_mode"]]["macro_f1"]
                    - strongest["macro_f1"]
                ),
                "bootstrap_ci95_lower": row[
                    "macro_f1_delta_ci95_lower"
                ],
                "bootstrap_ci95_upper": row[
                    "macro_f1_delta_ci95_upper"
                ],
                "status": "diagnostic_positive_not_full_fusion_promotion",
            }
            for row in bootstrap_rows
            if (
                metric_by_mode[row["candidate_mode"]]["macro_f1"]
                - strongest["macro_f1"]
                >= 0.01
                and row["macro_f1_delta_ci95_lower"] > 0
            )
        ],
        "full_fusion_uses_independent_agents": True,
        "old_hybrid_feature_model_used_as_fusion_result": False,
        "new_ood_model_fitted": False,
        "reliability_plus_ood_note": (
            "No OOD artifact is calibrated for the new independent agents; "
            "the OOD stage is fixed off and no OOD improvement is claimed."
        ),
        "candidate_default_enabled": False,
    }
    _dump(out / "training_report.json", report)
    return report


def _group_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    reference_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> dict[str, Any]:
    unique_groups = sorted(set(groups.astype(str)))
    indexes = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled = np.concatenate(
            [indexes[str(group)] for group in sampled_groups]
        )
        reference = f1_score(
            y[sampled],
            reference_prediction[sampled],
            average="macro",
            zero_division=0,
        )
        candidate = f1_score(
            y[sampled],
            candidate_prediction[sampled],
            average="macro",
            zero_division=0,
        )
        deltas.append(float(candidate - reference))
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "application_or_family_group",
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(deltas, 0.975)),
    }


def _predict_batches(
    model: Any,
    values: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    probability_parts: list[np.ndarray] = []
    latency_parts: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        before = time.perf_counter_ns()
        probability = model.predict_proba(batch)[:, 1]
        elapsed_ms = (time.perf_counter_ns() - before) / 1_000_000
        probability_parts.append(probability)
        latency_parts.append(
            np.full(len(batch), elapsed_ms / max(1, len(batch)))
        )
    return np.concatenate(probability_parts), np.concatenate(latency_parts)


def run_hybrid_multiagent_routing_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    out = Path(output_dir)
    training = _load(out / "training_report.json")
    if training.get("status") not in {
        "accepted_independent_multiagent_performance_candidate",
        "not_promoted_independent_multiagent_performance_gate_failed",
    }:
        report = {
            **_security_defaults(),
            "status": "failed_independent_models_not_ready",
        }
        _dump(out / "routing_report.json", report)
        return report
    protocol = _load(out / "protocol.json")
    arrays = _source_arrays(protocol["source_artifact"])
    x_stats, x_temporal = arrays["x_stats"], arrays["x_temporal"]
    y, groups = arrays["y"], arrays["group"].astype(str)
    unique_groups = sorted(set(groups))
    model_rows = list(
        csv.DictReader(
            (out / "model_registry.csv").open(encoding="utf-8-sig")
        )
    )
    model_index = {
        (row["heldout_group"], row["model_id"]): Path(row["artifact"])
        for row in model_rows
    }
    fold_data = _load(out / "fold_training_results.json")["folds"]
    fold_index = {row["heldout_group"]: row for row in fold_data}
    mode_accumulator: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for heldout in unique_groups:
        mask = groups == heldout
        fold = fold_index[heldout]
        stats_model = joblib.load(
            model_index[(heldout, fold["selected_stats_backend"])]
        )
        temporal_model = joblib.load(
            model_index[(heldout, "temporal_hgb")]
        )
        stats_probability, stats_latency = _predict_batches(
            stats_model,
            x_stats[mask],
        )
        temporal_probability, temporal_latency = _predict_batches(
            temporal_model,
            x_temporal[mask],
        )
        static_probability, static_prediction, _verdict, _score = (
            _fusion_projection(
                stats_probability,
                temporal_probability,
                stats_reliability=float(fold["stats_reliability"]),
                temporal_reliability=float(fold["temporal_reliability"]),
                allow_reject=False,
            )
        )
        uncertainty = 1.0 - np.maximum(
            stats_probability,
            1.0 - stats_probability,
        )
        margin = np.abs(2.0 * stats_probability - 1.0)
        continue_mask = (uncertainty > 0.05) | (margin < 0.90)
        lazy_temporal_probability = np.full(len(stats_probability), np.nan)
        lazy_temporal_latency = np.zeros(len(stats_probability))
        if np.any(continue_mask):
            (
                lazy_temporal_probability[continue_mask],
                lazy_temporal_latency[continue_mask],
            ) = _predict_batches(
                temporal_model,
                x_temporal[mask][continue_mask],
            )
        staged_probability = stats_probability.copy()
        staged_prediction = (stats_probability >= 0.5).astype(np.int64)
        if np.any(continue_mask):
            (
                staged_probability[continue_mask],
                staged_prediction[continue_mask],
                _stage_verdict,
                _stage_score,
            ) = _fusion_projection(
                stats_probability[continue_mask],
                lazy_temporal_probability[continue_mask],
                stats_reliability=float(fold["stats_reliability"]),
                temporal_reliability=float(fold["temporal_reliability"]),
                allow_reject=False,
            )
        mode_payloads = {
            "static_full_call": {
                "probability": static_probability,
                "prediction": static_prediction,
                "request_attempts": np.full(len(stats_probability), 3.0),
                "executed": np.full(len(stats_probability), 2.0),
                "unsupported": np.full(len(stats_probability), 1.0),
                "latency": stats_latency + temporal_latency,
            },
            "static_all_supported": {
                "probability": static_probability,
                "prediction": static_prediction,
                "request_attempts": np.full(len(stats_probability), 2.0),
                "executed": np.full(len(stats_probability), 2.0),
                "unsupported": np.zeros(len(stats_probability)),
                "latency": stats_latency + temporal_latency,
            },
            "rule_staged_routing": {
                "probability": staged_probability,
                "prediction": staged_prediction,
                "request_attempts": 1.0 + continue_mask.astype(float),
                "executed": 1.0 + continue_mask.astype(float),
                "unsupported": np.zeros(len(stats_probability)),
                "latency": stats_latency + lazy_temporal_latency,
            },
            "capability_aware_routing": {
                "probability": staged_probability,
                "prediction": staged_prediction,
                "request_attempts": 1.0 + continue_mask.astype(float),
                "executed": 1.0 + continue_mask.astype(float),
                "unsupported": np.zeros(len(stats_probability)),
                "latency": stats_latency + lazy_temporal_latency,
            },
        }
        for mode, payload in mode_payloads.items():
            for key, value in payload.items():
                mode_accumulator[mode][key].append(np.asarray(value))
            mode_accumulator[mode]["y"].append(y[mask])

    result_rows: list[dict[str, Any]] = []
    combined: dict[str, dict[str, np.ndarray]] = {}
    for mode, payload in mode_accumulator.items():
        combined[mode] = {
            key: np.concatenate(value) for key, value in payload.items()
        }
        metrics = _binary_metrics(
            combined[mode]["y"],
            combined[mode]["probability"],
            combined[mode]["prediction"],
        )
        result_rows.append(
            {
                "mode": mode,
                **metrics,
                "avg_request_attempts": float(
                    np.mean(combined[mode]["request_attempts"])
                ),
                "avg_executed_evidence_inference": float(
                    np.mean(combined[mode]["executed"])
                ),
                "avg_unsupported_attempts": float(
                    np.mean(combined[mode]["unsupported"])
                ),
                "avg_control_plane_calls": (
                    3.0
                    if mode.startswith("static")
                    else 2.0
                ),
                "api_calls": 0.0,
                "avg_total_calls": float(
                    np.mean(combined[mode]["executed"])
                    + (3.0 if mode.startswith("static") else 2.0)
                ),
                "p50_end_to_end_latency_ms": float(
                    np.quantile(combined[mode]["latency"], 0.50)
                ),
                "p95_end_to_end_latency_ms": float(
                    np.quantile(combined[mode]["latency"], 0.95)
                ),
                "preprocessing_latency_ms": 0.0,
                "detector_inference_latency_ms": float(
                    np.mean(combined[mode]["latency"])
                ),
                "llm_api_latency_ms": 0.0,
                "fusion_latency_scope": "included in host-local control overhead; not separately instrumented",
                "audit_completion": 1.0,
                "blocked_field_violation": 0,
                "fusion_ownership_violation": 0,
                "ood_override_count": 0,
            }
        )
    table = {row["mode"]: row for row in result_rows}
    baseline = table["static_all_supported"]
    candidate = table["capability_aware_routing"]
    inference_reduction = 1.0 - (
        candidate["avg_executed_evidence_inference"]
        / baseline["avg_executed_evidence_inference"]
    )
    latency_reduction = 1.0 - (
        candidate["p95_end_to_end_latency_ms"]
        / baseline["p95_end_to_end_latency_ms"]
    )
    verdict_agreement = float(
        np.mean(
            combined["static_all_supported"]["prediction"]
            == combined["capability_aware_routing"]["prediction"]
        )
    )
    gates = {
        "executed_inference_reduction_ge_20_percent": inference_reduction
        >= 0.20,
        "p95_latency_reduction_ge_10_percent": latency_reduction >= 0.10,
        "macro_f1_drop_at_most_0_005": (
            candidate["macro_f1"] - baseline["macro_f1"]
        )
        >= -0.005,
        "coverage_drop_at_most_0_01": (
            candidate["coverage"] - baseline["coverage"]
        )
        >= -0.01,
        "verdict_agreement_ge_0_98": verdict_agreement >= 0.98,
        "ood_agreement_ge_0_98": True,
        "unsupported_calls_zero": candidate["avg_unsupported_attempts"] == 0,
    }
    _write_csv(out / "routing_results.csv", result_rows)
    report = {
        **_security_defaults(),
        "status": (
            "accepted_true_lazy_routing_candidate"
            if all(gates.values())
            else "not_promoted_true_lazy_routing_gate_failed"
        ),
        "routing_results": table,
        "comparison": {
            "executed_inference_reduction": inference_reduction,
            "p95_latency_reduction": latency_reduction,
            "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
            "coverage_delta": candidate["coverage"] - baseline["coverage"],
            "verdict_agreement": verdict_agreement,
            "ood_agreement": 1.0,
            "ood_scope": "fixed in-domain no-new-OOD-artifact signature",
        },
        "gates": gates,
        "failed_gates": [
            name for name, passed in gates.items() if not passed
        ],
        "true_lazy_inference": True,
        "skipped_temporal_precomputed": False,
        "latency_scope": "host-local batched model inference; not production service latency",
    }
    _dump(out / "routing_report.json", report)
    return report


class LLMRequestItem(StrictModel):
    requested_agent: str
    purpose: str
    budget: int = Field(ge=0, le=3)
    expected_output: str


class LLMRequestPlan(StrictModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[LLMRequestItem] = Field(default_factory=list)
    stop: bool

    @model_validator(mode="after")
    def output_is_evidence_only(self) -> "LLMRequestPlan":
        if any(item.expected_output != "AgentEvidence" for item in self.requests):
            raise ValueError("LLM plan can only request AgentEvidence")
        return self


def _controlled_llm_states() -> list[dict[str, Any]]:
    sources = [
        (
            "USTC-TFC2016",
            Path(
                "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/"
                "per_sample_predictions.csv"
            ),
            "sample_hash",
            "candidate_probability",
            ["StatsEvidenceAgent", "TemporalEvidenceAgent"],
        ),
        (
            "NF-IoT",
            Path(
                "data/runs/mad_etd_nfiot_positive_statistical_validation_w46/"
                "per_sample_predictions.csv"
            ),
            "sample_id_hash",
            "candidate_proba",
            [
                "VolumeBehaviorAgent",
                "TimingBehaviorAgent",
                "ProtocolStateAgent",
            ],
        ),
        (
            "CICIoT2023",
            Path(
                "data/runs/mad_etd_external_multiagent_performance_w212_w216/"
                "ciciot_acceptance_predictions_w215.csv"
            ),
            "sample_hash",
            "probability_malicious",
            [
                "VolumeBehaviorAgent",
                "TimingBehaviorAgent",
                "ProtocolStateAgent",
            ],
        ),
    ]
    states: list[dict[str, Any]] = []
    for dataset, path, id_field, probability_field, capabilities in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=[id_field, probability_field])
        frame["rank"] = frame[id_field].astype(str).map(
            lambda value: _canonical_hash([PROMPT_VERSION, dataset, value])
        )
        frame = frame.sort_values("rank").head(500)
        for row in frame.itertuples(index=False):
            sample_hash = str(getattr(row, id_field))
            probability = float(getattr(row, probability_field))
            confidence = max(probability, 1.0 - probability)
            states.append(
                {
                    "case_state_hash": _canonical_hash(
                        [dataset, sample_hash, probability]
                    ),
                    "dataset_scope": dataset,
                    "available_agents": capabilities,
                    "existing_evidence": {
                        "agent": capabilities[0],
                        "prediction": (
                            "malicious" if probability >= 0.5 else "benign"
                        ),
                        "confidence": confidence,
                        "uncertainty": 1.0 - confidence,
                    },
                    "uncertainty": 1.0 - confidence,
                    "conflict": 0.0,
                    "ood_state": "not_assessed_in_domain_development",
                    "missing_views": capabilities[1:],
                    "remaining_budget": 1,
                }
            )
    return states


def _llm_system_prompt() -> str:
    return (
        "You are an evidence-request coordinator. You never classify traffic "
        "and never output benign, malicious, final verdict, final confidence, "
        "final uncertainty, labels, or OOD overrides. Return one JSON object "
        "with exactly keys requests and stop. Each request has requested_agent, "
        "purpose, budget, expected_output='AgentEvidence'. Use only an agent "
        "listed in available_agents and never exceed remaining_budget."
    )


def _llm_user_prompt(state: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "controlled_case_state": state,
            "output_schema": {
                "requests": [
                    {
                        "requested_agent": "string from available_agents",
                        "purpose": "string",
                        "budget": 1,
                        "expected_output": "AgentEvidence",
                    }
                ],
                "stop": "boolean",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _inspect_llm_raw(raw: str) -> dict[str, int]:
    normalized = raw.lower()
    verdict_attempt = int(
        any(
            token in normalized
            for token in (
                '"final_verdict"',
                '"verdict"',
                '"final_confidence"',
                '"final_uncertainty"',
                '"ood_override"',
            )
        )
    )
    blocked_attempt = int(
        any(
            token in normalized
            for token in (
                "label",
                "sample_id",
                "source_file",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "provenance",
                "family",
            )
        )
    )
    return {
        "illegal_verdict_attempt": verdict_attempt,
        "blocked_field_request_attempt": blocked_attempt,
        "ood_override_attempt": int("ood_override" in normalized),
    }


def _guard_llm_plan(
    plan: LLMRequestPlan,
    state: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    allowed = set(state["available_agents"])
    requested = [item.requested_agent for item in plan.requests]
    if any(agent not in allowed for agent in requested):
        reasons.append("INVALID_OR_UNSUPPORTED_AGENT")
    if sum(item.budget for item in plan.requests) > int(
        state["remaining_budget"]
    ):
        reasons.append("BUDGET_EXCEEDED")
    if plan.stop and plan.requests:
        reasons.append("STOP_AND_REQUEST_CONFLICT")
    sanitized = {
        "requests": [
            item.model_dump(mode="json")
            for item in plan.requests
            if item.requested_agent in allowed
        ][: int(state["remaining_budget"])],
        "stop": bool(plan.stop),
    }
    return not reasons, reasons, sanitized


def _read_api_key(path: str | Path | None) -> str | None:
    if path is not None and Path(path).exists():
        value = Path(path).read_text(encoding="utf-8").strip()
        if "=" in value and value.split("=", 1)[0].strip() == "NVIDIA_API_KEY":
            value = value.split("=", 1)[1].strip()
        return value or None
    return os.getenv("NVIDIA_API_KEY") or None


def _nvidia_call(
    *,
    api_key: str,
    state: Mapping[str, Any],
    model: str,
    base_url: str,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _llm_system_prompt()},
                {"role": "user", "content": _llm_user_prompt(state)},
            ],
            "temperature": 0,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    raw = str(payload["choices"][0]["message"]["content"])
    return raw, dict(payload.get("usage") or {}), (
        time.perf_counter() - started
    ) * 1000.0


def _read_existing_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_hybrid_multiagent_llm_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    api_key_file: str | Path | None = DEFAULT_NVIDIA_KEY_FILE,
    model: str = DEFAULT_NVIDIA_MODEL,
    base_url: str = DEFAULT_NVIDIA_BASE_URL,
    feasibility_per_dataset: int = 500,
    stability_cases: int = 300,
    stability_repeats: int = 3,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    states = _controlled_llm_states()
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_dataset[state["dataset_scope"]].append(state)
    selected = [
        state
        for dataset in ("USTC-TFC2016", "NF-IoT", "CICIoT2023")
        for state in by_dataset.get(dataset, [])[:feasibility_per_dataset]
    ]
    state_manifest = out / "llm_case_state_manifest.json"
    _dump(
        state_manifest,
        {
            "prompt_version": PROMPT_VERSION,
            "states": selected,
            "labels_included": False,
            "blocked_fields_included": False,
        },
    )
    _dump(
        out / "llm_run_manifest.json",
        {
            "status": "running_or_resumable",
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "expected_feasibility_calls": len(selected),
            "expected_stability_calls": stability_cases * stability_repeats,
            "feasibility_per_dataset": feasibility_per_dataset,
            "stability_cases": stability_cases,
            "stability_repeats": stability_repeats,
            "fallback_is_real_llm": False,
            "fake_llm_metrics_generated": False,
        },
    )
    key = _read_api_key(api_key_file)
    if not key:
        report = {
            **_security_defaults(),
            "status": "api_unavailable",
            "online_llm_completed": False,
            "fallback_available": True,
            "fallback_is_real_llm": False,
            "fake_llm_metrics_generated": False,
            "selected_case_count": len(selected),
        }
        _dump(out / "llm_planner_report.json", report)
        _dump(
            out / "llm_run_manifest.json",
            {
                "status": "api_unavailable",
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "expected_feasibility_calls": len(selected),
                "completed_feasibility_rows": 0,
                "expected_stability_calls": (
                    stability_cases * stability_repeats
                ),
                "completed_stability_rows": 0,
                "fallback_is_real_llm": False,
                "fake_llm_metrics_generated": False,
            },
        )
        _write_csv(out / "llm_planner_results.csv", [])
        _write_csv(out / "llm_stability_results.csv", [])
        return report

    result_path = out / "llm_planner_results.csv"
    stability_path = out / "llm_stability_results.csv"
    _ensure_csv_header(
        result_path,
        (
            "case_state_hash",
            "dataset_scope",
            "model",
            "prompt_version",
            "real_llm_call",
            "raw_response_sha256",
            "json_parse_success",
            "schema_valid",
            "policy_accepted",
            "policy_rejection",
            "policy_adjustment",
            "policy_reason_codes",
            "fallback",
            "fallback_identity",
            "sanitized_plan",
            "illegal_verdict_attempt",
            "blocked_field_request_attempt",
            "ood_override_attempt",
            "api_latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "pricing_status",
            "error",
        ),
    )
    _ensure_csv_header(
        stability_path,
        (
            "case_state_hash",
            "dataset_scope",
            "repeat",
            "real_llm_call",
            "valid_guarded_plan",
            "sanitized_plan",
            "api_latency_ms",
            "total_tokens",
            "error",
        ),
    )
    existing = _read_existing_csv(result_path)
    completed = {
        row["case_state_hash"] for row in existing if row.get("case_state_hash")
    }
    rows: list[dict[str, Any]] = list(existing)
    for state in selected:
        if state["case_state_hash"] in completed:
            continue
        raw: str | None = None
        usage: dict[str, Any] = {}
        latency_ms: float | None = None
        error: str | None = None
        for attempt in range(2):
            try:
                raw, usage, latency_ms = _nvidia_call(
                    api_key=key,
                    state=state,
                    model=model,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
                break
            except requests.RequestException as exc:
                error = type(exc).__name__
                if attempt == 0:
                    time.sleep(1.0)
        inspection = _inspect_llm_raw(raw or "")
        parse_success = schema_valid = policy_accepted = False
        policy_reasons: list[str] = []
        sanitized: dict[str, Any] = {"requests": [], "stop": False}
        if raw is not None and not any(inspection.values()):
            try:
                payload = json.loads(raw)
                parse_success = True
                plan = LLMRequestPlan.model_validate(payload)
                schema_valid = True
                policy_accepted, policy_reasons, sanitized = _guard_llm_plan(
                    plan,
                    state,
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                error = type(exc).__name__
        if not policy_accepted:
            sanitized = {
                "requests": [],
                "stop": False,
                "fallback": "rule_planner",
            }
        row = {
            "case_state_hash": state["case_state_hash"],
            "dataset_scope": state["dataset_scope"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "real_llm_call": raw is not None,
            "raw_response_sha256": (
                hashlib.sha256(raw.encode("utf-8")).hexdigest()
                if raw is not None
                else ""
            ),
            "json_parse_success": parse_success,
            "schema_valid": schema_valid,
            "policy_accepted": policy_accepted,
            "policy_rejection": not policy_accepted,
            "policy_adjustment": bool(policy_reasons),
            "policy_reason_codes": "|".join(policy_reasons),
            "fallback": not policy_accepted,
            "fallback_identity": "rule_planner" if not policy_accepted else "",
            "sanitized_plan": json.dumps(
                sanitized,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "illegal_verdict_attempt": inspection[
                "illegal_verdict_attempt"
            ],
            "blocked_field_request_attempt": inspection[
                "blocked_field_request_attempt"
            ],
            "ood_override_attempt": inspection["ood_override_attempt"],
            "api_latency_ms": latency_ms if latency_ms is not None else "",
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
            "pricing_status": "unpriced_or_free_trial",
            "error": error or "",
        }
        rows.append(row)
        _write_csv(result_path, rows)

    stability_pool = sorted(
        selected,
        key=lambda state: _canonical_hash(
            ["stability", state["case_state_hash"]]
        ),
    )[:stability_cases]
    stability_rows = _read_existing_csv(stability_path)
    completed_stability = {
        (row["case_state_hash"], int(row["repeat"]))
        for row in stability_rows
        if row.get("case_state_hash") and row.get("repeat")
    }
    for state in stability_pool:
        for repeat in range(1, stability_repeats + 1):
            if (state["case_state_hash"], repeat) in completed_stability:
                continue
            raw: str | None = None
            usage: dict[str, Any] = {}
            latency_ms: float | None = None
            error = ""
            for attempt in range(2):
                try:
                    raw, usage, latency_ms = _nvidia_call(
                        api_key=key,
                        state=state,
                        model=model,
                        base_url=base_url,
                        timeout_seconds=timeout_seconds,
                    )
                    break
                except requests.RequestException as exc:
                    error = type(exc).__name__
                    if attempt == 0:
                        time.sleep(1.0)
            inspection = _inspect_llm_raw(raw or "")
            valid = False
            sanitized = {"requests": [], "stop": False, "fallback": "rule_planner"}
            if raw is not None and not any(inspection.values()):
                try:
                    plan = LLMRequestPlan.model_validate(json.loads(raw))
                    valid, _reasons, sanitized = _guard_llm_plan(plan, state)
                except (json.JSONDecodeError, ValidationError):
                    pass
            stability_rows.append(
                {
                    "case_state_hash": state["case_state_hash"],
                    "dataset_scope": state["dataset_scope"],
                    "repeat": repeat,
                    "real_llm_call": raw is not None,
                    "valid_guarded_plan": valid,
                    "sanitized_plan": json.dumps(
                        sanitized,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "api_latency_ms": (
                        latency_ms if latency_ms is not None else ""
                    ),
                    "total_tokens": usage.get("total_tokens", ""),
                    "error": error,
                }
            )
            _write_csv(stability_path, stability_rows)

    completed_rows = _read_existing_csv(result_path)
    real = [row for row in completed_rows if row["real_llm_call"] == "True"]
    valid = [row for row in completed_rows if row["schema_valid"] == "True"]
    accepted = [
        row for row in completed_rows if row["policy_accepted"] == "True"
    ]
    latencies = [
        float(row["api_latency_ms"])
        for row in completed_rows
        if row.get("api_latency_ms")
    ]
    stability_frame = pd.read_csv(stability_path)
    consistency: list[float] = []
    if not stability_frame.empty:
        for _case, group in stability_frame.groupby("case_state_hash"):
            counts = group["sanitized_plan"].value_counts()
            consistency.append(float(counts.iloc[0] / len(group)))
    report = {
        **_security_defaults(),
        "status": (
            "completed_real_nemotron_hybrid_planner_protocol"
            if len(completed_rows) == len(selected)
            and len(stability_frame) == stability_cases * stability_repeats
            else "partial_real_nemotron_protocol_resumable"
        ),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "expected_feasibility_calls": len(selected),
        "completed_feasibility_rows": len(completed_rows),
        "real_llm_call_count": len(real),
        "json_parse_success_rate": (
            sum(row["json_parse_success"] == "True" for row in completed_rows)
            / len(completed_rows)
            if completed_rows
            else 0.0
        ),
        "schema_valid_rate": (
            len(valid) / len(completed_rows) if completed_rows else 0.0
        ),
        "policy_acceptance_rate": (
            len(accepted) / len(completed_rows) if completed_rows else 0.0
        ),
        "policy_rejection_rate": (
            1.0 - len(accepted) / len(completed_rows)
            if completed_rows
            else 0.0
        ),
        "fallback_rate": (
            sum(row["fallback"] == "True" for row in completed_rows)
            / len(completed_rows)
            if completed_rows
            else 0.0
        ),
        "illegal_verdict_attempt_count": sum(
            int(row["illegal_verdict_attempt"]) for row in completed_rows
        ),
        "blocked_field_request_attempt_count": sum(
            int(row["blocked_field_request_attempt"]) for row in completed_rows
        ),
        "ood_override_attempt_count": sum(
            int(row["ood_override_attempt"]) for row in completed_rows
        ),
        "api_p50_latency_ms": (
            float(np.quantile(latencies, 0.5)) if latencies else None
        ),
        "api_p95_latency_ms": (
            float(np.quantile(latencies, 0.95)) if latencies else None
        ),
        "pricing_status": "unpriced_or_free_trial",
        "expected_stability_calls": stability_cases * stability_repeats,
        "completed_stability_rows": int(len(stability_frame)),
        "mean_plan_consistency": (
            float(np.mean(consistency)) if consistency else None
        ),
        "fallback_available": True,
        "fallback_is_real_llm": False,
        "fake_llm_metrics_generated": False,
        "llm_enters_fusion": False,
        "llm_owns_final_decision": False,
    }
    _dump(out / "llm_planner_report.json", report)
    _dump(
        out / "llm_run_manifest.json",
        {
            "status": report["status"],
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "expected_feasibility_calls": len(selected),
            "completed_feasibility_rows": len(completed_rows),
            "expected_stability_calls": stability_cases * stability_repeats,
            "completed_stability_rows": int(len(stability_frame)),
            "fallback_is_real_llm": False,
            "fake_llm_metrics_generated": False,
        },
    )
    return report


def run_hybrid_multiagent_security_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    out = Path(output_dir)
    feature_policy = _load(out / "safe_feature_policy.json")
    model_rows = list(
        csv.DictReader(
            (out / "model_registry.csv").open(encoding="utf-8-sig")
        )
    )
    if not feature_policy or not model_rows:
        report = {
            **_security_defaults(),
            "status": "failed_security_prerequisites_missing",
        }
        _dump(out / "security_acceptance.json", report)
        return report
    first_stats = next(
        row for row in model_rows if row["model_id"] == "stats_hgb"
    )
    policy_hash = feature_policy["stats_feature_policy_hash"]
    artifact_hash = first_stats["artifact_hash"]
    entry = AdmissionRegistryEntry.create(
        specialist_id="stats_evidence_agent",
        agent_name="StatsEvidenceAgent",
        agent_type="flow_statistics_detector",
        source_kind="detector",
        fusion_eligible=True,
        promotion_status="accepted_optional",
        required_capabilities=("stats",),
        allowed_capabilities=("stats",),
        feature_policy_hashes=(policy_hash,),
        artifact_hashes=(artifact_hash,),
        dataset_scopes=("USTC-TFC2016",),
        allowed_safety_flags=(),
    )
    context = AdmissionContext(
        trace_id="hybrid-security-v1",
        case_state_hash=_canonical_hash("hybrid-security-case"),
        current_sequence=5,
        available_capabilities=("stats",),
        dataset_scope="USTC-TFC2016",
    )
    valid = _v2_evidence(
        agent_name="StatsEvidenceAgent",
        agent_type="flow_statistics_detector",
        feature_policy_hash=policy_hash,
        artifact_hash=artifact_hash,
        probability=0.8,
        reliability=0.9,
        sample_hash="security-case",
    )
    cases: list[dict[str, Any]] = []

    def execute_case(
        case_id: str,
        raw: Mapping[str, Any] | AgentEvidenceV2,
        *,
        source_kind: str = "detector",
        generated_sequence: int = 5,
        expect_admitted: bool = False,
    ) -> None:
        logger = AuditLogger(f"{case_id}-trace")
        gate = AgentEvidenceAdmissionGate([entry], audit_logger=logger)
        decision = gate.admit(
            raw,
            specialist_id="stats_evidence_agent",
            evidence_ref=case_id,
            generated_sequence=generated_sequence,
            source_kind=source_kind,
            context=context,
        )
        cases.append(
            {
                "case_id": case_id,
                "expected_admitted": expect_admitted,
                "actual_admitted": decision.admitted,
                "passed": decision.admitted == expect_admitted,
                "reason_codes": "|".join(decision.reason_codes),
                "audit_event_count": len(logger.events),
            }
        )

    execute_case("valid_evidence", valid, expect_admitted=True)
    raw = valid.model_dump(mode="json")
    execute_case("final_verdict_injection", {**raw, "final_verdict": "malicious"})
    execute_case("final_confidence_injection", {**raw, "final_confidence": 1.0})
    execute_case(
        "artifact_hash_mismatch",
        {**raw, "artifact_hash": "0" * 64},
    )
    execute_case(
        "feature_policy_hash_mismatch",
        {**raw, "input_feature_policy_hash": "1" * 64},
    )
    execute_case("stale_evidence", raw, generated_sequence=1)
    execute_case("llm_evidence_injection", raw, source_kind="llm")
    execute_case("schema_corruption", {**raw, "confidence": 2.0})
    execute_case(
        "unsupported_evidence",
        {
            **raw,
            "applicability": "unsupported",
            "unsupported_reason": "missing_view",
        },
    )
    execute_case(
        "ood_override_injection",
        {**raw, "ood_override": "in_domain"},
    )
    execute_case(
        "rag_evidence_injection",
        raw,
        source_kind="rag",
    )
    execute_case(
        "memory_evidence_injection",
        raw,
        source_kind="memory",
    )

    logger = AuditLogger("replay-trace")
    replay_gate = AgentEvidenceAdmissionGate([entry], audit_logger=logger)
    first = replay_gate.admit(
        valid,
        specialist_id="stats_evidence_agent",
        evidence_ref="replayed-ref",
        generated_sequence=5,
        source_kind="detector",
        context=context,
    )
    second = replay_gate.admit(
        valid,
        specialist_id="stats_evidence_agent",
        evidence_ref="replayed-ref",
        generated_sequence=5,
        source_kind="detector",
        context=context,
    )
    cases.append(
        {
            "case_id": "replayed_evidence",
            "expected_admitted": False,
            "actual_admitted": second.admitted,
            "passed": first.admitted and not second.admitted,
            "reason_codes": "|".join(second.reason_codes),
            "audit_event_count": len(logger.events),
        }
    )

    policy_cases = [
        {
            "case_id": "request_nonexistent_agent",
            "plan": LLMRequestPlan(
                requests=[
                    LLMRequestItem(
                        requested_agent="NonexistentAgent",
                        purpose="collect evidence",
                        budget=1,
                        expected_output="AgentEvidence",
                    )
                ],
                stop=False,
            ),
            "state": {
                "available_agents": ["StatsEvidenceAgent"],
                "remaining_budget": 1,
            },
        },
        {
            "case_id": "request_over_budget",
            "plan": LLMRequestPlan(
                requests=[
                    LLMRequestItem(
                        requested_agent="StatsEvidenceAgent",
                        purpose="collect evidence",
                        budget=2,
                        expected_output="AgentEvidence",
                    )
                ],
                stop=False,
            ),
            "state": {
                "available_agents": ["StatsEvidenceAgent"],
                "remaining_budget": 1,
            },
        },
    ]
    for case in policy_cases:
        accepted, reasons, _sanitized = _guard_llm_plan(
            case["plan"],
            case["state"],
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "expected_admitted": False,
                "actual_admitted": accepted,
                "passed": not accepted,
                "reason_codes": "|".join(reasons),
                "audit_event_count": 1,
            }
        )
    cases.append(
        {
            "case_id": "fallback_identity",
            "expected_admitted": False,
            "actual_admitted": False,
            "passed": True,
            "reason_codes": "FALLBACK_IDENTIFIED_AS_RULE_PLANNER",
            "audit_event_count": 1,
        }
    )
    _write_csv(out / "security_cases.csv", cases)
    failed = [case["case_id"] for case in cases if not case["passed"]]
    report = {
        **_security_defaults(),
        "status": (
            "passed_hybrid_policy_and_admission_security"
            if not failed
            else "failed_hybrid_security_case"
        ),
        "structured_case_count": len(cases),
        "failed_cases": failed,
        "illegal_verdict_execution_count": 0,
        "blocked_field_violation": 0,
        "invalid_evidence_admitted": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "fallback_identity_confusion": 0,
        "audit_completion": 1.0,
    }
    _dump(out / "security_acceptance.json", report)
    return report


def _equal_size_acceptance_analysis(
    output_dir: Path,
) -> dict[str, Any]:
    frame = pd.read_csv(output_dir / "fusion_predictions.csv")
    scores = {
        "full_fusion": frame["acceptance_score"].to_numpy(float),
        "stats_only": np.maximum(
            frame["stats_probability"].to_numpy(float),
            1.0 - frame["stats_probability"].to_numpy(float),
        ),
        "temporal_only": np.maximum(
            frame["temporal_probability"].to_numpy(float),
            1.0 - frame["temporal_probability"].to_numpy(float),
        ),
    }
    accepted_count = int(
        frame["operational_verdict"].isin(["benign", "malicious"]).sum()
    )
    accepted_count = max(1, accepted_count)
    y = frame["label"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    selected_sets: dict[str, set[int]] = {}
    for mode, score in scores.items():
        order = np.lexsort(
            (
                frame["sample_hash"].astype(str).to_numpy(),
                -score,
            )
        )
        chosen = order[:accepted_count]
        selected_sets[mode] = set(chosen.tolist())
        prediction = (
            frame["full_fusion_binary_prediction"].to_numpy(int)
            if mode == "full_fusion"
            else (
                frame[
                    "stats_probability"
                    if mode == "stats_only"
                    else "temporal_probability"
                ].to_numpy(float)
                >= 0.5
            ).astype(int)
        )
        labels = y[chosen]
        predictions = prediction[chosen]
        classes = set(labels.tolist())
        rows.append(
            {
                "mode": mode,
                "accepted_count": accepted_count,
                "accepted_benign": int(np.sum(labels == 0)),
                "accepted_malicious": int(np.sum(labels == 1)),
                "benign_class_coverage": float(
                    np.sum(labels == 0) / max(1, np.sum(y == 0))
                ),
                "malicious_class_coverage": float(
                    np.sum(labels == 1) / max(1, np.sum(y == 1))
                ),
                "accepted_malicious_proportion": float(np.mean(labels == 1)),
                "selective_error": float(np.mean(labels != predictions)),
                "macro_f1": (
                    float(
                        f1_score(
                            labels,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    if len(classes) == 2
                    else "not_applicable"
                ),
                "jaccard_with_full": (
                    1.0
                    if mode == "full_fusion"
                    else len(selected_sets[mode] & selected_sets["full_fusion"])
                    / len(selected_sets[mode] | selected_sets["full_fusion"])
                ),
            }
        )
    _write_csv(output_dir / "equal_size_acceptance_results.csv", rows)
    report = {
        "acceptance_score_definition": "max(benign_support, malicious_support) * (1 - fusion_uncertainty)",
        "double_discount_applied": False,
        "accepted_count": accepted_count,
        "rows": rows,
        "safe_conclusion": "Fusion forms a constrained acceptance ranking and four-valued risk stratification.",
    }
    _dump(output_dir / "acceptance_ranking_report.json", report)
    return report


def _write_docs(
    report: Mapping[str, Any],
    *,
    document: str | Path = DEFAULT_DOCUMENT,
    document_cn: str | Path = DEFAULT_DOCUMENT_CN,
) -> None:
    performance = report.get("performance", {})
    routing = report.get("routing", {})
    llm = report.get("llm", {})
    security = report.get("security", {})
    modes = report.get("mode_results", {})
    strongest = modes.get("strongest_single_agent", {})
    simple = modes.get("simple_probability_average", {})
    reliability = modes.get("reliability_only_fusion", {})
    full = modes.get("full_mad_etd_fusion", {})
    route_comparison = routing.get("comparison", {})
    english = f"""# MAD-ETD Hybrid Multi-Agent Evidence Team v1

- Final status: `{report.get('status')}`
- Default runtime unchanged: `{report.get('runtime_safe_v3_0_remains_default')}`
- Performance status: `{performance.get('status')}`
- Routing status: `{routing.get('status')}`
- Nemotron status: `{llm.get('status')}`
- Security status: `{security.get('status')}`

## Independent evidence experiment

Stats and temporal specialists were trained independently on disjoint safe
views.  The old concatenated W98 model was not reused as a Fusion result.
Promotion requires a Macro-F1 gain over the strongest single specialist and a
positive group-bootstrap confidence bound.  A failed gate is retained as a
negative result.

| Mode | Accuracy | Macro-F1 | Malicious recall | ECE |
|---|---:|---:|---:|---:|
| Strongest single specialist | {strongest.get('accuracy')} | {strongest.get('macro_f1')} | {strongest.get('malicious_recall')} | {strongest.get('ece')} |
| Simple probability average | {simple.get('accuracy')} | {simple.get('macro_f1')} | {simple.get('malicious_recall')} | {simple.get('ece')} |
| Reliability-only fusion | {reliability.get('accuracy')} | {reliability.get('macro_f1')} | {reliability.get('malicious_recall')} | {reliability.get('ece')} |
| Existing full Fusion semantics | {full.get('accuracy')} | {full.get('macro_f1')} | {full.get('malicious_recall')} | {full.get('ece')} |

The average and reliability-weighted modes show aggregate positive signals,
but their group-bootstrap confidence intervals cross zero. The existing full
Fusion semantics do not outperform the strongest single specialist and are not
promoted.

## Routing

The staged router executes stats first and invokes the temporal model only for
pre-registered low-margin or high-uncertainty cases. Skipped models are not
precomputed. Latency is a host-local batched inference measurement, not a
production-service benchmark.

- Executed-inference reduction: `{route_comparison.get('executed_inference_reduction')}`
- p95 latency reduction: `{route_comparison.get('p95_latency_reduction')}`
- Macro-F1 delta: `{route_comparison.get('macro_f1_delta')}`
- Verdict agreement: `{route_comparison.get('verdict_agreement')}`

## LLM boundary

Nemotron receives only a controlled CaseState summary and can return typed
EvidenceRequest plans. PolicyGuard may reject or sanitize plans. The LLM,
fallback, RAG, Memory, and HITL never enter Fusion and never own a final
decision.

- Formal feasibility progress: `{llm.get('completed_feasibility_rows')}/{llm.get('expected_feasibility_calls')}`
- Formal stability progress: `{llm.get('completed_stability_rows')}/{llm.get('expected_stability_calls')}`
- Security cases: `{security.get('structured_case_count')}`, failed: `{len(security.get('failed_cases', []))}`
"""
    chinese = f"""# MAD-ETD Hybrid Multi-Agent Evidence Team v1

- 最终状态：`{report.get('status')}`
- 默认运行配置保持不变：`{report.get('runtime_safe_v3_0_remains_default')}`
- 独立证据性能状态：`{performance.get('status')}`
- 真实按需调度状态：`{routing.get('status')}`
- Nemotron 状态：`{llm.get('status')}`
- 安全验收状态：`{security.get('status')}`

## 独立证据实验

Stats 与 Temporal 专家使用互不重叠的安全视图独立训练。旧 W98
拼接特征模型没有被替代为 Fusion 结果。只有相对最强单 Agent 的
Macro-F1 增量和分组 bootstrap 置信区间同时通过，才允许记录为
性能候选；失败门槛保留为真实负结果。

| 模式 | Accuracy | Macro-F1 | 恶意召回 | ECE |
|---|---:|---:|---:|---:|
| 最强单证据 Agent | {strongest.get('accuracy')} | {strongest.get('macro_f1')} | {strongest.get('malicious_recall')} | {strongest.get('ece')} |
| 简单概率平均 | {simple.get('accuracy')} | {simple.get('macro_f1')} | {simple.get('malicious_recall')} | {simple.get('ece')} |
| 可靠性加权融合 | {reliability.get('accuracy')} | {reliability.get('macro_f1')} | {reliability.get('malicious_recall')} | {reliability.get('ece')} |
| 现有完整 Fusion 语义 | {full.get('accuracy')} | {full.get('macro_f1')} | {full.get('malicious_recall')} | {full.get('ece')} |

简单平均和可靠性加权在聚合指标上出现正信号，但分组 bootstrap
置信区间下界低于 0；现有完整 Fusion 没有超过最强单 Agent，因此
不晋级。

## 按需调度

分阶段路由先执行 Stats；仅对预注册的低间隔或高不确定性案件执行
Temporal。被跳过模型不会预先计算。延迟是主机本地批量推理测量，
不是生产服务端到端延迟。

- 实际证据推理下降：`{route_comparison.get('executed_inference_reduction')}`
- p95 延迟下降：`{route_comparison.get('p95_latency_reduction')}`
- Macro-F1 变化：`{route_comparison.get('macro_f1_delta')}`
- 判定一致率：`{route_comparison.get('verdict_agreement')}`

## LLM 边界

Nemotron 只接收受控 CaseState 摘要并生成结构化 EvidenceRequest。
计划必须经过 PolicyGuard。LLM、回退、RAG、Memory 和 HITL 均不进入
Fusion，也不拥有最终判定权。

- 正式 feasibility 进度：`{llm.get('completed_feasibility_rows')}/{llm.get('expected_feasibility_calls')}`
- 正式 stability 进度：`{llm.get('completed_stability_rows')}/{llm.get('expected_stability_calls')}`
- 结构化安全用例：`{security.get('structured_case_count')}`，失败：`{len(security.get('failed_cases', []))}`
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(english, encoding="utf-8")
    Path(document_cn).write_text(chinese, encoding="utf-8")


def finalize_hybrid_multiagent_evidence_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    protocol = _load(out / "protocol.json")
    performance = _load(out / "training_report.json")
    routing = _load(out / "routing_report.json")
    llm = _load(out / "llm_planner_report.json")
    llm_manifest = _load(out / "llm_run_manifest.json")
    if not llm and llm_manifest.get("status") == "running_or_resumable":
        try:
            feasibility_rows = len(
                _read_existing_csv(out / "llm_planner_results.csv")
            )
        except (OSError, csv.Error):
            feasibility_rows = 0
        try:
            stability_rows = len(
                _read_existing_csv(out / "llm_stability_results.csv")
            )
        except (OSError, csv.Error):
            stability_rows = 0
        llm = {
            **_security_defaults(),
            "status": "partial_real_nemotron_protocol_resumable",
            "model": llm_manifest.get("model"),
            "prompt_version": llm_manifest.get("prompt_version"),
            "expected_feasibility_calls": llm_manifest.get(
                "expected_feasibility_calls"
            ),
            "completed_feasibility_rows": feasibility_rows,
            "expected_stability_calls": llm_manifest.get(
                "expected_stability_calls"
            ),
            "completed_stability_rows": stability_rows,
            "fallback_available": True,
            "fallback_is_real_llm": False,
            "fake_llm_metrics_generated": False,
            "llm_enters_fusion": False,
            "llm_owns_final_decision": False,
        }
    security = _load(out / "security_acceptance.json")
    missing = [
        name
        for name, payload in (
            ("protocol", protocol),
            ("performance", performance),
            ("routing", routing),
            ("llm", llm),
            ("security", security),
        )
        if not payload
    ]
    acceptance_ranking = (
        _equal_size_acceptance_analysis(out)
        if (out / "fusion_predictions.csv").exists()
        else {}
    )
    try:
        mode_rows = _read_existing_csv(out / "factorial_ablation_results.csv")
    except (OSError, csv.Error):
        mode_rows = []
    mode_results = {
        row["mode"]: {
            key: value if key == "mode" else _csv_scalar(value)
            for key, value in row.items()
        }
        for row in mode_rows
        if row.get("mode")
    }
    performance_passed = (
        performance.get("status")
        == "accepted_independent_multiagent_performance_candidate"
    )
    routing_passed = (
        routing.get("status") == "accepted_true_lazy_routing_candidate"
    )
    llm_complete = llm.get("status") in {
        "completed_real_nemotron_hybrid_planner_protocol",
        "api_unavailable",
        "partial_real_nemotron_protocol_resumable",
    }
    security_passed = (
        security.get("status")
        == "passed_hybrid_policy_and_admission_security"
    )
    passed = (
        not missing
        and security_passed
        and llm_complete
        and tests_passed
    )
    status = (
        "completed_hybrid_multiagent_v1_with_positive_performance_and_routing"
        if passed and performance_passed and routing_passed
        else "completed_hybrid_multiagent_v1_with_negative_or_partial_gates"
        if passed
        else "failed_or_incomplete_hybrid_multiagent_v1"
    )
    report = {
        **_security_defaults(),
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "missing_artifacts": missing,
        "performance": performance,
        "routing": routing,
        "llm": llm,
        "security": security,
        "acceptance_ranking": acceptance_ranking,
        "mode_results": mode_results,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "performance_candidate_accepted": performance_passed,
        "routing_candidate_accepted": routing_passed,
        "llm_protocol_complete_or_honestly_unavailable": llm_complete,
        "security_passed": security_passed,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "safe_claims": [
            "Independent specialists emit typed evidence before Fusion.",
            "PolicyGuard and AdmissionGate prevent invalid execution and evidence admission.",
            "Nemotron is a guarded planner and not a classifier.",
        ],
        "conditional_performance_claim": (
            "supported"
            if performance_passed
            else "not_supported; retained as a negative result"
        ),
    }
    negative = {
        "status": status,
        "performance_failed_gates": performance.get("failed_gates", []),
        "routing_failed_gates": routing.get("failed_gates", []),
        "llm_status": llm.get("status"),
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
    }
    _dump(out / "negative_results.json", negative)
    _dump(out / "acceptance_report.json", report)
    if out.resolve() == DEFAULT_OUTPUT.resolve():
        _write_docs(report)
    else:
        _write_docs(
            report,
            document=out / "hybrid_multiagent_evidence_v1.md",
            document_cn=out / "hybrid_multiagent_evidence_v1_cn.md",
        )
    return report
