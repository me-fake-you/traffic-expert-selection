from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .base import CaseState
from .schemas import EvidenceUtilityEstimate


UTILITY_FEATURE_NAMES = (
    "remaining_budget",
    "round_no",
    "evidence_count",
    "called_stats",
    "called_temporal",
    "called_tls",
    "uncertainty",
    "conflict",
    "distribution_shift",
    "stats_reliability",
    "sequence_reliability",
    "tls_reliability",
    "input_completeness",
    "sequence_missing",
    "tls_missing",
    "candidate_temporal",
    "candidate_tls",
)


def utility_features(state: CaseState, agent: str) -> np.ndarray:
    """Runtime-only features. Labels and provenance are intentionally absent."""

    fusion = state.interim_fusion
    reliability = state.reliability
    flow = state.safe_flow or state.flow
    values = [
        float(state.remaining_budget),
        float(state.round_no),
        float(len(state.evidence)),
        float("StatsDetectorAgent" in state.called_agents),
        float("TemporalBehaviorAgent" in state.called_agents),
        float("TLSProtocolAgent" in state.called_agents),
        float(fusion.uncertainty if fusion else 1.0),
        float(fusion.conflict_score if fusion else 0.0),
        float(fusion.distribution_shift_score if fusion else 0.0),
        float(reliability.stats_reliability if reliability else 0.0),
        float(reliability.sequence_reliability if reliability else 0.0),
        float(reliability.tls_reliability if reliability else 0.0),
        float(reliability.input_completeness if reliability else 0.0),
        float(not bool(flow.sequence.packet_lengths)),
        float(not bool(flow.tls)),
        float(agent == "TemporalBehaviorAgent"),
        float(agent == "TLSProtocolAgent"),
    ]
    return np.asarray(values, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceUtilityPolicyV2:
    name = "EvidenceUtilityPolicyV2"
    version = "2.0"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        median_path = root / "median.joblib"
        lower_path = root / "lower.joblib"
        if not all(path.exists() for path in (metadata_path, median_path, lower_path)):
            raise ValueError(f"incomplete evidence utility artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(self.metadata.get("feature_names", [])) != UTILITY_FEATURE_NAMES:
            raise ValueError("evidence utility feature schema mismatch")
        if _sha256(median_path) != self.metadata.get("median_sha256"):
            raise ValueError("evidence utility median checksum mismatch")
        if _sha256(lower_path) != self.metadata.get("lower_sha256"):
            raise ValueError("evidence utility lower checksum mismatch")
        self.median = joblib.load(median_path)
        self.lower = joblib.load(lower_path)

    def estimate(self, state: CaseState, agent: str) -> EvidenceUtilityEstimate:
        features = utility_features(state, agent).reshape(1, -1)
        median = float(self.median.predict(features)[0])
        lower = float(self.lower.predict(features)[0])
        return EvidenceUtilityEstimate(
            agent=agent,
            median_utility=median,
            lower_utility=lower,
            expected_uncertainty_reduction=max(0.0, median),
            estimated_cost=1.0,
            should_dispatch=lower > 0,
            feature_names=list(UTILITY_FEATURE_NAMES),
        )


class EvidenceUtilityPolicyV21:
    name = "EvidenceUtilityPolicyV21"
    version = "2.1"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        median_path = root / "median.joblib"
        lower_path = root / "lower.joblib"
        if not all(
            path.exists() for path in (metadata_path, median_path, lower_path)
        ):
            raise ValueError(f"incomplete evidence utility v2.1 artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("policy") != "evidence_utility_v2_1":
            raise ValueError("evidence utility v2.1 policy metadata mismatch")
        if tuple(self.metadata.get("feature_names", [])) != UTILITY_FEATURE_NAMES:
            raise ValueError("evidence utility v2.1 feature schema mismatch")
        if _sha256(median_path) != self.metadata.get("median_sha256"):
            raise ValueError("evidence utility v2.1 median checksum mismatch")
        if _sha256(lower_path) != self.metadata.get("lower_sha256"):
            raise ValueError("evidence utility v2.1 lower checksum mismatch")
        self.median = joblib.load(median_path)
        self.lower = joblib.load(lower_path)

    def estimate(self, state: CaseState, agent: str) -> EvidenceUtilityEstimate:
        flow = state.safe_flow or state.flow
        unavailable = (
            agent == "TemporalBehaviorAgent"
            and len(flow.sequence.packet_lengths) < 4
        ) or (
            agent == "TLSProtocolAgent"
            and not (
                flow.tls.get("record_lengths")
                or flow.tls.get("tls_record_lengths")
            )
        )
        if unavailable:
            return EvidenceUtilityEstimate(
                agent=agent,
                median_utility=-1.0,
                lower_utility=-1.0,
                expected_uncertainty_reduction=0.0,
                estimated_cost=0.0,
                should_dispatch=False,
                feature_names=list(UTILITY_FEATURE_NAMES),
                policy_version="evidence-utility-v2.1",
            )
        features = utility_features(state, agent).reshape(1, -1)
        median = float(self.median.predict(features)[0])
        lower = float(self.lower.predict(features)[0])
        return EvidenceUtilityEstimate(
            agent=agent,
            median_utility=median,
            lower_utility=lower,
            expected_uncertainty_reduction=max(0.0, median),
            estimated_cost=1.0,
            should_dispatch=lower > 0,
            feature_names=list(UTILITY_FEATURE_NAMES),
            policy_version="evidence-utility-v2.1",
        )


def fit_evidence_utility_policy(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    materialized = list(rows)
    if len(materialized) < 100:
        raise ValueError("utility policy training requires at least 100 replay rows")
    x = np.asarray(
        [[float(row[name]) for name in UTILITY_FEATURE_NAMES] for row in materialized],
        dtype=np.float32,
    )
    y = np.asarray(
        [float(row["observed_utility"]) for row in materialized],
        dtype=np.float32,
    )
    median = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.5,
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x, y)
    lower = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.1,
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x, y)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    median_path = root / "median.joblib"
    lower_path = root / "lower.joblib"
    joblib.dump(median, median_path)
    joblib.dump(lower, lower_path)
    metadata = {
        "schema_version": "1.0",
        "policy": "evidence_utility_v2",
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "training_row_count": len(materialized),
        "seed": seed,
        "median_quantile": 0.5,
        "lower_quantile": 0.1,
        "dispatch_condition": "lower_utility > 0",
        "labels_available_only_for_offline_target_construction": True,
        "runtime_label_features": False,
        "automatic_retraining": False,
        "median_sha256": _sha256(median_path),
        "lower_sha256": _sha256(lower_path),
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def fit_evidence_utility_policy_v21(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    materialized = list(rows)
    if len(materialized) < 100:
        raise ValueError(
            "utility policy v2.1 training requires at least 100 replay rows"
        )
    x = np.asarray(
        [
            [float(row[name]) for name in UTILITY_FEATURE_NAMES]
            for row in materialized
        ],
        dtype=np.float32,
    )
    y = np.asarray(
        [float(row["observed_utility"]) for row in materialized],
        dtype=np.float32,
    )
    median = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.5,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x, y)
    lower = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.25,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x, y)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    median_path = root / "median.joblib"
    lower_path = root / "lower.joblib"
    joblib.dump(median, median_path)
    joblib.dump(lower, lower_path)
    metadata = {
        "schema_version": "1.0",
        "policy": "evidence_utility_v2_1",
        "feature_names": list(UTILITY_FEATURE_NAMES),
        "training_row_count": len(materialized),
        "seed": seed,
        "median_quantile": 0.5,
        "lower_quantile": 0.25,
        "dispatch_condition": "view_available and lower_utility > 0",
        "labels_available_only_for_offline_target_construction": True,
        "runtime_label_features": False,
        "runtime_dataset_identity_features": False,
        "runtime_provenance_features": False,
        "automatic_retraining": False,
        "median_sha256": _sha256(median_path),
        "lower_sha256": _sha256(lower_path),
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata
