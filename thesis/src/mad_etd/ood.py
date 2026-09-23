from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from .features import FEATURE_SCHEMA_VERSION, feature_extractor, feature_names
from .models import ModelArtifactError
from .schemas import AgentEvidence, DetectorInput
from .training import FeatureMatrix, _sha256_file, _validation_partition, load_feature_matrix


OOD_ARTIFACT_SCHEMA_VERSION = "1.0"
OOD_POLICIES = ("off", "soft", "hybrid")
OODPolicy = Literal["off", "soft", "hybrid"]


@dataclass(slots=True)
class OODAssessment:
    raw_score: float
    shift_score: float
    level: Literal["in_domain", "warning", "hard"]
    activation: float
    model_reliability: float


def ood_adjustment(shift_score: float) -> tuple[float, float]:
    activation = max(0.0, min(1.0, (shift_score - 0.95) / 0.04))
    reliability = max(0.10, 1 - 0.90 * activation)
    return activation, reliability


class OODReliabilityGate:
    def __init__(self, gate_dir: str | Path, *, expected_agent: str) -> None:
        self.gate_dir = Path(gate_dir)
        metadata_path = self.gate_dir / "metadata.json"
        artifact_path = self.gate_dir / "ood_gate.joblib"
        scores_path = self.gate_dir / "score_calibration.npz"
        if not all(path.exists() for path in (metadata_path, artifact_path, scores_path)):
            raise ModelArtifactError(
                f"OOD gate artifact is incomplete: {self.gate_dir}"
            )
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"invalid OOD metadata: {metadata_path}") from exc
        self._validate_metadata(expected_agent)
        expected_hash = self.metadata.get("gate_sha256")
        if expected_hash and _sha256_file(artifact_path) != expected_hash:
            raise ModelArtifactError(f"OOD gate checksum mismatch: {artifact_path}")
        try:
            payload = joblib.load(artifact_path)
            scores = np.load(scores_path)
        except Exception as exc:
            raise ModelArtifactError(
                f"cannot load OOD gate artifact: {self.gate_dir}"
            ) from exc
        if not isinstance(payload, dict) or {
            "imputer",
            "scaler",
            "estimator",
        } - set(payload):
            raise ModelArtifactError(f"invalid OOD gate payload: {artifact_path}")
        self.imputer = payload["imputer"]
        self.scaler = payload["scaler"]
        self.estimator = payload["estimator"]
        self.policy_scores = np.asarray(
            scores["policy_raw_scores"], dtype=np.float64
        )
        if not len(self.policy_scores):
            raise ModelArtifactError("OOD score calibration is empty")
        self.extract = feature_extractor(expected_agent)
        self.agent = expected_agent

    def _validate_metadata(self, expected_agent: str) -> None:
        if self.metadata.get("schema_version") != OOD_ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("unsupported OOD artifact schema version")
        if self.metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ModelArtifactError("OOD feature schema version mismatch")
        if self.metadata.get("agent") != expected_agent:
            raise ModelArtifactError(
                f"OOD gate agent mismatch: expected {expected_agent}, "
                f"got {self.metadata.get('agent')}"
            )
        if tuple(self.metadata.get("feature_names", [])) != feature_names(
            expected_agent
        ):
            raise ModelArtifactError("OOD feature names do not match runtime schema")

    def assess(self, detector_input: DetectorInput) -> OODAssessment:
        return self.assess_feature_matrix(
            self.extract(detector_input).reshape(1, -1)
        )[0]

    def assess_feature_matrix(
        self, features: np.ndarray
    ) -> list[OODAssessment]:
        transformed = self.scaler.transform(self.imputer.transform(features))
        raw_scores = -self.estimator.score_samples(transformed)
        shift_scores = (
            np.searchsorted(
                self.policy_scores, raw_scores, side="right"
            )
            / len(self.policy_scores)
        )
        return [
            self._assessment(float(raw), float(shift))
            for raw, shift in zip(raw_scores, shift_scores, strict=True)
        ]

    @staticmethod
    def _assessment(raw_score: float, shift_score: float) -> OODAssessment:
        level: Literal["in_domain", "warning", "hard"]
        if shift_score >= 0.99:
            level = "hard"
        elif shift_score >= 0.95:
            level = "warning"
        else:
            level = "in_domain"
        activation, reliability = ood_adjustment(shift_score)
        return OODAssessment(
            raw_score=raw_score,
            shift_score=shift_score,
            level=level,
            activation=activation,
            model_reliability=reliability,
        )


def apply_ood_policy_to_evidence(
    evidence: AgentEvidence,
    policy: OODPolicy,
) -> AgentEvidence:
    if policy not in OOD_POLICIES:
        raise ValueError(f"unsupported OOD policy: {policy}")
    if policy == "off" or evidence.distribution_shift_level == "off":
        return evidence.model_copy(
            update={"model_reliability": 1.0}
        )
    activation, reliability = ood_adjustment(
        evidence.distribution_shift_score
    )
    if policy == "hybrid" and evidence.distribution_shift_level == "hard":
        return evidence.model_copy(
            update={
                "benign_support": 0.0,
                "malicious_support": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "model_reliability": reliability,
                "abstained": True,
                "evidence": [
                    *evidence.evidence,
                    "hard model-distribution shift caused abstention",
                ],
            }
        )
    uncertainty = max(
        evidence.uncertainty,
        0.25 + 0.70 * activation,
    )
    assigned = evidence.benign_support + evidence.malicious_support
    risk = (
        evidence.malicious_support / assigned if assigned else 0.5
    )
    certainty = 1 - uncertainty
    malicious = risk * certainty
    benign = (1 - risk) * certainty
    return evidence.model_copy(
        update={
            "benign_support": benign,
            "malicious_support": malicious,
            "confidence": max(benign, malicious),
            "uncertainty": uncertainty,
            "model_reliability": reliability,
            "evidence": [
                *evidence.evidence,
                (
                    "model-distribution shift reliability discount applied "
                    f"(activation={activation:.3f})"
                ),
            ],
        }
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_ood_gate(
    *,
    agent: str,
    train: FeatureMatrix,
    policy: FeatureMatrix,
    output_dir: str | Path,
    seed: int,
    split_manifest_path: str | Path,
    base_model_dir: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = RobustScaler()
    train_transformed = scaler.fit_transform(imputer.fit_transform(train.x))
    estimator = IsolationForest(
        n_estimators=200,
        max_samples=min(4096, len(train.x)),
        contamination="auto",
        n_jobs=-1,
        random_state=seed,
    )
    estimator.fit(train_transformed)
    policy_transformed = scaler.transform(imputer.transform(policy.x))
    policy_raw_scores = np.sort(
        -estimator.score_samples(policy_transformed)
    ).astype(np.float64)
    warning_raw = float(np.quantile(policy_raw_scores, 0.95))
    hard_raw = float(np.quantile(policy_raw_scores, 0.99))
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "estimator": estimator,
        },
        target / "ood_gate.joblib",
        compress=3,
    )
    np.savez_compressed(
        target / "score_calibration.npz",
        policy_raw_scores=policy_raw_scores,
    )
    gate_sha = _sha256_file(target / "ood_gate.joblib")
    split_path = Path(split_manifest_path)
    base_root = Path(base_model_dir) if base_model_dir else None
    base_metadata = (
        base_root / agent / "metadata.json" if base_root else None
    )
    metadata = {
        "schema_version": OOD_ARTIFACT_SCHEMA_VERSION,
        "agent": agent,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(feature_names(agent)),
        "algorithm": "isolation_forest",
        "parameters": {
            "n_estimators": 200,
            "max_samples": min(4096, len(train.x)),
            "random_state": seed,
        },
        "warning_percentile": 0.95,
        "hard_percentile": 0.99,
        "warning_raw_score": warning_raw,
        "hard_raw_score": hard_raw,
        "train_sample_count": int(len(train.y)),
        "policy_sample_count": int(len(policy.y)),
        "gate_sha256": gate_sha,
        "split_manifest_sha256": _sha256_file(split_path),
        "base_model_metadata_sha256": (
            _sha256_file(base_metadata)
            if base_metadata and base_metadata.exists()
            else None
        ),
        "external_dataset_used": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
        **(extra_metadata or {}),
    }
    _write_json(target / "metadata.json", metadata)
    shift_scores = (
        np.searchsorted(
            policy_raw_scores, policy_raw_scores, side="right"
        )
        / len(policy_raw_scores)
    )
    _write_json(
        target / "calibration_report.json",
        {
            "schema_version": "1.0",
            "agent": agent,
            "warning_percentile": 0.95,
            "hard_percentile": 0.99,
            "policy_warning_or_higher_rate": float(
                np.mean(shift_scores >= 0.95)
            ),
            "policy_hard_rate": float(np.mean(shift_scores >= 0.99)),
            "raw_score_quantiles": {
                str(q): float(np.quantile(policy_raw_scores, q))
                for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            },
        },
    )
    return metadata


def train_ood_gates(
    dataset_dir: str | Path,
    base_model_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    flows = dataset / "flows"
    split_manifest = dataset / "splits" / "split-manifest.json"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    for agent in ("stats", "temporal"):
        train = load_feature_matrix(flows, split_manifest, "train", agent)
        validation = load_feature_matrix(
            flows, split_manifest, "validation", agent
        )
        _, policy_mask = _validation_partition(
            validation.y, validation.groups
        )
        policy = FeatureMatrix(
            x=validation.x[policy_mask],
            y=validation.y[policy_mask],
            groups=validation.groups[policy_mask],
            sample_ids=list(
                np.asarray(validation.sample_ids, dtype=object)[policy_mask]
            ),
        )
        models[agent] = train_ood_gate(
            agent=agent,
            train=train,
            policy=policy,
            output_dir=target / agent,
            seed=seed,
            split_manifest_path=split_manifest,
            base_model_dir=base_model_dir,
            extra_metadata={
                "dataset": "USTC-TFC2016",
                "training_split": "train",
                "threshold_split": "validation_policy_partition",
            },
        )
    summary = {
        "schema_version": "1.0",
        "dataset_dir": str(dataset.resolve()),
        "base_model_dir": str(Path(base_model_dir).resolve()),
        "output_dir": str(target.resolve()),
        "seed": seed,
        "external_dataset_used": False,
        "gates": models,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(target / "training_summary.json", summary)
    return summary
