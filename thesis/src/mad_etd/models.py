from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import FEATURE_SCHEMA_VERSION, feature_extractor, feature_names
from .schemas import DetectorInput


MODEL_ARTIFACT_SCHEMA_VERSION = "1.0"


class ModelArtifactError(RuntimeError):
    pass


@dataclass(slots=True)
class DetectorPrediction:
    malicious_probability: float
    uncertainty: float
    calibration_quality: float
    accepted_class: str | None


class SigmoidCalibrator:
    """Small Platt-scaling wrapper that remains stable in joblib artifacts."""

    def __init__(self) -> None:
        self.model: Any | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "SigmoidCalibrator":
        from sklearn.linear_model import LogisticRegression

        if len(np.unique(labels)) < 2:
            raise ValueError("calibration requires both benign and malicious labels")
        self.model = LogisticRegression(C=1_000_000, max_iter=1000, random_state=42)
        self.model.fit(self._logit(probabilities), labels)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        return self.model.predict_proba(self._logit(probabilities))[:, 1]


class LearnedDetectorModel:
    def __init__(self, model_dir: str | Path, *, expected_agent: str) -> None:
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "metadata.json"
        model_path = self.model_dir / "model.joblib"
        if not metadata_path.exists() or not model_path.exists():
            raise ModelArtifactError(
                f"learned detector artifact is incomplete: {self.model_dir}"
            )
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactError(f"invalid metadata: {metadata_path}") from exc
        self._validate_metadata(expected_agent)
        expected_hash = self.metadata.get("model_sha256")
        if expected_hash and self._sha256(model_path) != expected_hash:
            raise ModelArtifactError(f"model checksum mismatch: {model_path}")
        try:
            artifact = joblib.load(model_path)
        except Exception as exc:  # joblib exposes backend-specific exceptions
            raise ModelArtifactError(f"cannot load model artifact: {model_path}") from exc
        if not isinstance(artifact, dict) or {
            "estimator",
            "calibrator",
        } - set(artifact):
            raise ModelArtifactError(f"invalid model payload: {model_path}")
        self.estimator = artifact["estimator"]
        self.calibrator = artifact["calibrator"]
        self.extract = feature_extractor(expected_agent)
        self.agent = expected_agent

    def _validate_metadata(self, expected_agent: str) -> None:
        if self.metadata.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("unsupported model artifact schema version")
        if self.metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ModelArtifactError("feature schema version mismatch")
        if self.metadata.get("agent") != expected_agent:
            raise ModelArtifactError(
                f"model agent mismatch: expected {expected_agent}, "
                f"got {self.metadata.get('agent')}"
            )
        if tuple(self.metadata.get("feature_names", [])) != feature_names(expected_agent):
            raise ModelArtifactError("feature names do not match runtime schema")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def predict(self, detector_input: DetectorInput) -> DetectorPrediction:
        return self.batch_predict([detector_input])[0]

    def batch_predict(
        self,
        detector_inputs: list[DetectorInput],
    ) -> list[DetectorPrediction]:
        if not detector_inputs:
            return []
        features = np.vstack(
            [self.extract(detector_input) for detector_input in detector_inputs]
        )
        raw_probability = self.estimator.predict_proba(features)[:, 1]
        probabilities = self.calibrator.predict(raw_probability)
        return [self._prediction_from_probability(float(item)) for item in probabilities]

    def _prediction_from_probability(self, probability: float) -> DetectorPrediction:
        thresholds = self.metadata["decision_policy"]
        benign_max = float(thresholds["benign_max_probability"])
        malicious_min = float(thresholds["malicious_min_probability"])
        if probability <= benign_max:
            accepted_class = "benign"
            distance = (benign_max - probability) / max(benign_max, 1e-6)
            uncertainty = 0.08 + 0.17 * (1 - min(1.0, distance))
        elif probability >= malicious_min:
            accepted_class = "malicious"
            distance = (probability - malicious_min) / max(
                1 - malicious_min, 1e-6
            )
            uncertainty = 0.08 + 0.17 * (1 - min(1.0, distance))
        else:
            accepted_class = None
            half_gap = max((malicious_min - benign_max) / 2, 1e-6)
            boundary_distance = min(
                probability - benign_max,
                malicious_min - probability,
            )
            centrality = min(1.0, boundary_distance / half_gap)
            uncertainty = 0.55 + 0.35 * centrality
        calibration_quality = float(
            self.metadata.get("calibration_quality", 0.8)
        )
        return DetectorPrediction(
            malicious_probability=max(0.0, min(1.0, probability)),
            uncertainty=max(0.0, min(1.0, uncertainty)),
            calibration_quality=max(0.0, min(1.0, calibration_quality)),
            accepted_class=accepted_class,
        )


def probability_entropy(probability: float) -> float:
    probability = max(1e-9, min(1 - 1e-9, probability))
    return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )
