from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ConformalAssessment


CONFORMAL_V32_SCHEMA_VERSION = "1.0"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values),
        where=norms != 0,
    )


def _nearest_distance(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    queries = _normalize(queries)
    references = _normalize(references)
    if not len(references):
        return np.full(len(queries), np.inf, dtype=np.float32)
    output = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        current = queries[start : start + chunk_size]
        output[start : start + len(current)] = 1.0 - np.max(
            current @ references.T,
            axis=1,
        )
    return output


def fit_regime_conformal_v32(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    output_dir: str | Path,
    *,
    regime: str,
    alpha: float = 0.01,
    hard_p_value: float = 0.01,
    max_references_per_class: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    if regime not in {"short", "long"}:
        raise ValueError(f"unsupported conformal_v3_2 regime: {regime}")
    if alpha != 0.01 or hard_p_value != 0.01:
        raise ValueError("conformal_v3_2 freezes alpha and hard p-value at 0.01")
    references_input = np.asarray(
        reference_embeddings, dtype=np.float32
    )
    reference_labels = np.asarray(reference_labels, dtype=np.int8)
    calibration_embeddings = np.asarray(
        calibration_embeddings, dtype=np.float32
    )
    calibration_labels = np.asarray(calibration_labels, dtype=np.int8)
    if set(np.unique(reference_labels)) != {0, 1}:
        raise ValueError("conformal_v3_2 reference requires both classes")
    if set(np.unique(calibration_labels)) != {0, 1}:
        raise ValueError("conformal_v3_2 calibration requires both classes")
    rng = np.random.default_rng(seed)
    references: dict[int, np.ndarray] = {}
    scores: dict[int, np.ndarray] = {}
    for label in (0, 1):
        values = references_input[reference_labels == label]
        if len(values) > max_references_per_class:
            indices = np.sort(
                rng.choice(
                    len(values),
                    size=max_references_per_class,
                    replace=False,
                )
            )
            values = values[indices]
        references[label] = _normalize(values)
        calibration = calibration_embeddings[calibration_labels == label]
        scores[label] = np.sort(
            _nearest_distance(calibration, references[label])
        )
    root = Path(output_dir) / regime
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "conformal_artifact.npz"
    np.savez_compressed(
        artifact_path,
        benign_references=references[0],
        malicious_references=references[1],
        benign_calibration_scores=scores[0],
        malicious_calibration_scores=scores[1],
    )
    metadata = {
        "schema_version": CONFORMAL_V32_SCHEMA_VERSION,
        "policy": "conformal_v3_2",
        "regime": regime,
        "method": "embedding_1nn",
        "alpha": alpha,
        "hard_p_value": hard_p_value,
        "reference_counts": {
            "benign": int(len(references[0])),
            "malicious": int(len(references[1])),
        },
        "calibration_counts": {
            "benign": int(len(scores[0])),
            "malicious": int(len(scores[1])),
        },
        "seed": seed,
        "selection_performed": False,
        "test_or_external_used": False,
        "cross_dataset_coverage_guarantee_claimed": False,
        "artifact_sha256": _sha256(artifact_path),
    }
    _dump(root / "metadata.json", metadata)
    return metadata


class RegimeConformalGateV32:
    name = "RegimeConformalGateV32"
    version = "3.2"

    def __init__(self, artifact_dir: str | Path, *, regime: str) -> None:
        root = Path(artifact_dir) / regime
        metadata_path = root / "metadata.json"
        artifact_path = root / "conformal_artifact.npz"
        if not metadata_path.exists() or not artifact_path.exists():
            raise ValueError(
                f"incomplete conformal_v3_2 {regime} artifact: {root}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.metadata.get("schema_version")
            != CONFORMAL_V32_SCHEMA_VERSION
            or self.metadata.get("policy") != "conformal_v3_2"
            or self.metadata.get("regime") != regime
        ):
            raise ValueError("conformal_v3_2 metadata mismatch")
        if _sha256(artifact_path) != self.metadata.get("artifact_sha256"):
            raise ValueError("conformal_v3_2 artifact checksum mismatch")
        artifact = np.load(artifact_path)
        self.references = {
            0: artifact["benign_references"].astype(np.float32),
            1: artifact["malicious_references"].astype(np.float32),
        }
        self.scores = {
            0: artifact["benign_calibration_scores"].astype(np.float32),
            1: artifact["malicious_calibration_scores"].astype(np.float32),
        }
        self.alpha = float(self.metadata["alpha"])
        self.hard_p_value = float(self.metadata["hard_p_value"])
        self.regime = regime

    def _p_value(self, embedding: np.ndarray, label: int) -> float:
        distance = float(
            _nearest_distance(
                np.asarray(embedding, dtype=np.float32).reshape(1, -1),
                self.references[label],
            )[0]
        )
        calibration = self.scores[label]
        return float(
            (1 + np.count_nonzero(calibration >= distance))
            / (len(calibration) + 1)
        )

    def assess(self, embedding: np.ndarray) -> ConformalAssessment:
        benign = self._p_value(embedding, 0)
        malicious = self._p_value(embedding, 1)
        prediction_set = []
        if benign >= self.alpha:
            prediction_set.append("benign")
        if malicious >= self.alpha:
            prediction_set.append("malicious")
        hard = benign < self.hard_p_value and malicious < self.hard_p_value
        level = (
            "hard"
            if hard
            else "warning"
            if len(prediction_set) != 1
            else "in_domain"
        )
        return ConformalAssessment(
            benign_p_value=benign,
            malicious_p_value=malicious,
            prediction_set=prediction_set,
            shift_score=max(0.0, min(1.0, 1 - max(benign, malicious))),
            level=level,
            calibration_scope=f"within_dataset_{self.regime}_regime",
        )
