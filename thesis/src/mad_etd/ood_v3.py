from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ConformalAssessment


CONFORMAL_V3_SCHEMA_VERSION = "1.0"


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
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        current = queries[start : start + chunk_size]
        similarities = current @ references.T
        result[start : start + len(current)] = 1.0 - np.max(
            similarities, axis=1
        )
    return result


def fit_conformal_artifact(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    output_dir: str | Path,
    *,
    alpha: float = 0.1,
    hard_p_value: float = 0.01,
    max_references_per_class: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    if not 0 < hard_p_value < alpha < 1:
        raise ValueError("require 0 < hard_p_value < alpha < 1")
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int8)
    calibration_embeddings = np.asarray(
        calibration_embeddings, dtype=np.float32
    )
    calibration_labels = np.asarray(calibration_labels, dtype=np.int8)
    if train_embeddings.ndim != 2 or calibration_embeddings.ndim != 2:
        raise ValueError("conformal embeddings must be two-dimensional")
    if train_embeddings.shape[1] != calibration_embeddings.shape[1]:
        raise ValueError("train and calibration embedding dimensions differ")
    if set(np.unique(train_labels)) != {0, 1}:
        raise ValueError("conformal reference fitting requires both classes")
    if set(np.unique(calibration_labels)) != {0, 1}:
        raise ValueError("conformal calibration requires both classes")

    rng = np.random.default_rng(seed)
    references: dict[int, np.ndarray] = {}
    scores: dict[int, np.ndarray] = {}
    for label in (0, 1):
        class_train = train_embeddings[train_labels == label]
        if len(class_train) > max_references_per_class:
            indices = np.sort(
                rng.choice(
                    len(class_train),
                    size=max_references_per_class,
                    replace=False,
                )
            )
            class_train = class_train[indices]
        references[label] = _normalize(class_train)
        class_calibration = calibration_embeddings[
            calibration_labels == label
        ]
        scores[label] = np.sort(
            _nearest_distance(class_calibration, references[label])
        )

    root = Path(output_dir)
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
        "schema_version": CONFORMAL_V3_SCHEMA_VERSION,
        "policy": "conformal_v3",
        "alpha": alpha,
        "hard_p_value": hard_p_value,
        "embedding_dim": int(train_embeddings.shape[1]),
        "reference_counts": {
            "benign": int(len(references[0])),
            "malicious": int(len(references[1])),
        },
        "calibration_counts": {
            "benign": int(len(scores[0])),
            "malicious": int(len(scores[1])),
        },
        "seed": seed,
        "external_dataset_used": False,
        "cross_dataset_coverage_guarantee_claimed": False,
        "artifact_sha256": _sha256(artifact_path),
    }
    _dump(root / "metadata.json", metadata)
    return metadata


class ConformalReliabilityGate:
    name = "ConformalReliabilityGate"
    version = "3.0"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        artifact_path = root / "conformal_artifact.npz"
        if not metadata_path.exists() or not artifact_path.exists():
            raise ValueError(f"incomplete conformal_v3 artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != CONFORMAL_V3_SCHEMA_VERSION:
            raise ValueError("unsupported conformal_v3 artifact schema")
        if _sha256(artifact_path) != self.metadata.get("artifact_sha256"):
            raise ValueError("conformal_v3 artifact checksum mismatch")
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
        level = "hard" if hard else "warning" if len(prediction_set) != 1 else "in_domain"
        return ConformalAssessment(
            benign_p_value=benign,
            malicious_p_value=malicious,
            prediction_set=prediction_set,
            shift_score=max(0.0, min(1.0, 1 - max(benign, malicious))),
            level=level,
        )
