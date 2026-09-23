from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .schemas import ConformalAssessment


SCHEMA_VERSION = "1.0"


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


def _distance(queries: np.ndarray, references: np.ndarray) -> np.ndarray:
    queries = _normalize(queries)
    references = _normalize(references)
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), 512):
        current = queries[start : start + 512]
        result[start : start + len(current)] = (
            1.0 - current @ references.T
        ).min(axis=1)
    return result


def _p_value(score: float, calibration: np.ndarray) -> float:
    return float(
        (1 + np.count_nonzero(calibration >= score))
        / (len(calibration) + 1)
    )


def fit_prefix_conformal(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    output_dir: str | Path,
    *,
    alpha: float = 0.01,
    hard_p_value: float = 0.01,
    max_references_per_class: int = 2048,
    seed: int = 42,
) -> dict:
    if alpha != 0.01 or hard_p_value != 0.01:
        raise ValueError("prefix conformal alpha/hard p-value are frozen")
    rng = np.random.default_rng(seed)
    references = {}
    scores = {}
    for label in (0, 1):
        values = np.asarray(
            reference_embeddings[reference_labels == label],
            dtype=np.float32,
        )
        if not len(values):
            raise ValueError(f"missing reference embeddings for class {label}")
        if len(values) > max_references_per_class:
            values = values[
                np.sort(
                    rng.choice(
                        len(values),
                        max_references_per_class,
                        replace=False,
                    )
                )
            ]
        references[label] = _normalize(values)
        class_calibration = calibration_embeddings[
            calibration_labels == label
        ]
        if not len(class_calibration):
            raise ValueError(f"missing calibration embeddings for class {label}")
        scores[label] = np.sort(
            _distance(class_calibration, references[label])
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "conformal_artifact.npz"
    np.savez_compressed(
        artifact,
        benign_references=references[0],
        malicious_references=references[1],
        benign_calibration_scores=scores[0],
        malicious_calibration_scores=scores[1],
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "policy": "conformal_v3_1_prefix",
        "method": "class_conditional_embedding_1nn",
        "alpha": alpha,
        "hard_p_value": hard_p_value,
        "reference_counts": {
            "benign": len(references[0]),
            "malicious": len(references[1]),
        },
        "calibration_counts": {
            "benign": len(scores[0]),
            "malicious": len(scores[1]),
        },
        "seed": seed,
        "artifact_sha256": _sha256(artifact),
        "test_or_external_used": False,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


class ConformalPrefixGate:
    name = "ConformalPrefixGate"
    version = "3.1-prefix"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        artifact_path = root / "conformal_artifact.npz"
        if not metadata_path.exists() or not artifact_path.exists():
            raise ValueError(f"incomplete prefix conformal artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("policy") != "conformal_v3_1_prefix":
            raise ValueError("prefix conformal policy mismatch")
        if _sha256(artifact_path) != self.metadata["artifact_sha256"]:
            raise ValueError("prefix conformal artifact checksum mismatch")
        artifact = np.load(artifact_path)
        self.references = {
            0: artifact["benign_references"],
            1: artifact["malicious_references"],
        }
        self.scores = {
            0: artifact["benign_calibration_scores"],
            1: artifact["malicious_calibration_scores"],
        }
        self.alpha = float(self.metadata["alpha"])
        self.hard_p_value = float(self.metadata["hard_p_value"])

    def assess(
        self,
        embedding: np.ndarray,
        probability: float,
    ) -> ConformalAssessment:
        del probability
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        p_values = {
            label: _p_value(
                float(_distance(query, self.references[label])[0]),
                self.scores[label],
            )
            for label in (0, 1)
        }
        prediction_set = [
            name
            for label, name in ((0, "benign"), (1, "malicious"))
            if p_values[label] >= self.alpha
        ]
        hard = all(value < self.hard_p_value for value in p_values.values())
        level = (
            "hard"
            if hard
            else "warning"
            if len(prediction_set) != 1
            else "in_domain"
        )
        return ConformalAssessment(
            benign_p_value=p_values[0],
            malicious_p_value=p_values[1],
            prediction_set=prediction_set,
            shift_score=max(0.0, min(1.0, 1 - max(p_values.values()))),
            level=level,
            calibration_scope="within_dataset_only",
        )
