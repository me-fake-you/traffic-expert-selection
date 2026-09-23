from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ood_v32 import _nearest_distance, _normalize
from .schemas import ConformalAssessment


CONFORMAL_V34_ORBIT_SCHEMA_VERSION = "1.0"


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


def _p_values(
    distances: np.ndarray,
    calibration_scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(calibration_scores, dtype=np.float32)
    indices = np.searchsorted(scores, distances, side="left")
    counts = len(scores) - indices
    return (1 + counts) / (len(scores) + 1)


def _accelerated_nearest_distance(
    queries: np.ndarray,
    references: np.ndarray,
) -> np.ndarray:
    try:
        import torch
    except ImportError:
        return _nearest_distance(queries, references)
    if not torch.cuda.is_available():
        return _nearest_distance(queries, references)
    normalized_queries = _normalize(queries)
    normalized_references = _normalize(references)
    reference_tensor = torch.from_numpy(
        normalized_references
    ).to("cuda")
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(normalized_queries), 4096):
            current = torch.from_numpy(
                normalized_queries[start : start + 4096]
            ).to("cuda")
            distances = 1.0 - torch.max(
                current @ reference_tensor.T,
                dim=1,
            ).values
            output.append(distances.float().cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def fit_orbit_conformal_v34(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_labels: np.ndarray,
    output_dir: str | Path,
    *,
    regime: str,
    alpha: float = 0.01,
    hard_p_value: float = 0.01,
    max_references_per_class: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    if regime not in {"short", "long"}:
        raise ValueError(f"unsupported orbit conformal regime: {regime}")
    if alpha != 0.01 or hard_p_value != 0.01:
        raise ValueError(
            "conformal_v3_4_orbit freezes alpha and hard p-value at 0.01"
        )
    reference_embeddings = np.asarray(
        reference_embeddings,
        dtype=np.float32,
    )
    reference_labels = np.asarray(reference_labels, dtype=np.int8)
    calibration_embeddings = np.asarray(
        calibration_embeddings,
        dtype=np.float32,
    )
    calibration_labels = np.asarray(calibration_labels, dtype=np.int8)
    if set(np.unique(reference_labels)) != {0, 1}:
        raise ValueError("orbit reference requires both classes")
    if set(np.unique(calibration_labels)) != {0, 1}:
        raise ValueError("orbit calibration requires both classes")

    rng = np.random.default_rng(seed)
    references: dict[int, np.ndarray] = {}
    scores: dict[int, np.ndarray] = {}
    for label in (0, 1):
        values = reference_embeddings[reference_labels == label]
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
        calibration = calibration_embeddings[
            calibration_labels == label
        ]
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
        "schema_version": CONFORMAL_V34_ORBIT_SCHEMA_VERSION,
        "policy": "conformal_v3_4_orbit",
        "regime": regime,
        "method": "orbit_embedding_1nn",
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
        "max_references_per_class": max_references_per_class,
        "seed": seed,
        "orbit_kinds": [
            "clean",
            "padding",
            "dummy_packet",
            "iat_jitter",
            "sequence_truncation",
        ],
        "strength": 0.2,
        "orbit_seed": 42,
        "selection_performed": False,
        "test_or_external_used": False,
        "cross_dataset_coverage_guarantee_claimed": False,
        "artifact_sha256": _sha256(artifact_path),
    }
    _dump(root / "metadata.json", metadata)
    return metadata


class OrbitConformalGateV34:
    name = "OrbitConformalGateV34"
    version = "3.4"

    def __init__(self, artifact_dir: str | Path, *, regime: str) -> None:
        root = Path(artifact_dir) / regime
        metadata_path = root / "metadata.json"
        artifact_path = root / "conformal_artifact.npz"
        if not metadata_path.exists() or not artifact_path.exists():
            raise ValueError(
                f"incomplete conformal_v3_4_orbit {regime}: {root}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.metadata.get("schema_version")
            != CONFORMAL_V34_ORBIT_SCHEMA_VERSION
            or self.metadata.get("policy") != "conformal_v3_4_orbit"
            or self.metadata.get("regime") != regime
        ):
            raise ValueError("conformal_v3_4_orbit metadata mismatch")
        if _sha256(artifact_path) != self.metadata.get("artifact_sha256"):
            raise ValueError("conformal_v3_4_orbit checksum mismatch")
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

    def p_values_many(
        self,
        embeddings: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(embeddings, dtype=np.float32)
        benign_distance = _accelerated_nearest_distance(
            values,
            self.references[0],
        )
        malicious_distance = _accelerated_nearest_distance(
            values,
            self.references[1],
        )
        return (
            _p_values(benign_distance, self.scores[0]),
            _p_values(malicious_distance, self.scores[1]),
        )

    def assess(self, embedding: np.ndarray) -> ConformalAssessment:
        benign_values, malicious_values = self.p_values_many(
            np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        )
        benign = float(benign_values[0])
        malicious = float(malicious_values[0])
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
            calibration_scope=f"within_dataset_orbit_{self.regime}_regime",
        )
