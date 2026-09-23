from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ConformalAssessment


CONFORMAL_V31_SCHEMA_VERSION = "1.0"
CONFORMAL_V31_METHODS = (
    "embedding_1nn",
    "embedding_knn",
    "probability_mondrian",
)
CONFORMAL_V31_ALPHAS = (0.01, 0.02, 0.05, 0.10)


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


def _knn_distance(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    k: int,
    chunk_size: int = 512,
) -> np.ndarray:
    queries = _normalize(queries)
    references = _normalize(references)
    if not len(references):
        return np.full(len(queries), np.inf, dtype=np.float32)
    k = max(1, min(k, len(references)))
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        current = queries[start : start + chunk_size]
        distances = 1.0 - current @ references.T
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        result[start : start + len(current)] = nearest.mean(axis=1)
    return result


def _probability_scores(
    probabilities: np.ndarray,
    candidate_label: int,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return probabilities if candidate_label == 0 else 1.0 - probabilities


def _p_values(
    candidate_scores: np.ndarray,
    calibration_scores: np.ndarray,
) -> np.ndarray:
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float32)
    sorted_scores = np.sort(calibration_scores)
    first_ge = np.searchsorted(
        sorted_scores,
        candidate_scores,
        side="left",
    )
    greater_equal = len(sorted_scores) - first_ge
    return (1 + greater_equal) / (len(sorted_scores) + 1)


def _method_scores(
    method: str,
    *,
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    references: dict[int, np.ndarray],
    k: int,
) -> dict[int, np.ndarray]:
    if method == "probability_mondrian":
        return {
            label: _probability_scores(probabilities, label)
            for label in (0, 1)
        }
    if method not in {"embedding_1nn", "embedding_knn"}:
        raise ValueError(f"unsupported conformal_v3_1 method: {method}")
    neighbors = 1 if method == "embedding_1nn" else k
    return {
        label: _knn_distance(
            embeddings,
            references[label],
            k=neighbors,
        )
        for label in (0, 1)
    }


def fit_conformal_v31_artifact(
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    policy_embeddings: np.ndarray,
    policy_probabilities: np.ndarray,
    policy_labels: np.ndarray,
    output_dir: str | Path,
    *,
    benign_max_probability: float,
    malicious_min_probability: float,
    hard_p_value: float = 0.01,
    alphas: tuple[float, ...] = CONFORMAL_V31_ALPHAS,
    k: int = 5,
    max_references_per_class: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    reference_embeddings = np.asarray(
        reference_embeddings, dtype=np.float32
    )
    reference_labels = np.asarray(reference_labels, dtype=np.int8)
    calibration_embeddings = np.asarray(
        calibration_embeddings, dtype=np.float32
    )
    calibration_probabilities = np.asarray(
        calibration_probabilities, dtype=np.float32
    )
    calibration_labels = np.asarray(calibration_labels, dtype=np.int8)
    policy_embeddings = np.asarray(policy_embeddings, dtype=np.float32)
    policy_probabilities = np.asarray(
        policy_probabilities, dtype=np.float32
    )
    policy_labels = np.asarray(policy_labels, dtype=np.int8)
    for name, labels in (
        ("reference", reference_labels),
        ("calibration", calibration_labels),
        ("policy", policy_labels),
    ):
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(
                f"conformal_v3_1 {name} split requires both classes"
            )
    if any(not 0 < alpha <= 0.1 for alpha in alphas):
        raise ValueError("conformal_v3_1 alpha candidates must be in (0, 0.1]")
    if not 0 < hard_p_value <= min(alphas):
        raise ValueError(
            "hard_p_value must be positive and no larger than minimum alpha"
        )

    rng = np.random.default_rng(seed)
    references: dict[int, np.ndarray] = {}
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

    candidates: list[dict[str, Any]] = []
    calibration_by_method: dict[str, dict[int, np.ndarray]] = {}
    policy_by_method: dict[str, dict[int, np.ndarray]] = {}
    for method in CONFORMAL_V31_METHODS:
        calibration_candidate_scores = _method_scores(
            method,
            embeddings=calibration_embeddings,
            probabilities=calibration_probabilities,
            references=references,
            k=k,
        )
        calibration_scores = {
            label: np.sort(
                calibration_candidate_scores[label][
                    calibration_labels == label
                ]
            )
            for label in (0, 1)
        }
        calibration_by_method[method] = calibration_scores
        policy_candidate_scores = _method_scores(
            method,
            embeddings=policy_embeddings,
            probabilities=policy_probabilities,
            references=references,
            k=k,
        )
        policy_p_values = {
            label: _p_values(
                policy_candidate_scores[label],
                calibration_scores[label],
            )
            for label in (0, 1)
        }
        policy_by_method[method] = policy_p_values
        probability_accepted = np.full(len(policy_labels), -1, dtype=np.int8)
        probability_accepted[
            policy_probabilities <= benign_max_probability
        ] = 0
        probability_accepted[
            policy_probabilities >= malicious_min_probability
        ] = 1
        for alpha in alphas:
            benign_in = policy_p_values[0] >= alpha
            malicious_in = policy_p_values[1] >= alpha
            singleton = benign_in ^ malicious_in
            conformal_class = np.where(malicious_in, 1, 0)
            covered = (
                singleton
                & (probability_accepted >= 0)
                & (conformal_class == probability_accepted)
            )
            selective_error = (
                float(
                    np.mean(
                        probability_accepted[covered]
                        != policy_labels[covered]
                    )
                )
                if covered.any()
                else 1.0
            )
            coverage = float(covered.mean())
            hard = (
                (policy_p_values[0] < hard_p_value)
                & (policy_p_values[1] < hard_p_value)
            )
            candidates.append(
                {
                    "method": method,
                    "alpha": float(alpha),
                    "coverage": coverage,
                    "selective_error": selective_error,
                    "hard_ood_rate": float(hard.mean()),
                    "ambiguous_rate": float(
                        np.mean(benign_in & malicious_in)
                    ),
                    "empty_set_rate": float(
                        np.mean(~benign_in & ~malicious_in)
                    ),
                    "constraints_satisfied": (
                        coverage >= 0.60 and selective_error <= 0.05
                    ),
                }
            )

    method_order = {
        method: index for index, method in enumerate(CONFORMAL_V31_METHODS)
    }
    eligible = [
        candidate
        for candidate in candidates
        if candidate["constraints_satisfied"]
    ]
    pool = eligible or candidates
    selected = sorted(
        pool,
        key=lambda candidate: (
            -candidate["coverage"],
            candidate["selective_error"],
            candidate["hard_ood_rate"],
            method_order[candidate["method"]],
            candidate["alpha"],
        ),
    )[0]
    selected_method = selected["method"]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "conformal_artifact.npz"
    np.savez_compressed(
        artifact_path,
        benign_references=references[0],
        malicious_references=references[1],
        benign_calibration_scores=calibration_by_method[selected_method][0],
        malicious_calibration_scores=calibration_by_method[selected_method][1],
    )
    metadata = {
        "schema_version": CONFORMAL_V31_SCHEMA_VERSION,
        "policy": "conformal_v3_1",
        "selected_method": selected_method,
        "alpha": selected["alpha"],
        "hard_p_value": hard_p_value,
        "k": k,
        "embedding_dim": int(reference_embeddings.shape[1]),
        "reference_counts": {
            "benign": int(len(references[0])),
            "malicious": int(len(references[1])),
        },
        "calibration_counts": {
            "benign": int(
                len(calibration_by_method[selected_method][0])
            ),
            "malicious": int(
                len(calibration_by_method[selected_method][1])
            ),
        },
        "policy_count": int(len(policy_labels)),
        "candidates": candidates,
        "selected": selected,
        "selection_split": "validation_hash_policy",
        "calibration_split": "validation_hash_calibration",
        "test_or_external_used_for_selection": False,
        "cross_dataset_coverage_guarantee_claimed": False,
        "seed": seed,
        "artifact_sha256": _sha256(artifact_path),
    }
    _dump(root / "metadata.json", metadata)
    return metadata


class ConformalReliabilityGateV31:
    name = "ConformalReliabilityGateV31"
    version = "3.1"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        artifact_path = root / "conformal_artifact.npz"
        if not metadata_path.exists() or not artifact_path.exists():
            raise ValueError(f"incomplete conformal_v3_1 artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.metadata.get("schema_version")
            != CONFORMAL_V31_SCHEMA_VERSION
        ):
            raise ValueError("unsupported conformal_v3_1 artifact schema")
        if _sha256(artifact_path) != self.metadata.get("artifact_sha256"):
            raise ValueError("conformal_v3_1 artifact checksum mismatch")
        artifact = np.load(artifact_path)
        self.references = {
            0: artifact["benign_references"].astype(np.float32),
            1: artifact["malicious_references"].astype(np.float32),
        }
        self.scores = {
            0: artifact["benign_calibration_scores"].astype(np.float32),
            1: artifact["malicious_calibration_scores"].astype(np.float32),
        }
        self.method = str(self.metadata["selected_method"])
        self.alpha = float(self.metadata["alpha"])
        self.hard_p_value = float(self.metadata["hard_p_value"])
        self.k = int(self.metadata.get("k", 5))

    def _candidate_score(
        self,
        embedding: np.ndarray,
        probability: float,
        label: int,
    ) -> float:
        if self.method == "probability_mondrian":
            return float(probability if label == 0 else 1 - probability)
        neighbors = 1 if self.method == "embedding_1nn" else self.k
        return float(
            _knn_distance(
                np.asarray(embedding, dtype=np.float32).reshape(1, -1),
                self.references[label],
                k=neighbors,
            )[0]
        )

    def _p_value(
        self,
        embedding: np.ndarray,
        probability: float,
        label: int,
    ) -> float:
        score = self._candidate_score(embedding, probability, label)
        calibration = self.scores[label]
        return float(
            (1 + np.count_nonzero(calibration >= score))
            / (len(calibration) + 1)
        )

    def assess(
        self,
        embedding: np.ndarray,
        probability: float,
    ) -> ConformalAssessment:
        benign = self._p_value(embedding, probability, 0)
        malicious = self._p_value(embedding, probability, 1)
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
            calibration_scope="within_dataset_validation_only",
        )
