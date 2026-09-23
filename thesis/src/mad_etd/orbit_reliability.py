from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .deep_models import DeepDetectorModel, encode_temporal_input
from .io import iter_split_records
from .ood_v34_orbit import (
    OrbitConformalGateV34,
    fit_orbit_conformal_v34,
)
from .perturb import apply_perturbation
from .schemas import DetectorInput, FlowRecord
from .short_flow_robust import RobustShortFlowModel
from .v2_training import TensorDatasetBundle, _partition_validation, _predict


ORBIT_KINDS = (
    "clean",
    "padding",
    "dummy_packet",
    "iat_jitter",
    "sequence_truncation",
)
ORBIT_RELIABILITY_FEATURE_NAMES = (
    "clean_probability",
    "orbit_class_consistency",
    "support_min",
    "support_mean",
    "support_max",
    "support_span",
    "max_uncertainty",
    "min_class_p_value",
    "hard_ood_count",
    "route_crossing_count",
)


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


def _safe_input(record: FlowRecord) -> DetectorInput:
    return DetectorInput(
        stats=record.stats,
        sequence=record.sequence,
        tls=record.tls,
        payload_tokens=record.payload_tokens,
        context={
            key: value
            for key, value in record.context.items()
            if key in {"transport", "protocols"}
        },
    )


def orbit_records(record: FlowRecord) -> list[FlowRecord]:
    values = [record.model_copy(deep=True)]
    for kind in ORBIT_KINDS[1:]:
        values.append(
            apply_perturbation(
                record,
                kind,
                strength=0.2,
                seed=42,
            )
        )
    return values


def _cell_key(record: FlowRecord) -> tuple[int, int] | None:
    count = len(record.sequence.packet_lengths)
    label = str(record.labels.get("binary", "")).lower()
    if count < 1 or label not in {"benign", "malicious"}:
        return None
    bucket = min(count, 4)
    return bucket, int(label == "malicious")


def select_orbit_records(
    records: Iterable[FlowRecord],
    *,
    max_per_bucket_label: int,
) -> list[FlowRecord]:
    reservoirs: dict[
        tuple[int, int],
        list[tuple[int, str, FlowRecord]],
    ] = {
        (bucket, label): []
        for bucket in (1, 2, 3, 4)
        for label in (0, 1)
    }
    for record in records:
        key = _cell_key(record)
        if key is None:
            continue
        score = int(
            hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:16],
            16,
        )
        item = (-score, record.sample_id, record)
        current = reservoirs[key]
        if len(current) < max_per_bucket_label:
            heapq.heappush(current, item)
        elif item[:2] > current[0][:2]:
            heapq.heapreplace(current, item)
    selected = [
        record
        for values in reservoirs.values()
        for _, _, record in values
    ]
    return sorted(selected, key=lambda record: record.sample_id)


def validation_calibration_records(
    records: Iterable[FlowRecord],
    *,
    max_per_bucket_label: int,
) -> list[FlowRecord]:
    materialized = list(records)
    mask, _ = _partition_validation(
        [record.sample_id for record in materialized]
    )
    return select_orbit_records(
        (
            record
            for record, selected in zip(
                materialized,
                mask,
                strict=True,
            )
            if selected
        ),
        max_per_bucket_label=max_per_bucket_label,
    )


def validation_selection_records(
    records: Iterable[FlowRecord],
) -> list[FlowRecord]:
    materialized = list(records)
    _, mask = _partition_validation(
        [record.sample_id for record in materialized]
    )
    return [
        record
        for record, selected in zip(materialized, mask, strict=True)
        if selected
    ]


def _accepted_class(
    probability: float,
    policy: dict[str, Any],
) -> str | None:
    if probability <= float(policy["benign_max_probability"]):
        return "benign"
    if probability >= float(policy["malicious_min_probability"]):
        return "malicious"
    return None


def _entropy(probability: float) -> float:
    probability = max(1e-9, min(1 - 1e-9, probability))
    return -(
        probability * math.log2(probability)
        + (1 - probability) * math.log2(1 - probability)
    )


def batch_temporal_predictions(
    inputs: list[DetectorInput],
    *,
    model_dir: str | Path,
    conformal_dir: str | Path | None = None,
    batch_size: int = 2048,
) -> list[dict[str, Any]]:
    if not inputs:
        return []
    short_model = RobustShortFlowModel(
        Path(model_dir) / "temporal" / "short"
    )
    long_model = DeepDetectorModel(
        Path(model_dir) / "temporal" / "long",
        expected_agent="temporal",
        ood_policy="off",
    )
    gates = (
        {
            "short": OrbitConformalGateV34(
                conformal_dir,
                regime="short",
            ),
            "long": OrbitConformalGateV34(
                conformal_dir,
                regime="long",
            ),
        }
        if conformal_dir is not None
        else {}
    )
    result: list[dict[str, Any] | None] = [None] * len(inputs)
    for regime, indices, wrapper in (
        (
            "short",
            [
                index
                for index, value in enumerate(inputs)
                if len(value.sequence.packet_lengths) <= 3
            ],
            short_model,
        ),
        (
            "long",
            [
                index
                for index, value in enumerate(inputs)
                if len(value.sequence.packet_lengths) >= 4
            ],
            long_model,
        ),
    ):
        if not indices:
            continue
        encoded = [encode_temporal_input(inputs[index]) for index in indices]
        bundle = TensorDatasetBundle(
            sequence=np.stack([item[0] for item in encoded]).astype(
                np.float32
            ),
            mask=np.stack([item[1] for item in encoded]).astype(np.float32),
            static=np.empty((len(indices), 0), dtype=np.float32),
            labels=np.zeros(len(indices), dtype=np.int8),
            sample_ids=[str(index) for index in indices],
        )
        logits, embeddings = _predict(
            wrapper.model,
            bundle,
            str(wrapper.device),
            batch_size,
        )
        probabilities = 1 / (
            1 + np.exp(-logits / max(wrapper.temperature, 1e-6))
        )
        if regime in gates:
            benign_p, malicious_p = gates[regime].p_values_many(embeddings)
        else:
            benign_p = np.ones(len(indices), dtype=np.float32)
            malicious_p = np.ones(len(indices), dtype=np.float32)
        policy = wrapper.metadata["decision_policy"]
        for local, original in enumerate(indices):
            probability = float(probabilities[local])
            prediction_set = []
            if benign_p[local] >= 0.01:
                prediction_set.append("benign")
            if malicious_p[local] >= 0.01:
                prediction_set.append("malicious")
            hard = (
                benign_p[local] < 0.01
                and malicious_p[local] < 0.01
            )
            result[original] = {
                "probability": probability,
                "accepted_class": _accepted_class(probability, policy),
                "uncertainty": max(
                    0.08,
                    min(0.95, _entropy(probability)),
                ),
                "embedding": embeddings[local].astype(np.float32),
                "regime": regime,
                "benign_p_value": float(benign_p[local]),
                "malicious_p_value": float(malicious_p[local]),
                "prediction_set": prediction_set,
                "hard_ood": bool(hard),
            }
    if any(item is None for item in result):
        raise RuntimeError("temporal orbit prediction routing is incomplete")
    return [item for item in result if item is not None]


def _support(prediction: dict[str, Any], class_name: str) -> float:
    certainty = max(0.0, 1 - float(prediction["uncertainty"]))
    probability = float(prediction["probability"])
    return (
        probability * certainty
        if class_name == "malicious"
        else (1 - probability) * certainty
    )


def orbit_feature_vector(
    predictions: list[dict[str, Any]],
) -> tuple[np.ndarray, str | None, list[float]]:
    if len(predictions) != len(ORBIT_KINDS):
        raise ValueError("orbit features require five predictions")
    clean_class = predictions[0]["accepted_class"]
    classes = [item["accepted_class"] for item in predictions]
    consistent = (
        clean_class is not None
        and all(value == clean_class for value in classes)
    )
    supports = (
        [_support(item, clean_class) for item in predictions]
        if clean_class is not None
        else [0.0] * len(predictions)
    )
    class_p_values = []
    for item in predictions:
        class_p_values.append(
            float(
                item[
                    "malicious_p_value"
                    if clean_class == "malicious"
                    else "benign_p_value"
                ]
            )
            if clean_class is not None
            else 0.0
        )
    clean_regime = predictions[0]["regime"]
    route_crossings = sum(
        item["regime"] != clean_regime for item in predictions[1:]
    )
    features = np.asarray(
        [
            float(predictions[0]["probability"]),
            float(sum(value == clean_class for value in classes) / 5)
            if clean_class is not None
            else 0.0,
            min(supports),
            float(np.mean(supports)),
            max(supports),
            max(supports) - min(supports),
            max(float(item["uncertainty"]) for item in predictions),
            min(class_p_values),
            float(sum(bool(item["hard_ood"]) for item in predictions)),
            float(route_crossings),
        ],
        dtype=np.float32,
    )
    return features, clean_class if consistent else None, supports


class OrbitReliabilityCalibrator:
    name = "OrbitReliabilityCalibrator"
    version = "2.5"

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        metadata_path = root / "metadata.json"
        model_path = root / "lower.joblib"
        if not metadata_path.exists() or not model_path.exists():
            raise ValueError(f"incomplete orbit reliability artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            self.metadata.get("policy") != "orbit_reliability_v2_5"
            or tuple(self.metadata.get("feature_names", []))
            != ORBIT_RELIABILITY_FEATURE_NAMES
        ):
            raise ValueError("orbit reliability metadata mismatch")
        if _sha256(model_path) != self.metadata.get("model_sha256"):
            raise ValueError("orbit reliability checksum mismatch")
        self.model = joblib.load(model_path)

    def lower_support(self, features: np.ndarray) -> float:
        value = float(
            self.model.predict(
                np.asarray(features, dtype=np.float32).reshape(1, -1)
            )[0]
        )
        return max(0.0, min(1.0, value))


def fit_orbit_reliability(
    records: Iterable[FlowRecord],
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    conformal_dir: str | Path,
    max_per_bucket_label: int = 5000,
    batch_size: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    if max_per_bucket_label != 5000:
        raise ValueError(
            "v2.5 freezes max_per_bucket_label at 5000"
        )
    selected = select_orbit_records(
        records,
        max_per_bucket_label=max_per_bucket_label,
    )
    if len(selected) < 100:
        raise ValueError("orbit reliability training data are insufficient")
    inputs: list[DetectorInput] = []
    labels: list[str] = []
    for record in selected:
        orbit = orbit_records(record)
        inputs.extend(_safe_input(item) for item in orbit)
        labels.append(str(record.labels["binary"]).lower())
    predictions = batch_temporal_predictions(
        inputs,
        model_dir=model_dir,
        conformal_dir=conformal_dir,
        batch_size=batch_size,
    )
    x: list[np.ndarray] = []
    y: list[float] = []
    for index, truth in enumerate(labels):
        orbit = predictions[index * 5 : index * 5 + 5]
        features, consistent_class, supports = orbit_feature_vector(orbit)
        correct = (
            consistent_class == truth
            and not any(bool(item["hard_ood"]) for item in orbit)
        )
        x.append(features)
        y.append(min(supports) if correct else 0.0)
    estimator = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.1,
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(np.stack(x), np.asarray(y, dtype=np.float32))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "lower.joblib"
    joblib.dump(estimator, model_path)
    metadata = {
        "schema_version": "1.0",
        "policy": "orbit_reliability_v2_5",
        "feature_names": list(ORBIT_RELIABILITY_FEATURE_NAMES),
        "target": (
            "minimum true-class orbit support; zero on any class error"
        ),
        "quantile": 0.1,
        "seed": seed,
        "training_split": "USTC train only",
        "training_record_count": len(selected),
        "training_orbit_prediction_count": len(predictions),
        "max_per_packet_bucket_and_label": max_per_bucket_label,
        "runtime_label_features": False,
        "runtime_identifier_features": False,
        "runtime_provenance_features": False,
        "runtime_context_features": False,
        "automatic_retraining": False,
        "model_sha256": _sha256(model_path),
    }
    _dump(root / "metadata.json", metadata)
    return metadata


def fit_conformal_v34_orbit(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    reference_per_bucket_label: int = 1024,
    calibration_per_bucket_label: int = 1000,
    batch_size: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    input_path = root / "flows"
    manifest = root / "splits" / "split-manifest.json"
    reference_records = select_orbit_records(
        iter_split_records(input_path, manifest, "train"),
        max_per_bucket_label=reference_per_bucket_label,
    )
    calibration_records = validation_calibration_records(
        iter_split_records(input_path, manifest, "validation"),
        max_per_bucket_label=calibration_per_bucket_label,
    )
    reference_inputs: list[DetectorInput] = []
    reference_labels: list[int] = []
    for record in reference_records:
        orbit = orbit_records(record)
        reference_inputs.extend(_safe_input(item) for item in orbit)
        reference_labels.extend(
            [int(record.labels["binary"] == "malicious")] * len(orbit)
        )
    calibration_inputs: list[DetectorInput] = []
    calibration_labels: list[int] = []
    for record in calibration_records:
        orbit = orbit_records(record)
        calibration_inputs.extend(_safe_input(item) for item in orbit)
        calibration_labels.extend(
            [int(record.labels["binary"] == "malicious")] * len(orbit)
        )
    reference_predictions = batch_temporal_predictions(
        reference_inputs,
        model_dir=model_dir,
        batch_size=batch_size,
    )
    calibration_predictions = batch_temporal_predictions(
        calibration_inputs,
        model_dir=model_dir,
        batch_size=batch_size,
    )
    reference_labels_array = np.asarray(reference_labels, dtype=np.int8)
    calibration_labels_array = np.asarray(calibration_labels, dtype=np.int8)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "policy": "conformal_v3_4_orbit",
        "alpha": 0.01,
        "hard_p_value": 0.01,
        "strength": 0.2,
        "orbit_seed": 42,
        "reference_split": "USTC train",
        "calibration_split": "USTC validation calibration partition",
        "selection_partition_used": False,
        "test_or_external_used": False,
        "regimes": {},
    }
    for regime in ("short", "long"):
        reference_mask = np.asarray(
            [
                item["regime"] == regime
                for item in reference_predictions
            ],
            dtype=bool,
        )
        calibration_mask = np.asarray(
            [
                item["regime"] == regime
                for item in calibration_predictions
            ],
            dtype=bool,
        )
        metadata = fit_orbit_conformal_v34(
            np.stack(
                [
                    item["embedding"]
                    for item, selected in zip(
                        reference_predictions,
                        reference_mask,
                        strict=True,
                    )
                    if selected
                ]
            ),
            reference_labels_array[reference_mask],
            np.stack(
                [
                    item["embedding"]
                    for item, selected in zip(
                        calibration_predictions,
                        calibration_mask,
                        strict=True,
                    )
                    if selected
                ]
            ),
            calibration_labels_array[calibration_mask],
            output_dir,
            regime=regime,
            seed=seed,
        )
        summary["regimes"][regime] = metadata
    _dump(Path(output_dir) / "metadata.json", summary)
    return summary
