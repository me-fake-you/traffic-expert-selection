from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .features import (
    FEATURE_SCHEMA_VERSION,
    feature_extractor,
    feature_names,
)
from .io import iter_split_records
from .models import MODEL_ARTIFACT_SCHEMA_VERSION, SigmoidCalibrator
from .schemas import DetectorInput, FlowRecord


SUPPORTED_TRAINING_AGENTS = ("stats", "temporal")


@dataclass(slots=True)
class FeatureMatrix:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: list[str]


def _group_key(record: FlowRecord, time_block_seconds: int) -> str:
    source_file = record.provenance.get("source_file")
    capture_start = record.provenance.get("capture_start_epoch")
    session_id = record.context.get("session_id")
    if source_file and isinstance(capture_start, (int, float)):
        return (
            f"{source_file}::timeblock="
            f"{int(float(capture_start) // time_block_seconds)}"
        )
    if session_id:
        return f"session={session_id}"
    if source_file:
        return f"source={source_file}"
    return f"sample={record.sample_id}"


def _group_hash(record: FlowRecord, time_block_seconds: int) -> np.uint64:
    digest = hashlib.sha256(
        _group_key(record, time_block_seconds).encode("utf-8")
    ).digest()
    return np.uint64(int.from_bytes(digest[:8], "big"))


def _safe_detector_input(record: FlowRecord) -> DetectorInput:
    # The extractor accepts DetectorInput rather than FlowRecord. Labels,
    # provenance, identifiers, and context therefore cannot enter features.
    return DetectorInput(stats=record.stats, sequence=record.sequence)


def load_feature_matrix(
    input_path: str | Path,
    split_manifest: str | Path,
    split: str,
    agent: str,
) -> FeatureMatrix:
    if agent not in SUPPORTED_TRAINING_AGENTS:
        raise ValueError(f"unsupported training agent: {agent}")
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    capacity = len(manifest["assignments"][split])
    names = feature_names(agent)
    x = np.empty((capacity, len(names)), dtype=np.float32)
    y = np.empty(capacity, dtype=np.int8)
    groups = np.empty(capacity, dtype=np.uint64)
    sample_ids: list[str] = []
    cursor = 0
    extract = feature_extractor(agent)
    time_block_seconds = int(manifest.get("time_block_seconds", 300))

    for record in iter_split_records(input_path, split_manifest, split):
        label = str(record.labels.get("binary", "")).lower()
        if label not in {"benign", "malicious"}:
            continue
        if agent == "temporal" and len(record.sequence.packet_lengths) < 4:
            continue
        x[cursor] = extract(_safe_detector_input(record))
        y[cursor] = int(label == "malicious")
        groups[cursor] = _group_hash(record, time_block_seconds)
        sample_ids.append(record.sample_id)
        cursor += 1
    return FeatureMatrix(
        x=x[:cursor],
        y=y[:cursor],
        groups=groups[:cursor],
        sample_ids=sample_ids,
    )


def _validation_partition(
    labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        label_groups = sorted(
            int(value) for value in np.unique(groups[labels == label])
        )
        calibration_groups = set(label_groups[::2])
        calibration |= (labels == label) & np.isin(
            groups, list(calibration_groups)
        )
    policy = ~calibration
    if len(np.unique(labels[calibration])) < 2 or len(np.unique(labels[policy])) < 2:
        raise ValueError(
            "validation split cannot be divided into two group-disjoint "
            "binary-label subsets"
        )
    return calibration, policy


def _candidate_models(seed: int) -> list[tuple[str, str, Any]]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    candidates: list[tuple[str, str, Any]] = []
    for c_value in (0.1, 1.0, 10.0):
        candidates.append(
            (
                f"logistic_c{c_value:g}",
                "logistic_regression",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c_value,
                                class_weight="balanced",
                                max_iter=1000,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    for name, leaves in (("hist_small", 15), ("hist_medium", 31)):
        candidates.append(
            (
                name,
                "hist_gradient_boosting",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "model",
                            HistGradientBoostingClassifier(
                                learning_rate=0.05,
                                max_iter=200,
                                max_leaf_nodes=leaves,
                                min_samples_leaf=30,
                                l2_regularization=1.0,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    return candidates


def _cross_validate_candidates(
    matrix: FeatureMatrix,
    *,
    seed: int,
    cv_splits: int,
) -> list[dict[str, Any]]:
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    group_counts = [
        len(np.unique(matrix.groups[matrix.y == label])) for label in (0, 1)
    ]
    folds = min(cv_splits, *group_counts)
    if folds < 2:
        raise ValueError("at least two groups per class are required for training")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    results: list[dict[str, Any]] = []
    for name, family, candidate in _candidate_models(seed):
        fold_metrics: list[dict[str, float]] = []
        for train_index, test_index in splitter.split(
            matrix.x, matrix.y, matrix.groups
        ):
            estimator = clone(candidate)
            estimator.fit(matrix.x[train_index], matrix.y[train_index])
            probabilities = estimator.predict_proba(matrix.x[test_index])[:, 1]
            predictions = (probabilities >= 0.5).astype(np.int8)
            fold_metrics.append(
                {
                    "PR_AUC": float(
                        average_precision_score(matrix.y[test_index], probabilities)
                    ),
                    "macro_f1": float(
                        f1_score(
                            matrix.y[test_index],
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
        results.append(
            {
                "candidate": name,
                "family": family,
                "folds": fold_metrics,
                "mean_PR_AUC": float(
                    np.mean([item["PR_AUC"] for item in fold_metrics])
                ),
                "mean_macro_f1": float(
                    np.mean([item["macro_f1"] for item in fold_metrics])
                ),
            }
        )
    return sorted(
        results,
        key=lambda item: (item["mean_PR_AUC"], item["mean_macro_f1"]),
        reverse=True,
    )


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (probabilities >= low) & (
            (probabilities < high)
            | ((index == bins - 1) & (probabilities == 1))
        )
        if not np.any(mask):
            continue
        result += float(np.mean(mask)) * abs(
            float(np.mean(labels[mask])) - float(np.mean(probabilities[mask]))
        )
    return result


def _probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss

    return {
        "PR_AUC": float(average_precision_score(labels, probabilities)),
        "Brier_score": float(brier_score_loss(labels, probabilities)),
        "ECE": _ece(labels, probabilities),
    }


def _tune_decision_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_selective_risk: float = 0.10,
    max_benign_fpr: float = 0.05,
) -> dict[str, Any]:
    benign_thresholds = np.arange(0.05, 0.451, 0.02)
    malicious_thresholds = np.arange(0.55, 0.951, 0.02)
    benign_total = max(1, int(np.sum(labels == 0)))
    candidates: list[dict[str, float | bool]] = []
    for benign_max in benign_thresholds:
        for malicious_min in malicious_thresholds:
            covered = (probabilities <= benign_max) | (
                probabilities >= malicious_min
            )
            covered_count = int(np.sum(covered))
            if not covered_count:
                continue
            predicted = (probabilities >= malicious_min).astype(np.int8)
            errors = int(np.sum(predicted[covered] != labels[covered]))
            false_positives = int(
                np.sum((labels == 0) & (probabilities >= malicious_min))
            )
            true_positives = int(
                np.sum((labels == 1) & (probabilities >= malicious_min))
            )
            malicious_total = max(1, int(np.sum(labels == 1)))
            selective_risk = errors / covered_count
            fpr = false_positives / benign_total
            candidates.append(
                {
                    "benign_max_probability": float(benign_max),
                    "malicious_min_probability": float(malicious_min),
                    "coverage": covered_count / len(labels),
                    "selective_risk": selective_risk,
                    "benign_false_positive_rate": fpr,
                    "malicious_recall": true_positives / malicious_total,
                    "constraints_satisfied": selective_risk
                    <= max_selective_risk
                    and fpr <= max_benign_fpr,
                }
            )
    feasible = [item for item in candidates if item["constraints_satisfied"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda item: (
                item["coverage"],
                item["malicious_recall"],
                -item["selective_risk"],
            ),
        )
    elif candidates:
        chosen = min(
            candidates,
            key=lambda item: (
                item["selective_risk"],
                item["benign_false_positive_rate"],
                -item["coverage"],
            ),
        )
    else:
        chosen = {
            "benign_max_probability": 0.49,
            "malicious_min_probability": 0.51,
            "coverage": 0.0,
            "selective_risk": 0.0,
            "benign_false_positive_rate": 0.0,
            "malicious_recall": 0.0,
            "constraints_satisfied": False,
        }
    return {
        **chosen,
        "max_selective_risk": max_selective_risk,
        "max_benign_false_positive_rate": max_benign_fpr,
    }


def _coverage_risk_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    targets: Iterable[float] = (0.01, 0.05, 0.10),
) -> list[dict[str, float | None]]:
    confidence = np.abs(probabilities - 0.5)
    order = np.argsort(-confidence)
    sorted_labels = labels[order]
    sorted_predictions = (probabilities[order] >= 0.5).astype(np.int8)
    cumulative_errors = np.cumsum(sorted_predictions != sorted_labels)
    result: list[dict[str, float | None]] = []
    for target in targets:
        risks = cumulative_errors / np.arange(1, len(labels) + 1)
        eligible = np.flatnonzero(risks <= target)
        if len(eligible):
            count = int(eligible[-1] + 1)
            result.append(
                {
                    "target_risk": target,
                    "coverage": count / len(labels),
                    "observed_risk": float(risks[count - 1]),
                    "minimum_confidence_margin": float(confidence[order[count - 1]]),
                }
            )
        else:
            result.append(
                {
                    "target_risk": target,
                    "coverage": 0.0,
                    "observed_risk": None,
                    "minimum_confidence_margin": None,
                }
            )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_detector(
    *,
    agent: str,
    train: FeatureMatrix,
    validation: FeatureMatrix | None = None,
    calibration: FeatureMatrix | None = None,
    policy: FeatureMatrix | None = None,
    output_dir: str | Path,
    seed: int,
    cv_splits: int,
    split_manifest_path: str | Path,
    extraction_manifest_path: str | Path | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from sklearn.base import clone

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cv_results = _cross_validate_candidates(
        train, seed=seed, cv_splits=cv_splits
    )
    candidates = {name: (family, model) for name, family, model in _candidate_models(seed)}
    family_winners: list[str] = []
    for family in ("logistic_regression", "hist_gradient_boosting"):
        family_winners.append(
            next(item["candidate"] for item in cv_results if item["family"] == family)
        )

    if calibration is not None or policy is not None:
        if calibration is None or policy is None:
            raise ValueError("calibration and policy matrices must be supplied together")
        calibration_x = calibration.x
        calibration_y = calibration.y
        calibration_ids = calibration.sample_ids
        policy_x = policy.x
        policy_y = policy.y
        policy_ids = policy.sample_ids
    else:
        if validation is None:
            raise ValueError("validation or explicit calibration/policy matrices required")
        calibration_mask, policy_mask = _validation_partition(
            validation.y, validation.groups
        )
        calibration_x = validation.x[calibration_mask]
        calibration_y = validation.y[calibration_mask]
        calibration_ids = list(
            np.asarray(validation.sample_ids, dtype=object)[calibration_mask]
        )
        policy_x = validation.x[policy_mask]
        policy_y = validation.y[policy_mask]
        policy_ids = list(
            np.asarray(validation.sample_ids, dtype=object)[policy_mask]
        )
    finalist_results: list[dict[str, Any]] = []
    fitted: dict[str, tuple[Any, SigmoidCalibrator, np.ndarray]] = {}
    for name in family_winners:
        family, candidate = candidates[name]
        estimator = clone(candidate)
        estimator.fit(train.x, train.y)
        calibration_raw = estimator.predict_proba(calibration_x)[:, 1]
        calibrator = SigmoidCalibrator().fit(
            calibration_raw, calibration_y
        )
        policy_raw = estimator.predict_proba(policy_x)[:, 1]
        policy_probabilities = calibrator.predict(policy_raw)
        metrics = _probability_metrics(policy_y, policy_probabilities)
        finalist_results.append(
            {"candidate": name, "family": family, **metrics}
        )
        fitted[name] = (estimator, calibrator, policy_probabilities)

    selected = min(
        finalist_results,
        key=lambda item: (
            item["Brier_score"],
            item["ECE"],
            -item["PR_AUC"],
        ),
    )
    selected_name = selected["candidate"]
    estimator, calibrator, policy_probabilities = fitted[selected_name]
    decision_policy = _tune_decision_policy(
        policy_y, policy_probabilities
    )
    curve = _coverage_risk_curve(policy_y, policy_probabilities)
    calibration_quality = max(0.5, min(1.0, 1 - selected["ECE"]))

    joblib.dump(
        {"estimator": estimator, "calibrator": calibrator},
        target / "model.joblib",
        compress=3,
    )
    model_sha256 = _sha256_file(target / "model.joblib")
    split_manifest_file = Path(split_manifest_path)
    extraction_manifest_file = (
        Path(extraction_manifest_path) if extraction_manifest_path else None
    )
    metadata = {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "agent": agent,
        "agent_class": (
            "StatsDetectorAgent"
            if agent == "stats"
            else "TemporalBehaviorAgent"
        ),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(feature_names(agent)),
        "selected_candidate": selected_name,
        "selected_family": selected["family"],
        "calibration_method": "platt_sigmoid",
        "calibration_quality": calibration_quality,
        "decision_policy": decision_policy,
        "model_sha256": model_sha256,
        "seed": seed,
        "train_sample_count": int(len(train.y)),
        "calibration_sample_count": int(len(calibration_y)),
        "policy_sample_count": int(len(policy_y)),
        "split_manifest_sha256": _sha256_file(split_manifest_file),
        "extraction_manifest_sha256": (
            _sha256_file(extraction_manifest_file)
            if extraction_manifest_file and extraction_manifest_file.exists()
            else None
        ),
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
    _write_json(
        target / "feature_schema.json",
        {
            "schema_version": "1.0",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "agent": agent,
            "input_schema": "DetectorInput",
            "feature_names": list(feature_names(agent)),
            "forbidden_input_groups": [
                "labels",
                "provenance",
                "trace_id",
                "sample_id",
            ],
        },
    )
    calibration_report = {
        "schema_version": "1.0",
        "agent": agent,
        "candidate_cross_validation": cv_results,
        "calibrated_finalists": finalist_results,
        "selected_candidate": selected_name,
        "selected_metrics": {
            key: selected[key] for key in ("PR_AUC", "Brier_score", "ECE")
        },
        "decision_policy": decision_policy,
        "coverage_risk_curve": curve,
    }
    _write_json(target / "calibration_report.json", calibration_report)
    with (target / "validation_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "schema_version",
                "sample_id",
                "true_label",
                "malicious_probability",
                "accepted_class",
            ],
        )
        writer.writeheader()
        benign_max = decision_policy["benign_max_probability"]
        malicious_min = decision_policy["malicious_min_probability"]
        for sample_id, label, probability in zip(
            policy_ids,
            policy_y,
            policy_probabilities,
            strict=True,
        ):
            accepted = (
                "benign"
                if probability <= benign_max
                else "malicious"
                if probability >= malicious_min
                else "reject"
            )
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "sample_id": sample_id,
                    "true_label": "malicious" if label else "benign",
                    "malicious_probability": float(probability),
                    "accepted_class": accepted,
                }
            )
    return metadata


def train_detectors(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    agents: Iterable[str] = SUPPORTED_TRAINING_AGENTS,
    seed: int = 42,
    cv_splits: int = 3,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    flows = dataset / "flows"
    split_manifest = dataset / "splits" / "split-manifest.json"
    extraction_manifest = dataset / "manifests" / "extraction_manifest.json"
    if not flows.is_dir():
        raise FileNotFoundError(f"flow directory not found: {flows}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"split manifest not found: {split_manifest}")

    requested = tuple(dict.fromkeys(agents))
    unsupported = set(requested) - set(SUPPORTED_TRAINING_AGENTS)
    if unsupported:
        raise ValueError(f"unsupported training agents: {sorted(unsupported)}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for agent in requested:
        train = load_feature_matrix(flows, split_manifest, "train", agent)
        validation = load_feature_matrix(
            flows, split_manifest, "validation", agent
        )
        summaries[agent] = train_detector(
            agent=agent,
            train=train,
            validation=validation,
            output_dir=target / agent,
            seed=seed,
            cv_splits=cv_splits,
            split_manifest_path=split_manifest,
            extraction_manifest_path=extraction_manifest,
        )
    summary = {
        "schema_version": "1.0",
        "dataset_dir": str(dataset.resolve()),
        "output_dir": str(target.resolve()),
        "agents": list(requested),
        "seed": seed,
        "cv_splits": cv_splits,
        "models": summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(target / "training_summary.json", summary)
    return summary
