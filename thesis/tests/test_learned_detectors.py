import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from mad_etd.detectors import StatsDetectorAgent, TemporalBehaviorAgent
from mad_etd.engine import build_default_engine
from mad_etd.features import (
    FEATURE_SCHEMA_VERSION,
    STATS_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    extract_stats_features,
    extract_temporal_features,
)
from mad_etd.models import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    LearnedDetectorModel,
    ModelArtifactError,
    SigmoidCalibrator,
)
from mad_etd.schemas import AgentEvidence, DetectorInput, SequenceFeatures
from mad_etd.training import FeatureMatrix, _validation_partition, train_detector


def _write_artifact(path, agent, features):
    path.mkdir(parents=True)
    x = np.vstack([features, features + 0.1, features + 1, features + 1.1])
    y = np.asarray([0, 0, 1, 1])
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("model", LogisticRegression(random_state=42)),
        ]
    ).fit(x, y)
    raw = estimator.predict_proba(x)[:, 1]
    calibrator = SigmoidCalibrator().fit(raw, y)
    joblib.dump(
        {"estimator": estimator, "calibrator": calibrator},
        path / "model.joblib",
    )
    names = (
        STATS_FEATURE_NAMES if agent == "stats" else TEMPORAL_FEATURE_NAMES
    )
    metadata = {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "agent": agent,
        "feature_names": list(names),
        "calibration_quality": 0.9,
        "decision_policy": {
            "benign_max_probability": 0.35,
            "malicious_min_probability": 0.65,
        },
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_stats_features_are_deterministic_and_fixed_width(benign_detector_input):
    first = extract_stats_features(benign_detector_input)
    second = extract_stats_features(benign_detector_input.model_copy(deep=True))

    np.testing.assert_array_equal(first, second)
    assert len(first) == len(STATS_FEATURE_NAMES)


def test_feature_extractors_accept_only_detector_input(benign_flow):
    with pytest.raises(TypeError, match="DetectorInput"):
        extract_stats_features(benign_flow)


def test_stats_features_ignore_context_and_identifiers(benign_detector_input):
    first = extract_stats_features(benign_detector_input)
    changed = benign_detector_input.model_copy(deep=True)
    changed.context = {"src_ip": "label-proxy", "session_id": "malicious"}

    np.testing.assert_array_equal(first, extract_stats_features(changed))


def test_temporal_features_include_periodicity_signal():
    regular = DetectorInput(
        sequence=SequenceFeatures(
            packet_lengths=[100] * 8,
            directions=[1, -1] * 4,
            iats=[5.0] * 7,
        )
    )
    noisy = regular.model_copy(deep=True)
    noisy.sequence.iats = [0.1, 9, 0.2, 8, 0.3, 7, 0.4]

    regular_features = extract_temporal_features(regular)
    noisy_features = extract_temporal_features(noisy)

    periodicity_index = TEMPORAL_FEATURE_NAMES.index("iat_periodicity_score")
    assert regular_features[periodicity_index] > noisy_features[periodicity_index]
    burst_mean_index = TEMPORAL_FEATURE_NAMES.index("burst_mean")
    burst_max_index = TEMPORAL_FEATURE_NAMES.index("burst_max")
    assert regular_features[burst_mean_index] == 0
    assert regular_features[burst_max_index] == 0


def test_learned_model_rejects_missing_artifact(tmp_path):
    with pytest.raises(ModelArtifactError, match="incomplete"):
        LearnedDetectorModel(tmp_path / "missing", expected_agent="stats")


def test_learned_model_rejects_agent_schema_mismatch(
    tmp_path, benign_detector_input
):
    features = extract_stats_features(benign_detector_input)
    _write_artifact(tmp_path / "stats", "stats", features)

    with pytest.raises(ModelArtifactError, match="agent mismatch"):
        LearnedDetectorModel(tmp_path / "stats", expected_agent="temporal")


def test_learned_model_rejects_checksum_mismatch(
    tmp_path, benign_detector_input
):
    artifact = tmp_path / "stats"
    _write_artifact(
        artifact, "stats", extract_stats_features(benign_detector_input)
    )
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="checksum"):
        LearnedDetectorModel(artifact, expected_agent="stats")


def test_learned_stats_detector_returns_valid_calibrated_evidence(
    tmp_path, benign_detector_input
):
    features = extract_stats_features(benign_detector_input)
    _write_artifact(tmp_path / "stats", "stats", features)

    result = StatsDetectorAgent(
        backend="learned", model_dir=tmp_path / "stats"
    ).analyze(benign_detector_input)

    assert AgentEvidence.model_validate(result.model_dump()) == result
    assert result.agent_version == "2.1"
    assert result.calibration_quality == 0.9
    assert "learned statistical model" in result.evidence[0]


def test_learned_temporal_detector_still_abstains_on_short_sequence(
    tmp_path, benign_detector_input
):
    temporal_input = DetectorInput(
        sequence=SequenceFeatures(
            packet_lengths=[100] * 8,
            directions=[1, -1] * 4,
            iats=[1.0] * 7,
        )
    )
    _write_artifact(
        tmp_path / "temporal",
        "temporal",
        extract_temporal_features(temporal_input),
    )
    detector = TemporalBehaviorAgent(
        backend="learned", model_dir=tmp_path / "temporal"
    )

    result = detector.analyze(
        DetectorInput(
            sequence=SequenceFeatures(
                packet_lengths=[100, 120, 80],
                directions=[1, -1, 1],
            )
        )
    )

    assert result.abstained is True
    assert result.uncertainty == 1


def test_learned_engine_requires_explicit_model_directory():
    with pytest.raises(ValueError, match="model_dir"):
        build_default_engine(detector_backend="learned")


def test_validation_partition_keeps_groups_disjoint_and_both_labels():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    groups = np.asarray([10, 10, 11, 11, 20, 20, 21, 21], dtype=np.uint64)

    calibration, policy = _validation_partition(labels, groups)

    assert set(groups[calibration]).isdisjoint(set(groups[policy]))
    assert set(labels[calibration]) == {0, 1}
    assert set(labels[policy]) == {0, 1}


def test_train_detector_writes_versioned_fixed_artifacts(tmp_path):
    rng = np.random.default_rng(42)
    train_labels = np.asarray([0] * 20 + [1] * 20, dtype=np.int8)
    validation_labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int8)
    train = FeatureMatrix(
        x=np.vstack(
            [
                rng.normal(0, 0.2, (20, len(STATS_FEATURE_NAMES))),
                rng.normal(1, 0.2, (20, len(STATS_FEATURE_NAMES))),
            ]
        ).astype(np.float32),
        y=train_labels,
        groups=np.asarray(
            [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
            + [11] * 5 + [12] * 5 + [13] * 5 + [14] * 5,
            dtype=np.uint64,
        ),
        sample_ids=[f"train-{index}" for index in range(40)],
    )
    validation = FeatureMatrix(
        x=np.vstack(
            [
                rng.normal(0, 0.2, (12, len(STATS_FEATURE_NAMES))),
                rng.normal(1, 0.2, (12, len(STATS_FEATURE_NAMES))),
            ]
        ).astype(np.float32),
        y=validation_labels,
        groups=np.asarray(
            [101] * 3 + [102] * 3 + [103] * 3 + [104] * 3
            + [201] * 3 + [202] * 3 + [203] * 3 + [204] * 3,
            dtype=np.uint64,
        ),
        sample_ids=[f"validation-{index}" for index in range(24)],
    )
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text('{"version":"1.0"}', encoding="utf-8")
    output = tmp_path / "stats"

    metadata = train_detector(
        agent="stats",
        train=train,
        validation=validation,
        output_dir=output,
        seed=42,
        cv_splits=2,
        split_manifest_path=split_manifest,
        extraction_manifest_path=None,
    )

    assert {
        "model.joblib",
        "metadata.json",
        "feature_schema.json",
        "calibration_report.json",
        "validation_predictions.csv",
    } == {path.name for path in output.iterdir()}
    assert metadata["schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["decision_policy"]["coverage"] > 0
    assert metadata["runtime_versions"]["scikit_learn"]
    assert len(metadata["model_sha256"]) == 64
