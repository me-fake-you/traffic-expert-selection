from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from .deep_models import (
    DEEP_V2_ARTIFACT_SCHEMA_VERSION,
    TCNConfig,
    build_tcn,
    config_as_json,
    encode_tls_input,
)
from .io import iter_flow_records
from .models import SigmoidCalibrator
from .schemas import DetectorInput, FlowRecord
from .v2_training import _ece, _predict, _select_policy, _temperature_scale


TLS_V4_VARIANTS = ("records_only", "records_handshake")
TLS_V4_SEEDS = (42, 43, 44)
HGB_FEATURE_NAMES = (
    "record_count",
    "total_absolute_bytes",
    "mean_absolute_length",
    "std_absolute_length",
    "min_absolute_length",
    "max_absolute_length",
    "outbound_record_ratio",
    "outbound_byte_ratio",
    "direction_transition_ratio",
    "first_record_log_length",
    "second_record_log_length",
    "last_record_log_length",
)


@dataclass(slots=True)
class TLSV4DatasetBundle:
    sequence: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    hgb: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    capture_ids: list[str]
    generators: list[str]
    resolvers: list[str]


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_input(record: FlowRecord) -> DetectorInput:
    return DetectorInput(
        tls={
            key: value
            for key, value in record.tls.items()
            if key
            in {
                "record_lengths",
                "tls_record_lengths",
                "server_version",
                "client_cipher_count",
                "client_extension_count",
                "server_extension_count",
                "alpn",
            }
        }
    )


def hgb_tls_features(detector_input: DetectorInput) -> np.ndarray:
    raw = detector_input.tls.get("record_lengths") or detector_input.tls.get(
        "tls_record_lengths"
    ) or []
    records = np.asarray([int(item) for item in raw if int(item) != 0], dtype=float)
    if len(records) < 2:
        return np.zeros(len(HGB_FEATURE_NAMES), dtype=np.float32)
    absolute = np.abs(records)
    outbound = records > 0
    transitions = np.count_nonzero(np.sign(records[1:]) != np.sign(records[:-1]))
    return np.asarray(
        [
            len(records),
            absolute.sum(),
            absolute.mean(),
            absolute.std(),
            absolute.min(),
            absolute.max(),
            outbound.mean(),
            absolute[outbound].sum() / max(absolute.sum(), 1.0),
            transitions / max(len(records) - 1, 1),
            np.sign(records[0]) * np.log1p(absolute[0]),
            np.sign(records[1]) * np.log1p(absolute[1]),
            np.sign(records[-1]) * np.log1p(absolute[-1]),
        ],
        dtype=np.float32,
    )


def _empty_bundle() -> TLSV4DatasetBundle:
    return TLSV4DatasetBundle(
        sequence=np.empty((0, 2, 64), dtype=np.float32),
        mask=np.empty((0, 64), dtype=np.float32),
        static=np.empty((0, 4), dtype=np.float32),
        hgb=np.empty((0, len(HGB_FEATURE_NAMES)), dtype=np.float32),
        labels=np.empty(0, dtype=np.int8),
        sample_ids=[],
        capture_ids=[],
        generators=[],
        resolvers=[],
    )


def load_tls_v4_dataset(
    dataset_dir: str | Path,
    sample_ids: Iterable[str],
) -> TLSV4DatasetBundle:
    selected = set(sample_ids)
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    static: list[np.ndarray] = []
    hgb: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    captures: list[str] = []
    generators: list[str] = []
    resolvers: list[str] = []
    for record in iter_flow_records(Path(dataset_dir) / "flows"):
        if record.sample_id not in selected:
            continue
        label = str(record.labels.get("binary", ""))
        records = record.tls.get("record_lengths") or []
        if label not in {"benign", "malicious"} or len(records) < 2:
            continue
        detector_input = _safe_input(record)
        sequence, mask, local_static = encode_tls_input(detector_input)
        sequences.append(sequence)
        masks.append(mask)
        static.append(local_static)
        hgb.append(hgb_tls_features(detector_input))
        labels.append(int(label == "malicious"))
        ids.append(record.sample_id)
        captures.append(str(record.provenance.get("capture_id", "")))
        generators.append(str(record.labels.get("generator") or ""))
        resolvers.append(str(record.labels.get("resolver") or ""))
    if not labels:
        return _empty_bundle()
    return TLSV4DatasetBundle(
        sequence=np.stack(sequences).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        static=np.stack(static).astype(np.float32),
        hgb=np.stack(hgb).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        sample_ids=ids,
        capture_ids=captures,
        generators=generators,
        resolvers=resolvers,
    )


def _macro_f1(
    labels: np.ndarray,
    probabilities: np.ndarray,
    policy: dict[str, Any],
) -> tuple[float, float, float]:
    benign_max = float(policy["benign_max_probability"])
    malicious_min = float(policy["malicious_min_probability"])
    predictions = np.full(len(probabilities), -1, dtype=np.int8)
    predictions[probabilities <= benign_max] = 0
    predictions[probabilities >= malicious_min] = 1
    covered = predictions >= 0
    scores = []
    for label in (0, 1):
        tp = np.sum((labels == label) & (predictions == label))
        fp = np.sum((labels != label) & (predictions == label))
        fn = np.sum(covered & (labels == label) & (predictions != label))
        scores.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
    error = (
        float(np.mean(predictions[covered] != labels[covered]))
        if covered.any()
        else 0.0
    )
    return float(np.mean(scores)), float(covered.mean()), error


def _train_hgb(
    train: TLSV4DatasetBundle,
    calibration: TLSV4DatasetBundle,
    selection: TLSV4DatasetBundle,
    output_dir: Path,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42,
    )
    estimator.fit(train.hgb, train.labels)
    calibrator = SigmoidCalibrator().fit(
        estimator.predict_proba(calibration.hgb)[:, 1],
        calibration.labels,
    )
    probabilities = calibrator.predict(
        estimator.predict_proba(selection.hgb)[:, 1]
    )
    policy = _select_policy(probabilities, selection.labels)
    macro_f1, coverage, selective_error = _macro_f1(
        selection.labels,
        probabilities,
        policy,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": estimator, "calibrator": calibrator},
        output_dir / "model.joblib",
        compress=3,
    )
    metadata = {
        "schema_version": "1.0",
        "agent": "tls",
        "backend": "hgb_tls_v4",
        "feature_names": list(HGB_FEATURE_NAMES),
        "decision_policy": policy,
        "calibration_quality": max(
            0.0,
            min(1.0, 1 - _ece(probabilities, selection.labels)),
        ),
        "model_sha256": _sha256(output_dir / "model.joblib"),
        "train_count": len(train.labels),
        "calibration_count": len(calibration.labels),
        "selection_count": len(selection.labels),
        "blocked_inputs": [
            "IP",
            "port",
            "SNI",
            "JA3",
            "tool",
            "browser",
            "resolver",
            "source_file",
        ],
    }
    _dump(output_dir / "metadata.json", metadata)
    report = {
        "model": "hgb_tls_v4",
        "selection_macro_f1": macro_f1,
        "selection_coverage": coverage,
        "selection_selective_error": selective_error,
        "selection_ece": _ece(probabilities, selection.labels),
        "decision_policy": policy,
    }
    _dump(output_dir / "selection_report.json", report)
    return report


def _tensor_bundle(
    bundle: TLSV4DatasetBundle,
    *,
    include_handshake: bool,
):
    from .v2_training import TensorDatasetBundle

    return TensorDatasetBundle(
        sequence=bundle.sequence,
        mask=bundle.mask,
        static=(
            bundle.static
            if include_handshake
            else np.empty((len(bundle.labels), 0), dtype=np.float32)
        ),
        labels=bundle.labels,
        sample_ids=bundle.sample_ids,
    )


def _train_tcn_variant(
    train: TLSV4DatasetBundle,
    calibration: TLSV4DatasetBundle,
    selection: TLSV4DatasetBundle,
    output_dir: Path,
    *,
    variant: str,
    seeds: Iterable[int],
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    include_handshake = variant == "records_handshake"
    train_tensor = _tensor_bundle(train, include_handshake=include_handshake)
    calibration_tensor = _tensor_bundle(
        calibration, include_handshake=include_handshake
    )
    selection_tensor = _tensor_bundle(
        selection, include_handshake=include_handshake
    )
    config = TCNConfig(
        input_channels=2,
        static_dim=4 if include_handshake else 0,
        hidden_channels=64,
        embedding_dim=128,
        dilations=(1, 2, 4, 8),
    )
    candidates = []
    indices = np.arange(len(train.labels))
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_tcn(config).to("cuda")
        positives = max(1, int(train.labels.sum()))
        negatives = max(1, len(train.labels) - positives)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device="cuda")
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler("cuda")
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = seed_dir / "training_checkpoint.pt"
        start_epoch = 0
        if checkpoint.exists():
            state = torch.load(
                checkpoint, map_location="cuda", weights_only=False
            )
            if state.get("seed") != seed or state.get("variant") != variant:
                raise ValueError("TLS v4 checkpoint contract changed")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["next_epoch"])
        for epoch in range(start_epoch, epochs):
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(indices)
            model.train()
            for start in range(0, len(indices), batch_size):
                current = indices[start : start + batch_size]
                sequence = torch.from_numpy(train_tensor.sequence[current]).to(
                    "cuda"
                )
                mask = torch.from_numpy(train_tensor.mask[current]).to("cuda")
                static = (
                    torch.from_numpy(train_tensor.static[current]).to("cuda")
                    if train_tensor.static.shape[1]
                    else None
                )
                labels = torch.from_numpy(
                    train_tensor.labels[current].astype(np.float32)
                ).to("cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    logits = model(sequence, mask, static)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.save(
                {
                    "seed": seed,
                    "variant": variant,
                    "next_epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint,
            )
            _dump(
                seed_dir / "training_progress.json",
                {
                    "seed": seed,
                    "variant": variant,
                    "completed_epochs": epoch + 1,
                    "expected_epochs": epochs,
                    "resumable": True,
                },
            )
        calibration_logits, _ = _predict(
            model, calibration_tensor, "cuda", batch_size
        )
        selection_logits, _ = _predict(
            model, selection_tensor, "cuda", batch_size
        )
        temperature = _temperature_scale(
            calibration_logits,
            calibration.labels,
        )
        probabilities = 1 / (
            1 + np.exp(-selection_logits / max(temperature, 1e-6))
        )
        policy = _select_policy(probabilities, selection.labels)
        macro_f1, coverage, selective_error = _macro_f1(
            selection.labels,
            probabilities,
            policy,
        )
        ece = _ece(probabilities, selection.labels)
        brier = float(np.mean((probabilities - selection.labels) ** 2))
        model_path = seed_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        metadata = {
            "schema_version": DEEP_V2_ARTIFACT_SCHEMA_VERSION,
            "agent": "tls",
            "backend": "deep_tls_v4",
            "variant": variant,
            "task_scope": "benign_doh_vs_malicious_doh",
            "architecture": "residual_tcn",
            "tcn_config": config_as_json(config),
            "temperature": temperature,
            "calibration_quality": max(0.0, min(1.0, 1 - ece)),
            "decision_policy": policy,
            "model_sha256": _sha256(model_path),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "mixed_precision": True,
            "embedding_dim": 128,
            "minimum_tls_records": 2,
            "blocked_inputs": [
                "IP",
                "port",
                "SNI",
                "JA3",
                "tool",
                "browser",
                "resolver",
                "source_file",
            ],
        }
        _dump(seed_dir / "metadata.json", metadata)
        report = {
            "seed": seed,
            "variant": variant,
            "selection_macro_f1": macro_f1,
            "selection_coverage": coverage,
            "selection_selective_error": selective_error,
            "selection_ece": ece,
            "selection_brier": brier,
            "decision_policy": policy,
        }
        _dump(seed_dir / "selection_report.json", report)
        candidates.append({"path": seed_dir.as_posix(), **report})
        checkpoint.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()
    selected = min(
        candidates,
        key=lambda item: (
            -item["selection_macro_f1"],
            item["selection_brier"],
            item["selection_ece"],
            item["seed"],
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(selected["path"])
    for name in ("model.pt", "metadata.json", "selection_report.json"):
        shutil.copy2(selected_path / name, output_dir / name)
    summary = {
        "variant": variant,
        "status": "trained",
        "selected_seed": selected["seed"],
        "candidates": candidates,
        "train_count": len(train.labels),
        "calibration_count": len(calibration.labels),
        "selection_count": len(selection.labels),
    }
    _dump(output_dir / "training_summary.json", summary)
    return summary


def train_tls_v4(
    dataset_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    output_dir: str | Path = "data/models/mad_etd_v4/tls",
    *,
    seeds: Iterable[int] = TLS_V4_SEEDS,
    epochs: int = 12,
    batch_size: int = 512,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TLS v4 training requires CUDA")
    dataset = Path(dataset_dir)
    manifest_path = dataset / "splits" / "split-manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("freeze-dohbrw-v4-splits must complete first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_used_for_selection") or not manifest.get("locked_test"):
        raise ValueError("TLS v4 requires an untouched locked test")
    train = load_tls_v4_dataset(dataset, manifest["assignments"]["train"])
    calibration = load_tls_v4_dataset(
        dataset,
        manifest["validation_partitions"]["calibration_sample_ids"],
    )
    selection = load_tls_v4_dataset(
        dataset,
        manifest["validation_partitions"]["selection_sample_ids"],
    )
    for name, bundle in (
        ("train", train),
        ("calibration", calibration),
        ("selection", selection),
    ):
        if len(bundle.labels) < 40 or set(np.unique(bundle.labels)) != {0, 1}:
            raise ValueError(f"TLS v4 {name} data are insufficient")
    root = Path(output_dir)
    hgb = _train_hgb(train, calibration, selection, root / "hgb")
    tcns = {
        variant: _train_tcn_variant(
            train,
            calibration,
            selection,
            root / variant,
            variant=variant,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
        )
        for variant in TLS_V4_VARIANTS
    }
    summary = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v4_tls_training",
        "status": "trained",
        "scope": "benign_doh_vs_malicious_doh",
        "hgb": hgb,
        "tcn": tcns,
        "split_manifest_sha256": _sha256(manifest_path),
        "locked_test_used": False,
        "external_used_for_selection": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(root / "training_summary.json", summary)
    return summary
