from __future__ import annotations

import hashlib
import heapq
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .deep_models import DeepDetectorModel, encode_temporal_input
from .io import iter_split_records
from .ood_v32 import fit_regime_conformal_v32
from .schemas import DetectorInput, FlowRecord
from .short_flow import (
    SHORT_FLOW_SCHEMA_VERSION,
    ShortFlowConfig,
    build_short_flow_encoder,
    config_as_json,
)
from .v2_training import (
    TensorDatasetBundle,
    _augment_batch,
    _ece,
    _partition_validation,
    _predict,
    _select_policy,
    _temperature_scale,
    load_v2_tensor_dataset,
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
        context={
            key: value
            for key, value in record.context.items()
            if key in {"transport", "protocols"}
        },
    )


@dataclass(slots=True)
class RegimeBundle:
    tensors: TensorDatasetBundle
    packet_counts: np.ndarray


def load_short_flow_dataset(
    input_path: str | Path,
    split_manifest: str | Path,
    split: str,
    *,
    minimum_packets: int = 1,
    maximum_packets: int = 4,
    limit: int | None = None,
) -> RegimeBundle:
    if not 1 <= minimum_packets <= maximum_packets:
        raise ValueError("invalid short-flow packet bounds")
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    packet_counts: list[int] = []
    reservoirs: dict[int, list[tuple[int, str, Any]]] | None = None
    capacities = None
    if limit is not None:
        capacities = {0: limit // 2, 1: limit - limit // 2}
        reservoirs = {0: [], 1: []}
    for record in iter_split_records(input_path, split_manifest, split):
        count = len(record.sequence.packet_lengths)
        label = str(record.labels.get("binary", "")).lower()
        if (
            count < minimum_packets
            or count > maximum_packets
            or label not in {"benign", "malicious"}
        ):
            continue
        sequence, mask, _ = encode_temporal_input(_safe_input(record))
        binary = int(label == "malicious")
        payload = (sequence, mask, binary, count)
        if reservoirs is not None and capacities is not None:
            score = int(
                hashlib.sha256(
                    record.sample_id.encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            item = (-score, record.sample_id, payload)
            current = reservoirs[binary]
            if len(current) < capacities[binary]:
                heapq.heappush(current, item)
            elif item[:2] > current[0][:2]:
                heapq.heapreplace(current, item)
            continue
        sequences.append(sequence)
        masks.append(mask)
        labels.append(binary)
        sample_ids.append(record.sample_id)
        packet_counts.append(count)
    if reservoirs is not None:
        selected = sorted(
            (
                (sample_id, payload)
                for values in reservoirs.values()
                for _, sample_id, payload in values
            ),
            key=lambda item: item[0],
        )
        for sample_id, (sequence, mask, binary, count) in selected:
            sequences.append(sequence)
            masks.append(mask)
            labels.append(binary)
            sample_ids.append(sample_id)
            packet_counts.append(count)
    tensors = TensorDatasetBundle(
        sequence=(
            np.stack(sequences).astype(np.float32)
            if sequences
            else np.empty((0, 4, 64), dtype=np.float32)
        ),
        mask=(
            np.stack(masks).astype(np.float32)
            if masks
            else np.empty((0, 64), dtype=np.float32)
        ),
        static=np.empty((len(labels), 0), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        sample_ids=sample_ids,
    )
    return RegimeBundle(
        tensors=tensors,
        packet_counts=np.asarray(packet_counts, dtype=np.int8),
    )


def audit_sequence_regimes(
    dataset_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_2_sequence_regime_audit",
        "dataset": "USTC-TFC2016",
        "splits": {},
        "extraction_failure_interpretation": False,
        "terminology": {
            "0_packets": "missing_sequence",
            "1_to_3_packets": "short_sequence_unsupported_v2",
            "4_plus_packets": "long_sequence_supported_v2",
        },
    }
    for split in ("train", "validation", "test"):
        counts: dict[str, int] = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4+": 0,
        }
        by_label: dict[str, dict[str, int]] = {}
        for record in iter_split_records(
            root / "flows",
            root / "splits" / "split-manifest.json",
            split,
        ):
            count = len(record.sequence.packet_lengths)
            bucket = str(count) if count < 4 else "4+"
            counts[bucket] += 1
            label = str(record.labels.get("binary", "unknown"))
            by_label.setdefault(
                label,
                {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0},
            )[bucket] += 1
        result["splits"][split] = {
            "counts": counts,
            "by_label": by_label,
            "short_1_to_3_count": (
                counts["1"] + counts["2"] + counts["3"]
            ),
        }
    result["interpretation"] = (
        "The extractor preserves every observed packet up to 64. "
        "One-to-three-packet flows are genuine short flows, not missing "
        "extraction output."
    )
    _dump(Path(output_path), result)
    return result


def _copy_frozen_artifacts(
    output_root: Path,
    *,
    base_model_dir: Path,
) -> dict[str, Any]:
    stats_target = output_root / "stats"
    long_target = output_root / "temporal" / "long"
    stats_target.mkdir(parents=True, exist_ok=True)
    long_target.mkdir(parents=True, exist_ok=True)
    for source, target, names in (
        (
            base_model_dir / "stats",
            stats_target,
            ("model.joblib", "calibrator.joblib", "metadata.json"),
        ),
        (
            base_model_dir / "temporal",
            long_target,
            ("model.pt", "metadata.json", "calibration_report.json"),
        ),
    ):
        for name in names:
            source_path = source / name
            if source_path.exists():
                shutil.copy2(source_path, target / name)
    return {
        "stats_model_sha256": _sha256(stats_target / "model.joblib"),
        "long_model_sha256": _sha256(long_target / "model.pt"),
        "source_stats_model_sha256": _sha256(
            base_model_dir / "stats" / "model.joblib"
        ),
        "source_long_model_sha256": _sha256(
            base_model_dir / "temporal" / "model.pt"
        ),
    }


def train_temporal_short_v22(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model_dir: str | Path,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 12,
    batch_size: int = 1024,
    consistency_weight: float = 0.2,
    distillation_weight: float = 0.25,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("short-flow v2.2 training requires CUDA")
    dataset_root = Path(dataset_dir)
    input_path = dataset_root / "flows"
    split_manifest = dataset_root / "splits" / "split-manifest.json"
    train = load_short_flow_dataset(
        input_path,
        split_manifest,
        "train",
        minimum_packets=1,
        maximum_packets=4,
        limit=limit,
    )
    validation = load_short_flow_dataset(
        input_path,
        split_manifest,
        "validation",
        minimum_packets=1,
        maximum_packets=4,
        limit=limit,
    )
    if len(train.tensors.labels) < 100:
        raise ValueError("short-flow training data are insufficient")
    output_root = Path(output_dir)
    frozen_hashes = _copy_frozen_artifacts(
        output_root,
        base_model_dir=Path(base_model_dir),
    )
    long_model = DeepDetectorModel(
        output_root / "temporal" / "long",
        expected_agent="temporal",
        ood_policy="off",
        device="cuda",
    )
    teacher = long_model.model
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    config = ShortFlowConfig()
    calibration_mask, policy_mask = _partition_validation(
        validation.tensors.sample_ids
    )
    short_root = output_root / "temporal" / "short"
    candidates: list[dict[str, Any]] = []
    device = "cuda"
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_short_flow_encoder(config).to(device)
        positives = max(1, int(train.tensors.labels.sum()))
        negatives = max(1, len(train.tensors.labels) - positives)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [negatives / positives],
                device=device,
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = np.arange(len(train.tensors.labels))
        seed_root = (
            short_root
            / "truncated-boundary-distillation-v2"
            / f"seed-{seed}"
        )
        seed_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_root / "training_checkpoint.pt"
        start_epoch = 0
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            if int(checkpoint.get("seed", -1)) != seed:
                raise ValueError("short-flow checkpoint seed mismatch")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["next_epoch"])
        for epoch in range(start_epoch, epochs):
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(indices)
            model.train()
            for start in range(0, len(indices), batch_size):
                current = indices[start : start + batch_size]
                sequence = torch.from_numpy(
                    train.tensors.sequence[current]
                ).to(device)
                mask = torch.from_numpy(train.tensors.mask[current]).to(device)
                labels = torch.from_numpy(
                    train.tensors.labels[current].astype(np.float32)
                ).to(device)
                counts = torch.from_numpy(
                    train.packet_counts[current]
                ).to(device)
                augmented, augmented_mask = _augment_batch(
                    sequence,
                    mask,
                    generator=generator,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    clean_logits = model(sequence, mask)
                    augmented_logits = model(augmented, augmented_mask)
                    supervised = criterion(clean_logits, labels)
                    consistency = torch.mean(
                        (
                            torch.sigmoid(clean_logits)
                            - torch.sigmoid(augmented_logits)
                        )
                        ** 2
                    )
                    four = counts == 4
                    if four.any():
                        with torch.no_grad():
                            teacher_logits = teacher(
                                sequence[four],
                                mask[four],
                            )
                            teacher_probability = torch.sigmoid(
                                teacher_logits
                                / max(long_model.temperature, 1e-6)
                            )
                        student_boundary_sequence = sequence[four].clone()
                        student_boundary_mask = mask[four].clone()
                        student_boundary_sequence[:, :, 3:] = 0
                        student_boundary_mask[:, 3:] = 0
                        student_boundary_logits = model(
                            student_boundary_sequence,
                            student_boundary_mask,
                        )
                        distillation = torch.mean(
                            (
                                torch.sigmoid(student_boundary_logits)
                                - teacher_probability
                            )
                            ** 2
                        )
                    else:
                        distillation = clean_logits.sum() * 0
                    loss = (
                        supervised
                        + consistency_weight * consistency
                        + distillation_weight * distillation
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.save(
                {
                    "seed": seed,
                    "next_epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint_path,
            )
            _dump(
                seed_root / "training_progress.json",
                {
                    "seed": seed,
                    "completed_epochs": epoch + 1,
                    "expected_epochs": epochs,
                    "resumable": True,
                },
            )
        train_logits, _ = _predict(
            model,
            train.tensors,
            device,
            batch_size,
        )
        validation_logits, _ = _predict(
            model,
            validation.tensors,
            device,
            batch_size,
        )
        temperature = _temperature_scale(
            validation_logits[calibration_mask],
            validation.tensors.labels[calibration_mask],
        )
        probabilities = 1 / (
            1 + np.exp(-validation_logits / max(temperature, 1e-6))
        )
        policy = _select_policy(
            probabilities[policy_mask],
            validation.tensors.labels[policy_mask],
        )
        brier = float(
            np.mean(
                (
                    probabilities
                    - validation.tensors.labels.astype(np.float32)
                )
                ** 2
            )
        )
        ece = _ece(probabilities, validation.tensors.labels)
        four_mask = validation.packet_counts == 4
        with torch.inference_mode():
            teacher_logits, _ = _predict(
                teacher,
                TensorDatasetBundle(
                    sequence=validation.tensors.sequence[four_mask],
                    mask=validation.tensors.mask[four_mask],
                    static=np.empty(
                        (int(four_mask.sum()), 0),
                        dtype=np.float32,
                    ),
                    labels=validation.tensors.labels[four_mask],
                    sample_ids=[
                        sample_id
                        for sample_id, selected in zip(
                            validation.tensors.sample_ids,
                            four_mask,
                            strict=True,
                        )
                        if selected
                    ],
                ),
                device,
                batch_size,
            )
        teacher_probabilities = 1 / (
            1 + np.exp(-teacher_logits / max(long_model.temperature, 1e-6))
        )
        boundary_agreement = float(
            np.mean(
                (probabilities[four_mask] >= 0.5)
                == (teacher_probabilities >= 0.5)
            )
        )
        model_path = seed_root / "model.pt"
        torch.save(model.state_dict(), model_path)
        metadata = {
            "schema_version": SHORT_FLOW_SCHEMA_VERSION,
            "backend": "deep_v2_2_short",
            "architecture": "two_block_residual_mlp",
            "config": config_as_json(config),
            "temperature": temperature,
            "calibration_quality": max(0.0, min(1.0, 1 - ece)),
            "decision_policy": policy,
            "model_sha256": _sha256(model_path),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_count": int(len(train.tensors.labels)),
            "validation_count": int(len(validation.tensors.labels)),
            "runtime_packet_regime": "1_to_3",
            "training_packet_regime": "1_to_4",
            "consistency_weight": consistency_weight,
            "distillation_weight": distillation_weight,
            "distillation_objective": (
                "three_packet_student_matches_four_packet_frozen_long_teacher"
            ),
            "teacher_long_model_sha256": frozen_hashes[
                "long_model_sha256"
            ],
            "blocked_inputs": [
                "stats",
                "labels",
                "provenance",
                "sample_id",
                "trace_id",
                "IP",
                "port",
                "family",
                "application",
            ],
        }
        _dump(seed_root / "metadata.json", metadata)
        report = {
            "seed": seed,
            "brier_score": brier,
            "ece": ece,
            "temperature": temperature,
            "decision_policy": policy,
            "four_packet_teacher_agreement": boundary_agreement,
        }
        _dump(seed_root / "calibration_report.json", report)
        candidates.append(
            {
                "seed": seed,
                "path": str(seed_root),
                "brier_score": brier,
                "ece": ece,
                "four_packet_teacher_agreement": boundary_agreement,
            }
        )
        checkpoint_path.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()

    selected = min(
        candidates,
        key=lambda item: (
            item["brier_score"],
            item["ece"],
            -item["four_packet_teacher_agreement"],
        ),
    )
    selected_root = Path(selected["path"])
    short_root.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "metadata.json", "calibration_report.json"):
        shutil.copy2(selected_root / name, short_root / name)
    combined_metadata = {
        "schema_version": "1.0",
        "backend": "deep_v2_2",
        "routing": {
            "empty": "abstain",
            "1_to_3": "short",
            "4_plus": "long",
        },
        "short_model_sha256": _sha256(short_root / "model.pt"),
        "long_model_sha256": frozen_hashes["long_model_sha256"],
        "stats_model_sha256": frozen_hashes["stats_model_sha256"],
        "source_hashes_match": (
            frozen_hashes["stats_model_sha256"]
            == frozen_hashes["source_stats_model_sha256"]
            and frozen_hashes["long_model_sha256"]
            == frozen_hashes["source_long_model_sha256"]
        ),
        "fusion_modified": False,
        "ood_v1_v2_v3_v31_modified": False,
    }
    _dump(output_root / "temporal" / "metadata.json", combined_metadata)
    summary = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_2_short_training",
        "status": "trained",
        "selected_seed": selected["seed"],
        "candidates": candidates,
        "train_count": int(len(train.tensors.labels)),
        "validation_count": int(len(validation.tensors.labels)),
        "frozen_hashes": frozen_hashes,
        "automatic_training": False,
        "automatic_deployment": False,
        "distillation_objective_version": "truncated-boundary-v2",
    }
    _dump(output_root / "training_summary.json", summary)
    return summary


def calibrate_conformal_v32(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    reference_limit: int = 8192,
    validation_limit: int = 12000,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    from .short_flow import ShortFlowModel

    dataset_root = Path(dataset_dir)
    model_root = Path(model_dir)
    input_path = dataset_root / "flows"
    split_manifest = dataset_root / "splits" / "split-manifest.json"
    short_reference = load_short_flow_dataset(
        input_path,
        split_manifest,
        "train",
        minimum_packets=1,
        maximum_packets=3,
        limit=reference_limit,
    )
    short_validation = load_short_flow_dataset(
        input_path,
        split_manifest,
        "validation",
        minimum_packets=1,
        maximum_packets=3,
        limit=validation_limit,
    )
    long_reference = load_v2_tensor_dataset(
        input_path,
        split_manifest,
        "train",
        "temporal",
        limit=reference_limit,
    )
    long_validation = load_v2_tensor_dataset(
        input_path,
        split_manifest,
        "validation",
        "temporal",
        limit=validation_limit,
    )
    short_model = ShortFlowModel(model_root / "temporal" / "short")
    long_model = DeepDetectorModel(
        model_root / "temporal" / "long",
        expected_agent="temporal",
        ood_policy="off",
    )
    short_train_logits, short_train_embeddings = _predict(
        short_model.model,
        short_reference.tensors,
        str(short_model.device),
        batch_size,
    )
    del short_train_logits
    short_val_logits, short_val_embeddings = _predict(
        short_model.model,
        short_validation.tensors,
        str(short_model.device),
        batch_size,
    )
    del short_val_logits
    long_train_logits, long_train_embeddings = _predict(
        long_model.model,
        long_reference,
        str(long_model.device),
        batch_size,
    )
    del long_train_logits
    long_val_logits, long_val_embeddings = _predict(
        long_model.model,
        long_validation,
        str(long_model.device),
        batch_size,
    )
    del long_val_logits
    short_calibration, _ = _partition_validation(
        short_validation.tensors.sample_ids
    )
    long_calibration, _ = _partition_validation(
        long_validation.sample_ids
    )
    short_meta = fit_regime_conformal_v32(
        short_train_embeddings,
        short_reference.tensors.labels,
        short_val_embeddings[short_calibration],
        short_validation.tensors.labels[short_calibration],
        output_dir,
        regime="short",
        seed=seed,
    )
    long_meta = fit_regime_conformal_v32(
        long_train_embeddings,
        long_reference.labels,
        long_val_embeddings[long_calibration],
        long_validation.labels[long_calibration],
        output_dir,
        regime="long",
        seed=seed,
    )
    summary = {
        "schema_version": "1.0",
        "policy": "conformal_v3_2",
        "method": "embedding_1nn",
        "alpha": 0.01,
        "hard_p_value": 0.01,
        "short": short_meta,
        "long": long_meta,
        "validation_only_calibration": True,
        "test_or_external_used": False,
    }
    _dump(Path(output_dir) / "metadata.json", summary)
    return summary
