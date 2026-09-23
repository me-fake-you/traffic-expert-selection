from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .deep_models import DeepDetectorModel
from .ood_v33 import fit_regime_conformal_v33
from .short_flow import (
    SHORT_FLOW_SCHEMA_VERSION,
    ShortFlowConfig,
    build_short_flow_encoder,
    config_as_json,
)
from .short_flow_robust import RobustShortFlowModel
from .v2_training import (
    TensorDatasetBundle,
    _ece,
    _partition_validation,
    _predict,
    _select_policy,
    _temperature_scale,
    load_v2_tensor_dataset,
)
from .v22_training import load_short_flow_dataset


PERTURBATION_KINDS = (
    "padding",
    "dummy_packet",
    "iat_jitter",
    "sequence_truncation",
)
TRAINING_STRENGTHS = (0.1, 0.2, 0.3)


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


def _copy_v22_frozen_artifacts(
    output_root: Path,
    *,
    base_v22_model_dir: Path,
) -> dict[str, Any]:
    stats_target = output_root / "stats"
    long_target = output_root / "temporal" / "long"
    stats_target.mkdir(parents=True, exist_ok=True)
    long_target.mkdir(parents=True, exist_ok=True)
    for source, target, names in (
        (
            base_v22_model_dir / "stats",
            stats_target,
            ("model.joblib", "calibrator.joblib", "metadata.json"),
        ),
        (
            base_v22_model_dir / "temporal" / "long",
            long_target,
            ("model.pt", "metadata.json", "calibration_report.json"),
        ),
    ):
        for name in names:
            source_path = source / name
            if source_path.exists():
                shutil.copy2(source_path, target / name)
    hashes = {
        "stats_model_sha256": _sha256(stats_target / "model.joblib"),
        "long_model_sha256": _sha256(long_target / "model.pt"),
        "source_stats_model_sha256": _sha256(
            base_v22_model_dir / "stats" / "model.joblib"
        ),
        "source_long_model_sha256": _sha256(
            base_v22_model_dir / "temporal" / "long" / "model.pt"
        ),
        "source_short_model_sha256": _sha256(
            base_v22_model_dir / "temporal" / "short" / "model.pt"
        ),
    }
    hashes["source_hashes_match"] = (
        hashes["stats_model_sha256"]
        == hashes["source_stats_model_sha256"]
        and hashes["long_model_sha256"]
        == hashes["source_long_model_sha256"]
    )
    return hashes


def paired_short_augmentations(
    sequence,
    mask,
    *,
    strength: float,
    generator,
) -> dict[str, tuple[Any, Any]]:
    """Create all four train-only counterfactual views in encoded space."""
    if strength not in TRAINING_STRENGTHS and strength != 0.2:
        raise ValueError("v2.3 strength must be one of 0.1, 0.2, or 0.3")
    torch = __import__("torch")
    batch, channels, length = sequence.shape
    output: dict[str, tuple[Any, Any]] = {}
    valid_counts = mask.sum(dim=1).to(dtype=torch.long)
    positions = torch.arange(
        length,
        device=sequence.device,
    ).unsqueeze(0).expand(batch, -1)

    padded = sequence.clone()
    padded_mask = mask.clone()
    selected = (
        torch.rand(
            (batch, length),
            generator=generator,
            device=sequence.device,
        )
        < strength
    ) & (mask > 0)
    signed = padded[:, 0, :]
    rounded = torch.sign(signed) * torch.log1p(
        torch.ceil(torch.expm1(torch.abs(signed)) / 64) * 64
    )
    padded[:, 0, :] = torch.where(selected, rounded, signed)
    output["padding"] = (padded, padded_mask)

    insertion_positions = torch.floor(
        torch.rand(
            batch,
            generator=generator,
            device=sequence.device,
        )
        * (valid_counts + 1).to(sequence.dtype)
    ).to(dtype=torch.long).clamp(max=length - 1)
    source_positions = torch.where(
        positions < insertion_positions.unsqueeze(1),
        positions,
        positions - 1,
    ).clamp(min=0, max=length - 1)
    dummy = torch.gather(
        sequence,
        2,
        source_positions.unsqueeze(1).expand(-1, channels, -1),
    )
    dummy_mask = positions < (
        valid_counts + 1
    ).clamp(max=length).unsqueeze(1)
    insert_mask = positions == insertion_positions.unsqueeze(1)
    raw_lengths = torch.randint(
        1,
        17,
        (batch,),
        generator=generator,
        device=sequence.device,
    )
    directions = (
        torch.randint(
            0,
            2,
            (batch,),
            generator=generator,
            device=sequence.device,
        )
        * 2
        - 1
    ).to(sequence.dtype)
    dummy[:, 0, :] = torch.where(
        insert_mask,
        directions.unsqueeze(1) * torch.log1p(raw_lengths).unsqueeze(1),
        dummy[:, 0, :],
    )
    dummy[:, 1, :] = torch.where(
        insert_mask,
        directions.unsqueeze(1),
        dummy[:, 1, :],
    )
    if channels > 2:
        raw_iats = (
            torch.rand(
                batch,
                generator=generator,
                device=sequence.device,
            )
            * 0.01
        )
        dummy[:, 2, :] = torch.where(
            insert_mask,
            torch.log1p(raw_iats).unsqueeze(1),
            dummy[:, 2, :],
        )
        dummy[:, 3, :] = torch.where(
            insert_mask,
            torch.ones_like(dummy[:, 3, :]),
            dummy[:, 3, :],
        )
    dummy = dummy * dummy_mask.unsqueeze(1)
    dummy_mask = dummy_mask.to(mask.dtype)
    output["dummy_packet"] = (dummy, dummy_mask)

    jitter = sequence.clone()
    jitter_mask = mask.clone()
    if channels > 2:
        raw_iats = torch.expm1(jitter[:, 2, :]).clamp_min(0)
        multipliers = 1 + (
            torch.rand(
                (batch, length),
                generator=generator,
                device=sequence.device,
            )
            * 2
            - 1
        ) * strength * 3
        transformed = torch.log1p((raw_iats * multipliers).clamp_min(0))
        jitter[:, 2, :] = torch.where(
            mask > 0,
            transformed,
            jitter[:, 2, :],
        )
    output["iat_jitter"] = (jitter, jitter_mask)

    truncated = sequence.clone()
    keep_counts = torch.floor(
        valid_counts.to(sequence.dtype) * (1 - 0.8 * strength)
    ).to(dtype=torch.long).clamp(min=1)
    truncated_mask = (
        positions < keep_counts.unsqueeze(1)
    ).to(mask.dtype)
    truncated = truncated * truncated_mask.unsqueeze(1)
    output["sequence_truncation"] = (truncated, truncated_mask)
    return output


def _bundle_slice(bundle: TensorDatasetBundle, selected: np.ndarray):
    return TensorDatasetBundle(
        sequence=bundle.sequence[selected],
        mask=bundle.mask[selected],
        static=np.empty((int(selected.sum()), 0), dtype=np.float32),
        labels=bundle.labels[selected],
        sample_ids=[
            sample_id
            for sample_id, keep in zip(
                bundle.sample_ids,
                selected,
                strict=True,
            )
            if keep
        ],
    )


def _validation_robustness(
    model,
    validation,
    *,
    temperature: float,
    device: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    deltas: dict[str, list[float]] = {kind: [] for kind in PERTURBATION_KINDS}
    agreements: dict[str, list[bool]] = {
        kind: [] for kind in PERTURBATION_KINDS
    }
    with torch.inference_mode():
        for start in range(0, len(validation.tensors.labels), batch_size):
            stop = start + batch_size
            sequence = torch.from_numpy(
                validation.tensors.sequence[start:stop]
            ).to(device)
            mask = torch.from_numpy(
                validation.tensors.mask[start:stop]
            ).to(device)
            clean = torch.sigmoid(
                model(sequence, mask) / max(temperature, 1e-6)
            )
            variants = paired_short_augmentations(
                sequence,
                mask,
                strength=0.2,
                generator=generator,
            )
            for kind, (values, visible) in variants.items():
                perturbed = torch.sigmoid(
                    model(values, visible) / max(temperature, 1e-6)
                )
                deltas[kind].extend(
                    torch.abs(clean - perturbed).cpu().tolist()
                )
                agreements[kind].extend(
                    ((clean >= 0.5) == (perturbed >= 0.5)).cpu().tolist()
                )
    by_kind = {
        kind: {
            "mean_absolute_probability_delta": float(np.mean(deltas[kind])),
            "binary_probability_agreement": float(
                np.mean(agreements[kind])
            ),
        }
        for kind in PERTURBATION_KINDS
    }
    return {
        "strength": 0.2,
        "seed": seed,
        "by_kind": by_kind,
        "mean_absolute_probability_delta": float(
            np.mean(
                [
                    item
                    for values in deltas.values()
                    for item in values
                ]
            )
        ),
        "mean_binary_probability_agreement": float(
            np.mean(
                [
                    item
                    for values in agreements.values()
                    for item in values
                ]
            )
        ),
    }


def train_temporal_robust_v23(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    base_v22_model_dir: str | Path,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 6,
    batch_size: int = 1024,
    perturbed_supervised_weight: float = 0.5,
    probability_consistency_weight: float = 0.25,
    embedding_consistency_weight: float = 0.1,
    boundary_consistency_weight: float = 0.25,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("short-flow v2.3 training requires CUDA")
    if (
        perturbed_supervised_weight,
        probability_consistency_weight,
        embedding_consistency_weight,
        boundary_consistency_weight,
    ) != (0.5, 0.25, 0.1, 0.25):
        raise ValueError("v2.3 loss weights are preregistered and immutable")
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
        raise ValueError("short-flow v2.3 training data are insufficient")

    output_root = Path(output_dir)
    base_v22 = Path(base_v22_model_dir)
    frozen_hashes = _copy_v22_frozen_artifacts(
        output_root,
        base_v22_model_dir=base_v22,
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
    initial_state = torch.load(
        base_v22 / "temporal" / "short" / "model.pt",
        map_location="cuda",
        weights_only=True,
    )
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
        model.load_state_dict(initial_state)
        positives = max(1, int(train.tensors.labels.sum()))
        negatives = max(1, len(train.tensors.labels) - positives)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = np.arange(len(train.tensors.labels))
        seed_root = short_root / "paired-counterfactual-v3" / f"seed-{seed}"
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
                raise ValueError("v2.3 checkpoint seed mismatch")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["next_epoch"])

        for epoch in range(start_epoch, epochs):
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(indices)
            model.train()
            for batch_index, start in enumerate(
                range(0, len(indices), batch_size)
            ):
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
                strength = TRAINING_STRENGTHS[
                    (epoch + batch_index) % len(TRAINING_STRENGTHS)
                ]
                variants = paired_short_augmentations(
                    sequence,
                    mask,
                    strength=strength,
                    generator=generator,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    clean_logits, clean_embeddings = model(
                        sequence,
                        mask,
                        return_embedding=True,
                    )
                    clean_supervised = criterion(clean_logits, labels)
                    perturbed_supervised = clean_logits.sum() * 0
                    probability_consistency = clean_logits.sum() * 0
                    embedding_consistency = clean_logits.sum() * 0
                    outputs = {}
                    for kind, (values, visible) in variants.items():
                        variant_logits, variant_embeddings = model(
                            values,
                            visible,
                            return_embedding=True,
                        )
                        outputs[kind] = (
                            variant_logits,
                            variant_embeddings,
                            values,
                            visible,
                        )
                        perturbed_supervised = (
                            perturbed_supervised
                            + criterion(variant_logits, labels)
                        )
                        probability_consistency = (
                            probability_consistency
                            + torch.mean(
                                (
                                    torch.sigmoid(clean_logits)
                                    - torch.sigmoid(variant_logits)
                                )
                                ** 2
                            )
                        )
                        embedding_consistency = (
                            embedding_consistency
                            + torch.mean(
                                1
                                - functional.cosine_similarity(
                                    clean_embeddings,
                                    variant_embeddings,
                                    dim=1,
                                )
                            )
                        )
                    divisor = float(len(PERTURBATION_KINDS))
                    perturbed_supervised = perturbed_supervised / divisor
                    probability_consistency = (
                        probability_consistency / divisor
                    )
                    embedding_consistency = embedding_consistency / divisor

                    boundary_losses = []
                    three = counts == 3
                    if three.any():
                        dummy_logits, _, dummy_values, dummy_visible = outputs[
                            "dummy_packet"
                        ]
                        with torch.no_grad():
                            teacher_dummy = torch.sigmoid(
                                teacher(
                                    dummy_values[three],
                                    dummy_visible[three],
                                )
                                / max(long_model.temperature, 1e-6)
                            )
                        boundary_losses.append(
                            torch.mean(
                                (
                                    torch.sigmoid(dummy_logits[three])
                                    - teacher_dummy
                                )
                                ** 2
                            )
                        )
                    four = counts == 4
                    if four.any():
                        truncated_logits = outputs[
                            "sequence_truncation"
                        ][0]
                        with torch.no_grad():
                            teacher_clean = torch.sigmoid(
                                teacher(sequence[four], mask[four])
                                / max(long_model.temperature, 1e-6)
                            )
                        boundary_losses.append(
                            torch.mean(
                                (
                                    torch.sigmoid(truncated_logits[four])
                                    - teacher_clean
                                )
                                ** 2
                            )
                        )
                    boundary_consistency = (
                        torch.stack(boundary_losses).mean()
                        if boundary_losses
                        else clean_logits.sum() * 0
                    )
                    loss = (
                        clean_supervised
                        + perturbed_supervised_weight
                        * perturbed_supervised
                        + probability_consistency_weight
                        * probability_consistency
                        + embedding_consistency_weight
                        * embedding_consistency
                        + boundary_consistency_weight
                        * boundary_consistency
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
        robustness = _validation_robustness(
            model,
            validation,
            temperature=temperature,
            device=device,
            batch_size=batch_size,
            seed=42,
        )
        model_path = seed_root / "model.pt"
        torch.save(model.state_dict(), model_path)
        metadata = {
            "schema_version": SHORT_FLOW_SCHEMA_VERSION,
            "backend": "deep_v2_3_short",
            "architecture": "two_block_residual_mlp",
            "config": config_as_json(config),
            "temperature": temperature,
            "calibration_quality": max(0.0, min(1.0, 1 - ece)),
            "decision_policy": policy,
            "model_sha256": _sha256(model_path),
            "initialized_from_v2_2_sha256": frozen_hashes[
                "source_short_model_sha256"
            ],
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "train_count": int(len(train.tensors.labels)),
            "validation_count": int(len(validation.tensors.labels)),
            "runtime_packet_regime": "1_to_4_boundary_overlap",
            "training_packet_regime": "1_to_4",
            "paired_perturbations": list(PERTURBATION_KINDS),
            "training_strengths": list(TRAINING_STRENGTHS),
            "loss_weights": {
                "clean_bce": 1.0,
                "perturbed_bce": perturbed_supervised_weight,
                "probability_consistency": probability_consistency_weight,
                "embedding_consistency": embedding_consistency_weight,
                "boundary_consistency": boundary_consistency_weight,
            },
            "teacher_long_model_sha256": frozen_hashes[
                "long_model_sha256"
            ],
            "blocked_inputs": [
                "stats",
                "labels_at_runtime",
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
            "validation_robustness": robustness,
        }
        _dump(seed_root / "calibration_report.json", report)
        candidates.append(
            {
                "seed": seed,
                "path": str(seed_root),
                "brier_score": brier,
                "ece": ece,
                "mean_absolute_probability_delta": robustness[
                    "mean_absolute_probability_delta"
                ],
                "mean_binary_probability_agreement": robustness[
                    "mean_binary_probability_agreement"
                ],
            }
        )
        checkpoint_path.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()

    selected = min(
        candidates,
        key=lambda item: (
            item["mean_absolute_probability_delta"],
            item["brier_score"],
            item["ece"],
            item["seed"],
        ),
    )
    selected_root = Path(selected["path"])
    short_root.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "metadata.json", "calibration_report.json"):
        shutil.copy2(selected_root / name, short_root / name)
    combined_metadata = {
        "schema_version": "1.0",
        "backend": "deep_v2_3",
        "routing": {
            "empty": "abstain",
            "1_to_2": "short",
            "3": "short_primary_long_consistency_veto",
            "4": "long_primary_short_consistency_veto",
            "5_plus": "long",
        },
        "secondary_confidence_threshold": 0.9,
        "boundary_disagreement_action": "temporal_abstain",
        "short_model_sha256": _sha256(short_root / "model.pt"),
        "long_model_sha256": frozen_hashes["long_model_sha256"],
        "stats_model_sha256": frozen_hashes["stats_model_sha256"],
        "source_hashes_match": frozen_hashes["source_hashes_match"],
        "fusion_modified": False,
        "ood_v1_v2_v3_v31_v32_modified": False,
    }
    _dump(output_root / "temporal" / "metadata.json", combined_metadata)
    summary = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_3_short_robust_training",
        "status": "trained",
        "selected_seed": selected["seed"],
        "selection_rule": (
            "minimum validation paired probability delta, then brier/ece/seed"
        ),
        "candidates": candidates,
        "train_count": int(len(train.tensors.labels)),
        "validation_count": int(len(validation.tensors.labels)),
        "frozen_hashes": frozen_hashes,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(output_root / "training_summary.json", summary)
    return summary


def calibrate_conformal_v33(
    dataset_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    reference_limit: int = 8192,
    validation_limit: int = 12000,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    model_root = Path(model_dir)
    input_path = dataset_root / "flows"
    split_manifest = dataset_root / "splits" / "split-manifest.json"
    short_reference = load_short_flow_dataset(
        input_path,
        split_manifest,
        "train",
        minimum_packets=1,
        maximum_packets=4,
        limit=reference_limit,
    )
    short_validation = load_short_flow_dataset(
        input_path,
        split_manifest,
        "validation",
        minimum_packets=1,
        maximum_packets=4,
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
    short_model = RobustShortFlowModel(model_root / "temporal" / "short")
    long_model = DeepDetectorModel(
        model_root / "temporal" / "long",
        expected_agent="temporal",
        ood_policy="off",
    )
    _, short_train_embeddings = _predict(
        short_model.model,
        short_reference.tensors,
        str(short_model.device),
        batch_size,
    )
    _, short_val_embeddings = _predict(
        short_model.model,
        short_validation.tensors,
        str(short_model.device),
        batch_size,
    )
    _, long_train_embeddings = _predict(
        long_model.model,
        long_reference,
        str(long_model.device),
        batch_size,
    )
    _, long_val_embeddings = _predict(
        long_model.model,
        long_validation,
        str(long_model.device),
        batch_size,
    )
    short_calibration, _ = _partition_validation(
        short_validation.tensors.sample_ids
    )
    long_calibration, _ = _partition_validation(long_validation.sample_ids)
    short_meta = fit_regime_conformal_v33(
        short_train_embeddings,
        short_reference.tensors.labels,
        short_val_embeddings[short_calibration],
        short_validation.tensors.labels[short_calibration],
        output_dir,
        regime="short",
        seed=seed,
    )
    long_meta = fit_regime_conformal_v33(
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
        "policy": "conformal_v3_3",
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
