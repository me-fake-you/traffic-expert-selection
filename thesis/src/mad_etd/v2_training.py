from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .deep_models import (
    DEEP_V2_ARTIFACT_SCHEMA_VERSION,
    TCNConfig,
    build_tcn,
    config_as_json,
    encode_temporal_input,
    encode_tls_input,
)
from .io import iter_split_records
from .ood_v3 import fit_conformal_artifact
from .schemas import DetectorInput, FlowRecord


V2_TRAINING_AGENTS = ("temporal", "tls")


@dataclass(slots=True)
class TensorDatasetBundle:
    sequence: np.ndarray
    mask: np.ndarray
    static: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]


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


def load_v2_tensor_dataset(
    input_path: str | Path,
    split_manifest: str | Path,
    split: str,
    agent: str,
    *,
    limit: int | None = None,
) -> TensorDatasetBundle:
    if agent not in V2_TRAINING_AGENTS:
        raise ValueError(f"unsupported v2 training agent: {agent}")
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    static_values: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    encode = encode_temporal_input if agent == "temporal" else encode_tls_input
    reservoirs: dict[int, list[tuple[int, str, Any]]] | None = None
    capacities = None
    if limit is not None:
        capacities = {0: limit // 2, 1: limit - limit // 2}
        reservoirs = {0: [], 1: []}
    for record in iter_split_records(input_path, split_manifest, split):
        label = str(record.labels.get("binary", "")).lower()
        if label not in {"benign", "malicious"}:
            continue
        if agent == "temporal" and len(record.sequence.packet_lengths) < 4:
            continue
        if agent == "tls":
            records = record.tls.get("record_lengths") or record.tls.get(
                "tls_record_lengths"
            ) or []
            if len(records) < 2:
                continue
        sequence, mask, static = encode(_safe_input(record))
        binary = int(label == "malicious")
        if reservoirs is not None and capacities is not None:
            score = int(
                hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:16],
                16,
            )
            item = (
                -score,
                record.sample_id,
                (sequence, mask, static, binary),
            )
            current = reservoirs[binary]
            if len(current) < capacities[binary]:
                heapq.heappush(current, item)
            elif item[:2] > current[0][:2]:
                heapq.heapreplace(current, item)
            continue
        sequences.append(sequence)
        masks.append(mask)
        static_values.append(static)
        labels.append(binary)
        sample_ids.append(record.sample_id)
    if reservoirs is not None:
        selected = sorted(
            (
                (sample_id, payload)
                for values in reservoirs.values()
                for _, sample_id, payload in values
            ),
            key=lambda item: item[0],
        )
        for sample_id, (sequence, mask, static, binary) in selected:
            sequences.append(sequence)
            masks.append(mask)
            static_values.append(static)
            labels.append(binary)
            sample_ids.append(sample_id)
    if not labels:
        channels = 4 if agent == "temporal" else 2
        static_dim = 0 if agent == "temporal" else 4
        return TensorDatasetBundle(
            sequence=np.empty((0, channels, 64), dtype=np.float32),
            mask=np.empty((0, 64), dtype=np.float32),
            static=np.empty((0, static_dim), dtype=np.float32),
            labels=np.empty(0, dtype=np.int8),
            sample_ids=[],
        )
    static_dim = len(static_values[0])
    return TensorDatasetBundle(
        sequence=np.stack(sequences).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        static=(
            np.stack(static_values).astype(np.float32)
            if static_dim
            else np.empty((len(labels), 0), dtype=np.float32)
        ),
        labels=np.asarray(labels, dtype=np.int8),
        sample_ids=sample_ids,
    )


def _partition_validation(sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.asarray(
        [
            int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:8], 16) % 2
            == 0
            for item in sample_ids
        ],
        dtype=bool,
    )
    if calibration.all() or (~calibration).all():
        calibration[::2] = True
        calibration[1::2] = False
    return calibration, ~calibration


def _augment_batch(sequence, mask, *, generator):
    torch = __import__("torch")
    augmented = sequence.clone()
    augmented_mask = mask.clone()
    batch, _, length = augmented.shape
    choices = torch.randint(
        0,
        4,
        (batch,),
        generator=generator,
        device=augmented.device,
    )
    for row in range(batch):
        choice = int(choices[row])
        valid = int(augmented_mask[row].sum().item())
        if valid <= 1:
            continue
        if choice == 0:  # padding
            signed = augmented[row, 0, :valid]
            augmented[row, 0, :valid] = torch.sign(signed) * torch.log1p(
                torch.ceil(torch.expm1(torch.abs(signed)) / 64) * 64
            )
        elif choice == 1:  # dummy record/packet
            position = min(valid, length - 1)
            augmented[row, 0, position] = math.log1p(8)
            augmented[row, 1, position] = 1
            if augmented.shape[1] > 2:
                augmented[row, 2, position] = 0
                augmented[row, 3, position] = 1
            augmented_mask[row, position] = 1
        elif choice == 2 and augmented.shape[1] > 2:  # IAT jitter
            noise = torch.randn(
                valid,
                generator=generator,
                device=augmented.device,
            ) * 0.15
            augmented[row, 2, :valid] = torch.clamp(
                augmented[row, 2, :valid] + noise,
                min=0,
            )
        else:  # truncation
            keep = max(1, int(valid * 0.7))
            augmented[row, :, keep:valid] = 0
            augmented_mask[row, keep:valid] = 0
    return augmented, augmented_mask


def _predict(model, bundle: TensorDatasetBundle, device: str, batch_size: int):
    import torch

    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(bundle.labels), batch_size):
            stop = start + batch_size
            sequence = torch.from_numpy(bundle.sequence[start:stop]).to(device)
            mask = torch.from_numpy(bundle.mask[start:stop]).to(device)
            static = (
                torch.from_numpy(bundle.static[start:stop]).to(device)
                if bundle.static.shape[1]
                else None
            )
            current_logits, current_embeddings = model(
                sequence,
                mask,
                static,
                return_embedding=True,
            )
            logits.append(current_logits.float().cpu().numpy())
            embeddings.append(current_embeddings.float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(embeddings)


def _temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.linspace(0.5, 10.0, 191)
    losses = []
    for temperature in candidates:
        probabilities = 1 / (1 + np.exp(-logits / temperature))
        clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        losses.append(
            -np.mean(
                labels * np.log(clipped)
                + (1 - labels) * np.log(1 - clipped)
            )
        )
    return float(candidates[int(np.argmin(losses))])


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1 else probabilities <= upper
        )
        if not selected.any():
            continue
        result += selected.mean() * abs(
            probabilities[selected].mean() - labels[selected].mean()
        )
    return float(result)


def _select_policy(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for benign_max in np.linspace(0.05, 0.45, 17):
        for malicious_min in np.linspace(0.55, 0.95, 17):
            covered = (probabilities <= benign_max) | (
                probabilities >= malicious_min
            )
            if not covered.any():
                continue
            predictions = probabilities[covered] >= malicious_min
            truth = labels[covered].astype(bool)
            risk = float(np.mean(predictions != truth))
            benign = labels == 0
            false_positive_rate = float(
                np.mean(probabilities[benign] >= malicious_min)
            ) if benign.any() else 0.0
            coverage = float(covered.mean())
            valid = risk <= 0.05 and false_positive_rate <= 0.05
            candidate = {
                "benign_max_probability": float(benign_max),
                "malicious_min_probability": float(malicious_min),
                "coverage": coverage,
                "selective_risk": risk,
                "benign_false_positive_rate": false_positive_rate,
                "constraints_satisfied": valid,
            }
            if valid and (
                best is None
                or coverage > best["coverage"]
                or (
                    coverage == best["coverage"]
                    and risk < best["selective_risk"]
                )
            ):
                best = candidate
    return best or {
        "benign_max_probability": 0.2,
        "malicious_min_probability": 0.8,
        "coverage": 0.0,
        "selective_risk": 1.0,
        "benign_false_positive_rate": 1.0,
        "constraints_satisfied": False,
    }


def _copy_selected(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "metadata.json", "calibration_report.json"):
        shutil.copy2(source / name, target / name)
    source_conformal = source / "conformal_v3"
    target_conformal = target / "conformal_v3"
    if target_conformal.exists():
        shutil.rmtree(target_conformal)
    shutil.copytree(source_conformal, target_conformal)


def train_deep_v2_agent(
    dataset_dir: str | Path,
    output_dir: str | Path,
    agent: str,
    *,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 12,
    batch_size: int = 512,
    consistency_weight: float = 0.2,
    limit: int | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("deep_v2 training requires CUDA")
    dataset_root = Path(dataset_dir)
    input_path = dataset_root / "flows"
    split_manifest = dataset_root / "splits" / "split-manifest.json"
    train = load_v2_tensor_dataset(
        input_path, split_manifest, "train", agent, limit=limit
    )
    validation = load_v2_tensor_dataset(
        input_path, split_manifest, "validation", agent, limit=limit
    )
    if len(train.labels) < 100 or len(np.unique(train.labels)) < 2:
        return {
            "agent": agent,
            "status": "skipped_insufficient_labeled_sequences",
            "train_count": int(len(train.labels)),
            "validation_count": int(len(validation.labels)),
        }
    if len(validation.labels) < 40 or len(np.unique(validation.labels)) < 2:
        raise ValueError(f"{agent} validation data are insufficient")

    config = TCNConfig(
        input_channels=int(train.sequence.shape[1]),
        static_dim=int(train.static.shape[1]),
    )
    calibration_mask, policy_mask = _partition_validation(validation.sample_ids)
    root = Path(output_dir)
    candidates: list[dict[str, Any]] = []
    device = "cuda"
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_tcn(config).to(device)
        positives = max(1, int(train.labels.sum()))
        negatives = max(1, len(train.labels) - positives)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scaler = torch.amp.GradScaler("cuda")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        indices = np.arange(len(train.labels))
        seed_root = root / f"seed-{seed}"
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
                raise ValueError("deep_v2 checkpoint seed mismatch")
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
                sequence = torch.from_numpy(train.sequence[current]).to(device)
                mask = torch.from_numpy(train.mask[current]).to(device)
                static = (
                    torch.from_numpy(train.static[current]).to(device)
                    if train.static.shape[1]
                    else None
                )
                labels = torch.from_numpy(
                    train.labels[current].astype(np.float32)
                ).to(device)
                augmented, augmented_mask = _augment_batch(
                    sequence, mask, generator=generator
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    clean_logits = model(sequence, mask, static)
                    augmented_logits = model(
                        augmented, augmented_mask, static
                    )
                    supervised = criterion(clean_logits, labels)
                    consistency = torch.mean(
                        (
                            torch.sigmoid(clean_logits)
                            - torch.sigmoid(augmented_logits)
                        )
                        ** 2
                    )
                    loss = supervised + consistency_weight * consistency
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

        train_logits, train_embeddings = _predict(
            model, train, device, batch_size
        )
        validation_logits, validation_embeddings = _predict(
            model, validation, device, batch_size
        )
        temperature = _temperature_scale(
            validation_logits[calibration_mask],
            validation.labels[calibration_mask],
        )
        probabilities = 1 / (
            1 + np.exp(-validation_logits / max(temperature, 1e-6))
        )
        policy = _select_policy(
            probabilities[policy_mask],
            validation.labels[policy_mask],
        )
        brier = float(
            np.mean(
                (probabilities - validation.labels.astype(np.float32)) ** 2
            )
        )
        ece = _ece(probabilities, validation.labels)
        model_path = seed_root / "model.pt"
        torch.save(model.state_dict(), model_path)
        fit_conformal_artifact(
            train_embeddings,
            train.labels,
            validation_embeddings[calibration_mask],
            validation.labels[calibration_mask],
            seed_root / "conformal_v3",
            seed=seed,
        )
        metadata = {
            "schema_version": DEEP_V2_ARTIFACT_SCHEMA_VERSION,
            "agent": agent,
            "backend": "deep_v2",
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
            "cuda_required": True,
            "consistency_regularization": {
                "enabled": consistency_weight > 0,
                "weight": consistency_weight,
                "perturbations": [
                    "padding",
                    "dummy_packet",
                    "iat_jitter",
                    "sequence_truncation",
                ],
                "train_only": True,
            },
            "train_count": int(len(train.labels)),
            "validation_count": int(len(validation.labels)),
            "embedding_dim": config.embedding_dim,
            "blocked_inputs": [
                "labels",
                "provenance",
                "sample_id",
                "trace_id",
                "IP",
                "port",
                "SNI",
                "JA3",
                "JA4",
            ],
        }
        _dump(seed_root / "metadata.json", metadata)
        report = {
            "seed": seed,
            "brier_score": brier,
            "ece": ece,
            "temperature": temperature,
            "decision_policy": policy,
        }
        _dump(seed_root / "calibration_report.json", report)
        candidates.append(
            {
                "seed": seed,
                "path": str(seed_root),
                "brier_score": brier,
                "ece": ece,
            }
        )
        checkpoint_path.unlink(missing_ok=True)
        del model
        torch.cuda.empty_cache()

    selected = min(candidates, key=lambda item: (item["brier_score"], item["ece"]))
    _copy_selected(Path(selected["path"]), root)
    summary = {
        "agent": agent,
        "status": "trained",
        "selected_seed": selected["seed"],
        "candidates": candidates,
        "train_count": int(len(train.labels)),
        "validation_count": int(len(validation.labels)),
        "output_dir": str(root),
    }
    _dump(root / "training_summary.json", summary)
    return summary


def recalibrate_deep_v2_agent(
    dataset_dir: str | Path,
    model_dir: str | Path,
    agent: str,
    *,
    batch_size: int = 512,
) -> dict[str, Any]:
    import torch

    root = Path(model_dir)
    metadata_path = root / "metadata.json"
    model_path = root / "model.pt"
    if not metadata_path.exists() or not model_path.exists():
        raise ValueError(f"incomplete selected deep_v2 artifact: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_data = dict(metadata["tcn_config"])
    config_data["dilations"] = tuple(config_data["dilations"])
    config = TCNConfig(**config_data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_tcn(config).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    dataset_root = Path(dataset_dir)
    validation = load_v2_tensor_dataset(
        dataset_root / "flows",
        dataset_root / "splits" / "split-manifest.json",
        "validation",
        agent,
    )
    calibration_mask, policy_mask = _partition_validation(validation.sample_ids)
    logits, _ = _predict(model, validation, device, batch_size)
    temperature = _temperature_scale(
        logits[calibration_mask],
        validation.labels[calibration_mask],
    )
    probabilities = 1 / (1 + np.exp(-logits / max(temperature, 1e-6)))
    policy = _select_policy(
        probabilities[policy_mask],
        validation.labels[policy_mask],
    )
    brier = float(
        np.mean((probabilities - validation.labels.astype(np.float32)) ** 2)
    )
    ece = _ece(probabilities, validation.labels)
    metadata["temperature"] = temperature
    metadata["calibration_quality"] = max(0.0, min(1.0, 1 - ece))
    metadata["decision_policy"] = policy
    metadata["posthoc_recalibrated"] = True
    _dump(metadata_path, metadata)
    report = {
        "seed": metadata.get("seed"),
        "brier_score": brier,
        "ece": ece,
        "temperature": temperature,
        "decision_policy": policy,
        "posthoc_recalibrated": True,
    }
    _dump(root / "calibration_report.json", report)
    return report


def train_v2_detectors(
    ustc_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    base_stats_model_dir: str | Path | None = None,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 12,
    batch_size: int = 512,
    consistency_weight: float = 0.2,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if base_stats_model_dir is not None:
        target = root / "stats"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(Path(base_stats_model_dir), target)
    results = {}
    for agent in V2_TRAINING_AGENTS:
        results[agent] = train_deep_v2_agent(
            ustc_dataset_dir,
            root / agent,
            agent,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            consistency_weight=consistency_weight,
            limit=limit,
        )
    summary = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_training",
        "agents": results,
        "base_stats_copied": base_stats_model_dir is not None,
        "external_dataset_used_for_thresholds": False,
        "locked_test_used": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(root / "training_summary.json", summary)
    return summary
