from __future__ import annotations

import random
import statistics

from .schemas import FlowRecord, SequenceFeatures


def recompute_stats(flow: FlowRecord) -> FlowRecord:
    updated = flow.model_copy(deep=True)
    lengths = updated.sequence.packet_lengths
    directions = updated.sequence.directions
    iats = updated.sequence.iats
    if not lengths:
        return updated

    outbound = sum(
        length for length, direction in zip(lengths, directions, strict=False) if direction == 1
    )
    inbound = sum(
        length for length, direction in zip(lengths, directions, strict=False) if direction == -1
    )
    total = sum(lengths)
    updated.stats.update(
        {
            "packet_count": float(len(lengths)),
            "total_bytes": float(total),
            "outbound_bytes": float(outbound),
            "inbound_bytes": float(inbound),
            "outbound_ratio": float(outbound / max(total, 1)),
            "mean_packet_length": float(statistics.fmean(lengths)),
            "packet_length_variance": float(statistics.pvariance(lengths))
            if len(lengths) > 1
            else 0.0,
            "duration": float(sum(iats)),
        }
    )
    updated.sequence.original_packet_count = len(lengths)
    updated.sequence.truncated = False
    return updated


def apply_perturbation(
    flow: FlowRecord,
    kind: str,
    *,
    strength: float = 0.2,
    seed: int = 7,
) -> FlowRecord:
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    rng = random.Random(seed)
    updated = flow.model_copy(deep=True)
    seq = updated.sequence
    original_packet_count = (
        seq.original_packet_count
        if seq.original_packet_count is not None
        else len(seq.packet_lengths)
    )

    if kind == "padding":
        block_size = 64
        seq.packet_lengths = [
            (
                ((length + block_size - 1) // block_size) * block_size
                if rng.random() < strength
                else length
            )
            for length in seq.packet_lengths
        ]
    elif kind == "dummy_packet":
        count = max(1, int(len(seq.packet_lengths) * strength))
        for _ in range(count):
            index = rng.randint(0, len(seq.packet_lengths))
            seq.packet_lengths.insert(index, rng.randint(1, 16))
            direction = rng.choice([-1, 1])
            seq.directions.insert(min(index, len(seq.directions)), direction)
            seq.iats.insert(min(index, len(seq.iats)), rng.random() * 0.01)
    elif kind == "iat_jitter":
        seq.iats = [
            max(0.0, iat * (1 + rng.uniform(-strength, strength) * 3))
            for iat in seq.iats
        ]
    elif kind == "time_scale":
        factor = 1 + strength * 4
        seq.iats = [iat * factor for iat in seq.iats]
    elif kind == "sequence_truncation":
        keep = max(1, int(len(seq.packet_lengths) * (1 - 0.8 * strength)))
        seq.packet_lengths = seq.packet_lengths[:keep]
        seq.directions = seq.directions[:keep]
        seq.iats = seq.iats[: max(0, keep - 1)]
    else:
        raise ValueError(f"unsupported perturbation: {kind}")

    updated.sequence = SequenceFeatures.model_validate(seq.model_dump())
    updated.context["synthetic_perturbation"] = {
        "kind": kind,
        "strength": strength,
        "seed": seed,
    }
    result = recompute_stats(updated)
    if kind == "sequence_truncation":
        result.sequence.original_packet_count = original_packet_count
        result.sequence.truncated = keep < original_packet_count
    return result
