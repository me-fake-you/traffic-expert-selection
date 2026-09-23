from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, pstdev
from typing import Callable

import numpy as np

from .schemas import DetectorInput


FEATURE_SCHEMA_VERSION = "1.1"

STATS_FEATURE_NAMES = (
    "log_packet_count",
    "log_total_bytes",
    "log_outbound_bytes",
    "log_inbound_bytes",
    "outbound_ratio",
    "direction_imbalance",
    "log_mean_packet_length",
    "log_packet_length_variance",
    "log_duration",
    "log_bytes_per_packet",
    "log_bytes_per_second",
)

TEMPORAL_FEATURE_NAMES = (
    "sequence_length",
    "log_original_packet_count",
    "truncated",
    "length_mean",
    "length_std",
    "length_min",
    "length_max",
    "length_q25",
    "length_median",
    "length_q75",
    "length_repeated_ratio",
    "length_tiny_ratio",
    "length_large_ratio",
    "length_entropy",
    "direction_outbound_ratio",
    "direction_switch_ratio",
    "direction_longest_run_ratio",
    "iat_mean",
    "iat_std",
    "iat_cv",
    "iat_min",
    "iat_max",
    "iat_q25",
    "iat_median",
    "iat_q75",
    "iat_zero_ratio",
    "iat_periodicity_score",
    "burst_count",
    "burst_mean",
    "burst_max",
)


def _log1p(value: float | None) -> float:
    if value is None:
        return math.nan
    return math.log1p(max(0.0, float(value)))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _distribution_entropy(values: list[int]) -> float:
    if not values:
        return math.nan
    counts = Counter(values)
    total = len(values)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )
    maximum = math.log2(max(1, len(counts)))
    return entropy / maximum if maximum else 0.0


def _longest_run_ratio(values: list[int]) -> float:
    if not values:
        return math.nan
    longest = current = 1
    for left, right in zip(values, values[1:]):
        current = current + 1 if left == right else 1
        longest = max(longest, current)
    return longest / len(values)


def extract_stats_features(detector_input: DetectorInput) -> np.ndarray:
    """Extract behavior-only statistics from an already audited input."""

    if not isinstance(detector_input, DetectorInput):
        raise TypeError("feature extraction requires an audited DetectorInput")
    stats = detector_input.stats
    packet_count = stats.get("packet_count")
    total_bytes = stats.get("total_bytes")
    outbound_bytes = stats.get("outbound_bytes")
    inbound_bytes = stats.get("inbound_bytes")
    duration = stats.get("duration")
    outbound_ratio = stats.get("outbound_ratio")
    if outbound_ratio is None and total_bytes is not None:
        outbound_ratio = float(outbound_bytes or 0) / max(float(total_bytes), 1.0)
    mean_length = stats.get("mean_packet_length")
    if mean_length is None and packet_count and total_bytes is not None:
        mean_length = float(total_bytes) / max(float(packet_count), 1.0)
    bytes_per_packet = (
        float(total_bytes) / max(float(packet_count), 1.0)
        if total_bytes is not None and packet_count is not None
        else None
    )
    bytes_per_second = (
        float(total_bytes) / max(float(duration), 1e-6)
        if total_bytes is not None and duration is not None
        else None
    )
    ratio = float(outbound_ratio) if outbound_ratio is not None else math.nan
    return np.asarray(
        [
            _log1p(packet_count),
            _log1p(total_bytes),
            _log1p(outbound_bytes),
            _log1p(inbound_bytes),
            ratio,
            abs(ratio - 0.5) if math.isfinite(ratio) else math.nan,
            _log1p(mean_length),
            _log1p(stats.get("packet_length_variance")),
            _log1p(duration),
            _log1p(bytes_per_packet),
            _log1p(bytes_per_second),
        ],
        dtype=np.float32,
    )


def extract_temporal_features(detector_input: DetectorInput) -> np.ndarray:
    """Extract sequence behavior without consulting context, labels, or provenance."""

    if not isinstance(detector_input, DetectorInput):
        raise TypeError("feature extraction requires an audited DetectorInput")
    seq = detector_input.sequence
    lengths = [float(value) for value in seq.packet_lengths]
    directions = list(seq.directions)
    iats = [float(value) for value in seq.iats]
    bursts = [float(value) for value in seq.bursts]

    length_mean = fmean(lengths) if lengths else math.nan
    length_std = pstdev(lengths) if len(lengths) >= 2 else 0.0 if lengths else math.nan
    repeated_ratio = (
        max(Counter(lengths).values()) / len(lengths) if lengths else math.nan
    )
    outbound_ratio = (
        sum(direction == 1 for direction in directions) / len(directions)
        if directions
        else math.nan
    )
    switch_ratio = (
        sum(left != right for left, right in zip(directions, directions[1:]))
        / (len(directions) - 1)
        if len(directions) >= 2
        else 0.0 if directions else math.nan
    )

    iat_mean = fmean(iats) if iats else math.nan
    iat_std = pstdev(iats) if len(iats) >= 2 else 0.0 if iats else math.nan
    iat_cv = (
        iat_std / iat_mean
        if iats and iat_mean > 0
        else 0.0 if iats else math.nan
    )
    periodicity = (
        1.0 / (1.0 + iat_cv)
        if iats and math.isfinite(iat_cv)
        else math.nan
    )
    original_count = seq.original_packet_count
    if original_count is None:
        original_count = len(lengths)

    return np.asarray(
        [
            float(len(lengths)),
            _log1p(float(original_count)),
            float(seq.truncated),
            length_mean,
            length_std,
            min(lengths) if lengths else math.nan,
            max(lengths) if lengths else math.nan,
            _quantile(lengths, 0.25),
            _quantile(lengths, 0.5),
            _quantile(lengths, 0.75),
            repeated_ratio,
            sum(value <= 16 for value in lengths) / len(lengths)
            if lengths
            else math.nan,
            sum(value >= 1400 for value in lengths) / len(lengths)
            if lengths
            else math.nan,
            _distribution_entropy([int(value) for value in lengths]),
            outbound_ratio,
            switch_ratio,
            _longest_run_ratio(directions),
            iat_mean,
            iat_std,
            iat_cv,
            min(iats) if iats else math.nan,
            max(iats) if iats else math.nan,
            _quantile(iats, 0.25),
            _quantile(iats, 0.5),
            _quantile(iats, 0.75),
            sum(value == 0 for value in iats) / len(iats)
            if iats
            else math.nan,
            periodicity,
            float(len(bursts)),
            fmean(bursts) if bursts else 0.0,
            max(bursts) if bursts else 0.0,
        ],
        dtype=np.float32,
    )


def feature_names(agent: str) -> tuple[str, ...]:
    if agent == "stats":
        return STATS_FEATURE_NAMES
    if agent == "temporal":
        return TEMPORAL_FEATURE_NAMES
    raise ValueError(f"unsupported learned detector: {agent}")


def feature_extractor(agent: str) -> Callable[[DetectorInput], np.ndarray]:
    if agent == "stats":
        return extract_stats_features
    if agent == "temporal":
        return extract_temporal_features
    raise ValueError(f"unsupported learned detector: {agent}")
