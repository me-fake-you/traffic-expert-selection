from __future__ import annotations

import math
import statistics

from .schemas import FlowRecord, ReliabilityProfile
from .view_contract import populated_tls_fields


def _relative_error(observed: float, expected: float) -> float:
    denominator = max(abs(observed), abs(expected), 1.0)
    return abs(observed - expected) / denominator


class PseudoFeatureGuardAgent:
    name = "PseudoFeatureGuardAgent"
    version = "1.1"

    def __init__(
        self,
        *,
        routing_policy: str = "legacy",
        tls_backend: str = "rule",
    ) -> None:
        self.routing_policy = routing_policy
        self.tls_backend = tls_backend

    def assess(self, flow: FlowRecord) -> ReliabilityProfile:
        stats = flow.stats
        seq = flow.sequence
        indicators: list[str] = []

        stats_rel = 1.0 if stats else 0.0
        sequence_rel = 1.0 if seq.packet_lengths else 0.0
        tls_rel = self._tls_reliability(flow.tls)
        payload_rel = 0.0 if not flow.payload_tokens else 0.6

        lengths = seq.packet_lengths
        directions = seq.directions
        iats = seq.iats

        if lengths:
            if directions and len(directions) != len(lengths):
                sequence_rel -= 0.35
                indicators.append("packet length and direction sequence mismatch")
            if iats and len(iats) not in {len(lengths), len(lengths) - 1}:
                sequence_rel -= 0.25
                indicators.append("packet length and IAT sequence mismatch")

            if not seq.truncated:
                derived_count = float(len(lengths))
                derived_bytes = float(sum(lengths))
                derived_mean = derived_bytes / max(derived_count, 1)
                derived_duration = float(sum(iats))
                consistency_checks = [
                    ("packet_count", derived_count),
                    ("total_bytes", derived_bytes),
                    ("mean_packet_length", derived_mean),
                    ("duration", derived_duration),
                ]
                for key, expected in consistency_checks:
                    if key in stats and _relative_error(stats[key], expected) > 0.2:
                        stats_rel -= 0.18
                        sequence_rel -= 0.08
                        indicators.append(f"{key} conflicts with packet sequence")
            elif seq.original_packet_count is None:
                sequence_rel -= 0.1
                indicators.append("truncated sequence lacks original packet count")

            small_ratio = sum(length <= 16 for length in lengths) / len(lengths)
            if len(lengths) >= 8 and small_ratio >= 0.35:
                sequence_rel -= 0.2
                stats_rel -= 0.1
                indicators.append("high tiny-packet ratio may indicate dummy insertion")

            repeated_ratio = max(lengths.count(item) for item in set(lengths)) / len(lengths)
            if len(lengths) >= 12 and repeated_ratio >= 0.8:
                sequence_rel -= 0.12
                indicators.append("packet sizes are unusually repetitive")
            aligned_ratio = sum(length % 64 == 0 for length in lengths) / len(lengths)
            if len(lengths) >= 8 and aligned_ratio >= 0.6:
                sequence_rel -= 0.18
                stats_rel -= 0.12
                indicators.append("block-aligned packet sizes may indicate padding")

        if iats and len(iats) >= 6:
            mean_iat = statistics.fmean(iats)
            if mean_iat > 0:
                cv = statistics.pstdev(iats) / mean_iat
                if cv > 1.5:
                    sequence_rel -= 0.15
                    indicators.append("extreme IAT dispersion")
            zero_ratio = sum(value == 0 for value in iats) / len(iats)
            if zero_ratio >= 0.2:
                sequence_rel -= 0.15
                indicators.append("IAT jitter produced an unusual zero-delay ratio")
            if any(value > 3600 for value in iats):
                sequence_rel -= 0.2
                indicators.append("extreme IAT outlier")

        if stats:
            if stats.get("packet_count", 1) <= 0 or stats.get("duration", 0) < 0:
                stats_rel -= 0.5
                indicators.append("invalid flow statistics")
            if any(abs(value) > 1e12 for value in stats.values()):
                stats_rel -= 0.4
                indicators.append("out-of-range statistical feature")

        # Stats and packet sequences are the MVP's required views. TLS and payload
        # are optional enrichments, so their absence must not make every USTC flow
        # unknowable.
        completeness = (
            0.1
            + 0.45 * bool(stats)
            + 0.45 * bool(lengths)
        )
        key_missing = not stats or not lengths
        ood = any(
            text
            for text in indicators
            if "out-of-range" in text or "extreme" in text or "invalid" in text
        )
        return ReliabilityProfile(
            stats_reliability=max(0.0, min(1.0, stats_rel)),
            sequence_reliability=max(0.0, min(1.0, sequence_rel)),
            tls_reliability=tls_rel,
            payload_reliability=payload_rel,
            input_completeness=completeness,
            ood_suspected=ood,
            key_features_missing=key_missing,
            indicators=indicators,
        )

    def _tls_reliability(self, tls: dict[str, object]) -> float:
        if not tls:
            return 0.0
        if self.routing_policy in {"contract_v2_9", "capability_v3_0"}:
            present = len(
                populated_tls_fields(tls, backend=self.tls_backend)
            )
            return 0.0 if present == 0 else min(1.0, 0.35 + 0.13 * present)
        useful = {
            "version",
            "cipher_suite",
            "ja3",
            "ja4",
            "certificate_valid",
            "handshake_complete",
        }
        present = sum(key in tls and tls[key] not in {None, ""} for key in useful)
        return min(1.0, 0.35 + 0.13 * present)
