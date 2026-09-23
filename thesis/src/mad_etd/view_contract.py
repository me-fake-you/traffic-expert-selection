from __future__ import annotations

from typing import Any, Literal

from .schemas import (
    DetectorCapabilityProfile,
    DetectorInput,
    ViewAvailabilityProfile,
)


RoutingPolicy = Literal["legacy", "contract_v2_9", "capability_v3_0"]

VERDICT_AGENT_NAMES = {
    "StatsDetectorAgent",
    "TemporalBehaviorAgent",
    "TLSProtocolAgent",
}
ENRICHMENT_AGENT_NAMES = {
    "FamilyAttributionAgent",
    "IntentAgent",
}

TLS_RULE_CONSUMED_FIELDS = (
    "version",
    "certificate_valid",
    "handshake_complete",
)
TLS_DEEP_CONSUMED_FIELDS = (
    "record_lengths",
    "tls_record_lengths",
    "server_version",
    "client_cipher_count",
    "client_extension_count",
    "server_extension_count",
    "alpn",
)
TLS_ALL_BEHAVIOR_FIELDS = tuple(
    dict.fromkeys((*TLS_RULE_CONSUMED_FIELDS, *TLS_DEEP_CONSUMED_FIELDS))
)


def _populated(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def tls_consumed_fields(backend: str) -> tuple[str, ...]:
    return (
        TLS_DEEP_CONSUMED_FIELDS
        if backend in {"deep_v2", "deep_tls_v4"}
        else TLS_RULE_CONSUMED_FIELDS
    )


def populated_tls_fields(
    tls: dict[str, Any],
    *,
    backend: str,
) -> list[str]:
    return [
        key
        for key in tls_consumed_fields(backend)
        if key in tls and _populated(tls[key])
    ]


def build_view_availability(
    detector_input: DetectorInput,
    *,
    routing_policy: RoutingPolicy,
    tls_backend: str,
) -> ViewAvailabilityProfile:
    if routing_policy == "legacy":
        tls_fields = [
            key for key, value in detector_input.tls.items() if _populated(value)
        ]
    else:
        tls_fields = populated_tls_fields(
            detector_input.tls,
            backend=tls_backend,
        )
    stats_fields = [
        key
        for key, value in detector_input.stats.items()
        if _populated(value)
    ]
    sequence_fields = (
        ["packet_lengths"]
        if detector_input.sequence.packet_lengths
        else []
    )
    payload_fields = (
        ["payload_tokens"] if detector_input.payload_tokens else []
    )
    return ViewAvailabilityProfile(
        routing_policy=routing_policy,
        stats=bool(stats_fields),
        sequence=bool(sequence_fields),
        tls=bool(tls_fields),
        payload=bool(payload_fields),
        available_fields={
            "stats": sorted(stats_fields),
            "sequence": sequence_fields,
            "tls": sorted(tls_fields),
            "payload": payload_fields,
        },
    )


def build_detector_capabilities(
    detector_input: DetectorInput,
    *,
    stats_backend: str,
    temporal_backend: str,
    tls_backend: str,
) -> dict[str, DetectorCapabilityProfile]:
    stats_fields = sorted(
        key
        for key, value in detector_input.stats.items()
        if _populated(value)
    )
    sequence_fields = (
        ["sequence.packet_lengths"]
        if detector_input.sequence.packet_lengths
        else []
    )
    tls_observed = sorted(
        key
        for key, value in detector_input.tls.items()
        if _populated(value)
    )
    tls_supported = populated_tls_fields(
        detector_input.tls,
        backend=tls_backend,
    )
    other_tls_behavior = sorted(
        set(tls_observed)
        & (set(TLS_ALL_BEHAVIOR_FIELDS) - set(tls_consumed_fields(tls_backend)))
    )
    if tls_supported:
        tls_status = "available"
        tls_reasons = ["TLS_BACKEND_FIELDS_AVAILABLE"]
    elif other_tls_behavior:
        tls_status = "unsupported_schema"
        tls_reasons = ["TLS_SCHEMA_UNSUPPORTED_BY_BACKEND"]
    else:
        tls_status = "missing_view"
        tls_reasons = ["TLS_VIEW_MISSING"]
    return {
        "StatsDetectorAgent": DetectorCapabilityProfile(
            agent_name="StatsDetectorAgent",
            backend=stats_backend,
            status="available" if stats_fields else "missing_view",
            consumed_fields=[f"stats.{key}" for key in stats_fields],
            available_fields=[f"stats.{key}" for key in stats_fields],
            observed_fields=[f"stats.{key}" for key in stats_fields],
            reason_codes=(
                ["STATS_FIELDS_AVAILABLE"]
                if stats_fields
                else ["STATS_VIEW_MISSING"]
            ),
        ),
        "TemporalBehaviorAgent": DetectorCapabilityProfile(
            agent_name="TemporalBehaviorAgent",
            backend=temporal_backend,
            status="available" if sequence_fields else "missing_view",
            consumed_fields=["sequence.packet_lengths"],
            available_fields=sequence_fields,
            observed_fields=sequence_fields,
            reason_codes=(
                ["SEQUENCE_FIELDS_AVAILABLE"]
                if sequence_fields
                else ["SEQUENCE_VIEW_MISSING"]
            ),
        ),
        "TLSProtocolAgent": DetectorCapabilityProfile(
            agent_name="TLSProtocolAgent",
            backend=tls_backend,
            status=tls_status,
            consumed_fields=[
                f"tls.{key}" for key in tls_consumed_fields(tls_backend)
            ],
            available_fields=[f"tls.{key}" for key in tls_supported],
            observed_fields=[f"tls.{key}" for key in tls_observed],
            unsupported_fields=[f"tls.{key}" for key in other_tls_behavior],
            reason_codes=tls_reasons,
        ),
    }
