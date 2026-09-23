from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schemas import FieldRole


FIELD_CONTRACT_VERSION = "2.8"
FIELD_CONTRACT_VERSIONS = ("2.8", "2.9")
FIELD_CONTRACT_SCHEMA_VERSION = "1.0"


def _entry(
    role: FieldRole,
    *,
    consumers: tuple[str, ...] = (),
    reason: str,
) -> dict[str, Any]:
    return {
        "role": role.value,
        "consumers": list(consumers),
        "reason": reason,
    }


FIELD_CONTRACT: dict[str, dict[str, Any]] = {
    "stats.packet_count": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.total_bytes": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.outbound_bytes": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.inbound_bytes": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.outbound_ratio": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.mean_packet_length": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.packet_length_variance": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "stats.duration": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("StatsDetectorAgent",),
        reason="Frozen v1 statistical feature.",
    ),
    "sequence.packet_lengths": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Audited packet-sequence behavior.",
    ),
    "sequence.directions": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Audited packet direction sequence.",
    ),
    "sequence.iats": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Audited inter-arrival-time sequence.",
    ),
    "sequence.bursts": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Audited behavior-only burst sequence.",
    ),
    "sequence.original_packet_count": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Required to interpret truncation without identity metadata.",
    ),
    "sequence.truncated": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TemporalBehaviorAgent",),
        reason="Sequence quality indicator.",
    ),
    "tls.version": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Protocol-version evidence consumed by the accepted TLS rule.",
    ),
    "tls.server_version": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Server protocol version consumed by the TLS backend.",
    ),
    "tls.record_lengths": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Signed TLS record-length sequence.",
    ),
    "tls.tls_record_lengths": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Legacy alias for signed TLS record-length sequence.",
    ),
    "tls.client_cipher_count": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Count-only TLS handshake feature.",
    ),
    "tls.client_extension_count": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Count-only TLS handshake feature.",
    ),
    "tls.server_extension_count": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Count-only TLS handshake feature.",
    ),
    "tls.alpn": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Controlled ALPN encoding consumed by the TLS backend.",
    ),
    "tls.certificate_valid": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Certificate-quality evidence consumed by the accepted TLS rule.",
    ),
    "tls.handshake_complete": _entry(
        FieldRole.DETECTION_ALLOWED,
        consumers=("TLSProtocolAgent",),
        reason="Handshake-completeness evidence consumed by the accepted TLS rule.",
    ),
    "tls.known_bad_fingerprint": _entry(
        FieldRole.BLOCKED,
        reason="Direct threat-intelligence match can act as a label proxy.",
    ),
    "tls.cipher_suite": _entry(
        FieldRole.CONTEXT_ONLY,
        reason="Exact negotiated cipher is environment-sensitive and unused by accepted detectors.",
    ),
    "tls.selected_cipher": _entry(
        FieldRole.CONTEXT_ONLY,
        reason="Exact negotiated cipher is environment-sensitive and unused by accepted detectors.",
    ),
    "tls.client_version": _entry(
        FieldRole.CONTEXT_ONLY,
        reason="Client-advertised version is retained for audit only.",
    ),
    "context.transport": _entry(
        FieldRole.CONTEXT_ONLY,
        reason="Transport context is not consumed by accepted detectors.",
    ),
    "context.protocols": _entry(
        FieldRole.CONTEXT_ONLY,
        reason="Protocol-name context is not consumed by accepted detectors.",
    ),
    "payload_tokens": _entry(
        FieldRole.BLOCKED,
        reason="Payload representation is disabled in the accepted runtime.",
    ),
}


def contract_entry(
    path: str,
    contract_version: str = FIELD_CONTRACT_VERSION,
) -> dict[str, Any] | None:
    if contract_version not in FIELD_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported field contract: {contract_version}")
    return FIELD_CONTRACT.get(path.lower().replace("-", "_"))


def field_contract_payload(
    contract_version: str = FIELD_CONTRACT_VERSION,
) -> dict[str, Any]:
    if contract_version not in FIELD_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported field contract: {contract_version}")
    payload = {
        "schema_version": FIELD_CONTRACT_SCHEMA_VERSION,
        "contract_version": contract_version,
        "policy": "strict_fail_closed",
        "unknown_populated_field_policy": "UNKNOWN and excluded from DetectorInput",
        "empty_field_policy": "ignored_empty_fields and excluded from DetectorInput",
        "fields": dict(sorted(FIELD_CONTRACT.items())),
    }
    if contract_version == "2.9":
        payload["routing_semantics"] = (
            "contract-native detector-consumer fields"
        )
    return payload


def field_contract_sha256(
    contract_version: str = FIELD_CONTRACT_VERSION,
) -> str:
    encoded = json.dumps(
        field_contract_payload(contract_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_field_contract(
    path: str | Path,
    contract_version: str = FIELD_CONTRACT_VERSION,
) -> dict[str, Any]:
    payload = {
        **field_contract_payload(contract_version),
        "contract_sha256": field_contract_sha256(contract_version),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
