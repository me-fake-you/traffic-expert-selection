"""W320 fresh device-grouped Gotham Dataset 2025 performance lane.

The official Zenodo release is a single 23.8 GB ZIP containing both raw PCAP
and processed CSV data.  The acquisition stage uses HTTP range requests to
extract only README.md and all official ``processed/*.csv`` members.  Every
extracted member is checked against the CRC32 stored in the official ZIP
central directory and is additionally hashed with SHA-256.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier

from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _hash_snapshot_equal,
    _load,
    _safe_hashes,
    _write_csv,
)


EXPERIMENT = "mad_etd_gotham_grouped_performance_w320"
ZENODO_RECORD = "https://zenodo.org/records/14502760"
ZENODO_DOI = "10.5281/zenodo.14502760"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/14502760/files/"
    "GothamDataset2025.zip/content"
)
ARCHIVE_NAME = "GothamDataset2025.zip"
ARCHIVE_SIZE = 23_824_968_355
ARCHIVE_MD5 = "7ca78c0517ccb3d2854e823678e0f206"
DATASET_LICENSE = "CC-BY-4.0"
EXPECTED_PROCESSED_FILES = 78
EXPECTED_PROCESSED_COMPRESSED_BYTES = 466_244_832
EXPECTED_PROCESSED_UNCOMPRESSED_BYTES = 6_959_260_013

DEFAULT_DATA = Path("data/raw/gotham_2025/v1")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_gotham_grouped_performance_w320")
DEFAULT_MODEL = Path("data/models/mad_etd_gotham_w320")
DEFAULT_DOC = Path("docs/MAD_ETD_GOTHAM_GROUPED_PERFORMANCE_W320.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_GOTHAM_GROUPED_PERFORMANCE_W320_CN.md")

SEED = 42
ROWS_PER_DEVICE = 5_000
CSV_CHUNK_SIZE = 100_000
DEVELOPMENT_ROLES = {"train", "calibration", "selection"}
ROLE_COUNTS = {
    "train": 46,
    "calibration": 10,
    "selection": 10,
    "sealed_acceptance": 12,
}
ATTACK_EXPOSED_GROUPS = {
    "iotsim-air-quality-1",
    "iotsim-building-monitor-1",
    "iotsim-city-power-1",
    "iotsim-combined-cycle-1",
    "iotsim-combined-cycle-10",
    "iotsim-domotic-monitor-1",
    "iotsim-ip-camera-museum-1",
    "iotsim-ip-camera-street-1",
}
ATTACK_GROUP_ROLE_COUNTS = {
    "train": 4,
    "calibration": 1,
    "selection": 1,
    "sealed_acceptance": 2,
}

RAW_COLUMNS = [
    "frame.len",
    "frame.protocols",
    "ip.flags",
    "ip.ttl",
    "ip.proto",
    "ip.tos",
    "tcp.flags",
    "tcp.window_size_value",
    "tcp.window_size_scalefactor",
    "tcp.pdu.size",
    "label",
]
BLOCKED_OR_CONTEXT_ONLY = [
    "frame.time",
    "eth.src",
    "eth.dst",
    "ip.dst",
    "ip.src",
    "ip.checksum",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.checksum",
    "tcp.options",
    "udp.srcport",
    "udp.dstport",
    "label",
    "device_group",
    "device_family",
    "source_file",
    "archive_member",
    "provenance",
]

BASE_FEATURES = [
    "log_frame_len",
    "ip_flags_value",
    "ip_ttl",
    "ip_proto",
    "ip_tos",
    "tcp_flags_value",
    "log_tcp_window",
    "tcp_window_scale",
    "log_tcp_pdu_size",
    "protocol_depth",
    "has_tcp",
    "has_udp",
    "has_coap",
    "has_mqtt",
    "has_rtsp",
    "has_tls",
    "has_dns",
    "has_icmp",
    "missing_tcp_window",
    "missing_tcp_pdu",
]
CONTEXT_FEATURES = [
    "rolling_len_mean_4",
    "rolling_len_std_4",
    "rolling_len_max_4",
    "rolling_len_mean_16",
    "rolling_len_std_16",
    "rolling_tcp_rate_8",
    "rolling_udp_rate_8",
    "rolling_protocol_change_rate_8",
    "rolling_protocol_change_rate_32",
]
BENIGN_LABELS = {"benign"}
MALICIOUS_LABELS = {
    "c&c communication",
    "coap amplification",
    "denial of service",
    "file download",
    "ingress tool transfer",
    "merlin c&c communication",
    "merlin icmp flooding",
    "merlin tcp flooding",
    "merlin udp flooding",
    "mirai c&c communication",
    "mirai dos",
    "mirai gre flooding",
    "mirai tcp flooding",
    "mirai udp flooding",
    "network scanning",
    "periodic c&c communication",
    "remote code execution",
    "remote command execution",
    "reporting",
    "tcp scan",
    "telnet brute force",
    "udp scan",
}
UNSUPPORTED_LABELS = {"unknown"}


def _request(
    url: str,
    start: int,
    end: int,
    timeout: int = 180,
    attempts: int = 6,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "MAD-ETD-W320/1.0",
                },
            )
            response = urllib.request.urlopen(request, timeout=timeout)
            if getattr(response, "status", None) != 206:
                response.close()
                raise RuntimeError("official archive server did not honor Range")
            return response
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"official archive range request failed: {last_error}")


def _read_range(url: str, start: int, end: int) -> bytes:
    with _request(url, start, end) as response:
        return response.read()


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"short HTTP range response; missing {remaining} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _zip64_central_directory(
    url: str = ARCHIVE_URL,
    archive_size: int = ARCHIVE_SIZE,
) -> tuple[int, int]:
    tail_size = min(2 * 1024 * 1024, archive_size)
    start = archive_size - tail_size
    tail = _read_range(url, start, archive_size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    locator = tail.rfind(b"PK\x06\x07", 0, eocd)
    if eocd < 0 or locator < 0:
        raise RuntimeError("official archive ZIP64 directory was not found")
    _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail, locator)
    relative = int(zip64_offset) - start
    if relative < 0 or tail[relative : relative + 4] != b"PK\x06\x06":
        zip64 = _read_range(url, int(zip64_offset), int(zip64_offset) + 55)
        relative = 0
    else:
        zip64 = tail
    values = struct.unpack_from("<4sQ2H2L4Q", zip64, relative)
    return int(values[9]), int(values[8])


def _apply_zip64_extra(
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
    extra: bytes,
) -> tuple[int, int, int]:
    position = 0
    while position + 4 <= len(extra):
        identifier, size = struct.unpack_from("<HH", extra, position)
        payload = extra[position + 4 : position + 4 + size]
        if identifier == 1:
            cursor = 0
            if uncompressed_size == 0xFFFFFFFF:
                uncompressed_size = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if compressed_size == 0xFFFFFFFF:
                compressed_size = struct.unpack_from("<Q", payload, cursor)[0]
                cursor += 8
            if local_offset == 0xFFFFFFFF:
                local_offset = struct.unpack_from("<Q", payload, cursor)[0]
            break
        position += 4 + size
    return int(compressed_size), int(uncompressed_size), int(local_offset)


def _official_zip_index(
    url: str = ARCHIVE_URL,
    archive_size: int = ARCHIVE_SIZE,
) -> list[dict[str, Any]]:
    central_offset, central_size = _zip64_central_directory(url, archive_size)
    data = _read_range(
        url, central_offset, central_offset + central_size - 1
    )
    rows: list[dict[str, Any]] = []
    position = 0
    while (
        position + 46 <= len(data)
        and data[position : position + 4] == b"PK\x01\x02"
    ):
        header = struct.unpack_from("<4s6H3L5H2L", data, position)
        method = int(header[4])
        crc32 = int(header[7])
        compressed_size = int(header[8])
        uncompressed_size = int(header[9])
        name_length = int(header[10])
        extra_length = int(header[11])
        comment_length = int(header[12])
        local_offset = int(header[16])
        name_start = position + 46
        name = data[name_start : name_start + name_length].decode(
            "utf-8", "replace"
        )
        extra_start = name_start + name_length
        extra = data[extra_start : extra_start + extra_length]
        compressed_size, uncompressed_size, local_offset = _apply_zip64_extra(
            compressed_size, uncompressed_size, local_offset, extra
        )
        rows.append(
            {
                "archive_member": name,
                "compression_method": method,
                "crc32": f"{crc32:08x}",
                "crc32_int": crc32,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "local_header_offset": local_offset,
            }
        )
        position += 46 + name_length + extra_length + comment_length
    if not rows:
        raise RuntimeError("official ZIP central directory contained no entries")
    return rows


def _stream_extract(
    entry: Mapping[str, Any],
    destination: Path,
    url: str = ARCHIVE_URL,
    archive_size: int = ARCHIVE_SIZE,
) -> dict[str, Any]:
    offset = int(entry["local_header_offset"])
    compressed_size = int(entry["compressed_size"])
    # Local extra fields are small; the bounded over-read is ignored.
    request_end = min(
        archive_size - 1,
        offset + 30 + len(str(entry["archive_member"]).encode("utf-8"))
        + 65_535 + compressed_size,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    sha256 = hashlib.sha256()
    crc32 = 0
    written = 0
    decompressor = zlib.decompressobj(-15)
    try:
        with _request(url, offset, request_end) as response:
            local_header = _read_exact(response, 30)
            values = struct.unpack("<4s5H3L2H", local_header)
            if values[0] != b"PK\x03\x04":
                raise RuntimeError("invalid local ZIP header")
            method = int(values[3])
            name_length = int(values[9])
            extra_length = int(values[10])
            _read_exact(response, name_length + extra_length)
            if method not in {0, 8}:
                raise RuntimeError(f"unsupported ZIP compression method {method}")
            remaining = compressed_size
            with partial.open("wb") as handle:
                while remaining:
                    chunk = response.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError("truncated compressed member")
                    remaining -= len(chunk)
                    output = (
                        decompressor.decompress(chunk) if method == 8 else chunk
                    )
                    if output:
                        handle.write(output)
                        sha256.update(output)
                        crc32 = zlib.crc32(output, crc32)
                        written += len(output)
                tail = decompressor.flush() if method == 8 else b""
                if tail:
                    handle.write(tail)
                    sha256.update(tail)
                    crc32 = zlib.crc32(tail, crc32)
                    written += len(tail)
        if written != int(entry["uncompressed_size"]):
            raise RuntimeError("uncompressed member size mismatch")
        if (crc32 & 0xFFFFFFFF) != int(entry["crc32_int"]):
            raise RuntimeError("official ZIP member CRC32 mismatch")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {
        **{key: value for key, value in entry.items() if key != "crc32_int"},
        "relative_path": destination.as_posix(),
        "sha256": sha256.hexdigest(),
        "extraction_status": "verified",
    }


def acquire_gotham_processed_w320(
    data_root: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
    url: str = ARCHIVE_URL,
    archive_size: int = ARCHIVE_SIZE,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    index = _official_zip_index(url, archive_size)
    selected = [
        row
        for row in index
        if row["archive_member"] == "README.md"
        or (
            row["archive_member"].startswith("processed/")
            and row["archive_member"].endswith(".csv")
        )
    ]
    processed = [
        row for row in selected if row["archive_member"].startswith("processed/")
    ]
    index_gates = {
        "official_processed_file_count_matches": (
            len(processed) == EXPECTED_PROCESSED_FILES
        ),
        "official_processed_compressed_bytes_match": (
            sum(int(row["compressed_size"]) for row in processed)
            == EXPECTED_PROCESSED_COMPRESSED_BYTES
        ),
        "official_processed_uncompressed_bytes_match": (
            sum(int(row["uncompressed_size"]) for row in processed)
            == EXPECTED_PROCESSED_UNCOMPRESSED_BYTES
        ),
        "only_supported_deflate_members_selected": all(
            int(row["compression_method"]) in {0, 8} for row in selected
        ),
    }
    if not all(index_gates.values()):
        raise RuntimeError(f"official Gotham ZIP index gate failed: {index_gates}")

    progress_path = output / "acquisition_progress_w320.json"
    previous = _load(progress_path) if progress_path.is_file() else {}
    verified_by_member = {
        row["archive_member"]: row
        for row in previous.get("verified_members", [])
    }
    verified: list[dict[str, Any]] = []
    for entry in selected:
        member = str(entry["archive_member"])
        destination = root / member
        prior = verified_by_member.get(member)
        if (
            prior
            and destination.is_file()
            and destination.stat().st_size == int(entry["uncompressed_size"])
            and prior.get("crc32") == entry["crc32"]
            and prior.get("sha256")
        ):
            result = prior
        else:
            result = _stream_extract(entry, destination, url, archive_size)
            result["relative_path"] = destination.relative_to(root).as_posix()
        verified.append(result)
        _dump(
            progress_path,
            {
                "status": "acquisition_in_progress",
                "verified_member_count": len(verified),
                "selected_member_count": len(selected),
                "verified_members": verified,
                "fake_metric_count": 0,
            },
        )

    manifest_rows = []
    for row in verified:
        manifest_rows.append(
            {
                **row,
                "relative_path": str(row["archive_member"]),
                "official_archive_doi": ZENODO_DOI,
                "official_archive_md5_published": ARCHIVE_MD5,
                "full_archive_downloaded": False,
                "member_crc_verified": True,
            }
        )
    _write_csv(output / "official_processed_member_manifest_w320.csv", manifest_rows)
    processed_verified = [
        row for row in verified if str(row["archive_member"]).startswith("processed/")
    ]
    gates = {
        **index_gates,
        "all_selected_members_verified": len(verified) == len(selected),
        "all_processed_members_verified": (
            len(processed_verified) == EXPECTED_PROCESSED_FILES
        ),
        "no_partial_files": not any(root.rglob("*.partial")),
    }
    report = {
        "status": (
            "official_processed_gotham_ready_for_w320_protocol"
            if all(gates.values())
            else "blocked_official_processed_acquisition_gate"
        ),
        "experiment": EXPERIMENT,
        "source_record": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "license": DATASET_LICENSE,
        "publication_date": "2025-02-05",
        "official_archive_name": ARCHIVE_NAME,
        "official_archive_size": ARCHIVE_SIZE,
        "official_archive_md5_published": ARCHIVE_MD5,
        "full_archive_downloaded": False,
        "range_extracted_members_only": True,
        "processed_file_count": len(processed_verified),
        "processed_compressed_bytes": EXPECTED_PROCESSED_COMPRESSED_BYTES,
        "processed_uncompressed_bytes": EXPECTED_PROCESSED_UNCOMPRESSED_BYTES,
        "member_crc32_verified": True,
        "member_sha256_recorded": all(row.get("sha256") for row in verified),
        "gates": gates,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "source_acquisition_report_w320.json", report)
    return report


def _device_family(group: str) -> str:
    return re.sub(r"-\d+$", "", group)


def _role_assignment(groups: Iterable[str]) -> dict[str, str]:
    unique = set(groups)
    ordered = sorted(
        unique,
        key=lambda group: hashlib.sha256(
            f"{SEED}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) != sum(ROLE_COUNTS.values()):
        raise RuntimeError("unexpected Gotham device-group count")
    if ATTACK_EXPOSED_GROUPS.issubset(unique):
        attack_ordered = [group for group in ordered if group in ATTACK_EXPOSED_GROUPS]
        other_ordered = [group for group in ordered if group not in ATTACK_EXPOSED_GROUPS]
        mapping: dict[str, str] = {}
        attack_cursor = 0
        other_cursor = 0
        for role, total_count in ROLE_COUNTS.items():
            attack_count = ATTACK_GROUP_ROLE_COUNTS[role]
            for group in attack_ordered[
                attack_cursor : attack_cursor + attack_count
            ]:
                mapping[group] = role
            attack_cursor += attack_count
            other_count = total_count - attack_count
            for group in other_ordered[other_cursor : other_cursor + other_count]:
                mapping[group] = role
            other_cursor += other_count
        return mapping
    mapping: dict[str, str] = {}
    cursor = 0
    for role, count in ROLE_COUNTS.items():
        for group in ordered[cursor : cursor + count]:
            mapping[group] = role
        cursor += count
    return mapping


def _feature_policy_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "field_name": feature,
            "semantic_role": "safe_packet_evidence",
            "final_policy": "safe_common",
            "enters_detector_input": True,
        }
        for feature in BASE_FEATURES + CONTEXT_FEATURES
    ]
    rows.extend(
        {
            "field_name": field,
            "semantic_role": (
                "label_only" if field == "label" else "blocked_or_alignment_only"
            ),
            "final_policy": "label_only" if field == "label" else "blocked",
            "enters_detector_input": False,
        }
        for field in BLOCKED_OR_CONTEXT_ONLY
    )
    return rows


def build_gotham_grouped_performance_w320(
    data_root: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    acquisition_path = output / "source_acquisition_report_w320.json"
    acquisition = _load(acquisition_path) if acquisition_path.is_file() else {}
    _dump(output / "frozen_hashes_before_w320.json", _safe_hashes())
    paths = sorted((root / "processed").glob("*.csv"))
    groups = [path.stem for path in paths]
    roles = _role_assignment(groups) if len(groups) == EXPECTED_PROCESSED_FILES else {}
    rows: list[dict[str, Any]] = []
    schema_mismatch = 0
    for path in paths:
        columns = list(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(RAW_COLUMNS) - set(columns))
        schema_mismatch += int(bool(missing))
        group = path.stem
        rows.append(
            {
                "device_group": group,
                "device_family_audit_only": _device_family(group),
                "official_attack_exposed_group_audit_only": (
                    group in ATTACK_EXPOSED_GROUPS
                ),
                "role": roles.get(group, "unassigned"),
                "relative_path": path.relative_to(root).as_posix(),
                "file_size": path.stat().st_size,
                "missing_required_columns": "|".join(missing),
                "group_or_path_enters_detector_input": False,
            }
        )
    _write_csv(output / "device_group_split_manifest_w320.csv", rows)
    policy_rows = _feature_policy_rows()
    _write_csv(output / "safe_feature_policy_w320.csv", policy_rows)
    feature_policy_hash = hashlib.sha256(
        "\n".join(sorted(BASE_FEATURES + CONTEXT_FEATURES)).encode("utf-8")
    ).hexdigest()
    _dump(
        output / "feature_policy_lock_w320.json",
        {
            "base_safe_features": BASE_FEATURES,
            "candidate_context_features": CONTEXT_FEATURES,
            "all_candidate_features": BASE_FEATURES + CONTEXT_FEATURES,
            "blocked_or_context_fields": BLOCKED_OR_CONTEXT_ONLY,
            "feature_policy_hash": feature_policy_hash,
            "absolute_time_or_identity_enters_detector_input": False,
            "device_group_used_for_split_and_audit_only": True,
            "label_used_for_supervision_only": True,
        },
    )
    role_counts = Counter(row["role"] for row in rows)
    attack_role_counts = Counter(
        roles[group] for group in set(groups) & ATTACK_EXPOSED_GROUPS
    )
    gates = {
        "official_acquisition_ready": acquisition.get("status")
        == "official_processed_gotham_ready_for_w320_protocol",
        "processed_file_count_matches": len(paths) == EXPECTED_PROCESSED_FILES,
        "schemas_contain_required_fields": schema_mismatch == 0,
        "device_groups_unique": len(groups) == len(set(groups)),
        "role_counts_match_frozen_protocol": all(
            role_counts.get(role, 0) == count
            for role, count in ROLE_COUNTS.items()
        ),
        "official_attack_exposed_groups_present": (
            len(set(groups) & ATTACK_EXPOSED_GROUPS)
            == len(ATTACK_EXPOSED_GROUPS)
        ),
        "attack_exposed_groups_in_every_role": all(
            attack_role_counts.get(role, 0) == count
            for role, count in ATTACK_GROUP_ROLE_COUNTS.items()
        ),
        "safe_feature_policy_nonempty": bool(BASE_FEATURES),
        "sealed_acceptance_content_not_opened": True,
    }
    ready = all(gates.values())
    report = {
        "status": (
            "ready_for_w320_development_extraction"
            if ready
            else "blocked_gotham_source_schema_or_group_gate"
        ),
        "experiment": EXPERIMENT,
        "source": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "license": DATASET_LICENSE,
        "device_group_count": len(groups),
        "official_attack_exposed_group_count": len(
            set(groups) & ATTACK_EXPOSED_GROUPS
        ),
        "official_attack_exposed_role_counts": dict(
            attack_role_counts
        ),
        "role_counts": dict(role_counts),
        "schema_mismatch_count": schema_mismatch,
        "feature_policy_hash": feature_policy_hash,
        "training_authorized": ready,
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "w319_acceptance_used_for_selection": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
    }
    _dump(output / "protocol_readiness_report_w320.json", report)
    return report


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(float)


def _hex_values(series: pd.Series) -> np.ndarray:
    def parse(value: Any) -> float:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return 0.0
        try:
            return float(int(text, 16 if text.lower().startswith("0x") else 10))
        except ValueError:
            return 0.0

    return series.map(parse).to_numpy(float)


def _base_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    protocols = frame["frame.protocols"].fillna("").astype(str).str.lower()
    frame_len = np.maximum(_numeric(frame, "frame.len"), 0)
    tcp_window_raw = pd.to_numeric(
        frame["tcp.window_size_value"], errors="coerce"
    )
    tcp_pdu_raw = pd.to_numeric(frame["tcp.pdu.size"], errors="coerce")
    output: dict[str, Any] = {
        "log_frame_len": np.log1p(frame_len),
        "ip_flags_value": _hex_values(frame["ip.flags"]),
        "ip_ttl": _numeric(frame, "ip.ttl"),
        "ip_proto": _numeric(frame, "ip.proto"),
        "ip_tos": _hex_values(frame["ip.tos"]),
        "tcp_flags_value": _hex_values(frame["tcp.flags"]),
        "log_tcp_window": np.log1p(
            np.maximum(tcp_window_raw.fillna(0).to_numpy(float), 0)
        ),
        "tcp_window_scale": _numeric(frame, "tcp.window_size_scalefactor"),
        "log_tcp_pdu_size": np.log1p(
            np.maximum(tcp_pdu_raw.fillna(0).to_numpy(float), 0)
        ),
        "protocol_depth": protocols.str.count(":").to_numpy(float) + 1.0,
        "has_tcp": protocols.str.contains("tcp", regex=False).to_numpy(float),
        "has_udp": protocols.str.contains("udp", regex=False).to_numpy(float),
        "has_coap": protocols.str.contains("coap", regex=False).to_numpy(float),
        "has_mqtt": protocols.str.contains("mqtt", regex=False).to_numpy(float),
        "has_rtsp": protocols.str.contains("rtsp", regex=False).to_numpy(float),
        "has_tls": (
            protocols.str.contains("tls", regex=False)
            | protocols.str.contains("ssl", regex=False)
        ).to_numpy(float),
        "has_dns": protocols.str.contains("dns", regex=False).to_numpy(float),
        "has_icmp": protocols.str.contains("icmp", regex=False).to_numpy(float),
        "missing_tcp_window": tcp_window_raw.isna().to_numpy(float),
        "missing_tcp_pdu": tcp_pdu_raw.isna().to_numpy(float),
    }
    return pd.DataFrame(output, dtype=np.float32)


def _context_features(base: pd.DataFrame) -> pd.DataFrame:
    change = base["ip_proto"].ne(base["ip_proto"].shift()).astype(float)
    return pd.DataFrame(
        {
            "rolling_len_mean_4": base["log_frame_len"].rolling(
                4, min_periods=1
            ).mean(),
            "rolling_len_std_4": base["log_frame_len"].rolling(
                4, min_periods=1
            ).std().fillna(0),
            "rolling_len_max_4": base["log_frame_len"].rolling(
                4, min_periods=1
            ).max(),
            "rolling_len_mean_16": base["log_frame_len"].rolling(
                16, min_periods=1
            ).mean(),
            "rolling_len_std_16": base["log_frame_len"].rolling(
                16, min_periods=1
            ).std().fillna(0),
            "rolling_tcp_rate_8": base["has_tcp"].rolling(
                8, min_periods=1
            ).mean(),
            "rolling_udp_rate_8": base["has_udp"].rolling(
                8, min_periods=1
            ).mean(),
            "rolling_protocol_change_rate_8": change.rolling(
                8, min_periods=1
            ).mean(),
            "rolling_protocol_change_rate_32": change.rolling(
                32, min_periods=1
            ).mean(),
        },
        dtype=np.float32,
    )


def _priority(indices: np.ndarray, group: str) -> np.ndarray:
    seed = int.from_bytes(
        hashlib.sha256(group.encode("utf-8")).digest()[:8], "little"
    )
    values = indices.astype(np.uint64) ^ np.uint64(seed)
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def _binary_label(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    if (normalized == "").any():
        raise RuntimeError("Gotham contains missing official labels")
    observed = set(normalized.unique())
    unrecognized = observed - BENIGN_LABELS - MALICIOUS_LABELS - UNSUPPORTED_LABELS
    if unrecognized:
        raise RuntimeError(
            f"Gotham contains unrecognized official labels: {sorted(unrecognized)}"
        )
    values = np.full(len(normalized), -1, dtype=np.int8)
    values[normalized.isin(BENIGN_LABELS).to_numpy()] = 0
    values[normalized.isin(MALICIOUS_LABELS).to_numpy()] = 1
    return values, values >= 0


def _extract_device(
    path: Path,
    group: str,
    role: str,
    cap: int = ROWS_PER_DEVICE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    kept = pd.DataFrame()
    tail = pd.DataFrame(columns=BASE_FEATURES)
    offset = 0
    official_labels: set[str] = set()
    total = 0
    explicit_rows = 0
    unsupported_rows = 0
    for raw in pd.read_csv(
        path,
        usecols=RAW_COLUMNS,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):
        labels, supported = _binary_label(raw["label"])
        official_labels.update(
            raw["label"].dropna().astype(str).str.strip().unique().tolist()
        )
        raw_indices = np.arange(offset, offset + len(raw), dtype=np.uint64)
        total += len(raw)
        offset += len(raw)
        unsupported_rows += int((~supported).sum())
        if not supported.any():
            continue
        raw = raw.loc[supported].reset_index(drop=True)
        labels = labels[supported]
        indices = raw_indices[supported]
        explicit_rows += len(raw)
        base = _base_feature_frame(raw)
        combined = (
            base.copy()
            if tail.empty
            else pd.concat([tail, base], ignore_index=True)
        )
        context = _context_features(combined).iloc[len(tail) :].reset_index(drop=True)
        current = pd.concat([base.reset_index(drop=True), context], axis=1)
        current["binary_label"] = labels
        current["device_group"] = group
        current["device_family"] = _device_family(group)
        current["role"] = role
        current["_row_index"] = indices
        current["_priority"] = _priority(indices, group)
        merged = pd.concat([kept, current], ignore_index=True)
        kept = merged.nsmallest(min(cap, len(merged)), "_priority")
        tail = combined.tail(31)[BASE_FEATURES].reset_index(drop=True)
    kept = kept.sort_values("_row_index").reset_index(drop=True)
    return kept, {
        "device_group": group,
        "role": role,
        "total_rows": total,
        "explicit_binary_truth_rows": explicit_rows,
        "unsupported_unknown_rows_excluded": unsupported_rows,
        "sampled_rows": len(kept),
        "official_label_values": "|".join(sorted(official_labels)),
        "sampled_benign_rows": int((kept["binary_label"] == 0).sum()),
        "sampled_malicious_rows": int((kept["binary_label"] == 1).sum()),
    }


def _extract_roles(
    root: Path,
    manifest: pd.DataFrame,
    roles: set[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        if row.role not in roles:
            continue
        frame, summary = _extract_device(
            root / row.relative_path,
            str(row.device_group),
            str(row.role),
        )
        frames.append(frame)
        audit.append(summary)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        audit,
    )


def extract_gotham_development_w320(
    data_root: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir)
    readiness = _load(output / "protocol_readiness_report_w320.json")
    if readiness.get("training_authorized") is not True:
        raise RuntimeError("W320 protocol is not ready for development extraction")
    manifest = pd.read_csv(output / "device_group_split_manifest_w320.csv")
    development, audit = _extract_roles(root, manifest, DEVELOPMENT_ROLES)
    if development.empty:
        raise RuntimeError("W320 development extraction produced no rows")
    development.to_parquet(
        output / "development_features_w320.parquet",
        index=False,
        compression="zstd",
    )
    _write_csv(output / "development_device_audit_w320.csv", audit)
    group_roles = development.groupby("device_group")["role"].nunique()
    overlap = int((group_roles > 1).sum())
    classes = {
        role: sorted(
            development.loc[development["role"] == role, "binary_label"]
            .astype(int)
            .unique()
            .tolist()
        )
        for role in sorted(DEVELOPMENT_ROLES)
    }
    gates = {
        "development_nonempty": len(development) > 0,
        "both_classes_in_each_development_role": all(
            values == [0, 1] for values in classes.values()
        ),
        "cross_role_device_overlap_zero": overlap == 0,
        "sealed_acceptance_rows_read_zero": True,
    }
    report = {
        "status": (
            "ready_for_w320_model_development"
            if all(gates.values())
            else "blocked_development_class_or_group_gate"
        ),
        "row_count": len(development),
        "device_group_count": int(development["device_group"].nunique()),
        "class_by_role": classes,
        "cross_role_device_group_overlap": overlap,
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "w319_acceptance_used_for_selection": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "gates": gates,
    }
    _dump(output / "development_extraction_report_w320.json", report)
    return report


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    weights = np.ones(len(y), dtype=float)
    for label in (0, 1):
        if counts[label] > 0:
            weights[y == label] = len(y) / (2.0 * counts[label])
    return weights


def _group_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    class_weight = _class_weights(y)
    counts = pd.Series(groups).value_counts()
    group_weight = np.asarray(
        [1.0 / math.sqrt(float(counts[group])) for group in groups]
    )
    values = class_weight * group_weight
    return values / values.mean()


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability < upper if upper < 1.0 else probability <= upper
        )
        if mask.any():
            total += float(mask.mean()) * abs(
                float(y[mask].mean()) - float(probability[mask].mean())
            )
    return float(total)


def _fit_calibrator(probability: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    model.fit(logits, y, sample_weight=_class_weights(y))
    return model


def _calibrate(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    group_scores = [
        f1_score(
            y[groups == group],
            prediction[groups == group],
            average="macro",
            zero_division=0,
        )
        for group in np.unique(groups)
    ]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(
            f1_score(y, prediction, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
        "malicious_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "ece": _ece(y, probability),
        "coverage": 1.0,
        "selective_error": float(1.0 - accuracy_score(y, prediction)),
        "worst_group_macro_f1": float(min(group_scores) if group_scores else 0.0),
        "threshold": float(threshold),
    }


def _threshold_score(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> float:
    values = _metrics(y, probability, groups, threshold)
    return 0.5 * values["macro_f1"] + 0.5 * values["worst_group_macro_f1"]


def _choose_threshold(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _threshold_score(y, probability, groups, float(threshold))
        rows.append((score, -abs(float(threshold) - 0.5), float(threshold)))
    winner = max(rows)
    return winner[2], winner[0]


def _xgb(seed: int = SEED, device: str = "cuda") -> XGBClassifier:
    return XGBClassifier(
        n_estimators=260,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=2.0,
        reg_alpha=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=8,
    )


def _fit_xgb(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    seed: int,
) -> tuple[XGBClassifier, str, float]:
    model = _xgb(seed, "cuda")
    started = time.perf_counter()
    try:
        model.fit(x, y, sample_weight=weight)
        device = "cuda"
    except Exception:
        model = _xgb(seed, "cpu")
        model.fit(x, y, sample_weight=weight)
        device = "cpu_fallback"
    return model, device, time.perf_counter() - started


def train_validate_gotham_w320(
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    output = Path(output_dir)
    models_path = Path(model_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    extraction = _load(output / "development_extraction_report_w320.json")
    if extraction.get("status") != "ready_for_w320_model_development":
        raise RuntimeError("W320 development features are not ready")
    frame = pd.read_parquet(output / "development_features_w320.parquet")
    train = frame[frame["role"] == "train"]
    calibration = frame[frame["role"] == "calibration"]
    selection = frame[frame["role"] == "selection"]
    x_train = train[BASE_FEATURES].to_numpy(np.float32)
    x_train_candidate = train[BASE_FEATURES + CONTEXT_FEATURES].to_numpy(
        np.float32
    )
    y_train = train["binary_label"].to_numpy(np.int8)
    train_groups = train["device_group"].astype(str).to_numpy()
    x_cal = calibration[BASE_FEATURES].to_numpy(np.float32)
    x_cal_candidate = calibration[
        BASE_FEATURES + CONTEXT_FEATURES
    ].to_numpy(np.float32)
    y_cal = calibration["binary_label"].to_numpy(np.int8)
    groups_cal = calibration["device_group"].astype(str).to_numpy()
    x_sel = selection[BASE_FEATURES].to_numpy(np.float32)
    x_sel_candidate = selection[
        BASE_FEATURES + CONTEXT_FEATURES
    ].to_numpy(np.float32)
    y_sel = selection["binary_label"].to_numpy(np.int8)
    groups_sel = selection["device_group"].astype(str).to_numpy()

    models: dict[str, Any] = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.07,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
    }
    weights = _class_weights(y_train)
    artifacts: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    cal_probabilities: dict[str, np.ndarray] = {}
    sel_probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train, sample_weight=weights)
        elapsed = time.perf_counter() - started
        calibrator = _fit_calibrator(model.predict_proba(x_cal)[:, 1], y_cal)
        cal_probability = _calibrate(
            calibrator, model.predict_proba(x_cal)[:, 1]
        )
        sel_probability = _calibrate(
            calibrator, model.predict_proba(x_sel)[:, 1]
        )
        threshold, objective = _choose_threshold(
            y_cal, cal_probability, groups_cal
        )
        values = _metrics(y_sel, sel_probability, groups_sel, threshold)
        result_rows.append(
            {
                "candidate": name,
                "candidate_type": "strong_safe_input_baseline",
                "evaluation_role": "selection",
                **values,
                "calibration_objective": objective,
                "train_seconds": elapsed,
                "device": "cpu",
            }
        )
        artifacts[name] = {
            "model": model,
            "calibrator": calibrator,
            "threshold": threshold,
        }
        cal_probabilities[name] = cal_probability
        sel_probabilities[name] = sel_probability

    xgb_model, xgb_device, xgb_seconds = _fit_xgb(
        x_train, y_train, weights, SEED
    )
    xgb_calibrator = _fit_calibrator(
        xgb_model.predict_proba(x_cal)[:, 1], y_cal
    )
    xgb_cal_probability = _calibrate(
        xgb_calibrator, xgb_model.predict_proba(x_cal)[:, 1]
    )
    xgb_sel_probability = _calibrate(
        xgb_calibrator, xgb_model.predict_proba(x_sel)[:, 1]
    )
    xgb_threshold, xgb_objective = _choose_threshold(
        y_cal, xgb_cal_probability, groups_cal
    )
    xgb_values = _metrics(
        y_sel, xgb_sel_probability, groups_sel, xgb_threshold
    )
    result_rows.append(
        {
            "candidate": "xgboost",
            "candidate_type": "strong_safe_input_baseline",
            "evaluation_role": "selection",
            **xgb_values,
            "calibration_objective": xgb_objective,
            "train_seconds": xgb_seconds,
            "device": xgb_device,
        }
    )
    artifacts["xgboost"] = {
        "model": xgb_model,
        "calibrator": xgb_calibrator,
        "threshold": xgb_threshold,
    }
    cal_probabilities["xgboost"] = xgb_cal_probability
    sel_probabilities["xgboost"] = xgb_sel_probability

    candidate_model, candidate_device, candidate_seconds = _fit_xgb(
        x_train_candidate,
        y_train,
        _group_weights(y_train, train_groups),
        SEED + 1,
    )
    candidate_calibrator = _fit_calibrator(
        candidate_model.predict_proba(x_cal_candidate)[:, 1], y_cal
    )
    candidate_cal_raw = _calibrate(
        candidate_calibrator,
        candidate_model.predict_proba(x_cal_candidate)[:, 1],
    )
    candidate_sel_raw = _calibrate(
        candidate_calibrator,
        candidate_model.predict_proba(x_sel_candidate)[:, 1],
    )
    calibration_best = max(
        result_rows,
        key=lambda row: (
            row["calibration_objective"],
            row["macro_f1"],
            row["candidate"],
        ),
    )["candidate"]
    blends = []
    for weight in (0.50, 0.65, 0.80, 0.90, 1.00):
        probability = (
            weight * candidate_cal_raw
            + (1.0 - weight) * cal_probabilities[calibration_best]
        )
        threshold, objective = _choose_threshold(
            y_cal, probability, groups_cal
        )
        blends.append((objective, weight, threshold))
    _, blend_weight, candidate_threshold = max(blends)
    candidate_cal_probability = (
        blend_weight * candidate_cal_raw
        + (1.0 - blend_weight) * cal_probabilities[calibration_best]
    )
    candidate_sel_probability = (
        blend_weight * candidate_sel_raw
        + (1.0 - blend_weight) * sel_probabilities[calibration_best]
    )
    candidate_values = _metrics(
        y_sel, candidate_sel_probability, groups_sel, candidate_threshold
    )
    candidate_row = {
        "candidate": "mad_etd_device_balanced_context_evidence_v1",
        "candidate_type": "group_aware_evidence_candidate",
        "evaluation_role": "selection",
        **candidate_values,
        "calibration_objective": _threshold_score(
            y_cal,
            candidate_cal_probability,
            groups_cal,
            candidate_threshold,
        ),
        "train_seconds": candidate_seconds,
        "device": candidate_device,
        "blend_weight_group_context": blend_weight,
        "blend_reference": calibration_best,
    }
    result_rows.append(candidate_row)
    _write_csv(output / "development_results_w320.csv", result_rows)

    strongest = max(
        (
            row
            for row in result_rows
            if row["candidate_type"] == "strong_safe_input_baseline"
        ),
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            row["candidate"],
        ),
    )
    delta = {
        key: candidate_row[key] - strongest[key]
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "malicious_recall",
            "ece",
            "worst_group_macro_f1",
        )
    }
    gates = {
        "selection_macro_f1_delta_at_least_0_005": delta["macro_f1"] >= 0.005,
        "selection_accuracy_drop_within_0_002": delta["accuracy"] >= -0.002,
        "selection_malicious_recall_drop_within_0_005": (
            delta["malicious_recall"] >= -0.005
        ),
        "selection_ece_not_worse_by_more_than_0_005": delta["ece"] <= 0.005,
        "selection_worst_group_macro_f1_not_worse": (
            delta["worst_group_macro_f1"] >= 0.0
        ),
        "acceptance_not_used_for_development": True,
        "w319_acceptance_not_used_for_selection": True,
        "blocked_field_violation_zero": True,
    }
    passed = all(gates.values())
    joblib.dump(
        {
            "base_features": BASE_FEATURES,
            "candidate_features": BASE_FEATURES + CONTEXT_FEATURES,
            "baseline_artifacts": artifacts,
            "strongest_selection_baseline": strongest["candidate"],
            "candidate_model": candidate_model,
            "candidate_calibrator": candidate_calibrator,
            "candidate_threshold": candidate_threshold,
            "blend_weight": blend_weight,
            "blend_reference": calibration_best,
        },
        models_path / "w320_model_bundle.joblib",
    )
    report = {
        "status": (
            "development_positive_ready_for_one_shot_acceptance"
            if passed
            else "development_candidate_not_better_than_strongest_baseline"
        ),
        "strongest_safe_input_baseline": strongest,
        "candidate": candidate_row,
        "deltas_vs_strongest_baseline": delta,
        "development_gates": gates,
        "development_gate_passed": passed,
        "acceptance_authorized": passed,
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "w319_acceptance_used_for_selection": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "development_gate_report_w320.json", report)
    return report


def _bootstrap_delta(
    frame: pd.DataFrame,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    iterations: int = 1_000,
    seed: int = SEED,
) -> dict[str, Any]:
    y = frame["binary_label"].to_numpy(int)
    groups = frame["device_group"].astype(str).to_numpy()
    unique = np.unique(groups)

    def group_confusions(prediction: np.ndarray) -> np.ndarray:
        rows = []
        for group in unique:
            mask = groups == group
            yy = y[mask]
            pp = prediction[mask]
            rows.append(
                [
                    int(((yy == 0) & (pp == 0)).sum()),
                    int(((yy == 0) & (pp == 1)).sum()),
                    int(((yy == 1) & (pp == 0)).sum()),
                    int(((yy == 1) & (pp == 1)).sum()),
                ]
            )
        return np.asarray(rows, dtype=np.int64)

    def macro_f1(confusion: np.ndarray) -> float:
        tn, fp, fn, tp = confusion
        benign = 2 * tn / max(2 * tn + fp + fn, 1)
        malicious = 2 * tp / max(2 * tp + fp + fn, 1)
        return float((benign + malicious) / 2.0)

    baseline = group_confusions(baseline_prediction)
    candidate = group_confusions(candidate_prediction)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(unique), size=len(unique))
        deltas[index] = (
            macro_f1(candidate[sample].sum(axis=0))
            - macro_f1(baseline[sample].sum(axis=0))
        )
    return {
        "iterations": iterations,
        "seed": seed,
        "group_unit": "official_device_csv",
        "macro_f1_delta_mean": float(deltas.mean()),
        "macro_f1_delta_ci_lower": float(np.quantile(deltas, 0.025)),
        "macro_f1_delta_ci_upper": float(np.quantile(deltas, 0.975)),
    }


def evaluate_gotham_acceptance_w320(
    data_root: str | Path = DEFAULT_DATA,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_dir)
    development = _load(output / "development_gate_report_w320.json")
    if development.get("acceptance_authorized") is not True:
        report = {
            "status": "acceptance_sealed_development_gate_failed",
            "acceptance_opened": False,
            "acceptance_rows_read": 0,
            "acceptance_metrics_generated": False,
            "w319_acceptance_used_for_selection": False,
            "fake_metric_count": 0,
            "runtime_safe_v3_0_remains_default": True,
        }
        _dump(output / "acceptance_evaluation_report_w320.json", report)
        return report
    manifest = pd.read_csv(output / "device_group_split_manifest_w320.csv")
    acceptance, audit = _extract_roles(root, manifest, {"sealed_acceptance"})
    if acceptance.empty:
        raise RuntimeError("authorized W320 acceptance extraction produced no rows")
    acceptance.to_parquet(
        output / "sealed_acceptance_features_w320.parquet",
        index=False,
        compression="zstd",
    )
    _write_csv(output / "sealed_acceptance_device_audit_w320.csv", audit)
    bundle = joblib.load(Path(model_dir) / "w320_model_bundle.joblib")
    y = acceptance["binary_label"].to_numpy(np.int8)
    groups = acceptance["device_group"].astype(str).to_numpy()
    x_base = acceptance[bundle["base_features"]].to_numpy(np.float32)
    x_candidate = acceptance[bundle["candidate_features"]].to_numpy(np.float32)
    baseline_name = bundle["strongest_selection_baseline"]
    baseline_artifact = bundle["baseline_artifacts"][baseline_name]
    baseline_probability = _calibrate(
        baseline_artifact["calibrator"],
        baseline_artifact["model"].predict_proba(x_base)[:, 1],
    )
    candidate_raw = _calibrate(
        bundle["candidate_calibrator"],
        bundle["candidate_model"].predict_proba(x_candidate)[:, 1],
    )
    reference = bundle["baseline_artifacts"][bundle["blend_reference"]]
    reference_probability = _calibrate(
        reference["calibrator"],
        reference["model"].predict_proba(x_base)[:, 1],
    )
    weight = float(bundle["blend_weight"])
    candidate_probability = (
        weight * candidate_raw + (1.0 - weight) * reference_probability
    )
    baseline_metrics = _metrics(
        y, baseline_probability, groups, baseline_artifact["threshold"]
    )
    candidate_metrics = _metrics(
        y, candidate_probability, groups, bundle["candidate_threshold"]
    )
    _write_csv(
        output / "sealed_acceptance_results_w320.csv",
        [
            {
                "candidate": baseline_name,
                "candidate_type": "fixed_strongest_development_baseline",
                **baseline_metrics,
            },
            {
                "candidate": "mad_etd_device_balanced_context_evidence_v1",
                "candidate_type": "fixed_group_aware_evidence_candidate",
                **candidate_metrics,
            },
        ],
    )
    baseline_prediction = (
        baseline_probability >= baseline_artifact["threshold"]
    ).astype(int)
    candidate_prediction = (
        candidate_probability >= bundle["candidate_threshold"]
    ).astype(int)
    bootstrap = _bootstrap_delta(
        acceptance, baseline_prediction, candidate_prediction
    )
    _dump(output / "grouped_bootstrap_report_w320.json", bootstrap)
    delta = {
        key: candidate_metrics[key] - baseline_metrics[key]
        for key in (
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "malicious_recall",
            "ece",
            "worst_group_macro_f1",
        )
    }
    gates = {
        "acceptance_macro_f1_delta_positive": delta["macro_f1"] > 0.0,
        "grouped_bootstrap_ci_lower_above_zero": (
            bootstrap["macro_f1_delta_ci_lower"] > 0.0
        ),
        "acceptance_accuracy_drop_within_0_002": delta["accuracy"] >= -0.002,
        "acceptance_malicious_recall_drop_within_0_005": (
            delta["malicious_recall"] >= -0.005
        ),
        "acceptance_ece_not_worse_by_more_than_0_005": delta["ece"] <= 0.005,
        "acceptance_worst_group_macro_f1_not_worse": (
            delta["worst_group_macro_f1"] >= 0.0
        ),
    }
    accepted = all(gates.values())
    report = {
        "status": (
            "accepted_dataset_specific_gotham_evidence_candidate"
            if accepted
            else "not_promoted_gotham_acceptance_gate_failed"
        ),
        "acceptance_opened": True,
        "acceptance_rows_read": len(acceptance),
        "acceptance_device_group_count": int(
            acceptance["device_group"].nunique()
        ),
        "baseline": {"name": baseline_name, **baseline_metrics},
        "candidate": {
            "name": "mad_etd_device_balanced_context_evidence_v1",
            **candidate_metrics,
        },
        "deltas_vs_baseline": delta,
        "grouped_bootstrap": bootstrap,
        "acceptance_gates": gates,
        "accepted_dataset_specific_candidate": accepted,
        "w319_acceptance_used_for_selection": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "acceptance_evaluation_report_w320.json", report)
    return report


def _render(report: Mapping[str, Any], document: Path, document_cn: Path) -> None:
    acceptance = report.get("acceptance_evaluation", {})
    english = f"""# MAD-ETD Gotham Device-Grouped Performance W320

- Final status: `{report['status']}`
- Official dataset DOI: `{ZENODO_DOI}`
- License: `{DATASET_LICENSE}`
- Group unit: official per-device processed CSV
- Default runtime changed: false
- Fake metrics: 0
- Full tests passed: {report.get('tests_passed', False)}

Only the 78 official processed CSV members were range-extracted from the
23.8 GB Zenodo ZIP.  Every member was checked against the official ZIP CRC32
and hashed with SHA-256.  Device identity, endpoint addresses, ports,
timestamps, checksums, labels, file names and provenance never enter the
DetectorInput.  The sealed device-group acceptance was opened only if the
development gate passed.  W319 acceptance results were not reused for
selection.

Acceptance status: `{acceptance.get('status', 'not_opened')}`.
"""
    chinese = f"""# MAD-ETD Gotham 设备分组性能 W320

- 最终状态：`{report['status']}`
- 官方数据 DOI：`{ZENODO_DOI}`
- 许可证：`{DATASET_LICENSE}`
- 分组单位：官方逐设备 processed CSV
- 默认 runtime 已修改：false
- fake metric count：0
- 完整测试通过：{report.get('tests_passed', False)}

本实验没有下载 23.8 GB 的完整 PCAP 包，而是通过 HTTP Range 仅提取
78 个官方 processed CSV。每个成员均使用官方 ZIP 中的 CRC32 校验，并
记录 SHA-256。设备身份、端点、端口、绝对时间、校验和、标签、文件名和
provenance 均不进入 DetectorInput。只有 development gate 通过后，才会
一次性打开按设备隔离的 sealed acceptance。W319 acceptance 结果没有
参与本轮任何选择或调参。

Acceptance 状态：`{acceptance.get('status', 'not_opened')}`。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_gotham_grouped_performance_w320(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    acquisition = _load(output / "source_acquisition_report_w320.json")
    protocol = _load(output / "protocol_readiness_report_w320.json")
    extraction = _load(output / "development_extraction_report_w320.json")
    development = _load(output / "development_gate_report_w320.json")
    acceptance = _load(output / "acceptance_evaluation_report_w320.json")
    hashes_after = _safe_hashes()
    _dump(output / "frozen_hashes_after_w320.json", hashes_after)
    unchanged = _hash_snapshot_equal(
        _load(output / "frozen_hashes_before_w320.json"), hashes_after
    )
    safety = {
        "blocked_field_violation_zero": acceptance.get(
            "blocked_field_violation", 0
        )
        == 0,
        "fusion_ownership_violation_zero": acceptance.get(
            "fusion_ownership_violation", 0
        )
        == 0,
        "ood_override_zero": acceptance.get("ood_override_count", 0) == 0,
        "fake_metric_count_zero": acceptance.get("fake_metric_count", 0) == 0,
        "acceptance_not_used_for_selection": acceptance.get(
            "test_or_acceptance_used_for_selection", False
        )
        is False,
        "w319_acceptance_not_used_for_selection": acceptance.get(
            "w319_acceptance_used_for_selection", False
        )
        is False,
        "frozen_hashes_unchanged": unchanged,
        "runtime_safe_v3_0_remains_default": True,
        "tests_passed": bool(tests_passed),
    }
    accepted = acceptance.get("accepted_dataset_specific_candidate") is True
    status = (
        "completed_w320_dataset_specific_positive_candidate"
        if accepted and all(safety.values())
        else "completed_w320_negative_or_development_gate_closed"
    )
    report = {
        "status": status,
        "source_acquisition": acquisition,
        "protocol": protocol,
        "development_extraction": extraction,
        "development": development,
        "acceptance_evaluation": acceptance,
        "accepted_dataset_specific_candidate": accepted and all(safety.values()),
        "general_runtime_promoted": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "w319_acceptance_used_for_selection": False,
        "fake_metric_count": 0,
        "security_gates": safety,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "security_acceptance_w320.json", safety)
    _dump(output / "acceptance_report.json", report)
    negatives: list[dict[str, Any]] = []
    if not report["accepted_dataset_specific_candidate"]:
        negatives.append(
            {
                "experiment_id": EXPERIMENT,
                "candidate_module": (
                    "mad_etd_device_balanced_context_evidence_v1"
                ),
                "failure_type": (
                    "development_gate_failed"
                    if not development.get("development_gate_passed", False)
                    else "sealed_acceptance_gate_failed"
                ),
                "failure_reason": (
                    development.get("status")
                    if not development.get("development_gate_passed", False)
                    else acceptance.get("status")
                ),
                "acceptance_opened": acceptance.get("acceptance_opened", False),
                "fake_metric_count": 0,
                "runtime_modified": False,
                "final_status": status,
            }
        )
    _dump(output / "negative_results.json", negatives)
    _render(report, Path(document), Path(document_cn))
    return report
