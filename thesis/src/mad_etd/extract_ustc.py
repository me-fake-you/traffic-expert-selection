from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .io import write_jsonl
from .schemas import FlowRecord, SequenceFeatures


TSHARK_FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ipv6.src",
    "tcp.srcport",
    "udp.srcport",
    "ip.dst",
    "ipv6.dst",
    "tcp.dstport",
    "udp.dstport",
    "frame.len",
    "_ws.col.Protocol",
    "tcp.stream",
    "udp.stream",
    "ip.proto",
    "tls.handshake.version",
    "tls.handshake.ciphersuite",
]


def resolve_tshark() -> str:
    candidates = [
        os.getenv("MAD_ETD_TSHARK"),
        shutil.which("tshark"),
        str(
            Path(__file__).resolve().parents[2]
            / ".tools"
            / "Wireshark"
            / "Wireshark"
            / "tshark.exe"
        ),
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    raise RuntimeError(
        "tshark is required for PCAP extraction. Install Wireshark/tshark "
        "and ensure tshark is on PATH, or set MAD_ETD_TSHARK to tshark.exe."
    )


def tshark_version() -> str:
    executable = resolve_tshark()
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=True,
    )
    return result.stdout.splitlines()[0].strip()


@dataclass(slots=True)
class FlowBuilder:
    first_timestamp: float
    last_timestamp: float
    origin: tuple[str, str]
    peer: tuple[str, str]
    transport: str
    stream_id: str
    sequence_limit: int
    packet_count: int = 0
    total_bytes: int = 0
    outbound_bytes: int = 0
    length_sum_squares: float = 0
    sequence_lengths: list[int] = field(default_factory=list)
    sequence_directions: list[int] = field(default_factory=list)
    sequence_iats: list[float] = field(default_factory=list)
    sequence_last_timestamp: float | None = None
    protocols: set[str] = field(default_factory=set)
    tls_versions: set[str] = field(default_factory=set)
    cipher_suites: set[str] = field(default_factory=set)
    heap_version: int = 0

    def add(
        self,
        timestamp: float,
        src: tuple[str, str],
        length: int,
        protocol: str,
        tls_version: str,
        cipher_suite: str,
    ) -> None:
        self.last_timestamp = timestamp
        self.packet_count += 1
        self.total_bytes += length
        self.length_sum_squares += length * length
        direction = 1 if src == self.origin else -1
        if direction == 1:
            self.outbound_bytes += length

        if len(self.sequence_lengths) < self.sequence_limit:
            if self.sequence_last_timestamp is not None:
                self.sequence_iats.append(
                    max(0.0, timestamp - self.sequence_last_timestamp)
                )
            self.sequence_lengths.append(length)
            self.sequence_directions.append(direction)
            self.sequence_last_timestamp = timestamp

        if protocol:
            self.protocols.add(protocol)
        if tls_version:
            self.tls_versions.add(tls_version)
        if cipher_suite:
            self.cipher_suites.add(cipher_suite)
        self.heap_version += 1


def _endpoint(ipv4: str, ipv6: str, tcp_port: str, udp_port: str) -> tuple[str, str]:
    return (ipv4 or ipv6 or "unknown", tcp_port or udp_port or "0")


def _stream_key(
    src: tuple[str, str],
    dst: tuple[str, str],
    tcp_stream: str,
    udp_stream: str,
    ip_protocol: str,
) -> tuple[str, str]:
    if tcp_stream != "":
        return "tcp", tcp_stream
    if udp_stream != "":
        return "udp", udp_stream
    left, right = sorted((src, dst))
    digest = hashlib.sha256(
        f"{left}|{right}|{ip_protocol}".encode("utf-8")
    ).hexdigest()[:16]
    return f"ip-{ip_protocol or 'unknown'}", digest


def _iter_packets(pcap: Path) -> Iterator[list[str]]:
    tshark = resolve_tshark()
    command = [
        tshark,
        "-n",
        "-r",
        str(pcap),
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-E",
        "occurrence=f",
    ]
    for field_name in TSHARK_FIELDS:
        command.extend(["-e", field_name])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    for line in process.stdout:
        values = line.rstrip("\n").split("\t")
        yield values + [""] * (len(TSHARK_FIELDS) - len(values))
    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"tshark failed for {pcap}: {stderr.strip()}")


def _labels_for(relative: Path) -> dict[str, str]:
    parts = list(relative.parts)
    lower = [part.lower() for part in parts]
    if "malware" in lower:
        index = lower.index("malware")
        class_part = parts[index + 1] if index + 1 < len(parts) else relative.stem
        family = Path(class_part).stem
        return {"binary": "malicious", "family": family, "application": ""}
    if "benign" in lower:
        index = lower.index("benign")
        class_part = parts[index + 1] if index + 1 < len(parts) else relative.stem
        application = Path(class_part).stem
        return {"binary": "benign", "family": "", "application": application}
    return {"binary": "", "family": "", "application": ""}


def _record_from_builder(
    builder: FlowBuilder,
    *,
    relative: Path,
    segment_index: int,
) -> FlowRecord | None:
    if builder.packet_count < 1:
        return None
    mean_length = builder.total_bytes / builder.packet_count
    variance = max(
        0.0,
        builder.length_sum_squares / builder.packet_count - mean_length * mean_length,
    )
    inbound = builder.total_bytes - builder.outbound_bytes
    digest = hashlib.sha256(
        (
            f"{relative}|{builder.transport}|{builder.stream_id}|"
            f"{builder.first_timestamp:.6f}|{segment_index}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    tls: dict[str, object] = {}
    if builder.tls_versions or builder.cipher_suites:
        tls = {
            "version": sorted(builder.tls_versions)[0]
            if builder.tls_versions
            else "",
            "cipher_suite": sorted(builder.cipher_suites)[0]
            if builder.cipher_suites
            else "",
            "handshake_complete": True,
        }
    return FlowRecord(
        trace_id=f"ustc-{digest}",
        sample_id=digest,
        stats={
            "packet_count": builder.packet_count,
            "total_bytes": builder.total_bytes,
            "outbound_bytes": builder.outbound_bytes,
            "inbound_bytes": inbound,
            "outbound_ratio": builder.outbound_bytes / max(builder.total_bytes, 1),
            "mean_packet_length": mean_length,
            "packet_length_variance": variance,
            "duration": builder.last_timestamp - builder.first_timestamp,
        },
        sequence=SequenceFeatures(
            packet_lengths=builder.sequence_lengths,
            directions=builder.sequence_directions,
            iats=builder.sequence_iats,
            original_packet_count=builder.packet_count,
            truncated=builder.packet_count > len(builder.sequence_lengths),
        ),
        tls=tls,
        context={
            "src_ip": builder.origin[0],
            "src_port": builder.origin[1],
            "dst_ip": builder.peer[0],
            "dst_port": builder.peer[1],
            "transport": builder.transport,
            "stream_id": builder.stream_id,
            "protocols": sorted(builder.protocols),
        },
        provenance={
            "source_file": str(relative),
            "capture_start_epoch": builder.first_timestamp,
            "segment_index": segment_index,
        },
        labels=_labels_for(relative),
    )


def extract_pcap(
    pcap: str | Path,
    *,
    dataset_root: str | Path,
    idle_timeout: float = 60.0,
    sequence_limit: int = 64,
) -> Iterator[FlowRecord]:
    source = Path(pcap)
    root = Path(dataset_root)
    relative = source.relative_to(root)
    active: dict[tuple[str, str], FlowBuilder] = {}
    expiry_heap: list[tuple[float, tuple[str, str], int]] = []
    segment_counts: dict[tuple[str, str], int] = {}

    for values in _iter_packets(source):
        (
            timestamp_text,
            src4,
            src6,
            tcp_src,
            udp_src,
            dst4,
            dst6,
            tcp_dst,
            udp_dst,
            length_text,
            protocol,
            tcp_stream,
            udp_stream,
            ip_protocol,
            tls_version,
            cipher_suite,
        ) = values[: len(TSHARK_FIELDS)]
        if not timestamp_text or not length_text:
            continue
        timestamp = float(timestamp_text)

        while expiry_heap and expiry_heap[0][0] <= timestamp - idle_timeout:
            _, expired_key, version = heapq.heappop(expiry_heap)
            candidate = active.get(expired_key)
            if candidate is None or candidate.heap_version != version:
                continue
            del active[expired_key]
            segment_index = segment_counts.get(expired_key, 0)
            segment_counts[expired_key] = segment_index + 1
            record = _record_from_builder(
                candidate,
                relative=relative,
                segment_index=segment_index,
            )
            if record is not None:
                yield record

        src = _endpoint(src4, src6, tcp_src, udp_src)
        dst = _endpoint(dst4, dst6, tcp_dst, udp_dst)
        key = _stream_key(src, dst, tcp_stream, udp_stream, ip_protocol)
        builder = active.get(key)
        if builder is None:
            builder = FlowBuilder(
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                origin=src,
                peer=dst,
                transport=key[0],
                stream_id=key[1],
                sequence_limit=sequence_limit,
            )
            active[key] = builder
        builder.add(
            timestamp,
            src,
            int(length_text),
            protocol,
            tls_version,
            cipher_suite,
        )
        heapq.heappush(expiry_heap, (builder.last_timestamp, key, builder.heap_version))

    for key, builder in sorted(
        active.items(), key=lambda item: item[1].first_timestamp
    ):
        segment_index = segment_counts.get(key, 0)
        record = _record_from_builder(
            builder,
            relative=relative,
            segment_index=segment_index,
        )
        if record is not None:
            yield record


def extract_dataset(
    dataset_root: str | Path,
    *,
    idle_timeout: float = 60.0,
    sequence_limit: int = 64,
) -> Iterator[FlowRecord]:
    root = Path(dataset_root)
    pcaps = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )
    if not pcaps:
        raise FileNotFoundError(f"no PCAP files found under {root}")
    for pcap in pcaps:
        yield from extract_pcap(
            pcap,
            dataset_root=root,
            idle_timeout=idle_timeout,
            sequence_limit=sequence_limit,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_ustc_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    idle_timeout: float = 60.0,
    sequence_limit: int = 64,
    limit_files: int | None = None,
    resume: bool = True,
) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    target = Path(output_dir).resolve()
    flows_dir = target / "flows"
    manifests_dir = target / "manifests"
    flows_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "extraction_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "flow_schema_version": "1.1",
        "dataset": "USTC-TFC2016",
        "dataset_root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tshark_version": tshark_version(),
        "parameters": {
            "idle_timeout_seconds": idle_timeout,
            "sequence_limit": sequence_limit,
            "flow_key": "tcp.stream/udp.stream with idle segmentation",
        },
        "captures": [],
    }
    existing_by_source: dict[str, dict[str, object]] = {}
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_by_source = {
            item["source_file"]: item for item in existing.get("captures", [])
        }

    pcaps = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )
    if limit_files is not None:
        pcaps = pcaps[:limit_files]
    captures: list[dict[str, object]] = []
    total_flows = 0
    for source in pcaps:
        relative = source.relative_to(root)
        output = flows_dir / relative.parent / f"{relative.name}.jsonl.gz"
        source_key = str(relative)
        previous = existing_by_source.get(source_key)
        if resume and previous and output.exists():
            captures.append(previous)
            total_flows += int(previous["flow_count"])
            continue
        source_hash = _sha256_file(source)
        count = write_jsonl(
            extract_pcap(
                source,
                dataset_root=root,
                idle_timeout=idle_timeout,
                sequence_limit=sequence_limit,
            ),
            output,
        )
        entry = {
            "source_file": source_key,
            "source_size": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "source_sha256": source_hash,
            "output_file": str(output.relative_to(target)),
            "flow_count": count,
        }
        captures.append(entry)
        total_flows += count
        manifest["captures"] = captures
        manifest["total_flows"] = total_flows
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest["captures"] = captures
    manifest["total_flows"] = total_flows
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
