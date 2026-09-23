from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests

from .schemas import FlowRecord, SequenceFeatures


DOHBRW_DATASET = "CIRA-CIC-DoHBrw-2020"
DOHBRW_OFFICIAL_PAGE = "https://www.unb.ca/cic/datasets/dohbrw-2020.html"
TSHARK_VERSION = "4.6.6"
TSHARK_OFFICIAL_URL = (
    "https://2.na.dl.wireshark.org/win64/"
    "WiresharkPortable64_4.6.6.paf.exe"
)
TSHARK_INSTALLER_SHA256 = (
    "a26b74b2c8e10a82b4e9f60f632135151525c1003b34bbbc35dde538df637fa6"
)
CAPTURE_COLUMNS = (
    "capture_id",
    "relative_path",
    "traffic_role",
    "binary_label",
    "generator",
    "resolver",
    "label_source",
)
MALICIOUS_GENERATORS = ("dns2tcp", "dnscat2", "iodine")
BENIGN_GENERATORS = ("chrome", "firefox")
RESOLVERS = ("adguard", "cloudflare", "google", "quad9")
TSHARK_FIELDS = (
    "frame.number",
    "frame.time_epoch",
    "tcp.stream",
    "ip.src",
    "ipv6.src",
    "tls.record.length",
    "tls.handshake.type",
    "tls.handshake.version",
    "tls.handshake.ciphersuites",
    "tls.handshake.extension.type",
    "tls.handshake.extensions_alpn_str",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_id(relative_path: str) -> str:
    stem = Path(relative_path).stem.lower()
    digest = hashlib.sha256(relative_path.lower().encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:40] or "capture"
    return f"{safe}-{digest}"


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _infer_capture(relative_path: str) -> dict[str, str]:
    normalized = _normalize(relative_path)
    if "nondoh" in normalized:
        role, label = "non_doh_ood", ""
    elif "maliciousdoh" in normalized:
        role, label = "malicious_doh", "malicious"
    elif "benigndoh" in normalized:
        role, label = "benign_doh", "benign"
    else:
        role, label = "unresolved", ""
    generators = [
        name
        for name in (*MALICIOUS_GENERATORS, *BENIGN_GENERATORS)
        if name in normalized
    ]
    resolvers = [name for name in RESOLVERS if name in normalized]
    return {
        "capture_id": _capture_id(relative_path),
        "relative_path": relative_path,
        "traffic_role": role,
        "binary_label": label,
        "generator": generators[0] if len(generators) == 1 else "",
        "resolver": resolvers[0] if len(resolvers) == 1 else "",
        "label_source": (
            "official_directory_category" if role != "unresolved" else ""
        ),
    }


def resumable_download(
    url: str,
    output_path: str | Path,
    *,
    expected_sha256: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.stat().st_size if target.exists() else 0
    headers = {"Range": f"bytes={current}-"} if current else {}
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        if current and response.status_code == 200:
            current = 0
            mode = "wb"
        elif response.status_code in {200, 206}:
            mode = "ab" if current else "wb"
        else:
            response.raise_for_status()
        with target.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    digest = _sha256(target)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"download checksum mismatch: {target}")
    return {
        "url": url,
        "path": target.as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": digest,
        "resumable": True,
    }


def _pcap_files(source_dir: Path) -> list[Path]:
    return sorted(
        {
            *source_dir.rglob("*.pcap"),
            *source_dir.rglob("*.pcapng"),
            *source_dir.rglob("*.cap"),
        }
    )


def _write_capture_manifest(source_dir: Path, path: Path) -> list[dict[str, str]]:
    rows = [
        _infer_capture(item.relative_to(source_dir).as_posix())
        for item in _pcap_files(source_dir)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_capture_manifest(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(rows[0]) != CAPTURE_COLUMNS if rows else False:
        raise ValueError("capture manifest columns changed")
    ids = [row["capture_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("capture manifest contains duplicate capture_id")
    for row in rows:
        role = row["traffic_role"]
        label = row["binary_label"]
        if role == "benign_doh" and label != "benign":
            raise ValueError("benign DoH capture has invalid label")
        if role == "malicious_doh" and label != "malicious":
            raise ValueError("malicious DoH capture has invalid label")
        if role == "non_doh_ood" and label:
            raise ValueError("non-DoH capture must remain unlabeled")
        if role not in {
            "benign_doh",
            "malicious_doh",
            "non_doh_ood",
            "unresolved",
        }:
            raise ValueError(f"unsupported traffic role: {role}")
        if label not in {"", "benign", "malicious"}:
            raise ValueError(f"unsupported binary label: {label}")
    return rows


def locate_tshark(
    explicit_path: str | Path | None = None,
) -> Path | None:
    candidates = [
        Path(explicit_path) if explicit_path else None,
        Path("tools/wireshark/4.6.6/portable/App/Wireshark/tshark.exe"),
        Path("tools/wireshark/4.6.6/extracted/App/Wireshark/tshark.exe"),
        Path("C:/Program Files/Wireshark/tshark.exe"),
    ]
    return next((item for item in candidates if item and item.exists()), None)


def tshark_identity(path: str | Path) -> dict[str, Any]:
    executable = Path(path)
    completed = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )
    first = completed.stdout.splitlines()[0] if completed.stdout else ""
    match = re.search(r"TShark.*?(\d+\.\d+\.\d+)", first)
    version = match.group(1) if match else ""
    return {
        "path": executable.as_posix(),
        "sha256": _sha256(executable),
        "version": version,
        "version_line": first,
        "accepted": version == TSHARK_VERSION,
    }


def prepare_dohbrw_v4(
    raw_dir: str | Path = "data/raw/cira_cic_dohbrw_2020/v1",
    processed_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    *,
    source_dir: str | Path | None = None,
    capture_manifest: str | Path | None = None,
    tshark_path: str | Path | None = None,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    raw.mkdir(parents=True, exist_ok=True)
    installer = Path(
        "tools/wireshark/4.6.6/WiresharkPortable64_4.6.6.paf.exe"
    )
    installer_status: dict[str, Any]
    if installer.exists():
        installer_status = {
            "path": installer.as_posix(),
            "sha256": _sha256(installer),
            "expected_sha256": TSHARK_INSTALLER_SHA256,
            "verified": _sha256(installer) == TSHARK_INSTALLER_SHA256,
        }
    else:
        installer_status = resumable_download(
            TSHARK_OFFICIAL_URL,
            installer,
            expected_sha256=TSHARK_INSTALLER_SHA256,
        )
        installer_status["verified"] = True
    source = Path(source_dir) if source_dir else raw / "official_download"
    pcaps = _pcap_files(source) if source.exists() else []
    manifest_path = (
        Path(capture_manifest)
        if capture_manifest
        else processed / "manifests" / "capture_manifest.csv"
    )
    # The default manifest is a derived inventory, not a user-supplied split
    # artifact.  Rebuild it for every readiness pass so a removed, added, or
    # restaged PCAP cannot silently leave a stale capture-group inventory that
    # would later be used for extraction or split freezing.  An explicit
    # manifest remains caller-owned for compatibility with audited custom
    # manifests and is therefore loaded unchanged when it already exists.
    if not pcaps:
        rows = []
        manifest_origin = "not_generated_no_pcaps"
    elif capture_manifest and manifest_path.exists():
        rows = load_capture_manifest(manifest_path)
        manifest_origin = "explicit_existing_manifest"
    else:
        rows = _write_capture_manifest(source, manifest_path)
        manifest_origin = (
            "explicit_generated_manifest"
            if capture_manifest
            else "default_refreshed_from_source"
        )
    tshark = locate_tshark(tshark_path)
    tshark_status = (
        tshark_identity(tshark)
        if tshark
        else {
            "accepted": False,
            "reason": "verified installer present but tshark is not installed",
        }
    )
    unresolved = sum(row["traffic_role"] == "unresolved" for row in rows)
    status = (
        "registration_required"
        if not pcaps
        else "capture_manifest_incomplete"
        if unresolved
        else "tshark_unavailable"
        if not tshark_status["accepted"]
        else "ready_for_extraction"
    )
    report = {
        "schema_version": "1.0",
        "dataset": DOHBRW_DATASET,
        "status": status,
        "official_page": DOHBRW_OFFICIAL_PAGE,
        "download_access": "CIC_registration_form_required",
        "source_dir": source.as_posix(),
        "pcap_count": len(pcaps),
        "capture_manifest": manifest_path.as_posix(),
        "capture_manifest_origin": manifest_origin,
        "capture_count": len(rows),
        "unresolved_capture_count": unresolved,
        "tshark": tshark_status,
        "tshark_installer": installer_status,
        "automatic_label_inference": False,
        "non_doh_supervised": False,
    }
    _dump(processed / "manifests" / "source_manifest.json", report)
    return report


def audit_dohbrw_v4(
    processed_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    *,
    source_dir: str | Path | None = None,
    capture_manifest: str | Path | None = None,
    tshark_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(processed_dir)
    source_manifest_path = root / "manifests" / "source_manifest.json"
    if not source_manifest_path.exists():
        prepare_dohbrw_v4(
            processed_dir=root,
            source_dir=source_dir,
            capture_manifest=capture_manifest,
            tshark_path=tshark_path,
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_path = Path(
        capture_manifest or source_manifest["capture_manifest"]
    )
    rows = load_capture_manifest(manifest_path) if manifest_path.exists() else []
    role_counts = Counter(row["traffic_role"] for row in rows)
    label_counts = Counter(row["binary_label"] for row in rows if row["binary_label"])
    malicious_tools = {
        row["generator"]
        for row in rows
        if row["binary_label"] == "malicious" and row["generator"]
    }
    quad9_labels = {
        row["binary_label"]
        for row in rows
        if row["resolver"] == "quad9" and row["binary_label"]
    }
    checks = {
        "pcaps_available": source_manifest["pcap_count"] > 0,
        "capture_manifest_complete": bool(rows)
        and role_counts.get("unresolved", 0) == 0,
        "both_supervised_classes_present": set(label_counts)
        == {"benign", "malicious"},
        "non_doh_is_unlabeled": all(
            not row["binary_label"]
            for row in rows
            if row["traffic_role"] == "non_doh_ood"
        ),
        "malicious_tools_explicit": malicious_tools
        == set(MALICIOUS_GENERATORS),
        "quad9_has_both_classes": quad9_labels == {"benign", "malicious"},
        "tshark_version_fixed": bool(source_manifest["tshark"].get("accepted")),
    }
    report = {
        "schema_version": "1.0",
        "dataset": DOHBRW_DATASET,
        "status": "accepted_for_extraction"
        if all(checks.values())
        else source_manifest["status"],
        "checks": checks,
        "role_counts": dict(role_counts),
        "label_counts": dict(label_counts),
        "malicious_generators": sorted(malicious_tools),
        "quad9_labels": sorted(quad9_labels),
        "capture_manifest_sha256": (
            _sha256(manifest_path) if manifest_path.exists() else None
        ),
        "supervised_training_enabled": all(checks.values()),
        "automatic_label_inference": False,
    }
    _dump(root / "audits" / "dohbrw_v4_audit.json", report)
    return report


def _split_values(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _parse_tshark_rows(
    path: str | Path,
    capture: dict[str, str],
) -> Iterator[FlowRecord]:
    streams: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "client": None,
            "records": [],
            "first_epoch": None,
            "server_version": "",
            "client_cipher_count": 0,
            "client_extension_count": 0,
            "server_extension_count": 0,
            "alpn": set(),
        }
    )
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            stream = row.get("tcp.stream", "")
            source = row.get("ip.src") or row.get("ipv6.src") or ""
            if not stream or not source:
                continue
            state = streams[stream]
            state["client"] = state["client"] or source
            epoch = row.get("frame.time_epoch", "")
            if epoch and state["first_epoch"] is None:
                state["first_epoch"] = float(epoch)
            sign = 1 if source == state["client"] else -1
            for value in _split_values(row.get("tls.record.length", "")):
                length = int(value)
                if length > 0 and len(state["records"]) < 64:
                    state["records"].append(sign * length)
            handshake_types = set(
                _split_values(row.get("tls.handshake.type", ""))
            )
            versions = _split_values(row.get("tls.handshake.version", ""))
            if versions:
                state["server_version"] = versions[-1]
            ciphers = _split_values(row.get("tls.handshake.ciphersuites", ""))
            if "1" in handshake_types and source == state["client"]:
                state["client_cipher_count"] = max(
                    state["client_cipher_count"],
                    len(ciphers),
                )
            extensions = _split_values(
                row.get("tls.handshake.extension.type", "")
            )
            if source == state["client"]:
                state["client_extension_count"] = max(
                    state["client_extension_count"],
                    len(extensions),
                )
            else:
                state["server_extension_count"] = max(
                    state["server_extension_count"],
                    len(extensions),
                )
            state["alpn"].update(
                _split_values(
                    row.get("tls.handshake.extensions_alpn_str", "")
                )
            )
    for stream, state in sorted(streams.items()):
        if len(state["records"]) < 2:
            continue
        sample_id = hashlib.sha256(
            f"{capture['capture_id']}:{stream}".encode("utf-8")
        ).hexdigest()[:24]
        label = capture["binary_label"]
        labels = {
            "traffic_role": capture["traffic_role"],
            "generator": capture["generator"] or None,
            "resolver": capture["resolver"] or None,
        }
        if label:
            labels["binary"] = label
        yield FlowRecord(
            trace_id=f"dohbrw-{sample_id}",
            sample_id=sample_id,
            stats={},
            sequence=SequenceFeatures(),
            tls={
                "record_lengths": state["records"],
                "server_version": state["server_version"],
                "client_cipher_count": state["client_cipher_count"],
                "client_extension_count": state["client_extension_count"],
                "server_extension_count": state["server_extension_count"],
                "alpn": sorted(state["alpn"])[:8],
            },
            context={"transport": "tcp", "protocols": ["tls", "doh"]},
            provenance={
                "dataset": DOHBRW_DATASET,
                "capture_id": capture["capture_id"],
                "source_file": capture["relative_path"],
                "capture_start_epoch": state["first_epoch"],
                "generator": capture["generator"] or None,
                "resolver": capture["resolver"] or None,
            },
            labels=labels,
        )


def _tshark_execution_environment(plugin_dir: Path) -> dict[str, str]:
    """Return an isolated environment for deterministic TLS-only parsing.

    The portable Wireshark bundle includes optional Falco plugins whose
    third-party DLL dependencies are not shipped with the bundle. TShark can
    decode the PCAP correctly, but reports those unrelated optional-plugin
    failures with a non-zero exit code. The fixed TShark binary is left
    untouched; the extraction process points WIRESHARK_PLUGIN_DIR at a
    verified empty directory instead.
    """
    plugin_dir.mkdir(parents=True, exist_ok=True)
    if any(plugin_dir.iterdir()):
        raise RuntimeError(
            "TLS extraction plugin isolation directory must remain empty"
        )
    environment = os.environ.copy()
    environment["WIRESHARK_PLUGIN_DIR"] = str(plugin_dir.resolve())
    return environment


def _run_tshark(
    tshark: Path,
    pcap: Path,
    output_path: Path,
    *,
    plugin_dir: Path,
) -> None:
    command = [
        str(tshark),
        "-r",
        str(pcap),
        "-o",
        "tcp.desegment_tcp_streams:TRUE",
        "-o",
        "tls.desegment_ssl_records:TRUE",
        "-Y",
        "tls && tcp.stream",
        "-T",
        "fields",
        "-E",
        "header=y",
        "-E",
        "separator=\t",
        "-E",
        "quote=n",
        "-E",
        "occurrence=a",
        "-E",
        "aggregator=,",
    ]
    for field in TSHARK_FIELDS:
        command.extend(["-e", field])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=60 * 60,
            env=_tshark_execution_environment(plugin_dir),
        )


def _extract_capture_tsv(
    capture: dict[str, str],
    *,
    source: Path,
    root: Path,
    tshark: Path,
    plugin_dir: Path,
    quarantine_invalid_pcaps: bool = False,
) -> dict[str, Any]:
    """Create or verify one immutable per-capture TShark extraction state."""
    pcap = source / capture["relative_path"]
    if not pcap.exists():
        raise FileNotFoundError(pcap)
    tsv = root / "tshark" / f"{capture['capture_id']}.tsv"
    state_path = root / "tshark" / f"{capture['capture_id']}.json"
    pcap_sha = _sha256(pcap)
    if state_path.exists() and tsv.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("pcap_sha256") != pcap_sha:
            raise RuntimeError("PCAP changed after partial extraction")
    else:
        try:
            _run_tshark(tshark, pcap, tsv, plugin_dir=plugin_dir)
        except subprocess.CalledProcessError as error:
            diagnostic = str(error.stderr or "")
            error_path = root / "tshark" / f"{capture['capture_id']}.error.log"
            error_path.write_text(diagnostic, encoding="utf-8")
            # A TShark failure can leave an apparently usable partial TSV.
            # It is never admissible as learned-TLS evidence.
            tsv.unlink(missing_ok=True)
            failure = {
                "capture_id": capture["capture_id"],
                "relative_path": capture["relative_path"],
                "pcap_sha256": pcap_sha,
                "tsv_path": None,
                "integrity_status": "quarantined",
                "tshark_exit_code": error.returncode,
                "error_path": error_path.as_posix(),
                "error_kind": (
                    "pcap_truncated"
                    if "appears to have been cut short" in diagnostic.lower()
                    else "tshark_parse_failure"
                ),
            }
            if quarantine_invalid_pcaps:
                return failure
            raise RuntimeError(
                "DoHBrw extraction is fail-closed for "
                f"{capture['relative_path']}: {failure['error_kind']}; "
                f"see {error_path}"
            ) from error
        _dump(
            state_path,
            {
                "capture_id": capture["capture_id"],
                "pcap_sha256": pcap_sha,
                "tshark_sha256": _sha256(tshark),
                "tshark_version": TSHARK_VERSION,
                "command_fields": list(TSHARK_FIELDS),
                "tcp_reassembly": True,
                "tls_record_reassembly": True,
            },
        )
    return {
        "capture_id": capture["capture_id"],
        "relative_path": capture["relative_path"],
        "pcap_sha256": pcap_sha,
        "tsv_path": tsv.as_posix(),
        "integrity_status": "valid",
    }


def extract_dohbrw_v4(
    processed_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    *,
    source_dir: str | Path | None = None,
    capture_manifest: str | Path | None = None,
    tshark_path: str | Path | None = None,
    workers: int = 1,
    quarantine_invalid_pcaps: bool = False,
) -> dict[str, Any]:
    root = Path(processed_dir)
    audit_path = root / "audits" / "dohbrw_v4_audit.json"
    if not audit_path.exists():
        audit_dohbrw_v4(
            root,
            source_dir=source_dir,
            capture_manifest=capture_manifest,
            tshark_path=tshark_path,
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("supervised_training_enabled"):
        raise RuntimeError(
            f"DoHBrw v4 extraction is fail-closed: {audit.get('status')}"
        )
    source_manifest = json.loads(
        (root / "manifests" / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = Path(source_dir or source_manifest["source_dir"])
    manifest_path = Path(
        capture_manifest or source_manifest["capture_manifest"]
    )
    captures = load_capture_manifest(manifest_path)
    if workers < 1:
        raise ValueError("DoHBrw extraction workers must be at least one")
    tshark = locate_tshark(tshark_path)
    if tshark is None or not tshark_identity(tshark)["accepted"]:
        raise RuntimeError("fixed tshark 4.6.6 is required")
    flow_path = root / "flows" / "part-00000.jsonl.gz"
    flow_path.parent.mkdir(parents=True, exist_ok=True)
    plugin_dir = root / "tshark" / "isolated_empty_plugins"
    tshark_environment = _tshark_execution_environment(plugin_dir)
    extractor = lambda capture: _extract_capture_tsv(
        capture,
        source=source,
        root=root,
        tshark=tshark,
        plugin_dir=plugin_dir,
        quarantine_invalid_pcaps=quarantine_invalid_pcaps,
    )
    if workers == 1:
        tsv_rows = [extractor(capture) for capture in captures]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map retains manifest order, which makes the later flow
            # merge reproducible while allowing independent PCAP decoding.
            tsv_rows = list(pool.map(extractor, captures))
    quarantined = [row for row in tsv_rows if row["integrity_status"] != "valid"]
    _dump(
        root / "manifests" / "extraction_progress.json",
        {
            "capture_manifest_sha256": _sha256(manifest_path),
            "pcap_count": len(captures),
            "tsv_ready_count": len(tsv_rows),
            "valid_tsv_count": len(tsv_rows) - len(quarantined),
            "quarantined_capture_count": len(quarantined),
            "workers": workers,
            "status": (
                "tsv_extraction_complete_with_quarantine"
                if quarantined
                else "tsv_extraction_complete"
            ),
        },
    )
    quarantine_fields = (
        "capture_id",
        "relative_path",
        "pcap_sha256",
        "tshark_exit_code",
        "error_kind",
        "error_path",
    )
    quarantine_path = root / "audits" / "pcap_integrity_quarantine.csv"
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    with quarantine_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=quarantine_fields)
        writer.writeheader()
        writer.writerows(
            {name: row.get(name) for name in quarantine_fields}
            for row in quarantined
        )
    extraction_rows = []
    count = 0
    with gzip.open(flow_path, "wt", encoding="utf-8", newline="") as output:
        for capture, tsv_row in zip(captures, tsv_rows, strict=True):
            if tsv_row["integrity_status"] != "valid":
                extraction_rows.append(
                    {
                        "capture_id": capture["capture_id"],
                        "relative_path": capture["relative_path"],
                        "pcap_sha256": tsv_row["pcap_sha256"],
                        "flow_count": 0,
                        "integrity_status": "quarantined",
                    }
                )
                continue
            tsv = Path(tsv_row["tsv_path"])
            local_count = 0
            for record in _parse_tshark_rows(tsv, capture):
                output.write(record.model_dump_json() + "\n")
                count += 1
                local_count += 1
            extraction_rows.append(
                {
                    "capture_id": capture["capture_id"],
                    "relative_path": capture["relative_path"],
                    "pcap_sha256": tsv_row["pcap_sha256"],
                    "flow_count": local_count,
                    "integrity_status": "valid",
                }
            )
    report = {
        "schema_version": "1.0",
        "dataset": DOHBRW_DATASET,
        "status": (
            "extracted_with_integrity_quarantine"
            if quarantined
            else "extracted"
        ),
        "flow_count": count,
        "flow_path": flow_path.as_posix(),
        "flow_sha256": _sha256(flow_path),
        "capture_manifest_sha256": _sha256(manifest_path),
        "extraction_workers": workers,
        "quarantine_invalid_pcaps": quarantine_invalid_pcaps,
        "quarantined_capture_count": len(quarantined),
        "pcap_integrity_quarantine": quarantine_path.as_posix(),
        "tshark": tshark_identity(tshark),
        "tshark_plugin_loading": {
            "mode": "isolated_empty_directory",
            "directory": plugin_dir.as_posix(),
            "wireshark_plugin_dir": tshark_environment["WIRESHARK_PLUGIN_DIR"],
            "directory_is_empty": not any(plugin_dir.iterdir()),
        },
        "captures": extraction_rows,
        "blocked_feature_policy": [
            "IP",
            "port",
            "SNI",
            "JA3",
            "tool",
            "browser",
            "resolver",
            "source_file",
        ],
    }
    _dump(root / "manifests" / "extraction_manifest.json", report)
    return report


def _group_rank(seed: int, label: str, capture_id: str) -> str:
    return hashlib.sha256(
        f"mad-etd-v4:{seed}:{label}:{capture_id}".encode("utf-8")
    ).hexdigest()


def freeze_dohbrw_v4_splits(
    processed_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    *,
    seed: int = 42,
) -> dict[str, Any]:
    from .io import iter_flow_records

    root = Path(processed_dir)
    extraction = root / "manifests" / "extraction_manifest.json"
    if not extraction.exists():
        raise FileNotFoundError("extract-dohbrw-v4 must complete first")
    records = list(iter_flow_records(root / "flows"))
    supervised = [
        record
        for record in records
        if record.labels.get("binary") in {"benign", "malicious"}
    ]
    by_capture: dict[str, list[FlowRecord]] = defaultdict(list)
    for record in supervised:
        by_capture[str(record.provenance["capture_id"])].append(record)
    capture_meta = {
        capture_id: {
            "label": str(items[0].labels["binary"]),
            "resolver": str(items[0].labels.get("resolver") or ""),
            "generator": str(items[0].labels.get("generator") or ""),
        }
        for capture_id, items in by_capture.items()
    }
    assignments = {"train": [], "validation": [], "test": []}
    group_assignments: dict[str, str] = {}
    quad9_captures: list[str] = []
    calibration_captures: set[str] = set()
    selection_captures: set[str] = set()
    for label in ("benign", "malicious"):
        groups = [
            capture_id
            for capture_id, meta in capture_meta.items()
            if meta["label"] == label and meta["resolver"] != "quad9"
        ]
        groups.sort(key=lambda item: _group_rank(seed, label, item))
        if len(groups) < 8:
            raise ValueError(
                f"v4 requires at least eight non-Quad9 {label} capture groups"
            )
        train_cut = max(1, math.floor(len(groups) * 0.60))
        validation_cut = max(
            train_cut + 2,
            math.floor(len(groups) * 0.80),
        )
        split_groups = {
            "train": groups[:train_cut],
            "validation": groups[train_cut:validation_cut],
            "test": groups[validation_cut:],
        }
        if any(not values for values in split_groups.values()):
            raise ValueError(f"v4 {label} capture split is incomplete")
        if len(split_groups["validation"]) < 2:
            raise ValueError(
                f"v4 {label} needs independent calibration and selection captures"
            )
        ordered_validation = split_groups["validation"]
        calibration_captures.add(ordered_validation[0])
        selection_captures.add(ordered_validation[1])
        for extra_index, capture_id in enumerate(ordered_validation[2:]):
            (
                calibration_captures
                if extra_index % 2 == 0
                else selection_captures
            ).add(capture_id)
        for split, values in split_groups.items():
            for capture_id in values:
                group_assignments[capture_id] = split
                assignments[split].extend(
                    record.sample_id for record in by_capture[capture_id]
                )
        local_quad9 = [
            capture_id
            for capture_id, meta in capture_meta.items()
            if meta["label"] == label and meta["resolver"] == "quad9"
        ]
        if not local_quad9:
            raise ValueError(f"v4 Quad9 holdout lacks {label} capture")
        for capture_id in local_quad9:
            group_assignments[capture_id] = "test"
            quad9_captures.append(capture_id)
            assignments["test"].extend(
                record.sample_id for record in by_capture[capture_id]
            )
    calibration_ids = [
        record.sample_id
        for capture_id in calibration_captures
        for record in by_capture[capture_id]
    ]
    selection_ids = [
        record.sample_id
        for capture_id in selection_captures
        for record in by_capture[capture_id]
    ]
    if not calibration_ids or not selection_ids:
        raise ValueError("v4 calibration/selection capture groups are incomplete")
    split_capture_sets = {
        split: {
            capture_id
            for capture_id, assigned in group_assignments.items()
            if assigned == split
        }
        for split in assignments
    }
    overlaps = {
        "train_validation": len(
            split_capture_sets["train"] & split_capture_sets["validation"]
        ),
        "train_test": len(
            split_capture_sets["train"] & split_capture_sets["test"]
        ),
        "validation_test": len(
            split_capture_sets["validation"] & split_capture_sets["test"]
        ),
    }
    manifest = {
        "schema_version": "1.0",
        "dataset": DOHBRW_DATASET,
        "seed": seed,
        "ratios": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "grouping": "capture_id",
        "assignments": {
            key: sorted(set(values)) for key, values in assignments.items()
        },
        "group_assignments": group_assignments,
        "validation_partitions": {
            "calibration_capture_ids": sorted(calibration_captures),
            "selection_capture_ids": sorted(selection_captures),
            "calibration_sample_ids": sorted(calibration_ids),
            "selection_sample_ids": sorted(selection_ids),
        },
        "holdouts": {
            "quad9_capture_ids": sorted(quad9_captures),
            "quad9_sample_ids": sorted(
                record.sample_id
                for capture_id in quad9_captures
                for record in by_capture[capture_id]
            ),
        },
        "capture_overlap": overlaps,
        "non_doh_excluded_from_supervision": True,
        "locked_test": True,
        "test_used_for_selection": False,
    }
    if any(overlaps.values()):
        raise RuntimeError("v4 capture group leakage detected")
    canonical = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    _dump(root / "splits" / "split-manifest.json", manifest)
    return manifest
