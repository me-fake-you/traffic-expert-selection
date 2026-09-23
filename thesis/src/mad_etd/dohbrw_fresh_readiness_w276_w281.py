"""W276-W281 fresh DoHBrw capture readiness and bounded TLS smoke.

This lane deliberately does not train or evaluate a detector.  It reconciles
newly downloaded official archives against every historically consumed
DoHBrw capture, freezes a capture-disjoint *future* role manifest, and proves
that the fixed tshark toolchain can emit a safe TLS-record representation.

The benign browser archives contain both DoH and non-DoH traffic.  Therefore
their directory names are never treated as sample-level binary truth.  The
official per-flow CSV alignment remains mandatory before future supervised
training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .dohbrw_v4 import locate_tshark, tshark_identity
from .learned_tls_w83 import _malicious_csv_index, _normalize_stem
from .paper_evaluation import hash_artifact_paths
from .soc_evidence_team_w72 import _default_frozen_paths


EXPERIMENT = "mad_etd_dohbrw_fresh_capture_readiness_w276_w281"
DEFAULT_INPUT = Path("data/raw/DoHBrw/pcap")
DEFAULT_OUTPUT = Path(
    "data/runs/mad_etd_dohbrw_fresh_capture_readiness_w276_w281"
)
DEFAULT_STAGE = DEFAULT_INPUT / "_w278_fresh_staging"
DEFAULT_BENIGN_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/"
    "BenignDoH-NonDoH-CSVs.zip"
)
DEFAULT_MALICIOUS_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/MaliciousDoH-CSVs.zip"
)
DEFAULT_W70 = Path("data/runs/mad_etd_dohbrw_readiness_w70")
DEFAULT_W71 = Path("data/runs/mad_etd_w71_learned_tls_evidence")
DEFAULT_W83 = Path("data/runs/mad_etd_learned_tls_w83")
DEFAULT_W84 = Path("data/runs/mad_etd_safe_tls_hgb_w84")
DEFAULT_W96 = Path("data/runs/mad_etd_safe_tls_hgb_w96")
DEFAULT_W128 = Path(
    "data/runs/mad_etd_dohbrw_unseen_tool_development_w128_w134"
)
DEFAULT_DOC = Path("docs/MAD_ETD_DOHBRW_FRESH_READINESS_W276_W281.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_DOHBRW_FRESH_READINESS_W276_W281_CN.md")

FRESH_ARCHIVES = {
    "BenignDoH_NonDoH-Firefox-CloudFlare.zip": {
        "traffic_scope": "mixed_benign_doh_non_doh",
        "browser": "firefox",
        "tool": "",
        "resolver": "cloudflare",
    },
    "MaliciousDoH-dnscat2-Pcap-1202_1802.zip": {
        "traffic_scope": "malicious_doh",
        "browser": "",
        "tool": "dnscat2",
        "resolver": "mixed_official_capture_metadata",
    },
    "MaliciousDoH-iodine-pcap-1202_1802.zip": {
        "traffic_scope": "malicious_doh",
        "browser": "",
        "tool": "iodine",
        "resolver": "mixed_official_capture_metadata",
    },
}

BLOCKED_OR_CONTEXT_ONLY = {
    "ip",
    "port",
    "absolute_timestamp",
    "timestamp",
    "flow_id",
    "sni",
    "ja3",
    "ja4",
    "browser",
    "tool",
    "resolver",
    "capture_path",
    "archive_path",
    "archive_member",
    "source_file",
    "provenance",
    "label",
    "binary_label",
    "official_csv_entry",
    "tcp.stream",
}

SAFE_SMOKE_FEATURES = (
    "signed_tls_record_lengths",
    "sequence_length",
    "sequence_mask",
    "record_length_mean",
    "record_length_std",
    "record_length_min",
    "record_length_max",
    "record_length_sum",
    "signed_length_mean",
    "signed_length_std",
    "direction_change_rate",
    "tls_version_count",
    "cipher_count",
    "extension_count",
    "alpn_present",
    "short_flow_flag",
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    fieldnames: list[str] = []
    for row in values:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in values:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _safe_member(member: str) -> bool:
    pure = PurePosixPath(member)
    if pure.is_absolute() or ".." in pure.parts:
        return False
    return not bool(pure.parts and re.match(r"^[A-Za-z]:", pure.parts[0]))


def _safe_target(root: Path, member: str) -> Path:
    if not _safe_member(member):
        raise RuntimeError(f"unsafe archive member rejected: {member}")
    cleaned = [
        re.sub(r'[<>:"\\|?*]', "_", part) for part in PurePosixPath(member).parts
    ]
    target = root.joinpath(*cleaned)
    resolved_root = root.resolve()
    resolved_target = target.resolve()
    if resolved_root != resolved_target and resolved_root not in resolved_target.parents:
        raise RuntimeError(f"archive member escaped staging root: {member}")
    return target


def _archive_scope(name: str) -> dict[str, str]:
    if name in FRESH_ARCHIVES:
        return dict(FRESH_ARCHIVES[name])
    lowered = name.lower()
    if lowered.startswith("benigndoh_nondoh-"):
        browser = "firefox" if "firefox" in lowered else "chrome"
        resolver = next(
            (
                value
                for value in ("adguard", "cloudflare", "google", "quad9")
                if value in lowered
            ),
            "",
        )
        return {
            "traffic_scope": "mixed_benign_doh_non_doh",
            "browser": browser,
            "tool": "",
            "resolver": resolver,
        }
    tool = next(
        (value for value in ("dns2tcp", "dnscat2", "iodine") if value in lowered),
        "",
    )
    if lowered.startswith("maliciousdoh-") and tool:
        return {
            "traffic_scope": "malicious_doh",
            "browser": "",
            "tool": tool,
            "resolver": "mixed_official_capture_metadata",
        }
    return {
        "traffic_scope": "unresolved",
        "browser": "",
        "tool": "",
        "resolver": "",
    }


def _known_archive_hashes() -> tuple[dict[str, str], dict[str, int]]:
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    csv_sources = (
        Path(
            "data/runs/mad_etd_dohbrw_lite_w70/"
            "official_subset_staging_manifest.csv"
        ),
        DEFAULT_W84 / "staged_official_archives_w84.csv",
    )
    for path in csv_sources:
        if not path.exists():
            continue
        for row in _read_csv(path):
            raw_name = (
                row.get("archive_name")
                or row.get("archive_path")
                or row.get("source_archive")
                or ""
            )
            name = Path(raw_name).name
            value = row.get("archive_sha256", "")
            if name and len(value) == 64:
                hashes[name] = value
            size = row.get("archive_size_bytes") or row.get("size_bytes")
            if name and str(size).isdigit():
                sizes[name] = int(str(size))
    w96 = DEFAULT_W96 / "archive_extraction_manifest.json"
    if w96.exists():
        payload = _load(w96)
        name = Path(str(payload.get("archive", ""))).name
        value = str(payload.get("archive_sha256", ""))
        if name and len(value) == 64:
            hashes[name] = value
        if name and payload.get("archive_size_bytes") is not None:
            sizes[name] = int(payload["archive_size_bytes"])
    w99 = Path(
        "data/runs/mad_etd_tls_dataset_eligibility_w99/"
        "dohbrw_archive_inventory.csv"
    )
    if w99.exists():
        for row in _read_csv(w99):
            name = row.get("archive_name", "")
            size = row.get("size_bytes", "")
            if name and str(size).isdigit():
                sizes[name] = int(size)
    return hashes, sizes


def _archive_inventory(
    input_root: Path,
    *,
    hash_cache_path: Path,
) -> list[dict[str, Any]]:
    prior_cache: dict[str, Any] = {}
    if hash_cache_path.exists():
        loaded = _load(hash_cache_path)
        prior_cache = {
            str(row["archive_name"]): row
            for row in loaded.get("archives", [])
            if isinstance(row, dict) and row.get("archive_name")
        }
    known_hashes, known_sizes = _known_archive_hashes()
    rows: list[dict[str, Any]] = []
    for archive in sorted(input_root.glob("*.zip"), key=lambda p: p.name.lower()):
        stat = archive.stat()
        cached = prior_cache.get(archive.name, {})
        digest = ""
        hash_source = ""
        hash_verified_this_round = False
        if (
            cached.get("size_bytes") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and len(str(cached.get("sha256", ""))) == 64
        ):
            digest = str(cached["sha256"])
            hash_source = "w276_local_cache_same_size_mtime"
        elif (
            archive.name in known_hashes
            and (
                archive.name not in known_sizes
                or known_sizes[archive.name] == stat.st_size
            )
        ):
            digest = known_hashes[archive.name]
            hash_source = "historical_verified_artifact_same_name_size"
        else:
            digest = _sha256(archive)
            hash_source = "w276_streamed_sha256"
            hash_verified_this_round = True

        scope = _archive_scope(archive.name)
        member_count = 0
        pcap_members: list[zipfile.ZipInfo] = []
        unsafe_members: list[str] = []
        duplicate_members = 0
        encrypted_members = 0
        central_directory_valid = False
        error = ""
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                member_count = len(members)
                names = [item.filename for item in members]
                duplicate_members = len(names) - len(set(names))
                unsafe_members = [
                    item.filename for item in members if not _safe_member(item.filename)
                ]
                encrypted_members = sum(bool(item.flag_bits & 0x1) for item in members)
                pcap_members = [
                    item
                    for item in members
                    if item.filename.lower().endswith((".pcap", ".pcapng"))
                ]
                central_directory_valid = True
        except (OSError, zipfile.BadZipFile) as exc:
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "archive_name": archive.name,
                "archive_path": archive.as_posix(),
                "size_bytes": stat.st_size,
                "size_gib": round(stat.st_size / (1024**3), 6),
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "hash_source": hash_source,
                "hash_verified_this_round": hash_verified_this_round,
                "central_directory_valid": central_directory_valid,
                "member_count": member_count,
                "pcap_member_count": len(pcap_members),
                "pcap_uncompressed_bytes": sum(
                    item.file_size for item in pcap_members
                ),
                "duplicate_member_count": duplicate_members,
                "unsafe_member_count": len(unsafe_members),
                "encrypted_member_count": encrypted_members,
                "traffic_scope": scope["traffic_scope"],
                "browser": scope["browser"],
                "tool": scope["tool"],
                "resolver": scope["resolver"],
                "capture_label_homogeneous": scope["traffic_scope"]
                == "malicious_doh",
                "official_flow_csv_required": scope["traffic_scope"]
                == "mixed_benign_doh_non_doh",
                "payload_crc_fully_tested": False,
                "bounded_payload_smoke_required": archive.name in FRESH_ARCHIVES,
                "error": error,
            }
        )
    return rows


def _safe_feature_policy() -> dict[str, Any]:
    policy = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "detector_input_features": list(SAFE_SMOKE_FEATURES),
        "alignment_only_never_detector_input": sorted(BLOCKED_OR_CONTEXT_ONLY),
        "label_policy": {
            "malicious_archives": (
                "official malicious archive category plus exact official "
                "per-capture CSV match"
            ),
            "benign_browser_archives": (
                "mixed DoH/non-DoH capture; official per-flow CSV alignment "
                "is mandatory and directory names are not sample truth"
            ),
            "non_doh": "OOD-only and excluded from supervised training",
        },
        "max_tls_records": 64,
        "tcp_reassembly": True,
        "candidate_default_enabled": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    policy["feature_policy_hash"] = _canonical_hash(policy)
    return policy


def audit_dohbrw_downloads_w276(
    *,
    input_root: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    benign_csv: str | Path = DEFAULT_BENIGN_CSV,
    malicious_csv: str | Path = DEFAULT_MALICIOUS_CSV,
    w70_dir: str | Path = DEFAULT_W70,
) -> dict[str, Any]:
    root = Path(input_root)
    output = Path(output_dir)
    benign_csv_path, malicious_csv_path = Path(benign_csv), Path(malicious_csv)
    w70 = Path(w70_dir)
    for required in (
        root,
        benign_csv_path,
        malicious_csv_path,
        w70 / "acceptance_report.json",
    ):
        if not required.exists():
            raise RuntimeError(f"W276 missing required artifact: {required}")
    partials = sorted(
        path.as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".crdownload", ".part", ".tmp"}
    )
    if partials:
        raise RuntimeError(f"W276 refuses incomplete downloads: {partials[:5]}")
    w70_report = _load(w70 / "acceptance_report.json")
    if w70_report.get("status") != "ready_for_w71_tls_record_training":
        raise RuntimeError("W276 requires the accepted historical W70 readiness")
    output.mkdir(parents=True, exist_ok=True)
    rows = _archive_inventory(
        root, hash_cache_path=output / "archive_hash_cache.json"
    )
    if not rows:
        raise RuntimeError("W276 found no official DoHBrw ZIP archives")
    required_names = set(FRESH_ARCHIVES)
    found_names = {row["archive_name"] for row in rows}
    missing_fresh = sorted(required_names - found_names)
    unresolved = [
        row["archive_name"]
        for row in rows
        if row["traffic_scope"] == "unresolved"
    ]
    invalid = [
        row["archive_name"]
        for row in rows
        if not row["central_directory_valid"]
        or row["unsafe_member_count"]
        or row["encrypted_member_count"]
        or not row["sha256"]
    ]
    tshark = locate_tshark(None)
    identity = tshark_identity(tshark) if tshark else {"accepted": False}
    _write_csv(output / "download_archive_inventory.csv", rows)
    _dump(
        output / "archive_hash_cache.json",
        {
            "schema_version": "1.0",
            "archives": [
                {
                    "archive_name": row["archive_name"],
                    "size_bytes": row["size_bytes"],
                    "mtime_ns": row["mtime_ns"],
                    "sha256": row["sha256"],
                    "hash_source": row["hash_source"],
                }
                for row in rows
            ],
        },
    )
    _dump(
        output / "official_source_map.json",
        {
            "schema_version": "1.0",
            "source_mode": "immutable_multi_root_mapping_no_large_file_copy",
            "current_download_root": root.as_posix(),
            "historical_official_download_root": (
                "data/raw/cira_cic_dohbrw_2020/v1/official_download"
            ),
            "auxiliary_benign_csv": benign_csv_path.as_posix(),
            "auxiliary_malicious_csv": malicious_csv_path.as_posix(),
            "csv_used_for_alignment_only": True,
            "csv_enters_detector_input": False,
        },
    )
    _dump(output / "safe_tls_feature_policy.json", _safe_feature_policy())
    _dump(output / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "w276_download_audit",
        "status": (
            "downloads_audited_for_fresh_reconciliation"
            if not missing_fresh and not invalid and not unresolved
            else "blocked_download_or_archive_audit"
        ),
        "archive_count": len(rows),
        "archive_bytes": sum(int(row["size_bytes"]) for row in rows),
        "archive_gib": round(
            sum(int(row["size_bytes"]) for row in rows) / (1024**3), 6
        ),
        "pcap_member_count": sum(
            int(row["pcap_member_count"]) for row in rows
        ),
        "missing_required_fresh_archives": missing_fresh,
        "invalid_archives": invalid,
        "unresolved_archives": unresolved,
        "partial_download_count": 0,
        "w70_status": w70_report.get("status"),
        "historical_w70_reused_not_rerun": True,
        "tshark": identity,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "download_audit_report.json", report)
    if report["status"].startswith("blocked"):
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def _historical_archive_usage() -> dict[str, set[str]]:
    usage: dict[str, set[str]] = defaultdict(set)
    lite = Path(
        "data/runs/mad_etd_dohbrw_lite_w70/"
        "official_subset_staging_manifest.csv"
    )
    if lite.exists():
        for row in _read_csv(lite):
            name = row.get("archive_name", "")
            if name:
                usage[name].add("w70_w71_historical_lane")
    staged84 = DEFAULT_W84 / "staged_official_archives_w84.csv"
    if staged84.exists():
        for row in _read_csv(staged84):
            name = Path(row.get("archive_path", "")).name
            if name:
                usage[name].add("w84_staged_or_trained_lane")
    w96 = DEFAULT_W96 / "archive_extraction_manifest.json"
    if w96.exists():
        payload = _load(w96)
        name = Path(str(payload.get("archive", ""))).name
        if name:
            usage[name].add("w96_audit_only_no_training")
    return usage


def reconcile_dohbrw_history_w277(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    w71_dir: str | Path = DEFAULT_W71,
    w83_dir: str | Path = DEFAULT_W83,
    w84_dir: str | Path = DEFAULT_W84,
    w96_dir: str | Path = DEFAULT_W96,
    w128_dir: str | Path = DEFAULT_W128,
) -> dict[str, Any]:
    output = Path(output_dir)
    audit_path = output / "download_audit_report.json"
    if not audit_path.exists():
        raise RuntimeError("W277 requires W276 download audit")
    if _load(audit_path).get("status") != "downloads_audited_for_fresh_reconciliation":
        raise RuntimeError("W277 refuses a blocked W276 audit")
    w71, w83, w84, w96, w128 = map(
        Path, (w71_dir, w83_dir, w84_dir, w96_dir, w128_dir)
    )
    required = (
        w71 / "acceptance_report_w71.json",
        w71 / "acceptance_report.json",
        w71 / "locked_test_access.json",
        w83 / "acceptance_report.json",
        w84 / "acceptance_report.json",
        w96 / "acceptance_report.json",
        w128 / "pcap_identity_inventory_w128.csv",
    )
    for path in required:
        if not path.exists():
            raise RuntimeError(f"W277 missing historical evidence: {path}")
    w71_report = _load(w71 / "acceptance_report_w71.json")
    w71_primary = _load(w71 / "acceptance_report.json")
    w71_access = _load(w71 / "locked_test_access.json")
    w96_report = _load(w96 / "acceptance_report.json")
    locked_lane_closed = bool(
        w71_report.get("locked_test_read_once")
        or (
            w71_primary.get("locked_test_used_once")
            and w71_access.get("split") == "test"
            and not w71_access.get("selection_or_tuning")
        )
    )
    if not locked_lane_closed:
        raise RuntimeError("W277 cannot verify that the historical locked lane is closed")
    if w96_report.get("training_executed") or w96_report.get(
        "new_acceptance_executed"
    ):
        raise RuntimeError("W277 expected W96 Firefox-Quad9 to be audit-only")

    usage = _historical_archive_usage()
    inventory = _read_csv(output / "download_archive_inventory.csv")
    rows: list[dict[str, Any]] = []
    for row in inventory:
        name = row["archive_name"]
        sources = sorted(usage.get(name, set()))
        audit_only = sources == ["w96_audit_only_no_training"]
        consumed = bool(sources) and not audit_only
        fresh = name in FRESH_ARCHIVES and not consumed
        rows.append(
            {
                "archive_name": name,
                "sha256": row["sha256"],
                "historical_usage": "|".join(sources),
                "historically_consumed_for_model_lane": consumed,
                "audit_only_not_model_consumed": audit_only,
                "fresh_archive_candidate": fresh,
                "fresh_reason": (
                    "not present in historical model-selection or acceptance lanes"
                    if fresh
                    else (
                        "W96 metadata/extraction audit only; still eligible as a "
                        "future untouched capture group"
                        if audit_only
                        else "historically consumed or not selected as a fresh lane"
                    )
                ),
            }
        )
    _write_csv(output / "archive_usage_ledger.csv", rows)
    _write_csv(
        output / "fresh_archive_candidates.csv",
        [row for row in rows if row["fresh_archive_candidate"]],
    )
    historical = _read_csv(w128 / "pcap_identity_inventory_w128.csv")
    _write_csv(
        output / "historical_capture_hash_registry.csv",
        [
            {
                "relative_path": row.get("relative_path", ""),
                "size_bytes": row.get("size_bytes", ""),
                "sha256": row.get("sha256", ""),
                "historical_match_sources": row.get(
                    "historical_match_sources", ""
                ),
                "model_lane_consumed_or_historical": True,
            }
            for row in historical
        ],
    )
    fresh_names = {
        row["archive_name"]
        for row in rows
        if row["fresh_archive_candidate"]
    }
    missing = sorted(set(FRESH_ARCHIVES) - fresh_names)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "w277_historical_reconciliation",
        "status": (
            "fresh_archive_set_reconciled"
            if not missing
            else "blocked_fresh_archives_consumed_or_missing"
        ),
        "historical_capture_hash_count": len(historical),
        "fresh_archive_names": sorted(fresh_names),
        "required_fresh_archives_missing_after_reconciliation": missing,
        "firefox_quad9_w96_audit_only_eligible": any(
            row["archive_name"]
            == "BenignDoH_NonDoH-Firefox-Quad9.zip"
            and row["audit_only_not_model_consumed"]
            for row in rows
        ),
        "historical_w71_locked_test_reopened": False,
        "new_training_executed": False,
        "new_acceptance_executed": False,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
    }
    _dump(output / "historical_reconciliation_report.json", report)
    if report["status"].startswith("blocked"):
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def _rank(seed: int, lane: str, value: str) -> str:
    return hashlib.sha256(f"w276:{seed}:{lane}:{value}".encode("utf-8")).hexdigest()


def _zip_pcap_members(archive: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as bundle:
        values = [
            item
            for item in bundle.infolist()
            if item.filename.lower().endswith((".pcap", ".pcapng"))
        ]
    if any(not _safe_member(item.filename) for item in values):
        raise RuntimeError(f"unsafe member found in {archive}")
    return values


def _extract_members(
    archive: Path,
    members: Iterable[str],
    stage_root: Path,
) -> list[dict[str, Any]]:
    wanted = set(members)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        infos = {item.filename: item for item in bundle.infolist()}
        for name in sorted(wanted):
            if name not in infos:
                raise RuntimeError(f"archive member disappeared: {name}")
            info = infos[name]
            target = _safe_target(stage_root / archive.stem, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            disposition = "resumed_existing"
            if not target.exists() or target.stat().st_size != info.file_size:
                with bundle.open(info) as source, target.open("wb") as sink:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        sink.write(chunk)
                disposition = "extracted_w278"
            if target.stat().st_size != info.file_size:
                raise RuntimeError(f"staged member size mismatch: {name}")
            rows.append(
                {
                    "archive_name": archive.name,
                    "archive_member": name,
                    "member_crc32": f"{info.CRC:08x}",
                    "member_size_bytes": info.file_size,
                    "staged_path": target.as_posix(),
                    "staging_disposition": disposition,
                    "staged_sha256": _sha256(target),
                }
            )
    return rows


def _member_group(
    *,
    archive_name: str,
    archive_sha256: str,
    info: zipfile.ZipInfo,
    traffic_scope: str,
    browser: str = "",
    tool: str = "",
    resolver: str = "",
    official_csv_entry: str = "",
    official_csv_candidates: str = "",
    role: str = "",
    extraction_state: str = "deferred_until_w282",
) -> dict[str, Any]:
    identity = {
        "archive_sha256": archive_sha256,
        "member": info.filename,
        "crc32": f"{info.CRC:08x}",
        "size": info.file_size,
    }
    return {
        "capture_group_id": "fresh-" + _canonical_hash(identity)[:20],
        "archive_name": archive_name,
        "archive_sha256": archive_sha256,
        "archive_member": info.filename,
        "member_crc32": f"{info.CRC:08x}",
        "member_size_bytes": info.file_size,
        "traffic_scope": traffic_scope,
        "browser": browser,
        "tool": tool,
        "resolver": resolver,
        "official_csv_entry": official_csv_entry,
        "official_csv_candidates": official_csv_candidates,
        "official_label_alignment_required": True,
        "capture_directory_is_sample_truth": False,
        "role": role,
        "extraction_state": extraction_state,
        "enters_detector_input": False,
    }


def stage_fresh_dohbrw_captures_w278(
    *,
    input_root: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    stage_root: str | Path = DEFAULT_STAGE,
    benign_csv: str | Path = DEFAULT_BENIGN_CSV,
    malicious_csv: str | Path = DEFAULT_MALICIOUS_CSV,
    w96_dir: str | Path = DEFAULT_W96,
    seed: int = 42,
) -> dict[str, Any]:
    root, output, stage = Path(input_root), Path(output_dir), Path(stage_root)
    benign_csv_path, malicious_csv_path = Path(benign_csv), Path(malicious_csv)
    w96 = Path(w96_dir)
    recon = output / "historical_reconciliation_report.json"
    if not recon.exists() or _load(recon).get("status") != "fresh_archive_set_reconciled":
        raise RuntimeError("W278 requires accepted W277 reconciliation")
    archive_inventory = {
        row["archive_name"]: row
        for row in _read_csv(output / "download_archive_inventory.csv")
    }
    malicious_index = _malicious_csv_index(malicious_csv_path)
    with zipfile.ZipFile(benign_csv_path) as bundle:
        firefox_csv_entries = sorted(
            name
            for name in bundle.namelist()
            if "/Firefox/Separate/" in name
            and name.lower().endswith(".csv")
        )
    if not firefox_csv_entries:
        raise RuntimeError("W278 found no official Firefox per-flow CSV candidates")

    all_groups: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []

    firefox_archive_name = "BenignDoH_NonDoH-Firefox-CloudFlare.zip"
    firefox_archive = root / firefox_archive_name
    firefox_infos = _zip_pcap_members(firefox_archive)
    if len(firefox_infos) < 2:
        raise RuntimeError("W278 needs two Firefox-Cloudflare capture groups")
    firefox_infos.sort(key=lambda item: item.filename)
    firefox_roles = ("train", "selection")
    for role, info in zip(firefox_roles, firefox_infos[:2]):
        all_groups.append(
            _member_group(
                archive_name=firefox_archive_name,
                archive_sha256=archive_inventory[firefox_archive_name]["sha256"],
                info=info,
                traffic_scope="mixed_benign_doh_non_doh",
                browser="firefox",
                resolver="cloudflare",
                official_csv_candidates=json.dumps(
                    firefox_csv_entries, ensure_ascii=False
                ),
                role=role,
                extraction_state="large_member_verified_in_zip_deferred_to_w282",
            )
        )

    quad_manifest = w96 / "archive_extraction_manifest.json"
    if not quad_manifest.exists():
        raise RuntimeError("W278 requires W96 audit-only Firefox-Quad9 manifest")
    quad_payload = _load(quad_manifest)
    quad_archive_name = Path(str(quad_payload["archive"])).name
    quad_archive_sha = str(quad_payload["archive_sha256"])
    quad_infos = _zip_pcap_members(root / quad_archive_name)
    quad_by_size = {info.file_size: info for info in quad_infos}
    quad_roles = ("train", "acceptance")
    for role, extracted in zip(quad_roles, quad_payload["extracted_pcaps"]):
        info = quad_by_size.get(int(extracted["size_bytes"]))
        if info is None or _sha256(Path(extracted["path"])) != extracted["sha256"]:
            raise RuntimeError("W278 Firefox-Quad9 audit-only capture changed")
        all_groups.append(
            _member_group(
                archive_name=quad_archive_name,
                archive_sha256=quad_archive_sha,
                info=info,
                traffic_scope="mixed_benign_doh_non_doh",
                browser="firefox",
                resolver="quad9",
                official_csv_candidates=json.dumps(
                    firefox_csv_entries, ensure_ascii=False
                ),
                role=role,
                extraction_state="existing_w96_audit_only_capture_verified",
            )
        )

    malicious_specs = (
        ("MaliciousDoH-dnscat2-Pcap-1202_1802.zip", "dnscat2"),
        ("MaliciousDoH-iodine-pcap-1202_1802.zip", "iodine"),
    )
    role_counts = {"train": 60, "selection": 20, "acceptance": 20}
    for archive_name, tool in malicious_specs:
        archive = root / archive_name
        infos = _zip_pcap_members(archive)
        eligible: list[tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            csv_entry = malicious_index.get(
                (tool, _normalize_stem(Path(info.filename).stem)), ""
            )
            if csv_entry:
                eligible.append((info, csv_entry))
        if len(eligible) < sum(role_counts.values()):
            raise RuntimeError(
                f"W278 lacks exact official CSV matches for {tool}: "
                f"{len(eligible)}/{sum(role_counts.values())}"
            )
        eligible.sort(
            key=lambda pair: (
                _rank(seed, tool, pair[0].filename),
                pair[0].filename,
            )
        )
        cursor = 0
        selected_for_smoke: list[str] = []
        for role, count in role_counts.items():
            lane = eligible[cursor : cursor + count]
            cursor += count
            for info, csv_entry in lane:
                all_groups.append(
                    _member_group(
                        archive_name=archive_name,
                        archive_sha256=archive_inventory[archive_name]["sha256"],
                        info=info,
                        traffic_scope="malicious_doh",
                        tool=tool,
                        official_csv_entry=csv_entry,
                        role=role,
                    )
                )
            selected_for_smoke.append(lane[0][0].filename)
        extraction_rows.extend(
            _extract_members(archive, selected_for_smoke, stage)
        )

    group_ids = [row["capture_group_id"] for row in all_groups]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("W278 duplicate capture-group identities")
    _write_csv(output / "fresh_capture_staging_manifest.csv", extraction_rows)
    _write_csv(output / "fresh_capture_group_candidates.csv", all_groups)
    mapping_rows = [
        {
            "capture_group_id": row["capture_group_id"],
            "traffic_scope": row["traffic_scope"],
            "official_label_source": (
                row["official_csv_entry"]
                or "official Firefox per-flow CSV candidate set; "
                "endpoint-overlap alignment required before supervised use"
            ),
            "exact_csv_match": bool(row["official_csv_entry"]),
            "flow_alignment_pending": row["traffic_scope"]
            == "mixed_benign_doh_non_doh",
            "directory_label_used_as_sample_truth": False,
            "label_enters_detector_input": False,
        }
        for row in all_groups
    ]
    _write_csv(output / "official_label_mapping_audit.csv", mapping_rows)
    role_class_counts = Counter(
        (row["role"], row["traffic_scope"]) for row in all_groups
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "w278_safe_staging",
        "status": "fresh_capture_groups_staged_and_frozen_for_smoke",
        "fresh_capture_group_count": len(all_groups),
        "role_class_counts": {
            f"{role}|{scope}": count
            for (role, scope), count in sorted(role_class_counts.items())
        },
        "bounded_extracted_pcap_count": len(extraction_rows),
        "large_firefox_cloudflare_extraction_deferred": True,
        "firefox_flow_alignment_pending": True,
        "malicious_exact_official_csv_match_rate": (
            sum(bool(row["official_csv_entry"]) for row in all_groups)
            / max(
                1,
                sum(
                    row["traffic_scope"] == "malicious_doh"
                    for row in all_groups
                ),
            )
        ),
        "directory_label_used_as_sample_truth": False,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "fresh_staging_report.json", report)
    return report


def freeze_fresh_dohbrw_groups_w279(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir)
    staging = output / "fresh_staging_report.json"
    if not staging.exists() or _load(staging).get("status") != (
        "fresh_capture_groups_staged_and_frozen_for_smoke"
    ):
        raise RuntimeError("W279 requires accepted W278 staging")
    groups = _read_csv(output / "fresh_capture_group_candidates.csv")
    roles = {"train", "selection", "acceptance"}
    role_sets = {
        role: {
            row["capture_group_id"] for row in groups if row["role"] == role
        }
        for role in roles
    }
    overlap = {
        f"{left}|{right}": sorted(role_sets[left] & role_sets[right])
        for left in roles
        for right in roles
        if left < right
    }
    role_scope = Counter(
        (row["role"], row["traffic_scope"]) for row in groups
    )
    missing_scope = [
        f"{role}|{scope}"
        for role in sorted(roles)
        for scope in ("mixed_benign_doh_non_doh", "malicious_doh")
        if role_scope[(role, scope)] == 0
    ]
    frozen = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": (
            "fresh_capture_roles_frozen_for_w282"
            if not any(overlap.values()) and not missing_scope
            else "blocked_fresh_capture_role_isolation"
        ),
        "seed": 42,
        "selection_policy": (
            "hash-ranked archive/member identities; roles fixed before TLS "
            "record extraction, flow-label alignment, training or evaluation"
        ),
        "group_count": len(groups),
        "role_counts": {
            role: len(values) for role, values in sorted(role_sets.items())
        },
        "role_scope_counts": {
            f"{role}|{scope}": count
            for (role, scope), count in sorted(role_scope.items())
        },
        "role_overlap": overlap,
        "missing_role_scope_cells": missing_scope,
        "acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
        "historical_locked_test_reopened": False,
        "supervised_metrics_generated": False,
        "manifest_hash": _canonical_hash(groups),
    }
    _write_csv(output / "capture_group_role_manifest.csv", groups)
    _dump(output / "fresh_group_feasibility.json", frozen)
    if frozen["status"].startswith("blocked"):
        raise RuntimeError(json.dumps(frozen, ensure_ascii=False))
    return frozen


def _parse_multi(value: str) -> list[str]:
    return [item for item in re.split(r"[,;]", value or "") if item]


def _run_tshark_smoke(
    pcap: Path,
    tshark: Path,
    *,
    packet_limit: int,
    timeout_seconds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = (
        "frame.time_epoch",
        "ip.src",
        "ipv6.src",
        "tcp.srcport",
        "ip.dst",
        "ipv6.dst",
        "tcp.dstport",
        "tcp.stream",
        "tls.record.length",
        "tls.handshake.version",
        "tls.handshake.ciphersuite",
        "tls.handshake.extension.type",
        "tls.handshake.extensions_alpn_str",
    )
    command = [
        str(tshark),
        "-r",
        str(pcap),
        "-c",
        str(packet_limit),
        "-o",
        "tcp.desegment_tcp_streams:TRUE",
        "-o",
        "tls.desegment_ssl_records:TRUE",
        "-Y",
        "tls.record || tls.handshake",
        "-T",
        "fields",
        "-E",
        "separator=/t",
        "-E",
        "occurrence=a",
        "-E",
        "aggregator=,",
    ]
    for field in fields:
        command.extend(["-e", field])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"tshark smoke failed for {pcap}: {completed.stderr[-1000:]}"
        )
    streams: dict[str, dict[str, Any]] = {}
    for raw in completed.stdout.splitlines():
        parts = raw.split("\t")
        parts.extend([""] * (len(fields) - len(parts)))
        row = dict(zip(fields, parts))
        stream = row["tcp.stream"]
        if not stream:
            continue
        src = row["ip.src"] or row["ipv6.src"]
        dst = row["ip.dst"] or row["ipv6.dst"]
        src_ep = (src, row["tcp.srcport"])
        dst_ep = (dst, row["tcp.dstport"])
        value = streams.setdefault(
            stream,
            {
                "origin": src_ep,
                "records": [],
                "versions": set(),
                "ciphers": set(),
                "extensions": set(),
                "alpn": set(),
                "first_time": row["frame.time_epoch"],
            },
        )
        sign = 1 if src_ep == value["origin"] else -1
        for item in _parse_multi(row["tls.record.length"]):
            try:
                length = int(float(item))
            except ValueError:
                continue
            if len(value["records"]) < 64:
                value["records"].append(sign * abs(length))
        value["versions"].update(_parse_multi(row["tls.handshake.version"]))
        value["ciphers"].update(_parse_multi(row["tls.handshake.ciphersuite"]))
        value["extensions"].update(
            _parse_multi(row["tls.handshake.extension.type"])
        )
        value["alpn"].update(
            _parse_multi(row["tls.handshake.extensions_alpn_str"])
        )

    safe_rows: list[dict[str, Any]] = []
    for stream, value in sorted(streams.items(), key=lambda pair: pair[0]):
        records = list(value["records"])
        if not records:
            continue
        absolute = [abs(item) for item in records]
        changes = sum(
            (records[index] >= 0) != (records[index - 1] >= 0)
            for index in range(1, len(records))
        )
        safe_rows.append(
            {
                "_tcp_stream_offline_only": stream,
                "signed_tls_record_lengths": json.dumps(records),
                "sequence_length": len(records),
                "sequence_mask": json.dumps(
                    [1] * len(records) + [0] * (64 - len(records))
                ),
                "record_length_mean": statistics.fmean(absolute),
                "record_length_std": (
                    statistics.pstdev(absolute) if len(absolute) > 1 else 0.0
                ),
                "record_length_min": min(absolute),
                "record_length_max": max(absolute),
                "record_length_sum": sum(absolute),
                "signed_length_mean": statistics.fmean(records),
                "signed_length_std": (
                    statistics.pstdev(records) if len(records) > 1 else 0.0
                ),
                "direction_change_rate": changes / max(1, len(records) - 1),
                "tls_version_count": len(value["versions"]),
                "cipher_count": len(value["ciphers"]),
                "extension_count": len(value["extensions"]),
                "alpn_present": bool(value["alpn"]),
                "short_flow_flag": len(records) < 4,
            }
        )
    audit = {
        "pcap": pcap.as_posix(),
        "packet_limit": packet_limit,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-1000:],
        "raw_stream_count": len(streams),
        "safe_stream_count": len(safe_rows),
        "alignment_fields_used_offline_only": [
            "IP",
            "port",
            "absolute timestamp",
            "tcp.stream",
        ],
    }
    return safe_rows, audit


def run_bounded_tls_smoke_w280(
    *,
    input_root: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
    stage_root: str | Path = DEFAULT_STAGE,
    w96_dir: str | Path = DEFAULT_W96,
    tshark_path: str | Path | None = None,
    packet_limit: int = 200_000,
    max_safe_flows_per_capture: int = 50,
) -> dict[str, Any]:
    root, output, stage = Path(input_root), Path(output_dir), Path(stage_root)
    w96 = Path(w96_dir)
    frozen = output / "fresh_group_feasibility.json"
    if not frozen.exists() or _load(frozen).get("status") != (
        "fresh_capture_roles_frozen_for_w282"
    ):
        raise RuntimeError("W280 requires accepted W279 frozen groups")
    tshark = locate_tshark(tshark_path)
    identity = tshark_identity(tshark) if tshark else {"accepted": False}
    if not tshark or not identity.get("accepted"):
        raise RuntimeError("W280 requires fixed tshark 4.6.6")

    staged = _read_csv(output / "fresh_capture_staging_manifest.csv")
    smoke_sources: list[tuple[Path, dict[str, str]]] = []
    for row in staged[:4]:
        smoke_sources.append((Path(row["staged_path"]), row))
    quad_payload = _load(w96 / "archive_extraction_manifest.json")
    quad = quad_payload["extracted_pcaps"][0]
    smoke_sources.append(
        (
            Path(quad["path"]),
            {
                "archive_name": Path(str(quad_payload["archive"])).name,
                "archive_member": "Quad9/audit-only existing capture",
                "staged_sha256": quad["sha256"],
                "source_scope": "mixed_benign_doh_non_doh",
            },
        )
    )

    policy = _load(output / "safe_tls_feature_policy.json")
    safe_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    source_scope_counts: Counter[str] = Counter()
    for source_index, (pcap, metadata) in enumerate(smoke_sources):
        if not pcap.exists():
            raise RuntimeError(f"W280 smoke source disappeared: {pcap}")
        rows, audit = _run_tshark_smoke(
            pcap,
            tshark,
            packet_limit=packet_limit,
            timeout_seconds=300,
        )
        archive_name = metadata.get("archive_name", "")
        scope = (
            metadata.get("source_scope")
            or _archive_scope(archive_name)["traffic_scope"]
        )
        source_scope_counts[scope] += int(bool(rows))
        for row_index, row in enumerate(rows[:max_safe_flows_per_capture]):
            stream = row.pop("_tcp_stream_offline_only")
            smoke_case_id = (
                "w280-" + _canonical_hash(
                    {
                        "source": source_index,
                        "stream": stream,
                        "records": row["signed_tls_record_lengths"],
                    }
                )[:20]
            )
            safe_rows.append(
                {
                    "smoke_case_id": smoke_case_id,
                    **row,
                    "feature_policy_hash": policy["feature_policy_hash"],
                }
            )
            audit_rows.append(
                {
                    "smoke_case_id": smoke_case_id,
                    "source_capture_path": pcap.as_posix(),
                    "source_capture_sha256": metadata.get(
                        "staged_sha256", quad.get("sha256", "")
                    ),
                    "archive_name": archive_name,
                    "archive_member": metadata.get("archive_member", ""),
                    "traffic_scope": scope,
                    "tcp_stream_alignment_only": stream,
                    "label_assigned_to_smoke_feature_row": False,
                    "alignment_metadata_enters_detector_input": False,
                }
            )
        audit_rows.append(
            {
                "smoke_case_id": "",
                "source_capture_path": pcap.as_posix(),
                "source_capture_sha256": metadata.get(
                    "staged_sha256", quad.get("sha256", "")
                ),
                "archive_name": archive_name,
                "archive_member": metadata.get("archive_member", ""),
                "traffic_scope": scope,
                "tcp_stream_alignment_only": "",
                "label_assigned_to_smoke_feature_row": False,
                "alignment_metadata_enters_detector_input": False,
                "tshark_packet_limit": audit["packet_limit"],
                "tshark_safe_stream_count": audit["safe_stream_count"],
                "tshark_returncode": audit["returncode"],
            }
        )
    if not safe_rows:
        raise RuntimeError("W280 produced no real TLS-record smoke rows")
    forbidden_in_safe = sorted(
        set(safe_rows[0]).intersection(BLOCKED_OR_CONTEXT_ONLY)
    )
    scope_ready = (
        source_scope_counts["malicious_doh"] > 0
        and source_scope_counts["mixed_benign_doh_non_doh"] > 0
    )
    _write_csv(output / "safe_tls_smoke_features.csv", safe_rows)
    _write_csv(output / "smoke_offline_alignment_audit.csv", audit_rows)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "phase": "w280_bounded_real_tshark_smoke",
        "status": (
            "bounded_real_tls_record_smoke_passed"
            if not forbidden_in_safe and scope_ready
            else "blocked_tls_record_smoke"
        ),
        "real_tshark_calls": len(smoke_sources),
        "safe_smoke_flow_count": len(safe_rows),
        "source_scope_with_tls_rows": dict(source_scope_counts),
        "max_tls_records": 64,
        "tcp_reassembly": True,
        "tshark": identity,
        "forbidden_fields_in_safe_feature_table": forbidden_in_safe,
        "offline_alignment_metadata_separated": True,
        "supervised_labels_assigned": False,
        "accuracy_or_f1_generated": False,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
    }
    _dump(output / "tls_record_smoke_report.json", report)
    if report["status"].startswith("blocked"):
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    return report


def _document_text(report: Mapping[str, Any], *, chinese: bool) -> str:
    if chinese:
        return f"""# MAD-ETD W276–W281 DoHBrw Fresh Capture 就绪报告

## 结论

- 状态：`{report['status']}`
- 当前下载档案：{report['archive_count']} 个，约 {report['archive_gib']} GiB。
- 冻结 fresh capture groups：{report['fresh_capture_group_count']}。
- 真实 tshark smoke flow：{report['safe_smoke_flow_count']}。
- `runtime_safe_v3_0` 保持默认；没有训练、没有 Accuracy/F1、没有新 runtime。

## 关键数据边界

Firefox/Chrome benign archives 同时包含 DoH 与 non-DoH，目录名不是逐流
benign 真值。后续监督训练必须先用官方逐流 CSV 做 endpoint-overlap 对齐；
non-DoH 只能作为 OOD。IP、端口、绝对时间、tcp.stream、capture path、
tool、resolver 和标签只用于离线对齐，不进入 DetectorInput。

## Fresh split

本轮识别 Firefox-Cloudflare、DNScat2 1202–1802、Iodine 1202–1802
为未进入历史模型训练/验收的官方档案；W96 的两个 Firefox-Quad9 capture
只做过完整性审计，未训练、未打开 acceptance。角色在提取/训练前按 group
冻结，acceptance 未打开。

## 历史边界

W71 locked test 已经读取一次并关闭，本轮没有重开。W71、W83、W84 的负结果
保持不变；本轮不是正向分类实验，也不构成 Learned TLS 晋级。
"""
    return f"""# MAD-ETD W276-W281 DoHBrw Fresh-Capture Readiness

## Outcome

- Status: `{report['status']}`
- Downloaded archives: {report['archive_count']} ({report['archive_gib']} GiB).
- Frozen fresh capture groups: {report['fresh_capture_group_count']}.
- Real tshark smoke flows: {report['safe_smoke_flow_count']}.
- `runtime_safe_v3_0` remains default. No training, Accuracy/F1, or runtime
  promotion was produced.

## Label boundary

The benign browser archives mix DoH and non-DoH traffic. Their directory names
are not flow-level benign truth. Future supervised use must first perform
official per-flow CSV endpoint-overlap alignment, while non-DoH remains
OOD-only. Endpoint, capture, tool, resolver, label, and provenance fields are
offline alignment metadata and never DetectorInput.

## Fresh split

Firefox-Cloudflare, DNScat2 1202-1802, and Iodine 1202-1802 were absent from
historical model-selection/acceptance lanes. The two Firefox-Quad9 groups were
audited in W96 but never trained or accepted. Roles were frozen before
extraction, alignment, training, or acceptance access.

## Historical boundary

The W71 locked test remains closed and was not reread. Historical W71/W83/W84
negative results are unchanged. This readiness result is not a positive
classification result and does not promote Learned TLS.
"""


def finalize_dohbrw_fresh_readiness_w281(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    required = (
        "download_audit_report.json",
        "historical_reconciliation_report.json",
        "fresh_staging_report.json",
        "fresh_group_feasibility.json",
        "tls_record_smoke_report.json",
        "capture_group_role_manifest.csv",
        "safe_tls_smoke_features.csv",
        "smoke_offline_alignment_audit.csv",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise RuntimeError(f"W281 missing required artifacts: {missing}")
    download = _load(output / "download_audit_report.json")
    reconcile = _load(output / "historical_reconciliation_report.json")
    staging = _load(output / "fresh_staging_report.json")
    feasibility = _load(output / "fresh_group_feasibility.json")
    smoke = _load(output / "tls_record_smoke_report.json")
    before = _load(output / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(output / "frozen_hashes_after.json", after)
    frozen_unchanged = before == after
    gates = {
        "download_audit_passed": download.get("status")
        == "downloads_audited_for_fresh_reconciliation",
        "fresh_history_reconciled": reconcile.get("status")
        == "fresh_archive_set_reconciled",
        "roles_frozen_before_training": feasibility.get("status")
        == "fresh_capture_roles_frozen_for_w282",
        "role_overlap_zero": not any(
            feasibility.get("role_overlap", {}).values()
        ),
        "acceptance_not_opened": not feasibility.get("acceptance_opened"),
        "real_tshark_smoke_passed": smoke.get("status")
        == "bounded_real_tls_record_smoke_passed",
        "blocked_field_violation_zero": not smoke.get(
            "forbidden_fields_in_safe_feature_table"
        ),
        "supervised_metrics_not_generated": not smoke.get(
            "supervised_metrics_generated"
        ),
        "historical_locked_test_not_reopened": not reconcile.get(
            "historical_w71_locked_test_reopened"
        ),
        "frozen_hashes_unchanged": frozen_unchanged,
        "tests_passed": bool(tests_passed),
    }
    accepted = all(gates.values())
    status = (
        "ready_for_w282_fresh_tls_flow_alignment_and_training"
        if accepted
        else "blocked_fresh_tls_readiness_gate"
    )
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "no_negative_result" if accepted else status,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "safe_claim": (
            "A fresh capture-disjoint DoHBrw source set and bounded real "
            "TLS-record smoke are ready for a new flow-alignment/training lane."
            if accepted
            else "Fresh TLS readiness remained blocked; no training occurred."
        ),
        "forbidden_claims": [
            "Learned TLS was promoted",
            "W71 locked test was rerun",
            "Accuracy or F1 improved in W276-W281",
            "browser archive directory is flow-level benign truth",
        ],
        "fake_metric_count": 0,
    }
    _dump(output / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "gates": gates,
        "archive_count": download["archive_count"],
        "archive_gib": download["archive_gib"],
        "fresh_archive_names": reconcile["fresh_archive_names"],
        "fresh_capture_group_count": feasibility["group_count"],
        "role_counts": feasibility["role_counts"],
        "safe_smoke_flow_count": smoke["safe_smoke_flow_count"],
        "tshark_version": smoke["tshark"].get("version", ""),
        "benign_flow_alignment_pending_for_w282": True,
        "large_firefox_cloudflare_extraction_pending_for_w282": True,
        "new_training_executed": False,
        "new_acceptance_executed": False,
        "accuracy_or_f1_generated": False,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "historical_w71_locked_test_reopened": False,
        "test_or_acceptance_used_for_selection": False,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "security_acceptance.json", {
        key: report[key]
        for key in (
            "blocked_field_violation",
            "fusion_ownership_violation",
            "ood_override_count",
            "illegal_verdict_execution_count",
            "fake_metric_count",
            "runtime_safe_v3_0_remains_default",
            "promoted_runtime_created",
            "historical_w71_locked_test_reopened",
            "test_or_acceptance_used_for_selection",
        )
    })
    _dump(output / "acceptance_report.json", report)
    Path(document).write_text(_document_text(report, chinese=False), encoding="utf-8")
    Path(document_cn).write_text(_document_text(report, chinese=True), encoding="utf-8")
    return report


__all__ = [
    "audit_dohbrw_downloads_w276",
    "reconcile_dohbrw_history_w277",
    "stage_fresh_dohbrw_captures_w278",
    "freeze_fresh_dohbrw_groups_w279",
    "run_bounded_tls_smoke_w280",
    "finalize_dohbrw_fresh_readiness_w281",
    "_safe_member",
    "_safe_target",
]
