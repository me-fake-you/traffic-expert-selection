from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT = "mad_etd_external_multiagent_v5_1"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_external_multiagent_v5_1")
DEFAULT_CICIDS_RAW = Path("data/raw/cicids2017/v1/official_download")
DEFAULT_CICIDS_PROCESSED = Path("data/processed/cicids2017/v1")
DEFAULT_NF_RAW = Path("data/raw/nf_bot_iot_ton_iot/v1/official_download")
DEFAULT_NF_PROCESSED = Path("data/processed/nf_bot_iot_ton_iot/v1")
DEFAULT_DOC = Path("docs/MAD_ETD_EXTERNAL_MULTIAGENT_V5_1.md")


CICIDS_PRIMARY_ARCHIVE = "MachineLearningCSV/MachineLearningCSV.zip"
CICIDS_COMPAT_ARCHIVE = "GeneratedLabelledFlows/GeneratedLabelledFlows.zip"

NF_ARCHIVES: tuple[tuple[str, str], ...] = (
    ("NF-BoT-IoT", "NF-BoT-IoT/*.zip"),
    ("NF-ToN-IoT", "NF-ToN-IoT/*.zip"),
    ("NF-BoT-IoT-v2", "NF-BoT-IoT-v2/*.zip"),
    ("NF-ToN-IoT-v2", "NF-ToN-IoT-v2/*.zip"),
)

BLOCKED_PATTERNS = (
    "flow_id",
    "source_ip",
    "src_ip",
    "ip_src",
    "ipv4_src_addr",
    "destination_ip",
    "dst_ip",
    "ip_dst",
    "ipv4_dst_addr",
    "source_port",
    "src_port",
    "sport",
    "l4_src_port",
    "destination_port",
    "dst_port",
    "dport",
    "l4_dst_port",
    "timestamp",
    "time_stamp",
    "label",
    "attack",
    "family",
    "tool",
    "application",
    "app",
    "source_file",
    "file_name",
    "pcap",
    "payload",
)

CICIDS_VALIDATION_GROUP_MARKERS = (
    "Thursday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
)
CICIDS_TEST_GROUP_MARKERS = (
    "Thursday-WorkingHours-Afternoon",
    "Friday-WorkingHours-Afternoon-DDos",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _write_doc(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lanes = report.get("lanes", [])
    lines = [
        "# MAD-ETD External Multi-Agent v5.1 Data Gate",
        "",
        f"- status: `{report.get('status')}`",
        f"- experiment: `{EXPERIMENT}`",
        f"- default runtime changed: `{report.get('default_runtime_changed', False)}`",
        f"- external numeric results fabricated: `{report.get('external_numeric_results_fabricated', False)}`",
        "",
        "## Lane Status",
        "",
        "| Lane | Status | Ready for Round 2 | Notes |",
        "|---|---:|---:|---|",
    ]
    for lane in lanes:
        lines.append(
            "| {lane} | {status} | {ready} | {notes} |".format(
                lane=lane.get("lane"),
                status=lane.get("status"),
                ready=lane.get("ready_for_round2"),
                notes=lane.get("notes", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- CSV datasets are admitted only after inventory, hash, label audit, and blocked-field audit.",
            "- Flow IDs, IP addresses, ports, timestamps, attack/family fields, and file provenance are context-only.",
            "- No external paper metrics are filled in this round.",
            "- `runtime_safe_v3_0` remains the default runtime.",
        ]
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256_path(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: str, *, salt: str = EXPERIMENT) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _normalise_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _is_blocked_column(name: str) -> bool:
    normalised = _normalise_name(name)
    return any(pattern in normalised for pattern in BLOCKED_PATTERNS)


def _label_column(headers: list[str]) -> str | None:
    for header in headers:
        if _normalise_name(header) == "label":
            return header
    return None


def _attack_column(headers: list[str]) -> str | None:
    for header in headers:
        if _normalise_name(header) == "attack":
            return header
    return None


def _normalise_binary_label(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return None
    if text in {"0", "benign", "normal"}:
        return "benign"
    if text in {"1", "malicious", "attack", "anomaly"}:
        return "malicious"
    if text == "benign":
        return "benign"
    return "malicious"


def _split_from_hash(sample_id: str) -> str:
    value = int(_stable_hash(sample_id)[:8], 16) / 0xFFFFFFFF
    if value < 0.60:
        return "train"
    if value < 0.80:
        return "validation"
    return "test"


def _cicids_group_split(entry: str) -> str:
    if any(marker in entry for marker in CICIDS_VALIDATION_GROUP_MARKERS):
        return "validation"
    if any(marker in entry for marker in CICIDS_TEST_GROUP_MARKERS):
        return "test"
    return "train"


def _zip_csv_entries(zip_path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(zip_path) as archive:
        return [
            info
            for info in archive.infolist()
            if info.filename.lower().endswith(".csv")
            and not info.filename.endswith("/")
        ]


def _primary_entries(zip_path: Path) -> list[str]:
    return [
        info.filename
        for info in _zip_csv_entries(zip_path)
        if "features" not in info.filename.lower()
    ]


def _scan_zip_csv_entry(
    zip_path: Path,
    entry: str,
    *,
    split_mode: str = "hash",
    selection_size: int = 6000,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    selection_ids: list[str] = []
    headers: list[str] = []
    row_count = 0
    import io

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(entry) as raw:
            text = io.TextIOWrapper(
                raw, encoding="utf-8-sig", errors="replace", newline=""
            )
            reader = csv.DictReader(text)
            headers = list(reader.fieldnames or [])
            label_col = _label_column(headers)
            attack_col = _attack_column(headers)
            for index, row in enumerate(reader):
                row_count += 1
                label = _normalise_binary_label(row.get(label_col, "")) if label_col else None
                if label:
                    label_counts[label] += 1
                attack = str(row.get(attack_col, "")).strip() if attack_col else ""
                if attack:
                    attack_counts[attack] += 1
                sample_id = f"{zip_path.name}:{entry}:{index}"
                split = (
                    _cicids_group_split(entry)
                    if split_mode == "cicids_group"
                    else _split_from_hash(sample_id)
                )
                split_counts[split] += 1
                if label:
                    split_label_counts[split][label] += 1
                if split == "validation" and len(selection_ids) < selection_size:
                    selection_ids.append(_stable_hash(sample_id)[:24])
    blocked_columns = [column for column in headers if _is_blocked_column(column)]
    safe_columns = [
        column
        for column in headers
        if column not in blocked_columns and _normalise_name(column) != "label"
    ]
    return {
        "archive": str(zip_path).replace("\\", "/"),
        "entry": entry,
        "row_count": row_count,
        "headers": headers,
        "label_column": _label_column(headers),
        "attack_column": _attack_column(headers),
        "label_counts": dict(label_counts),
        "attack_counts": dict(attack_counts),
        "blocked_columns": blocked_columns,
        "safe_feature_columns": safe_columns,
        "split_counts": dict(split_counts),
        "split_label_counts": {
            split: dict(counts) for split, counts in split_label_counts.items()
        },
        "selection_sample_ids": selection_ids,
    }


def _archive_inventory(zip_path: Path) -> dict[str, Any]:
    entries = _zip_csv_entries(zip_path)
    return {
        "path": str(zip_path).replace("\\", "/"),
        "exists": zip_path.exists(),
        "size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "sha256": _sha256_path(zip_path) if zip_path.exists() else "",
        "zip_readable": True,
        "csv_entries": [
            {
                "name": info.filename,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
            }
            for info in entries
        ],
    }


def _combine_counts(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        counter.update({str(key): int(value) for key, value in item.items()})
    return dict(counter)


def _write_processed_common(
    processed_dir: Path,
    manifest: Mapping[str, Any],
    feature_policy: Mapping[str, Any],
    split_manifest: Mapping[str, Any] | None = None,
) -> None:
    _dump(processed_dir / "dataset_manifest.json", dict(manifest))
    _dump(processed_dir / "feature_policy.json", dict(feature_policy))
    if split_manifest is not None:
        _dump(processed_dir / "splits" / "split-manifest.json", dict(split_manifest))


def prepare_cicids2017_v5(
    raw_dir: str | Path = DEFAULT_CICIDS_RAW,
    processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    selection_size: int = 6000,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    out = Path(output_dir)
    primary_zip = raw / CICIDS_PRIMARY_ARCHIVE
    compat_zip = raw / CICIDS_COMPAT_ARCHIVE
    missing = [str(path) for path in (primary_zip, compat_zip) if not path.exists()]
    if missing:
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "dataset": "CICIDS2017",
            "status": "failed_missing_raw_archive",
            "missing": missing,
        }
        _dump(out / "cicids2017_audit_report.json", report)
        return report

    primary_entries = _primary_entries(primary_zip)
    scanned = [
        _scan_zip_csv_entry(
            primary_zip,
            entry,
            split_mode="cicids_group",
            selection_size=selection_size,
        )
        for entry in primary_entries
    ]
    split_counts = _combine_counts(item["split_counts"] for item in scanned)
    label_counts = _combine_counts(item["label_counts"] for item in scanned)
    selection_ids: list[str] = []
    for item in scanned:
        for sample_id in item["selection_sample_ids"]:
            if len(selection_ids) < selection_size:
                selection_ids.append(sample_id)
    split_label_counts = {
        split: _combine_counts(item["split_label_counts"].get(split, {}) for item in scanned)
        for split in ("train", "validation", "test")
    }
    raw_inventory = {
        "schema_version": "1.0",
        "dataset": "CICIDS2017",
        "archives": [_archive_inventory(primary_zip), _archive_inventory(compat_zip)],
    }
    feature_policy = {
        "schema_version": "1.0",
        "dataset": "CICIDS2017",
        "payload_bytes_allowed": False,
        "blocked_field_policy": "FieldAudit/context-only",
        "blocked_patterns": list(BLOCKED_PATTERNS),
        "safe_feature_columns": sorted(
            set().union(*(set(item["safe_feature_columns"]) for item in scanned))
        ),
        "blocked_columns": sorted(
            set().union(*(set(item["blocked_columns"]) for item in scanned))
        ),
        "label_context_only": True,
        "attack_type_context_only": True,
    }
    split_manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "CICIDS2017",
        "split_strategy": "cicids_day_file_group_v5_1",
        "split_counts": split_counts,
        "split_label_counts": split_label_counts,
        "selection_sample_ids": selection_ids,
        "selection_count": len(selection_ids),
        "final_acceptance_source_split": "test",
        "final_acceptance_count": split_counts.get("test", 0),
        "final_acceptance_sample_ids_included": False,
        "test_used_for_selection": False,
        "locked_test_used_for_selection": False,
    }
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "CICIDS2017",
        "raw_dir": str(raw).replace("\\", "/"),
        "processed_dir": str(processed).replace("\\", "/"),
        "primary_archive": str(primary_zip).replace("\\", "/"),
        "compatibility_archive": str(compat_zip).replace("\\", "/"),
        "primary_entries": scanned,
        "row_count": sum(item["row_count"] for item in scanned),
        "label_counts": label_counts,
        "split_counts": split_counts,
        "status": "prepared",
    }
    processed.mkdir(parents=True, exist_ok=True)
    _write_processed_common(processed, manifest, feature_policy, split_manifest)
    _dump(out / "raw_inventory.json", raw_inventory)
    _write_csv(
        out / "raw_hashes.csv",
        [
            {
                "dataset": "CICIDS2017",
                "path": archive["path"],
                "size_bytes": archive["size_bytes"],
                "sha256": archive["sha256"],
            }
            for archive in raw_inventory["archives"]
        ],
    )
    _write_csv(
        out / "label_distribution_report.csv",
        [
            {
                "dataset": "CICIDS2017",
                "scope": "primary",
                "label": label,
                "count": count,
            }
            for label, count in sorted(label_counts.items())
        ],
    )
    report = audit_cicids2017_v5(processed, out)
    return report


def audit_cicids2017_v5(
    processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    out = Path(output_dir)
    manifest_path = processed / "dataset_manifest.json"
    policy_path = processed / "feature_policy.json"
    split_path = processed / "splits" / "split-manifest.json"
    if not manifest_path.exists() or not policy_path.exists() or not split_path.exists():
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "dataset": "CICIDS2017",
            "status": "failed_missing_processed_manifest",
        }
        _dump(out / "cicids2017_audit_report.json", report)
        return report
    manifest = _read_json(manifest_path)
    policy = _read_json(policy_path)
    split = _read_json(split_path)
    labels = manifest.get("label_counts", {})
    blocked_leakage = [
        column
        for column in policy.get("safe_feature_columns", [])
        if _is_blocked_column(column)
    ]
    split_counts = split.get("split_counts", {})
    split_has_rows = all(split_counts.get(name, 0) > 0 for name in ("train", "validation", "test"))
    split_has_binary_labels = all(
        split.get("split_label_counts", {}).get(name, {}).get("benign", 0) > 0
        and split.get("split_label_counts", {}).get(name, {}).get("malicious", 0) > 0
        for name in ("train", "validation", "test")
    )
    passed = (
        labels.get("benign", 0) > 0
        and labels.get("malicious", 0) > 0
        and not blocked_leakage
        and split_has_rows
        and split_has_binary_labels
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "CICIDS2017",
        "status": "passed" if passed else "failed_data_audit",
        "row_count": manifest.get("row_count", 0),
        "label_counts": labels,
        "blocked_feature_leakage_count": len(blocked_leakage),
        "blocked_feature_leakage": blocked_leakage,
        "split_counts": split_counts,
        "split_has_binary_labels": split_has_binary_labels,
        "test_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "ready_for_round2": passed,
    }
    _dump(out / "cicids2017_audit_report.json", report)
    return report


def freeze_cicids2017_v5_splits(
    processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    selection_size: int = 6000,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    out = Path(output_dir)
    split_path = processed / "splits" / "split-manifest.json"
    if not split_path.exists():
        report = prepare_cicids2017_v5(output_dir=out, selection_size=selection_size)
        if report.get("status") != "passed":
            return report
    split = _read_json(split_path)
    split["selection_sample_ids"] = split.get("selection_sample_ids", [])[:selection_size]
    split["selection_count"] = len(split["selection_sample_ids"])
    _dump(split_path, split)
    _dump(out / "cicids2017_split_manifest.json", split)
    return split


def _nf_archives(raw: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for dataset, pattern in NF_ARCHIVES:
        matches = sorted(raw.glob(pattern))
        if matches:
            found.append((dataset, matches[0]))
    return found


def prepare_nf_iot_v5(
    raw_dir: str | Path = DEFAULT_NF_RAW,
    processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    selection_size: int = 6000,
) -> dict[str, Any]:
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    out = Path(output_dir)
    archives = _nf_archives(raw)
    missing = [dataset for dataset, _pattern in NF_ARCHIVES if dataset not in {item[0] for item in archives}]
    if len(archives) < 2:
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "dataset": "NF-BoT-IoT/NF-ToN-IoT",
            "status": "failed_missing_raw_archive",
            "missing": missing,
        }
        _dump(out / "nf_iot_audit_report.json", report)
        return report
    scanned: list[dict[str, Any]] = []
    inventories = []
    for dataset, archive in archives:
        inventories.append({"dataset": dataset, **_archive_inventory(archive)})
        for entry in _primary_entries(archive):
            item = _scan_zip_csv_entry(
                archive,
                entry,
                split_mode="hash",
                selection_size=math.ceil(selection_size / max(1, len(archives))),
            )
            item["dataset_variant"] = dataset
            scanned.append(item)
    label_counts = _combine_counts(item["label_counts"] for item in scanned)
    split_counts = _combine_counts(item["split_counts"] for item in scanned)
    split_label_counts = {
        split: _combine_counts(item["split_label_counts"].get(split, {}) for item in scanned)
        for split in ("train", "validation", "test")
    }
    selection_ids: list[str] = []
    for item in scanned:
        for sample_id in item["selection_sample_ids"]:
            if len(selection_ids) < selection_size:
                selection_ids.append(sample_id)
    raw_inventory = {
        "schema_version": "1.0",
        "dataset": "NF-BoT-IoT/NF-ToN-IoT",
        "archives": inventories,
    }
    feature_policy = {
        "schema_version": "1.0",
        "dataset": "NF-BoT-IoT/NF-ToN-IoT",
        "payload_bytes_allowed": False,
        "blocked_field_policy": "FieldAudit/context-only",
        "blocked_patterns": list(BLOCKED_PATTERNS),
        "safe_feature_columns": sorted(
            set().union(*(set(item["safe_feature_columns"]) for item in scanned))
        ),
        "blocked_columns": sorted(
            set().union(*(set(item["blocked_columns"]) for item in scanned))
        ),
        "label_context_only": True,
        "attack_type_context_only": True,
        "v1_v2_mixed_reporting_allowed": False,
    }
    split_manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "NF-BoT-IoT/NF-ToN-IoT",
        "split_strategy": "stable_row_hash_per_dataset_variant_v5_1",
        "split_counts": split_counts,
        "split_label_counts": split_label_counts,
        "selection_sample_ids": selection_ids,
        "selection_count": len(selection_ids),
        "final_acceptance_source_split": "test",
        "final_acceptance_count": split_counts.get("test", 0),
        "final_acceptance_sample_ids_included": False,
        "test_used_for_selection": False,
        "locked_test_used_for_selection": False,
    }
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "NF-BoT-IoT/NF-ToN-IoT",
        "raw_dir": str(raw).replace("\\", "/"),
        "processed_dir": str(processed).replace("\\", "/"),
        "archive_count": len(archives),
        "missing_optional_variants": missing,
        "primary_entries": scanned,
        "row_count": sum(item["row_count"] for item in scanned),
        "label_counts": label_counts,
        "split_counts": split_counts,
        "status": "prepared",
    }
    processed.mkdir(parents=True, exist_ok=True)
    _write_processed_common(processed, manifest, feature_policy, split_manifest)
    _dump(out / "nf_iot_raw_inventory.json", raw_inventory)
    _write_csv(
        out / "nf_iot_raw_hashes.csv",
        [
            {
                "dataset": archive["dataset"],
                "path": archive["path"],
                "size_bytes": archive["size_bytes"],
                "sha256": archive["sha256"],
            }
            for archive in inventories
        ],
    )
    _write_csv(
        out / "nf_iot_label_distribution_report.csv",
        [
            {
                "dataset": item["dataset_variant"],
                "entry": item["entry"],
                "label": label,
                "count": count,
            }
            for item in scanned
            for label, count in sorted(item["label_counts"].items())
        ],
    )
    report = audit_nf_iot_v5(processed, out)
    return report


def audit_nf_iot_v5(
    processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    out = Path(output_dir)
    manifest_path = processed / "dataset_manifest.json"
    policy_path = processed / "feature_policy.json"
    split_path = processed / "splits" / "split-manifest.json"
    if not manifest_path.exists() or not policy_path.exists() or not split_path.exists():
        report = {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "dataset": "NF-BoT-IoT/NF-ToN-IoT",
            "status": "failed_missing_processed_manifest",
        }
        _dump(out / "nf_iot_audit_report.json", report)
        return report
    manifest = _read_json(manifest_path)
    policy = _read_json(policy_path)
    split = _read_json(split_path)
    labels = manifest.get("label_counts", {})
    blocked_leakage = [
        column
        for column in policy.get("safe_feature_columns", [])
        if _is_blocked_column(column)
    ]
    split_counts = split.get("split_counts", {})
    split_has_rows = all(split_counts.get(name, 0) > 0 for name in ("train", "validation", "test"))
    split_has_binary_labels = all(
        split.get("split_label_counts", {}).get(name, {}).get("benign", 0) > 0
        and split.get("split_label_counts", {}).get(name, {}).get("malicious", 0) > 0
        for name in ("train", "validation", "test")
    )
    passed = (
        labels.get("benign", 0) > 0
        and labels.get("malicious", 0) > 0
        and not blocked_leakage
        and split_has_rows
        and split_has_binary_labels
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "dataset": "NF-BoT-IoT/NF-ToN-IoT",
        "status": "passed" if passed else "failed_data_audit",
        "row_count": manifest.get("row_count", 0),
        "label_counts": labels,
        "blocked_feature_leakage_count": len(blocked_leakage),
        "blocked_feature_leakage": blocked_leakage,
        "split_counts": split_counts,
        "split_has_binary_labels": split_has_binary_labels,
        "test_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "ready_for_round2": passed,
    }
    _dump(out / "nf_iot_audit_report.json", report)
    return report


def freeze_nf_iot_v5_splits(
    processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    selection_size: int = 6000,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    out = Path(output_dir)
    split_path = processed / "splits" / "split-manifest.json"
    if not split_path.exists():
        report = prepare_nf_iot_v5(output_dir=out, selection_size=selection_size)
        if report.get("status") != "passed":
            return report
    split = _read_json(split_path)
    split["selection_sample_ids"] = split.get("selection_sample_ids", [])[:selection_size]
    split["selection_count"] = len(split["selection_sample_ids"])
    _dump(split_path, split)
    _dump(out / "nf_iot_split_manifest.json", split)
    return split


def refresh_external_multiagent_lanes_v5(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    cicids_processed_dir: str | Path = DEFAULT_CICIDS_PROCESSED,
    nf_processed_dir: str | Path = DEFAULT_NF_PROCESSED,
    document: str | Path = DEFAULT_DOC,
) -> dict[str, Any]:
    out = Path(output_dir)
    cic = audit_cicids2017_v5(cicids_processed_dir, out)
    nf = audit_nf_iot_v5(nf_processed_dir, out)
    cic_processed = Path(cicids_processed_dir)
    nf_processed = Path(nf_processed_dir)
    raw_inventory: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "datasets": [],
    }
    hash_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for dataset_name, processed_dir in (
        ("CICIDS2017", cic_processed),
        ("NF-BoT-IoT/NF-ToN-IoT", nf_processed),
    ):
        manifest_path = processed_dir / "dataset_manifest.json"
        policy_path = processed_dir / "feature_policy.json"
        if not manifest_path.exists() or not policy_path.exists():
            continue
        manifest = _read_json(manifest_path)
        policy = _read_json(policy_path)
        raw_inventory["datasets"].append(
            {
                "dataset": dataset_name,
                "row_count": manifest.get("row_count", 0),
                "label_counts": manifest.get("label_counts", {}),
                "status": manifest.get("status"),
            }
        )
        for label, count in sorted(manifest.get("label_counts", {}).items()):
            label_rows.append(
                {
                    "dataset": dataset_name,
                    "scope": "processed_primary",
                    "label": label,
                    "count": count,
                }
            )
        field_rows.append(
            {
                "dataset": dataset_name,
                "safe_feature_count": len(policy.get("safe_feature_columns", [])),
                "blocked_column_count": len(policy.get("blocked_columns", [])),
                "blocked_feature_leakage_count": sum(
                    1
                    for column in policy.get("safe_feature_columns", [])
                    if _is_blocked_column(column)
                ),
                "payload_bytes_allowed": policy.get("payload_bytes_allowed"),
                "label_context_only": policy.get("label_context_only"),
                "attack_type_context_only": policy.get("attack_type_context_only"),
            }
        )
    for source in (out / "raw_hashes.csv", out / "nf_iot_raw_hashes.csv"):
        if not source.exists():
            continue
        with source.open(encoding="utf-8-sig", newline="") as handle:
            hash_rows.extend(csv.DictReader(handle))
    _dump(out / "raw_inventory.json", raw_inventory)
    _write_csv(out / "raw_hashes.csv", hash_rows)
    _write_csv(out / "label_distribution_report.csv", label_rows)
    _dump(
        out / "field_safety_report.json",
        {
            "schema_version": "1.0",
            "experiment": EXPERIMENT,
            "status": "passed"
            if all(row["blocked_feature_leakage_count"] == 0 for row in field_rows)
            else "failed_blocked_feature_leakage",
            "rows": field_rows,
        },
    )
    lanes = [
        {
            "lane": "osf_eimtc",
            "status": "encrypted_traffic_baseline_not_multi_agent",
            "ready_for_round2": False,
            "notes": "retained only for encrypted-traffic single-model/framework comparison",
        },
        {
            "lane": "cicids2017_multiagent_lane",
            "status": "prepared" if cic.get("status") == "passed" else "failed_data_audit",
            "ready_for_round2": cic.get("status") == "passed",
            "notes": "Continual-Federated-IDS / MARL-NIDS candidate lane",
        },
        {
            "lane": "nf_bot_iot_ton_iot_ma_ids_lane",
            "status": "prepared" if nf.get("status") == "passed" else "failed_data_audit",
            "ready_for_round2": nf.get("status") == "passed",
            "notes": "MA-IDS-style NF-BoT-IoT/NF-ToN-IoT candidate lane",
        },
    ]
    passed = cic.get("status") == "passed" and nf.get("status") == "passed"
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "stage": "external_multiagent_lane_refresh",
        "status": "passed" if passed else "partial",
        "lanes": lanes,
        "cicids2017_status": cic.get("status"),
        "nf_iot_status": nf.get("status"),
        "default_runtime_changed": False,
        "runtime_safe_v3_0_remains_default": True,
        "external_numeric_results_fabricated": False,
        "test_used_for_selection": False,
        "locked_test_used_for_selection": False,
        "next_recommended_action": "run_multiagent_baseline_round2_after_lane_review",
    }
    _dump(out / "external_multiagent_lane_status.json", report)
    _dump(out / "acceptance_report.json", report)
    _write_doc(document, report)
    return report
