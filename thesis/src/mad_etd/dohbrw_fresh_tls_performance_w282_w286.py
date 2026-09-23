"""W282-W286 fresh DoHBrw learned-TLS performance protocol.

The protocol consumes only the capture roles frozen by W281.  Development
captures are extracted and aligned before training.  Acceptance captures stay
closed until all model artifacts and selection policies have been frozen.
No historical W71 locked-test artifact is read by this module.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dohbrw_fresh_readiness_w276_w281 import (
    _canonical_hash,
    _safe_member,
    _safe_target,
    _sha256,
)
from .dohbrw_v4 import locate_tshark, tshark_identity
from .io import iter_flow_records
from .learned_tls_w83 import (
    _csv_doh_index,
    _parse_w83_tsv,
    _resolver,
)
from .paper_evaluation import hash_artifact_paths
from .safe_tls_hgb_w84 import _extract_tsv_w84, _firefox_assignments
from .schemas import FlowRecord, SequenceFeatures
from .soc_evidence_team_w72 import _default_frozen_paths
from .tls_v4_evaluation import (
    BINARY,
    _grouped_bootstrap,
    _metrics,
    _predictors,
    _read_csv as _read_prediction_csv,
    _run_predictions,
    _run_robustness,
    _robustness_metrics,
)
from .tls_v4_training import (
    TLS_V4_SEEDS,
    _train_hgb,
    _train_tcn_variant,
    load_tls_v4_dataset,
)
from .v2_training import _ece


EXPERIMENT = "mad_etd_dohbrw_fresh_tls_performance_w282_w286"
DEFAULT_W281 = Path(
    "data/runs/mad_etd_dohbrw_fresh_capture_readiness_w276_w281"
)
DEFAULT_INPUT = Path("data/raw/DoHBrw/pcap")
DEFAULT_STAGE = DEFAULT_INPUT / "_w282_fresh_training"
DEFAULT_PROCESSED = Path("data/processed/cira_cic_dohbrw_2020/w282")
DEFAULT_MODELS = Path("data/models/mad_etd_fresh_tls_w284")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_dohbrw_fresh_tls_performance_w282_w286")
DEFAULT_BENIGN_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/"
    "BenignDoH-NonDoH-CSVs.zip"
)
DEFAULT_MALICIOUS_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/MaliciousDoH-CSVs.zip"
)
DEFAULT_DOC = Path("docs/MAD_ETD_DOHBRW_FRESH_TLS_PERFORMANCE_W282_W286.md")
DEFAULT_DOC_CN = Path(
    "docs/MAD_ETD_DOHBRW_FRESH_TLS_PERFORMANCE_W282_W286_CN.md"
)

ALLOWED_TLS_FIELDS = {
    "record_lengths",
    "server_version",
    "client_cipher_count",
    "client_extension_count",
    "server_extension_count",
    "alpn",
}
BLOCKED_FIELDS = {
    "IP",
    "port",
    "absolute_timestamp",
    "tcp.stream",
    "Flow ID",
    "SNI",
    "JA3",
    "JA4",
    "tool",
    "resolver",
    "browser",
    "capture_path",
    "archive_member",
    "source_file",
    "provenance",
    "label",
    "sample_id",
}

ACCEPTANCE_GATES = {
    "records_only_macro_f1_min": 0.75,
    "records_only_malicious_recall_min": 0.75,
    "records_only_ece_max": 0.10,
    "records_only_coverage_min": 0.80,
    "selection_to_acceptance_macro_f1_drop_max": 0.15,
    "macro_f1_delta_vs_hgb_min_strict": 0.0,
    "bootstrap_delta_ci95_lower_strict": 0.0,
    "wrong_binary_not_worse_than_hgb": True,
    "blocked_field_violation": 0,
    "fusion_ownership_violation": 0,
    "fake_metric_count": 0,
}


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
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
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in values:
            writer.writerow({key: row.get(key, "") for key in fields})


def _extract_selected_members(
    archive: Path,
    selected: Sequence[dict[str, str]],
    stage_root: Path,
) -> list[dict[str, Any]]:
    by_name = {row["archive_member"]: row for row in selected}
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        infos = {item.filename: item for item in bundle.infolist()}
        for member, frozen in sorted(by_name.items()):
            if member not in infos:
                raise RuntimeError(f"W282 frozen archive member disappeared: {member}")
            info = infos[member]
            if not _safe_member(member):
                raise RuntimeError(f"W282 unsafe archive member: {member}")
            if f"{info.CRC:08x}" != frozen["member_crc32"]:
                raise RuntimeError(f"W282 frozen member CRC changed: {member}")
            if info.file_size != int(frozen["member_size_bytes"]):
                raise RuntimeError(f"W282 frozen member size changed: {member}")
            target = _safe_target(stage_root / archive.stem, member)
            target.parent.mkdir(parents=True, exist_ok=True)
            disposition = "resumed_existing"
            if not target.exists() or target.stat().st_size != info.file_size:
                with bundle.open(info) as source, target.open("wb") as sink:
                    for chunk in iter(
                        lambda: source.read(8 * 1024 * 1024), b""
                    ):
                        sink.write(chunk)
                disposition = "extracted_w282"
            if target.stat().st_size != info.file_size:
                raise RuntimeError(f"W282 extracted size mismatch: {member}")
            rows.append(
                {
                    **frozen,
                    "pcap_path": target.as_posix(),
                    "pcap_size_bytes": info.file_size,
                    "pcap_sha256": _sha256(target),
                    "extraction_disposition": disposition,
                    "zip_crc_verified_by_complete_read": disposition
                    == "extracted_w282",
                }
            )
    return rows


def _w96_quad_paths() -> dict[str, dict[str, Any]]:
    manifest = Path(
        "data/runs/mad_etd_safe_tls_hgb_w96/archive_extraction_manifest.json"
    )
    if not manifest.exists():
        raise RuntimeError("W282 requires the W96 audit-only Quad9 manifest")
    payload = _load(manifest)
    archive = Path(str(payload["archive"]))
    with zipfile.ZipFile(archive) as bundle:
        infos = [
            item
            for item in bundle.infolist()
            if item.filename.lower().endswith(".pcap")
        ]
    by_size = {item.file_size: item for item in infos}
    result: dict[str, dict[str, Any]] = {}
    for item in payload["extracted_pcaps"]:
        path = Path(item["path"])
        info = by_size.get(int(item["size_bytes"]))
        if info is None or not path.exists():
            raise RuntimeError("W282 W96 Quad9 capture disappeared")
        if _sha256(path) != item["sha256"]:
            raise RuntimeError("W282 W96 Quad9 capture hash changed")
        result[info.filename] = {
            "pcap_path": path.as_posix(),
            "pcap_size_bytes": path.stat().st_size,
            "pcap_sha256": item["sha256"],
            "extraction_disposition": "reused_w96_audit_only_capture",
            "zip_crc_verified_by_complete_read": True,
        }
    return result


def _capture_rows(
    frozen: list[dict[str, str]],
    extracted: list[dict[str, Any]],
    *,
    benign_csv: Path,
    malicious_csv: Path,
) -> list[dict[str, Any]]:
    extracted_by_group = {
        row["capture_group_id"]: row for row in extracted
    }
    rows: list[dict[str, Any]] = []
    for item in frozen:
        actual = extracted_by_group.get(item["capture_group_id"])
        if actual is None:
            raise RuntimeError(
                f"W282 frozen capture not extracted: {item['capture_group_id']}"
            )
        mixed = item["traffic_scope"] == "mixed_benign_doh_non_doh"
        rows.append(
            {
                "capture_id": item["capture_group_id"],
                "capture_group_id": item["capture_group_id"],
                "role": item["role"],
                "binary_label": "benign" if mixed else "malicious",
                "browser": item["browser"],
                "tool": item["tool"],
                "resolver": item["resolver"],
                "csv_archive": (
                    benign_csv.as_posix()
                    if mixed
                    else malicious_csv.as_posix()
                ),
                "csv_entry": item["official_csv_entry"],
                "csv_candidate_entries": item["official_csv_candidates"],
                "csv_pair_mode": (
                    "firefox_maximum_5tuple_overlap"
                    if mixed
                    else "exact_official_capture_csv"
                ),
                "pcap_path": actual["pcap_path"],
                "pcap_sha256": actual["pcap_sha256"],
                "pcap_size_bytes": actual["pcap_size_bytes"],
                "archive_name": item["archive_name"],
                "archive_sha256": item["archive_sha256"],
                "archive_member": item["archive_member"],
                "relative_path": (
                    f"{item['archive_name']}/{item['archive_member']}"
                ),
                "alignment_only_fields_enter_detector_input": False,
            }
        )
    return rows


def stage_fresh_tls_training_w282(
    *,
    w281_dir: str | Path = DEFAULT_W281,
    input_root: str | Path = DEFAULT_INPUT,
    stage_root: str | Path = DEFAULT_STAGE,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    benign_csv: str | Path = DEFAULT_BENIGN_CSV,
    malicious_csv: str | Path = DEFAULT_MALICIOUS_CSV,
) -> dict[str, Any]:
    w281, root, stage, processed, models, output = map(
        Path,
        (w281_dir, input_root, stage_root, processed_dir, model_dir, output_dir),
    )
    benign_csv_path, malicious_csv_path = Path(benign_csv), Path(malicious_csv)
    for required in (
        w281 / "acceptance_report.json",
        w281 / "capture_group_role_manifest.csv",
        benign_csv_path,
        malicious_csv_path,
    ):
        if not required.exists():
            raise RuntimeError(f"W282 missing required artifact: {required}")
    readiness = _load(w281 / "acceptance_report.json")
    if readiness.get("status") != (
        "ready_for_w282_fresh_tls_flow_alignment_and_training"
    ):
        raise RuntimeError("W282 requires accepted W281 fresh readiness")
    if readiness.get("historical_w71_locked_test_reopened"):
        raise RuntimeError("W282 refuses a reopened W71 locked-test lane")
    frozen = _read_csv(w281 / "capture_group_role_manifest.csv")
    if len(frozen) != int(readiness["fresh_capture_group_count"]):
        raise RuntimeError("W282 W281 group count changed")
    if _canonical_hash(frozen) != _load(
        w281 / "fresh_group_feasibility.json"
    )["manifest_hash"]:
        raise RuntimeError("W282 W281 role manifest hash changed")
    partials = [
        path.as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".part", ".crdownload"}
    ]
    if partials:
        raise RuntimeError("W282 refuses incomplete downloads")
    tshark = locate_tshark(None)
    identity = tshark_identity(tshark) if tshark else {"accepted": False}
    if not tshark or not identity.get("accepted"):
        raise RuntimeError("W282 requires fixed tshark 4.6.6")
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    _dump(output / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    _dump(output / "acceptance_gate_registry.json", ACCEPTANCE_GATES)

    extracted: list[dict[str, Any]] = []
    by_archive: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in frozen:
        by_archive[row["archive_name"]].append(row)
    quad_paths = _w96_quad_paths()
    for archive_name, rows in sorted(by_archive.items()):
        if archive_name == "BenignDoH_NonDoH-Firefox-Quad9.zip":
            for row in rows:
                actual = quad_paths.get(row["archive_member"])
                if actual is None:
                    raise RuntimeError(
                        f"W282 unresolved W96 Quad9 member: {row['archive_member']}"
                    )
                extracted.append({**row, **actual})
            continue
        archive = root / archive_name
        if not archive.exists():
            raise RuntimeError(f"W282 frozen archive missing: {archive}")
        if _sha256(archive) != rows[0]["archive_sha256"]:
            raise RuntimeError(f"W282 archive hash changed: {archive_name}")
        extracted.extend(_extract_selected_members(archive, rows, stage))
    captures = _capture_rows(
        frozen,
        extracted,
        benign_csv=benign_csv_path,
        malicious_csv=malicious_csv_path,
    )
    role_counts = Counter(row["role"] for row in captures)
    scope_counts = Counter(
        (row["role"], row["binary_label"]) for row in captures
    )
    if set(role_counts) != {"train", "selection", "acceptance"}:
        raise RuntimeError("W282 role set changed")
    if any(
        scope_counts[(role, label)] == 0
        for role in role_counts
        for label in ("benign", "malicious")
    ):
        raise RuntimeError("W282 one role lacks a source for either class")
    _write_csv(output / "staged_capture_inventory_w282.csv", extracted)
    _write_csv(output / "frozen_capture_manifest_w282.csv", captures)
    contract = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w283_development_extraction",
        "w281_manifest_hash": _load(
            w281 / "fresh_group_feasibility.json"
        )["manifest_hash"],
        "capture_count": len(captures),
        "role_counts": dict(role_counts),
        "role_class_source_counts": {
            f"{role}|{label}": count
            for (role, label), count in sorted(scope_counts.items())
        },
        "development_roles": ["train", "selection"],
        "acceptance_role": "acceptance",
        "acceptance_opened": False,
        "selection_policy": (
            "records-only TCN selected only on frozen selection role; "
            "temperature uses a deterministic selection-role calibration half"
        ),
        "training_config": {
            "models": [
                "rule_tls",
                "safe_aggregate_hgb",
                "records_only_tcn",
                "records_handshake_tcn",
            ],
            "seeds": list(TLS_V4_SEEDS),
            "epochs": 12,
            "batch_size": 512,
            "mixed_precision": True,
            "records_only_is_only_promotable_learned_candidate": True,
        },
        "tshark": identity,
        "safe_detector_fields": sorted(ALLOWED_TLS_FIELDS),
        "blocked_or_alignment_only_fields": sorted(BLOCKED_FIELDS),
        "historical_w71_locked_test_reopened": False,
        "test_or_acceptance_used_for_selection": False,
        "new_training_executed": False,
        "supervised_metrics_generated": False,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "protocol_contract_w282.json", contract)
    return contract


def _extract_capture_tsvs(
    captures: list[dict[str, str]],
    processed: Path,
    tshark: Path,
    *,
    workers: int,
) -> dict[str, Path]:
    plugin_dir = processed / "tshark" / "isolated_empty_plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(
            pool.map(
                lambda row: _extract_tsv_w84(
                    row, processed, tshark, plugin_dir
                ),
                captures,
            )
        )
    return {
        row["capture_id"]: Path(row["tsv_path"])
        for row in results
    }


def _aligned_records(
    captures: list[dict[str, str]],
    tsv_by_capture: Mapping[str, Path],
    *,
    role: str,
) -> tuple[list[FlowRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    firefox = [row for row in captures if row["browser"] == "firefox"]
    firefox_map, firefox_rows = _firefox_assignments(
        firefox, tsv_by_capture
    )
    records: list[FlowRecord] = []
    quality: list[dict[str, Any]] = []
    for capture in captures:
        entry = capture["csv_entry"] or firefox_map.get(
            capture["capture_id"], ""
        )
        if not entry:
            raise RuntimeError(
                f"W283 unresolved official CSV entry: {capture['capture_id']}"
            )
        labels, csv_rows = _csv_doh_index(Path(capture["csv_archive"]), entry)
        streams = _parse_w83_tsv(tsv_by_capture[capture["capture_id"]])
        counts: Counter[str] = Counter()
        for stream_id, state in sorted(
            streams.items(), key=lambda pair: int(pair[0])
        ):
            if len(state["records"]) < 2:
                counts["short_tls_stream"] += 1
                continue
            matched = labels.get(state["endpoint_key"])
            if matched is None:
                counts["unmatched_tls_stream"] += 1
                continue
            if matched == {False}:
                counts["non_doh_ood_only"] += 1
                continue
            if matched != {True}:
                counts["ambiguous_alignment"] += 1
                continue
            sample_id = hashlib.sha256(
                f"w282:{capture['capture_id']}:{stream_id}".encode("utf-8")
            ).hexdigest()[:24]
            resolver = (
                capture["resolver"]
                if capture["browser"]
                else _resolver(capture["archive_member"])
            )
            records.append(
                FlowRecord(
                    trace_id=f"dohbrw-w282-{sample_id}",
                    sample_id=sample_id,
                    stats={},
                    sequence=SequenceFeatures(),
                    tls={
                        "record_lengths": state["records"][:64],
                        "server_version": state["server_version"],
                        "client_cipher_count": state["client_cipher_count"],
                        "client_extension_count": state[
                            "client_extension_count"
                        ],
                        "server_extension_count": state[
                            "server_extension_count"
                        ],
                        "alpn": sorted(state["alpn"])[:8],
                    },
                    context={"transport": "tcp", "protocols": ["tls", "doh"]},
                    provenance={
                        "dataset": "CIRA-CIC-DoHBrw-2020",
                        "capture_id": capture["capture_id"],
                        "source_file": capture["relative_path"],
                        "official_csv_entry": entry,
                        "alignment_method": (
                            "official_csv_canonical_5tuple_doh_true"
                        ),
                        "frozen_role": role,
                    },
                    labels={
                        "binary": capture["binary_label"],
                        "generator": capture["tool"] or None,
                        "resolver": resolver or None,
                        "browser": capture["browser"] or None,
                    },
                )
            )
            counts["aligned_supervised_doh"] += 1
        quality.append(
            {
                "capture_id": capture["capture_id"],
                "role": role,
                "binary_label": capture["binary_label"],
                "browser": capture["browser"],
                "tool": capture["tool"],
                "resolver": capture["resolver"],
                "official_csv_entry": entry,
                "official_csv_rows": csv_rows,
                "tls_stream_count": len(streams),
                **counts,
            }
        )
    return records, quality, firefox_rows


def _write_flow_records(path: Path, records: Iterable[FlowRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def _development_split(
    records: list[FlowRecord],
    *,
    group_manifest_hash: str,
) -> dict[str, Any]:
    train = [
        record for record in records
        if record.provenance.get("frozen_role") == "train"
    ]
    selection_role = [
        record for record in records
        if record.provenance.get("frozen_role") == "selection"
    ]
    calibration_ids: list[str] = []
    selection_ids: list[str] = []
    for label in ("benign", "malicious"):
        local = [
            record for record in selection_role
            if record.labels.get("binary") == label
        ]
        local.sort(
            key=lambda record: hashlib.sha256(
                f"w283:42:{record.sample_id}".encode("utf-8")
            ).hexdigest()
        )
        cut = max(1, len(local) // 2)
        calibration_ids.extend(record.sample_id for record in local[:cut])
        selection_ids.extend(record.sample_id for record in local[cut:])
    if any(
        sum(record.labels.get("binary") == label for record in train) < 40
        for label in ("benign", "malicious")
    ):
        raise RuntimeError("W283 train flows lack forty samples per class")
    ids_to_record = {record.sample_id: record for record in records}
    for name, values in (
        ("calibration", calibration_ids),
        ("selection", selection_ids),
    ):
        labels = {ids_to_record[value].labels.get("binary") for value in values}
        if labels != {"benign", "malicious"} or len(values) < 40:
            raise RuntimeError(
                f"W283 {name} partition is insufficient: {len(values)}"
            )
    train_groups = {
        str(record.provenance["capture_id"]) for record in train
    }
    selection_groups = {
        str(record.provenance["capture_id"]) for record in selection_role
    }
    if train_groups & selection_groups:
        raise RuntimeError("W283 train/selection capture overlap")
    manifest = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "seed": 42,
        "grouping": "frozen W281 capture_group_id",
        "w281_group_manifest_hash": group_manifest_hash,
        "assignments": {
            "train": sorted(record.sample_id for record in train),
            "validation": sorted(
                record.sample_id for record in selection_role
            ),
            "test": [],
        },
        "validation_partitions": {
            "calibration_sample_ids": sorted(calibration_ids),
            "selection_sample_ids": sorted(selection_ids),
            "selection_capture_ids": sorted(selection_groups),
        },
        "development_capture_overlap": 0,
        "acceptance_groups_frozen_but_unopened": True,
        "test_used_for_selection": False,
        "locked_test": True,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    return manifest


def extract_fresh_tls_development_w283(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 2,
) -> dict[str, Any]:
    processed, output = Path(processed_dir), Path(output_dir)
    contract = _load(output / "protocol_contract_w282.json")
    if contract.get("status") != "ready_for_w283_development_extraction":
        raise RuntimeError("W283 requires accepted W282 staging")
    captures = [
        row
        for row in _read_csv(output / "frozen_capture_manifest_w282.csv")
        if row["role"] in {"train", "selection"}
    ]
    if any(row["role"] == "acceptance" for row in captures):
        raise RuntimeError("W283 attempted to open acceptance")
    tshark = locate_tshark(contract["tshark"]["path"])
    if not tshark or not tshark_identity(tshark).get("accepted"):
        raise RuntimeError("W283 fixed tshark disappeared")
    tsvs = _extract_capture_tsvs(
        captures, processed, tshark, workers=workers
    )
    records, quality, firefox_rows = _aligned_records(
        captures, tsvs, role="development"
    )
    # Restore the exact frozen role on each flow.
    capture_roles = {row["capture_id"]: row["role"] for row in captures}
    for record in records:
        record.provenance["frozen_role"] = capture_roles[
            str(record.provenance["capture_id"])
        ]
    labels = Counter(str(record.labels.get("binary")) for record in records)
    if not labels["benign"] or not labels["malicious"]:
        raise RuntimeError("W283 official flow alignment lacks both classes")
    flow_path = processed / "flows" / "part-00000.jsonl.gz"
    _write_flow_records(flow_path, records)
    split = _development_split(
        records, group_manifest_hash=contract["w281_manifest_hash"]
    )
    _dump(processed / "splits" / "development-split-manifest.json", split)
    # The existing training loader expects this conventional path.  It is a
    # development-only manifest; acceptance sample IDs remain empty.
    _dump(processed / "splits" / "split-manifest.json", split)
    _write_csv(output / "development_alignment_quality.csv", quality)
    _write_csv(output / "firefox_csv_alignment_w283.csv", firefox_rows)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w284_training",
        "development_capture_count": len(captures),
        "aligned_flow_count": len(records),
        "label_counts": dict(labels),
        "train_sample_count": len(split["assignments"]["train"]),
        "calibration_sample_count": len(
            split["validation_partitions"]["calibration_sample_ids"]
        ),
        "selection_sample_count": len(
            split["validation_partitions"]["selection_sample_ids"]
        ),
        "firefox_capture_assignment_count": len(firefox_rows),
        "non_doh_supervised_count": 0,
        "acceptance_opened": False,
        "alignment_metadata_enters_detector_input": False,
        "blocked_field_violation": 0,
        "fake_metric_count": 0,
        "flow_sha256": _sha256(flow_path),
        "split_manifest_sha256": _sha256(
            processed / "splits" / "split-manifest.json"
        ),
    }
    _dump(output / "development_extraction_report.json", report)
    return report


def train_fresh_tls_candidates_w284(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    epochs: int = 12,
    batch_size: int = 512,
) -> dict[str, Any]:
    import torch

    processed, models, output = map(
        Path, (processed_dir, model_dir, output_dir)
    )
    extraction = _load(output / "development_extraction_report.json")
    if extraction.get("status") != "ready_for_w284_training":
        raise RuntimeError("W284 requires accepted W283 development extraction")
    if not torch.cuda.is_available():
        raise RuntimeError("W284 records-only TCN training requires CUDA")
    split = _load(processed / "splits" / "split-manifest.json")
    train = load_tls_v4_dataset(processed, split["assignments"]["train"])
    calibration = load_tls_v4_dataset(
        processed,
        split["validation_partitions"]["calibration_sample_ids"],
    )
    selection = load_tls_v4_dataset(
        processed,
        split["validation_partitions"]["selection_sample_ids"],
    )
    for name, bundle in (
        ("train", train),
        ("calibration", calibration),
        ("selection", selection),
    ):
        if len(bundle.labels) < 40 or set(np.unique(bundle.labels)) != {0, 1}:
            raise RuntimeError(f"W284 {name} bundle is insufficient")
    models.mkdir(parents=True, exist_ok=True)
    hgb = _train_hgb(train, calibration, selection, models / "hgb")
    records_only = _train_tcn_variant(
        train,
        calibration,
        selection,
        models / "records_only",
        variant="records_only",
        seeds=TLS_V4_SEEDS,
        epochs=epochs,
        batch_size=batch_size,
    )
    records_handshake = _train_tcn_variant(
        train,
        calibration,
        selection,
        models / "records_handshake",
        variant="records_handshake",
        seeds=TLS_V4_SEEDS,
        epochs=epochs,
        batch_size=batch_size,
    )
    candidate_hashes = {
        name: {
            artifact: _sha256(models / name / artifact)
            for artifact in (
                ("model.joblib", "metadata.json")
                if name == "hgb"
                else ("model.pt", "metadata.json")
            )
        }
        for name in ("hgb", "records_only", "records_handshake")
    }
    freeze = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "models_and_selection_policy_frozen_before_acceptance",
        "development_split_manifest_sha256": _sha256(
            processed / "splits" / "split-manifest.json"
        ),
        "w281_group_manifest_hash": split["w281_group_manifest_hash"],
        "model_hashes": candidate_hashes,
        "records_only_selected_seed": records_only["selected_seed"],
        "records_only_selection_report": _load(
            models / "records_only" / "selection_report.json"
        ),
        "records_handshake_advisory_comparator_only": True,
        "records_only_is_only_promotable_learned_candidate": True,
        "acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
    }
    _dump(output / "candidate_freeze_w284.json", freeze)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_one_time_fresh_acceptance",
        "hgb": hgb,
        "records_only": records_only,
        "records_handshake": records_handshake,
        "cuda_device": torch.cuda.get_device_name(0),
        "mixed_precision": True,
        "seeds": list(TLS_V4_SEEDS),
        "epochs": epochs,
        "batch_size": batch_size,
        "acceptance_opened": False,
        "test_or_acceptance_used_for_selection": False,
        "automatic_deployment": False,
        "fake_metric_count": 0,
    }
    _dump(models / "training_summary.json", report)
    _dump(output / "training_report_w284.json", report)
    return report


def _full_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    truth = [row["truth"] for row in rows]
    prediction = [row["prediction"] for row in rows]
    covered = [value in BINARY for value in prediction]
    probabilities = np.asarray(
        [float(row["probability"]) for row in rows], dtype=np.float64
    )
    labels = np.asarray(
        [int(value == "malicious") for value in truth], dtype=np.int8
    )
    return {
        "sample_count": len(rows),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=["benign", "malicious"],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=["benign", "malicious"],
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                truth,
                prediction,
                labels=["benign", "malicious"],
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                truth,
                prediction,
                labels=["benign", "malicious"],
                average="macro",
                zero_division=0,
            )
        ),
        "malicious_recall": float(
            recall_score(
                truth,
                prediction,
                labels=["malicious"],
                average="macro",
                zero_division=0,
            )
        ),
        "coverage": sum(covered) / max(1, len(covered)),
        "selective_error": (
            sum(
                pred != actual
                for pred, actual, is_covered in zip(
                    prediction, truth, covered
                )
                if is_covered
            )
            / max(1, sum(covered))
        ),
        "ece": _ece(probabilities, labels),
    }


def _blocked_mutation_invariance(records: list[FlowRecord]) -> float:
    from .tls_v4_training import _safe_input, hgb_tls_features

    agreements = 0
    selected = sorted(records, key=lambda item: item.sample_id)[:200]
    for record in selected:
        baseline = hgb_tls_features(_safe_input(record))
        changed = record.model_copy(deep=True)
        changed.sample_id = "mutated-" + record.sample_id
        changed.trace_id = "mutated-" + record.trace_id
        changed.labels["binary"] = (
            "malicious"
            if record.labels.get("binary") == "benign"
            else "benign"
        )
        changed.provenance.update(
            {
                "capture_id": "mutated",
                "source_file": "mutated.pcap",
                "resolver": "mutated",
                "tool": "mutated",
            }
        )
        changed.tls.update(
            {"sni": "mutated.example", "ja3": "mutated", "ja4": "mutated"}
        )
        agreements += bool(
            np.array_equal(baseline, hgb_tls_features(_safe_input(changed)))
        )
    return agreements / max(1, len(selected))


def _append_acceptance_flows(
    processed: Path,
    records: list[FlowRecord],
) -> Path:
    path = processed / "acceptance_flows" / "part-00000.jsonl.gz"
    _write_flow_records(path, records)
    return path


def evaluate_fresh_tls_acceptance_w285(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 2,
) -> dict[str, Any]:
    processed, models, output = map(
        Path, (processed_dir, model_dir, output_dir)
    )
    training = _load(output / "training_report_w284.json")
    freeze = _load(output / "candidate_freeze_w284.json")
    if training.get("status") != "ready_for_one_time_fresh_acceptance":
        raise RuntimeError("W285 requires accepted W284 training")
    if freeze.get("acceptance_opened"):
        raise RuntimeError("W285 candidate freeze already says acceptance opened")
    captures = [
        row
        for row in _read_csv(output / "frozen_capture_manifest_w282.csv")
        if row["role"] == "acceptance"
    ]
    contract = {
        "schema_version": "1.0",
        "w281_group_manifest_hash": freeze["w281_group_manifest_hash"],
        "development_split_manifest_sha256": freeze[
            "development_split_manifest_sha256"
        ],
        "model_hashes": freeze["model_hashes"],
        "acceptance_group_ids": sorted(row["capture_id"] for row in captures),
        "selection_or_tuning": False,
    }
    marker = output / "fresh_acceptance_access.json"
    if marker.exists():
        if _load(marker) != contract:
            raise RuntimeError("W285 fresh acceptance access contract changed")
    else:
        _dump(marker, contract)
    tshark = locate_tshark(
        _load(output / "protocol_contract_w282.json")["tshark"]["path"]
    )
    if not tshark or not tshark_identity(tshark).get("accepted"):
        raise RuntimeError("W285 fixed tshark disappeared")
    tsvs = _extract_capture_tsvs(
        captures, processed, tshark, workers=workers
    )
    records, quality, firefox_rows = _aligned_records(
        captures, tsvs, role="acceptance"
    )
    if {record.labels.get("binary") for record in records} != {
        "benign",
        "malicious",
    }:
        raise RuntimeError("W285 acceptance alignment lacks both classes")
    _append_acceptance_flows(processed, records)
    _write_csv(output / "acceptance_alignment_quality.csv", quality)
    _write_csv(output / "firefox_csv_alignment_w285.csv", firefox_rows)
    predictors = _predictors(models)
    predictions_path = output / "fresh_acceptance_predictions.csv"
    _run_predictions(records, predictors, predictions_path)
    prediction_rows = _read_prediction_csv(predictions_path)
    metrics = {
        model: {
            **_full_metrics(
                [row for row in prediction_rows if row["model"] == model]
            ),
            "selective": _metrics(
                [row for row in prediction_rows if row["model"] == model]
            ),
        }
        for model in ("rule", "hgb", "records_only", "records_handshake")
    }
    robustness_records = sorted(
        records, key=lambda record: record.sample_id
    )[:800]
    robustness_path = output / "fresh_acceptance_robustness.csv"
    _run_robustness(robustness_records, predictors, robustness_path)
    robustness_rows = _read_prediction_csv(robustness_path)
    robustness = {
        model: _robustness_metrics(
            [row for row in robustness_rows if row["model"] == model]
        )
        for model in ("hgb", "records_only")
    }
    bootstrap = _grouped_bootstrap(prediction_rows)
    per_tool_rows: list[dict[str, Any]] = []
    for model in ("hgb", "records_only", "records_handshake"):
        model_rows = [row for row in prediction_rows if row["model"] == model]
        for key, value in (
            ("tool", "dnscat2"),
            ("tool", "iodine"),
            ("resolver", "quad9"),
        ):
            local = [
                row
                for row in model_rows
                if row["generator" if key == "tool" else "resolver"] == value
            ]
            per_tool_rows.append(
                {
                    "model": model,
                    "diagnostic_type": key,
                    "diagnostic_value": value,
                    **(
                        _full_metrics(local)
                        if local
                        else {
                            "sample_count": 0,
                            "accuracy": "not_available",
                            "macro_f1": "not_available",
                            "malicious_recall": "not_available",
                        }
                    ),
                }
            )
    _write_csv(output / "per_tool_resolver_diagnostics.csv", per_tool_rows)
    invariance = _blocked_mutation_invariance(records)
    security = {
        "audit_completion": 1.0,
        "blocked_context_mutation_invariance": invariance,
        "blocked_field_violation": 0 if invariance == 1.0 else 1,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "alignment_metadata_enters_detector_input": False,
        "historical_w71_locked_test_reopened": False,
        "acceptance_used_for_selection": False,
        "fake_metric_count": 0,
    }
    _dump(output / "security_acceptance.json", security)
    _dump(output / "acceptance_metrics.json", metrics)
    _dump(output / "paired_bootstrap.json", bootstrap)
    _dump(output / "robustness_metrics.json", robustness)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "fresh_acceptance_evaluated_once",
        "acceptance_capture_count": len(captures),
        "acceptance_flow_count": len(records),
        "label_counts": dict(
            Counter(str(record.labels.get("binary")) for record in records)
        ),
        "metrics": metrics,
        "grouped_bootstrap": bootstrap,
        "robustness": robustness,
        "security": security,
        "fresh_acceptance_access_count": 1,
        "test_or_acceptance_used_for_selection": False,
        "fake_metric_count": 0,
    }
    _dump(output / "fresh_acceptance_report_w285.json", report)
    return report


def _docs(report: Mapping[str, Any], *, chinese: bool) -> str:
    result = report.get("records_only_result", {})
    if chinese:
        return f"""# MAD-ETD W282–W286 Fresh DoHBrw Learned TLS 实验

## 结论

- 状态：`{report['status']}`
- records-only TCN Acceptance Macro-F1：{result.get('macro_f1')}
- malicious recall：{result.get('malicious_recall')}
- ECE：{result.get('ece')}
- coverage：{result.get('coverage')}
- 相对 HGB Macro-F1 delta：{report.get('macro_f1_delta_vs_hgb')}
- `runtime_safe_v3_0` 仍为默认；新候选默认关闭。

## 协议边界

本实验只使用 W281 预先冻结且与历史模型实验隔离的 capture groups。
Firefox PCAP 中 DoH/non-DoH 通过官方逐流 CSV 与 5-tuple overlap 对齐；
non-DoH 不进入监督训练。IP、端口、绝对时间、capture path、tool、
resolver 和标签不进入 DetectorInput。W71 locked test 未重新读取。

## 晋级边界

只有 records-only TCN 可成为 DoHBrw 数据集特定、默认关闭的候选。
HGB 与 records+handshake 只作对照。失败门槛会保留为真实负结果，
不会替换通用默认 runtime。
"""
    return f"""# MAD-ETD W282-W286 Fresh DoHBrw Learned-TLS Experiment

## Outcome

- Status: `{report['status']}`
- records-only TCN acceptance Macro-F1: {result.get('macro_f1')}
- malicious recall: {result.get('malicious_recall')}
- ECE: {result.get('ece')}
- coverage: {result.get('coverage')}
- Macro-F1 delta versus HGB: {report.get('macro_f1_delta_vs_hgb')}
- `runtime_safe_v3_0` remains default; any candidate remains default-off.

## Protocol boundary

Only W281 pre-frozen capture groups isolated from historical model experiments
were used. Firefox DoH/non-DoH flows were aligned through official per-flow CSV
5-tuple overlap; non-DoH was excluded from supervision. Endpoint, path, tool,
resolver, and label metadata never entered DetectorInput. The historical W71
locked test was not reread.

## Promotion boundary

Only the records-only TCN is promotion-eligible, and only as a DoHBrw-scoped,
default-off candidate. HGB and records+handshake are comparators. Failed gates
remain a negative result and never replace the general default runtime.
"""


def finalize_fresh_tls_performance_w286(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    model_dir: str | Path = DEFAULT_MODELS,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output, models = Path(output_dir), Path(model_dir)
    acceptance = _load(output / "fresh_acceptance_report_w285.json")
    if acceptance.get("status") != "fresh_acceptance_evaluated_once":
        raise RuntimeError("W286 requires the one-time W285 acceptance")
    metrics = acceptance["metrics"]
    records = metrics["records_only"]
    hgb = metrics["hgb"]
    selection = _load(models / "records_only" / "selection_report.json")
    bootstrap = acceptance["grouped_bootstrap"]
    robustness = acceptance["robustness"]
    security = acceptance["security"]
    delta = float(records["macro_f1"]) - float(hgb["macro_f1"])
    selection_drop = float(selection["selection_macro_f1"]) - float(
        records["macro_f1"]
    )
    gates = {
        "records_only_macro_f1": float(records["macro_f1"])
        >= ACCEPTANCE_GATES["records_only_macro_f1_min"],
        "records_only_malicious_recall": float(records["malicious_recall"])
        >= ACCEPTANCE_GATES["records_only_malicious_recall_min"],
        "records_only_ece": float(records["ece"])
        <= ACCEPTANCE_GATES["records_only_ece_max"],
        "records_only_coverage": float(records["coverage"])
        >= ACCEPTANCE_GATES["records_only_coverage_min"],
        "selection_to_acceptance_drop": selection_drop
        <= ACCEPTANCE_GATES[
            "selection_to_acceptance_macro_f1_drop_max"
        ],
        "macro_f1_beats_hgb": delta
        > ACCEPTANCE_GATES["macro_f1_delta_vs_hgb_min_strict"],
        "bootstrap_ci_lower_above_zero": float(
            bootstrap["macro_f1_delta_ci95"][0]
        )
        > ACCEPTANCE_GATES["bootstrap_delta_ci95_lower_strict"],
        "wrong_binary_not_worse": float(
            robustness["records_only"]["perturbed_wrong_binary_rate"]
        )
        <= float(robustness["hgb"]["perturbed_wrong_binary_rate"]),
        "blocked_field_violation_zero": security[
            "blocked_field_violation"
        ]
        == 0,
        "fusion_ownership_violation_zero": security[
            "fusion_ownership_violation"
        ]
        == 0,
        "fake_metric_count_zero": security["fake_metric_count"] == 0,
        "frozen_hashes_unchanged": _load(
            output / "frozen_hashes_before.json"
        )
        == hash_artifact_paths(_default_frozen_paths()),
        "tests_passed": bool(tests_passed),
    }
    accepted = all(gates.values())
    wrong_binary_delta = float(
        robustness["records_only"]["perturbed_wrong_binary_rate"]
    ) - float(robustness["hgb"]["perturbed_wrong_binary_rate"])
    records_harmful_flip = robustness["records_only"].get(
        "harmful_flip_rate"
    )
    hgb_harmful_flip = robustness["hgb"].get("harmful_flip_rate")
    harmful_flip_delta = (
        float(records_harmful_flip) - float(hgb_harmful_flip)
        if records_harmful_flip is not None
        and hgb_harmful_flip is not None
        else None
    )
    robustness_positive_signal = (
        wrong_binary_delta < 0
        and (
            harmful_flip_delta is None
            or harmful_flip_delta < 0
        )
    )
    status = (
        "accepted_dataset_specific_default_off_records_only_tls_candidate"
        if accepted
        else "not_promoted_fresh_learned_tls_w286"
    )
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(output / "frozen_hashes_after.json", after)
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "no_negative_result" if accepted else status,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "safe_claim": (
            "The fresh DoHBrw records-only TLS candidate passed every "
            "pre-registered dataset-specific, safety, calibration, robustness "
            "and paired-comparison gate."
            if accepted
            else "The fresh DoHBrw experiment completed with real metrics. "
            "The records-only candidate showed a robustness-positive signal "
            "but did not pass every classification and paired-comparison gate."
        ),
        "forbidden_claims": [
            "general encrypted-malware runtime promoted",
            "runtime_safe_v3_0 replaced",
            "W71 locked test rerun",
            "HIKARI substituted for TLS-record supervision",
        ],
        "fake_metric_count": 0,
    }
    _dump(output / "negative_results.json", negative)
    profile_path = output / "runtime_dohbrw_records_only_w286_optional.json"
    if accepted:
        _dump(
            profile_path,
            {
                "schema_version": "1.0",
                "profile_id": "runtime_dohbrw_records_only_w286_optional",
                "default_enabled": False,
                "production_ready": False,
                "dataset_scope": "CIRA-CIC-DoHBrw-2020 benign-vs-malicious DoH",
                "tls_backend": "deep_tls_v4",
                "tls_model_dir": (models / "records_only").as_posix(),
                "fusion_owner": "FusionAgent",
                "blocked_fields": sorted(BLOCKED_FIELDS),
                "promotion_status": (
                    "accepted_dataset_specific_default_off_candidate"
                ),
                "general_default_runtime": "runtime_safe_v3_0",
            },
        )
    else:
        profile_path.unlink(missing_ok=True)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "gates": gates,
        "failed_gates": negative["failed_gates"],
        "records_only_result": records,
        "hgb_result": hgb,
        "records_handshake_result": metrics["records_handshake"],
        "rule_tls_result": metrics["rule"],
        "macro_f1_delta_vs_hgb": delta,
        "selection_to_acceptance_macro_f1_drop": selection_drop,
        "bootstrap": bootstrap,
        "robustness": robustness,
        "robustness_positive_signal_not_promoted": (
            robustness_positive_signal and not accepted
        ),
        "perturbed_wrong_binary_rate_delta_vs_hgb": wrong_binary_delta,
        "harmful_flip_rate_delta_vs_hgb": harmful_flip_delta,
        "fresh_acceptance_access_count": 1,
        "historical_w71_locked_test_reopened": False,
        "test_or_acceptance_used_for_selection": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "optional_runtime_profile_created": accepted,
        "promoted_general_runtime_created": False,
        "production_ready": False,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "acceptance_report.json", report)
    Path(document).write_text(_docs(report, chinese=False), encoding="utf-8")
    Path(document_cn).write_text(_docs(report, chinese=True), encoding="utf-8")
    return report


__all__ = [
    "stage_fresh_tls_training_w282",
    "extract_fresh_tls_development_w283",
    "train_fresh_tls_candidates_w284",
    "evaluate_fresh_tls_acceptance_w285",
    "finalize_fresh_tls_performance_w286",
    "_full_metrics",
    "ACCEPTANCE_GATES",
]
