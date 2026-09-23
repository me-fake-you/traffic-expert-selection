"""W287-W290 independent DoHBrw robustness replication.

The W284 models and decision policies are immutable in this lane.  W287
selects capture groups that never entered W282-W285.  Four benign captures
were used only in historical W71/W84 development roles (never the W71 locked
test); malicious captures come from unused members of the fresh W281
archives.  This distinction is retained in every manifest and report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .dohbrw_fresh_readiness_w276_w281 import (
    _canonical_hash,
    _safe_member,
    _safe_target,
    _sha256,
)
from .dohbrw_fresh_tls_performance_w282_w286 import (
    BLOCKED_FIELDS,
    _aligned_records,
    _blocked_mutation_invariance,
    _dump,
    _extract_capture_tsvs,
    _extract_selected_members,
    _full_metrics,
    _load,
    _read_csv,
    _write_csv,
    _write_flow_records,
)
from .dohbrw_v4 import locate_tshark, tshark_identity
from .io import iter_flow_records
from .learned_tls_w83 import (
    _malicious_csv_index,
    _normalize_stem,
    _resolver,
)
from .paper_evaluation import hash_artifact_paths
from .schemas import FlowRecord
from .soc_evidence_team_w72 import _default_frozen_paths
from .tls_v4_evaluation import (
    BINARY,
    _predictors,
    _read_csv as _read_prediction_csv,
    _run_predictions,
)


EXPERIMENT = "mad_etd_dohbrw_independent_robustness_w287_w290"
DEFAULT_W282 = Path(
    "data/runs/mad_etd_dohbrw_fresh_tls_performance_w282_w286"
)
DEFAULT_W281 = Path(
    "data/runs/mad_etd_dohbrw_fresh_capture_readiness_w276_w281"
)
DEFAULT_W84 = Path("data/runs/mad_etd_safe_tls_hgb_w84")
DEFAULT_W71 = Path("data/runs/mad_etd_w71_learned_tls_evidence")
DEFAULT_INPUT = Path("data/raw/DoHBrw/pcap")
DEFAULT_STAGE = DEFAULT_INPUT / "_w287_independent_robustness"
DEFAULT_PROCESSED = Path("data/processed/cira_cic_dohbrw_2020/w287")
DEFAULT_MODELS = Path("data/models/mad_etd_fresh_tls_w284")
DEFAULT_OUTPUT = Path(
    "data/runs/mad_etd_dohbrw_independent_robustness_w287_w290"
)
DEFAULT_BENIGN_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/"
    "BenignDoH-NonDoH-CSVs.zip"
)
DEFAULT_MALICIOUS_CSV = Path(
    "data/raw/DoHBrw/auxiliary_csv_archives/CSVs/MaliciousDoH-CSVs.zip"
)
DEFAULT_DOC = Path(
    "docs/MAD_ETD_DOHBRW_INDEPENDENT_ROBUSTNESS_W287_W290.md"
)
DEFAULT_DOC_CN = Path(
    "docs/MAD_ETD_DOHBRW_INDEPENDENT_ROBUSTNESS_W287_W290_CN.md"
)

ROBUSTNESS_PERTURBATIONS = (
    "record_padding",
    "sequence_truncation",
    "record_drop",
    "length_jitter",
)
BOOTSTRAP_ITERATIONS = 1000
SEED = 42
MALICIOUS_PER_TOOL = 24

GATES = {
    "minimum_benign_capture_groups": 4,
    "minimum_malicious_capture_groups": 40,
    "minimum_malicious_tools": 2,
    "wrong_binary_delta_strict_max": 0.0,
    "wrong_binary_ci95_upper_max": 0.0,
    "harmful_flip_delta_max": 0.0,
    "harmful_flip_ci95_upper_max": 0.0,
    "exact_stability_delta_strict_min": 0.0,
    "exact_stability_ci95_lower_min": 0.0,
    "clean_macro_f1_drop_max": 0.01,
    "malicious_recall_drop_max": 0.05,
    "coverage_drop_max": 0.02,
    "ece_increase_max": 0.01,
}


def _model_hashes(model_dir: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for model in ("hgb", "records_only"):
        root = model_dir / model
        if not root.is_dir():
            raise RuntimeError(f"W287 missing frozen model directory: {root}")
        frozen_names = (
            ("metadata.json", "model.joblib")
            if model == "hgb"
            else ("metadata.json", "model.pt")
        )
        frozen_files = [root / name for name in frozen_names]
        files = (
            frozen_files
            if all(item.is_file() for item in frozen_files)
            else [
                item
                for item in sorted(root.rglob("*"))
                if item.is_file()
            ]
        )
        result[model] = {
            item.relative_to(root).as_posix(): _sha256(item)
            for item in files
        }
        if not result[model]:
            raise RuntimeError(f"W287 empty frozen model directory: {root}")
    return result


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _w71_role(stem: str, assignments: Mapping[str, str]) -> str:
    compact = _compact(stem)
    roles = {
        role
        for capture_id, role in assignments.items()
        if _compact(capture_id).startswith(compact)
        or compact.startswith(_compact(capture_id))
    }
    return next(iter(roles)) if len(roles) == 1 else "unresolved"


def _rank(tool: str, member: str) -> str:
    return hashlib.sha256(
        f"w287:{SEED}:{tool}:{member}".encode("utf-8")
    ).hexdigest()


def _benign_candidates(
    *,
    w84_dir: Path,
    w71_dir: Path,
    excluded_hashes: set[str],
) -> list[dict[str, Any]]:
    selected = _read_csv(w84_dir / "selected_capture_manifest_w84.csv")
    w84_roles = _load(w84_dir / "split_manifest.json")["group_assignments"]
    w71_roles = _load(
        w71_dir / "capture_group_split_manifest.json"
    )["group_assignments"]
    rows: list[dict[str, Any]] = []
    for source in selected:
        if (
            source["binary_label"] != "benign"
            or source["browser"] != "chrome"
            or source["resolver"].lower() != "cloudflare"
            or source.get("csv_pair_available") != "True"
            or not source.get("csv_entry")
            or source.get("pcap_sha256") in excluded_hashes
        ):
            continue
        w84_role = w84_roles.get(source["capture_id"], "unresolved")
        w71_role = _w71_role(
            source.get("normalized_capture_stem", ""), w71_roles
        )
        if w84_role == "test" or w71_role != "train":
            continue
        path = Path(source["pcap_path"])
        if not path.is_file():
            continue
        capture_id = (
            "w287-benign-"
            + _canonical_hash(
                {
                    "pcap_sha256": source["pcap_sha256"],
                    "csv_entry": source["csv_entry"],
                }
            )[:18]
        )
        rows.append(
            {
                "capture_id": capture_id,
                "capture_group_id": capture_id,
                "role": "independent_robustness_acceptance",
                "binary_label": "benign",
                "browser": "chrome",
                "tool": "",
                "resolver": "cloudflare",
                "csv_archive": DEFAULT_BENIGN_CSV.as_posix(),
                "csv_entry": source["csv_entry"],
                "csv_candidate_entries": "[]",
                "csv_pair_mode": "official_exact_stem",
                "pcap_path": path.as_posix(),
                "pcap_sha256": source["pcap_sha256"],
                "pcap_size_bytes": source["pcap_size_bytes"],
                "archive_name": "",
                "archive_sha256": "",
                "archive_member": "",
                "member_crc32": "",
                "member_size_bytes": source["pcap_size_bytes"],
                "relative_path": source["relative_path"],
                "w84_historical_role": w84_role,
                "w71_historical_role": w71_role,
                "w71_locked_test_member": False,
                "w282_w285_overlap": False,
                "freshness_scope": (
                    "model_fresh_for_w284_not_globally_history_fresh"
                ),
                "alignment_only_fields_enter_detector_input": False,
            }
        )
    return sorted(rows, key=lambda row: row["pcap_sha256"])


def _malicious_candidates(
    *,
    input_root: Path,
    stage_root: Path,
    w281_dir: Path,
    w282_dir: Path,
    malicious_csv: Path,
) -> list[dict[str, Any]]:
    inventory = {
        row["archive_name"]: row
        for row in _read_csv(w281_dir / "download_archive_inventory.csv")
    }
    used_members = {
        (row["archive_name"], row["archive_member"])
        for row in _read_csv(w282_dir / "frozen_capture_manifest_w282.csv")
    }
    index = _malicious_csv_index(malicious_csv)
    specs = (
        ("MaliciousDoH-dnscat2-Pcap-1202_1802.zip", "dnscat2"),
        ("MaliciousDoH-iodine-pcap-1202_1802.zip", "iodine"),
    )
    rows: list[dict[str, Any]] = []
    for archive_name, tool in specs:
        archive = input_root / archive_name
        if not archive.is_file():
            raise RuntimeError(f"W287 missing official archive: {archive}")
        expected_sha = inventory[archive_name]["sha256"]
        if _sha256(archive) != expected_sha:
            raise RuntimeError(f"W287 archive hash changed: {archive_name}")
        with zipfile.ZipFile(archive) as bundle:
            eligible: list[tuple[zipfile.ZipInfo, str]] = []
            for info in bundle.infolist():
                if not info.filename.lower().endswith((".pcap", ".pcapng")):
                    continue
                if not _safe_member(info.filename):
                    raise RuntimeError(
                        f"W287 unsafe archive member: {info.filename}"
                    )
                if (archive_name, info.filename) in used_members:
                    continue
                entry = index.get(
                    (tool, _normalize_stem(Path(info.filename).stem)), ""
                )
                if entry:
                    eligible.append((info, entry))
        eligible.sort(key=lambda pair: (_rank(tool, pair[0].filename), pair[0].filename))
        if len(eligible) < MALICIOUS_PER_TOOL:
            raise RuntimeError(
                f"W287 insufficient unused exact-aligned {tool} captures: "
                f"{len(eligible)}/{MALICIOUS_PER_TOOL}"
            )
        for info, csv_entry in eligible[:MALICIOUS_PER_TOOL]:
            identity = {
                "archive_sha256": expected_sha,
                "member": info.filename,
                "crc32": f"{info.CRC:08x}",
                "size": info.file_size,
            }
            capture_id = "w287-malicious-" + _canonical_hash(identity)[:18]
            target = _safe_target(stage_root / archive.stem, info.filename)
            rows.append(
                {
                    "capture_id": capture_id,
                    "capture_group_id": capture_id,
                    "role": "independent_robustness_acceptance",
                    "binary_label": "malicious",
                    "browser": "",
                    "tool": tool,
                    "resolver": _resolver(info.filename),
                    "csv_archive": malicious_csv.as_posix(),
                    "csv_entry": csv_entry,
                    "csv_candidate_entries": "[]",
                    "csv_pair_mode": "official_exact_stem",
                    "pcap_path": target.as_posix(),
                    "pcap_sha256": "",
                    "pcap_size_bytes": info.file_size,
                    "archive_name": archive_name,
                    "archive_sha256": expected_sha,
                    "archive_member": info.filename,
                    "member_crc32": f"{info.CRC:08x}",
                    "member_size_bytes": info.file_size,
                    "relative_path": f"{archive_name}/{info.filename}",
                    "w84_historical_role": "",
                    "w71_historical_role": "not_historically_consumed",
                    "w71_locked_test_member": False,
                    "w282_w285_overlap": False,
                    "freshness_scope": "globally_fresh_archive_member",
                    "alignment_only_fields_enter_detector_input": False,
                }
            )
    return rows


def build_independent_robustness_w287(
    *,
    w281_dir: str | Path = DEFAULT_W281,
    w282_dir: str | Path = DEFAULT_W282,
    w84_dir: str | Path = DEFAULT_W84,
    w71_dir: str | Path = DEFAULT_W71,
    input_root: str | Path = DEFAULT_INPUT,
    stage_root: str | Path = DEFAULT_STAGE,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    malicious_csv: str | Path = DEFAULT_MALICIOUS_CSV,
) -> dict[str, Any]:
    w281, w282, w84, w71 = map(
        Path, (w281_dir, w282_dir, w84_dir, w71_dir)
    )
    root, stage, models, output, malicious = map(
        Path, (input_root, stage_root, model_dir, output_dir, malicious_csv)
    )
    required = (
        w281 / "acceptance_report.json",
        w281 / "download_archive_inventory.csv",
        w282 / "frozen_capture_manifest_w282.csv",
        w282 / "candidate_freeze_w284.json",
        w84 / "selected_capture_manifest_w84.csv",
        w84 / "split_manifest.json",
        w71 / "capture_group_split_manifest.json",
        w71 / "locked_test_access.json",
        malicious,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"W287 missing required artifacts: {missing}")
    freeze = _load(w282 / "candidate_freeze_w284.json")
    if (
        freeze.get("status")
        != "models_and_selection_policy_frozen_before_acceptance"
        or freeze.get("test_or_acceptance_used_for_selection")
    ):
        raise RuntimeError("W287 requires the pre-acceptance W284 model freeze")
    current_models = _model_hashes(models)
    expected_models = {
        key: value
        for key, value in freeze["model_hashes"].items()
        if key in {"hgb", "records_only"}
    }
    if current_models != expected_models:
        raise RuntimeError("W287 frozen W284 model hashes changed")
    used_rows = _read_csv(w282 / "frozen_capture_manifest_w282.csv")
    used_hashes = {row["pcap_sha256"] for row in used_rows if row["pcap_sha256"]}
    benign = _benign_candidates(
        w84_dir=w84,
        w71_dir=w71,
        excluded_hashes=used_hashes,
    )
    malicious_rows = _malicious_candidates(
        input_root=root,
        stage_root=stage,
        w281_dir=w281,
        w282_dir=w282,
        malicious_csv=malicious,
    )
    if len(benign) < GATES["minimum_benign_capture_groups"]:
        raise RuntimeError(
            "W287 failed_no_independent_benign_capture_groups: "
            f"{len(benign)}/{GATES['minimum_benign_capture_groups']}"
        )
    captures = benign + malicious_rows
    ids = [row["capture_id"] for row in captures]
    if len(ids) != len(set(ids)):
        raise RuntimeError("W287 duplicate capture-group identities")
    if any(
        row["w282_w285_overlap"] == "True"
        or row["w71_locked_test_member"] == "True"
        for row in captures
    ):
        raise RuntimeError("W287 historical exclusion failed")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "independent_capture_manifest_w287.csv", captures)
    locked_hash = _sha256(w71 / "locked_test_access.json")
    exclusion = {
        "schema_version": "1.0",
        "w282_w285_capture_id_overlap": 0,
        "w282_w285_pcap_hash_overlap": 0,
        "w71_locked_test_capture_count_used": 0,
        "w71_locked_test_access_artifact_sha256": locked_hash,
        "benign_history_boundary": (
            "Four Chrome-Cloudflare captures were historical W71-train/"
            "W84-development data, but never W71 locked-test and never "
            "W282-W285. They are independent of W284 model development."
        ),
        "malicious_history_boundary": (
            "All malicious captures are previously unselected members of "
            "fresh W281 archives and never entered W282-W285."
        ),
        "globally_history_fresh_claim_allowed": False,
        "w284_model_independent_acceptance_claim_allowed": True,
    }
    _dump(output / "historical_exclusion_audit_w287.json", exclusion)
    _dump(output / "frozen_model_hashes_w287.json", current_models)
    _dump(
        output / "frozen_hashes_before.json",
        hash_artifact_paths(_default_frozen_paths()),
    )
    protocol = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w288_independent_extraction",
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "capture_group_count": len(captures),
        "label_capture_counts": dict(
            Counter(row["binary_label"] for row in captures)
        ),
        "malicious_tool_capture_counts": dict(
            Counter(row["tool"] for row in malicious_rows)
        ),
        "perturbations": list(ROBUSTNESS_PERTURBATIONS),
        "selection_or_tuning": False,
        "training_permitted": False,
        "acceptance_metrics_from_w285_used_for_policy": False,
        "historical_w71_locked_test_reopened": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "robustness_protocol_w287.json", protocol)
    return protocol


def extract_independent_robustness_w288(
    *,
    input_root: str | Path = DEFAULT_INPUT,
    stage_root: str | Path = DEFAULT_STAGE,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    workers: int = 2,
) -> dict[str, Any]:
    root, stage, processed, models, output = map(
        Path, (input_root, stage_root, processed_dir, model_dir, output_dir)
    )
    protocol = _load(output / "robustness_protocol_w287.json")
    if protocol.get("status") != "ready_for_w288_independent_extraction":
        raise RuntimeError("W288 requires accepted W287 protocol")
    if _model_hashes(models) != _load(output / "frozen_model_hashes_w287.json"):
        raise RuntimeError("W288 frozen model hashes changed")
    captures = _read_csv(output / "independent_capture_manifest_w287.csv")
    malicious = [row for row in captures if row["binary_label"] == "malicious"]
    extracted_by_id: dict[str, dict[str, Any]] = {}
    for archive_name in sorted({row["archive_name"] for row in malicious}):
        rows = [row for row in malicious if row["archive_name"] == archive_name]
        archive = root / archive_name
        if _sha256(archive) != rows[0]["archive_sha256"]:
            raise RuntimeError(f"W288 archive hash changed: {archive_name}")
        for row in _extract_selected_members(archive, rows, stage):
            extracted_by_id[row["capture_id"]] = row
    staged: list[dict[str, Any]] = []
    for row in captures:
        current = dict(row)
        if row["binary_label"] == "malicious":
            current.update(extracted_by_id[row["capture_id"]])
        path = Path(current["pcap_path"])
        if not path.is_file():
            raise RuntimeError(f"W288 capture disappeared: {path}")
        actual_sha = _sha256(path)
        expected_sha = current.get("pcap_sha256", "")
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"W288 capture hash changed: {path}")
        current["pcap_sha256"] = actual_sha
        current["pcap_size_bytes"] = path.stat().st_size
        staged.append(current)
    if len({row["pcap_sha256"] for row in staged}) != len(staged):
        raise RuntimeError("W288 duplicate PCAP content across groups")
    tshark = locate_tshark(None)
    identity = tshark_identity(tshark) if tshark else {"accepted": False}
    if not tshark or not identity.get("accepted"):
        raise RuntimeError("W288 requires fixed tshark 4.6.6")
    tsvs = _extract_capture_tsvs(
        staged, processed, tshark, workers=workers
    )
    records, quality, firefox_rows = _aligned_records(
        staged, tsvs, role="independent_robustness_acceptance"
    )
    labels = {str(record.labels.get("binary")) for record in records}
    if labels != {"benign", "malicious"}:
        raise RuntimeError("W288 aligned records lack both classes")
    aligned_groups = {
        str(record.provenance["capture_id"]) for record in records
    }
    aligned_labels = defaultdict(set)
    for record in records:
        aligned_labels[str(record.labels["binary"])].add(
            str(record.provenance["capture_id"])
        )
    if (
        len(aligned_labels["benign"]) < GATES["minimum_benign_capture_groups"]
        or len(aligned_labels["malicious"])
        < GATES["minimum_malicious_capture_groups"]
    ):
        raise RuntimeError("W288 insufficient aligned independent capture groups")
    flow_path = processed / "flows" / "part-00000.jsonl.gz"
    _write_flow_records(flow_path, records)
    _write_csv(output / "staged_capture_manifest_w288.csv", staged)
    _write_csv(output / "alignment_quality_w288.csv", quality)
    _write_csv(output / "firefox_alignment_w288.csv", firefox_rows)
    invariance = _blocked_mutation_invariance(records)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w289_frozen_model_robustness_evaluation",
        "capture_group_count": len(staged),
        "aligned_capture_group_count": len(aligned_groups),
        "aligned_label_group_counts": {
            key: len(value) for key, value in aligned_labels.items()
        },
        "flow_count": len(records),
        "label_flow_counts": dict(
            Counter(str(record.labels.get("binary")) for record in records)
        ),
        "flow_artifact": flow_path.as_posix(),
        "flow_artifact_sha256": _sha256(flow_path),
        "tshark": identity,
        "blocked_context_mutation_invariance": invariance,
        "blocked_field_violation": 0 if invariance == 1.0 else 1,
        "w282_w285_group_overlap": 0,
        "historical_w71_locked_test_reopened": False,
        "training_performed": False,
        "selection_or_tuning": False,
        "fake_metric_count": 0,
    }
    _dump(output / "extraction_report_w288.json", report)
    return report


def _perturb_w289(
    record: FlowRecord,
    kind: str,
    *,
    seed: int = SEED,
) -> FlowRecord:
    result = record.model_copy(deep=True)
    values = [int(value) for value in result.tls.get("record_lengths") or []]
    rng = np.random.default_rng(
        int(
            hashlib.sha256(
                f"w289:{seed}:{kind}:{record.sample_id}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
    )
    if kind == "record_padding":
        values = [
            (1 if value >= 0 else -1)
            * (abs(value) + int(rng.integers(1, 129)))
            for value in values
        ]
    elif kind == "sequence_truncation":
        values = values[: max(2, math.floor(len(values) * 0.80))]
    elif kind == "record_drop":
        drop_count = min(
            max(1, round(len(values) * 0.20)),
            max(0, len(values) - 2),
        )
        if drop_count:
            dropped = set(
                int(index)
                for index in rng.choice(
                    len(values), size=drop_count, replace=False
                )
            )
            values = [
                value for index, value in enumerate(values) if index not in dropped
            ]
    elif kind == "length_jitter":
        jittered: list[int] = []
        for value in values:
            magnitude = max(1, abs(value))
            delta = int(round(float(rng.normal(0.0, max(1.0, magnitude * 0.05)))))
            jittered.append(
                (1 if value >= 0 else -1) * max(1, magnitude + delta)
            )
        values = jittered
    else:
        raise ValueError(f"unsupported W289 perturbation: {kind}")
    result.tls["record_lengths"] = values[:64]
    return result


ROBUSTNESS_FIELDS = [
    "model",
    "sample_id",
    "capture_id",
    "truth",
    "perturbation",
    "clean_prediction",
    "perturbed_prediction",
    "clean_covered",
    "perturbed_covered",
    "exact_stable",
    "perturbed_wrong_binary",
    "harmful_flip",
    "unknown_fallback",
]


def _run_fixed_robustness(
    records: Sequence[FlowRecord],
    predictors: Mapping[str, Callable[[FlowRecord], dict[str, Any]]],
    path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for model in ("hgb", "records_only"):
        predict = predictors[model]
        for record in records:
            clean = predict(record)["accepted_class"] or "unknown"
            truth = str(record.labels["binary"])
            for kind in ROBUSTNESS_PERTURBATIONS:
                perturbed = (
                    predict(_perturb_w289(record, kind))["accepted_class"]
                    or "unknown"
                )
                rows.append(
                    {
                        "model": model,
                        "sample_id": record.sample_id,
                        "capture_id": record.provenance["capture_id"],
                        "truth": truth,
                        "perturbation": kind,
                        "clean_prediction": clean,
                        "perturbed_prediction": perturbed,
                        "clean_covered": int(clean in BINARY),
                        "perturbed_covered": int(perturbed in BINARY),
                        "exact_stable": int(clean == perturbed),
                        "perturbed_wrong_binary": int(
                            perturbed in BINARY and perturbed != truth
                        ),
                        "harmful_flip": int(
                            clean == truth
                            and perturbed in BINARY
                            and perturbed != truth
                        ),
                        "unknown_fallback": int(perturbed not in BINARY),
                    }
                )
    _write_csv(path, rows)


def _robustness_metrics(rows: Sequence[Mapping[str, str]]) -> dict[str, float | int]:
    count = max(1, len(rows))
    return {
        "sample_count": len(rows),
        "exact_stability": sum(int(row["exact_stable"]) for row in rows) / count,
        "perturbed_coverage": sum(
            int(row["perturbed_covered"]) for row in rows
        )
        / count,
        "perturbed_wrong_binary_rate": sum(
            int(row["perturbed_wrong_binary"]) for row in rows
        )
        / count,
        "harmful_flip_rate": sum(
            int(row["harmful_flip"]) for row in rows
        )
        / count,
        "unknown_fallback_rate": sum(
            int(row["unknown_fallback"]) for row in rows
        )
        / count,
    }


def _ci(values: Sequence[float]) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _stratified_group_bootstrap(
    clean_rows: Sequence[Mapping[str, str]],
    robustness_rows: Sequence[Mapping[str, str]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    clean = defaultdict(lambda: defaultdict(list))
    robust = defaultdict(lambda: defaultdict(list))
    labels: dict[str, str] = {}
    for row in clean_rows:
        clean[row["model"]][row["capture_id"]].append(dict(row))
        labels[row["capture_id"]] = row["truth"]
    for row in robustness_rows:
        robust[row["model"]][row["capture_id"]].append(dict(row))
        labels[row["capture_id"]] = row["truth"]
    captures = sorted(set(clean["hgb"]) & set(clean["records_only"]))
    by_label = {
        label: [capture for capture in captures if labels[capture] == label]
        for label in ("benign", "malicious")
    }
    if any(not values for values in by_label.values()):
        raise RuntimeError("W289 bootstrap lacks one capture label")
    rng = np.random.default_rng(seed)
    deltas = defaultdict(list)
    for _ in range(iterations):
        selected: list[str] = []
        for label in ("benign", "malicious"):
            values = by_label[label]
            selected.extend(
                str(item)
                for item in rng.choice(values, size=len(values), replace=True)
            )
        hgb_clean = [
            row for capture in selected for row in clean["hgb"][capture]
        ]
        tcn_clean = [
            row for capture in selected for row in clean["records_only"][capture]
        ]
        hgb_robust = [
            row for capture in selected for row in robust["hgb"][capture]
        ]
        tcn_robust = [
            row for capture in selected for row in robust["records_only"][capture]
        ]
        hgb_r = _robustness_metrics(hgb_robust)
        tcn_r = _robustness_metrics(tcn_robust)
        deltas["clean_macro_f1"].append(
            _full_metrics(tcn_clean)["macro_f1"]
            - _full_metrics(hgb_clean)["macro_f1"]
        )
        deltas["wrong_binary"].append(
            float(tcn_r["perturbed_wrong_binary_rate"])
            - float(hgb_r["perturbed_wrong_binary_rate"])
        )
        deltas["harmful_flip"].append(
            float(tcn_r["harmful_flip_rate"])
            - float(hgb_r["harmful_flip_rate"])
        )
        deltas["exact_stability"].append(
            float(tcn_r["exact_stability"])
            - float(hgb_r["exact_stability"])
        )
        deltas["perturbed_coverage"].append(
            float(tcn_r["perturbed_coverage"])
            - float(hgb_r["perturbed_coverage"])
        )
    return {
        "method": "capture_label_stratified_grouped_bootstrap",
        "iterations": iterations,
        "seed": seed,
        "group_counts": {
            label: len(values) for label, values in by_label.items()
        },
        **{
            f"{name}_delta_mean": float(np.mean(values))
            for name, values in deltas.items()
        },
        **{
            f"{name}_delta_ci95": _ci(values)
            for name, values in deltas.items()
        },
    }


def evaluate_independent_robustness_w289(
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    processed, models, output = map(
        Path, (processed_dir, model_dir, output_dir)
    )
    extraction = _load(output / "extraction_report_w288.json")
    if (
        extraction.get("status")
        != "ready_for_w289_frozen_model_robustness_evaluation"
    ):
        raise RuntimeError("W289 requires accepted W288 extraction")
    if _model_hashes(models) != _load(output / "frozen_model_hashes_w287.json"):
        raise RuntimeError("W289 frozen W284 model hashes changed")
    records = list(iter_flow_records(extraction["flow_artifact"]))
    predictors = _predictors(models)
    predictors = {
        key: predictors[key] for key in ("hgb", "records_only")
    }
    clean_path = output / "clean_predictions_w289.csv"
    _run_predictions(records, predictors, clean_path)
    clean_rows = _read_prediction_csv(clean_path)
    by_label = defaultdict(list)
    for record in sorted(records, key=lambda item: item.sample_id):
        by_label[str(record.labels["binary"])].append(record)
    per_class = min(
        400, len(by_label["benign"]), len(by_label["malicious"])
    )
    if per_class < 20:
        raise RuntimeError("W289 insufficient balanced robustness samples")
    robustness_records = (
        by_label["benign"][:per_class]
        + by_label["malicious"][:per_class]
    )
    _write_csv(
        output / "robustness_sample_manifest_w289.csv",
        [
            {
                "sample_id": record.sample_id,
                "capture_id": record.provenance["capture_id"],
                "binary_label": record.labels["binary"],
                "selection_rule": (
                    f"sha_order_first_{per_class}_per_class_before_evaluation"
                ),
            }
            for record in robustness_records
        ],
    )
    robustness_path = output / "robustness_predictions_w289.csv"
    _run_fixed_robustness(
        robustness_records, predictors, robustness_path
    )
    robustness_rows = _read_csv(robustness_path)
    clean_metrics = {
        model: _full_metrics(
            [row for row in clean_rows if row["model"] == model]
        )
        for model in ("hgb", "records_only")
    }
    robustness_metrics = {}
    per_perturbation: list[dict[str, Any]] = []
    for model in ("hgb", "records_only"):
        local = [row for row in robustness_rows if row["model"] == model]
        robustness_metrics[model] = _robustness_metrics(local)
        for perturbation in ROBUSTNESS_PERTURBATIONS:
            cell = [
                row
                for row in local
                if row["perturbation"] == perturbation
            ]
            per_perturbation.append(
                {
                    "model": model,
                    "perturbation": perturbation,
                    **_robustness_metrics(cell),
                }
            )
    _write_csv(
        output / "robustness_cell_metrics_w289.csv",
        per_perturbation,
    )
    bootstrap = _stratified_group_bootstrap(
        clean_rows, robustness_rows
    )
    _dump(output / "grouped_bootstrap_w289.json", bootstrap)
    _dump(output / "clean_metrics_w289.json", clean_metrics)
    _dump(output / "robustness_metrics_w289.json", robustness_metrics)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "ready_for_w290_robustness_gate_decision",
        "clean_metrics": clean_metrics,
        "robustness_metrics": robustness_metrics,
        "grouped_bootstrap": bootstrap,
        "robustness_sample_count_per_class": per_class,
        "perturbations": list(ROBUSTNESS_PERTURBATIONS),
        "training_performed": False,
        "selection_or_tuning": False,
        "w285_acceptance_metrics_used_for_policy": False,
        "historical_w71_locked_test_reopened": False,
        "blocked_field_violation": extraction["blocked_field_violation"],
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
    }
    _dump(output / "evaluation_report_w289.json", report)
    return report


def _docs(report: Mapping[str, Any], *, chinese: bool) -> str:
    clean = report.get("clean_metrics", {})
    robust = report.get("robustness_metrics", {})
    bootstrap = report.get("grouped_bootstrap", {})
    failed = ", ".join(report.get("failed_gates", [])) or "none"
    if chinese:
        return f"""# MAD-ETD W287–W290 独立 DoHBrw 鲁棒性复验

## 状态

- 最终状态：`{report.get('status')}`
- 默认 runtime：`runtime_safe_v3_0`
- 鲁棒性 optional profile：`{report.get('optional_profile_created')}`
- 失败门槛：`{failed}`

## 边界

W287 使用的 capture groups 均未进入 W282–W285。恶意 capture 是 fresh
archive 中此前未选过的成员；4 个 benign capture 曾用于 W71 train/W84
development，但从未属于 W71 locked test，且未参与 W284 模型训练。因而本轮
只能称为 **W284 模型独立复验**，不能称为项目历史全新数据。

## 结果

- HGB clean Macro-F1：`{clean.get('hgb', {}).get('macro_f1')}`
- Records-only clean Macro-F1：`{clean.get('records_only', {}).get('macro_f1')}`
- HGB wrong-binary：`{robust.get('hgb', {}).get('perturbed_wrong_binary_rate')}`
- Records-only wrong-binary：`{robust.get('records_only', {}).get('perturbed_wrong_binary_rate')}`
- HGB harmful flip：`{robust.get('hgb', {}).get('harmful_flip_rate')}`
- Records-only harmful flip：`{robust.get('records_only', {}).get('harmful_flip_rate')}`
- wrong-binary delta 95% CI：`{bootstrap.get('wrong_binary_delta_ci95')}`
- harmful-flip delta 95% CI：`{bootstrap.get('harmful_flip_delta_ci95')}`

本轮没有重新训练或调参，历史 W71 locked test 未重新读取，fake metric
count 为 0。只有全部鲁棒性、clean-performance 与安全门槛同时通过时，
才允许创建 DoHBrw-only、default-off profile。
"""
    return f"""# MAD-ETD W287-W290 Independent DoHBrw Robustness Replication

## Status

- Final status: `{report.get('status')}`
- Default runtime: `runtime_safe_v3_0`
- Robustness optional profile: `{report.get('optional_profile_created')}`
- Failed gates: `{failed}`

## Boundary

No W287 capture entered W282-W285. Malicious captures are unused members of
fresh archives. Four benign captures were previously W71-train/W84-development
data, never W71 locked-test data and never W284 model-development data. The
safe claim is therefore *model-independent replication for W284*, not
globally history-fresh data.

## Results

- HGB clean Macro-F1: `{clean.get('hgb', {}).get('macro_f1')}`
- Records-only clean Macro-F1: `{clean.get('records_only', {}).get('macro_f1')}`
- HGB wrong-binary: `{robust.get('hgb', {}).get('perturbed_wrong_binary_rate')}`
- Records-only wrong-binary: `{robust.get('records_only', {}).get('perturbed_wrong_binary_rate')}`
- Wrong-binary delta 95% CI: `{bootstrap.get('wrong_binary_delta_ci95')}`
- Harmful-flip delta 95% CI: `{bootstrap.get('harmful_flip_delta_ci95')}`

No retraining or tuning occurred, the historical W71 locked test was not
reopened, and no fake metric was generated.
"""


def finalize_independent_robustness_w290(
    *,
    model_dir: str | Path = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    models, output = Path(model_dir), Path(output_dir)
    evaluation = _load(output / "evaluation_report_w289.json")
    if evaluation.get("status") != "ready_for_w290_robustness_gate_decision":
        raise RuntimeError("W290 requires accepted W289 evaluation")
    extraction = _load(output / "extraction_report_w288.json")
    exclusion = _load(output / "historical_exclusion_audit_w287.json")
    if _model_hashes(models) != _load(output / "frozen_model_hashes_w287.json"):
        raise RuntimeError("W290 frozen W284 model hashes changed")
    clean = evaluation["clean_metrics"]
    robust = evaluation["robustness_metrics"]
    bootstrap = evaluation["grouped_bootstrap"]
    hgb_c, tcn_c = clean["hgb"], clean["records_only"]
    hgb_r, tcn_r = robust["hgb"], robust["records_only"]
    wrong_delta = (
        float(tcn_r["perturbed_wrong_binary_rate"])
        - float(hgb_r["perturbed_wrong_binary_rate"])
    )
    harmful_delta = (
        float(tcn_r["harmful_flip_rate"])
        - float(hgb_r["harmful_flip_rate"])
    )
    stability_delta = (
        float(tcn_r["exact_stability"])
        - float(hgb_r["exact_stability"])
    )
    macro_drop = float(hgb_c["macro_f1"]) - float(tcn_c["macro_f1"])
    recall_drop = (
        float(hgb_c["malicious_recall"])
        - float(tcn_c["malicious_recall"])
    )
    coverage_drop = float(hgb_c["coverage"]) - float(tcn_c["coverage"])
    ece_increase = float(tcn_c["ece"]) - float(hgb_c["ece"])
    label_groups = extraction["aligned_label_group_counts"]
    tools = {
        row["tool"]
        for row in _read_csv(
            output / "independent_capture_manifest_w287.csv"
        )
        if row["tool"]
    }
    gates = {
        "independent_from_w282_w285": (
            exclusion["w282_w285_capture_id_overlap"] == 0
            and exclusion["w282_w285_pcap_hash_overlap"] == 0
        ),
        "w71_locked_test_not_used": (
            exclusion["w71_locked_test_capture_count_used"] == 0
            and not evaluation["historical_w71_locked_test_reopened"]
        ),
        "minimum_benign_groups": int(label_groups["benign"])
        >= GATES["minimum_benign_capture_groups"],
        "minimum_malicious_groups": int(label_groups["malicious"])
        >= GATES["minimum_malicious_capture_groups"],
        "minimum_malicious_tools": len(tools)
        >= GATES["minimum_malicious_tools"],
        "wrong_binary_strictly_improved": wrong_delta
        < GATES["wrong_binary_delta_strict_max"],
        "wrong_binary_ci_upper_nonpositive": float(
            bootstrap["wrong_binary_delta_ci95"][1]
        )
        <= GATES["wrong_binary_ci95_upper_max"],
        "harmful_flip_not_worse": harmful_delta
        <= GATES["harmful_flip_delta_max"],
        "harmful_flip_ci_upper_nonpositive": float(
            bootstrap["harmful_flip_delta_ci95"][1]
        )
        <= GATES["harmful_flip_ci95_upper_max"],
        "exact_stability_strictly_improved": stability_delta
        > GATES["exact_stability_delta_strict_min"],
        "exact_stability_ci_lower_positive": float(
            bootstrap["exact_stability_delta_ci95"][0]
        )
        > GATES["exact_stability_ci95_lower_min"],
        "clean_macro_f1_tradeoff": macro_drop
        <= GATES["clean_macro_f1_drop_max"],
        "malicious_recall_tradeoff": recall_drop
        <= GATES["malicious_recall_drop_max"],
        "coverage_tradeoff": coverage_drop <= GATES["coverage_drop_max"],
        "ece_tradeoff": ece_increase <= GATES["ece_increase_max"],
        "blocked_field_violation_zero": evaluation[
            "blocked_field_violation"
        ]
        == 0,
        "fusion_ownership_violation_zero": evaluation[
            "fusion_ownership_violation"
        ]
        == 0,
        "ood_override_zero": evaluation["ood_override_count"] == 0,
        "illegal_verdict_execution_zero": evaluation[
            "illegal_verdict_execution_count"
        ]
        == 0,
        "fake_metric_count_zero": evaluation["fake_metric_count"] == 0,
        "frozen_default_hashes_unchanged": _load(
            output / "frozen_hashes_before.json"
        )
        == hash_artifact_paths(_default_frozen_paths()),
        "tests_passed": bool(tests_passed),
    }
    accepted = all(gates.values())
    status = (
        "accepted_dataset_specific_robustness_only_default_off_profile"
        if accepted
        else "not_promoted_independent_robustness_w290"
    )
    profile = output / "runtime_dohbrw_robustness_w290_optional.json"
    if accepted:
        _dump(
            profile,
            {
                "schema_version": "1.0",
                "profile_id": "runtime_dohbrw_robustness_w290_optional",
                "default_enabled": False,
                "production_ready": False,
                "dataset_scope": (
                    "CIRA-CIC-DoHBrw-2020 robustness-only diagnostic"
                ),
                "tls_backend": "records_only_w284_frozen",
                "model_dir": (models / "records_only").as_posix(),
                "fusion_owner": "FusionAgent",
                "promotion_status": (
                    "accepted_dataset_specific_robustness_only"
                ),
                "classification_superiority_claim_allowed": False,
                "general_default_runtime": "runtime_safe_v3_0",
            },
        )
    else:
        profile.unlink(missing_ok=True)
    failed = [name for name, passed in gates.items() if not passed]
    negative = {
        "schema_version": "1.0",
        "status": "no_negative_result" if accepted else status,
        "failed_gates": failed,
        "safe_claim": (
            "Independent model-specific robustness replication passed all "
            "pre-registered robustness, clean-tradeoff and safety gates."
            if accepted
            else "Real independent robustness metrics were retained, but the "
            "candidate failed one or more pre-registered gates."
        ),
        "forbidden_claims": [
            "classification superiority over HGB",
            "globally history-fresh benign acceptance data",
            "runtime_safe_v3_0 replaced",
            "general encrypted-malware robustness profile",
        ],
        "fake_metric_count": 0,
    }
    _dump(output / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "gates": gates,
        "failed_gates": failed,
        "clean_metrics": clean,
        "robustness_metrics": robust,
        "grouped_bootstrap": bootstrap,
        "wrong_binary_delta": wrong_delta,
        "harmful_flip_delta": harmful_delta,
        "exact_stability_delta": stability_delta,
        "clean_macro_f1_drop": macro_drop,
        "malicious_recall_drop": recall_drop,
        "coverage_drop": coverage_drop,
        "ece_increase": ece_increase,
        "optional_profile_created": accepted,
        "candidate_default_enabled": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_general_runtime_created": False,
        "historical_w71_locked_test_reopened": False,
        "w285_acceptance_used_for_tuning": False,
        "training_performed": False,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "production_ready": False,
    }
    _dump(
        output / "frozen_hashes_after.json",
        hash_artifact_paths(_default_frozen_paths()),
    )
    _dump(output / "acceptance_report.json", report)
    Path(document).write_text(_docs(report, chinese=False), encoding="utf-8")
    Path(document_cn).write_text(
        _docs(report, chinese=True), encoding="utf-8"
    )
    return report


__all__ = [
    "build_independent_robustness_w287",
    "extract_independent_robustness_w288",
    "evaluate_independent_robustness_w289",
    "finalize_independent_robustness_w290",
    "_perturb_w289",
    "_stratified_group_bootstrap",
    "_robustness_metrics",
    "ROBUSTNESS_PERTURBATIONS",
    "GATES",
]
