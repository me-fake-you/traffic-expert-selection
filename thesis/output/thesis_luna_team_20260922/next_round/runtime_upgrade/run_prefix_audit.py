"""Small real-PCAP equal-prefix support and feature-cost audit.

This script deliberately stops at a preregistered packet/segment cap.  It
does not load labels or models.  The historical ``raw_all.Segment`` and its
already locked feature functions are reused without editing that source.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
LOGS = OUT / "logs"
PREFIXES = (4, 8, 16, 32, 64)
MAX_PACKETS = 20_000
MAX_SEGMENTS = 500
RUN_V2 = "--v2" in sys.argv[1:]
PER_CAPTURE_SEGMENT_QUOTA = 40 if RUN_V2 else None

SPLIT_PATH = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001/split_manifest.json"
SOURCE_MANIFEST = ROOT / "output/mad_etd_icassp2027_upgrade_013/raw_pcap_run/source_manifest.csv"
RAW_ALL_PATH = ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts/raw_all.py"
FEATURE_PATH = ROOT / "src/mad_etd/ustc_group_heldout_hybrid_w98.py"
PREREG_PATH = OUT / ("preregistration_v2.json" if RUN_V2 else "preregistration.json")
RESULT_SUFFIX = "_v2" if RUN_V2 else ""

for path in (ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# The bundled runtime is the execution interpreter; this project-local site
# directory supplies the historical raw_all dependency set without changing
# system Python or the process environment.
PROJECT_SITE = ROOT / ".codex_mad_etd_v24_venv/Lib/site-packages"
if PROJECT_SITE.is_dir() and str(PROJECT_SITE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SITE))

import raw_all  # noqa: E402


class FailureLedger:
    """Retains every failure row instead of silently shrinking denominators."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, *, request_id: str, prefix_packets: int, status: str, stage: str, detail: str = "") -> None:
        self.rows.append(
            {
                "request_id": request_id,
                "prefix_packets": int(prefix_packets),
                "status": status,
                "stage": stage,
                "detail": detail,
            }
        )

    def counts(self) -> dict[str, int]:
        return dict(Counter(row["status"] for row in self.rows))


class ActualCallCache:
    """Tiny deterministic cache used only for the call-count contract test."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.actual_calls = 0
        self.cache_hits = 0

    def request(self, key: str, loader: Callable[[], Any]) -> Any:
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.actual_calls += 1
        value = loader()
        self.cache[key] = value
        return value


@dataclass(frozen=True)
class PrefixSnapshot:
    """Copied first-P packet state; no reference to a future tail."""

    lengths: tuple[float, ...]
    directions: tuple[float, ...]
    times: tuple[float, ...]

    def stats_record(self) -> dict[str, Any]:
        lengths = np.asarray(self.lengths, dtype=float)
        directions = np.asarray(self.directions, dtype=float)
        total = float(lengths.sum())
        outbound = float(lengths[directions > 0].sum())
        mean = total / len(lengths)
        duration = max(0.0, float(self.times[-1] - self.times[0]))
        return {
            "packet_count": len(lengths),
            "total_bytes": total,
            "outbound_bytes": outbound,
            "inbound_bytes": total - outbound,
            "outbound_ratio": outbound / max(total, 1.0),
            "mean_packet_length": mean,
            "packet_length_variance": max(0.0, float(np.mean(lengths * lengths) - mean * mean)),
            "duration": duration,
        }

    def sequence_record(self) -> dict[str, Any]:
        return {
            "sequence": {
                "packet_lengths": list(self.lengths),
                "directions": list(self.directions),
                "iats": [max(0.0, right - left) for left, right in zip(self.times, self.times[1:])],
            }
        }


def prefix_snapshot(segment: Any, prefix_packets: int) -> PrefixSnapshot | None:
    if segment.n < prefix_packets or len(segment.lengths) < prefix_packets:
        return None
    return PrefixSnapshot(
        lengths=tuple(float(value) for value in segment.lengths[:prefix_packets]),
        directions=tuple(float(value) for value in segment.directions[:prefix_packets]),
        times=tuple(float(value) for value in segment.times[:prefix_packets]),
    )


def timed(stage: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    start = time.perf_counter_ns()
    value = fn()
    end = time.perf_counter_ns()
    elapsed = end - start
    if elapsed < 0:
        raise AssertionError(f"negative stage duration: {stage}")
    return value, {"stage": stage, "start_ns": start, "end_ns": end, "duration_ns": elapsed}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_preregistered_sources() -> tuple[list[dict[str, str]], list[str]]:
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if prereg["status"] != "LOCKED_BEFORE_EXECUTION":
        raise ValueError("preregistration is not locked")
    scope = prereg["execution_scope"]
    if scope["max_total_packets"] != MAX_PACKETS or scope["max_segments"] != MAX_SEGMENTS:
        raise ValueError("runtime caps differ from locked preregistration")
    expected_quota = scope.get("per_capture_segment_quota")
    if expected_quota != PER_CAPTURE_SEGMENT_QUOTA:
        raise ValueError("per-capture quota differs from locked preregistration")
    if tuple(scope["prefixes"]) != PREFIXES:
        raise ValueError("prefix set differs from locked preregistration")
    expected = {
        SOURCE_MANIFEST: prereg["inputs"]["source_manifest_sha256"],
        SPLIT_PATH: prereg["inputs"]["split_manifest_sha256"],
        RAW_ALL_PATH: prereg["inputs"]["parser_source_sha256"],
        FEATURE_PATH: prereg["inputs"]["feature_source_sha256"],
    }
    for path, expected_hash in expected.items():
        if sha(path) != expected_hash:
            raise ValueError(f"locked input changed: {path}")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]
    train_groups = list(split["groups"]["train"])
    if train_groups != prereg["execution_scope"]["selected_groups"]:
        raise ValueError("train-group selection changed after preregistration")
    rows: list[dict[str, str]] = []
    # The label column is deliberately never accessed. Group is only used to
    # select the preregistered train role and is never passed to a feature fn.
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in train_groups:
                continue
            path = Path(row["raw_path"])
            if path.is_file():
                rows.append({"capture": row["capture"], "raw_path": row["raw_path"], "group": row["group"]})
    rows.sort(key=lambda row: row["capture"])
    if not rows:
        raise FileNotFoundError("no preregistered train-group PCAP is available")
    return rows, train_groups


def contract_tests() -> dict[str, Any]:
    ledger = FailureLedger()
    segment = raw_all.Segment("contract", 0.0, 0.0, ("src", "1"), "tcp")
    for index in range(8):
        segment.add(float(index), ("src", "1"), 100 + index)
    snapshot = prefix_snapshot(segment, 4)
    if snapshot is None:
        raise AssertionError("prefix snapshot unexpectedly missing")
    before_stats = raw_all.env["_stats_vector"]({"stats": snapshot.stats_record()})
    before_temporal = raw_all.env["_sequence_vector"](snapshot.sequence_record())
    segment.lengths[4:] = [999999.0] * 4
    segment.directions[4:] = [-1.0] * 4
    segment.times[4:] = [999999.0] * 4
    segment.total = 999999999
    segment.n = 999
    after = prefix_snapshot(segment, 4)
    assert after is not None
    assert np.array_equal(before_stats, raw_all.env["_stats_vector"]({"stats": after.stats_record()}))
    assert np.array_equal(before_temporal, raw_all.env["_sequence_vector"](after.sequence_record()))
    for prefix in PREFIXES:
        if prefix > 8:
            ledger.record(request_id=f"contract-{prefix}", prefix_packets=prefix, status="short_flow", stage="prefix_snapshot")
    timer_a = timed("contract_a", lambda: sum(range(10)))
    timer_b = timed("contract_b", lambda: sum(range(20)))
    assert timer_a[1]["duration_ns"] >= 0 and timer_b[1]["duration_ns"] >= 0
    assert timer_a[1]["end_ns"] <= timer_b[1]["start_ns"]

    calls = ActualCallCache()
    first = calls.request("same", lambda: 0.7)
    second = calls.request("same", lambda: (_ for _ in ()).throw(AssertionError("cache miss")))
    assert first == second == 0.7 and calls.actual_calls == 1 and calls.cache_hits == 1
    ledger.record(request_id="contract-timeout", prefix_packets=4, status="timeout", stage="second_inference", detail="synthetic")
    return {
        "future_tail_unchanged": True,
        "stage_durations_nonnegative": timer_a[1]["duration_ns"] >= 0 and timer_b[1]["duration_ns"] >= 0,
        "stage_spans_independent": True,
        "actual_call_count": calls.actual_calls,
        "cache_hit_count": calls.cache_hits,
        "failure_ledger_rows_retained": len(ledger.rows),
        "failure_ledger_statuses": ledger.counts(),
    }


def run_audit(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], FailureLedger, dict[str, Any]]:
    ledger = FailureLedger()
    support = {str(prefix): Counter() for prefix in PREFIXES}
    costs: list[dict[str, Any]] = []
    total_packets = 0
    total_segments = 0
    parser_counts: Counter[str] = Counter()
    source_records: list[dict[str, Any]] = []
    for source in rows:
        if total_packets >= MAX_PACKETS or total_segments >= MAX_SEGMENTS:
            break
        counts: Counter[str] = Counter()
        child_memory: dict[str, Any] = {}
        stderr_path = LOGS / (f"parser_v2r_{len(source_records):02d}.stderr.log" if RUN_V2 else f"parser_{len(source_records):02d}.stderr.log")
        generator = raw_all.parse_capture(Path(source["raw_path"]), source["capture"], counts, stderr_path, child_memory)
        segments: list[Any] = []
        parse_start = time.perf_counter_ns()
        try:
            for segment in generator:
                if (
                    total_segments >= MAX_SEGMENTS
                    or total_packets + segment.n > MAX_PACKETS
                    or (PER_CAPTURE_SEGMENT_QUOTA is not None and len(segments) >= PER_CAPTURE_SEGMENT_QUOTA)
                ):
                    break
                segments.append(segment)
                total_segments += 1
                total_packets += segment.n
        finally:
            generator.close()
        parse_ns = time.perf_counter_ns() - parse_start
        parser_counts.update(counts)
        source_records.append({"capture": source["capture"], "group": source["group"], "segments": len(segments), "packets": sum(s.n for s in segments), "parse_ns": parse_ns, "parser_counts": dict(counts)})
        for index, segment in enumerate(segments):
            for prefix in PREFIXES:
                request_id = f"{source['capture']}::{index}"
                snapshot = prefix_snapshot(segment, prefix)
                if snapshot is None:
                    status = "short_flow" if segment.n < prefix else "feature_error"
                    support[str(prefix)][status] += 1
                    ledger.record(request_id=request_id, prefix_packets=prefix, status=status, stage="prefix_snapshot")
                    continue
                support[str(prefix)]["supported"] += 1
                try:
                    _stats, stats_trace = timed("stats_feature", lambda: raw_all.env["_stats_vector"]({"stats": snapshot.stats_record()}))
                    _temporal, temporal_trace = timed("temporal_feature", lambda: raw_all.env["_sequence_vector"](snapshot.sequence_record()))
                    assert len(_stats) == 8 and len(_temporal) == 40
                    costs.append({
                        "request_id": request_id,
                        "capture": source["capture"],
                        "prefix_packets": prefix,
                        "observed_packets": segment.n,
                        "stats_feature_ns": stats_trace["duration_ns"],
                        "temporal_feature_ns": temporal_trace["duration_ns"],
                        "status": "ok",
                    })
                except Exception as exc:
                    support[str(prefix)]["feature_error"] += 1
                    ledger.record(request_id=request_id, prefix_packets=prefix, status="feature_error", stage="feature", detail=f"{type(exc).__name__}: {exc}")
        if total_packets >= MAX_PACKETS or total_segments >= MAX_SEGMENTS:
            break
    for key, value in parser_counts.items():
        if key in {"frames_missing_time_or_length", "frames_parse_failed"} and value:
            ledger.record(request_id="parser", prefix_packets=0, status="parse_error", stage="raw_parser", detail=f"{key}={value}")
    summaries = []
    for prefix in PREFIXES:
        counts = support[str(prefix)]
        prefix_costs = [row for row in costs if row["prefix_packets"] == prefix]
        summaries.append({
            "prefix_packets": prefix,
            "supported_count": counts["supported"],
            "short_flow_count": counts["short_flow"],
            "feature_error_count": counts["feature_error"],
            "stats_feature_mean_us": float(np.mean([row["stats_feature_ns"] for row in prefix_costs]) / 1000.0) if prefix_costs else None,
            "temporal_feature_mean_us": float(np.mean([row["temporal_feature_ns"] for row in prefix_costs]) / 1000.0) if prefix_costs else None,
            "stats_feature_p95_us": float(np.quantile([row["stats_feature_ns"] for row in prefix_costs], 0.95) / 1000.0) if prefix_costs else None,
            "temporal_feature_p95_us": float(np.quantile([row["temporal_feature_ns"] for row in prefix_costs], 0.95) / 1000.0) if prefix_costs else None,
        })
    return summaries, ledger, {
        "retained_segment_packets": total_packets,
        "retained_segments": total_segments,
        "scanned_frames": int(parser_counts.get("frames_total", 0)),
        "scanned_sessionized_frames": int(parser_counts.get("frames_sessionized", 0)),
        "scan_budget_status": "NOT_HARD_CAPPED",
        "scan_budget_caveat": "The retained-segment packet cap does not constrain TShark scanning because raw_all.parse_capture is reused without a TShark -c frame limit.",
        "parser_counts": dict(parser_counts),
        "sources": source_records,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    rows, train_groups = load_preregistered_sources()
    contracts = contract_tests()
    summaries, ledger, audit = run_audit(rows)
    payload = {
        "status": "PASS_REAL_PCAP_PREFIX_SUPPORT_AUDIT_V2" if RUN_V2 else "PASS_REAL_PCAP_PREFIX_SUPPORT_AUDIT_SHORT_SLICE",
        "protocol_status": "LOCKED_BEFORE_EXECUTION",
        "source_available": True,
        "tshark": raw_all.tshark_version(),
        "selected_train_groups": train_groups,
        "selected_captures": [row["capture"] for row in rows],
        "caps": {
            "requested_retained_segment_packet_cap": MAX_PACKETS,
            "retained_segment_packet_cap_enforced": True,
            "tshark_scan_frame_cap": None,
            "tshark_scan_frame_cap_enforced": False,
            "max_segments": MAX_SEGMENTS,
            "per_capture_segment_quota": PER_CAPTURE_SEGMENT_QUOTA,
        },
        "audit": audit,
        "prefix_summaries": summaries,
        "contract_tests": contracts,
        "model_prefix_bundle_available": False,
        "model_inference_performed": False,
        "end_to_end_acceleration_claim": False,
        "test_labels_read": False,
        "network_used": False,
    }
    (RESULTS / f"support_audit{RESULT_SUFFIX}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(RESULTS / f"prefix_support_and_cost{RESULT_SUFFIX}.csv", summaries)
    write_csv(RESULTS / f"failure_ledger{RESULT_SUFFIX}.csv", ledger.rows)
    (RESULTS / f"source_records{RESULT_SUFFIX}.json").write_text(json.dumps(audit["sources"], ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / f"run_prefix_audit{RESULT_SUFFIX}.log").write_text(
        json.dumps({
            "preregistration": str(PREREG_PATH),
            "status": payload["status"],
            "tshark": payload["tshark"],
            "audit": audit,
            "prefix_summaries": summaries,
            "contract_tests": contracts,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
