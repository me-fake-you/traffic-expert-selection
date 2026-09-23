"""Locked v62 prefix-runtime runner skeleton.

This file is intentionally incremental: it supplies the hard-cap, prefix
state, timing, and failure-ledger contracts without fitting or selecting a
model.  A real run is refused until ``preregistration_locked.json`` exists
and carries verified P=8/P=16 Stats and Temporal prefix-bundle metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for import_path in (
    ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts",
    ROOT / "src",
    HERE.parent / "runtime_upgrade",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import bounded_scan  # type: ignore
import raw_all  # type: ignore
from run_prefix_audit import PrefixSnapshot, prefix_snapshot  # type: ignore

PREFIXES = (8, 16)
FRAME_CAP = 20_000
OUTCOMES = {
    "supported", "short_flow", "budget_censored", "parser_failure",
    "feature_failure", "model_failure", "timeout", "router_failure",
    "fusion_failure", "abstain", "fallback_temporal",
}


@dataclass(frozen=True)
class PrefixCase:
    """Immutable prefix view; only the first P packet observations are exposed."""

    case_id: str
    capture: str
    group: str
    prefix_packets: int
    lengths: tuple[int, ...]
    directions: tuple[int, ...]
    times: tuple[float, ...]
    status: str
    tail_observation_status: str

    @property
    def supported(self) -> bool:
        return self.status == "supported"


@dataclass
class StageClock:
    """Per-row monotonic stage ledger. All intervals are integer nanoseconds."""

    starts: dict[str, int] = field(default_factory=dict)
    durations: dict[str, int] = field(default_factory=dict)

    def begin(self, name: str) -> None:
        self.starts[name] = time.perf_counter_ns()

    def end(self, name: str) -> int:
        started = self.starts.pop(name)
        elapsed = time.perf_counter_ns() - started
        if elapsed < 0:
            raise RuntimeError(f"negative monotonic duration for {name}")
        self.durations[name] = elapsed
        return elapsed


def single_thread_environment() -> dict[str, str]:
    """Return deterministic BLAS/OpenMP limits without changing the host env."""
    return {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def prefix_case(segment: Any, capture: str, group: str, prefix_packets: int,
                tail_observation_status: str) -> PrefixCase:
    p = int(prefix_packets)
    if p not in PREFIXES:
        raise ValueError(f"unsupported prefix P={p}; expected {PREFIXES}")
    lengths = tuple(int(x) for x in segment.lengths[:p])
    directions = tuple(int(x) for x in segment.directions[:p])
    times = tuple(float(x) for x in segment.times[:p])
    if len(lengths) != len(directions) or len(lengths) != len(times):
        raise ValueError("prefix arrays have inconsistent lengths")
    status = "supported" if len(lengths) >= p else "short_flow"
    if tail_observation_status == "budget_censored" and status != "supported":
        status = "budget_censored"
    case_id = str(segment.identity)
    return PrefixCase(case_id, capture, group, p, lengths, directions, times,
                      status, tail_observation_status)


def temporal_prefix_vector(case: PrefixCase) -> list[float]:
    """Build a Temporal input from the prefix only; reject unsupported rows."""
    if not case.supported:
        raise ValueError(f"Temporal prefix unavailable for {case.status}")
    snapshot = PrefixSnapshot(case.lengths, case.directions, case.times)
    # Reuse the frozen 40-dimensional vector contract, including log transforms
    # and fixed prefix summaries, rather than inventing a new raw layout.
    return [float(x) for x in raw_all.env["_sequence_vector"](snapshot.sequence_record())]


def stats_prefix_vector(case: PrefixCase) -> list[float]:
    """Build a Stats input from first-P packets, never the complete segment."""
    if not case.supported:
        raise ValueError(f"Stats prefix unavailable for {case.status}")
    snapshot = PrefixSnapshot(case.lengths, case.directions, case.times)
    # ``stats_record`` sums byte lengths for outbound/inbound; directions are
    # only the selector. This prevents the prior packet-count/byte-unit bug.
    return [float(x) for x in raw_all.env["_stats_vector"]({"stats": snapshot.stats_record()})]


def timed_failure(case: PrefixCase, stage: str, exc: BaseException,
                  clock: StageClock | None = None) -> dict[str, Any]:
    """Create a retained failure row; the exception text is diagnostic only."""
    return {
        "case_id": case.case_id,
        "capture": case.capture,
        "prefix_packets": case.prefix_packets,
        "status": stage,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:300],
        "stage_ns": dict(clock.durations) if clock else {},
        "prediction": None,
        "fallback_used": False,
    }


def verify_locked_bundle(lock: Mapping[str, Any]) -> None:
    """Reject full-segment/gain routers and incomplete prefix metadata."""
    if lock.get("status") != "LOCKED":
        raise RuntimeError("v62 execution requires status=LOCKED")
    if sorted(lock.get("prefix_packets", [])) != list(PREFIXES):
        raise RuntimeError("lock must register both P=8 and P=16")
    bundle = lock.get("prefix_model_bundle", {})
    for p in map(str, PREFIXES):
        item = bundle.get(p, {})
        if item.get("stats", {}).get("feature_scope") != "first_P_only":
            raise RuntimeError(f"P={p} Stats bundle is not first-P-only")
        if item.get("temporal", {}).get("feature_scope") != "first_P_only":
            raise RuntimeError(f"P={p} Temporal bundle is not first-P-only")
        if not item.get("stats", {}).get("sha256") or not item.get("temporal", {}).get("sha256"):
            raise RuntimeError(f"P={p} model hashes are required")
    if lock.get("frame_cap_per_capture") != FRAME_CAP:
        raise RuntimeError("lock must set frame_cap_per_capture=20000")
    if lock.get("evaluation_read", True):
        raise RuntimeError("evaluation access must remain false")


def contract_check() -> dict[str, Any]:
    """Offline contract checks; this does not open a PCAP or load a model."""
    assert bounded_scan.build_tshark_command(Path("sample.pcap"), FRAME_CAP)[
        bounded_scan.build_tshark_command(Path("sample.pcap"), FRAME_CAP).index("-c") + 1
    ] == str(FRAME_CAP)
    assert list(bounded_scan.bounded_lines((str(i) for i in range(100)), 3)) == ["0", "1", "2"]
    assert single_thread_environment()["OMP_NUM_THREADS"] == "1"
    # Known byte/direction answer: outbound is 10+30 bytes, not two packets.
    known = PrefixSnapshot((10.0, 20.0, 30.0, 40.0), (1.0, -1.0, 1.0, -1.0),
                           (0.0, 1.0, 3.0, 6.0))
    record = known.stats_record()
    assert record["outbound_bytes"] == 40.0
    assert record["inbound_bytes"] == 60.0
    assert len(raw_all.env["_stats_vector"]({"stats": record})) == 8
    assert len(raw_all.env["_sequence_vector"](known.sequence_record())) == 40
    # Mutating all observations after P=8 cannot change either expert vector.
    segment = raw_all.Segment("tail-contract", 0.0, 0.0, ("src", "1"), "tcp")
    for index in range(16):
        source = ("src", "1") if index % 2 == 0 else ("dst", "2")
        segment.add(float(index), source, 100 + index)
    snap8 = prefix_snapshot(segment, 8)
    assert snap8 is not None
    before_s = raw_all.env["_stats_vector"]({"stats": snap8.stats_record()})
    before_t = raw_all.env["_sequence_vector"](snap8.sequence_record())
    segment.lengths[8:] = [999999.0] * 8
    segment.directions[8:] = [-1.0] * 8
    segment.times[8:] = [999999.0] * 8
    segment.total = 10**12
    segment.outbound = 10**12
    segment.n = 10**6
    after8 = prefix_snapshot(segment, 8)
    assert after8 is not None
    assert before_s == raw_all.env["_stats_vector"]({"stats": after8.stats_record()})
    assert before_t == raw_all.env["_sequence_vector"](after8.sequence_record())
    return {
        "status": "PASS_V62_OFFLINE_CONTRACT",
        "frame_cap": FRAME_CAP,
        "prefixes": list(PREFIXES),
        "tshark_c_and_python_guard": True,
        "future_packet_access_in_prefix_vectors": False,
        "known_direction_byte_units": True,
        "post_prefix_tail_mutation_unchanged": True,
        "short_and_budget_rows_retained_by_policy": True,
        "evaluation_read": False,
        "model_training": False,
    }


def load_locked_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            "execution refused: preregistration_locked.json is absent; "
            "create it only after prefix model metadata and hashes are verified"
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    verify_locked_bundle(lock)
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-check", action="store_true",
                        help="run offline checks without a PCAP or model")
    parser.add_argument("--locked", type=Path,
                        default=HERE / "preregistration_locked.json")
    args = parser.parse_args(argv)
    if args.contract_check:
        print(json.dumps(contract_check(), ensure_ascii=False, indent=2))
        return 0
    load_locked_config(args.locked)
    raise RuntimeError(
        "the scoring loop is intentionally not activated in this draft; "
        "add the locked bundle adapter only after the preregistration review"
    )


if __name__ == "__main__":
    raise SystemExit(main())
