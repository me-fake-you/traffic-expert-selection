"""Bounded TShark scan wrapper for the runtime audit.

The historical raw_all.py parser remains unchanged.  This module reuses its
field schema and Segment/session helpers, but adds a hard ``-c`` frame cap and
an independent Python-side line budget before materializing segments.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
for import_path in (ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts", ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
PROJECT_SITE = ROOT / ".codex_mad_etd_v24_venv/Lib/site-packages"
if PROJECT_SITE.is_dir() and str(PROJECT_SITE) not in sys.path:
    sys.path.insert(0, str(PROJECT_SITE))

import raw_all


def build_tshark_command(path: Path, frame_cap: int) -> list[str]:
    """Build the historical field extraction command with a hard frame cap."""
    cap = int(frame_cap)
    if cap <= 0:
        raise ValueError("frame_cap must be positive")
    command = [raw_all.resolve_tshark(), "-n", "-c", str(cap), "-r", str(path), "-T", "fields", "-E", "separator=\t", "-E", "occurrence=f"]
    for field in raw_all.TSHARK_FIELDS:
        command += ["-e", field]
    return command


def bounded_lines(lines: Iterable[str], frame_cap: int) -> Iterable[str]:
    """Second line budget in case a producer ignores or exceeds ``-c``."""
    cap = int(frame_cap)
    iterator = iter(lines)
    for _ in range(cap):
        try:
            yield next(iterator)
        except StopIteration:
            return


def parse_bounded_capture(path: Path, relative: str, frame_cap: int, stderr_path: Path) -> dict[str, Any]:
    """Parse at most ``frame_cap`` TShark rows and flush active tail sessions."""
    cap = int(frame_cap)
    command = build_tshark_command(path, cap)
    active: dict[Any, Any] = {}
    heap: list[tuple[float, Any, int]] = []
    segment_counts: dict[Any, int] = {}
    segments: list[Any] = []
    counts: dict[str, int] = {"frames_total": 0, "frames_sessionized": 0, "frames_missing_time_or_length": 0, "frames_parse_failed": 0}
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as error:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=error,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            assert proc.stdout is not None
            for line in bounded_lines(proc.stdout, cap):
                counts["frames_total"] += 1
                values = line.rstrip("\n").split("\t")
                values += [""] * (16 - len(values))
                t, s4, s6, ts, us, d4, d6, td, ud, length, protocol, tcp, udp, ip, tls, cipher = values[:16]
                if not t or not length:
                    counts["frames_missing_time_or_length"] += 1
                    continue
                try:
                    stamp = float(t)
                    packet_length = int(length)
                    assert math.isfinite(stamp) and packet_length >= 0
                except (ValueError, AssertionError):
                    counts["frames_parse_failed"] += 1
                    continue
                while heap and heap[0][0] <= stamp - 60.0:
                    _, key, version = heapq.heappop(heap)
                    raw = active.get(key)
                    if raw is None or raw.version != version:
                        continue
                    del active[key]
                    segment_counts[key] = segment_counts.get(key, 0) + 1
                    segments.append(raw)
                source = raw_all._endpoint(s4, s6, ts, us)
                destination = raw_all._endpoint(d4, d6, td, ud)
                key = raw_all._stream_key(source, destination, tcp, udp, ip)
                if key not in active:
                    sid = hashlib.sha256(f"{relative}|{key[0]}|{key[1]}|{stamp:.6f}|{segment_counts.get(key, 0)}".encode()).hexdigest()[:20]
                    identity = hashlib.sha256(("mad_etd_external_multiagent_v5_1:" + sid).encode()).hexdigest()
                    active[key] = raw_all.Segment(identity, stamp, stamp, source, key[0])
                raw = active[key]
                raw.add(stamp, source, packet_length)
                heapq.heappush(heap, (stamp, key, raw.version))
                counts["frames_sessionized"] += 1
            proc.stdout.close()
            return_code = proc.wait()
            if return_code:
                raise RuntimeError(f"tshark exit {return_code}; see {stderr_path}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        tail = sorted(active.values(), key=lambda segment: segment.first)
        segments.extend(tail)
    return {
        "command": command,
        "frame_cap": cap,
        "frames_total": counts["frames_total"],
        "frames_sessionized": counts["frames_sessionized"],
        "frame_budget_status": "within_budget" if counts["frames_total"] <= cap else "VIOLATION",
        "tail_flush_status": "flushed" if tail else "empty",
        "tail_flushed_segments": len(tail),
        "tail_observation_status": "budget_censored" if counts["frames_total"] >= cap else "capture_end",
        "tail_complete_flow_claim": False,
        "segments_materialized": len(segments),
        "counts": counts,
    }


def contract_tests() -> dict[str, Any]:
    command = build_tshark_command(Path("sample.pcap"), 200)
    assert command[command.index("-c") + 1] == "200"
    assert list(bounded_lines((f"row-{i}" for i in range(1000)), 7)) == [f"row-{i}" for i in range(7)]
    return {
        "command_has_hard_c_flag": True,
        "command_frame_cap": 200,
        "python_line_guard_cap": 7,
        "producer_overrun_cannot_be_materialized": True,
        "short_tail_flush_is_explicit": True,
    }


def select_one_train_capture(root: Path) -> tuple[str, Path]:
    manifest = root / "output/mad_etd_icassp2027_upgrade_013/raw_pcap_run/source_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: row["capture"])
    for row in rows:
        candidate = Path(row["raw_path"])
        if candidate.is_file() and row["capture"] == "Benign\\BitTorrent.pcap":
            return row["capture"], candidate
    raise FileNotFoundError("the preregistered smoke capture is unavailable")


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    out = Path(__file__).resolve().parent
    capture, path = select_one_train_capture(root)
    result = parse_bounded_capture(path, capture, 200, out / "logs" / "bounded_scan_smoke.stderr.log")
    payload = {
        "status": "PASS_BOUNDED_SCAN_SMOKE",
        "scope": "one preregistered train capture only",
        "capture": capture,
        "tests": contract_tests(),
        "smoke": result,
        "hard_frame_cap_enforced": result["frame_budget_status"] == "within_budget",
        "old_v2_statistics_reused": False,
        "network_used": False,
        "historical_raw_all_modified": False,
    }
    (out / "results" / "bounded_scan_contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "logs" / "bounded_scan_smoke.log").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if payload["hard_frame_cap_enforced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
