"""Fit the locked v62 P=8/P=16 prefix experts on fold-0 train captures.

The command has two explicit phases. ``--lock-fit`` writes the fit protocol
and hashes this script plus all frozen inputs. ``--run-fit`` refuses to start
unless that lock still matches. It never reads evaluation captures and never
changes historical parser/source files.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
# For a directory path parents[3] is the workspace root; a file path would
# use parents[4]. Keep this explicit so source manifests resolve correctly.
ROOT = HERE.parents[3]
RUNTIME = HERE.parent / "runtime_upgrade"
for path in (ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts", ROOT / "src", RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import bounded_scan  # type: ignore
import raw_all  # type: ignore
from run_prefix_audit import PrefixSnapshot, prefix_snapshot  # type: ignore

SPLIT_PATH = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001/split_manifest.json"
SOURCE_MANIFEST = ROOT / "output/mad_etd_icassp2027_upgrade_013/raw_pcap_run/source_manifest.csv"
RAW_ALL_PATH = ROOT / "output/mad_etd_icassp2027_upgrade_013/scripts/raw_all.py"
FEATURE_PATH = ROOT / "src/mad_etd/ustc_group_heldout_hybrid_w98.py"
FIT_LOCK = HERE / "fit_protocol_locked.json"
OUT = HERE / "fit"
MODELS = HERE / "models"
CAP = 20_000
PREFIXES = (8, 16)
MODEL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "l2_regularization": 2.0,
    "random_state": 20260923,
    "early_stopping": False,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def split_train_groups() -> list[str]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]
    return list(split["groups"]["train"])


def sources() -> list[dict[str, str]]:
    groups = set(split_train_groups())
    result = []
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in groups:
                continue
            path = Path(row["raw_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            result.append({"capture": row["capture"], "raw_path": row["raw_path"],
                           "group": row["group"]})
    result.sort(key=lambda row: row["capture"])
    if len(result) != 13:
        raise RuntimeError(f"expected 13 fold-0 train captures, found {len(result)}")
    return result


def locked_payload() -> dict[str, Any]:
    return {
        "status": "LOCKED_FIT_BEFORE_EXECUTION",
        "version": "v62",
        "outer_fold": 0,
        "role": "fit_only_train_groups",
        "train_groups": split_train_groups(),
        "train_capture_count": 13,
        "prefix_packets": list(PREFIXES),
        "frame_cap_per_capture": CAP,
        "hard_cap": ["tshark -c 20000", "bounded_scan.bounded_lines"],
        "model": "sklearn HistGradientBoostingClassifier",
        "model_params": MODEL_PARAMS,
        "feature_scope": {
            "stats": "PrefixSnapshot.stats_record then frozen _stats_vector; first P only",
            "temporal": "PrefixSnapshot.sequence_record then frozen _sequence_vector; first P only",
            "dimensions": {"stats": 8, "temporal": 40},
            "labels": "group metadata only for y; never feature input",
            "future_access": False,
            "filename_feature": False,
        },
        "input_hashes": {
            "source_manifest": sha(SOURCE_MANIFEST),
            "split_manifest": sha(SPLIT_PATH),
            "raw_all": sha(RAW_ALL_PATH),
            "feature_source": sha(FEATURE_PATH),
            "fit_script": sha(Path(__file__).resolve()),
            "prefix_runtime_script": sha(HERE / "run_prefix_runtime_v62.py"),
        },
        "evaluation_read": False,
        "network_used": False,
        "new_labels": False,
    }


def lock_fit() -> None:
    payload = locked_payload()
    FIT_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def verify_lock() -> dict[str, Any]:
    if not FIT_LOCK.is_file():
        raise RuntimeError("fit lock missing; run --lock-fit first")
    lock = json.loads(FIT_LOCK.read_text(encoding="utf-8"))
    if lock != locked_payload():
        raise RuntimeError("fit lock or frozen input hash changed after lock")
    if lock["status"] != "LOCKED_FIT_BEFORE_EXECUTION":
        raise RuntimeError("unexpected fit lock status")
    return lock


def bounded_segments(path: Path, relative: str, stderr_path: Path) -> tuple[list[Any], dict[str, Any]]:
    """Sessionize at most CAP TShark rows, flushing active tails."""
    command = bounded_scan.build_tshark_command(path, CAP)
    active: dict[Any, Any] = {}
    heap: list[tuple[float, Any, int]] = []
    segment_counts: dict[Any, int] = {}
    segments: list[Any] = []
    counts = Counter(frames_total=0, frames_sessionized=0,
                     frames_missing_time_or_length=0, frames_parse_failed=0)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as error:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error,
                                text=True, encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            assert proc.stdout is not None
            for line in bounded_scan.bounded_lines(proc.stdout, CAP):
                counts["frames_total"] += 1
                values = line.rstrip("\n").split("\t") + [""] * 16
                t, s4, s6, ts, us, d4, d6, td, ud, length, protocol, tcp, udp, ip, tls, cipher = values[:16]
                if not t or not length:
                    counts["frames_missing_time_or_length"] += 1
                    continue
                try:
                    stamp = float(t); packet_length = int(length)
                    if not np.isfinite(stamp) or packet_length < 0:
                        raise ValueError("invalid packet")
                except (ValueError, TypeError):
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
            code = proc.wait()
            if code:
                raise RuntimeError(f"tshark exit {code}; see {stderr_path}")
        finally:
            if proc.poll() is None:
                proc.terminate(); proc.wait()
    tails = sorted(active.values(), key=lambda segment: segment.first)
    segments.extend(tails)
    counts["segments_total"] = len(segments)
    counts["tail_segments"] = len(tails)
    counts["frame_budget_status"] = int(counts["frames_total"] <= CAP)
    counts["tail_observation_status"] = "budget_censored" if counts["frames_total"] >= CAP else "capture_end"
    return segments, dict(counts)


def vectors(segment: Any, p: int) -> tuple[np.ndarray, np.ndarray] | None:
    snapshot = prefix_snapshot(segment, p)
    if snapshot is None:
        return None
    stats = np.asarray(raw_all.env["_stats_vector"]({"stats": snapshot.stats_record()}), dtype=np.float32)
    temporal = np.asarray(raw_all.env["_sequence_vector"](snapshot.sequence_record()), dtype=np.float32)
    if stats.shape != (8,) or temporal.shape != (40,):
        raise RuntimeError(f"unexpected feature dimensions: {stats.shape}, {temporal.shape}")
    return stats, temporal


def run_fit() -> dict[str, Any]:
    lock = verify_lock()
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    stderr_dir = HERE / "logs" / "fit_parser"
    all_xs = {p: [] for p in PREFIXES}; all_xt = {p: [] for p in PREFIXES}; all_y = {p: [] for p in PREFIXES}
    summaries: list[dict[str, Any]] = []
    extraction_start = time.perf_counter_ns()
    for index, source in enumerate(sources()):
        cap_start = time.perf_counter_ns()
        segments, counts = bounded_segments(Path(source["raw_path"]), source["capture"], stderr_dir / f"{index:02d}.stderr.log")
        supported = {str(p): 0 for p in PREFIXES}; short = {str(p): 0 for p in PREFIXES}
        label = int(source["group"].startswith("malware:"))
        for segment in segments:
            for p in PREFIXES:
                pair = vectors(segment, p)
                if pair is None:
                    short[str(p)] += 1
                    continue
                xs, xt = pair; all_xs[p].append(xs); all_xt[p].append(xt); all_y[p].append(label); supported[str(p)] += 1
        summaries.append({"capture": source["capture"], "group": source["group"],
                          "label_used_for_y_only": label, "parser": counts,
                          "supported": supported, "short": short,
                          "wall_ns": time.perf_counter_ns() - cap_start})
        (HERE / "logs" / "fit_progress.log").parent.mkdir(parents=True, exist_ok=True)
        with (HERE / "logs" / "fit_progress.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summaries[-1], ensure_ascii=False) + "\n")
    models_meta: dict[str, Any] = {}
    for p in PREFIXES:
        xs = np.stack(all_xs[p]); xt = np.stack(all_xt[p]); y = np.asarray(all_y[p], dtype=np.uint8)
        for view, x in (("stats", xs), ("temporal", xt)):
            model = HistGradientBoostingClassifier(**MODEL_PARAMS)
            fit_start = time.perf_counter_ns(); model.fit(x, y); fit_ns = time.perf_counter_ns() - fit_start
            path = MODELS / f"fold0_p{p}_{view}.joblib"; joblib.dump(model, path)
            models_meta[f"p{p}_{view}"] = {"path": str(path), "sha256": sha(path),
                                              "rows": int(len(y)), "features": int(x.shape[1]), "fit_ns": fit_ns}
    result = {"status": "PASS_PREFIX_FIT", "lock": lock, "extraction_wall_ns": time.perf_counter_ns() - extraction_start,
              "capture_summaries": summaries, "models": models_meta,
              "evaluation_read": False, "network_used": False}
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results" / "fit_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    if "--lock-fit" in sys.argv[1:]:
        lock_fit(); return 0
    if "--run-fit" in sys.argv[1:]:
        print(json.dumps(run_fit(), ensure_ascii=False, indent=2)); return 0
    raise SystemExit("use --lock-fit or --run-fit")


if __name__ == "__main__":
    raise SystemExit(main())
