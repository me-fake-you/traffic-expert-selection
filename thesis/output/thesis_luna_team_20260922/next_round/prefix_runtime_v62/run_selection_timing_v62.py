"""Formal local selection timing for the locked v62 prefix models.

The protocol is deliberately a replay benchmark: no network observation wait
is available, and ``observation_wait`` is recorded as N/A. Each method/P/round
reopens the five selection PCAPs and enforces the 20,000-frame cap. Source
parsing is reported at capture root scope rather than divided into per-case
latency.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
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
FIT_SCRIPT = HERE / "train_prefix_models_v62.py"
FIT_LOCK = HERE / "fit_protocol_locked.json"
FIT_RESULT = HERE / "results" / "fit_result.json"
MODELS = HERE / "models"
TIMING_LOCK = HERE / "timing_protocol_locked.json"
CAP = 20_000
PREFIXES = (8, 16)
ROUNDS = 3
BATCH = 128
METHODS = ("temporal", "stats", "allavg", "fixed_low_confidence")
FIXED_THRESHOLD = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def train_groups() -> set[str]:
    return set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]["groups"]["train"])


def selection_sources() -> list[dict[str, str]]:
    selection_groups = set(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]["groups"]["selection"])
    rows = []
    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in selection_groups:
                continue
            path = Path(row["raw_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({"capture": row["capture"], "raw_path": row["raw_path"], "group": row["group"]})
    rows.sort(key=lambda row: row["capture"])
    if len(rows) != 5:
        raise RuntimeError(f"expected five fold-0 selection captures, found {len(rows)}")
    return rows


def model_paths() -> dict[str, Path]:
    return {f"p{p}_{view}": MODELS / f"fold0_p{p}_{view}.joblib"
            for p in PREFIXES for view in ("stats", "temporal")}


def locked_payload() -> dict[str, Any]:
    if not FIT_LOCK.is_file() or not FIT_RESULT.is_file():
        raise RuntimeError("fit lock/result missing")
    fit_lock = json.loads(FIT_LOCK.read_text(encoding="utf-8"))
    fit_result = json.loads(FIT_RESULT.read_text(encoding="utf-8"))
    if fit_lock.get("status") != "LOCKED_FIT_BEFORE_EXECUTION":
        raise RuntimeError("fit protocol was not locked")
    expected = {k: value["sha256"] for k, value in fit_result["models"].items()}
    observed = {k: sha(path) for k, path in model_paths().items()}
    if observed != expected:
        raise RuntimeError(f"model sha mismatch: expected={expected}, observed={observed}")
    return {
        "status": "LOCKED_TIMING_BEFORE_EXECUTION",
        "version": "v62",
        "outer_fold": 0,
        "selection_groups": sorted(json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]["groups"]["selection"]),
        "selection_capture_count": 5,
        "prefix_packets": list(PREFIXES),
        "frame_cap_per_capture": CAP,
        "batch_size": BATCH,
        "rounds": ROUNDS,
        "method_order_base": list(METHODS),
        "method_order_rule": "rotate left by round index; same order for both P",
        "fixed_rule": "Stats call iff abs(p_temporal-0.5)<0.1; selected use (pT+pS)/2, otherwise Temporal; any processing failure abstains",
        "features": "compute only required first view; fixed rule constructs Stats only after selection",
        "root_scope": "capture read through in-memory result rows; excludes final JSONL persistence and live observation wait",
        "short_policy": "abstain",
        "budget_tail_policy": "retain; supported if P is available, otherwise budget_censored abstain",
        "observation_wait": "N/A",
        "parallelism": {"processes": 1, "threads": 1, "batch": BATCH},
        "fit_lock_sha256": sha(FIT_LOCK),
        "fit_result_sha256": sha(FIT_RESULT),
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "split_manifest_sha256": sha(SPLIT_PATH),
        "raw_all_sha256": sha(RAW_ALL_PATH),
        "feature_source_sha256": sha(FEATURE_PATH),
        "fit_script_sha256": sha(FIT_SCRIPT),
        "timing_script_sha256": sha(Path(__file__).resolve()),
        "model_sha256": observed,
        "evaluation_read": False,
        "network_used": False,
        "os_isolation": False,
    }


def lock_timing() -> None:
    payload = locked_payload()
    TIMING_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def verify_timing_lock() -> dict[str, Any]:
    if not TIMING_LOCK.is_file():
        raise RuntimeError("timing lock missing; run --lock-timing first")
    lock = json.loads(TIMING_LOCK.read_text(encoding="utf-8"))
    current = locked_payload()
    if lock != current:
        raise RuntimeError("timing lock or one of its frozen inputs changed")
    return lock


def bounded_segments(path: Path, relative: str, stderr_path: Path) -> tuple[list[Any], dict[str, Any]]:
    command = bounded_scan.build_tshark_command(path, CAP)
    active: dict[Any, Any] = {}; heap: list[tuple[float, Any, int]] = []
    segment_counts: dict[Any, int] = {}; segments: list[Any] = []
    counts = Counter(frames_total=0, frames_sessionized=0,
                     frames_missing_time_or_length=0, frames_parse_failed=0)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("w", encoding="utf-8") as error:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error, text=True,
                                encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            assert proc.stdout is not None
            for line in bounded_scan.bounded_lines(proc.stdout, CAP):
                counts["frames_total"] += 1
                v = line.rstrip("\n").split("\t") + [""] * 16
                t, s4, s6, ts, us, d4, d6, td, ud, length, protocol, tcp, udp, ip, tls, cipher = v[:16]
                if not t or not length:
                    counts["frames_missing_time_or_length"] += 1; continue
                try:
                    stamp = float(t); packet_length = int(length)
                    if not np.isfinite(stamp) or packet_length < 0: raise ValueError("invalid packet")
                except (ValueError, TypeError):
                    counts["frames_parse_failed"] += 1; continue
                while heap and heap[0][0] <= stamp - 60.0:
                    _, key, version = heapq.heappop(heap); raw = active.get(key)
                    if raw is None or raw.version != version: continue
                    del active[key]; segment_counts[key] = segment_counts.get(key, 0) + 1; segments.append(raw)
                source = raw_all._endpoint(s4, s6, ts, us); destination = raw_all._endpoint(d4, d6, td, ud)
                key = raw_all._stream_key(source, destination, tcp, udp, ip)
                if key not in active:
                    sid = hashlib.sha256(f"{relative}|{key[0]}|{key[1]}|{stamp:.6f}|{segment_counts.get(key, 0)}".encode()).hexdigest()[:20]
                    active[key] = raw_all.Segment(hashlib.sha256(("mad_etd_external_multiagent_v5_1:" + sid).encode()).hexdigest(), stamp, stamp, source, key[0])
                raw = active[key]; raw.add(stamp, source, packet_length); heapq.heappush(heap, (stamp, key, raw.version)); counts["frames_sessionized"] += 1
            proc.stdout.close(); code = proc.wait()
            if code: raise RuntimeError(f"tshark exit {code}; see {stderr_path}")
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait()
    tails = sorted(active.values(), key=lambda segment: segment.first); segments.extend(tails)
    counts["segments_total"] = len(segments); counts["tail_segments"] = len(tails)
    counts["frame_budget_status"] = "within_budget" if counts["frames_total"] <= CAP else "VIOLATION"
    counts["tail_observation_status"] = "budget_censored" if counts["frames_total"] >= CAP else "capture_end"
    counts["tail_complete_flow_claim"] = False
    return segments, dict(counts)


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values))
    classes = list(model.classes_)
    if 1 not in classes: return np.zeros(len(values), dtype=float)
    return probabilities[:, classes.index(1)]


def stage(ledger: list[dict[str, Any]], *, capture: str, round_id: int, p: int,
          method: str, batch_index: int, name: str, started: int, ended: int,
          scope: str = "batch") -> None:
    duration = ended - started
    if duration < 0: raise RuntimeError(f"negative duration {name}")
    ledger.append({"capture": capture, "round": round_id, "prefix_packets": p,
                   "method": method, "batch_index": batch_index, "stage": name,
                   "scope": scope, "start_ns": started, "end_ns": ended,
                   "duration_ns": duration})


def process_capture(source: dict[str, str], round_id: int, p: int, method: str,
                    models: dict[str, Any], stage_rows: list[dict[str, Any]],
                    result_rows: list[dict[str, Any]], capture_rows: list[dict[str, Any]],
                    stderr_root: Path) -> set[str]:
    capture = source["capture"]; group = source["group"]; label = int(group.startswith("malware:"))
    root_start = time.perf_counter_ns(); parser_start = time.perf_counter_ns()
    try:
        segments, parser = bounded_segments(Path(source["raw_path"]), capture,
                                             stderr_root / f"r{round_id}_p{p}_{method}_{hashlib.sha1(capture.encode()).hexdigest()[:8]}.stderr.log")
    except Exception as exc:
        capture_rows.append({"capture": capture, "round": round_id, "prefix_packets": p, "method": method,
                             "status": "pcap_failure", "stage": "source_parse", "error": str(exc)[:500],
                             "root_wall_ns": time.perf_counter_ns() - root_start})
        return set()
    stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=-1,
          name="source_parse", started=parser_start, ended=time.perf_counter_ns(), scope="capture")
    cases = []
    material_start = time.perf_counter_ns()
    for segment in segments:
        snap = prefix_snapshot(segment, p)
        tail_budget = parser["tail_observation_status"] == "budget_censored" and segment in segments[-parser["tail_segments"]:] if parser["tail_segments"] else False
        if snap is None:
            status = "budget_censored" if tail_budget else "short_flow"
            cases.append({"case_id": segment.identity, "segment": segment, "snapshot": None,
                          "status": status, "tail_budget": bool(tail_budget)})
        else:
            cases.append({"case_id": segment.identity, "segment": segment, "snapshot": snap,
                          "status": "supported", "tail_budget": bool(tail_budget)})
    stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=-1,
          name="prefix_materialization", started=material_start, ended=time.perf_counter_ns(), scope="capture")
    case_ids = {str(case["case_id"]) for case in cases}
    for batch_index, offset in enumerate(range(0, len(cases), BATCH)):
        batch = cases[offset:offset + BATCH]
        supported = [case for case in batch if case["snapshot"] is not None]
        feature_rows: list[dict[str, Any]] = []
        for case in supported:
            feature_rows.append({"case": case, "stats": None, "temporal": None})
        p_t = np.full(len(feature_rows), np.nan); p_s = np.full(len(feature_rows), np.nan)
        if feature_rows:
            def make_features(view: str, rows: list[dict[str, Any]]) -> None:
                batch_start = time.perf_counter_ns()
                for row in rows:
                    snap = row["case"]["snapshot"]
                    assert snap is not None
                    one_start = time.perf_counter_ns()
                    try:
                        value = (raw_all.env["_stats_vector"]({"stats": snap.stats_record()})
                                 if view == "stats" else raw_all.env["_sequence_vector"](snap.sequence_record()))
                        array = np.asarray(value, dtype=np.float32)
                        expected = (8,) if view == "stats" else (40,)
                        if array.shape != expected: raise ValueError(f"feature dimension mismatch: {array.shape}")
                        row[view] = array
                        row["case"][f"{view}_feature_ns"] = time.perf_counter_ns() - one_start
                    except Exception as exc:
                        row["case"]["status"] = "feature_failure"; row["case"]["error"] = str(exc)[:300]
                stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                      name=f"{view}_feature", started=batch_start, ended=time.perf_counter_ns())

            if method in {"temporal", "allavg", "fixed_low_confidence"}:
                make_features("temporal", feature_rows)
            if method in {"stats", "allavg"}:
                make_features("stats", feature_rows)
            valid_t = [row for row in feature_rows if row["temporal"] is not None]
            valid_s = [row for row in feature_rows if row["stats"] is not None]
            if method in {"temporal", "allavg", "fixed_low_confidence"} and valid_t:
                start = time.perf_counter_ns(); x_t = np.stack([row["temporal"] for row in valid_t])
                try:
                    values = positive_probability(models[f"p{p}_temporal"], x_t)
                    for row, value in zip(valid_t, values): row["p_t"] = float(value)
                except Exception as exc:
                    for row in valid_t: row["case"]["status"] = "model_failure"; row["case"]["error"] = str(exc)[:300]
                stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                      name="temporal_inference", started=start, ended=time.perf_counter_ns())
            if method == "fixed_low_confidence":
                coord_start = time.perf_counter_ns()
                need_stats = np.array([abs(float(row.get("p_t", np.nan)) - 0.5) < FIXED_THRESHOLD for row in feature_rows], dtype=bool)
                stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                      name="coordination", started=coord_start, ended=time.perf_counter_ns())
                selected = [row for index, row in enumerate(feature_rows) if need_stats[index] and row["case"]["status"] == "supported"]
                if selected:
                    make_features("stats", selected)
                    selected = [row for row in selected if row["stats"] is not None]
                    start = time.perf_counter_ns()
                    try:
                        values = (positive_probability(models[f"p{p}_stats"], np.stack([row["stats"] for row in selected])) if selected else [])
                        for row, value in zip(selected, values): row["p_s"] = float(value)
                    except Exception as exc:
                        for row in selected: row["case"]["status"] = "model_failure"; row["case"]["error"] = str(exc)[:300]
                    stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                          name="stats_inference", started=start, ended=time.perf_counter_ns())
                fusion_start = time.perf_counter_ns()
                for index, row in enumerate(feature_rows):
                    pt = row.get("p_t", np.nan); ps = row.get("p_s", np.nan)
                    row["p_final"] = ((pt + ps) / 2.0 if need_stats[index] and np.isfinite(ps) else pt)
                stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                      name="fusion", started=fusion_start, ended=time.perf_counter_ns())
            elif method == "allavg":
                valid_s = [row for row in feature_rows if row["stats"] is not None and row["case"]["status"] == "supported"]
                if valid_s:
                    start = time.perf_counter_ns(); x_s = np.stack([row["stats"] for row in valid_s])
                    try:
                        values = positive_probability(models[f"p{p}_stats"], x_s)
                        for row, value in zip(valid_s, values): row["p_s"] = float(value)
                    except Exception as exc:
                        for row in valid_s: row["case"]["status"] = "model_failure"; row["case"]["error"] = str(exc)[:300]
                    stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                          name="stats_inference", started=start, ended=time.perf_counter_ns())
                valid = [row for row in feature_rows if row["temporal"] is not None and row["stats"] is not None and "p_t" in row]
                start = time.perf_counter_ns()
                for row in valid: row["p_final"] = (row.get("p_t", np.nan) + row.get("p_s", np.nan)) / 2.0
                stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                      name="fusion", started=start, ended=time.perf_counter_ns())
            elif method == "temporal":
                for row in feature_rows: row["p_final"] = row.get("p_t", np.nan)
            else:
                if valid_s:
                    start = time.perf_counter_ns(); x_s = np.stack([row["stats"] for row in valid_s])
                    try:
                        values = positive_probability(models[f"p{p}_stats"], x_s)
                        for row, value in zip(valid_s, values): row["p_s"] = float(value)
                    except Exception as exc:
                        for row in valid_s: row["case"]["status"] = "model_failure"; row["case"]["error"] = str(exc)[:300]
                    stage(stage_rows, capture=capture, round_id=round_id, p=p, method=method, batch_index=batch_index,
                          name="stats_inference", started=start, ended=time.perf_counter_ns())
                for row in feature_rows: row["p_final"] = row.get("p_s", np.nan)
        else:
            p_final = np.array([])
        by_case = {id(row["case"]): (row, index) for index, row in enumerate(feature_rows)}
        for case in batch:
            row_info = by_case.get(id(case))
            if row_info is None:
                result_rows.append({"case_id": case["case_id"], "capture": capture, "group": group, "label": label,
                                    "round": round_id, "prefix_packets": p, "method": method,
                                    "status": case["status"], "budget_censored": case["tail_budget"],
                                    "tail_observation_status": parser["tail_observation_status"],
                                    "prediction": None, "first_probability": None, "second_probability": None,
                                    "feature_ns": None, "fallback_used": False})
                continue
            row, index = row_info; final = row.get("p_final", np.nan)
            prediction = int(final >= 0.5) if np.isfinite(final) and case["status"] == "supported" else None
            first = row.get("p_t", np.nan); second = row.get("p_s", np.nan)
            result_rows.append({"case_id": case["case_id"], "capture": capture, "group": group, "label": label,
                                "round": round_id, "prefix_packets": p, "method": method,
                                "status": case["status"], "budget_censored": case["tail_budget"],
                                "tail_observation_status": parser["tail_observation_status"],
                                "prediction": prediction, "first_probability": float(first) if np.isfinite(first) else None,
                                "second_probability": float(second) if second is not None and np.isfinite(second) else None,
                                "feature_ns": sum(case.get(f"{view}_feature_ns", 0) for view in ("stats", "temporal")),
                                "stats_feature_ns": case.get("stats_feature_ns"), "temporal_feature_ns": case.get("temporal_feature_ns"),
                                "fallback_used": False})
    root_end = time.perf_counter_ns()
    capture_rows.append({"capture": capture, "group": group, "round": round_id, "prefix_packets": p, "method": method,
                         "status": "ok", "parser": parser, "root_start_ns": root_start, "root_end_ns": root_end,
                         "root_wall_ns": root_end - root_start, "observation_wait": "N/A",
                         "supported_rows": sum(case["status"] == "supported" for case in cases),
                         "short_flow_rows": sum(case["status"] == "short_flow" for case in cases),
                         "budget_censored_rows": sum(case["status"] == "budget_censored" for case in cases),
                         "case_count": len(cases)})
    return case_ids


def class_f1(rows: list[dict[str, Any]], positive: int) -> float | None:
    supported = [row for row in rows if row["status"] == "supported" and row["prediction"] is not None]
    if not supported: return None
    tp = sum(row["label"] == positive and row["prediction"] == positive for row in supported)
    fp = sum(row["label"] != positive and row["prediction"] == positive for row in supported)
    fn = sum(row["label"] == positive and row["prediction"] != positive for row in supported)
    if tp == 0: return 0.0 if fp or fn else None
    return 2 * tp / (2 * tp + fp + fn)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    malicious = [row for row in rows if row["label"] == 1]
    malicious_recall = sum(row["prediction"] == 1 for row in malicious) / len(malicious) if malicious else None
    supported = [row for row in rows if row["status"] == "supported"]
    f1_malicious = class_f1(rows, 1); f1_benign = class_f1(rows, 0)
    macro = ((f1_malicious + f1_benign) / 2 if f1_malicious is not None and f1_benign is not None else None)
    return {"requests": len(rows), "malicious_requests": len(malicious),
            "supported_requests": len(supported), "supported_malicious_f1": f1_malicious,
            "supported_benign_f1": f1_benign, "supported_macro_f1": macro,
            "malicious_recall_all_requests": malicious_recall,
            "short_flow": sum(row["status"] == "short_flow" for row in rows),
            "budget_censored": sum(row["status"] == "budget_censored" for row in rows),
            "capture_end_tail_rows": sum(row.get("tail_observation_status") == "capture_end" for row in rows),
            "supported_budget_censored": sum(row["status"] == "supported" and row["budget_censored"] for row in rows),
            "prediction_failures": sum(row["prediction"] is None for row in rows),
            "prediction_failures_legacy_meaning": "all missing verdicts, including expected short/censored abstentions",
            "processing_failures": sum(row["status"] not in {"supported", "short_flow", "budget_censored"} for row in rows)}


def validate_stage_ledger(capture_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check stage bounds, non-overlap, and sum<=capture root wall."""
    checks = 0; failures: list[str] = []
    by_capture = defaultdict(list)
    for item in stage_rows:
        by_capture[(item["round"], item["prefix_packets"], item["method"], item["capture"])].append(item)
    for capture in capture_rows:
        if capture.get("status") != "ok":
            continue
        key = (capture["round"], capture["prefix_packets"], capture["method"], capture["capture"])
        entries = sorted(by_capture[key], key=lambda item: item["start_ns"])
        root_start = capture["root_start_ns"]; root_end = capture["root_end_ns"]
        previous_end = root_start; total = 0
        for item in entries:
            checks += 1
            if item["start_ns"] < root_start or item["end_ns"] > root_end:
                failures.append(f"out_of_root:{key}:{item['stage']}")
            if item["start_ns"] < previous_end:
                failures.append(f"overlap:{key}:{item['stage']}")
            if item["duration_ns"] < 0:
                failures.append(f"negative:{key}:{item['stage']}")
            previous_end = max(previous_end, item["end_ns"]); total += item["duration_ns"]
        if total > capture["root_wall_ns"]:
            failures.append(f"sum_exceeds_root:{key}:{total}>{capture['root_wall_ns']}")
    return {"checked_intervals": checks, "failures": failures,
            "pass": not failures}


def _run_timing_locked() -> dict[str, Any]:
    lock = verify_timing_lock()
    os_env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    model_load_start = time.perf_counter_ns()
    models = {key: joblib.load(path) for key, path in model_paths().items()}
    cold_load_end = time.perf_counter_ns()
    warmup = {}
    for key, model in models.items():
        x = np.zeros((1, 8 if "stats" in key else 40), dtype=np.float32); start = time.perf_counter_ns()
        for _ in range(3): model.predict_proba(x)
        warmup[key] = time.perf_counter_ns() - start
    sources = selection_sources(); stage_rows: list[dict[str, Any]] = []; result_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []; case_sets: dict[tuple[int, str], set[str]] = {}; formal_start = time.perf_counter_ns()
    base = list(METHODS)
    for round_id in range(ROUNDS):
        order = base[round_id % len(base):] + base[:round_id % len(base)]
        for method in order:
            for p in PREFIXES:
                for source in sources:
                    current = process_capture(source, round_id, p, method, models, stage_rows, result_rows,
                                              capture_rows, HERE / "logs" / "timing_parser")
                    key = (p, source["capture"])
                    if key in case_sets and current != case_sets[key]:
                        raise RuntimeError(f"case_id set mismatch for round={round_id}, P={p}, method={method}")
                    case_sets.setdefault(key, current)
    grouped_round = defaultdict(list)
    for row in result_rows: grouped_round[(row["method"], row["prefix_packets"], row["round"])].append(row)
    quality_by_round = {f"{method}_p{p}_round{round_id}": metrics(rows)
                        for (method, p, round_id), rows in grouped_round.items()}
    quality = {f"{method}_p{p}": metrics(grouped_round[(method, p, 0)])
               for method in METHODS for p in PREFIXES}
    consistency = {}
    for method in METHODS:
        for p in PREFIXES:
            by_case = defaultdict(list)
            for row in grouped_round[(method, p, 0)]: by_case[row["case_id"]].append(row["prediction"])
            for round_id in (1, 2):
                for row in grouped_round[(method, p, round_id)]: by_case[row["case_id"]].append(row["prediction"])
            consistency[f"{method}_p{p}"] = {"case_count": len(by_case),
                "prediction_disagreements_across_rounds": sum(len(set(values)) > 1 for values in by_case.values())}
    stage_checks = validate_stage_ledger(capture_rows, stage_rows)
    if not stage_checks["pass"]:
        raise RuntimeError(f"Invalid stage ledger: {stage_checks['failures']}")
    capture_by = defaultdict(list)
    for row in capture_rows: capture_by[(row["method"], row["prefix_packets"])].append(row)
    capture_summary = {f"{method}_p{p}": {"captures": len(rows),
                       "root_wall_ns_total": sum(row.get("root_wall_ns", 0) for row in rows),
                       "parser_frames_total": sum(row.get("parser", {}).get("frames_total", 0) for row in rows),
                       "parser_bad_frames": sum(row.get("parser", {}).get("frames_missing_time_or_length", 0) + row.get("parser", {}).get("frames_parse_failed", 0) for row in rows),
                       "pcap_failures": sum(row.get("status") == "pcap_failure" for row in rows),
                       "stage_duration_ns": sum(item["duration_ns"] for item in stage_rows if item["method"] == method and item["prefix_packets"] == p)}
                      for (method, p), rows in capture_by.items()}
    payload = {"status": "PASS_SELECTION_TIMING", "lock": lock, "environment_limits": os_env,
               "cold_model_load_ns": cold_load_end - model_load_start, "warmup3_ns": warmup,
               "formal_run_wall_ns": time.perf_counter_ns() - formal_start,
               "methods": METHODS, "prefixes": PREFIXES, "rounds": ROUNDS, "batch_size": BATCH,
               "case_set_checks": {f"p{p}_{capture}": len(ids) for (p, capture), ids in case_sets.items()},
               "quality": quality, "quality_by_round": quality_by_round, "prediction_consistency": consistency,
               "capture_summary": capture_summary, "stage_checks": stage_checks,
               "threadpool_info": threadpool_info(),
               "observation_wait": "N/A", "evaluation_read": False, "network_used": False,
               "stage_ledger_rows": len(stage_rows), "result_rows": len(result_rows),
               "capture_rows": len(capture_rows)}
    out = HERE / "results"; out.mkdir(parents=True, exist_ok=True)
    (out / "selection_timing_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "selection_result_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in result_rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "selection_stage_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in stage_rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "selection_capture_ledger.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in capture_rows) + "\n", encoding="utf-8")
    return payload


def run_timing() -> dict[str, Any]:
    # This is an actual process-wide native thread limit for the timing scope.
    with threadpool_limits(limits=1):
        return _run_timing_locked()


def main() -> int:
    if "--lock-timing" in sys.argv[1:]: lock_timing(); return 0
    if "--run-timing" in sys.argv[1:]: print(json.dumps(run_timing(), ensure_ascii=False, indent=2)); return 0
    raise SystemExit("use --lock-timing or --run-timing")


if __name__ == "__main__":
    raise SystemExit(main())
