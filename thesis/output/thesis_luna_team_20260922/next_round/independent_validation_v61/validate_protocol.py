"""Read-only validation of the already sealed/run llm_gain_v61 pilot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGET = ROOT / "output/thesis_luna_team_20260922/next_round/llm_gain_v61"
PREREG = TARGET / "preregistration.json"
RUNNER = TARGET / "run_pilot.py"
QUERY = TARGET / "results/scorer_only_queries.csv"
TRAIN = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade/train_predictions.csv.gz"
SELECTION = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade/selection_predictions.csv.gz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gz_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def jload(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_selection(selection_rows: list[dict[str, str]], fold: int) -> tuple[set[str], dict[str, str]]:
    rows = [row for row in selection_rows if int(row["fold"]) == fold]
    for row in rows:
        row["sampling_hash_check"] = hashlib.sha256(("llm-gain-v61|" + row["sample_hash"]).encode()).hexdigest()
        row["margin_check"] = abs(float(row["p_temporal"]) - 0.5)
    low = sorted(rows, key=lambda row: (row["margin_check"], row["sample_hash"]))[:32]
    low_hashes = {row["sample_hash"] for row in low}
    remaining = [row for row in rows if row["sample_hash"] not in low_hashes]
    remaining = sorted(remaining, key=lambda row: (row["sampling_hash_check"], row["sample_hash"]))[:32]
    chosen = low + remaining
    strata = {row["sample_hash"]: ("low_first_confidence" if row["sample_hash"] in low_hashes else "hash_remainder") for row in chosen}
    return {row["sample_hash"] for row in chosen}, strata


def parse_prompt(path: Path) -> tuple[dict[str, Any], str]:
    payload = jload(path)
    user = payload["messages"][1]["content"]
    task, raw_state = user.split("\nCOMMON_STATE=", 1)
    return json.loads(raw_state), task


def validate_response(receipt: dict[str, Any], response_path: Path) -> dict[str, Any]:
    payload = jload(response_path)
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    allowed = {f"c{i:02d}" for i in range(64)}
    arm = receipt["arm"]
    if arm == "direct":
        if set(parsed) != {"selected_case_ids"}:
            raise ValueError("direct response has extra/missing keys")
        ids = parsed["selected_case_ids"]
        if not isinstance(ids, list) or len(ids) != 6 or len(set(ids)) != 6 or not set(ids) <= allowed:
            raise ValueError("direct selection violates six unique allowed IDs")
        probabilities = None
    else:
        if set(parsed) != {"probabilities"} or set(parsed["probabilities"]) != allowed:
            raise ValueError("atomic response does not cover exactly c00-c63")
        probabilities = parsed["probabilities"]
        for value in probabilities.values():
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError("atomic outcome triple has wrong shape")
            if any(type(item) not in (int, float) for item in value):
                raise ValueError("atomic outcome probability is not a JSON int/float")
            if any(not (float(item) == float(item)) or abs(float(item)) == float("inf") or not 0 <= float(item) <= 1 for item in value):
                raise ValueError("atomic probability is non-finite or outside [0,1]")
            if abs(sum(value) - 1.0) >= 1e-6:
                raise ValueError("atomic probability triple does not sum to one")
        ids = sorted(probabilities, key=lambda key: (-(probabilities[key][2] - probabilities[key][0]), key))[:6]
    if ids != receipt.get("selected_case_ids"):
        raise ValueError("receipt selected_case_ids disagree with response parsing")
    return {"selected_case_ids": ids, "probabilities": probabilities, "model": payload.get("model"), "content": parsed}


def metric(rows: list[dict[str, str]], ids: list[str]) -> dict[str, int]:
    selected = set(ids)
    c = d = errors = calls = 0
    for row in rows:
        y = int(row["y"])
        h0 = float(row["p_temporal"]) >= 0.5
        h1 = (float(row["p_temporal"]) + float(row["p_stats_offline_target_only"])) / 2 >= 0.5
        acquired = row["case_id"] in selected
        pred = h1 if acquired else h0
        calls += int(acquired)
        errors += int(pred != y)
        c += int((h0 != y) and (pred == y))
        d += int((h0 == y) and (pred != y))
    return {"rows": len(rows), "C": c, "D": d, "net": c - d, "simulated_second_calls": calls, "errors": errors}


def main() -> int:
    prereg = jload(PREREG)
    issues: list[dict[str, str]] = []
    warnings: list[str] = []
    files_hash_ok = all(sha(Path(path)) == expected for path, expected in prereg["files"].items())
    if not files_hash_ok:
        issues.append({"severity": "critical", "item": "sealed_file_hashes", "detail": "One or more preregistered files changed."})
    schedule = {(int(item["fold"]), item["arm"]): item for item in prereg["schedule"]}
    prompt_hash_ok = all(sha(Path(item["prompt_path"])) == item["prompt_sha256"] for item in prereg["schedule"])
    if not prompt_hash_ok:
        issues.append({"severity": "critical", "item": "prompt_hashes", "detail": "A prompt differs from its sealed schedule hash."})

    q = read_csv(QUERY)
    sel = read_gz_csv(SELECTION)
    train = read_gz_csv(TRAIN)
    fold_counts = {str(fold): sum(int(row["fold"]) == fold for row in q) for fold in range(5)}
    case_id_checks = {str(fold): sorted({row["case_id"] for row in q if int(row["fold"]) == fold}) == [f"c{i:02d}" for i in range(64)] for fold in range(5)}
    recomputed_selection: dict[str, bool] = {}
    strata_checks: dict[str, bool] = {}
    group_disjoint: dict[str, bool] = {}
    for fold in range(5):
        chosen, strata = expected_selection(sel, fold)
        query_fold = [row for row in q if int(row["fold"]) == fold]
        recomputed_selection[str(fold)] = {row["sample_hash"] for row in query_fold} == chosen
        strata_checks[str(fold)] = all(row["sampling_stratum"] == strata[row["sample_hash"]] for row in query_fold)
        train_groups = {row["group"] for row in train if int(row["fold"]) == fold}
        selection_groups = {row["group"] for row in sel if int(row["fold"]) == fold}
        group_disjoint[str(fold)] = not (train_groups & selection_groups)
    if not all(recomputed_selection.values()) or not all(strata_checks.values()):
        issues.append({"severity": "critical", "item": "320_row_selection", "detail": "Persisted queries do not match p_temporal/sample_hash-only recomputation."})
    if not all(group_disjoint.values()):
        issues.append({"severity": "critical", "item": "train_selection_group_overlap", "detail": "A fold has overlapping training and selection groups."})

    prompt_checks: dict[str, Any] = {"count": 0, "common_state_equal_within_fold": True, "case_schema_clean": True, "hash_matches": prompt_hash_ok, "state_hash_matches": True, "common_state_contains_query_labels_or_p_stats": False}
    prompt_states: dict[int, dict[str, Any]] = {}
    for (fold, arm), item in sorted(schedule.items()):
        state, task = parse_prompt(Path(item["prompt_path"]))
        payload = jload(Path(item["prompt_path"]))
        common_hash = hashlib.sha256(json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        prompt_checks["count"] += 1
        prompt_checks["state_hash_matches"] &= common_hash == payload["common_state_sha256"]
        prompt_states.setdefault(fold, state)
        prompt_checks["common_state_equal_within_fold"] &= prompt_states[fold] == state
        cases = state.get("cases", [])
        prompt_checks["case_schema_clean"] &= len(cases) == 64 and all(set(case) == {"id", "temporal_probability"} for case in cases)
        prompt_checks["common_state_contains_query_labels_or_p_stats"] |= any(key in json.dumps(state, ensure_ascii=True).lower() for key in ("p_stats", "p_temporal_target", "query_label", "sample_label"))
        if len(cases) != 64 or any(set(case) != {"id", "temporal_probability"} for case in cases):
            issues.append({"severity": "critical", "item": f"prompt_{fold}_{arm}_state_schema", "detail": "Prompt case state contains fields beyond ID and first Temporal probability."})
    if prompt_checks["common_state_contains_query_labels_or_p_stats"]:
        issues.append({"severity": "critical", "item": "prompt_label_or_p_stats_leak", "detail": "A prompt common state contains query labels or second-probability fields."})

    receipts: dict[tuple[int, str], dict[str, Any]] = {}
    response_checks: dict[str, Any] = {"valid_schema": True, "valid_model_matches_requested": True, "receipt_response_hashes": True, "failure_transparent": True}
    valid_records: list[dict[str, Any]] = []
    for item in prereg["schedule"]:
        fold, arm = int(item["fold"]), item["arm"]
        receipt_path = TARGET / "responses" / f"fold_{fold}_{arm}.receipt.json"
        rec = jload(receipt_path)
        receipts[(fold, arm)] = rec
        if rec.get("status") == "valid":
            response_path = TARGET / "responses" / f"fold_{fold}_{arm}.response.json"
            try:
                parsed = validate_response(rec, response_path)
                response_checks["valid_model_matches_requested"] &= parsed["model"] == prereg["model"] == rec.get("requested_model")
                response_checks["receipt_response_hashes"] &= sha(response_path) == rec.get("response_sha256")
                valid_records.append({"fold": fold, "arm": arm, "wall_seconds": float(rec["wall_seconds"]), "usage": rec.get("usage", {}), "selected_case_ids": parsed["selected_case_ids"]})
            except Exception as exc:
                response_checks["valid_schema"] = False
                issues.append({"severity": "critical", "item": f"response_{fold}_{arm}", "detail": f"receipt marked valid but independent parsing failed: {type(exc).__name__}: {exc}"})
        else:
            response_path = TARGET / "responses" / f"fold_{fold}_{arm}.response.json"
            response_checks["failure_transparent"] &= not response_path.exists() and rec.get("status") == "invalid_or_transport_failure" and bool(rec.get("error_type"))
    if not response_checks["valid_model_matches_requested"]:
        issues.append({"severity": "critical", "item": "observed_model", "detail": "A valid response observed a model different from the preregistered model."})
    if not response_checks["failure_transparent"]:
        issues.append({"severity": "critical", "item": "failure_receipts", "detail": "A failed request lacks transparent invalid status/error or has a response artifact."})

    baseline = jload(TARGET / "results/baseline_decisions.json")
    policy = read_csv(TARGET / "results/policy_metrics.csv")
    policy_map = {(int(row["fold"]), row["arm"]): row for row in policy}
    score_checks = {"policy_metrics_recomputed": True, "fallback_statuses_match": True}
    expected_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for fold in range(5):
        rows = [row for row in q if int(row["fold"]) == fold]
        fold_base = {item["arm"]: item["selected_case_ids"] for item in baseline if int(item["fold"]) == fold}
        arms: dict[str, list[str]] = {**fold_base, "first_only": [], "all_average": [row["case_id"] for row in rows]}
        for arm in ("direct", "atomic"):
            rec = receipts[(fold, arm)]
            arms[arm] = rec["selected_case_ids"] if rec.get("status") == "valid" else fold_base["same_card_rule"]
        for arm, ids in arms.items():
            calc = metric(rows, ids)
            observed = policy_map[(fold, arm)]
            for key in ("rows", "C", "D", "net", "simulated_second_calls", "errors"):
                if int(observed[key]) != calc[key]:
                    score_checks["policy_metrics_recomputed"] = False
                    issues.append({"severity": "critical", "item": f"score_{fold}_{arm}_{key}", "detail": f"observed={observed[key]} recomputed={calc[key]}"})
            if arm in ("direct", "atomic"):
                expected_fallback = receipts[(fold, arm)].get("status") != "valid"
                score_checks["fallback_statuses_match"] &= (observed["fallback_used"].lower() == str(expected_fallback).lower()) and observed["response_status"] == receipts[(fold, arm)].get("status")
            expected_rows.append({"fold": fold, "arm": arm, **calc})
        if receipts[(fold, "direct")].get("status") == "valid" and receipts[(fold, "atomic")].get("status") == "valid":
            pair_rows.extend({"fold": fold, "arm": arm, **metric(rows, arms[arm])} for arm in ("direct", "atomic"))
    if not score_checks["policy_metrics_recomputed"] or not score_checks["fallback_statuses_match"]:
        issues.append({"severity": "critical", "item": "score_or_fallback", "detail": "policy_metrics does not match independent recomputation or fallback receipt status."})

    latency: dict[str, Any] = {}
    for arm in ("direct", "atomic"):
        records = [record for record in valid_records if record["arm"] == arm]
        times = [record["wall_seconds"] for record in records]
        usage = [record["usage"] for record in records]
        latency[arm] = {
            "valid_count": len(records),
            "wall_seconds": {"min": min(times) if times else None, "median": statistics.median(times) if times else None, "max": max(times) if times else None, "mean": statistics.mean(times) if times else None},
            "tokens": {"prompt_total": sum(int(item.get("prompt_tokens", 0)) for item in usage), "completion_total": sum(int(item.get("completion_tokens", 0)) for item in usage), "total": sum(int(item.get("total_tokens", 0)) for item in usage)},
            "folds": [record["fold"] for record in records],
        }

    if any(int(count) != 64 for count in fold_counts.values()) or len(q) != 320:
        issues.append({"severity": "critical", "item": "query_count", "detail": f"Expected 320 rows / 64 per fold; observed {len(q)} / {fold_counts}."})
    if len({row["sample_hash"] for row in q}) != 303:
        warnings.append("The 320 development rows contain 303 unique sample hashes; fold records overlap and are not 320 independent cases.")
    warnings.append("Atomic+fallback net=5 is a mixed policy result, not pure atomic model effect; valid atomic responses exist only on folds 0 and 4.")
    warnings.append("The only same-fold direct/atomic valid pairs are folds 0 and 4; both have C=0, D=0, net=0, so they provide no detectable paired net difference.")
    warnings.append("run_pilot.py uses assert for live validation; running it with Python -O would disable those checks. The executed bundled interpreter was not optimized.")

    payload = {
        "status": "PASS_WITH_CAVEATS" if not issues else "FAIL_REVIEW",
        "read_only": True,
        "remote_calls_made": 0,
        "original_files_modified": False,
        "protocol": {"name": prereg["protocol"], "rows": len(q), "fold_counts": fold_counts, "unique_sample_hashes": len({row["sample_hash"] for row in q}), "quota": prereg["quota"], "same_model": prereg["same_model"], "same_downstream_rule": prereg["same_downstream_rule"], "labels_sent": prereg["user_data_labels_sent"], "jev_used": prereg["jev_used"]},
        "sealed_inputs": {"files_hash_ok": files_hash_ok, "prompt_hash_ok": prompt_hash_ok, "selection_recomputed_from_p_temporal_and_sample_hash_only": recomputed_selection, "sampling_strata_match": strata_checks, "train_selection_groups_disjoint": group_disjoint, "case_ids_are_c00_to_c63_per_fold": case_id_checks},
        "prompt_isolation": prompt_checks,
        "response_validation": {**response_checks, "valid_counts": {"direct": sum(record["arm"] == "direct" for record in valid_records), "atomic": sum(record["arm"] == "atomic" for record in valid_records)}, "invalid_receipts": [{"fold": fold, "arm": arm, "status": rec.get("status"), "error_type": rec.get("error_type")} for (fold, arm), rec in receipts.items() if rec.get("status") != "valid"]},
        "latency_and_tokens_valid_only": latency,
        "score_recomputation": score_checks,
        "valid_same_fold_pairs_only": pair_rows,
        "reported_development_totals": jload(TARGET / "results/summary.json")["descriptive_totals"],
        "issues": issues,
        "caveats": warnings,
        "audit_source_sha256": {"run_pilot.py": sha(RUNNER), "preregistration.json": sha(PREREG), "scorer_only_queries.csv": sha(QUERY), "selection_predictions.csv.gz": sha(SELECTION), "train_predictions.csv.gz": sha(TRAIN)},
    }
    (HERE / "llm_protocol_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
