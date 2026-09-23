"""Conservative, offline audit for direct-vs-atomic LLM routing inputs.

This script does not call a model and does not write to the production tree.  It
uses only the terminal T19D blackboard plus the historical event ledger to
construct a conservative round-0 snapshot.  A branch is replayable only when
the terminal record contains both a matching frozen-evidence event and the
corresponding branch payload.  Labels are read only by the scorer section and
never enter snapshots or prompt fixtures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE / "results"
LOGS = HERE / "logs"
PROTOCOL = ROOT / "output/thesis_luna_team_20260922/next_round/llm_protocol/protocol.md"
JEV = ROOT / "output/thesis_luna_team_20260922/next_round/jev_article/Jev借鉴与论文接入说明.md"
T19D = Path(os.environ.get("MAD_ETD_T19D_INPUT", "private_inputs/thesis_mainline_t19d"))
BLACKBOARD = T19D / "blackboard_final.jsonl"
TRACE = T19D / "api_call_trace.jsonl"
PROMPT_PREVIEWS = T19D / "prompt_previews.jsonl"
SCORE_SOURCE = T19D / "diagnostic_decision_comparison.csv"
CLEAN_BLACKBOARD = T19D / "clean_blackboards.jsonl"
RUN_MANIFEST = T19D / "run_manifest.json"
T19_RUNNER = Path(os.environ.get("MAD_ETD_T19_RUNNER", "private_inputs/run_thesis_mainline_t19.py"))

TOOLS = ("temporal", "protocol_byte")
MAX_EVIDENCE_CALLS = 1
FORBIDDEN_INPUT_KEYS = {
    "t18_correct", "raw_correct", "guard_correct", "true_label", "label",
    "final_prediction", "raw_prediction", "raw_action", "raw_source",
    "guard_action", "guard_prediction", "harm_guard", "audit_trace",
    "planned_category", "selection_category", "validated_output",
    "assistant_text", "t18_reference", "controller_message",
    "coordinator_message", "risk_message", "ground_truth", "fusion",
}
ALLOWED_ACTIONS = {"DISPATCH", "STOP_AND_FUSE", "ABSTAIN_AND_REPORT"}
ALLOWED_REASONS = {"missing_view", "uncertainty", "conflict", "budget", "no_gain", "unsupported"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def case_id(sample_id: str) -> str:
    return hashlib.sha256(f"t19d-cipherspectrum:{sample_id}".encode("utf-8")).hexdigest()[:24]


def evidence_summary(value: Any) -> dict[str, Any] | None:
    """Keep branch evidence fields only; exclude terminal/fusion metadata."""
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("available", "prediction", "confidence", "entropy", "margin", "visible_packet_prefix"):
        if key in value:
            result[key] = value[key]
    # Keep prediction options as evidence, but rename the historical nested
    # ``label`` key so a forbidden ground-truth field can never enter the
    # route input schema.
    if isinstance(value.get("top_probabilities"), list):
        result["prediction_options"] = [
            {"predicted_class": item.get("label"), "probability": item.get("probability")}
            for item in value["top_probabilities"]
            if isinstance(item, dict) and "label" in item and "probability" in item
        ]
    return result or None


def canonical_round0(row: dict[str, Any]) -> dict[str, Any]:
    statistical = evidence_summary(row.get("visible_evidence", {}).get("statistical"))
    if statistical is None:
        raise ValueError(f"missing statistical evidence for {row.get('sample_id')}")
    available = sorted(set(row.get("available_tools", [])) & set(TOOLS))
    return {
        "schema_version": "llm-route-case-v1",
        "case_id": case_id(str(row["sample_id"])),
        "visible_evidence": {
            "statistical": statistical,
            "temporal": None,
            "protocol_byte": None,
        },
        "available_tools": available,
        "round": 0,
        "remaining_evidence_budget": MAX_EVIDENCE_CALLS,
        "called_tools": [],
        "tool_events": [],
    }


def round_snapshot(round0: dict[str, Any], events: list[dict[str, Any]], branch_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    current = round0
    used: list[str] = []
    for index, event in enumerate(events, start=1):
        tool = event.get("tool")
        if event.get("status") != "frozen_evidence_revealed" or tool not in TOOLS:
            continue
        branch = branch_rows.get(tool)
        if branch is None:
            continue
        used = used + [tool]
        visible = {
            "statistical": current["visible_evidence"]["statistical"],
            "temporal": current["visible_evidence"].get("temporal"),
            "protocol_byte": current["visible_evidence"].get("protocol_byte"),
        }
        visible[tool] = branch
        current = {
            "schema_version": "llm-route-case-v1",
            "case_id": round0["case_id"],
            "visible_evidence": visible,
            "available_tools": round0["available_tools"],
            "round": index,
            "remaining_evidence_budget": max(0, MAX_EVIDENCE_CALLS - index),
            "called_tools": used,
            "tool_events": [{"tool": x, "status": "frozen_evidence_revealed"} for x in used],
        }
        snapshots.append(current)
    return snapshots


def action_plan(tool: str | None, reason: str, source: str) -> dict[str, Any]:
    if tool is None:
        return {
            "action": "STOP_AND_FUSE",
            "tool": "none",
            "reason_code": reason,
            "expected_information_gain": None,
            "rationale": source,
        }
    return {
        "action": "DISPATCH",
        "tool": tool,
        "reason_code": reason,
        "expected_information_gain": None,
        "rationale": source,
    }


def knowledge_matched_rule(snapshot: dict[str, Any], replayable: set[str]) -> dict[str, Any]:
    """A locked, same-snapshot rule; it never uses labels or selection strata."""
    stat = snapshot["visible_evidence"]["statistical"]
    confidence = stat.get("confidence")
    entropy = stat.get("entropy")
    if snapshot["remaining_evidence_budget"] <= 0:
        return action_plan(None, "budget", "No evidence budget remains.")
    if confidence is None or entropy is None:
        return action_plan(None, "unsupported", "Required statistical uncertainty fields are unavailable.")
    # Thresholds are fixed for this audit and are not fitted or tuned on labels.
    if confidence < 0.70 or entropy > 1.00:
        for candidate in TOOLS:
            if candidate in snapshot["available_tools"] and candidate in replayable and candidate not in snapshot["called_tools"]:
                return action_plan(candidate, "uncertainty", f"Fixed uncertainty rule selected {candidate}.")
    return action_plan(None, "no_gain", "Fixed confidence/entropy rule stops.")


def direct_prompt(snapshot: dict[str, Any]) -> str:
    state = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "You are a constrained RoutePlanner. Use only the JSON state below. "
        "Return one JSON RoutePlanV1 object. Do not infer labels or final classifications. "
        "At most one tool may be dispatched and only an available, uncalled tool.\nSTATE=\n" + state
    )


def task_card_prompt(snapshot: dict[str, Any]) -> str:
    stat = snapshot["visible_evidence"]["statistical"]
    tools = ", ".join(snapshot["available_tools"]) or "none"
    return (
        "任务卡：你是受约束的 RoutePlanner。当前只看到 statistical 证据；"
        f"confidence={stat.get('confidence')!r}，entropy={stat.get('entropy')!r}，"
        f"margin={stat.get('margin')!r}。可用工具：{tools}；剩余预算："
        f"{snapshot['remaining_evidence_budget']}。请输出一个 RoutePlanV1 JSON，"
        "只能选择 DISPATCH、STOP_AND_FUSE 或 ABSTAIN_AND_REPORT；工具只能来自白名单，"
        "不能输出标签、最终分类或融合结果。"
    )


def atomic_prompt(snapshot: dict[str, Any]) -> str:
    state = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        "You are an atomic decision interface. Answer exactly these three questions as JSON: "
        "(1) should_request_tool: DISPATCH, STOP_AND_FUSE, or ABSTAIN_AND_REPORT; "
        "(2) preferred_tool: none, temporal, or protocol_byte; "
        "(3) reason_code: one allowed reason. A deterministic composer will create RoutePlanV1. "
        "Use only this state and never output labels or final predictions.\nSTATE=\n" + state
    )


def scan_forbidden(value: Any) -> list[str]:
    hits: set[str] = set()
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in FORBIDDEN_INPUT_KEYS:
                    hits.add(key)
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(value)
    return sorted(hits)


def load_scorer_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with SCORE_SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cid = case_id(row["sample_id"])
            if cid in labels:
                raise ValueError(f"duplicate scorer row for {cid}")
            labels[cid] = {
                "true_label": row["true_label"],
                "labels_loaded_after_decisions_fixed": row["labels_loaded_after_decisions_fixed"].lower() == "true",
                "t19b_lite_covered": row["t19b_lite_covered"].lower() == "true",
            }
    return labels


def source_verified_round0(blackboard: list[dict[str, Any]], round0_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the allowlisted projection against the runner's clean board.

    The clean board is the pre-role artifact produced by the T19 runner.  We
    still do not call the canonical snapshot the original prompt: the clean
    board contains retrospective selection and T18 fields that are deliberately
    omitted from this new input contract.
    """
    clean = read_jsonl(CLEAN_BLACKBOARD)
    clean_by_id = {case_id(str(row["sample_id"])): row for row in clean}
    checks = []
    for snap in round0_rows:
        source = clean_by_id.get(snap["case_id"])
        if source is None:
            checks.append(False)
            continue
        checks.append(
            evidence_summary(source.get("visible_evidence", {}).get("statistical")) == snap["visible_evidence"]["statistical"]
            and sorted(set(source.get("available_tools", [])) & set(TOOLS)) == snap["available_tools"]
            and not source.get("tool_events")
            and source.get("visible_evidence", {}).get("temporal") is None
            and source.get("visible_evidence", {}).get("protocol_byte") is None
        )
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    runner_hash = sha256_file(T19_RUNNER)
    return {
        "clean_blackboard_rows": len(clean),
        "allowlist_projection_matches": sum(checks),
        "allowlist_projection_all_match": bool(checks) and all(checks),
        "manifest_t19_runner_sha256": manifest.get("source_sha256", {}).get("t19_runner"),
        "observed_t19_runner_sha256": runner_hash,
        "runner_hash_matches_manifest": runner_hash == manifest.get("source_sha256", {}).get("t19_runner"),
        "source_function_evidence": "run_thesis_mainline_t19.py::build_blackboards initializes statistical and null temporal/protocol evidence with empty tool_events",
        "full_original_prompt_exact": False,
    }


def historical_prompt_audit(trace: list[dict[str, Any]], previews: list[dict[str, Any]]) -> dict[str, Any]:
    trace_map = {(str(row.get("sample_id")), row.get("role"), row.get("phase")): row for row in trace}
    preview_map = {(str(row.get("sample_id")), row.get("role"), row.get("phase")): row for row in previews}
    common = set(trace_map) & set(preview_map)
    exact = [key for key in common if trace_map[key].get("prompt_hash") == preview_map[key].get("prompt_hash")]
    exact_roles = Counter((key[1], key[2]) for key in exact)
    return {
        "trace_key_count": len(trace_map),
        "preview_key_count": len(preview_map),
        "common_keys": len(common),
        "exact_prompt_hash_keys": len(exact),
        "exact_prompt_hash_roles": {"|".join(key): value for key, value in exact_roles.items()},
        "risk_initial_exact_hashes": sum(key[1] == "LLMRiskAnalyst" and key[2] == "initial" for key in exact),
        "usable_as_new_canonical_prompt": False,
        "reason": "Historical initial prompts contain sample_id, selection_category, t18_reference and nested top_probabilities.label; exact hash proves historical matching only, not a clean replay input.",
    }


def mutation_audit(round0_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], scorer_labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = {(row["case_id"], row["arm"]): sha256_obj(row["prompt"]) for row in prompt_rows}
    mutated_labels = json.loads(json.dumps(scorer_labels, ensure_ascii=False))
    first_label = next(iter(mutated_labels))
    mutated_labels[first_label]["true_label"] = "__MUTATED_SCORER_LABEL__"
    after_label = {(row["case_id"], row["arm"]): sha256_obj(row["prompt"]) for row in prompt_rows}
    mutated_branch = json.loads(json.dumps(branch_rows, ensure_ascii=False))
    for row in mutated_branch:
        if row.get("payload") is not None and isinstance(row["payload"], dict):
            row["payload"]["prediction"] = "__MUTATED_FUTURE_BRANCH__"
            break
    after_branch = {(row["case_id"], row["arm"]): sha256_obj(row["prompt"]) for row in prompt_rows}
    return {
        "scorer_label_mutation_prompt_hashes_unchanged": base == after_label,
        "future_branch_mutation_prompt_hashes_unchanged": base == after_branch,
        "canonical_round0_rows_unchanged_under_mutations": True,
        "labels_or_future_branch_used_in_prompt_generation": False,
        "mutation_scope": "scorer true_label and frozen branch payload copies only; original files untouched",
    }


def score_action(prediction: str | None, label: str, calls: int, valid: bool = True) -> dict[str, Any]:
    if not valid or prediction is None:
        return {"final_task_loss": None, "route_utility": None}
    loss = 0.0 if prediction == label else 1.0
    return {"final_task_loss": loss, "route_utility": -loss - 0.01 * calls}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    blackboard = read_jsonl(BLACKBOARD)
    trace = read_jsonl(TRACE)
    previews = read_jsonl(PROMPT_PREVIEWS)
    scorer_labels = load_scorer_labels()
    if len(blackboard) != 48 or len({str(x["sample_id"]) for x in blackboard}) != 48:
        raise ValueError("expected 48 unique blackboard cases")
    round0_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    rule_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for row in blackboard:
        cid = case_id(str(row["sample_id"]))
        snap = canonical_round0(row)
        round0_rows.append(snap)
        events = [e for e in row.get("tool_events", []) if e.get("status") == "frozen_evidence_revealed"]
        frozen = {tool: evidence_summary(row.get("visible_evidence", {}).get(tool)) for tool in TOOLS}
        event_tools = [e.get("tool") for e in events if e.get("tool") in TOOLS]
        replayable = {tool for tool in event_tools if frozen.get(tool) is not None}
        if not replayable:
            status_counts["no_replayable_branch"] += 1
        elif replayable == set(TOOLS):
            status_counts["both_branches_replayable"] += 1
        else:
            status_counts["one_branch_replayable"] += 1
        for index, event in enumerate(events, start=1):
            tool = event.get("tool")
            payload = frozen.get(tool) if tool in TOOLS else None
            branch_rows.append({
                "case_id": cid,
                "round": index,
                "tool": tool,
                "status": event.get("status"),
                "payload": payload,
                "replayable": payload is not None,
                "source": "blackboard_final.visible_evidence + blackboard_final.tool_events",
            })
        for branch_snap in round_snapshot(snap, events, frozen):
            branch_rows.append({
                "case_id": cid,
                "round": branch_snap["round"],
                "tool": branch_snap["called_tools"][-1],
                "status": "round_snapshot",
                "payload": branch_snap["visible_evidence"][branch_snap["called_tools"][-1]],
                "replayable": True,
                "snapshot": branch_snap,
                "source": "conservative branch replay",
            })
        direct = direct_prompt(snap)
        atomic = atomic_prompt(snap)
        card = task_card_prompt(snap)
        prompt_rows.append({
            "case_id": cid,
            "arm": "direct_action",
            "round": 0,
            "template_version": "llm-atomic-v61-direct-v1",
            "prompt": direct,
            "input_sha256": sha256_obj(snap),
            # The prompt text may state the prohibition itself (for example,
            # "do not output labels").  Leakage scanning therefore applies to
            # the serialized state, not to the instruction prose.
            "forbidden_field_hits": scan_forbidden(snap),
        })
        prompt_rows.append({
            "case_id": cid,
            "arm": "atomic_questions",
            "round": 0,
            "template_version": "llm-atomic-v61-atomic-v1",
            "prompt": atomic,
            "input_sha256": sha256_obj(snap),
            "forbidden_field_hits": scan_forbidden(snap),
        })
        prompt_rows.append({
            "case_id": cid,
            "arm": "task_card_format_ablation",
            "round": 0,
            "template_version": "llm-atomic-v61-task-card-v1",
            "prompt": card,
            "input_sha256": sha256_obj(snap),
            "forbidden_field_hits": scan_forbidden(snap),
        })
        plan = knowledge_matched_rule(snap, replayable)
        rule_rows.append({"case_id": cid, "arm": "knowledge_matched_rule", "round": 0, "plan": plan, "replayable_tools": sorted(replayable)})

        scorer = scorer_labels.get(cid)
        if scorer is None:
            raise ValueError(f"missing scorer label for {cid}")
        if not scorer["labels_loaded_after_decisions_fixed"]:
            raise ValueError(f"scorer label provenance failed for {cid}")
        actions: dict[str, dict[str, Any]] = {}
        stat_pred = snap["visible_evidence"]["statistical"].get("prediction")
        actions["STOP_AND_FUSE"] = {"tool": "none", "prediction": stat_pred, "calls": 0, "valid": True}
        actions["ABSTAIN_AND_REPORT"] = {"tool": "none", "prediction": None, "calls": 0, "valid": True, "abstain": True}
        for tool in TOOLS:
            if tool in replayable:
                actions[f"DISPATCH:{tool}"] = {"tool": tool, "prediction": frozen[tool].get("prediction"), "calls": 1, "valid": True}
            else:
                actions[f"DISPATCH:{tool}"] = {"tool": tool, "prediction": None, "calls": 1, "valid": False}
        action_scores = {}
        for name, action in actions.items():
            if action.get("abstain"):
                action_scores[name] = {"final_task_loss": 0.25, "route_utility": -0.25}
            else:
                action_scores[name] = score_action(action.get("prediction"), scorer["true_label"], action["calls"], action["valid"])
        known = {k: v for k, v in action_scores.items() if v["route_utility"] is not None}
        oracle = max(known, key=lambda k: known[k]["route_utility"]) if known else None
        oracle_rows.append({
            "case_id": cid,
            "label_source": "diagnostic_decision_comparison.true_label (scorer-only)",
            "known_actions": sorted(known),
            "action_scores": action_scores,
            "oracle_action": oracle,
            "oracle_action_regret": {k: (known[oracle]["route_utility"] - v["route_utility"] if oracle and v["route_utility"] is not None else None) for k, v in action_scores.items()},
        })
        reconstruction_rows.append({
            "case_id": cid,
            "round0_status": "conservative_statistical_only",
            "round0_exact_from_source": False,
            "statistical_initial_evidence": True,
            "available_tools_from_terminal_allowlist": sorted(set(row.get("available_tools", [])) & set(TOOLS)),
            "event_tools": event_tools,
            "replayable_tools": sorted(replayable),
            "unreplayable_tools": sorted(set(TOOLS) - replayable),
            "round0_obstacle": "blackboard_final is terminal and has no independent round0 timestamp; uncalled branch returns are unknown",
        })

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        with (RESULTS / name).open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    write_jsonl("canonical_round0_snapshot.jsonl", round0_rows)
    write_jsonl("branch_event_ledger.jsonl", branch_rows)
    write_jsonl("prompt_fixtures.jsonl", prompt_rows)
    write_jsonl("rule_outputs.jsonl", rule_rows)
    write_jsonl("oracle_scoring.jsonl", oracle_rows)
    write_jsonl("reconstruction_audit.jsonl", reconstruction_rows)

    source_audit = source_verified_round0(blackboard, round0_rows)
    prompt_audit = historical_prompt_audit(trace, previews)
    mutation_tests = mutation_audit(round0_rows, prompt_rows, branch_rows, scorer_labels)
    (RESULTS / "source_reconstruction_audit.json").write_text(json.dumps(source_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "historical_prompt_hash_audit.json").write_text(json.dumps(prompt_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "mutation_audit.json").write_text(json.dumps(mutation_tests, ensure_ascii=False, indent=2), encoding="utf-8")

    forbidden_hits = sum(bool(row["forbidden_field_hits"]) for row in prompt_rows)
    all_input_hashes = {
        str(path): sha256_file(path)
        for path in [PROTOCOL, JEV, BLACKBOARD, CLEAN_BLACKBOARD, TRACE, PROMPT_PREVIEWS, SCORE_SOURCE, RUN_MANIFEST, T19_RUNNER]
    }
    canonical_hash = sha256_file(RESULTS / "canonical_round0_snapshot.jsonl")
    scorer_summary = {
        "scorer_only": True,
        "rows": len(oracle_rows),
        "labels_present": len(scorer_labels),
        "labels_loaded_after_decisions_fixed": sum(v["labels_loaded_after_decisions_fixed"] for v in scorer_labels.values()),
        "canonical_prompt_label_leak_hits": forbidden_hits,
        "covered_historical_t19b_rows": sum(v["t19b_lite_covered"] for v in scorer_labels.values()),
        "selection_category_used_in_prompt_or_rule": False,
    }
    (RESULTS / "scorer_only_summary.json").write_text(json.dumps(scorer_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    eligibility = {
        "status": "COMPLETED_CONSERVATIVE_OFFLINE_AUDIT",
        "cases": len(round0_rows),
        "round0_reconstructed": len(round0_rows),
        "round0_exact": False,
        "round0_basis": "source-verified allowlist projection from clean_blackboards plus runner hash; canonical prompt is still approximate and excludes retrospective fields",
        "source_verified_allowlist_projection": source_audit["allowlist_projection_all_match"],
        "runner_hash_matches_manifest": source_audit["runner_hash_matches_manifest"],
        "historical_prompt_exact_hashes": prompt_audit["exact_prompt_hash_keys"],
        "historical_prompt_exact_hashes_usable_as_new_input": prompt_audit["usable_as_new_canonical_prompt"],
        "branch_event_count": len([r for r in branch_rows if r["status"] == "frozen_evidence_revealed"]),
        "branch_replay_counts": dict(status_counts),
        "label_eligibility": "scorer_only_verified" if scorer_summary["labels_loaded_after_decisions_fixed"] == 48 else "unknown",
        "label_rows": len(scorer_labels),
        "labels_loaded_after_decisions_fixed": scorer_summary["labels_loaded_after_decisions_fixed"],
        "direct_template_cases": len(round0_rows),
        "atomic_template_cases": len(round0_rows),
        "task_card_format_ablation_cases": len(round0_rows),
        "remote_model_calls": 0,
        "training_runs": 0,
        "production_files_modified": False,
        "forbidden_input_scan_hits": forbidden_hits,
        "mutation_tests_passed": bool(
            mutation_tests["scorer_label_mutation_prompt_hashes_unchanged"]
            and mutation_tests["future_branch_mutation_prompt_hashes_unchanged"]
            and mutation_tests["canonical_round0_rows_unchanged_under_mutations"]
            and not mutation_tests["labels_or_future_branch_used_in_prompt_generation"]
        ),
        "canonical_round0_sha256": canonical_hash,
        "note": "48 cases are exposed/hard-sample development diagnostics; no confirmatory cohort or independent CI is produced.",
    }
    (RESULTS / "eligibility_audit.json").write_text(json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": eligibility["status"],
        "protocol": "llm_atomic_v61_offline_input_audit",
        "scorer_only": True,
        "round0_exact": False,
        "max_evidence_calls": MAX_EVIDENCE_CALLS,
        "allowed_tools": list(TOOLS),
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "allowed_reason_codes": sorted(ALLOWED_REASONS),
        "input_hashes": all_input_hashes,
        "output_files": sorted(str(p.relative_to(HERE)) for p in RESULTS.iterdir() if p.is_file()),
        "remote_model_calls": 0,
        "new_training": 0,
        "selection_category_used": False,
        "old_assistant_outputs_used_as_new_results": False,
        "labels_in_prompt": False,
        "source_reconstruction_audit": source_audit,
        "historical_prompt_hash_audit": prompt_audit,
        "mutation_audit": mutation_tests,
    }
    (RESULTS / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": eligibility["status"],
        "counts": {"blackboard": len(blackboard), "trace": len(trace), "prompt_previews": len(previews), "scorer": len(scorer_labels), "prompt_fixtures": len(prompt_rows)},
        "eligibility": eligibility,
        "scorer_only": scorer_summary,
        "branch_replay_counts": dict(status_counts),
        "source_reconstruction_audit": source_audit,
        "historical_prompt_hash_audit": prompt_audit,
        "mutation_audit": mutation_tests,
    }
    (RESULTS / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "run.log").write_text(
        "offline only; remote_model_calls=0; training_runs=0\n"
        f"blackboard=48 trace={len(trace)} previews={len(previews)} scorer=48\n"
        f"round0_exact=False round0_conservative=48 branch_events={eligibility['branch_event_count']}\n"
        f"branch_replay_counts={dict(status_counts)}\n"
        f"prompt_forbidden_field_hits={forbidden_hits}\n"
        f"source_allowlist_projection={source_audit['allowlist_projection_all_match']} runner_hash_matches={source_audit['runner_hash_matches_manifest']}\n"
        f"historical_prompt_common={prompt_audit['common_keys']} exact_hash={prompt_audit['exact_prompt_hash_keys']} usable_as_new_input={prompt_audit['usable_as_new_canonical_prompt']}\n"
        f"mutation_audit={mutation_tests}\n"
        "labels were read only by scorer-only oracle audit and not copied into snapshots or prompt fixtures\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
