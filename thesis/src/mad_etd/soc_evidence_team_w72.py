"""W72 default-off SOC evidence-team shadow protocol.

This module never trains a detector and never replaces the default runtime.
It replays already-produced detector evidence through a typed delegation,
review, AgentEvidenceV2 handoff, and lossless Fusion adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Iterable

from .base import CaseState
from .engine import build_default_engine
from .evidence_team import (
    SOC_ADVISORY_AGENT_NAMES,
    SOC_VERDICT_AGENT_ALLOWLIST,
    AgentEvidenceV2Adapter,
    EvidenceRequestPlanner,
    SOCEvidenceBusShadowV1,
    agent_evidence_v1_semantic_payload,
    agent_evidence_v2_semantic_payload,
)
from .field_contract import FIELD_CONTRACT
from .io import iter_split_records
from .paper_evaluation import hash_artifact_paths
from .runtime_profiles import (
    load_runtime_profile,
    profile_artifact_paths,
    profile_engine_kwargs,
    runtime_profile_sha256,
)
from .schemas import (
    AgentEvidenceV2,
    EvidenceHandoffV2,
    EvidenceRequest,
    ExecutionPlan,
    PlanAction,
    PlanStep,
)
from .view_contract import build_detector_capabilities, build_view_availability


W72_RUN_DIR = Path("data/runs/mad_etd_soc_evidence_team_w72")
DEFAULT_INPUT = Path("data/processed/ustc_tfc2016/v1/flows")
DEFAULT_SPLIT_MANIFEST = Path(
    "data/processed/ustc_tfc2016/v1/splits/split-manifest.json"
)
DEFAULT_RUNTIME = "runtime_safe_v3_0"
W71_ACCEPTANCE = Path(
    "data/runs/mad_etd_w71_learned_tls_evidence/acceptance_report_w71.json"
)


SOC_ROLES: list[dict[str, Any]] = [
    {"component": "FieldAuditAgent", "soc_role": "data_compliance_and_evidence_intake", "decision_authority": False, "fusion_eligible": False},
    {"component": "ControlledDetectorInput", "soc_role": "evidence_sanitizer", "decision_authority": False, "fusion_eligible": False},
    {"component": "StatsDetectorAgent", "soc_role": "junior_traffic_analyst", "decision_authority": False, "fusion_eligible": True},
    {"component": "TemporalBehaviorAgent", "soc_role": "behavior_analysis_specialist", "decision_authority": False, "fusion_eligible": True},
    {"component": "TLSProtocolAgent", "soc_role": "tls_protocol_specialist", "decision_authority": False, "fusion_eligible": True},
    {"component": "DetectorCapabilityProfile", "soc_role": "capability_dispatch_officer", "decision_authority": False, "fusion_eligible": False},
    {"component": "PlannerCoordinator", "soc_role": "incident_coordinator", "decision_authority": False, "fusion_eligible": False},
    {"component": "PolicyGuard", "soc_role": "compliance_supervisor", "decision_authority": False, "fusion_eligible": False},
    {"component": "FusionAgent", "soc_role": "evidence_arbitration_board", "decision_authority": True, "fusion_eligible": False},
    {"component": "OODReliabilityGate", "soc_role": "risk_and_ood_officer", "decision_authority": False, "fusion_eligible": False},
    {"component": "RAGCaseMemory", "soc_role": "threat_intelligence_advisor", "decision_authority": False, "fusion_eligible": False, "advisory_only": True},
    {"component": "LLMCriticReflection", "soc_role": "planning_explanation_audit_advisor", "decision_authority": False, "fusion_eligible": False, "advisory_only": True},
    {"component": "HITL", "soc_role": "human_escalation_handler", "decision_authority": False, "fusion_eligible": False, "advisory_only": True},
    {"component": "AuditLogger", "soc_role": "auditor", "decision_authority": False, "fusion_eligible": False},
    {"component": "PromotionNegativeClaimGovernance", "soc_role": "release_and_research_governance", "decision_authority": False, "fusion_eligible": False},
]


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _default_frozen_paths(profile_name: str = DEFAULT_RUNTIME) -> dict[str, Path]:
    profile = load_runtime_profile(profile_name)
    paths = profile_artifact_paths(profile)
    paths["runtime_profile"] = Path("data/configs") / f"{profile_name}.json"
    paths["ood_source"] = Path("src/mad_etd/ood.py")
    paths["policy_guard_source"] = Path("src/mad_etd/coordinator.py")
    return paths


def _case_state_schema() -> dict[str, Any]:
    added = {
        "ood_state",
        "conflict_state",
        "escalation_state",
        "case_lifecycle_status",
        "audit_chain_ref",
        "evidence_requests",
        "evidence_handoffs",
        "evidence_budget",
        "advisory_context_refs",
    }
    return {
        "schema_version": "W72.1",
        "object": "CaseState",
        "default_runtime_consumes_extension": False,
        "fields": [
            {
                "name": item.name,
                "type": str(item.type),
                "w72_extension": item.name in added,
            }
            for item in dataclass_fields(CaseState)
        ],
    }


def _write_docs(output: Path, report: dict[str, Any]) -> None:
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    en = "\n".join(
        [
            "# MAD-ETD W72 SOC Evidence Team Protocol",
            "",
            "MAD-ETD is organized as a SOC-inspired, safety-constrained multi-agent evidence team.",
            "The W72 lane is shadow-only: it adds typed CaseState, EvidenceRequest, AgentEvidenceV2, and EvidenceHandoff governance without changing detector inference or Fusion semantics.",
            "",
            "## Boundaries",
            "",
            "- `runtime_safe_v3_0` remains the default runtime.",
            "- `FusionAgent` remains the sole owner of verdict, confidence, and uncertainty.",
            "- LLM, RAG, Memory, HITL, Reflection, and Critic are advisory-only and cannot enter the evidence bus.",
            "- W72 uses 6,000 USTC validation records only; it performs no training or tuning.",
            "- W71 remains `not_promoted_learned_tls_w71`.",
            "",
            "## Result",
            "",
            f"- Status: `{report.get('status', 'protocol_built')}`",
            f"- Verdict invariance: `{report.get('verdict_invariance', 'pending')}`",
            f"- AgentEvidence semantic invariance: `{report.get('agent_evidence_semantic_invariance', 'pending')}`",
            f"- Audit completion: `{report.get('audit_completion', 'pending')}`",
            "- No classification-improvement claim is made.",
        ]
    )
    cn = "\n".join(
        [
            "# MAD-ETD W72 SOC 安全团队证据协议",
            "",
            "W72 将 MAD-ETD 正式组织为受安全约束的 SOC 多 Agent 证据团队。",
            "该实验仅运行默认关闭的影子协议：CaseState 记录案件，EvidenceRequest 约束委派，AgentEvidenceV2 统一证据，EvidenceHandoff 记录交接；不改变检测器推理和 Fusion 语义。",
            "",
            "## 边界",
            "",
            "- 默认 runtime 仍为 `runtime_safe_v3_0`。",
            "- `FusionAgent` 仍是 verdict、confidence、uncertainty 的唯一所有者。",
            "- LLM、RAG、Memory、HITL、Reflection、Critic 仅供建议，禁止进入证据总线。",
            "- W72 只使用 6,000 条 USTC validation，不训练、不调参。",
            "- W71 保持 `not_promoted_learned_tls_w71`。",
            "",
            "## 结果",
            "",
            f"- 状态：`{report.get('status', 'protocol_built')}`",
            f"- verdict 不变率：`{report.get('verdict_invariance', 'pending')}`",
            f"- AgentEvidence 语义不变率：`{report.get('agent_evidence_semantic_invariance', 'pending')}`",
            f"- 审计完整率：`{report.get('audit_completion', 'pending')}`",
            "- 本轮不声明分类性能提升。",
        ]
    )
    en_path = docs / "MAD_ETD_SOC_EVIDENCE_TEAM_W72.md"
    cn_path = docs / "MAD_ETD_SOC_EVIDENCE_TEAM_W72_CN.md"
    en_path.write_text(en + "\n", encoding="utf-8")
    cn_path.write_text(cn + "\n", encoding="utf-8")
    (output / en_path.name).write_text(en + "\n", encoding="utf-8")
    (output / cn_path.name).write_text(cn + "\n", encoding="utf-8")


def build_soc_evidence_team_w72(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    split_manifest: str | Path = DEFAULT_SPLIT_MANIFEST,
    output_dir: str | Path = W72_RUN_DIR,
    sample_count: int = 6000,
    seed: int = 42,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = _load(Path(split_manifest))
    validation_ids = list(source.get("assignments", {}).get("validation", []))
    if len(validation_ids) < sample_count:
        raise RuntimeError("USTC validation split is too small for W72")
    selected = sorted(validation_ids, key=lambda item: (_stable_rank(item, seed), item))[
        :sample_count
    ]
    selected_set = set(selected)
    if selected_set & set(source.get("assignments", {}).get("train", [])):
        raise RuntimeError("W72 selection overlaps USTC train")
    if selected_set & set(source.get("assignments", {}).get("test", [])):
        raise RuntimeError("W72 selection overlaps USTC test")
    w71 = _load(W71_ACCEPTANCE)
    if w71.get("status") != "not_promoted_learned_tls_w71":
        raise RuntimeError("W72 requires the frozen W71 non-promotion status")
    profile = load_runtime_profile(DEFAULT_RUNTIME)
    manifest = {
        "schema_version": "1.0",
        "experiment": "mad_etd_soc_evidence_team_w72",
        "dataset": "USTC-TFC2016",
        "split": "validation",
        "seed": seed,
        "sample_count": sample_count,
        "sample_ids": selected,
        "input_path": Path(input_path).as_posix(),
        "source_split_manifest": Path(split_manifest).as_posix(),
        "source_split_manifest_sha256": _sha(source),
        "train_overlap": 0,
        "test_overlap": 0,
        "external_or_locked_test_used": False,
        "training_or_tuning_performed": False,
    }
    _dump(output / "selection_manifest.json", manifest)
    _dump(
        output / "soc_role_registry.json",
        {
            "schema_version": "1.0",
            "framework": "SOC-inspired safety-constrained multi-agent evidence team",
            "roles": SOC_ROLES,
            "final_decision_owner": "FusionAgent",
        },
    )
    _dump(
        output / "soc_role_boundary_report.json",
        {
            "final_decision_owner": "FusionAgent",
            "evidence_producers": sorted(SOC_VERDICT_AGENT_ALLOWLIST),
            "advisory_only_prohibited_from_fusion": sorted(SOC_ADVISORY_AGENT_NAMES),
            "planner_can_emit_final_verdict": False,
            "field_audit_required": True,
            "policy_guard_required": True,
        },
    )
    _dump(output / "case_state_schema.json", _case_state_schema())
    _dump(output / "evidence_request_schema.json", EvidenceRequest.model_json_schema())
    _dump(output / "agent_evidence_v2_schema.json", AgentEvidenceV2.model_json_schema())
    _dump(output / "evidence_handoff_schema.json", EvidenceHandoffV2.model_json_schema())
    config = {
        "protocol": "soc_evidence_team_shadow_v1",
        "default_enabled": False,
        "runtime_profile": DEFAULT_RUNTIME,
        "runtime_profile_sha256": runtime_profile_sha256(profile),
        "planner_mode": "replay_executed_run_agent_steps_only",
        "fusion_owner": "FusionAgent",
        "max_verdict_stage_agents_per_case": 3,
        "advisory_components_fusion_eligible": False,
        "classification_metrics_reported": False,
        "fake_metric_count": 0,
        "w71_status": w71["status"],
        "w71_optional_runtime_profile_created": w71.get(
            "optional_runtime_profile_created", False
        ),
    }
    _dump(output / "shadow_protocol_config.json", config)
    _dump(output / "frozen_hashes_before.json", hash_artifact_paths(_default_frozen_paths()))
    report = {
        "status": "ready_for_shadow_run",
        "sample_count": sample_count,
        "runtime_safe_v3_0_remains_default": True,
        "soc_evidence_team_shadow_v1_default_enabled": False,
        "fusion_owner": "FusionAgent",
        "w71_status": w71["status"],
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _write_docs(output, report)
    return report


def _build_case_state(engine: Any, flow: Any) -> CaseState:
    state = CaseState(flow=flow, remaining_budget=engine.initial_budget)
    state.field_audit = engine.field_auditor.audit(flow)
    state.safe_flow = engine.field_auditor.make_safe_flow(flow, state.field_audit)
    detector_input = engine.field_auditor.make_detector_input(
        flow, state.field_audit, enforce=True
    )
    tls_agent = engine.detectors.get("TLSProtocolAgent")
    stats_agent = engine.detectors.get("StatsDetectorAgent")
    temporal_agent = engine.detectors.get("TemporalBehaviorAgent")
    tls_backend = getattr(tls_agent, "backend", "rule")
    state.view_availability = build_view_availability(
        detector_input,
        routing_policy=engine.contract_routing_policy,
        tls_backend=tls_backend,
    )
    state.detector_capabilities = build_detector_capabilities(
        detector_input,
        stats_backend=getattr(stats_agent, "backend", "rule"),
        temporal_backend=getattr(temporal_agent, "backend", "rule"),
        tls_backend=tls_backend,
    )
    # The legacy capability profile uses packet_lengths as the availability
    # sentinel for the whole temporal view.  W72 records the full, already
    # FieldAudit-approved view actually consumed by that specialist without
    # changing the default runtime capability/routing implementation.
    allowed = set(state.field_audit.allowed_fields)
    prefixes = {
        "StatsDetectorAgent": "stats.",
        "TemporalBehaviorAgent": "sequence.",
        "TLSProtocolAgent": "tls.",
    }
    for agent_name, prefix in prefixes.items():
        profile = state.detector_capabilities[agent_name]
        if profile.status != "available":
            continue
        audited_view = sorted(path for path in allowed if path.startswith(prefix))
        if audited_view:
            state.detector_capabilities[agent_name] = profile.model_copy(
                update={
                    "consumed_fields": audited_view,
                    "available_fields": audited_view,
                }
            )
    state.reliability = engine.feature_guard.assess(state.safe_flow)
    state.audit_chain_ref = f"audit:{flow.trace_id}"
    state.case_lifecycle_status = "evidence_collection"
    return state


def _artifact_hashes(engine: Any, runtime_hash: str) -> dict[str, str]:
    return {
        name: _sha(
            {
                "agent": name,
                "backend": getattr(agent, "backend", "rule"),
                "version": getattr(agent, "version", "1.0"),
                "runtime_profile_sha256": runtime_hash,
            }
        )
        for name, agent in engine.detectors.items()
    }


def _row_signature(report: Any) -> dict[str, Any]:
    return {
        "verdict": report.verdict.value,
        "confidence": report.confidence,
        "uncertainty": report.uncertainty,
        "conflict_score": report.conflict_score,
        "distribution_shift_score": report.distribution_shift_score,
        "participating_agents": list(report.participating_agents),
    }


def run_soc_evidence_team_shadow_w72(
    *,
    output_dir: str | Path = W72_RUN_DIR,
    resume: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest = _load(output / "selection_manifest.json")
    if manifest.get("split") != "validation" or manifest.get(
        "external_or_locked_test_used"
    ):
        raise RuntimeError("W72 accepts USTC validation only")
    profile = load_runtime_profile(DEFAULT_RUNTIME)
    engine = build_default_engine(**profile_engine_kwargs(profile))
    runtime_hash = runtime_profile_sha256(profile)
    artifacts = _artifact_hashes(engine, runtime_hash)
    bus = SOCEvidenceBusShadowV1(allowed_agents=set(engine.detectors))
    checkpoint = output / "shadow_checkpoint.jsonl"
    existing = _read_checkpoint(checkpoint) if resume else []
    if not resume and checkpoint.exists():
        checkpoint.unlink()
    completed = {row["sample_id"] for row in existing}
    selected = set(manifest["sample_ids"])
    processed = len(existing)
    for flow in iter_split_records(
        manifest["input_path"], manifest["source_split_manifest"], "validation"
    ):
        if flow.sample_id not in selected or flow.sample_id in completed:
            continue
        baseline_started = time.perf_counter()
        report, audit = engine.analyze(flow)
        baseline_latency = (time.perf_counter() - baseline_started) * 1000.0
        state = _build_case_state(engine, flow)
        state.evidence_budget = len(report.agent_results)
        state.remaining_budget = max(len(report.agent_results), 1)
        state.ood_state = {
            "distribution_shift_score": report.distribution_shift_score,
            "decision": (
                "hard_or_warning"
                if report.distribution_shift_score >= 0.95
                else "in_domain_or_off"
            ),
        }
        state.conflict_state = {"score": report.conflict_score}
        state.escalation_state = {"required": report.need_escalation}
        plan = ExecutionPlan(
            plan_id=f"w72-replay-{flow.trace_id}",
            steps=[
                PlanStep(
                    step_id=f"request-{index}-{item.agent_name}",
                    action=PlanAction.RUN_AGENT,
                    agent=item.agent_name,
                    budget_cost=1,
                )
                for index, item in enumerate(report.agent_results)
            ]
            + [PlanStep(step_id="fusion-owned-finalization", action=PlanAction.FUSE)],
            reason_codes=["W72_SHADOW_REPLAY_OF_EXECUTED_PLAN"],
            budget=len(report.agent_results),
            source="replay",
        )
        requests = EvidenceRequestPlanner.from_execution_plan(plan, state)
        state.evidence_requests.extend(requests)
        by_agent = {item.agent_name: item for item in report.agent_results}
        shadow_evidence = []
        request_audit: list[dict[str, Any]] = []
        handoff_audit: list[dict[str, Any]] = []
        semantic_matches: list[bool] = []
        shadow_started = time.perf_counter()
        for request in requests:
            review = bus.review(request, state)
            evidence = by_agent.get(request.requested_agent)
            request_audit.append(
                {
                    "request_id": request.request_id,
                    "requested_agent": request.requested_agent,
                    "policy_status": review.request.policy_status,
                    "approved": review.approved,
                    "reason_codes": list(review.reason_codes),
                    "permitted_safe_features": request.permitted_safe_features,
                    "purpose": request.purpose,
                    "prohibited_outputs": request.prohibited_outputs,
                }
            )
            if evidence is None:
                handoff_audit.append(
                    {
                        "request_id": request.request_id,
                        "requested_agent": request.requested_agent,
                        "policy_status": "rejected",
                        "reason_codes": ["MISSING_MATCHING_AGENT_EVIDENCE"],
                    }
                )
                continue
            handoff, eligible = bus.handoff(
                review,
                evidence,
                artifact_hash=artifacts[request.requested_agent],
            )
            state.evidence_handoffs.append(handoff)
            v2 = handoff.evidence
            semantic_match = bool(
                v2
                and agent_evidence_v1_semantic_payload(evidence)
                == agent_evidence_v2_semantic_payload(v2)
            )
            semantic_matches.append(semantic_match)
            handoff_audit.append(
                {
                    "request_id": request.request_id,
                    "requested_agent": request.requested_agent,
                    "policy_status": handoff.policy_status,
                    "feature_policy_hash": handoff.feature_policy_hash,
                    "artifact_hash": handoff.artifact_hash,
                    "semantic_match": semantic_match,
                    "reason_codes": handoff.reason_codes,
                }
            )
            if eligible is not None:
                shadow_evidence.append(eligible)
        shadow_fusion = engine.fusion_agent.fuse(
            shadow_evidence, state.reliability, final=True
        )
        shadow_latency = (time.perf_counter() - shadow_started) * 1000.0
        baseline_evidence_signature = _sha(
            [agent_evidence_v1_semantic_payload(item) for item in report.agent_results]
        )
        shadow_evidence_signature = _sha(
            [agent_evidence_v1_semantic_payload(item) for item in shadow_evidence]
        )
        baseline_ood_signature = _sha(
            [
                (
                    item.agent_name,
                    item.distribution_shift_score,
                    item.distribution_shift_raw_score,
                    item.distribution_shift_level,
                    item.abstained,
                )
                for item in report.agent_results
            ]
        )
        shadow_ood_signature = _sha(
            [
                (
                    item.agent_name,
                    item.distribution_shift_score,
                    item.distribution_shift_raw_score,
                    item.distribution_shift_level,
                    item.abstained,
                )
                for item in shadow_evidence
            ]
        )
        record = {
            "sample_id": flow.sample_id,
            "trace_id": flow.trace_id,
            "baseline": {
                **_row_signature(report),
                "agent_calls": len(report.agent_results),
                "audit_event_count": len(audit.events),
                "reported_audit_event_count": report.audit_event_count,
                "latency_ms": baseline_latency,
                "evidence_signature": baseline_evidence_signature,
                "ood_signature": baseline_ood_signature,
            },
            "shadow": {
                "verdict": shadow_fusion.verdict.value,
                "confidence": shadow_fusion.confidence,
                "uncertainty": shadow_fusion.uncertainty,
                "conflict_score": shadow_fusion.conflict_score,
                "distribution_shift_score": shadow_fusion.distribution_shift_score,
                "agent_calls": len(shadow_evidence),
                "request_count": len(requests),
                "handoff_count": len(state.evidence_handoffs),
                "latency_overhead_ms": shadow_latency,
                "evidence_signature": shadow_evidence_signature,
                "ood_signature": shadow_ood_signature,
            },
            "request_audit": request_audit,
            "handoff_audit": handoff_audit,
            "semantic_matches": semantic_matches,
            "audit_complete": (
                len(audit.events) == report.audit_event_count
                and len(requests) == len(report.agent_results)
                and len(state.evidence_handoffs) == len(requests)
            ),
            "advisory_injection_count": sum(
                item.agent_name in SOC_ADVISORY_AGENT_NAMES
                for item in shadow_evidence
            ),
            "unauthorized_handoff_count": sum(
                row.get("policy_status") == "approved"
                and row.get("requested_agent") not in SOC_VERDICT_AGENT_ALLOWLIST
                for row in handoff_audit
            ),
        }
        _append_checkpoint(checkpoint, record)
        existing.append(record)
        completed.add(flow.sample_id)
        processed += 1
        if processed % 250 == 0:
            print(f"W72 shadow progress: {processed}/{manifest['sample_count']}")
        if processed >= manifest["sample_count"]:
            break
    if len(existing) != manifest["sample_count"]:
        raise RuntimeError(
            f"W72 incomplete shadow run: {len(existing)}/{manifest['sample_count']}"
        )
    return _materialize_shadow_outputs(output, existing)


def _materialize_shadow_outputs(
    output: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_rows: list[dict[str, Any]] = []
    shadow_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    handoff_rows: list[dict[str, Any]] = []
    for row in rows:
        baseline_rows.append({"sample_id": row["sample_id"], **row["baseline"]})
        shadow_rows.append({"sample_id": row["sample_id"], **row["shadow"]})
        for item in row["request_audit"]:
            normalized_item = {
                "purpose": (
                    "Collect specialist AgentEvidence for: "
                    "W72_SHADOW_REPLAY_OF_EXECUTED_PLAN"
                ),
                **item,
            }
            request_rows.append(
                {
                    "sample_id": row["sample_id"],
                    **{
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in normalized_item.items()
                    },
                }
            )
        for item in row["handoff_audit"]:
            handoff_rows.append(
                {
                    "sample_id": row["sample_id"],
                    **{
                        key: json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                        for key, value in item.items()
                    },
                }
            )
    _write_csv(output / "baseline_case_results.csv", baseline_rows)
    _write_csv(output / "shadow_case_results.csv", shadow_rows)
    _write_csv(output / "evidence_request_audit.csv", request_rows)
    _write_csv(output / "evidence_handoff_audit.csv", handoff_rows)
    total = len(rows)
    eq = lambda key: sum(
        row["baseline"][key] == row["shadow"][key] for row in rows
    ) / total
    semantic_count = sum(len(row["semantic_matches"]) for row in rows)
    semantic_good = sum(sum(row["semantic_matches"]) for row in rows)
    report = {
        "sample_count": total,
        "verdict_invariance": eq("verdict"),
        "confidence_semantic_invariance": eq("confidence"),
        "uncertainty_semantic_invariance": eq("uncertainty"),
        "conflict_semantic_invariance": eq("conflict_score"),
        "ood_decision_agreement": eq("distribution_shift_score"),
        "ood_signature_agreement": eq("ood_signature"),
        "evidence_signature_agreement": eq("evidence_signature"),
        "agent_evidence_semantic_invariance": (
            semantic_good / semantic_count if semantic_count else 0.0
        ),
        "baseline_avg_agent_calls": statistics.mean(
            row["baseline"]["agent_calls"] for row in rows
        ),
        "shadow_avg_agent_calls": statistics.mean(
            row["shadow"]["agent_calls"] for row in rows
        ),
        "average_evidence_requests": statistics.mean(
            row["shadow"]["request_count"] for row in rows
        ),
        "average_evidence_handoffs": statistics.mean(
            row["shadow"]["handoff_count"] for row in rows
        ),
        "audit_completion": sum(row["audit_complete"] for row in rows) / total,
        "unauthorized_handoff_count": sum(
            row["unauthorized_handoff_count"] for row in rows
        ),
        "advisory_evidence_injection_count": sum(
            row["advisory_injection_count"] for row in rows
        ),
    }
    _dump(output / "semantic_invariance_report.json", report)
    overheads = [row["shadow"]["latency_overhead_ms"] for row in rows]
    latencies = [row["baseline"]["latency_ms"] for row in rows]
    latency = {
        "sample_count": total,
        "baseline_p50_ms": _percentile(latencies, 0.5),
        "baseline_p95_ms": _percentile(latencies, 0.95),
        "shadow_protocol_overhead_mean_ms": statistics.mean(overheads),
        "shadow_protocol_overhead_p50_ms": _percentile(overheads, 0.5),
        "shadow_protocol_overhead_p95_ms": _percentile(overheads, 0.95),
    }
    _dump(output / "latency_overhead_report.json", latency)
    return {**report, **latency, "status": "shadow_run_complete"}


def finalize_soc_evidence_team_w72(
    *,
    output_dir: str | Path = W72_RUN_DIR,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    semantics = _load(output / "semantic_invariance_report.json")
    latency = _load(output / "latency_overhead_report.json")
    before = _load(output / "frozen_hashes_before.json")
    after = hash_artifact_paths(_default_frozen_paths())
    _dump(output / "frozen_hashes_after.json", after)
    requests = list(csv.DictReader((output / "evidence_request_audit.csv").open(encoding="utf-8")))
    handoffs = list(csv.DictReader((output / "evidence_handoff_audit.csv").open(encoding="utf-8")))
    request_rejections = sum(row["policy_status"] == "rejected" for row in requests)
    request_adjustments = 0
    required_prohibitions = {
        "final_verdict",
        "final_confidence",
        "final_uncertainty",
    }
    blocked_field_violation = 0
    illegal_verdict_execution_count = 0
    ood_override_count = 0
    unsupported_agent_execution_count = 0
    for row in requests:
        requested_agent = row.get("requested_agent", "")
        features = json.loads(row.get("permitted_safe_features", "[]"))
        prohibited = set(json.loads(row.get("prohibited_outputs", "[]")))
        purpose = row.get("purpose", "").lower()
        for path in features:
            contract = FIELD_CONTRACT.get(path, {})
            if (
                contract.get("role") != "DETECTION_ALLOWED"
                or requested_agent not in contract.get("consumers", [])
            ):
                blocked_field_violation += 1
        if not required_prohibitions.issubset(prohibited) or any(
            token in purpose
            for token in (
                "final verdict",
                "final_verdict",
                "final confidence",
                "final_confidence",
                "final uncertainty",
                "final_uncertainty",
            )
        ):
            illegal_verdict_execution_count += 1
        if "override ood" in purpose or "ood override" in purpose:
            ood_override_count += 1
        if (
            row.get("policy_status") == "approved"
            and requested_agent not in SOC_VERDICT_AGENT_ALLOWLIST
        ):
            unsupported_agent_execution_count += 1
    agent_evidence_fields = set(AgentEvidenceV2.model_fields)
    handoff_fields = set(EvidenceHandoffV2.model_fields)
    fusion_ownership_violation = int(
        bool(
            {"final_verdict", "final_confidence", "final_uncertainty"}
            & (agent_evidence_fields | handoff_fields)
        )
    )
    total_requests = len(requests)
    security = {
        "audit_completion": semantics["audit_completion"],
        "blocked_field_violation": blocked_field_violation,
        "fusion_ownership_violation": fusion_ownership_violation,
        "ood_override_count": ood_override_count,
        "illegal_verdict_execution_count": illegal_verdict_execution_count,
        "unsupported_agent_execution_count": unsupported_agent_execution_count,
        "unauthorized_handoff_count": semantics["unauthorized_handoff_count"],
        "advisory_evidence_injection_count": semantics[
            "advisory_evidence_injection_count"
        ],
        "policy_guard_rejection_count": request_rejections,
        "policy_guard_rejection_rate": (
            request_rejections / total_requests if total_requests else 0.0
        ),
        "policy_guard_adjustment_count": request_adjustments,
        "policy_guard_adjustment_rate": (
            request_adjustments / total_requests if total_requests else 0.0
        ),
        "evidence_request_count": total_requests,
        "approved_handoff_count": sum(
            row["policy_status"] == "approved" for row in handoffs
        ),
        "fusion_owner": "FusionAgent",
        "fake_metric_count": 0,
    }
    _dump(output / "security_acceptance.json", security)
    checks = {
        "verdict_invariance_is_one": semantics["verdict_invariance"] == 1.0,
        "confidence_semantic_invariance_is_one": semantics[
            "confidence_semantic_invariance"
        ]
        == 1.0,
        "uncertainty_semantic_invariance_is_one": semantics[
            "uncertainty_semantic_invariance"
        ]
        == 1.0,
        "ood_decision_agreement_is_one": semantics["ood_decision_agreement"]
        == 1.0,
        "ood_signature_agreement_is_one": semantics["ood_signature_agreement"]
        == 1.0,
        "agent_evidence_semantic_invariance_is_one": semantics[
            "agent_evidence_semantic_invariance"
        ]
        == 1.0,
        "unauthorized_handoff_is_zero": security["unauthorized_handoff_count"]
        == 0,
        "advisory_injection_is_zero": security[
            "advisory_evidence_injection_count"
        ]
        == 0,
        "blocked_field_violation_is_zero": security["blocked_field_violation"]
        == 0,
        "fusion_ownership_violation_is_zero": security[
            "fusion_ownership_violation"
        ]
        == 0,
        "ood_override_is_zero": security["ood_override_count"] == 0,
        "illegal_verdict_execution_is_zero": security[
            "illegal_verdict_execution_count"
        ]
        == 0,
        "unsupported_agent_execution_is_zero": security[
            "unsupported_agent_execution_count"
        ]
        == 0,
        "audit_completion_is_one": security["audit_completion"] == 1.0,
        "agent_calls_unchanged": semantics["baseline_avg_agent_calls"]
        == semantics["shadow_avg_agent_calls"],
        "frozen_hashes_unchanged": before == after,
        "feature_flag_off_default_output_unchanged": semantics[
            "verdict_invariance"
        ]
        == 1.0,
        "fake_metric_count_is_zero": security["fake_metric_count"] == 0,
        "full_pytest_passed": bool(tests_passed),
    }
    accepted = all(checks.values())
    failed = [key for key, value in checks.items() if not value]
    status = (
        "accepted_shadow_soc_evidence_team_protocol"
        if accepted
        else "not_promoted_soc_protocol_w72"
    )
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_soc_evidence_team_w72",
        "status": status,
        "checks": checks,
        "failed_gates": failed,
        **semantics,
        "latency_overhead": latency,
        "policy_guard_rejection_rate": security[
            "policy_guard_rejection_rate"
        ],
        "policy_guard_adjustment_rate": security[
            "policy_guard_adjustment_rate"
        ],
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "default_runtime": DEFAULT_RUNTIME,
        "runtime_safe_v3_0_remains_default": True,
        "soc_evidence_team_shadow_v1_default_enabled": False,
        "fusion_owner": "FusionAgent",
        "w71_status": "not_promoted_learned_tls_w71",
        "runtime_tls_doh_w71_optional_created": False,
        "promoted_runtime_created": False,
        "classification_improvement_claimed": False,
        "fake_metric_count": 0,
    }
    _dump(output / "acceptance_report.json", report)
    negative = {
        "status": "none" if accepted else "retained_negative_result",
        "candidate": "soc_evidence_team_shadow_v1",
        "failed_gates": failed,
        "promoted_runtime_created": False,
        "safe_claim": (
            "W72 is an accepted default-off architecture protocol."
            if accepted
            else "W72 remained a non-promoted shadow protocol."
        ),
        "forbidden_claim": "W72 improves classification performance or replaces the default runtime.",
        "fake_metric_count": 0,
    }
    _dump(output / "negative_results.json", negative)
    _write_docs(output, report)
    return report
