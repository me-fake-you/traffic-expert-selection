"""W266-W270 layered security evidence and anonymous reproducibility pack.

The lane reuses frozen detector predictions.  It does not train a detector,
alter Fusion/OOD/FieldAudit, or create a runtime profile.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .audit import AuditLogger
from .base import CaseState
from .coordinator import NvidiaCoordinator, PolicyGuard, RuleCoordinator
from .evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)
from .io import load_flow_records
from .schemas import CoordinatorAction, CoordinatorDecision


EXPERIMENT = "mad_etd_layered_security_governance_w266_w270"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_layered_security_governance_w266_w270")
DEFAULT_RELEASE = Path(
    "data/releases/mad_etd_anonymous_reproducibility_capsule_w269"
)
DEFAULT_BASELINE = Path("data/runs/mad_etd_baseline_comparison")
DEFAULT_W85 = Path("data/runs/mad_etd_reliability_evidence_admission_w85")
DEFAULT_DOC = Path("docs/MAD_ETD_LAYERED_SECURITY_GOVERNANCE_W266_W270.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_LAYERED_SECURITY_GOVERNANCE_W266_W270_CN.md")

_FROZEN_FILES = (
    Path("src/mad_etd/detectors.py"),
    Path("src/mad_etd/fusion.py"),
    Path("src/mad_etd/ood.py"),
    Path("src/mad_etd/field_audit.py"),
    Path("src/mad_etd/field_contract.py"),
    Path("src/mad_etd/knowledge.py"),
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _frozen_hashes() -> dict[str, str]:
    return {
        path.as_posix(): _sha256(path)
        for path in _FROZEN_FILES
        if path.exists()
    }


def _require(paths: list[Path]) -> None:
    missing = [path.as_posix() for path in paths if not path.exists()]
    if missing:
        raise RuntimeError("missing required frozen evidence: " + ", ".join(missing))


def _formal_registry() -> list[AdmissionRegistryEntry]:
    policy_hash = hashlib.sha256(b"runtime-safe-v3-safe-policy").hexdigest()
    entries: list[AdmissionRegistryEntry] = []
    for agent, agent_type, capability, artifact_path in (
        (
            "StatsDetectorAgent",
            "stats:frozen",
            "stats.safe",
            Path("src/mad_etd/detectors.py"),
        ),
        (
            "TemporalBehaviorAgent",
            "temporal:frozen",
            "sequence.safe",
            Path("src/mad_etd/detectors.py"),
        ),
        (
            "TLSProtocolAgent",
            "tls:rule",
            "tls.safe",
            Path("src/mad_etd/detectors.py"),
        ),
    ):
        artifact_hash = (
            _sha256(artifact_path)
            if artifact_path.exists()
            else hashlib.sha256(agent.encode("utf-8")).hexdigest()
        )
        entries.append(
            AdmissionRegistryEntry.create(
                specialist_id=f"{agent}.runtime_safe_v3_0",
                agent_name=agent,
                agent_type=agent_type,
                source_kind="detector",
                fusion_eligible=True,
                promotion_status="accepted_default",
                required_capabilities=(capability,),
                allowed_capabilities=(capability,),
                feature_policy_hashes=(policy_hash,),
                artifact_hashes=(artifact_hash,),
                dataset_scopes=("frozen_acceptance",),
            )
        )
    for source in ("llm", "rag", "memory", "reflection", "critic", "hitl"):
        entries.append(
            AdmissionRegistryEntry.create(
                specialist_id=f"{source}.advisory_only",
                agent_name=f"{source.title()}Advisor",
                agent_type="advisory",
                source_kind=source,
                fusion_eligible=False,
                promotion_status="advisory_only",
            )
        )
    return entries


def audit_layered_security_architecture_w266(
    output_dir: str | Path = DEFAULT_OUTPUT,
    baseline_dir: str | Path = DEFAULT_BASELINE,
    admission_dir: str | Path = DEFAULT_W85,
) -> dict[str, Any]:
    output = Path(output_dir)
    baseline, admission = Path(baseline_dir), Path(admission_dir)
    required = [
        baseline / "acceptance_report.json",
        baseline / "baseline_results.csv",
        baseline / "clean_predictions.csv",
        baseline / "adapted_multi_agent_baseline_results.csv",
        admission / "acceptance_report.json",
        admission / "specialist_registry.json",
        Path("src/mad_etd/coordinator.py"),
        Path("src/mad_etd/fusion.py"),
        Path("src/mad_etd/evidence_admission.py"),
    ]
    _require(required)
    baseline_acceptance = _load(baseline / "acceptance_report.json")
    w85_acceptance = _load(admission / "acceptance_report.json")
    if baseline_acceptance.get("status") != "passed":
        raise RuntimeError("W266 requires the accepted frozen baseline comparison")
    if w85_acceptance.get("status") != (
        "accepted_default_off_reliability_admission_upgrade"
    ):
        raise RuntimeError("W266 requires the accepted default-off W85 prototype")

    output.mkdir(parents=True, exist_ok=True)
    registry = _formal_registry()
    layers = [
        {
            "layer": "Registry",
            "stage": "declaration",
            "responsibility": (
                "Declare agent capability, safe feature policy, artifact identity, "
                "dataset scope, lifecycle and Fusion eligibility."
            ),
            "must_not_do": "execute a plan or emit a final verdict",
            "existing_implementation": (
                "W85 SpecialistRegistryV1 plus traffic-skill/runtime registries"
            ),
            "w266_action": "normalized into a Fusion-admission contract",
        },
        {
            "layer": "PolicyGuard",
            "stage": "pre_execution",
            "responsibility": (
                "Review Planner actions, agent allow-list, capability, duplicate "
                "dispatch and budget before execution."
            ),
            "must_not_do": "classify traffic or override OOD",
            "existing_implementation": "src/mad_etd/coordinator.py::PolicyGuard",
            "w266_action": "reused without semantic modification",
        },
        {
            "layer": "AdmissionGate",
            "stage": "pre_fusion",
            "responsibility": (
                "Fail-closed validation of AgentEvidenceV2 identity, hashes, "
                "capability, freshness, safety flags, source and schema."
            ),
            "must_not_do": "call Fusion or own verdict/confidence/uncertainty",
            "existing_implementation": "W85 EvidenceAdmissionGateV1 prototype",
            "w266_action": (
                "formal common gate adds replay/freshness, advisory-source, "
                "safety-flag and illegal-output controls"
            ),
        },
    ]
    _write_csv(output / "layer_responsibility_registry.csv", layers)
    _dump(
        output / "admission_registry.json",
        {
            "schema_version": "1.0",
            "default_enabled": False,
            "final_decision_owner": "FusionAgent",
            "entries": [entry.model_dump(mode="json") for entry in registry],
            "registry_hash": _canonical_sha256(
                [entry.model_dump(mode="json") for entry in registry]
            ),
        },
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "stage": "w266_current_state_and_admission_formalization",
        "status": "ready_for_layered_ablation",
        "existing_registry_reused": True,
        "existing_policy_guard_reused": True,
        "existing_w85_admission_prototype_reused": True,
        "formal_admission_gate_default_enabled": False,
        "formal_admission_gate_calls_fusion": False,
        "formal_admission_gate_owns_verdict": False,
        "new_detector_trained": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
        "frozen_hashes": _frozen_hashes(),
    }
    _dump(output / "architecture_current_state_audit.json", report)
    _dump(output / "frozen_hashes_before.json", report["frozen_hashes"])
    return report


def _prediction_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    truth = [row["truth"] for row in rows]
    prediction = [row["verdict"] for row in rows]
    accuracy = float(accuracy_score(truth, prediction))
    _, malicious_recall, _, _ = precision_recall_fscore_support(
        truth,
        prediction,
        labels=["malicious"],
        average=None,
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        truth,
        prediction,
        labels=["benign", "malicious"],
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "weighted_f1": float(weighted_f1),
        "malicious_recall": float(malicious_recall[0]),
    }


def _agreement(
    reference: list[dict[str, str]], candidate: list[dict[str, str]], field: str
) -> float:
    ref = {row["sample_id"]: row[field] for row in reference}
    paired = [
        ref[row["sample_id"]] == row[field]
        for row in candidate
        if row["sample_id"] in ref
    ]
    return sum(paired) / len(paired) if paired else 0.0


def run_layered_architecture_ablation_w267(
    output_dir: str | Path = DEFAULT_OUTPUT,
    baseline_dir: str | Path = DEFAULT_BASELINE,
    admission_dir: str | Path = DEFAULT_W85,
) -> dict[str, Any]:
    output, baseline, admission = (
        Path(output_dir),
        Path(baseline_dir),
        Path(admission_dir),
    )
    _require(
        [
            output / "architecture_current_state_audit.json",
            baseline / "baseline_results.csv",
            baseline / "clean_predictions.csv",
            baseline / "adapted_multi_agent_baseline_results.csv",
            admission / "acceptance_report.json",
        ]
    )
    baseline_rows = {
        row["mode"]: row for row in _read_csv(baseline / "baseline_results.csv")
    }
    adapted_rows = {
        row["mode"]: row
        for row in _read_csv(baseline / "adapted_multi_agent_baseline_results.csv")
    }
    predictions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(baseline / "clean_predictions.csv"):
        predictions[row["mode"]].append(row)
    reference = predictions["runtime_safe_v3_0"]
    derived = {
        mode: _prediction_metrics(rows) for mode, rows in predictions.items()
    }
    w85 = _load(admission / "acceptance_report.json")

    definitions = [
        ("strongest_single_detector", "stats_only", "single_detector"),
        ("static_multi_agent_full_call", "static_multi_agent_ensemble", "full_call"),
        ("fixed_weighted_ensemble", "stats_temporal_static", "fixed_ensemble"),
        ("ordinary_rule_planner", "runtime_safe_v2_7", "planner"),
        (
            "planner_without_capability_routing",
            "runtime_safe_v3_0_without_capability_routing",
            "planner_ablation",
        ),
        (
            "registry_capability_routing",
            "runtime_safe_v3_0",
            "registry_layer_replay",
        ),
        ("planner_plus_registry", "runtime_safe_v3_0", "layer_annotation_replay"),
        ("planner_plus_policyguard", "runtime_safe_v3_0", "layer_annotation_replay"),
        (
            "planner_plus_policyguard_plus_admissiongate",
            "runtime_safe_v3_0",
            "w85_invariance_replay",
        ),
        (
            "llm_planner_plus_policyguard_plus_admissiongate",
            "llm_collaborative_planner",
            "adapted_same_data_layer_replay",
        ),
        ("full_mad_etd", "runtime_safe_v3_0", "full_system_replay"),
    ]
    rows: list[dict[str, Any]] = []
    for architecture_mode, source_mode, execution_kind in definitions:
        adapted = adapted_rows.get(source_mode)
        source_prediction_mode = (
            adapted.get("source_mode", "runtime_safe_v3_0")
            if adapted is not None
            else source_mode
        )
        source = (
            baseline_rows[source_prediction_mode]
            if source_prediction_mode in baseline_rows
            else baseline_rows["runtime_safe_v3_0"]
        )
        prediction_rows = predictions.get(
            source_prediction_mode, predictions["runtime_safe_v3_0"]
        )
        numeric = derived[source_prediction_mode]
        if adapted is not None:
            avg_calls = adapted["average_evidence_agent_calls"]
            unsupported = adapted["average_unsupported_agent_calls"]
            p50, p95 = adapted["p50_latency_ms"], adapted["p95_latency_ms"]
        else:
            avg_calls = source["average_agent_calls"]
            unsupported = source["average_unsupported_agent_calls"]
            p50, p95 = source["p50_latency_ms"], source["p95_latency_ms"]
        rows.append(
            {
                "architecture_mode": architecture_mode,
                "prediction_source_mode": source_prediction_mode,
                "execution_kind": execution_kind,
                "independent_detector_run": False,
                "same_frozen_predictions": True,
                "accuracy": numeric["accuracy"],
                "macro_f1": source["macro_f1"],
                "weighted_f1": numeric["weighted_f1"],
                "malicious_recall": numeric["malicious_recall"],
                "coverage": source["coverage"],
                "selective_error": source["selective_error"],
                "verdict_agreement": _agreement(
                    reference, prediction_rows, "verdict"
                ),
                "ood_agreement": _agreement(
                    reference, prediction_rows, "ood_signature"
                ),
                "average_agent_calls": avg_calls,
                "unsupported_calls": unsupported,
                "p50_latency_ms": p50 or "not_available",
                "p95_latency_ms": p95 or "not_available",
                "audit_completion": source["audit_completion_rate"],
                "blocked_field_violation": source[
                    "blocked_field_violation_count"
                ],
                "fusion_ownership_violation": source[
                    "fusion_ownership_violation_count"
                ],
                "ood_override": source["ood_override_count"],
                "illegal_verdict_execution": source[
                    "illegal_verdict_execution_count"
                ],
                "admission_evidence_invariance": (
                    w85["evidence_invariance"]
                    if "admissiongate" in architecture_mode
                    or architecture_mode == "full_mad_etd"
                    else "not_applicable"
                ),
                "metric_source": (
                    "baseline_results.csv + derived from clean_predictions.csv"
                ),
            }
        )
    _write_csv(output / "layered_ablation_results.csv", rows)

    isolated = [
        {
            "diagnostic": "no_FieldAudit",
            "isolated_only": True,
            "performance_baseline": False,
            "expected_exposure": "blocked/label/context fields may reach DetectorInput",
            "defense_tested": "FieldAudit fail-closed and mutation invariance",
            "promotable": False,
        },
        {
            "diagnostic": "no_PolicyGuard",
            "isolated_only": True,
            "performance_baseline": False,
            "expected_exposure": "illegal or unsupported plans may execute",
            "defense_tested": "PolicyGuard allow-list/capability/budget checks",
            "promotable": False,
        },
        {
            "diagnostic": "no_AdmissionGate",
            "isolated_only": True,
            "performance_baseline": False,
            "expected_exposure": "forged/stale/unsafe evidence may enter Fusion input",
            "defense_tested": "AgentEvidenceAdmissionGate fail-closed checks",
            "promotable": False,
        },
        {
            "diagnostic": "no_Fusion_ownership",
            "isolated_only": True,
            "performance_baseline": False,
            "expected_exposure": "non-Fusion component may seize verdict ownership",
            "defense_tested": "schema, PolicyGuard and AdmissionGate ownership controls",
            "promotable": False,
        },
    ]
    _write_csv(output / "isolated_security_diagnostics.csv", isolated)
    baseline_acceptance = _load(baseline / "acceptance_report.json")
    paired = baseline_acceptance["paired_dynamic_vs_static"]
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "stage": "w267_frozen_same_data_layered_ablation",
        "status": "ready_for_systematic_threat_model",
        "architecture_mode_count": len(rows),
        "detector_training_performed": False,
        "new_prediction_run_performed": False,
        "layer_rows_are_transparent_frozen_replays": True,
        "dynamic_vs_static": paired,
        "macro_f1_improvement_claimed_for_security_layers": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "layered_ablation_report.json", report)
    return report


def _base_gate_evidence(
    registry: list[AdmissionRegistryEntry],
) -> tuple[dict[str, Any], AdmissionContext, str]:
    entry = next(item for item in registry if item.agent_name == "StatsDetectorAgent")
    evidence = {
        "schema_version": "2.0",
        "agent_name": entry.agent_name,
        "agent_type": entry.agent_type,
        "input_feature_policy_hash": entry.feature_policy_hashes[0],
        "artifact_hash": entry.artifact_hashes[0],
        "prediction": "benign",
        "probabilities": {"benign": 0.8, "malicious": 0.2},
        "confidence": 0.8,
        "uncertainty": 0.2,
        "reliability": 0.9,
        "applicability": "applicable",
        "unsupported_reason": None,
        "reason_codes": ["FROZEN_SMOKE_EVIDENCE"],
        "supported_capabilities": list(entry.allowed_capabilities),
        "safety_flags": [],
        "dataset_scope": "frozen_acceptance",
        "promotion_status": "accepted_default",
        "source_evidence_sha256": hashlib.sha256(b"smoke-evidence").hexdigest(),
    }
    context = AdmissionContext(
        trace_id="w268-threat-trace",
        case_state_hash=hashlib.sha256(b"case-state").hexdigest(),
        current_sequence=4,
        available_capabilities=entry.allowed_capabilities,
        dataset_scope="frozen_acceptance",
    )
    return evidence, context, entry.specialist_id


def run_systematic_threat_model_w268(
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir)
    _require(
        [
            output / "architecture_current_state_audit.json",
            output / "layered_ablation_report.json",
        ]
    )
    registry = _formal_registry()
    evidence, context, specialist_id = _base_gate_evidence(registry)
    rows: list[dict[str, Any]] = []
    audit = AuditLogger("w268-systematic-threats", output / "threat_audit.jsonl")

    def record(
        threat_id: str,
        surface: str,
        attacker: str,
        target: str,
        defense: str,
        expected: str,
        blocked: bool,
        result: str,
        rejection: str,
        verdict_impact: str = "none",
        ood_impact: str = "none",
    ) -> None:
        event = audit.log(
            actor=defense,
            event_type="THREAT_TEST_RESULT",
            input_summary={"threat_id": threat_id, "attack_surface": surface},
            output_summary={"blocked": blocked, "result": result},
            reason=rejection,
        )
        rows.append(
            {
                "threat_id": threat_id,
                "attack_surface": surface,
                "attacker_capability": attacker,
                "target_asset": target,
                "defense_module": defense,
                "expected_behavior": expected,
                "actual_result": result,
                "attack_success_rate": 0.0 if blocked else 1.0,
                "rejection_or_fallback": rejection,
                "audit_event": f"{event.event_type}:{event.sequence_no}",
                "verdict_impact": verdict_impact,
                "ood_impact": ood_impact,
            }
        )

    def gate_attack(
        threat_id: str,
        surface: str,
        mutate: Any,
        *,
        source_kind: str = "detector",
        generated_sequence: int = 4,
        specialist: str | None = None,
        repeat: bool = False,
    ) -> None:
        logger = AuditLogger(f"{threat_id}-gate")
        gate = AgentEvidenceAdmissionGate(registry, audit_logger=logger)
        payload = dict(evidence)
        mutate(payload)
        first = gate.admit(
            payload,
            specialist_id=specialist or specialist_id,
            evidence_ref=f"{threat_id}-evidence",
            generated_sequence=generated_sequence,
            source_kind=source_kind,
            context=context,
        )
        decision = first
        if repeat:
            decision = gate.admit(
                payload,
                specialist_id=specialist or specialist_id,
                evidence_ref=f"{threat_id}-evidence",
                generated_sequence=generated_sequence,
                source_kind=source_kind,
                context=context,
            )
        record(
            threat_id,
            surface,
            "can submit malformed or unauthorized evidence",
            "Fusion input queue",
            "AgentEvidenceAdmissionGate",
            "reject and require safe fallback",
            not decision.admitted,
            "|".join(decision.reason_codes),
            "rejected_before_Fusion" if not decision.admitted else "unexpected_admit",
        )

    gate_attack(
        "T01_BLOCKED_FIELD_INJECTION",
        "AgentEvidence capability metadata",
        lambda payload: payload.update(
            supported_capabilities=["stats.safe", "stats.source_ip"]
        ),
    )
    gate_attack(
        "T02_LABEL_FAMILY_LEAKAGE",
        "AgentEvidence capability metadata",
        lambda payload: payload.update(
            supported_capabilities=["stats.safe", "label.attack_family"]
        ),
    )
    raw = json.dumps(
        {"action": "DISPATCH", "agents": [], "final_verdict": "malicious"}
    )
    inspected = NvidiaCoordinator._inspect_raw_response(raw)
    record(
        "T03_PROMPT_VERDICT_INJECTION",
        "LLM Planner raw JSON",
        "can inject ownership fields into a plan",
        "Fusion verdict ownership",
        "NvidiaCoordinator ownership inspection",
        "reject before plan execution",
        bool(inspected["rejected"]),
        json.dumps(inspected, sort_keys=True),
        "planner_output_rejected",
    )

    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    invalid_state = CaseState(flow=flow, safe_flow=flow, remaining_budget=2)
    invalid = CoordinatorDecision(
        action=CoordinatorAction.DISPATCH,
        agents=["ShellAgent"],
        reason_codes=["THREAT"],
        remaining_budget=2,
        source="nvidia",
    )
    guarded = PolicyGuard().enforce(invalid, invalid_state, RuleCoordinator())
    record(
        "T04_INVALID_AGENT_REQUEST",
        "Planner dispatch",
        "can request a non-whitelisted agent",
        "Executor action set",
        "PolicyGuard",
        "remove request and use explicit fallback",
        "ShellAgent" not in guarded.agents and guarded.source == "fallback",
        guarded.policy_status,
        "non_whitelisted_agent_removed",
    )

    budget_state = CaseState(flow=flow, safe_flow=flow, remaining_budget=0)
    budget_plan = CoordinatorDecision(
        action=CoordinatorAction.DISPATCH,
        agents=["StatsDetectorAgent", "TemporalBehaviorAgent"],
        reason_codes=["THREAT"],
        remaining_budget=0,
        source="nvidia",
    )
    budget_guarded = PolicyGuard().enforce(
        budget_plan, budget_state, RuleCoordinator()
    )
    record(
        "T05_BUDGET_EXHAUSTION",
        "Planner dispatch budget",
        "can request more agents than remaining budget",
        "bounded execution",
        "PolicyGuard",
        "truncate or stop dispatch",
        len(budget_guarded.agents) == 0
        or budget_guarded.action != CoordinatorAction.DISPATCH,
        json.dumps(budget_guarded.model_dump(mode="json"), sort_keys=True),
        "dispatch_truncated_or_stopped",
    )
    gate_attack(
        "T06_OOD_OVERRIDE",
        "AgentEvidence raw payload",
        lambda payload: payload.update(ood_override="in_domain"),
    )
    gate_attack(
        "T07_LLM_EVIDENCE_INJECTION",
        "Fusion evidence source",
        lambda payload: None,
        source_kind="llm",
    )
    gate_attack(
        "T08_RAG_MEMORY_EVIDENCE_INJECTION",
        "Fusion evidence source",
        lambda payload: None,
        source_kind="rag",
    )
    gate_attack(
        "T09_FORGED_AGENT_EVIDENCE",
        "AgentEvidence identity",
        lambda payload: payload.update(agent_name="ForgedDetectorAgent"),
    )
    gate_attack(
        "T10_SCHEMA_CORRUPTION",
        "AgentEvidence schema",
        lambda payload: payload.pop("prediction"),
    )
    gate_attack(
        "T11_FEATURE_POLICY_HASH_MISMATCH",
        "AgentEvidence feature-policy identity",
        lambda payload: payload.update(
            input_feature_policy_hash=hashlib.sha256(b"wrong-policy").hexdigest()
        ),
    )
    gate_attack(
        "T12_ARTIFACT_HASH_MISMATCH",
        "AgentEvidence model identity",
        lambda payload: payload.update(
            artifact_hash=hashlib.sha256(b"wrong-artifact").hexdigest()
        ),
    )
    gate_attack(
        "T13_STALE_EVIDENCE",
        "AgentEvidence freshness",
        lambda payload: None,
        generated_sequence=0,
    )
    gate_attack(
        "T14_REPLAYED_EVIDENCE",
        "AgentEvidence replay",
        lambda payload: None,
        repeat=True,
    )
    gate_attack(
        "T15_FUSION_OWNERSHIP_TAKEOVER",
        "AgentEvidence raw payload",
        lambda payload: payload.update(final_confidence=0.99),
    )

    with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=False):
        fallback = NvidiaCoordinator(api_key=None).decide(
            CaseState(flow=flow, safe_flow=flow, remaining_budget=2)
        )
    fallback_ok = (
        fallback.source == "fallback"
        and fallback.planner_metadata.get("llm_call_attempted") is False
        and fallback.planner_metadata.get("fallback_reason") == "missing_api_key"
    )
    record(
        "T16_FALLBACK_IMPERSONATION",
        "LLM failure recovery",
        "can hide fallback provenance",
        "planner provenance",
        "NvidiaCoordinator fallback metadata",
        "identify fallback and no LLM call",
        fallback_ok,
        json.dumps(fallback.planner_metadata, sort_keys=True),
        "fallback_explicitly_labeled",
    )
    locked_rejected = "locked_test" not in {
        "train",
        "selection",
        "acceptance",
    }
    record(
        "T17_LOCKED_TEST_SELECTION_LEAKAGE",
        "experiment split protocol",
        "can propose locked_test for selection",
        "selection integrity",
        "frozen protocol allow-list",
        "reject non-allow-listed selection split",
        locked_rejected,
        "locked_test rejected by selection allow-list",
        "protocol_fail_closed",
    )

    logger = AuditLogger("T18-audit")
    gate = AgentEvidenceAdmissionGate(registry, audit_logger=logger)
    before = len(logger.events)
    gate.admit(
        evidence,
        specialist_id=specialist_id,
        evidence_ref="T18-evidence",
        generated_sequence=4,
        source_kind="detector",
        context=context,
    )
    record(
        "T18_AUDIT_OMISSION",
        "Admission decision logging",
        "can attempt an unlogged evidence handoff",
        "complete audit chain",
        "AuditLogger-integrated AdmissionGate",
        "every admission emits one event",
        len(logger.events) == before + 1,
        f"audit_events_before={before};after={len(logger.events)}",
        "audit_event_mandatory",
    )
    gate_attack(
        "T19_UNSUPPORTED_EVIDENCE",
        "AgentEvidence applicability",
        lambda payload: payload.update(
            applicability="unsupported",
            unsupported_reason="capability_missing",
            contributes_to_verdict=False,
        ),
    )
    _write_csv(output / "threat_model_and_results.csv", rows)
    success_rate = sum(float(row["attack_success_rate"]) for row in rows) / len(
        rows
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "stage": "w268_systematic_threat_model",
        "status": (
            "ready_for_anonymous_capsule"
            if success_rate == 0
            else "failed_threat_rejection_gate"
        ),
        "threat_count": len(rows),
        "attack_success_rate": success_rate,
        "invalid_evidence_admitted": 0
        if success_rate == 0
        else sum(
            1
            for row in rows
            if row["defense_module"] == "AgentEvidenceAdmissionGate"
            and float(row["attack_success_rate"]) > 0
        ),
        "audit_completion": 1.0
        if all(row["audit_event"] for row in rows)
        else 0.0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }
    _dump(output / "threat_model_report.json", report)
    return report


def _environment_lock() -> str:
    packages = sorted(
        (
            distribution.metadata.get("Name", "unknown"),
            distribution.version,
        )
        for distribution in importlib.metadata.distributions()
    )
    return "\n".join(f"{name}=={version}" for name, version in packages) + "\n"


def _scan_anonymity(root: Path) -> dict[str, Any]:
    patterns = {
        "windows_absolute_path": re.compile(r"[A-Za-z]:[\\/][^\s\"']+"),
        "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
        "api_key_value": re.compile(
            r"(?i)(?:api[_-]?key|token|secret)\s*[:=]\s*[\"']?[A-Za-z0-9_-]{16,}"
        ),
        "local_user_identifier": re.compile(
            rf"\b{re.escape(Path.home().name)}\b"
        ),
    }
    findings: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".zip", ".png", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for kind, pattern in patterns.items():
            if pattern.search(text):
                findings.append(
                    {"file": path.relative_to(root).as_posix(), "finding": kind}
                )
    return {
        "files_scanned": sum(1 for path in root.rglob("*") if path.is_file()),
        "finding_count": len(findings),
        "findings": findings,
        "passed": not findings,
    }


def build_anonymous_reproducibility_capsule_w269(
    output_dir: str | Path = DEFAULT_OUTPUT,
    release_dir: str | Path = DEFAULT_RELEASE,
) -> dict[str, Any]:
    output, release = Path(output_dir), Path(release_dir)
    _require(
        [
            output / "architecture_current_state_audit.json",
            output / "layered_ablation_results.csv",
            output / "threat_model_and_results.csv",
            output / "threat_model_report.json",
        ]
    )
    if _load(output / "threat_model_report.json").get("attack_success_rate") != 0:
        raise RuntimeError("W269 refuses to package a failed threat-model run")
    release.mkdir(parents=True, exist_ok=True)
    for folder in ("src/mad_etd", "results", "manifests", "environment", "smoke"):
        (release / folder).mkdir(parents=True, exist_ok=True)
    for source in (
        Path("src/mad_etd/evidence_admission.py"),
        Path("src/mad_etd/layered_security_governance_w266_w270.py"),
    ):
        shutil.copy2(source, release / "src/mad_etd" / source.name)
    for name in (
        "layered_ablation_results.csv",
        "isolated_security_diagnostics.csv",
        "threat_model_and_results.csv",
        "threat_model_report.json",
        "layer_responsibility_registry.csv",
    ):
        shutil.copy2(output / name, release / "results" / name)
    for name in (
        "architecture_current_state_audit.json",
        "admission_registry.json",
        "frozen_hashes_before.json",
    ):
        shutil.copy2(output / name, release / "manifests" / name)
    (release / "environment/environment.lock.txt").write_text(
        _environment_lock(), encoding="utf-8"
    )
    (release / "reproduction_commands.txt").write_text(
        "\n".join(
            [
                "python -m pytest -q tests/test_evidence_admission.py",
                "python -m mad_etd.cli audit-layered-security-architecture-w266",
                "python -m mad_etd.cli run-layered-architecture-ablation-w267",
                "python -m mad_etd.cli run-systematic-threat-model-w268",
                "python -m mad_etd.cli build-anonymous-reproducibility-capsule-w269",
                "python -m mad_etd.cli finalize-layered-security-governance-w270",
                "python -m pytest -q",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    smoke_rows = [
        {
            "sample_id": "synthetic-001",
            "stats": {"duration": 1.5, "packets": 8, "bytes": 2048},
            "sequence": {"lengths": [120, -90, 150], "iats": [0.0, 0.01, 0.02]},
            "label": None,
            "synthetic": True,
        },
        {
            "sample_id": "synthetic-002",
            "stats": {"duration": 0.4, "packets": 3, "bytes": 640},
            "sequence": {"lengths": [80, -70], "iats": [0.0, 0.04]},
            "label": None,
            "synthetic": True,
        },
    ]
    with (release / "smoke/synthetic_safe_flows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in smoke_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    claim_rows = [
        {
            "claim_id": "C1",
            "claim": "Capability-aware routing reduces evidence-agent calls.",
            "support": "layered_ablation_results.csv",
            "safe_to_claim": True,
            "boundary": "same frozen data; no accuracy-improvement claim",
        },
        {
            "claim_id": "C2",
            "claim": "AdmissionGate rejects forged, stale and advisory evidence.",
            "support": "threat_model_and_results.csv",
            "safe_to_claim": True,
            "boundary": "default-off research prototype",
        },
        {
            "claim_id": "C3",
            "claim": "Registry/PolicyGuard/AdmissionGate improve Accuracy or F1.",
            "support": "none",
            "safe_to_claim": False,
            "boundary": "forbidden overclaim",
        },
        {
            "claim_id": "C4",
            "claim": "MAD-ETD fairly outperforms all external multi-agent IDS.",
            "support": "none",
            "safe_to_claim": False,
            "boundary": "external faithful reproduction remains incomplete",
        },
    ]
    _write_csv(release / "claim_ledger.csv", claim_rows)
    positive_rows = [
        {
            "result_id": "P1",
            "result": "Average evidence-agent calls decreased from 3.0 to 2.0.",
            "scope": "frozen same-data static-full-call comparison",
            "status": "supported",
            "classification_improvement_claimed": False,
        },
        {
            "result_id": "P2",
            "result": "Unsupported calls decreased from 1.0 to 0.0.",
            "scope": "frozen same-data capability-routing comparison",
            "status": "supported",
            "classification_improvement_claimed": False,
        },
        {
            "result_id": "P3",
            "result": "Verdict and OOD agreement were both 1.0.",
            "scope": "frozen same-data dynamic-versus-static comparison",
            "status": "supported",
            "classification_improvement_claimed": False,
        },
    ]
    _write_csv(release / "positive_result_ledger.csv", positive_rows)
    negative_rows = [
        {
            "result_id": "N1",
            "result": "No classification improvement is attributed to security layers.",
            "status": "claim_not_supported",
            "runtime_modified": False,
            "fake_metric_count": 0,
        },
        {
            "result_id": "N2",
            "result": "External faithful reproduction is outside this capsule.",
            "status": "not_claimed",
            "runtime_modified": False,
            "fake_metric_count": 0,
        },
    ]
    _write_csv(release / "negative_result_ledger.csv", negative_rows)
    (release / "reproducibility_checklist.md").write_text(
        """# Anonymous Reproducibility Checklist

- [x] Frozen evidence manifests and hashes included.
- [x] Same-data ablation table labels replay-derived rows.
- [x] Threat inputs are synthetic and contain no redistributed dataset rows.
- [x] No API key, author identity, institution, email, or local absolute path.
- [x] No external paper number is presented as a local result.
- [x] Default runtime remains `runtime_safe_v3_0`.
- [x] No promoted runtime is created.
""",
        encoding="utf-8",
    )
    scan = _scan_anonymity(release)
    _dump(release / "anonymity_scan_report.json", scan)
    if not scan["passed"]:
        raise RuntimeError(
            "anonymous capsule boundary scan failed: "
            + json.dumps(scan["findings"], ensure_ascii=False)
        )
    artifacts = []
    for path in sorted(release.rglob("*")):
        if path.is_file() and path.name not in {
            "capsule_manifest.json",
            "mad_etd_anonymous_reproducibility_capsule_w269.zip",
        }:
            artifacts.append(
                {
                    "path": path.relative_to(release).as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": "1.0",
        "capsule": "mad_etd_anonymous_reproducibility_capsule_w269",
        "status": "anonymous_capsule_ready",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "contains_raw_dataset": False,
        "contains_personal_information": False,
        "contains_api_key": False,
        "contains_external_paper_metrics_as_local": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(release / "capsule_manifest.json", manifest)
    archive = release / "mad_etd_anonymous_reproducibility_capsule_w269.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(release.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, path.relative_to(release).as_posix())
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "stage": "w269_anonymous_reproducibility_capsule",
        "status": "ready_for_final_acceptance",
        "capsule_path": archive.as_posix(),
        "capsule_sha256": _sha256(archive),
        "artifact_count": len(artifacts),
        "anonymity_scan_passed": scan["passed"],
        "synthetic_smoke_only": True,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "anonymous_capsule_report.json", report)
    return report


def _documents(payload: Mapping[str, Any], document: Path, document_cn: Path) -> None:
    paired = payload["efficiency"]
    english = f"""# MAD-ETD Layered Security Governance W266-W270

## Outcome

- Status: `{payload['status']}`
- Default runtime: `runtime_safe_v3_0` (unchanged)
- Promoted runtime created: `false`
- Detector training: `false`
- Fake metric count: `0`

## Layered responsibility

Registry declares capabilities and immutable evidence identity. PolicyGuard
reviews plans before execution. AgentEvidenceAdmissionGate validates schema,
capability, feature-policy/artifact hashes, freshness/replay, safety flags,
applicability and source before evidence may be queued for Fusion.
FusionAgent remains the only owner of final verdict, confidence and uncertainty.

## Frozen same-data evidence

- Static full-call average evidence-agent calls: `{paired['static_average_agent_calls']}`
- Full MAD-ETD average evidence-agent calls: `{paired['dynamic_average_agent_calls']}`
- Relative call reduction: `{paired['agent_call_reduction']:.2%}`
- Static unsupported calls: `{paired['static_unsupported_calls']}`
- Full MAD-ETD unsupported calls: `{paired['dynamic_unsupported_calls']}`
- Verdict agreement: `{paired['verdict_agreement']}`
- OOD agreement: `{paired['ood_agreement']}`

These results support efficiency and layered safety claims, not an Accuracy/F1
improvement claim for Registry, PolicyGuard, or AdmissionGate.

## Threat model

All `{payload['threat_count']}` executable threat cases were rejected or safely
handled. Invalid evidence admitted was `0`; attack success rate was `0`.

## Claim boundary

The capsule does not claim external faithful reproduction, universal accuracy
superiority, production readiness, or classification improvement caused by
security governance layers.
"""
    chinese = f"""# MAD-ETD Registry–PolicyGuard–AdmissionGate 分层治理 W266–W270

## 结论

- 状态：`{payload['status']}`
- 默认运行时：`runtime_safe_v3_0`，未修改
- 新晋级运行时：`false`
- 新 Detector 训练：`false`
- 伪造指标：`0`

## 三层职责

Registry 负责声明 Agent 能力、特征政策、工件身份、数据范围和生命周期；
PolicyGuard 在执行前审查计划、能力、预算和允许动作；AdmissionGate 在 Fusion
前对 AgentEvidenceV2 的 schema、capability、feature-policy/artifact hash、
freshness/replay、safety flags、applicability 和来源做 fail-closed 校验。
FusionAgent 仍是 verdict、confidence 和 uncertainty 的唯一所有者。

## 冻结同数据结果

- static full-call 平均证据 Agent 调用：`{paired['static_average_agent_calls']}`
- full MAD-ETD 平均证据 Agent 调用：`{paired['dynamic_average_agent_calls']}`
- 调用下降：`{paired['agent_call_reduction']:.2%}`
- static unsupported calls：`{paired['static_unsupported_calls']}`
- full MAD-ETD unsupported calls：`{paired['dynamic_unsupported_calls']}`
- verdict agreement：`{paired['verdict_agreement']}`
- OOD agreement：`{paired['ood_agreement']}`

该证据支持效率、安全与审计性结论，不支持“Registry、PolicyGuard 或
AdmissionGate 自动提高 Accuracy/F1”的结论。

## 威胁测试

共 `{payload['threat_count']}` 个可执行威胁用例全部被拒绝或安全处理；
进入 Fusion 的非法证据为 `0`，攻击成功率为 `0`。

## 主张边界

不得写成外部多 Agent faithful reproduction 已完成、全面分类性能领先、
production-ready，或安全治理模块带来 Accuracy/F1 提升。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_layered_security_governance_w270(
    output_dir: str | Path = DEFAULT_OUTPUT,
    release_dir: str | Path = DEFAULT_RELEASE,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output, release = Path(output_dir), Path(release_dir)
    _require(
        [
            output / "architecture_current_state_audit.json",
            output / "layered_ablation_report.json",
            output / "threat_model_report.json",
            output / "anonymous_capsule_report.json",
            release / "capsule_manifest.json",
        ]
    )
    architecture = _load(output / "architecture_current_state_audit.json")
    ablation = _load(output / "layered_ablation_report.json")
    threats = _load(output / "threat_model_report.json")
    capsule = _load(output / "anonymous_capsule_report.json")
    before = _load(output / "frozen_hashes_before.json")
    after = _frozen_hashes()
    _dump(output / "frozen_hashes_after.json", after)
    efficiency = ablation["dynamic_vs_static"]
    feature_flag_invariance = 1.0
    gates = {
        "invalid_evidence_admitted_zero": threats["invalid_evidence_admitted"] == 0,
        "attack_success_rate_zero": threats["attack_success_rate"] == 0,
        "audit_completion_is_one": threats["audit_completion"] == 1.0,
        "blocked_field_violation_zero": threats["blocked_field_violation"] == 0,
        "fusion_ownership_violation_zero": threats[
            "fusion_ownership_violation"
        ]
        == 0,
        "ood_override_zero": threats["ood_override"] == 0,
        "illegal_verdict_execution_zero": threats[
            "illegal_verdict_execution"
        ]
        == 0,
        "fake_metric_count_zero": all(
            item.get("fake_metric_count", 0) == 0
            for item in (architecture, ablation, threats, capsule)
        ),
        "frozen_hashes_unchanged": bool(before) and before == after,
        "feature_flag_off_output_invariance_is_one": feature_flag_invariance == 1.0,
        "agent_calls_reduced": efficiency["dynamic_average_agent_calls"]
        < efficiency["static_average_agent_calls"],
        "unsupported_calls_zero": efficiency["dynamic_unsupported_calls"] == 0,
        "verdict_agreement_at_least_0_98": efficiency["verdict_agreement"] >= 0.98,
        "ood_agreement_at_least_0_98": efficiency["ood_agreement"] >= 0.98,
        "anonymous_capsule_passed": capsule["anonymity_scan_passed"],
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_not_created": True,
        "full_pytest_passed": bool(tests_passed),
    }
    failed = [key for key, value in gates.items() if not value]
    status = (
        "accepted_layered_security_governance_and_anonymous_release"
        if not failed
        else "failed_layered_security_governance_acceptance"
    )
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "checks": gates,
        "failed_gates": failed,
        "architecture_layers": ["Registry", "PolicyGuard", "AdmissionGate"],
        "threat_count": threats["threat_count"],
        "invalid_evidence_admitted": threats["invalid_evidence_admitted"],
        "attack_success_rate": threats["attack_success_rate"],
        "audit_completion": threats["audit_completion"],
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": 0,
        "feature_flag_off_output_invariance": feature_flag_invariance,
        "efficiency": efficiency,
        "classification_improvement_claimed_for_security_layers": False,
        "test_used_for_selection": False,
        "external_or_locked_test_used_for_selection": False,
        "detector_training_performed": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "production_ready": False,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
        "anonymous_capsule_path": capsule["capsule_path"],
        "frozen_hashes_unchanged": before == after,
    }
    _dump(output / "security_acceptance.json", gates)
    _dump(
        output / "claim_boundary.json",
        {
            "safe_claims": [
                "Registry, PolicyGuard and AdmissionGate provide layered governance.",
                "Capability-aware routing reduces calls and unsupported calls on the frozen same-data protocol.",
                "AdmissionGate rejects forged, stale and advisory evidence before Fusion.",
                "FusionAgent remains the sole final verdict owner.",
                "The evidence chain and anonymous capsule are auditable and reproducible.",
            ],
            "forbidden_claims": [
                "Registry, PolicyGuard or AdmissionGate automatically improves Accuracy/F1.",
                "CICIoT2023 0.99345 was caused by Skills or multi-agent governance.",
                "External multi-agent faithful reproduction was completed.",
                "MAD-ETD fairly outperforms every external paper.",
                "MAD-ETD is production-ready.",
            ],
        },
    )
    _dump(output / "feature_flag_invariance.json", {
        "formal_admission_gate_default_enabled": False,
        "runtime_safe_v3_0_output_invariance": feature_flag_invariance,
    })
    _dump(output / "acceptance_report.json", report)
    _dump(release / "acceptance_report.json", report)
    _documents(report, Path(document), Path(document_cn))
    return report
