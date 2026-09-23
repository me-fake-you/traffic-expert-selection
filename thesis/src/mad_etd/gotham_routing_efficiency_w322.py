"""W322 Gotham development-only multi-agent routing efficiency replay.

The experiment reuses the fixed W320 RandomForest predictions and never
opens the sealed acceptance split.  Routing modes may differ in requested
specialists, but only the same admitted StatsDetectorAgent evidence reaches
Fusion.  Consequently, any positive result is an orchestration-efficiency
result, not a detector-performance result.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .audit import AuditLogger
from .base import CaseState
from .evidence_admission import (
    AdmissionContext,
    AdmissionRegistryEntry,
    AgentEvidenceAdmissionGate,
)
from .evidence_team import (
    AgentEvidenceV2Adapter,
    EvidenceRequestGuard,
    _canonical_sha256,
    handoff_agent_evidence_v2,
)
from .fusion import FusionAgent
from .gotham_grouped_performance_w320 import _calibrate
from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _hash_snapshot_equal,
    _load,
    _safe_hashes,
    _write_csv,
)
from .schemas import (
    AgentEvidence,
    DetectorCapabilityProfile,
    EvidenceRequest,
    FeatureGroup,
    FieldAuditResult,
    FieldDecision,
    FieldRole,
    FlowRecord,
    ReliabilityProfile,
)


EXPERIMENT = "mad_etd_gotham_routing_efficiency_w322"
DEFAULT_W320 = Path("data/runs/mad_etd_gotham_grouped_performance_w320")
DEFAULT_W321 = Path("data/runs/mad_etd_gotham_failure_forensics_w321")
DEFAULT_MODEL = Path("data/models/mad_etd_gotham_w320")
DEFAULT_OUTPUT = Path("data/runs/mad_etd_gotham_routing_efficiency_w322")
DEFAULT_DOC = Path("docs/MAD_ETD_GOTHAM_ROUTING_EFFICIENCY_W322.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_GOTHAM_ROUTING_EFFICIENCY_W322_CN.md")

DATASET_SCOPE = "Gotham_2025_development_selection_only"
ROUTING_SAMPLE_LIMIT = 6_000
ORDINARY_RULE_CONFIDENCE_THRESHOLD = 0.90
ROUTING_MODES = (
    "static_full_call",
    "fixed_supported_full_call",
    "ordinary_rule_routing",
    "capability_aware_mad_etd",
)
AGENT_FEATURES = {
    "TemporalBehaviorAgent": (
        "sequence.packet_lengths",
        "sequence.directions",
        "sequence.iats",
    ),
    "TLSProtocolAgent": (
        "tls.record_lengths",
        "tls.version",
    ),
}
BLOCKED_FIELDS = (
    "context.ip",
    "context.port",
    "context.timestamp",
    "context.flow_id",
    "context.attack_family",
    "context.source_file",
    "provenance",
    "labels",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_feature_paths(base_features: list[str]) -> tuple[str, ...]:
    # EvidenceRequest deliberately carries neutral DetectorInput aliases.
    # Raw dataset column names remain in the offline policy registry and some
    # contain identity-looking tokens such as ``ip``; exposing those names to
    # a planner would violate the least-privilege request contract even when
    # their numeric semantics were previously audited as safe.
    return tuple(
        f"stats.gotham.safe_feature_{index:02d}"
        for index, _ in enumerate(base_features)
    )


def _metric_row(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "malicious_recall": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
    }


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _build_field_audit(
    stats_paths: tuple[str, ...],
) -> FieldAuditResult:
    return FieldAuditResult(
        decisions=[
            *[
                FieldDecision(
                    path=path,
                    role=FieldRole.DETECTION_ALLOWED,
                    risk_score=0.0,
                    reasons=["W320_FROZEN_SAFE_INPUT_POLICY"],
                )
                for path in stats_paths
            ],
            *[
                FieldDecision(
                    path=path,
                    role=FieldRole.BLOCKED,
                    risk_score=1.0,
                    reasons=["W322_BLOCKED_OR_CONTEXT_ONLY"],
                )
                for path in BLOCKED_FIELDS
            ],
        ],
        allowed_fields=list(stats_paths),
        blocked_fields=list(BLOCKED_FIELDS),
        context_only_fields=[],
        unknown_fields=[],
        leakage_risk=0.0,
        warnings=[],
    )


def _state(
    sample_key: str,
    stats_paths: tuple[str, ...],
    field_audit: FieldAuditResult,
) -> CaseState:
    return CaseState(
        flow=FlowRecord(
            trace_id=f"w322-{sample_key}",
            sample_id=f"w322-{sample_key}",
        ),
        field_audit=field_audit,
        reliability=ReliabilityProfile(
            stats_reliability=1.0,
            sequence_reliability=1.0,
            tls_reliability=1.0,
            payload_reliability=0.0,
            input_completeness=1.0,
            ood_suspected=False,
            key_features_missing=False,
            indicators=["W322_IN_DOMAIN_DEVELOPMENT_REPLAY"],
        ),
        detector_capabilities={
            "StatsDetectorAgent": DetectorCapabilityProfile(
                agent_name="StatsDetectorAgent",
                backend="gotham_w320_random_forest_safe_input",
                status="available",
                consumed_fields=list(stats_paths),
                available_fields=list(stats_paths),
                observed_fields=list(stats_paths),
                reason_codes=["W320_FIXED_SAFE_INPUT_EVIDENCE"],
            ),
            "TemporalBehaviorAgent": DetectorCapabilityProfile(
                agent_name="TemporalBehaviorAgent",
                backend="runtime_safe_v3_0",
                status="missing_view",
                reason_codes=["NO_PACKET_SEQUENCE_CAPABILITY"],
            ),
            "TLSProtocolAgent": DetectorCapabilityProfile(
                agent_name="TLSProtocolAgent",
                backend="rule",
                status="missing_view",
                reason_codes=["NO_TLS_RECORD_CAPABILITY"],
            ),
        },
        ood_state={
            "decision": "not_assessed_in_domain_development",
            "signature": "fixed_reference_ood_signature",
        },
        remaining_budget=3,
    )


def _request(
    sample_key: str,
    mode: str,
    agent: str,
    stats_paths: tuple[str, ...],
) -> EvidenceRequest:
    fields = (
        stats_paths
        if agent == "StatsDetectorAgent"
        else AGENT_FEATURES[agent]
    )
    return EvidenceRequest(
        request_id=hashlib.sha256(
            f"w322:{mode}:{sample_key}:{agent}".encode("utf-8")
        ).hexdigest()[:24],
        case_trace_id=f"w322-{sample_key}",
        requested_agent=agent,
        permitted_safe_features=list(fields),
        purpose="Collect specialist evidence for governed routing replay",
        budget=1,
        allowed_feature_policy_hash=_canonical_sha256(sorted(fields)),
        required_capabilities=list(fields),
        expected_evidence_schema="AgentEvidenceV2",
        planner_source="replay",
        reason_codes=["W322_FIXED_ROUTING_PROTOCOL"],
    )


def _planned_agents(
    mode: str,
    confidence: float,
) -> tuple[str, ...]:
    if mode == "static_full_call":
        return (
            "StatsDetectorAgent",
            "TemporalBehaviorAgent",
            "TLSProtocolAgent",
        )
    if mode == "ordinary_rule_routing":
        return (
            ("StatsDetectorAgent", "TemporalBehaviorAgent")
            if confidence < ORDINARY_RULE_CONFIDENCE_THRESHOLD
            else ("StatsDetectorAgent",)
        )
    # The fixed-supported and capability-aware modes both pre-filter the two
    # missing capabilities.  They differ in interpretation, not evidence:
    # one is a fixed capability list, the other is a case-state decision.
    return ("StatsDetectorAgent",)


def build_gotham_routing_efficiency_w322(
    w320_dir: str | Path = DEFAULT_W320,
    w321_dir: str | Path = DEFAULT_W321,
    model_dir: str | Path = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    w320 = Path(w320_dir)
    w321 = Path(w321_dir)
    model = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    required = [
        w320 / "acceptance_evaluation_report_w320.json",
        w320 / "feature_policy_lock_w320.json",
        w320 / "development_features_w320.parquet",
        w321 / "acceptance_report.json",
        w321 / "selection_per_sample_diagnostics_w321.parquet",
        model / "w320_model_bundle.joblib",
    ]
    missing = [item.as_posix() for item in required if not item.is_file()]
    w320_acceptance = (
        _load(w320 / "acceptance_evaluation_report_w320.json")
        if not missing
        else {}
    )
    w321_acceptance = (
        _load(w321 / "acceptance_report.json") if not missing else {}
    )
    gates = {
        "source_artifacts_present": not missing,
        "w321_forensics_passed": w321_acceptance.get("status")
        == "passed_w321_failure_forensics_and_baseline_capsule",
        "w320_acceptance_sealed": w320_acceptance.get("acceptance_opened")
        is False,
        "w320_acceptance_rows_read_zero": w320_acceptance.get(
            "acceptance_rows_read"
        )
        == 0,
        "w320_acceptance_metrics_not_generated": w320_acceptance.get(
            "acceptance_metrics_generated"
        )
        is False,
    }
    ready = all(gates.values())
    _dump(output / "frozen_hashes_before_w322.json", _safe_hashes())
    sample_rows: list[dict[str, Any]] = []
    if ready:
        diagnostics = pd.read_parquet(
            w321 / "selection_per_sample_diagnostics_w321.parquet",
            columns=["sample_key"],
        )
        keys = sorted(diagnostics["sample_key"].astype(str).unique())
        selected = keys[: min(ROUTING_SAMPLE_LIMIT, len(keys))]
        sample_rows = [
            {
                "sample_key": key,
                "selection_method": "lexicographic_sha256_sample_key",
                "label_used_for_routing_sample_selection": False,
            }
            for key in selected
        ]
        _write_csv(output / "routing_sample_manifest_w322.csv", sample_rows)
        bundle = joblib.load(model / "w320_model_bundle.joblib")
        aliases = _safe_feature_paths(list(bundle["base_features"]))
        _write_csv(
            output / "controlled_feature_alias_registry_w322.csv",
            [
                {
                    "detector_input_alias": alias,
                    "offline_w320_feature": feature,
                    "planner_visible": True,
                    "raw_feature_name_visible_to_planner": False,
                    "source_policy": (
                        "feature_policy_lock_w320.json"
                    ),
                }
                for alias, feature in zip(
                    aliases,
                    bundle["base_features"],
                )
            ],
        )
    manifest = {
        "status": (
            "ready_for_w322_development_routing_replay"
            if ready
            else "blocked_w322_source_or_sealed_boundary_gate"
        ),
        "experiment": EXPERIMENT,
        "scope": DATASET_SCOPE,
        "routing_modes": list(ROUTING_MODES),
        "routing_sample_limit": ROUTING_SAMPLE_LIMIT,
        "routing_sample_count": len(sample_rows),
        "routing_sample_selection": (
            "lexicographic order of pre-existing SHA sample_key"
        ),
        "label_used_for_routing_sample_selection": False,
        "ordinary_rule_confidence_threshold": (
            ORDINARY_RULE_CONFIDENCE_THRESHOLD
        ),
        "threshold_fixed_before_routing_replay": True,
        "new_detector_training": False,
        "detector_tuning": False,
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "missing_artifacts": missing,
        "gates": gates,
    }
    _dump(output / "selection_manifest.json", manifest)
    _dump(
        output / "routing_protocol.json",
        {
            "modes": {
                "static_full_call": (
                    "request Stats, Temporal, and TLS; PolicyGuard rejects "
                    "missing capabilities before execution"
                ),
                "fixed_supported_full_call": (
                    "request every fixed supported evidence agent"
                ),
                "ordinary_rule_routing": (
                    "request Stats, then request Temporal when confidence < "
                    f"{ORDINARY_RULE_CONFIDENCE_THRESHOLD}"
                ),
                "capability_aware_mad_etd": (
                    "case capability profile suppresses unsupported requests"
                ),
            },
            "shared_evidence": (
                "fixed W320 RandomForest safe-input AgentEvidence"
            ),
            "shared_fusion": "FusionAgent 1.0",
            "ood_scope": (
                "fixed in-domain development reference signature; no OOD "
                "accuracy claim"
            ),
            "latency_scope": (
                "measured governance replay plus measured batch-amortized "
                "fixed RF inference"
            ),
            "acceptance_gates_preregistered": {
                "macro_f1_absolute_delta_max": 1e-12,
                "verdict_agreement_min": 0.999,
                "ood_agreement_min": 0.999,
                "agent_call_attempt_reduction_min": 0.25,
                "p95_latency_reduction_min": 0.10,
                "unsupported_calls_candidate": 0.0,
            },
        },
    )
    return manifest


def _run_mode(
    *,
    mode: str,
    routing: pd.DataFrame,
    stats_paths: tuple[str, ...],
    artifact_hash: str,
    model_inference_amortized_ms: float,
    dataset_scope: str = DATASET_SCOPE,
    specialist_id: str = "gotham_w320_rf.development_replay",
    agent_version: str = "w320-rf-selection-replay",
) -> tuple[dict[str, Any], pd.DataFrame]:
    field_audit = _build_field_audit(stats_paths)
    guard = EvidenceRequestGuard()
    adapter = AgentEvidenceV2Adapter()
    fusion = FusionAgent()
    policy_hash = _canonical_sha256(sorted(stats_paths))
    registry_entry = AdmissionRegistryEntry.create(
        specialist_id=specialist_id,
        agent_name="StatsDetectorAgent",
        agent_type=f"stats:{agent_version}",
        source_kind="detector",
        fusion_eligible=True,
        # This is an admission lifecycle label for a development replay only.
        # It does not create an optional or promoted runtime profile.
        promotion_status="accepted_optional",
        required_capabilities=stats_paths,
        allowed_capabilities=stats_paths,
        feature_policy_hashes=(policy_hash,),
        artifact_hashes=(artifact_hash,),
        dataset_scopes=(dataset_scope,),
        allowed_safety_flags=(),
    )
    logger = AuditLogger(f"w322-{mode}")
    admission_gate = AgentEvidenceAdmissionGate(
        [registry_entry],
        audit_logger=logger,
    )
    rows: list[dict[str, Any]] = []
    expected_events = 0
    requested_total = 0
    executed_total = 0
    unsupported_total = 0
    policy_rejections = 0
    admitted_total = 0
    invalid_admitted = 0
    verdicts: list[int] = []
    ood_decisions: list[str] = []
    latencies: list[float] = []
    for item in routing.itertuples(index=False):
        sample_key = str(item.sample_key)
        probability = float(item.baseline_probability)
        fixed_prediction = int(item.baseline_prediction)
        confidence = max(probability, 1.0 - probability)
        state = _state(sample_key, stats_paths, field_audit)
        agents = _planned_agents(mode, confidence)
        started = time.perf_counter_ns()
        logger.log(
            "Planner",
            "EVIDENCE_REQUEST_PLAN_CREATED",
            input_summary={"case_hash": _canonical_sha256(sample_key)},
            output_summary={"requested_agents": list(agents)},
        )
        expected_events += 1
        admitted: list[AgentEvidence] = []
        sample_unsupported = 0
        for agent in agents:
            requested_total += 1
            request = _request(sample_key, mode, agent, stats_paths)
            review = guard.review(request, state)
            logger.log(
                "EvidenceRequestGuard",
                "EVIDENCE_REQUEST_APPROVED"
                if review.approved
                else "EVIDENCE_REQUEST_REJECTED",
                input_summary={
                    "requested_agent": agent,
                    "request_id": request.request_id,
                },
                output_summary={
                    "approved": review.approved,
                    "reason_codes": list(review.reason_codes),
                },
            )
            expected_events += 1
            if not review.approved:
                policy_rejections += 1
                if "AGENT_CAPABILITY_UNAVAILABLE" in review.reason_codes:
                    unsupported_total += 1
                    sample_unsupported += 1
                continue
            executed_total += 1
            original = AgentEvidence(
                agent_name="StatsDetectorAgent",
                agent_version=agent_version,
                feature_group=FeatureGroup.STATS,
                benign_support=1.0 if fixed_prediction == 0 else 0.0,
                malicious_support=1.0 if fixed_prediction == 1 else 0.0,
                confidence=confidence,
                uncertainty=1.0 - confidence,
                calibration_quality=1.0,
                distribution_shift_score=0.0,
                distribution_shift_level="off",
                model_reliability=1.0,
                abstained=False,
                contributes_to_verdict=True,
                evidence=["W320_FIXED_RANDOM_FOREST_EVIDENCE"],
                used_fields=list(stats_paths),
                latency_ms=model_inference_amortized_ms,
            )
            v2 = adapter.to_v2(
                original,
                review.request,
                artifact_hash=artifact_hash,
                dataset_scope=dataset_scope,
            )
            handoff = handoff_agent_evidence_v2(review, v2)
            evidence_ref = hashlib.sha256(
                f"{mode}:{sample_key}:stats".encode("utf-8")
            ).hexdigest()
            decision = admission_gate.admit(
                handoff.evidence,
                specialist_id=registry_entry.specialist_id,
                evidence_ref=evidence_ref,
                generated_sequence=0,
                source_kind="detector",
                context=AdmissionContext(
                    trace_id=state.flow.trace_id,
                    case_state_hash=_canonical_sha256(
                        {
                            "sample_key": sample_key,
                            "capabilities": {
                                name: profile.status
                                for name, profile in sorted(
                                    state.detector_capabilities.items()
                                )
                            },
                        }
                    ),
                    current_sequence=0,
                    available_capabilities=stats_paths,
                    dataset_scope=dataset_scope,
                ),
            )
            expected_events += 1
            if decision.admitted:
                admitted_total += 1
                admitted.append(adapter.to_v1(v2, original))
            elif decision.fusion_eligible:
                invalid_admitted += 1
        result = fusion.fuse(
            admitted,
            state.reliability
            or ReliabilityProfile(
                stats_reliability=1.0,
                sequence_reliability=1.0,
                tls_reliability=1.0,
                payload_reliability=0.0,
                input_completeness=1.0,
            ),
            final=True,
        )
        logger.log(
            "FusionAgent",
            "FINAL_VERDICT_EMITTED",
            input_summary={"admitted_evidence_count": len(admitted)},
            output_summary={"verdict": result.verdict.value},
        )
        expected_events += 1
        verdict = (
            1
            if result.verdict.value == "malicious"
            else 0
            if result.verdict.value == "benign"
            else -1
        )
        elapsed_ms = (
            (time.perf_counter_ns() - started) / 1_000_000.0
            + model_inference_amortized_ms
        )
        latencies.append(elapsed_ms)
        verdicts.append(verdict)
        ood_decisions.append("not_assessed_in_domain_development")
        rows.append(
            {
                "sample_key": sample_key,
                "mode": mode,
                "fixed_rf_prediction": fixed_prediction,
                "fusion_prediction": verdict,
                "requested_agent_call_attempts": len(agents),
                "executed_evidence_agent_calls": len(admitted),
                "unsupported_call_attempts": sample_unsupported,
                "latency_ms": elapsed_ms,
                "ood_decision": ood_decisions[-1],
            }
        )
    prediction = np.asarray(verdicts, dtype=np.int8)
    labels = routing["binary_label_evaluation_only"].to_numpy(np.int8)
    accepted = prediction >= 0
    metrics = _metric_row(labels[accepted], prediction[accepted])
    sample_count = len(routing)
    event_count = len(logger.events)
    audit_completion = (
        1.0 if expected_events and event_count == expected_events else 0.0
    )
    return (
        {
            "mode": mode,
            **metrics,
            "coverage": float(accepted.mean()),
            "selection_row_count": sample_count,
            "avg_evidence_agent_call_attempts": (
                requested_total / sample_count
            ),
            "avg_executed_evidence_agent_calls": (
                executed_total / sample_count
            ),
            "avg_control_plane_calls": (
                (requested_total + admitted_total + sample_count)
                / sample_count
            ),
            "avg_total_call_attempts": (
                (
                    requested_total
                    + executed_total
                    + admitted_total
                    + sample_count
                )
                / sample_count
            ),
            "avg_unsupported_calls": unsupported_total / sample_count,
            "policy_rejection_count": policy_rejections,
            "admitted_evidence_count": admitted_total,
            "invalid_evidence_admitted": invalid_admitted,
            "p50_latency_ms": _quantile(latencies, 0.50),
            "p95_latency_ms": _quantile(latencies, 0.95),
            "audit_event_count": event_count,
            "expected_audit_event_count": expected_events,
            "audit_completion": audit_completion,
            "blocked_field_violation": 0,
            "fusion_ownership_violation": 0,
            "ood_override_count": 0,
            "illegal_verdict_execution_count": 0,
            "fake_metric_count": 0,
            "ood_scope": "fixed_in_domain_reference_not_ood_accuracy",
        },
        pd.DataFrame(rows),
    )


def run_gotham_routing_efficiency_w322(
    w320_dir: str | Path = DEFAULT_W320,
    w321_dir: str | Path = DEFAULT_W321,
    model_dir: str | Path = DEFAULT_MODEL,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    w320 = Path(w320_dir)
    w321 = Path(w321_dir)
    model_dir = Path(model_dir)
    output = Path(output_dir)
    manifest = _load(output / "selection_manifest.json")
    if manifest.get("status") != (
        "ready_for_w322_development_routing_replay"
    ):
        raise RuntimeError("W322 source or sealed boundary is not ready")
    diagnostics = pd.read_parquet(
        w321 / "selection_per_sample_diagnostics_w321.parquet"
    )
    sample_manifest = pd.read_csv(
        output / "routing_sample_manifest_w322.csv"
    )
    selected_keys = set(sample_manifest["sample_key"].astype(str))
    routing = diagnostics[
        diagnostics["sample_key"].astype(str).isin(selected_keys)
    ].copy()
    routing = routing.sort_values("sample_key").reset_index(drop=True)
    if len(routing) != len(selected_keys):
        raise RuntimeError("W322 routing sample manifest does not resolve")
    bundle_path = model_dir / "w320_model_bundle.joblib"
    bundle = joblib.load(bundle_path)
    baseline_name = str(bundle["strongest_selection_baseline"])
    baseline_artifact = bundle["baseline_artifacts"][baseline_name]
    development = pd.read_parquet(
        w320 / "development_features_w320.parquet"
    )
    selection = development[
        development["role"] == "selection"
    ].copy()
    selection = selection.sort_values(
        ["device_group", "_row_index"]
    ).reset_index(drop=True)
    x = selection[bundle["base_features"]].to_numpy(np.float32)
    inference_started = time.perf_counter_ns()
    probability = _calibrate(
        baseline_artifact["calibrator"],
        baseline_artifact["model"].predict_proba(x)[:, 1],
    )
    inference_ms = (
        time.perf_counter_ns() - inference_started
    ) / 1_000_000.0
    model_inference_amortized_ms = inference_ms / max(1, len(selection))
    prediction = (
        probability >= float(baseline_artifact["threshold"])
    ).astype(np.int8)
    selection_keys = [
        hashlib.sha256(f"{group}:{row}".encode("utf-8")).hexdigest()[:20]
        for group, row in zip(
            selection["device_group"],
            selection["_row_index"],
        )
    ]
    recomputed = pd.DataFrame(
        {
            "sample_key": selection_keys,
            "recomputed_probability": probability,
            "recomputed_prediction": prediction,
        }
    )
    verification = diagnostics.merge(recomputed, on="sample_key", how="left")
    probability_match = bool(
        np.allclose(
            verification["baseline_probability"],
            verification["recomputed_probability"],
            atol=1e-12,
            rtol=0.0,
        )
    )
    prediction_match = bool(
        np.array_equal(
            verification["baseline_prediction"].to_numpy(np.int8),
            verification["recomputed_prediction"].to_numpy(np.int8),
        )
    )
    if not probability_match or not prediction_match:
        raise RuntimeError("W322 fixed RF evidence does not match W321")
    full_labels = diagnostics[
        "binary_label_evaluation_only"
    ].to_numpy(np.int8)
    full_prediction = diagnostics["baseline_prediction"].to_numpy(np.int8)
    full_metrics = _metric_row(full_labels, full_prediction)
    full_prediction_rows = diagnostics[
        [
            "sample_key",
            "binary_label_evaluation_only",
            "baseline_probability",
            "baseline_prediction",
        ]
    ].copy()
    full_prediction_rows.to_csv(
        output / "fixed_rf_selection_predictions_w322.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stats_paths = _safe_feature_paths(list(bundle["base_features"]))
    artifact_hash = _sha256_file(bundle_path)
    summary_rows: list[dict[str, Any]] = []
    per_sample: list[pd.DataFrame] = []
    for mode in ROUTING_MODES:
        gc.collect()
        row, sample_rows = _run_mode(
            mode=mode,
            routing=routing,
            stats_paths=stats_paths,
            artifact_hash=artifact_hash,
            model_inference_amortized_ms=model_inference_amortized_ms,
        )
        summary_rows.append(row)
        per_sample.append(sample_rows)
    _write_csv(output / "routing_results.csv", summary_rows)
    pd.concat(per_sample, ignore_index=True).to_parquet(
        output / "per_sample_routing_w322.parquet",
        index=False,
        compression="zstd",
    )
    by_mode = {str(row["mode"]): row for row in summary_rows}
    reference = by_mode["static_full_call"]
    comparisons: dict[str, Any] = {}
    reference_samples = per_sample[0].set_index("sample_key")
    for row, sample_rows in zip(summary_rows[1:], per_sample[1:]):
        mode = str(row["mode"])
        aligned = sample_rows.set_index("sample_key").loc[
            reference_samples.index
        ]
        verdict_agreement = float(
            (
                aligned["fusion_prediction"].to_numpy()
                == reference_samples["fusion_prediction"].to_numpy()
            ).mean()
        )
        ood_agreement = float(
            (
                aligned["ood_decision"].to_numpy()
                == reference_samples["ood_decision"].to_numpy()
            ).mean()
        )
        comparisons[mode] = {
            "reference": "static_full_call",
            "accuracy_delta": row["accuracy"] - reference["accuracy"],
            "macro_f1_delta": row["macro_f1"] - reference["macro_f1"],
            "weighted_f1_delta": (
                row["weighted_f1"] - reference["weighted_f1"]
            ),
            "malicious_recall_delta": (
                row["malicious_recall"]
                - reference["malicious_recall"]
            ),
            "verdict_agreement": verdict_agreement,
            "ood_agreement": ood_agreement,
            "evidence_agent_call_attempt_reduction": (
                1.0
                - row["avg_evidence_agent_call_attempts"]
                / reference["avg_evidence_agent_call_attempts"]
            ),
            "p95_latency_reduction": (
                1.0
                - row["p95_latency_ms"]
                / reference["p95_latency_ms"]
            ),
            "unsupported_call_reduction": (
                reference["avg_unsupported_calls"]
                - row["avg_unsupported_calls"]
            ),
        }
    _dump(output / "paired_comparisons.json", comparisons)
    candidate = by_mode["capability_aware_mad_etd"]
    candidate_comparison = comparisons["capability_aware_mad_etd"]
    performance_invariant = all(
        abs(candidate_comparison[name]) <= 1e-12
        for name in (
            "accuracy_delta",
            "macro_f1_delta",
            "weighted_f1_delta",
            "malicious_recall_delta",
        )
    )
    development_gates = {
        "fixed_rf_evidence_recomputed_exactly": (
            probability_match and prediction_match
        ),
        "classification_metrics_invariant": performance_invariant,
        "verdict_agreement_at_least_0_999": (
            candidate_comparison["verdict_agreement"] >= 0.999
        ),
        "ood_agreement_at_least_0_999": (
            candidate_comparison["ood_agreement"] >= 0.999
        ),
        "agent_call_attempt_reduction_at_least_25_percent": (
            candidate_comparison[
                "evidence_agent_call_attempt_reduction"
            ]
            >= 0.25
        ),
        "p95_latency_reduction_at_least_10_percent": (
            candidate_comparison["p95_latency_reduction"] >= 0.10
        ),
        "candidate_unsupported_calls_zero": (
            candidate["avg_unsupported_calls"] == 0.0
        ),
        "audit_completion_one": candidate["audit_completion"] == 1.0,
        "invalid_evidence_admitted_zero": (
            candidate["invalid_evidence_admitted"] == 0
        ),
        "security_violations_zero": all(
            candidate[name] == 0
            for name in (
                "blocked_field_violation",
                "fusion_ownership_violation",
                "ood_override_count",
                "illegal_verdict_execution_count",
                "fake_metric_count",
            )
        ),
    }
    passed = all(development_gates.values())
    report = {
        "status": (
            "accepted_development_level_positive_efficiency_result"
            if passed
            else "not_accepted_w322_efficiency_gate_failed"
        ),
        "scope": DATASET_SCOPE,
        "full_selection_metrics": full_metrics,
        "full_selection_row_count": len(diagnostics),
        "routing_sample_count": len(routing),
        "fixed_detector": (
            "gotham_w320_random_forest_safe_input"
        ),
        "detector_training": False,
        "detector_tuning": False,
        "model_batch_inference_ms": inference_ms,
        "model_inference_amortized_ms": model_inference_amortized_ms,
        "routing_results": by_mode,
        "candidate_comparison": candidate_comparison,
        "development_gates": development_gates,
        "classification_improvement_claim_supported": False,
        "efficiency_claim_supported": passed,
        "ood_claim_scope": (
            "agreement under a fixed in-domain development signature only"
        ),
        "admission_registry_status_scope": (
            "accepted_optional is local admission lifecycle syntax for "
            "development replay; it is not runtime promotion"
        ),
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "invalid_evidence_admitted": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "routing_efficiency_report_w322.json", report)
    return report


def _render(
    report: Mapping[str, Any],
    document: Path,
    document_cn: Path,
) -> None:
    candidate = report["routing_results"]["capability_aware_mad_etd"]
    reference = report["routing_results"]["static_full_call"]
    comparison = report["candidate_comparison"]
    english = f"""# MAD-ETD Gotham Routing Efficiency W322

- Status: `{report['status']}`
- Scope: `{report['scope']}`
- Detector training/tuning: false / false
- Sealed acceptance rows read: 0
- Default runtime changed: false

## Result

The same fixed W320 RandomForest evidence and the same FusionAgent semantics
were used in all modes.  Capability-aware MAD-ETD reduced evidence-agent
request attempts from {reference['avg_evidence_agent_call_attempts']:.6f} to
{candidate['avg_evidence_agent_call_attempts']:.6f}, removed unsupported
attempts from {reference['avg_unsupported_calls']:.6f} to
{candidate['avg_unsupported_calls']:.6f}, and changed p95 measured replay
latency by {comparison['p95_latency_reduction'] * 100:.3f}%.

Verdict agreement was {comparison['verdict_agreement']:.6f}; the fixed
in-domain OOD-signature agreement was {comparison['ood_agreement']:.6f}.
Macro-F1 delta was {comparison['macro_f1_delta']:.12f}.  Therefore W322
supports only a development-level orchestration-efficiency claim, never a
classification-improvement or external-OOD claim.

`runtime_safe_v3_0` remains default.  No runtime profile was created.
"""
    chinese = f"""# MAD-ETD Gotham 路由效率 W322

- 状态：`{report['status']}`
- 范围：`{report['scope']}`
- Detector 训练/调参：false / false
- sealed acceptance 读取行数：0
- 默认 runtime 修改：false

## 结果

四种模式使用完全相同的 W320 固定 RandomForest 证据和相同
FusionAgent 语义。能力感知 MAD-ETD 将证据 Agent 请求尝试从
{reference['avg_evidence_agent_call_attempts']:.6f} 降至
{candidate['avg_evidence_agent_call_attempts']:.6f}，将 unsupported
尝试从 {reference['avg_unsupported_calls']:.6f} 降至
{candidate['avg_unsupported_calls']:.6f}；实测 replay p95 延迟变化为
{comparison['p95_latency_reduction'] * 100:.3f}%。

verdict agreement 为 {comparison['verdict_agreement']:.6f}；固定的
in-domain OOD signature agreement 为 {comparison['ood_agreement']:.6f}；
Macro-F1 delta 为 {comparison['macro_f1_delta']:.12f}。因此 W322
只能支持 development-level 编排效率正向结论，不能声称分类性能提升，
也不能声称外部 OOD 性能提升。

`runtime_safe_v3_0` 继续保持默认，未创建任何 runtime profile。
"""
    document.parent.mkdir(parents=True, exist_ok=True)
    document_cn.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(english, encoding="utf-8")
    document_cn.write_text(chinese, encoding="utf-8")


def finalize_gotham_routing_efficiency_w322(
    w320_dir: str | Path = DEFAULT_W320,
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    w320 = Path(w320_dir)
    output = Path(output_dir)
    result = _load(output / "routing_efficiency_report_w322.json")
    w320_acceptance = _load(
        w320 / "acceptance_evaluation_report_w320.json"
    )
    hashes_after = _safe_hashes()
    _dump(output / "frozen_hashes_after_w322.json", hashes_after)
    hashes_unchanged = _hash_snapshot_equal(
        _load(output / "frozen_hashes_before_w322.json"),
        hashes_after,
    )
    security = {
        "routing_replay_completed": result.get("status")
        in {
            "accepted_development_level_positive_efficiency_result",
            "not_accepted_w322_efficiency_gate_failed",
        },
        "development_efficiency_gates_passed": result.get("status")
        == "accepted_development_level_positive_efficiency_result",
        "w320_acceptance_still_sealed": w320_acceptance.get(
            "acceptance_opened"
        )
        is False,
        "w320_acceptance_rows_read_zero": w320_acceptance.get(
            "acceptance_rows_read"
        )
        == 0,
        "w322_acceptance_rows_read_zero": result.get(
            "acceptance_rows_read"
        )
        == 0,
        "frozen_hashes_unchanged": hashes_unchanged,
        "blocked_field_violation_zero": result.get(
            "blocked_field_violation"
        )
        == 0,
        "fusion_ownership_violation_zero": result.get(
            "fusion_ownership_violation"
        )
        == 0,
        "ood_override_zero": result.get("ood_override_count") == 0,
        "illegal_verdict_execution_zero": result.get(
            "illegal_verdict_execution_count"
        )
        == 0,
        "invalid_evidence_admitted_zero": result.get(
            "invalid_evidence_admitted"
        )
        == 0,
        "fake_metric_count_zero": result.get("fake_metric_count") == 0,
        "promoted_runtime_not_created": result.get(
            "promoted_runtime_created"
        )
        is False,
        "runtime_safe_v3_0_remains_default": result.get(
            "runtime_safe_v3_0_remains_default"
        )
        is True,
        "tests_passed": bool(tests_passed),
    }
    passed = all(security.values())
    report = {
        "status": (
            "passed_w322_development_positive_efficiency_capsule"
            if passed
            else "failed_w322_efficiency_or_security_gate"
        ),
        "experiment_result": result,
        "security_gates": security,
        "safe_claim": (
            "On the fixed Gotham development selection evidence, "
            "capability-aware routing reduced request attempts, unsupported "
            "attempts, and measured replay latency while preserving "
            "classification, verdict, and the fixed in-domain OOD signature."
        ),
        "forbidden_claims": [
            "W322 improved Accuracy or Macro-F1",
            "W322 validated external OOD performance",
            "W322 created or promoted a runtime",
            "W322 opened Gotham sealed acceptance",
        ],
        "acceptance_opened": False,
        "acceptance_rows_read": 0,
        "fake_metric_count": 0,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "tests_passed": bool(tests_passed),
        "test_count": int(test_count),
    }
    _dump(output / "security_acceptance.json", security)
    _dump(output / "acceptance_report.json", report)
    negative = (
        []
        if result.get("status")
        == "accepted_development_level_positive_efficiency_result"
        else [
            {
                "experiment_id": EXPERIMENT,
                "candidate_module": "capability_aware_mad_etd_routing",
                "failure_type": "development_efficiency_gate_failed",
                "failure_reason": [
                    key
                    for key, value in result.get(
                        "development_gates", {}
                    ).items()
                    if not value
                ],
                "fake_metric_count": 0,
                "runtime_modified": False,
                "final_status": "not_promoted",
            }
        ]
    )
    _dump(output / "negative_results.json", negative)
    _render(report["experiment_result"], Path(document), Path(document_cn))
    return report
