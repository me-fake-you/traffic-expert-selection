from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .audit import AuditLogger
from .base import CaseState, Coordinator, DetectorAgent
from .coordinator import (
    DEFAULT_ALLOWED_AGENTS,
    EvidenceBudgetRouterV1,
    GuardedCoordinator,
    NvidiaCoordinator,
    PolicyGuard,
    RuleCoordinator,
)
from .detectors import (
    FamilyAttributionAgent,
    IntentAgent,
    PayloadRepresentationAgent,
    StatsDetectorAgent,
    TLSProtocolAgent,
    TemporalBehaviorAgent,
)
from .field_audit import AuditPolicy, FieldAuditAgent
from .evidence_stability import (
    CounterfactualEvidenceEnvelope,
    EVIDENCE_STABILITY_POLICIES,
)
from .evidence_budget_router import EvidenceBudgetRouterV11, EvidenceBudgetRouterV12
from .fusion import FusionAgent
from .guard import PseudoFeatureGuardAgent
from .knowledge import KnowledgeRetrievalService
from .memory import JsonlCaseMemory
from .orchestration import (
    CoordinatorExecutionPlanner,
    PlanPolicyGuard,
    PlannerExecutorCoordinator,
)
from .reasoning import (
    AuditCritic,
    ConflictDeliberationService,
    SelfReflectionService,
)
from .reporter import ReporterAgent
from .schemas import (
    CoordinatorAction,
    DetectionReport,
    ExecutionPlan,
    ExecutorResult,
    FieldRole,
    FlowRecord,
    FutureFeatureFlags,
)
from .schemas import CoordinatorDecision, DetectorInput, ReliabilityProfile
from .view_contract import (
    VERDICT_AGENT_NAMES,
    build_detector_capabilities,
    build_view_availability,
)


class DetectionEngine:
    def __init__(
        self,
        *,
        coordinator: Coordinator,
        detectors: list[DetectorAgent],
        enrichment_agents: list[DetectorAgent] | None = None,
        field_auditor: FieldAuditAgent | None = None,
        feature_guard: PseudoFeatureGuardAgent | None = None,
        fusion_agent: FusionAgent | None = None,
        reporter: ReporterAgent | None = None,
        max_workers: int = 4,
        initial_budget: int = 4,
        enable_field_audit: bool = True,
        enable_pseudo_guard: bool = True,
        execution_policy: str = "dynamic",
        future_flags: FutureFeatureFlags | None = None,
        memory_service: JsonlCaseMemory | None = None,
        deliberation_service: ConflictDeliberationService | None = None,
        reflection_service: SelfReflectionService | None = None,
        audit_critic: AuditCritic | None = None,
        utility_policy=None,
        evidence_stability_policy: str = "off",
        orbit_reliability_dir: str | Path | None = None,
        routing_policy: str = "legacy",
        enrichment_policy: str = "legacy_inline",
        audit_payload_policy: str = "full",
        latency_profile: bool = False,
    ) -> None:
        if audit_payload_policy not in {"full", "compact"}:
            raise ValueError("audit_payload_policy must be 'full' or 'compact'")
        self.coordinator = coordinator
        self.detectors = {agent.name: agent for agent in detectors}
        self.enrichment_agents = {
            agent.name: agent for agent in (enrichment_agents or [])
        }
        self.field_auditor = field_auditor or FieldAuditAgent()
        self.feature_guard = feature_guard or PseudoFeatureGuardAgent()
        self.fusion_agent = fusion_agent or FusionAgent()
        self.reporter = reporter or ReporterAgent()
        self.max_workers = max_workers
        self.initial_budget = initial_budget
        self.enable_field_audit = enable_field_audit
        self.enable_pseudo_guard = enable_pseudo_guard
        self.execution_policy = execution_policy
        self.future_flags = future_flags or FutureFeatureFlags()
        self.memory_service = memory_service
        self.deliberation_service = (
            deliberation_service or ConflictDeliberationService()
        )
        self.reflection_service = reflection_service or SelfReflectionService()
        self.audit_critic = audit_critic or AuditCritic()
        self.utility_policy = utility_policy
        self.routing_policy = routing_policy
        self.contract_routing_policy = (
            "capability_v3_0"
            if routing_policy in {"efficiency_v1", "efficiency_v1_1", "efficiency_v1_2"}
            else routing_policy
        )
        self.enrichment_policy = enrichment_policy
        self.audit_payload_policy = audit_payload_policy
        self.latency_profile = latency_profile
        self.deliberation_fallback = RuleCoordinator(
            routing_policy=self.contract_routing_policy,
            enrichment_policy=enrichment_policy,
            allowed_agents=set(self.detectors),
        )
        self.deliberation_guard = PolicyGuard(
            allowed_agents=set(self.detectors),
            max_rounds=3,
            max_total_calls=4,
        )
        self.evidence_stability = CounterfactualEvidenceEnvelope(
            evidence_stability_policy,
            orbit_reliability_dir=(
                str(orbit_reliability_dir)
                if orbit_reliability_dir is not None
                else None
            ),
        )

    def analyze(
        self,
        flow: FlowRecord,
        *,
        audit_path: str | Path | None = None,
    ) -> tuple[DetectionReport, AuditLogger]:
        audit = AuditLogger(flow.trace_id, audit_path)
        state = CaseState(flow=flow, remaining_budget=self.initial_budget)
        audit.log(
            "FeatureCapture",
            "FLOW_VALIDATED",
            input_summary={"schema_version": flow.schema_version},
            output_summary={
                "stats_fields": sorted(flow.stats),
                "sequence_length": len(flow.sequence.packet_lengths),
                "tls_fields": sorted(flow.tls),
            },
            reason="Input passed strict schema validation.",
        )

        started = time.perf_counter()
        state.field_audit = self.field_auditor.audit(flow)
        state.safe_flow = (
            self.field_auditor.make_safe_flow(flow, state.field_audit)
            if self.enable_field_audit
            else flow.model_copy(deep=True)
        )
        detector_input = self.field_auditor.make_detector_input(
            flow,
            state.field_audit,
            enforce=self.enable_field_audit,
        )
        tls_agent = self.detectors.get("TLSProtocolAgent")
        tls_backend = getattr(tls_agent, "backend", "rule")
        state.view_availability = build_view_availability(
            detector_input,
            routing_policy=self.contract_routing_policy,
            tls_backend=tls_backend,
        )
        stats_agent = self.detectors.get("StatsDetectorAgent")
        temporal_agent = self.detectors.get("TemporalBehaviorAgent")
        state.detector_capabilities = build_detector_capabilities(
            detector_input,
            stats_backend=getattr(stats_agent, "backend", "rule"),
            temporal_backend=getattr(temporal_agent, "backend", "rule"),
            tls_backend=tls_backend,
        )
        audit.log(
            self.field_auditor.name,
            "FIELD_POLICY_APPLIED"
            if self.enable_field_audit
            else "FIELD_POLICY_BYPASSED",
            input_summary={"field_count": len(state.field_audit.decisions)},
            output_summary=self._field_audit_summary(
                state.field_audit,
                detector_input,
            ),
            reason=(
                "Strong field policy is applied before all detector calls."
                if self.enable_field_audit
                else "Field policy enforcement is disabled for ablation only."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        if self.contract_routing_policy == "capability_v3_0":
            audit.log(
                "StateBuilder",
                "VIEW_CAPABILITY_EVALUATED",
                output_summary={
                    name: item.model_dump(mode="json")
                    for name, item in state.detector_capabilities.items()
                },
                reason=(
                    "Backend-specific capability profiles guard detector "
                    "routing."
                ),
            )
        audit.log(
            "StateBuilder",
            "VIEW_AVAILABILITY_BUILT",
            output_summary=state.view_availability.model_dump(mode="json"),
            reason=(
                "Routing visibility was derived from detector-consumer fields."
                if self.contract_routing_policy == "contract_v2_9"
                else "Legacy view availability was retained for reproduction."
            ),
        )

        started = time.perf_counter()
        state.reliability = (
            self.feature_guard.assess(state.safe_flow)
            if self.enable_pseudo_guard
            else ReliabilityProfile(
                stats_reliability=1,
                sequence_reliability=1,
                tls_reliability=1,
                payload_reliability=1,
                input_completeness=1,
                indicators=["pseudo-feature guard disabled for ablation"],
            )
        )
        audit.log(
            self.feature_guard.name,
            "RELIABILITY_ASSESSED"
            if self.enable_pseudo_guard
            else "RELIABILITY_BYPASSED",
            input_summary={
                "visible_stats_fields": sorted(state.safe_flow.stats),
                "sequence_length": len(state.safe_flow.sequence.packet_lengths),
            },
            output_summary=state.reliability.model_dump(mode="json"),
            reason=(
                "Reliability is estimated from completeness and cross-view consistency."
                if self.enable_pseudo_guard
                else "Reliability profile is fixed to one for ablation."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        if any(
            (
                self.future_flags.memory,
                self.future_flags.planner_executor,
                self.future_flags.deliberation,
                self.future_flags.reflection,
                self.future_flags.audit_critic,
                self.future_flags.learning_queue,
                self.future_flags.evidence_utility_v2,
                self.future_flags.evidence_utility_v2_1,
            )
        ):
            audit.log(
                "FeatureFlags",
                "FEATURE_FLAG_EVALUATED",
                output_summary=self.future_flags.model_dump(mode="json"),
                reason="Research extensions are explicit and opt-in.",
            )
        if self.future_flags.memory:
            if self.memory_service is None:
                audit.log(
                    "CaseMemory",
                    "MEMORY_RETRIEVAL_FAILED",
                    reason="Memory feature enabled without a configured store.",
                )
            else:
                memory_hint = self.memory_service.retrieve(
                    state.safe_flow,
                    audit=audit,
                )
                state.future_artifacts.memory_hint = memory_hint
                if not self.future_flags.shadow_mode:
                    state.memory_hint = memory_hint

        if self.execution_policy in {"fixed_all", "fixed_supported"}:
            selected_agents = list(self.detectors)
            if self.execution_policy == "fixed_supported":
                selected_agents = [
                    name
                    for name in self.detectors
                    if state.detector_capabilities[name].status == "available"
                ]
            decision = CoordinatorDecision(
                action=CoordinatorAction.DISPATCH,
                agents=selected_agents,
                reason_codes=[
                    (
                        "FIXED_SUPPORTED_AGENTS_BASELINE"
                        if self.execution_policy == "fixed_supported"
                        else "FIXED_ALL_AGENTS_BASELINE"
                    )
                ],
                rationale=(
                    "Fixed supported baseline invokes every backend-compatible "
                    "detector exactly once."
                    if self.execution_policy == "fixed_supported"
                    else "Fixed baseline invokes every registered detector exactly once."
                ),
                remaining_budget=0,
                source="rule",
                policy_status="not_applicable",
                proposed_action=CoordinatorAction.DISPATCH,
                proposed_agents=selected_agents,
            )
            state.decisions.append(decision)
            self._log_coordination(audit, state, decision)
            self._run_detectors(
                [self.detectors[name] for name in selected_agents],
                detector_input,
                state,
                audit,
            )
            state.called_agents.update(selected_agents)
            state.remaining_budget = 0
            state.round_no = 1
            state.interim_fusion = self.fusion_agent.fuse(
                state.evidence, state.reliability, final=False
            )
            self._log_fusion(audit, state, state.interim_fusion, final=False)

        while (
            self.execution_policy == "dynamic"
            and state.round_no < 3
            and state.remaining_budget > 0
        ):
            decision = self.coordinator.decide(state)
            state.decisions.append(decision)
            self._log_coordination(audit, state, decision)

            if decision.action != CoordinatorAction.DISPATCH:
                break

            selected = [
                self.detectors[name]
                for name in decision.agents
                if name in self.detectors
            ]
            if not selected:
                break
            self._run_detectors(selected, detector_input, state, audit)
            state.called_agents.update(agent.name for agent in selected)
            state.remaining_budget = max(0, state.remaining_budget - len(selected))
            state.round_no += 1
            state.interim_fusion = self.fusion_agent.fuse(
                state.evidence,
                state.reliability,
                final=False,
            )
            self._log_fusion(audit, state, state.interim_fusion, final=False)
            self._run_deliberation(state, detector_input, audit)

        final_fusion = self.fusion_agent.fuse(
            state.evidence,
            state.reliability,
            final=True,
        )
        self._log_fusion(audit, state, final_fusion, final=True)
        self._run_post_fusion_enrichment(
            state,
            detector_input,
            final_fusion,
            audit,
        )
        if self.future_flags.reflection:
            state.reflection = self.reflection_service.reflect(
                state,
                final_fusion,
                audit=audit,
            )
            state.future_artifacts.reflection = state.reflection
        if self.future_flags.audit_critic:
            state.compliance = self.audit_critic.inspect(flow.trace_id, audit)
            state.future_artifacts.compliance = state.compliance
            audit.log(
                self.audit_critic.name,
                "COMPLIANCE_REPORT_GENERATED",
                output_summary=state.compliance.model_dump(mode="json"),
                reason="Audit Critic produced compliance findings only.",
            )

        knowledge_support = self.reporter.retrieve_knowledge(
            state,
            final_fusion,
            audit,
        )
        report = self.reporter.build(
            state,
            final_fusion,
            audit_event_count=len(audit.events) + 1,
            knowledge_support=knowledge_support,
        )
        immutable_fields = {
            "verdict": final_fusion.verdict,
            "confidence": final_fusion.confidence,
            "uncertainty": final_fusion.uncertainty,
            "conflict_score": final_fusion.conflict_score,
            "benign_support": final_fusion.benign_support,
            "malicious_support": final_fusion.malicious_support,
            "distribution_shift_score": final_fusion.distribution_shift_score,
            "severity": final_fusion.severity,
            "need_escalation": final_fusion.need_escalation,
        }
        if any(
            getattr(report, field) != expected
            for field, expected in immutable_fields.items()
        ):
            raise RuntimeError("Reporter modified an immutable FusionResult field")
        if self.latency_profile:
            audit.log(
                "RuntimeProfiler",
                "RUNTIME_STAGE_PROFILE",
                output_summary={
                    "audit_payload_policy": self.audit_payload_policy,
                    "event_count_before_report": len(audit.events),
                    "duration_by_event_type_ms": self._duration_by_event_type(audit),
                },
                reason="Latency profiling records timing metadata only.",
            )
        audit.log(
            self.reporter.name,
            "REPORT_GENERATED",
            output_summary=self._report_summary(report),
            reason="Reporter translated immutable evidence into a human-readable result.",
        )
        audit.future_artifacts = state.future_artifacts.model_copy(deep=True)
        return report, audit

    @staticmethod
    def _sha256_payload(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def _field_audit_summary(
        self,
        field_audit,
        detector_input: DetectorInput,
    ) -> dict[str, Any]:
        full = {
            "enforced": self.enable_field_audit,
            "decisions": [
                decision.model_dump(mode="json")
                for decision in field_audit.decisions
            ],
            "blocked_fields": field_audit.blocked_fields,
            "context_only_fields": field_audit.context_only_fields,
            "unknown_fields": field_audit.unknown_fields,
            "ignored_empty_fields": field_audit.ignored_empty_fields,
            "field_audit_mode": self.field_auditor.policy.mode,
            "leakage_risk": field_audit.leakage_risk,
            "detector_visible_fields": detector_input.visible_field_paths(),
        }
        if self.audit_payload_policy == "full":
            return full
        return {
            "enforced": full["enforced"],
            "decision_count": len(full["decisions"]),
            "decisions_sha256": self._sha256_payload(full["decisions"]),
            "blocked_fields": full["blocked_fields"],
            "context_only_fields": full["context_only_fields"],
            "unknown_fields": full["unknown_fields"],
            "ignored_empty_fields": full["ignored_empty_fields"],
            "field_audit_mode": full["field_audit_mode"],
            "leakage_risk": full["leakage_risk"],
            "detector_visible_fields": full["detector_visible_fields"],
            "compact_payload": True,
        }

    def _agent_evidence_summary(self, result) -> dict[str, Any]:
        full = result.model_dump(mode="json")
        if self.audit_payload_policy == "full":
            return full
        return {
            "agent_name": result.agent_name,
            "agent_version": result.agent_version,
            "feature_group": result.feature_group.value,
            "benign_support": result.benign_support,
            "malicious_support": result.malicious_support,
            "confidence": result.confidence,
            "uncertainty": result.uncertainty,
            "distribution_shift_score": result.distribution_shift_score,
            "distribution_shift_raw_score": result.distribution_shift_raw_score,
            "distribution_shift_level": result.distribution_shift_level,
            "model_reliability": result.model_reliability,
            "abstained": result.abstained,
            "contributes_to_verdict": result.contributes_to_verdict,
            "used_fields": list(result.used_fields),
            "latency_ms": result.latency_ms,
            "evidence_count": len(result.evidence),
            "full_evidence_sha256": self._sha256_payload(full),
            "compact_payload": True,
        }

    def _fusion_input_summary(self, state: CaseState) -> dict[str, Any]:
        reliability = state.reliability.model_dump(mode="json")
        evidence = [item.model_dump(mode="json") for item in state.evidence]
        if self.audit_payload_policy == "full":
            evidence_payload: Any = evidence
        else:
            evidence_payload = [
                {
                    "agent_name": item.agent_name,
                    "feature_group": item.feature_group.value,
                    "benign_support": item.benign_support,
                    "malicious_support": item.malicious_support,
                    "confidence": item.confidence,
                    "uncertainty": item.uncertainty,
                    "distribution_shift_score": item.distribution_shift_score,
                    "distribution_shift_level": item.distribution_shift_level,
                    "abstained": item.abstained,
                    "contributes_to_verdict": item.contributes_to_verdict,
                }
                for item in state.evidence
            ]
        payload = {
            "evidence": evidence_payload,
            "reliability": reliability,
            "reliability_discount_enabled": self.fusion_agent.use_reliability_discount,
        }
        if self.audit_payload_policy == "compact":
            payload["full_evidence_sha256"] = self._sha256_payload(evidence)
            payload["compact_payload"] = True
        return payload

    def _report_summary(self, report: DetectionReport) -> dict[str, Any]:
        full = report.model_dump(mode="json")
        if self.audit_payload_policy == "full":
            return full
        fields = {
            "schema_version",
            "trace_id",
            "sample_id",
            "verdict",
            "confidence",
            "uncertainty",
            "conflict_score",
            "benign_support",
            "malicious_support",
            "distribution_shift_score",
            "severity",
            "need_escalation",
            "participating_agents",
            "blocked_fields",
            "recommended_actions",
            "audit_event_count",
        }
        compact = {key: full[key] for key in fields if key in full}
        compact["agent_result_count"] = len(report.agent_results)
        compact["full_report_sha256"] = self._sha256_payload(full)
        compact["compact_payload"] = True
        return compact

    @staticmethod
    def _duration_by_event_type(audit: AuditLogger) -> dict[str, float]:
        result: dict[str, float] = {}
        for event in audit.events:
            result[event.event_type] = result.get(event.event_type, 0.0) + event.duration_ms
        return result

    def _run_post_fusion_enrichment(
        self,
        state: CaseState,
        detector_input: DetectorInput,
        final_fusion,
        audit: AuditLogger,
    ) -> None:
        if self.enrichment_policy != "post_fusion":
            audit.log(
                "EnrichmentPolicy",
                "ENRICHMENT_SKIPPED",
                output_summary={"policy": self.enrichment_policy},
                reason="Post-fusion enrichment is not enabled.",
            )
            return
        immutable_before = final_fusion.model_dump(mode="json")
        for agent in self.enrichment_agents.values():
            result = agent.analyze(detector_input.model_copy(deep=True))
            if result.contributes_to_verdict:
                raise RuntimeError(
                    f"enrichment agent may not contribute to verdict: {agent.name}"
                )
            state.enrichment_results.append(result)
            audit.log(
                agent.name,
                "ENRICHMENT_RESULT",
                input_summary={
                    "input_schema": "DetectorInput",
                    "post_fusion": True,
                    "budget_cost": 0,
                },
                output_summary=result.model_dump(mode="json"),
                reason=(
                    "Post-fusion enrichment is advisory and cannot trigger "
                    "another Fusion call."
                ),
                duration_ms=result.latency_ms,
            )
        if final_fusion.model_dump(mode="json") != immutable_before:
            raise RuntimeError("post-fusion enrichment modified FusionResult")
        audit.log(
            "EnrichmentPolicy",
            "ENRICHMENT_COMPLETED",
            output_summary={
                "policy": self.enrichment_policy,
                "agents": [
                    item.agent_name for item in state.enrichment_results
                ],
                "detection_budget_used": 0,
                "fusion_recomputed": False,
            },
            reason="Enrichment completed after immutable final Fusion.",
        )

    def _run_deliberation(
        self,
        state: CaseState,
        detector_input: DetectorInput,
        audit: AuditLogger,
    ) -> None:
        if not self.future_flags.deliberation or state.interim_fusion is None:
            return
        report = self.deliberation_service.analyze(
            state,
            state.interim_fusion,
            available_agents=set(self.detectors),
            audit=audit,
        )
        state.deliberations.append(report)
        state.future_artifacts.deliberations.append(report)
        if (
            self.future_flags.shadow_mode
            or not report.recommended_dispatch
            or state.remaining_budget <= 0
        ):
            return
        proposed = CoordinatorDecision(
            action=CoordinatorAction.DISPATCH,
            agents=list(report.recommended_dispatch),
            reason_codes=["CONFLICT_DELIBERATION_REDISPATCH"],
            rationale="Deliberation requested additional independent evidence.",
            remaining_budget=state.remaining_budget,
            source="rule",
        )
        guarded = self.deliberation_guard.enforce(
            proposed,
            state,
            self.deliberation_fallback,
        )
        if guarded.action == CoordinatorAction.DISPATCH and self.utility_policy:
            accepted_agents: list[str] = []
            adjustments = list(guarded.policy_adjustments)
            for agent in guarded.agents:
                if agent not in {
                    "TemporalBehaviorAgent",
                    "TLSProtocolAgent",
                }:
                    accepted_agents.append(agent)
                    continue
                estimate = self.utility_policy.estimate(state, agent)
                state.future_artifacts.utility_estimates.append(estimate)
                if estimate.should_dispatch:
                    accepted_agents.append(agent)
                else:
                    adjustments.append(
                        f"removed non-positive-utility agent: {agent}"
                    )
            guarded = guarded.model_copy(
                update={
                    "agents": accepted_agents,
                    "policy_adjustments": adjustments,
                    "policy_status": (
                        "adjusted" if adjustments else guarded.policy_status
                    ),
                    "action": (
                        CoordinatorAction.DISPATCH
                        if accepted_agents
                        else CoordinatorAction.STOP_AND_FUSE
                    ),
                }
            )
        state.decisions.append(guarded)
        audit.log(
            "PolicyGuard",
            "DELIBERATION_REDISPATCH_EVALUATED",
            input_summary=proposed.model_dump(mode="json"),
            output_summary=guarded.model_dump(mode="json"),
            reason="Deliberation recommendations cannot bypass PolicyGuard.",
        )
        if guarded.action != CoordinatorAction.DISPATCH:
            return
        selected = [
            self.detectors[name]
            for name in guarded.agents
            if name in self.detectors and name not in state.called_agents
        ]
        if not selected:
            return
        self._run_detectors(selected, detector_input, state, audit)
        state.called_agents.update(agent.name for agent in selected)
        state.remaining_budget = max(0, state.remaining_budget - len(selected))
        state.interim_fusion = self.fusion_agent.fuse(
            state.evidence,
            state.reliability,
            final=False,
        )
        self._log_fusion(audit, state, state.interim_fusion, final=False)

    def _run_detectors(
        self,
        agents: list[DetectorAgent],
        detector_input: DetectorInput,
        state: CaseState,
        audit: AuditLogger,
    ) -> None:
        if self.max_workers == 1 or len(agents) == 1:
            for agent in agents:
                envelope = self.evidence_stability.analyze(
                    agent,
                    detector_input.model_copy(deep=True),
                )
                self._record_detector_result(
                    agent,
                    envelope.evidence,
                    detector_input,
                    state,
                    audit,
                    stability_metadata=envelope.audit_metadata,
                )
            return
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(agents))) as pool:
            futures = {
                pool.submit(
                    self.evidence_stability.analyze,
                    agent,
                    detector_input.model_copy(deep=True),
                ): agent
                for agent in agents
            }
            for future, agent in futures.items():
                envelope = future.result()
                self._record_detector_result(
                    agent,
                    envelope.evidence,
                    detector_input,
                    state,
                    audit,
                    stability_metadata=envelope.audit_metadata,
                )

    def _record_detector_result(
        self,
        agent: DetectorAgent,
        result,
        detector_input: DetectorInput,
        state: CaseState,
        audit: AuditLogger,
        stability_metadata: dict | None = None,
    ) -> None:
        state.evidence.append(result)
        visible_fields = detector_input.visible_field_paths()
        input_summary = {
            "input_schema": "DetectorInput",
            "visible_fields": visible_fields,
            "blocked_field_intersection": sorted(
                set(visible_fields)
                & {
                    decision.path
                    for decision in state.field_audit.decisions
                    if decision.role
                    in {
                        FieldRole.BLOCKED,
                        FieldRole.LABEL_ONLY,
                        FieldRole.PROVENANCE,
                        FieldRole.UNKNOWN,
                    }
                }
            )
            if state.field_audit
            else [],
        }
        if (
            stability_metadata
            and stability_metadata.get("policy") != "off"
        ):
            input_summary["evidence_stability"] = stability_metadata
        audit.log(
            agent.name,
            "AGENT_EVIDENCE",
            input_summary=input_summary,
            output_summary=self._agent_evidence_summary(result),
            reason="Specialist evidence returned to the shared case state.",
            duration_ms=result.latency_ms,
        )

    def _log_coordination(
        self,
        audit: AuditLogger,
        state: CaseState,
        decision: CoordinatorDecision,
    ) -> None:
        plan_payload = decision.planner_metadata.get("execution_plan")
        result_payload = decision.planner_metadata.get("executor_result")
        if plan_payload:
            state.future_artifacts.execution_plans.append(
                ExecutionPlan.model_validate(plan_payload)
            )
        if result_payload:
            state.future_artifacts.executor_results.append(
                ExecutorResult.model_validate(result_payload)
            )
        audit.log(
            "CoordinatorAgent",
            "ROUTING_DECISION",
            input_summary={
                "round": state.round_no,
                "called_agents": sorted(state.called_agents),
                "remaining_budget": state.remaining_budget,
                "interim_fusion": state.interim_fusion.model_dump(mode="json")
                if state.interim_fusion
                else None,
            },
            output_summary=decision.model_dump(mode="json"),
            reason=decision.rationale,
        )
        audit.log(
            "PolicyGuard",
            "POLICY_GUARD_EVALUATION",
            input_summary={
                "proposed_action": decision.proposed_action,
                "proposed_agents": decision.proposed_agents,
            },
            output_summary={
                "status": decision.policy_status,
                "effective_action": decision.action,
                "effective_agents": decision.agents,
                "adjustments": decision.policy_adjustments,
                "remaining_budget": decision.remaining_budget,
                "utility_estimates": [
                    item.model_dump(mode="json")
                    for item in state.future_artifacts.utility_estimates
                ],
            },
            reason="Coordinator action was evaluated against routing policy.",
        )

    def _log_fusion(
        self,
        audit: AuditLogger,
        state: CaseState,
        fusion_result,
        *,
        final: bool,
    ) -> None:
        audit.log(
            self.fusion_agent.name,
            "FINAL_FUSION" if final else "INTERIM_FUSION",
            input_summary=self._fusion_input_summary(state),
            output_summary=fusion_result.model_dump(mode="json"),
            reason=(
                "FusionAgent exclusively owns the final verdict."
                if final
                else "Assess whether additional evidence is worth its cost."
            ),
        )


def build_default_engine(
    *,
    use_nvidia: bool = False,
    nvidia_api_key: str | None = None,
    nvidia_model: str | None = None,
    nvidia_timeout_seconds: float = 20,
    nvidia_max_retries: int = 3,
    nvidia_retry_backoff_seconds: float = 1.0,
    nvidia_min_request_interval_seconds: float = 0.25,
    nvidia_response_format_json: bool = False,
    nvidia_disable_thinking: bool = False,
    nvidia_max_tokens: int = 350,
    cache_dir: str | Path | None = ".cache/coordinator",
    enable_field_audit: bool = True,
    field_audit_mode: str = "strict",
    field_contract_version: str = "2.8",
    formal_acceptance: bool = False,
    enable_pseudo_guard: bool = True,
    use_reliability_discount: bool = True,
    allow_reject: bool = True,
    execution_policy: str = "dynamic",
    force_rule_coordinator: bool = False,
    detector_backend: str = "rule",
    model_dir: str | Path | None = None,
    stats_backend: str | None = None,
    stats_model_dir: str | Path | None = None,
    temporal_model_dir: str | Path | None = None,
    tls_backend: str | None = None,
    tls_model_dir: str | Path | None = None,
    max_workers: int = 4,
    ood_policy: str = "off",
    ood_gate_dir: str | Path | None = None,
    base_ood_policy: str = "off",
    base_ood_gate_dir: str | Path | None = None,
    enable_rag: bool = False,
    knowledge_base_dir: str | Path = "knowledge_base",
    future_flags: FutureFeatureFlags | None = None,
    memory_dir: str | Path = "data/memory",
    utility_model_dir: str | Path | None = None,
    evidence_stability_policy: str = "off",
    orbit_reliability_dir: str | Path | None = None,
    routing_policy: str = "legacy",
    enrichment_policy: str = "legacy_inline",
    audit_payload_policy: str = "full",
    latency_profile: bool = False,
) -> DetectionEngine:
    if field_audit_mode not in {"strict", "legacy"}:
        raise ValueError(f"unsupported field audit mode: {field_audit_mode}")
    if field_contract_version not in {"2.8", "2.9"}:
        raise ValueError(
            f"unsupported field contract version: {field_contract_version}"
        )
    if routing_policy not in {
        "legacy",
        "contract_v2_9",
        "capability_v3_0",
        "efficiency_v1",
        "efficiency_v1_1",
        "efficiency_v1_2",
    }:
        raise ValueError(f"unsupported routing policy: {routing_policy}")
    contract_routing_policy = (
        "capability_v3_0"
        if routing_policy in {"efficiency_v1", "efficiency_v1_1", "efficiency_v1_2"}
        else routing_policy
    )
    if enrichment_policy not in {"legacy_inline", "off", "post_fusion"}:
        raise ValueError(
            f"unsupported enrichment policy: {enrichment_policy}"
        )
    if audit_payload_policy not in {"full", "compact"}:
        raise ValueError(
            f"unsupported audit payload policy: {audit_payload_policy}"
        )
    if formal_acceptance and not enable_field_audit:
        raise ValueError("formal acceptance cannot disable FieldAudit")
    if formal_acceptance and field_audit_mode != "strict":
        raise ValueError("formal acceptance requires strict FieldAudit")
    if evidence_stability_policy not in EVIDENCE_STABILITY_POLICIES:
        raise ValueError(
            "unsupported evidence stability policy: "
            f"{evidence_stability_policy}"
        )
    if detector_backend not in {
        "rule",
        "learned",
        "deep_v2",
        "deep_v2_all_length",
        "deep_v2_2",
        "deep_v2_3",
        "deep_v2_6_continuous",
        "deep_v3_1_prefix",
    }:
        raise ValueError(f"unsupported detector backend: {detector_backend}")
    if detector_backend in {
        "learned",
        "deep_v2",
        "deep_v2_all_length",
        "deep_v2_2",
        "deep_v2_3",
        "deep_v2_6_continuous",
        "deep_v3_1_prefix",
    } and model_dir is None:
        raise ValueError(
            "model_dir is required when detector_backend is learned or deep_v2"
        )
    if detector_backend not in {
        "deep_v2",
        "deep_v2_all_length",
        "deep_v2_2",
        "deep_v2_3",
        "deep_v2_6_continuous",
        "deep_v3_1_prefix",
    } and ood_policy in {
        "conformal_v3",
        "conformal_v3_1",
        "conformal_v3_2",
        "conformal_v3_3",
        "conformal_v3_4_orbit",
        "conformal_v3_1_prefix",
    }:
        raise ValueError(f"{ood_policy} requires detector_backend='deep_v2'")
    if (
        detector_backend
        not in {
            "deep_v2",
            "deep_v2_all_length",
            "deep_v2_2",
            "deep_v2_3",
            "deep_v2_6_continuous",
        }
        and ood_policy != "off"
        and detector_backend != "learned"
    ):
        raise ValueError("OOD gating requires a learned detector backend")
    if (
        ood_policy != "off"
        and ood_gate_dir is None
        and not (
            detector_backend in {
                "deep_v2",
                "deep_v2_all_length",
                "deep_v2_2",
                "deep_v2_3",
                "deep_v2_6_continuous",
                "deep_v3_1_prefix",
            }
            and ood_policy in {
                "conformal_v3",
                "conformal_v3_1",
                "conformal_v3_2",
                "conformal_v3_3",
                "conformal_v3_4_orbit",
                "conformal_v3_1_prefix",
            }
        )
    ):
        raise ValueError("ood_gate_dir is required when OOD policy is enabled")
    learned_root = Path(model_dir) if model_dir is not None else None
    stats_root = (
        Path(stats_model_dir)
        if stats_model_dir is not None
        else learned_root / "stats"
        if learned_root is not None
        else None
    )
    temporal_root = (
        Path(temporal_model_dir)
        if temporal_model_dir is not None
        else learned_root / "temporal"
        if learned_root is not None
        else None
    )
    resolved_tls_backend = tls_backend or (
        "deep_v2"
        if detector_backend == "deep_v2"
        and learned_root is not None
        and (learned_root / "tls" / "model.pt").exists()
        else "rule"
    )
    if resolved_tls_backend not in {"rule", "deep_v2", "deep_tls_v4"}:
        raise ValueError(f"unsupported TLS backend: {resolved_tls_backend}")
    tls_root = (
        Path(tls_model_dir)
        if tls_model_dir is not None
        else learned_root / "tls"
        if resolved_tls_backend == "deep_v2" and learned_root is not None
        else None
    )
    if (
        resolved_tls_backend in {"deep_v2", "deep_tls_v4"}
        and (
            tls_root is None
            or not (tls_root / "model.pt").exists()
            or not (tls_root / "metadata.json").exists()
        )
    ):
        raise ValueError("learned TLS backend requires a complete model artifact")
    gate_root = Path(ood_gate_dir) if ood_gate_dir is not None else None
    allowed_detection_agents = (
        set(VERDICT_AGENT_NAMES)
        if routing_policy
        in {
            "contract_v2_9",
            "capability_v3_0",
            "efficiency_v1",
            "efficiency_v1_1",
            "efficiency_v1_2",
        }
        else set(DEFAULT_ALLOWED_AGENTS)
    )
    fallback = RuleCoordinator(
        routing_policy=contract_routing_policy,
        enrichment_policy=enrichment_policy,
        allowed_agents=allowed_detection_agents,
    )
    planner: Coordinator
    if routing_policy == "efficiency_v1":
        planner = EvidenceBudgetRouterV1()
    elif routing_policy == "efficiency_v1_1":
        planner = EvidenceBudgetRouterV11()
    elif routing_policy == "efficiency_v1_2":
        planner = EvidenceBudgetRouterV12()
    elif use_nvidia and not force_rule_coordinator:
        planner = NvidiaCoordinator(
            api_key=nvidia_api_key,
            model=nvidia_model,
            timeout_seconds=nvidia_timeout_seconds,
            max_retries=nvidia_max_retries,
            retry_backoff_seconds=nvidia_retry_backoff_seconds,
            min_request_interval_seconds=nvidia_min_request_interval_seconds,
            response_format_json=nvidia_response_format_json,
            disable_thinking=nvidia_disable_thinking,
            max_tokens=nvidia_max_tokens,
            cache_dir=cache_dir,
            fallback=fallback,
            allowed_agents=allowed_detection_agents,
        )
    else:
        planner = fallback
    flags = future_flags or FutureFeatureFlags()
    utility_policy = None
    if flags.evidence_utility_v2 and flags.evidence_utility_v2_1:
        raise ValueError("only one evidence utility policy may be enabled")
    if flags.evidence_utility_v2 or flags.evidence_utility_v2_1:
        if utility_model_dir is None:
            raise ValueError(
                "utility_model_dir is required when an evidence utility "
                "policy is enabled"
            )
        if flags.evidence_utility_v2_1:
            from .utility import EvidenceUtilityPolicyV21

            utility_policy = EvidenceUtilityPolicyV21(utility_model_dir)
        else:
            from .utility import EvidenceUtilityPolicyV2

            utility_policy = EvidenceUtilityPolicyV2(utility_model_dir)
    coordinator = (
        PlannerExecutorCoordinator(
            planner=CoordinatorExecutionPlanner(planner),
            plan_guard=PlanPolicyGuard(
                allowed_agents=allowed_detection_agents,
                utility_policy=utility_policy,
            ),
            legacy_guard=PolicyGuard(
                allowed_agents=allowed_detection_agents,
                max_rounds=3,
                max_total_calls=(
                    2
                    if routing_policy
                    in {"efficiency_v1", "efficiency_v1_1", "efficiency_v1_2"}
                    else 4
                ),
            ),
            fallback=fallback,
            utility_policy=utility_policy,
        )
        if flags.planner_executor
        else GuardedCoordinator(
            planner,
            policy_guard=PolicyGuard(
                allowed_agents=allowed_detection_agents,
                max_rounds=3,
                max_total_calls=(
                    2
                    if routing_policy
                    in {"efficiency_v1", "efficiency_v1_1", "efficiency_v1_2"}
                    else 4
                ),
            ),
            fallback=fallback,
        )
    )
    knowledge_service = (
        KnowledgeRetrievalService(knowledge_base_dir) if enable_rag else None
    )
    tls_agent = TLSProtocolAgent(
        backend=resolved_tls_backend,
        model_dir=tls_root,
        ood_policy=(
            ood_policy
            if resolved_tls_backend == "deep_v2"
            else "off"
        ),
        ood_gate_dir=(
            gate_root / "tls" / ood_policy
            if resolved_tls_backend == "deep_v2"
            and gate_root is not None
            and (gate_root / "tls" / ood_policy).exists()
            else gate_root / "tls"
            if resolved_tls_backend == "deep_v2"
            and gate_root is not None
            else tls_root / ood_policy
            if resolved_tls_backend == "deep_v2"
            and tls_root is not None
            else None
        ),
        contract_native_fields=routing_policy
        in {
            "contract_v2_9",
            "capability_v3_0",
            "efficiency_v1",
            "efficiency_v1_1",
            "efficiency_v1_2",
        },
    )
    verdict_detectors = [
        StatsDetectorAgent(
            backend=stats_backend
            or (
                "learned"
                if detector_backend in {
                    "deep_v2",
                    "deep_v2_all_length",
                    "deep_v2_2",
                    "deep_v2_3",
                    "deep_v2_6_continuous",
                    "deep_v3_1_prefix",
                }
                else detector_backend
            ),
            model_dir=stats_root,
            ood_policy=(
                base_ood_policy
                if detector_backend in {
                    "deep_v2",
                    "deep_v2_all_length",
                    "deep_v2_2",
                    "deep_v2_3",
                    "deep_v2_6_continuous",
                    "deep_v3_1_prefix",
                }
                else ood_policy
            ),
            ood_gate_dir=(
                Path(base_ood_gate_dir) / "stats"
                if detector_backend in {
                    "deep_v2",
                    "deep_v2_all_length",
                    "deep_v2_2",
                    "deep_v2_3",
                    "deep_v2_6_continuous",
                    "deep_v3_1_prefix",
                }
                and base_ood_gate_dir is not None
                else gate_root / "stats"
                if gate_root is not None
                and detector_backend == "learned"
                else None
            ),
        ),
        TemporalBehaviorAgent(
            backend=detector_backend,
            model_dir=temporal_root,
            ood_policy=ood_policy,
            ood_gate_dir=(
                None
                if ood_policy == "off"
                else gate_root / "temporal" / ood_policy
                if gate_root is not None
                and (gate_root / "temporal" / ood_policy).exists()
                else gate_root / "temporal"
                if gate_root is not None
                else learned_root / "temporal" / ood_policy
                if detector_backend in {
                    "deep_v2",
                    "deep_v2_all_length",
                    "deep_v2_2",
                    "deep_v2_3",
                    "deep_v2_6_continuous",
                    "deep_v3_1_prefix",
                }
                and temporal_root is not None
                else None
            ),
        ),
        tls_agent,
    ]
    enrichment_agents = [FamilyAttributionAgent(), IntentAgent()]
    legacy_extra_agents = [
        *enrichment_agents,
        PayloadRepresentationAgent(),
    ]
    return DetectionEngine(
        coordinator=coordinator,
        detectors=(
            verdict_detectors + legacy_extra_agents
            if enrichment_policy == "legacy_inline"
            else verdict_detectors
        ),
        enrichment_agents=(
            enrichment_agents
            if enrichment_policy in {"off", "post_fusion"}
            else []
        ),
        field_auditor=FieldAuditAgent(
            AuditPolicy(
                mode=field_audit_mode,
                contract_version=field_contract_version,
            )
        ),
        feature_guard=PseudoFeatureGuardAgent(
            routing_policy=contract_routing_policy,
            tls_backend=tls_agent.backend,
        ),
        enable_field_audit=enable_field_audit,
        enable_pseudo_guard=enable_pseudo_guard,
        execution_policy=execution_policy,
        fusion_agent=FusionAgent(
            use_reliability_discount=use_reliability_discount,
            allow_reject=allow_reject,
        ),
        reporter=ReporterAgent(knowledge_service=knowledge_service),
        max_workers=max_workers,
        future_flags=flags,
        memory_service=JsonlCaseMemory(memory_dir) if flags.memory else None,
        utility_policy=utility_policy,
        evidence_stability_policy=evidence_stability_policy,
        orbit_reliability_dir=orbit_reliability_dir,
        routing_policy=routing_policy,
        enrichment_policy=enrichment_policy,
        audit_payload_policy=audit_payload_policy,
        latency_profile=latency_profile,
    )
