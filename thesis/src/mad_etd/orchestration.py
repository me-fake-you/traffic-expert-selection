from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .base import CaseState, Coordinator
from .coordinator import DEFAULT_ALLOWED_AGENTS, PolicyGuard, RuleCoordinator
from .schemas import (
    CoordinatorAction,
    CoordinatorDecision,
    ExecutionPlan,
    ExecutorResult,
    PlanAction,
    PlanStep,
)
from .utility import EvidenceUtilityPolicyV2, EvidenceUtilityPolicyV21


class CoordinatorExecutionPlanner:
    """Convert an existing rule/LLM routing proposal into a typed plan."""

    name = "CoordinatorExecutionPlanner"
    version = "1.0"

    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator

    def plan(self, state: CaseState) -> ExecutionPlan:
        decision = self.coordinator.decide(state)
        steps: list[PlanStep] = []
        for index, agent in enumerate(decision.agents):
            steps.append(
                PlanStep(
                    step_id=f"step-{index + 1}",
                    action=PlanAction.RUN_AGENT,
                    agent=agent,
                    budget_cost=1,
                )
            )
        terminal = (
            PlanAction.FUSE
            if decision.action in {
                CoordinatorAction.STOP_AND_FUSE,
                CoordinatorAction.ABSTAIN_AND_REPORT,
            }
            else PlanAction.STOP
        )
        if not steps:
            steps.append(
                PlanStep(
                    step_id="step-1",
                    action=terminal,
                    budget_cost=0,
                )
            )
        fingerprint = (
            f"{state.flow.trace_id}:{state.round_no}:"
            + ",".join(step.agent or step.action.value for step in steps)
        )
        proposed_budget = sum(step.budget_cost for step in steps)
        return ExecutionPlan(
            plan_id=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24],
            steps=steps,
            reason_codes=list(decision.reason_codes),
            # The typed proposal declares the cost of everything it requested.
            # PlanPolicyGuard separately enforces the actual runtime budget.
            budget=proposed_budget,
            source=(
                "llm"
                if decision.source == "nvidia"
                else decision.source
                if decision.source in {"rule", "fallback", "replay"}
                else "fallback"
            ),
            planner_metadata={
                **decision.planner_metadata,
                "planner_version": self.version,
                "raw_response": decision.raw_response,
                "proposed_action": decision.action.value,
                "proposed_agents": list(decision.agents),
                "proposed_budget": proposed_budget,
                "available_budget": state.remaining_budget,
            },
        )


class RuleExecutionPlanner(CoordinatorExecutionPlanner):
    name = "RuleExecutionPlanner"

    def __init__(self, fallback: RuleCoordinator | None = None) -> None:
        super().__init__(fallback or RuleCoordinator())


@dataclass(slots=True)
class ValidatedPlan:
    plan: ExecutionPlan
    adjustments: list[str]


class PlanPolicyGuard:
    name = "PlanPolicyGuard"
    version = "1.0"

    def __init__(
        self,
        *,
        allowed_agents: set[str] | None = None,
        max_steps: int = 4,
        utility_policy: (
            EvidenceUtilityPolicyV2 | EvidenceUtilityPolicyV21 | None
        ) = None,
    ) -> None:
        self.allowed_agents = allowed_agents or set(DEFAULT_ALLOWED_AGENTS)
        self.max_steps = max_steps
        self.utility_policy = utility_policy

    def validate(self, plan: ExecutionPlan, state: CaseState) -> ValidatedPlan:
        adjustments: list[str] = []
        steps: list[PlanStep] = []
        remaining = state.remaining_budget
        for step in plan.steps[: self.max_steps]:
            if step.action != PlanAction.RUN_AGENT:
                steps.append(step)
                continue
            assert step.agent is not None
            if step.agent not in self.allowed_agents:
                adjustments.append(f"removed non-whitelisted agent: {step.agent}")
                continue
            if step.agent in state.called_agents:
                adjustments.append(f"removed duplicate agent: {step.agent}")
                continue
            if remaining < step.budget_cost:
                adjustments.append(f"removed over-budget step: {step.step_id}")
                continue
            if (
                self.utility_policy is not None
                and step.agent
                in {"TemporalBehaviorAgent", "TLSProtocolAgent"}
            ):
                if "StatsDetectorAgent" not in state.called_agents:
                    adjustments.append(
                        f"deferred utility agent until Stats evidence: {step.agent}"
                    )
                    continue
                estimate = self.utility_policy.estimate(state, step.agent)
                state.future_artifacts.utility_estimates.append(estimate)
                if not estimate.should_dispatch:
                    adjustments.append(
                        f"removed non-positive-utility agent: {step.agent}"
                    )
                    continue
            remaining -= step.budget_cost
            steps.append(step)
        if len(plan.steps) > self.max_steps:
            adjustments.append(f"truncated plan to {self.max_steps} steps")
        if not any(step.action == PlanAction.RUN_AGENT for step in steps):
            steps = [
                PlanStep(
                    step_id="guard-stop",
                    action=PlanAction.FUSE,
                    budget_cost=0,
                )
            ]
        normalized = plan.model_copy(
            update={
                "steps": steps,
                "budget": state.remaining_budget,
                "planner_metadata": {
                    **plan.planner_metadata,
                    "guard_version": self.version,
                    "adjustments": adjustments,
                },
            }
        )
        return ValidatedPlan(normalized, adjustments)


class PlanExecutor:
    name = "PlanExecutor"
    version = "1.0"

    @staticmethod
    def translate(
        validated: ValidatedPlan,
        state: CaseState,
    ) -> tuple[CoordinatorDecision, ExecutorResult]:
        plan = validated.plan
        agents = [
            step.agent
            for step in plan.steps
            if step.action == PlanAction.RUN_AGENT and step.agent is not None
        ]
        if agents:
            action = CoordinatorAction.DISPATCH
        else:
            action = CoordinatorAction.STOP_AND_FUSE
        result = ExecutorResult(
            plan_id=plan.plan_id,
            executed_steps=[
                step.step_id
                for step in plan.steps
                if step.action == PlanAction.RUN_AGENT
            ],
            skipped_steps=[],
            failures=[],
            budget_used=sum(
                step.budget_cost
                for step in plan.steps
                if step.action == PlanAction.RUN_AGENT
            ),
        )
        decision = CoordinatorDecision(
            action=action,
            agents=agents,
            reason_codes=list(plan.reason_codes),
            rationale="Typed execution plan accepted by PlanPolicyGuard.",
            remaining_budget=max(0, state.remaining_budget - result.budget_used),
            source=plan.source if plan.source != "llm" else "nvidia",
            planner_metadata={
                **plan.planner_metadata,
                "execution_plan": plan.model_dump(mode="json"),
                "executor_result": result.model_dump(mode="json"),
            },
            policy_adjustments=list(validated.adjustments),
            policy_status=(
                "adjusted" if validated.adjustments else "accepted"
            ),
            proposed_action=CoordinatorAction(
                plan.planner_metadata.get(
                    "proposed_action",
                    action.value,
                )
            ),
            proposed_agents=list(
                plan.planner_metadata.get("proposed_agents", agents)
            ),
        )
        return decision, result


class PlannerExecutorCoordinator(Coordinator):
    """Coordinator-compatible typed planner/guard/executor research path."""

    def __init__(
        self,
        planner: CoordinatorExecutionPlanner | None = None,
        plan_guard: PlanPolicyGuard | None = None,
        legacy_guard: PolicyGuard | None = None,
        fallback: RuleCoordinator | None = None,
        utility_policy: (
            EvidenceUtilityPolicyV2 | EvidenceUtilityPolicyV21 | None
        ) = None,
    ) -> None:
        self.planner = planner or RuleExecutionPlanner()
        self.plan_guard = plan_guard or PlanPolicyGuard(
            utility_policy=utility_policy
        )
        self.legacy_guard = legacy_guard or PolicyGuard()
        self.fallback = fallback or RuleCoordinator()

    def decide(self, state: CaseState) -> CoordinatorDecision:
        plan = self.planner.plan(state)
        validated = self.plan_guard.validate(plan, state)
        decision, _ = PlanExecutor.translate(validated, state)
        return self.legacy_guard.enforce(decision, state, self.fallback)
