from mad_etd.base import CaseState
from mad_etd.coordinator import PolicyGuard, RuleCoordinator
from mad_etd.io import load_flow_records
from mad_etd.schemas import CoordinatorAction, CoordinatorDecision


def test_policy_guard_removes_forbidden_and_unavailable_agents():
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    flow.tls = {}
    state = CaseState(flow=flow, safe_flow=flow, remaining_budget=2)
    proposed = CoordinatorDecision(
        action=CoordinatorAction.DISPATCH,
        agents=["ShellAgent", "TLSProtocolAgent"],
        reason_codes=["TEST"],
        remaining_budget=2,
        source="nvidia",
    )

    guarded = PolicyGuard().enforce(proposed, state, RuleCoordinator())

    assert "ShellAgent" not in guarded.agents
    assert "TLSProtocolAgent" not in guarded.agents
    assert guarded.source == "fallback"
    assert set(guarded.agents) == {
        "StatsDetectorAgent",
        "TemporalBehaviorAgent",
    }


def test_rule_coordinator_starts_with_independent_views():
    flow = load_flow_records("examples/sample_flows.jsonl")[0]
    state = CaseState(flow=flow, safe_flow=flow, remaining_budget=4)

    decision = RuleCoordinator().decide(state)

    assert decision.action == CoordinatorAction.DISPATCH
    assert decision.agents == [
        "StatsDetectorAgent",
        "TemporalBehaviorAgent",
    ]

