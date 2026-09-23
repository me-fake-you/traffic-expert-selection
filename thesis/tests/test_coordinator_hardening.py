import requests

from mad_etd.base import CaseState
from mad_etd.coordinator import (
    GuardedCoordinator,
    NvidiaCoordinator,
    PolicyGuard,
    RuleCoordinator,
)
from mad_etd.engine import build_default_engine
from mad_etd.orchestration import (
    CoordinatorExecutionPlanner,
    PlannerExecutorCoordinator,
)
from mad_etd.schemas import CoordinatorAction, CoordinatorDecision


class _FakeResponse:
    def __init__(
        self,
        content: str,
        *,
        usage=None,
        response_id="response-1",
        status_code=200,
        headers=None,
    ):
        self.content = content
        self.usage = usage
        self.response_id = response_id
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"status {self.status_code}", response=self
            )
        return None

    def json(self):
        payload = {
            "id": self.response_id,
            "choices": [{"message": {"content": self.content}}],
        }
        if self.usage is not None:
            payload["usage"] = self.usage
        return payload


def _nvidia_state(flow):
    return CaseState(flow=flow, safe_flow=flow.model_copy(deep=True), remaining_budget=4)


def test_nvidia_invalid_json_falls_back(monkeypatch, benign_flow):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _FakeResponse("not-json"),
    )
    coordinator = NvidiaCoordinator(
        api_key="test-key",
        cache_dir=None,
        fallback=RuleCoordinator(),
        max_retries=0,
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "fallback"
    assert decision.action == CoordinatorAction.DISPATCH
    assert any("ValueError" in item for item in decision.policy_adjustments)


def test_nvidia_timeout_falls_back(monkeypatch, benign_flow):
    def timeout(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(requests, "post", timeout)
    coordinator = NvidiaCoordinator(
        api_key="test-key",
        cache_dir=None,
        fallback=RuleCoordinator(),
        max_retries=0,
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "fallback"
    assert decision.planner_metadata["fallback_reason"] == "Timeout"
    assert decision.planner_metadata["llm_call_attempted"] is True


def test_llm_direct_final_verdict_is_rejected(monkeypatch, benign_flow):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            '{"action":"STOP_AND_FUSE","final_verdict":"malicious"}'
        ),
    )
    coordinator = NvidiaCoordinator(
        api_key="test-key", cache_dir=None, fallback=RuleCoordinator()
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "fallback"
    assert decision.action == CoordinatorAction.DISPATCH
    assert "final_verdict" in decision.raw_response
    assert decision.planner_metadata["illegal_verdict_attempt_count"] == 1
    assert decision.planner_metadata["real_llm"] is False


def test_llm_field_audit_override_is_rejected(monkeypatch, benign_flow):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            '{"action":"DISPATCH","agents":["StatsDetectorAgent"],'
            '"field_audit_override":{"blocked_fields":[]}}'
        ),
    )
    coordinator = NvidiaCoordinator(
        api_key="test-key", cache_dir=None, fallback=RuleCoordinator()
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "fallback"
    assert "field_audit_override" in decision.raw_response
    assert decision.planner_metadata["blocked_field_access_attempt_count"] >= 1


def test_nvidia_success_records_usage_latency_and_raw_hash(
    monkeypatch, benign_flow
):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            '{"action":"DISPATCH","agents":["StatsDetectorAgent"],'
            '"reason_codes":["STATS"],"rationale":"Route only."}',
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        ),
    )
    coordinator = NvidiaCoordinator(
        api_key="test-key",
        model="test-model",
        cache_dir=None,
        fallback=RuleCoordinator(),
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "nvidia"
    assert decision.planner_metadata["real_llm"] is True
    assert decision.planner_metadata["response_schema_valid"] is True
    assert decision.planner_metadata["usage"]["total_tokens"] == 30
    assert decision.planner_metadata["api_latency_ms"] >= 0
    assert len(decision.planner_metadata["raw_response_sha256"]) == 64


def test_nemotron_ultra_request_profile_is_forwarded(
    monkeypatch, benign_flow
):
    captured = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _FakeResponse(
            '{"action":"DISPATCH","agents":["StatsDetectorAgent"],'
            '"reason_codes":["STATS"],"rationale":"Route only."}'
        )

    monkeypatch.setattr(requests, "post", post)
    coordinator = NvidiaCoordinator(
        api_key="test-key",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        cache_dir=None,
        response_format_json=True,
        disable_thinking=True,
        max_tokens=350,
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "nvidia"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["max_tokens"] == 350
    assert decision.planner_metadata["response_format_json"] is True
    assert decision.planner_metadata["disable_thinking"] is True


def test_cached_over_budget_nvidia_plan_is_guarded_on_replay(
    monkeypatch, benign_flow, tmp_path
):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(
            '{"action":"DISPATCH","agents":['
            '"StatsDetectorAgent","TemporalBehaviorAgent",'
            '"TLSProtocolAgent"],"reason_codes":["MULTI"],'
            '"rationale":"Request all available views."}'
        ),
    )
    nvidia = NvidiaCoordinator(
        api_key="test-key",
        cache_dir=tmp_path,
        fallback=RuleCoordinator(),
        min_request_interval_seconds=0,
    )
    coordinator = PlannerExecutorCoordinator(
        planner=CoordinatorExecutionPlanner(nvidia)
    )
    first_state = _nvidia_state(benign_flow)
    first_state.remaining_budget = 1
    second_state = _nvidia_state(benign_flow)
    second_state.remaining_budget = 1

    first = coordinator.decide(first_state)
    second = coordinator.decide(second_state)

    assert first.source == "nvidia"
    assert second.source == "replay"
    assert first.agents == second.agents == ["StatsDetectorAgent"]
    assert first.policy_status == second.policy_status == "adjusted"
    assert any("over-budget" in item for item in first.policy_adjustments)
    assert any("over-budget" in item for item in second.policy_adjustments)


def test_nvidia_retries_429_and_records_attempts(monkeypatch, benign_flow):
    responses = iter(
        [
            _FakeResponse(
                "rate limited",
                status_code=429,
                headers={"Retry-After": "0"},
            ),
            _FakeResponse(
                '{"action":"DISPATCH","agents":["StatsDetectorAgent"],'
                '"reason_codes":["STATS"],"rationale":"Route only."}'
            ),
        ]
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: next(responses),
    )
    coordinator = NvidiaCoordinator(
        api_key="test-key",
        cache_dir=None,
        fallback=RuleCoordinator(),
        max_retries=2,
        retry_backoff_seconds=0,
        min_request_interval_seconds=0,
    )

    decision = coordinator.decide(_nvidia_state(benign_flow))

    assert decision.source == "nvidia"
    assert decision.planner_metadata["request_attempts"] == 2
    assert decision.planner_metadata["retry_count"] == 1


def test_policy_guard_truncates_over_budget_dispatch(benign_flow):
    state = _nvidia_state(benign_flow)
    state.remaining_budget = 1
    proposed = CoordinatorDecision(
        action=CoordinatorAction.DISPATCH,
        agents=[
            "StatsDetectorAgent",
            "TemporalBehaviorAgent",
            "TLSProtocolAgent",
        ],
        remaining_budget=1,
        source="nvidia",
    )

    decision = PolicyGuard().enforce(proposed, state, RuleCoordinator())

    assert len(decision.agents) == 1
    assert decision.remaining_budget == 0
    assert decision.policy_status == "adjusted"
    assert any("remaining capacity" in item for item in decision.policy_adjustments)


def test_policy_guard_blocks_duplicate_without_new_input(benign_flow):
    state = _nvidia_state(benign_flow)
    state.called_agents.add("StatsDetectorAgent")
    proposed = CoordinatorDecision(
        action=CoordinatorAction.DISPATCH,
        agents=["StatsDetectorAgent"],
        remaining_budget=3,
        source="nvidia",
    )

    decision = PolicyGuard().enforce(proposed, state, RuleCoordinator())

    assert "StatsDetectorAgent" not in decision.agents
    assert any("duplicate" in item for item in decision.policy_adjustments)


def test_rule_coordinator_completes_multiple_rounds(suspicious_flow):
    report, _ = build_default_engine(force_rule_coordinator=True).analyze(
        suspicious_flow
    )

    assert len(report.coordinator_decisions) >= 2
    assert report.coordinator_decisions[0].agents == [
        "StatsDetectorAgent",
        "TemporalBehaviorAgent",
    ]
    assert any(
        "TLSProtocolAgent" in decision.agents
        for decision in report.coordinator_decisions[1:]
    )
