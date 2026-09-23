from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from pydantic import ValidationError

from .base import CaseState, Coordinator
from .schemas import (
    CoordinatorAction,
    CoordinatorDecision,
    LLMPlannerResponse,
)
from .view_contract import VERDICT_AGENT_NAMES


DEFAULT_ALLOWED_AGENTS = {
    "StatsDetectorAgent",
    "TemporalBehaviorAgent",
    "TLSProtocolAgent",
    "FamilyAttributionAgent",
    "IntentAgent",
}

LLM_FORBIDDEN_OWNERSHIP_FIELDS = {
    "verdict",
    "final_verdict",
    "classification",
    "label",
    "confidence",
    "uncertainty",
    "severity",
    "benign_support",
    "malicious_support",
    "distribution_shift_score",
    "need_escalation",
}
LLM_BLOCKED_FIELD_MARKERS = {
    "labels",
    "labels.",
    "provenance",
    "provenance.",
    "trace_id",
    "sample_id",
    "field_audit_override",
    "blocked_fields",
}
LLM_VERDICT_PATTERN = re.compile(
    r"\b(?:traffic|flow|sample)\s+(?:is|=|:)\s*"
    r"(?:benign|malicious|suspicious|unknown)\b",
    re.IGNORECASE,
)


class RuleCoordinator(Coordinator):
    """Deterministic routing policy used for fallback and reproducible baselines."""

    def __init__(
        self,
        *,
        routing_policy: str = "legacy",
        enrichment_policy: str = "legacy_inline",
        allowed_agents: set[str] | None = None,
    ) -> None:
        self.routing_policy = routing_policy
        self.enrichment_policy = enrichment_policy
        self.allowed_agents = allowed_agents or (
            set(VERDICT_AGENT_NAMES)
            if routing_policy == "contract_v2_9"
            else set(DEFAULT_ALLOWED_AGENTS)
        )

    @staticmethod
    def _tls_available(state: CaseState) -> bool:
        capability = state.detector_capabilities.get("TLSProtocolAgent")
        if (
            capability is not None
            and state.view_availability is not None
            and state.view_availability.routing_policy == "capability_v3_0"
        ):
            return capability.status == "available"
        if state.view_availability is not None:
            return state.view_availability.tls
        flow = state.safe_flow or state.flow
        return bool(flow.tls)

    def decide(self, state: CaseState) -> CoordinatorDecision:
        available_budget = state.remaining_budget
        flow = state.safe_flow or state.flow

        initial = [
            name
            for name, available in (
                (
                    "StatsDetectorAgent",
                    (
                        state.detector_capabilities["StatsDetectorAgent"].status
                        == "available"
                        if (
                            state.view_availability is not None
                            and state.view_availability.routing_policy
                            == "capability_v3_0"
                            and "StatsDetectorAgent"
                            in state.detector_capabilities
                        )
                        else bool(flow.stats)
                    ),
                ),
                (
                    "TemporalBehaviorAgent",
                    (
                        state.detector_capabilities[
                            "TemporalBehaviorAgent"
                        ].status
                        == "available"
                        if (
                            state.view_availability is not None
                            and state.view_availability.routing_policy
                            == "capability_v3_0"
                            and "TemporalBehaviorAgent"
                            in state.detector_capabilities
                        )
                        else bool(flow.sequence.packet_lengths)
                    ),
                ),
            )
            if available and name not in state.called_agents
        ]
        if initial:
            return CoordinatorDecision(
                action=CoordinatorAction.DISPATCH,
                agents=initial[:available_budget],
                reason_codes=["INITIAL_MULTI_VIEW_EVIDENCE"],
                rationale="Collect independent statistical and temporal evidence.",
                remaining_budget=max(0, available_budget - len(initial[:available_budget])),
                source="rule",
            )

        if state.memory_hint and state.memory_hint.status == "success":
            hinted = [
                name
                for name in state.memory_hint.dispatch_hint
                if name in self.allowed_agents
                and name not in state.called_agents
            ]
            if hinted:
                return CoordinatorDecision(
                    action=CoordinatorAction.DISPATCH,
                    agents=hinted[:available_budget],
                    reason_codes=["MEMORY_DISPATCH_HINT"],
                    rationale=(
                        "Use advisory historical routing hints; PolicyGuard "
                        "still enforces availability and evidence prerequisites."
                    ),
                    remaining_budget=max(
                        0, available_budget - len(hinted[:available_budget])
                    ),
                    source="rule",
                )

        fusion = state.interim_fusion
        if fusion and not fusion.need_escalation:
            family_needed = (
                self.enrichment_policy == "legacy_inline"
                and
                fusion.malicious_support >= 0.6
                and "FamilyAttributionAgent" not in state.called_agents
                and available_budget > 0
            )
            if family_needed:
                return CoordinatorDecision(
                    action=CoordinatorAction.DISPATCH,
                    agents=["FamilyAttributionAgent"],
                    reason_codes=["MALICIOUS_SUPPORT_FOR_ATTRIBUTION"],
                    rationale="Binary evidence is strong enough to attempt family attribution.",
                    remaining_budget=available_budget - 1,
                    source="rule",
                )
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["DECISION_THRESHOLD_REACHED"],
                rationale="Current evidence satisfies a terminal fusion condition.",
                remaining_budget=available_budget,
                source="rule",
            )

        if (
            self._tls_available(state)
            and "TLSProtocolAgent" not in state.called_agents
            and available_budget > 0
        ):
            return CoordinatorDecision(
                action=CoordinatorAction.DISPATCH,
                agents=["TLSProtocolAgent"],
                reason_codes=["UNCERTAINTY_REQUIRES_PROTOCOL_VIEW"],
                rationale="Use protocol metadata to resolve uncertainty or conflict.",
                remaining_budget=available_budget - 1,
                source="rule",
            )

        if (
            self.enrichment_policy == "legacy_inline"
            and
            fusion
            and fusion.malicious_support >= 0.6
            and "FamilyAttributionAgent" not in state.called_agents
            and available_budget > 0
        ):
            return CoordinatorDecision(
                action=CoordinatorAction.DISPATCH,
                agents=["FamilyAttributionAgent"],
                reason_codes=["RISK_EVIDENCE_FOR_ATTRIBUTION"],
                rationale="Attempt attribution while preserving an unknown outcome.",
                remaining_budget=available_budget - 1,
                source="rule",
            )

        return CoordinatorDecision(
            action=CoordinatorAction.STOP_AND_FUSE,
            reason_codes=["NO_PRODUCTIVE_AGENT_REMAINS"],
            rationale="No uncalled compatible agent can materially reduce uncertainty.",
            remaining_budget=available_budget,
            source="rule",
        )


class EvidenceBudgetRouterV1(Coordinator):
    """Routing-only efficiency candidate.

    The router never emits verdict fields. It dispatches Stats first, then uses
    only already-produced evidence and capability status to decide whether a
    second evidence view is worth its budget.
    """

    name = "EvidenceBudgetRouterV1"
    version = "1.0"

    def __init__(
        self,
        *,
        suspicious_temporal_min_shift: float = 0.58,
        unknown_temporal_min_shift: float = 0.35,
        unknown_temporal_max_shift: float = 0.85,
        max_evidence_calls: int = 2,
    ) -> None:
        self.suspicious_temporal_min_shift = suspicious_temporal_min_shift
        self.unknown_temporal_min_shift = unknown_temporal_min_shift
        self.unknown_temporal_max_shift = unknown_temporal_max_shift
        self.max_evidence_calls = max_evidence_calls

    @staticmethod
    def _available(state: CaseState, agent: str) -> bool:
        capability = state.detector_capabilities.get(agent)
        return capability is not None and capability.status == "available"

    @staticmethod
    def _stats_evidence(state: CaseState):
        for item in reversed(state.evidence):
            if item.agent_name == "StatsDetectorAgent":
                return item
        return None

    def _should_supplement_after_stats(self, state: CaseState) -> bool:
        stats = self._stats_evidence(state)
        fusion = state.interim_fusion
        if stats is None or fusion is None:
            return False
        shift = max(
            stats.distribution_shift_score,
            fusion.distribution_shift_score,
        )
        if (
            stats.malicious_support >= stats.benign_support
            and stats.malicious_support >= 0.45
        ):
            return shift >= self.suspicious_temporal_min_shift
        if (
            stats.benign_support > stats.malicious_support
            and stats.benign_support >= 0.45
        ):
            return (
                self.unknown_temporal_min_shift
                <= shift
                <= self.unknown_temporal_max_shift
            )
        return (
            fusion.need_escalation
            and fusion.uncertainty >= 0.5
            and self.unknown_temporal_min_shift
            <= shift
            <= self.unknown_temporal_max_shift
        )

    def decide(self, state: CaseState) -> CoordinatorDecision:
        available_budget = state.remaining_budget
        if available_budget <= 0 or len(state.called_agents) >= self.max_evidence_calls:
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["EFFICIENCY_BUDGET_EXHAUSTED"],
                rationale="EvidenceBudgetRouterV1 reached its evidence-call budget.",
                remaining_budget=available_budget,
                source="rule",
                planner_metadata={"router": self.name, "version": self.version},
            )

        if not state.evidence:
            for agent in (
                "StatsDetectorAgent",
                "TemporalBehaviorAgent",
                "TLSProtocolAgent",
            ):
                if self._available(state, agent):
                    return CoordinatorDecision(
                        action=CoordinatorAction.DISPATCH,
                        agents=[agent],
                        reason_codes=[f"EFFICIENCY_INITIAL_{agent}"],
                        rationale=(
                            "EvidenceBudgetRouterV1 collects exactly one "
                            "initial available evidence view."
                        ),
                        remaining_budget=max(0, available_budget - 1),
                        source="rule",
                        planner_metadata={
                            "router": self.name,
                            "version": self.version,
                        },
                    )
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["EFFICIENCY_NO_AVAILABLE_INITIAL_AGENT"],
                rationale="No capability-available evidence agent exists.",
                remaining_budget=available_budget,
                source="rule",
                planner_metadata={"router": self.name, "version": self.version},
            )

        if (
            "StatsDetectorAgent" in state.called_agents
            and self._should_supplement_after_stats(state)
        ):
            for agent in ("TemporalBehaviorAgent", "TLSProtocolAgent"):
                if (
                    agent not in state.called_agents
                    and self._available(state, agent)
                    and len(state.called_agents) < self.max_evidence_calls
                    and available_budget > 0
                ):
                    return CoordinatorDecision(
                        action=CoordinatorAction.DISPATCH,
                        agents=[agent],
                        reason_codes=[f"EFFICIENCY_SUPPLEMENT_{agent}"],
                        rationale=(
                            "Stats evidence falls in the pre-registered "
                            "uncertainty/shift window where one supplemental "
                            "view may preserve coverage."
                        ),
                        remaining_budget=max(0, available_budget - 1),
                        source="rule",
                        planner_metadata={
                            "router": self.name,
                            "version": self.version,
                            "suspicious_temporal_min_shift": (
                                self.suspicious_temporal_min_shift
                            ),
                            "unknown_temporal_min_shift": (
                                self.unknown_temporal_min_shift
                            ),
                            "unknown_temporal_max_shift": (
                                self.unknown_temporal_max_shift
                            ),
                        },
                    )

        return CoordinatorDecision(
            action=CoordinatorAction.STOP_AND_FUSE,
            reason_codes=["EFFICIENCY_STOP_AFTER_STATS"],
            rationale=(
                "EvidenceBudgetRouterV1 stops because additional evidence was "
                "not selected by the fixed budget rule."
            ),
            remaining_budget=available_budget,
            source="rule",
            planner_metadata={"router": self.name, "version": self.version},
        )


class PolicyGuard:
    def __init__(
        self,
        allowed_agents: set[str] | None = None,
        *,
        max_rounds: int = 3,
        max_total_calls: int = 4,
    ) -> None:
        self.allowed_agents = allowed_agents or DEFAULT_ALLOWED_AGENTS
        self.max_rounds = max_rounds
        self.max_total_calls = max_total_calls

    def enforce(
        self,
        decision: CoordinatorDecision,
        state: CaseState,
        fallback: RuleCoordinator,
    ) -> CoordinatorDecision:
        adjustments = list(decision.policy_adjustments)
        proposed_action = decision.action
        proposed_agents = list(decision.agents)

        if state.round_no >= self.max_rounds:
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                reason_codes=["POLICY_MAX_ROUNDS"],
                rationale="PolicyGuard reached the maximum number of routing rounds.",
                remaining_budget=state.remaining_budget,
                source=decision.source,
                raw_response=decision.raw_response,
                planner_metadata=decision.planner_metadata,
                policy_adjustments=adjustments + ["forced stop at maximum rounds"],
                policy_status="forced_stop",
                proposed_action=proposed_action,
                proposed_agents=proposed_agents,
            )

        if decision.action != CoordinatorAction.DISPATCH:
            if not state.evidence and decision.action == CoordinatorAction.STOP_AND_FUSE:
                safe = fallback.decide(state)
                safe.source = "fallback"
                safe.raw_response = decision.raw_response
                safe.planner_metadata = decision.planner_metadata
                safe.policy_adjustments.append("cannot stop before collecting evidence")
                safe.policy_status = "fallback"
                safe.proposed_action = proposed_action
                safe.proposed_agents = proposed_agents
                return safe
            return decision.model_copy(
                update={
                    "policy_status": "accepted",
                    "proposed_action": proposed_action,
                    "proposed_agents": proposed_agents,
                }
            )

        agents: list[str] = []
        for agent in decision.agents:
            if agent not in self.allowed_agents:
                adjustments.append(f"removed non-whitelisted agent: {agent}")
                continue
            if agent in state.called_agents:
                adjustments.append(f"removed duplicate agent: {agent}")
                continue
            if (
                agent == "TLSProtocolAgent"
                and not RuleCoordinator._tls_available(state)
            ):
                capability = state.detector_capabilities.get(agent)
                if (
                    capability is not None
                    and state.view_availability is not None
                    and state.view_availability.routing_policy
                    == "capability_v3_0"
                    and capability.status == "unsupported_schema"
                ):
                    adjustments.append(
                        "removed TLSProtocolAgent: backend does not support "
                        "the observed TLS schema"
                    )
                else:
                    adjustments.append(
                        "removed TLSProtocolAgent: TLS metadata unavailable"
                    )
                continue
            if agent == "FamilyAttributionAgent":
                malicious_support = (
                    state.interim_fusion.malicious_support
                    if state.interim_fusion
                    else 0
                )
                if malicious_support < 0.6:
                    adjustments.append(
                        "removed FamilyAttributionAgent: insufficient malicious support"
                    )
                    continue
            agents.append(agent)

        remaining_capacity = min(
            state.remaining_budget,
            self.max_total_calls - len(state.called_agents),
        )
        if len(agents) > max(0, remaining_capacity):
            adjustments.append(
                f"truncated dispatch to remaining capacity: {max(0, remaining_capacity)}"
            )
        agents = agents[: max(0, remaining_capacity)]
        if not agents:
            safe = fallback.decide(state)
            safe.source = "fallback"
            safe.raw_response = decision.raw_response
            safe.planner_metadata = decision.planner_metadata
            safe.policy_adjustments.extend(adjustments + ["empty dispatch replaced"])
            safe.policy_status = "fallback"
            safe.proposed_action = proposed_action
            safe.proposed_agents = proposed_agents
            return safe

        return decision.model_copy(
            update={
                "agents": agents,
                "remaining_budget": max(0, state.remaining_budget - len(agents)),
                "policy_adjustments": adjustments,
                "policy_status": "adjusted" if adjustments else "accepted",
                "proposed_action": proposed_action,
                "proposed_agents": proposed_agents,
            }
        )


class NvidiaCoordinator(Coordinator):
    """NVIDIA-hosted LLM planner with strict JSON output and deterministic fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 20,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.25,
        response_format_json: bool = False,
        disable_thinking: bool = False,
        max_tokens: int = 350,
        cache_dir: str | Path | None = None,
        fallback: RuleCoordinator | None = None,
        allowed_agents: set[str] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model or os.getenv(
            "NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"
        )
        self.base_url = (
            base_url
            or os.getenv("NVIDIA_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.min_request_interval_seconds = max(
            0.0, min_request_interval_seconds
        )
        self.response_format_json = bool(response_format_json)
        self.disable_thinking = bool(disable_thinking)
        self.max_tokens = max(1, int(max_tokens))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fallback = fallback or RuleCoordinator()
        self.allowed_agents = allowed_agents or set(DEFAULT_ALLOWED_AGENTS)
        self._request_lock = threading.Lock()
        self._last_request_started = 0.0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def decide(self, state: CaseState) -> CoordinatorDecision:
        if not self.api_key:
            decision = self.fallback.decide(state)
            decision.source = "fallback"
            decision.policy_adjustments.append("NVIDIA_API_KEY not configured")
            decision.planner_metadata = {
                "provider": "nvidia",
                "model": self.model,
                "prompt_version": "coordinator-v1",
                "response_format_json": self.response_format_json,
                "disable_thinking": self.disable_thinking,
                "max_tokens": self.max_tokens,
                "fallback_reason": "missing_api_key",
                "llm_call_attempted": False,
                "real_llm": False,
                "cache_replay": False,
                "response_schema_valid": False,
                "api_latency_ms": None,
                "usage": None,
                "raw_response_sha256": None,
                "illegal_verdict_attempt_count": 0,
                "blocked_field_access_attempt_count": 0,
            }
            return decision

        prompt = self._build_prompt(state)
        cache_key = hashlib.sha256(
            (
                f"{self.model}\n{self.response_format_json}\n"
                f"{self.disable_thinking}\n{self.max_tokens}\n{prompt}"
            ).encode("utf-8")
        ).hexdigest()
        cached = self._read_cache(cache_key)
        if cached is not None:
            try:
                inspection = self._inspect_raw_response(cached)
                if inspection["rejected"]:
                    raise ValueError("cached response violates planner boundary")
                decision = self._parse_decision(cached, state.remaining_budget)
                decision.source = "replay"
                decision.raw_response = cached
                decision.planner_metadata = {
                    **self._planner_metadata(prompt),
                    "llm_call_attempted": False,
                    "real_llm": False,
                    "cache_replay": True,
                    "response_schema_valid": True,
                    "api_latency_ms": 0.0,
                    "usage": None,
                    "raw_response_sha256": self._raw_hash(cached),
                    **inspection,
                }
                return decision
            except (ValueError, ValidationError):
                pass

        raw: str | None = None
        api_latency_ms: float | None = None
        response_id: str | None = None
        usage: dict[str, int] | None = None
        inspection = self._empty_inspection()
        request_attempts = 0
        retry_count = 0
        try:
            started = time.perf_counter()
            response, request_attempts, retry_count = self._request(prompt)
            api_latency_ms = (time.perf_counter() - started) * 1000
            payload = response.json()
            response_id = payload.get("id")
            usage = self._normalize_usage(payload.get("usage"))
            raw = payload["choices"][0]["message"]["content"]
            inspection = self._inspect_raw_response(raw)
            if inspection["rejected"]:
                raise ValueError("LLM response violates planner boundary")
            decision = self._parse_decision(raw, state.remaining_budget)
            self._write_cache(cache_key, raw)
            decision.source = "nvidia"
            decision.raw_response = raw
            decision.planner_metadata = {
                **self._planner_metadata(prompt),
                "llm_call_attempted": True,
                "real_llm": True,
                "cache_replay": False,
                "response_schema_valid": True,
                "api_latency_ms": api_latency_ms,
                "response_id": response_id,
                "usage": usage,
                "request_attempts": request_attempts,
                "retry_count": retry_count,
                "raw_response_sha256": self._raw_hash(raw),
                **inspection,
            }
            return decision
        except (requests.RequestException, KeyError, ValueError, ValidationError) as exc:
            if api_latency_ms is None and "started" in locals():
                api_latency_ms = (time.perf_counter() - started) * 1000
            decision = self.fallback.decide(state)
            decision.source = "fallback"
            decision.raw_response = raw
            decision.policy_adjustments.append(
                f"NVIDIA coordinator failed: {type(exc).__name__}"
            )
            decision.planner_metadata = {
                **self._planner_metadata(prompt),
                "fallback_reason": type(exc).__name__,
                "llm_call_attempted": True,
                "real_llm": False,
                "cache_replay": False,
                "response_schema_valid": False,
                "api_latency_ms": api_latency_ms,
                "response_id": response_id,
                "usage": usage,
                "request_attempts": request_attempts,
                "retry_count": retry_count,
                "raw_response_sha256": self._raw_hash(raw),
                "parse_error": str(exc),
                **inspection,
            }
            return decision

    def _request(self, prompt: str) -> tuple[Any, int, int]:
        attempts = 0
        retries = 0
        while True:
            attempts += 1
            self._wait_for_request_slot()
            try:
                request_payload: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": self._system_prompt(),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": self.max_tokens,
                }
                if self.response_format_json:
                    request_payload["response_format"] = {
                        "type": "json_object"
                    }
                if self.disable_thinking:
                    request_payload["chat_template_kwargs"] = {
                        "enable_thinking": False
                    }
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.timeout_seconds,
                )
                status = int(getattr(response, "status_code", 200))
                if status == 429 or status >= 500:
                    if retries >= self.max_retries:
                        response.raise_for_status()
                    delay = self._retry_delay(response, retries)
                    retries += 1
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response, attempts, retries
            except (requests.Timeout, requests.ConnectionError):
                if retries >= self.max_retries:
                    raise
                delay = self.retry_backoff_seconds * (2**retries)
                retries += 1
                time.sleep(delay)

    def _wait_for_request_slot(self) -> None:
        with self._request_lock:
            now = time.perf_counter()
            remaining = (
                self.min_request_interval_seconds
                - (now - self._last_request_started)
            )
            if remaining > 0:
                time.sleep(remaining)
            self._last_request_started = time.perf_counter()

    def _retry_delay(self, response: Any, retry_index: int) -> float:
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        return self.retry_backoff_seconds * (2**retry_index)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a routing planner for encrypted traffic analysis. "
            "You may only choose the supplied agents. You never decide whether "
            "traffic is benign or malicious. Return one JSON object only with keys "
            "action, agents, reason_codes, rationale. action is DISPATCH, "
            "STOP_AND_FUSE, or ABSTAIN_AND_REPORT."
        )

    def _build_prompt(self, state: CaseState) -> str:
        reliability = (
            state.reliability.model_dump(mode="json") if state.reliability else {}
        )
        fusion = (
            state.interim_fusion.model_dump(mode="json")
            if state.interim_fusion
            else None
        )
        summary = {
            "round": state.round_no,
            "remaining_budget": state.remaining_budget,
            "available_views": (
                state.view_availability.model_dump(mode="json")
                if state.view_availability
                else {
                    "stats": bool((state.safe_flow or state.flow).stats),
                    "sequence": bool(
                        (state.safe_flow or state.flow).sequence.packet_lengths
                    ),
                    "tls": bool((state.safe_flow or state.flow).tls),
                    "payload": bool(
                        (state.safe_flow or state.flow).payload_tokens
                    ),
                }
            ),
            "detector_capabilities": {
                name: item.model_dump(mode="json")
                for name, item in state.detector_capabilities.items()
            },
            "called_agents": sorted(state.called_agents),
            "allowed_agents": sorted(self.allowed_agents),
            "reliability": reliability,
            "interim_fusion": fusion,
            "evidence_summaries": [
                {
                    "agent": item.agent_name,
                    "benign_support": item.benign_support,
                    "malicious_support": item.malicious_support,
                    "uncertainty": item.uncertainty,
                    "abstained": item.abstained,
                }
                for item in state.evidence
            ],
            "memory_hint": (
                state.memory_hint.model_dump(mode="json")
                if state.memory_hint
                else None
            ),
        }
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _parse_decision(raw: str, remaining_budget: int) -> CoordinatorDecision:
        data = NvidiaCoordinator._extract_json_object(raw)
        parsed = LLMPlannerResponse.model_validate(data)
        return CoordinatorDecision(
            action=parsed.action,
            agents=parsed.agents,
            reason_codes=parsed.reason_codes,
            rationale=parsed.rationale,
            remaining_budget=remaining_budget,
            source="nvidia",
        )

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM output does not contain a JSON object")
        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object")
        return data

    @staticmethod
    def _empty_inspection() -> dict[str, Any]:
        return {
            "rejected": False,
            "illegal_verdict_attempt_count": 0,
            "blocked_field_access_attempt_count": 0,
            "illegal_ownership_fields": [],
            "blocked_field_markers": [],
        }

    @classmethod
    def _inspect_raw_response(cls, raw: str) -> dict[str, Any]:
        data = cls._extract_json_object(raw)
        ownership_hits: set[str] = set()
        blocked_hits: set[str] = set()

        def visit(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    normalized = str(key).strip().lower()
                    field_path = f"{path}.{normalized}" if path else normalized
                    if normalized in LLM_FORBIDDEN_OWNERSHIP_FIELDS:
                        ownership_hits.add(field_path)
                    if any(
                        marker in normalized
                        for marker in LLM_BLOCKED_FIELD_MARKERS
                    ):
                        blocked_hits.add(field_path)
                    visit(nested, field_path)
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    visit(nested, f"{path}[{index}]")
            elif isinstance(value, str):
                normalized = value.lower()
                if LLM_VERDICT_PATTERN.search(value):
                    ownership_hits.add(path or "text")
                for marker in LLM_BLOCKED_FIELD_MARKERS:
                    if marker in normalized:
                        blocked_hits.add(f"{path}:{marker}" if path else marker)

        visit(data)
        return {
            "rejected": bool(ownership_hits or blocked_hits),
            "illegal_verdict_attempt_count": len(ownership_hits),
            "blocked_field_access_attempt_count": len(blocked_hits),
            "illegal_ownership_fields": sorted(ownership_hits),
            "blocked_field_markers": sorted(blocked_hits),
        }

    @staticmethod
    def _normalize_usage(value: Any) -> dict[str, int] | None:
        if not isinstance(value, dict):
            return None
        prompt = int(value.get("prompt_tokens", 0) or 0)
        completion = int(value.get("completion_tokens", 0) or 0)
        total = int(value.get("total_tokens", prompt + completion) or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    @staticmethod
    def _raw_hash(raw: str | None) -> str | None:
        if raw is None:
            return None
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.txt"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _write_cache(self, key: str, raw: str) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.txt").write_text(raw, encoding="utf-8")

    def _planner_metadata(self, prompt: str) -> dict[str, Any]:
        return {
            "provider": "nvidia",
            "model": self.model,
            "endpoint": f"{self.base_url}/chat/completions",
            "prompt_version": "coordinator-v1",
            "temperature": 0,
            "response_format_json": self.response_format_json,
            "disable_thinking": self.disable_thinking,
            "max_tokens": self.max_tokens,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "min_request_interval_seconds": self.min_request_interval_seconds,
        }


class GuardedCoordinator(Coordinator):
    def __init__(
        self,
        planner: Coordinator,
        policy_guard: PolicyGuard | None = None,
        fallback: RuleCoordinator | None = None,
    ) -> None:
        self.planner = planner
        self.fallback = fallback or RuleCoordinator()
        self.policy_guard = policy_guard or PolicyGuard()

    def decide(self, state: CaseState) -> CoordinatorDecision:
        proposed = self.planner.decide(state)
        return self.policy_guard.enforce(proposed, state, self.fallback)
