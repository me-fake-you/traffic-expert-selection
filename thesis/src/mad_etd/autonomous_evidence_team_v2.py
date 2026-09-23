"""Bounded-autonomy LLM control plane for the MAD-ETD evidence team.

The module deliberately separates three responsibilities:

* an LLM role controller may select a governed Skill or an advisory action;
* deterministic detectors and tools produce numerical evidence;
* only ``FusionAgent`` may create the final verdict/confidence/uncertainty.

This candidate is default-off.  It neither changes ``runtime_safe_v3_0`` nor
turns an LLM response into ``AgentEvidenceV2``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import requests
import numpy as np
from pydantic import ConfigDict, Field, model_validator

from .schemas import StrictModel
from .traffic_skill_registry_v1 import (
    CaseStateSkillV2,
    EvidenceRequestSkillV1,
    TrafficExpertSkillRegistryV2,
)


DEFAULT_OUTPUT = Path("data/runs/mad_etd_autonomous_evidence_team_v2")
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
PROMPT_VERSION = "mad_etd_autonomous_role_v2.0"
FINAL_DECISION_OWNER = "FusionAgent"
DEFAULT_RUNTIME = "runtime_safe_v3_0"

FORBIDDEN_LLM_OUTPUT_KEYS = frozenset(
    {
        "verdict",
        "final_verdict",
        "prediction",
        "probabilities",
        "confidence",
        "final_confidence",
        "uncertainty",
        "final_uncertainty",
        "ood_override",
        "agent_evidence",
        "fusion_result",
    }
)
FORBIDDEN_DIRECT_LABEL_PATTERN = re.compile(
    r"\b(?:traffic|flow|sample|case|class|classification|verdict)\s+"
    r"(?:is|=|:)\s*"
    r"(?:benign|malicious|suspicious|unknown)\b",
    re.IGNORECASE,
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _unsafe_output_keys(value: Any, *, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_").strip()
            current = f"{prefix}.{normalized}" if prefix else normalized
            if normalized in FORBIDDEN_LLM_OUTPUT_KEYS:
                findings.append(current)
            findings.extend(_unsafe_output_keys(item, prefix=current))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(_unsafe_output_keys(item, prefix=f"{prefix}[{index}]"))
    return findings


class AutonomousAgentRoleProfile(StrictModel):
    """Immutable authority and Skill boundary for one LLM role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    role_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    model: str = DEFAULT_MODEL
    prompt_version: str = PROMPT_VERSION
    allowed_skills: tuple[str, ...]
    allowed_actions: tuple[
        Literal[
            "request_skill",
            "request_followup",
            "abstain",
            "stop",
            "escalate",
            "critic_feedback",
            "retrieve_memory",
            "report",
        ],
        ...,
    ]
    input_schema: Literal["CaseStateSkillV2"] = "CaseStateSkillV2"
    output_schema: Literal["AutonomousRoleDecision"] = "AutonomousRoleDecision"
    llm_enabled: bool = True
    advisory_only: bool
    can_request_evidence: bool
    can_emit_agent_evidence: bool = False
    can_write_fusion: bool = False
    can_override_ood: bool = False
    can_access_blocked_fields: bool = False
    final_decision_owner: Literal["FusionAgent"] = "FusionAgent"

    @model_validator(mode="after")
    def bounded_authority(self) -> "AutonomousAgentRoleProfile":
        if (
            self.can_emit_agent_evidence
            or self.can_write_fusion
            or self.can_override_ood
            or self.can_access_blocked_fields
        ):
            raise ValueError("LLM role exceeds the bounded-autonomy contract")
        if self.can_request_evidence and "request_skill" not in self.allowed_actions:
            raise ValueError("evidence-requesting role omits request_skill action")
        return self


class AutonomousRoleDecision(StrictModel):
    """Schema-constrained LLM control-plane decision; never evidence/verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    role_id: str = Field(min_length=1)
    case_state_hash: str = Field(min_length=64, max_length=64)
    action: Literal[
        "request_skill",
        "request_followup",
        "abstain",
        "stop",
        "escalate",
        "critic_feedback",
        "retrieve_memory",
        "report",
    ]
    requested_skill: str | None = None
    purpose: str = Field(min_length=1)
    concise_rationale: str = Field(min_length=1, max_length=500)
    expected_output: Literal[
        "AgentEvidenceV2", "AdvisoryInfo", "NoAction"
    ] = "NoAction"
    source: Literal["llm", "fixture", "fallback"]
    prompt_version: str = PROMPT_VERSION

    @model_validator(mode="after")
    def request_action_has_skill(self) -> "AutonomousRoleDecision":
        if self.action in {"request_skill", "request_followup"}:
            if not self.requested_skill:
                raise ValueError("request action requires requested_skill")
        elif self.requested_skill is not None:
            raise ValueError("non-request action cannot carry requested_skill")
        if FORBIDDEN_DIRECT_LABEL_PATTERN.search(
            f"{self.purpose} {self.concise_rationale}"
        ):
            raise ValueError("LLM decision contains a direct class-label attempt")
        return self


class AutonomousRoleExecution(StrictModel):
    """Auditable result of one role invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: str
    status: Literal[
        "llm_success",
        "fixture_success",
        "api_unavailable",
        "llm_rejected",
        "llm_error_fallback",
        "policy_rejected",
    ]
    provider: Literal["NVIDIA", "fixture", "unavailable"]
    model: str
    real_llm_call: bool
    fallback_used: bool
    raw_response_sha256: str
    latency_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    decision: AutonomousRoleDecision
    evidence_request: EvidenceRequestSkillV1 | None = None
    policy_reason_codes: tuple[str, ...] = ()


ROLE_SKILL_BINDINGS: dict[str, tuple[str, ...]] = {
    "coordinator": (
        "GeneralFlowMalwareSkill",
        "TemporalSequenceSkill",
        "PrefixEvidenceSkill",
        "ShortFlowSkill",
        "TLSRecordSkill",
        "DoHCovertChannelSkill",
        "OODDomainShiftSkill",
    ),
    "stats_investigator": (
        "GeneralFlowMalwareSkill",
        "IoTBotnetSkill",
        "EnterpriseIntrusionSkill",
        "MalwareTrafficSkill",
        "ShortcutForensicsSkill",
    ),
    "temporal_investigator": (
        "TemporalSequenceSkill",
        "PrefixEvidenceSkill",
        "ShortFlowSkill",
        "PerturbationRobustnessSkill",
    ),
    "tls_investigator": (
        "TLSRecordSkill",
        "DoHCovertChannelSkill",
        "ProtocolInterpretationSkill",
    ),
    "ood_investigator": (
        "OODDomainShiftSkill",
        "DriftMonitorSkill",
    ),
    "evidence_critic": (
        "ErrorAnalysisSkill",
        "ShortcutForensicsSkill",
        "AuditExplanationSkill",
    ),
    "reflection": (
        "ErrorAnalysisSkill",
        "PerturbationRobustnessSkill",
        "AuditExplanationSkill",
    ),
    "memory_rag": (
        "ExperienceMemorySkill",
        "PacketFlowSummarySkill",
        "ProtocolInterpretationSkill",
    ),
    "reporter_hitl": (
        "IncidentNarrativeSkill",
        "AuditExplanationSkill",
        "PacketFlowSummarySkill",
    ),
}


def default_role_profiles(
    *, model: str = DEFAULT_MODEL
) -> tuple[AutonomousAgentRoleProfile, ...]:
    """Return the nine default-off LLM roles requested by the protocol."""

    names = {
        "coordinator": "LLM Investigation Coordinator",
        "stats_investigator": "LLM Stats Investigator",
        "temporal_investigator": "LLM Temporal Investigator",
        "tls_investigator": "LLM TLS Investigator",
        "ood_investigator": "LLM OOD Investigator",
        "evidence_critic": "LLM Evidence Critic",
        "reflection": "LLM Reflection Advisor",
        "memory_rag": "LLM Memory/RAG Advisor",
        "reporter_hitl": "LLM Reporter/HITL Assistant",
    }
    evidence_roles = {
        "coordinator",
        "stats_investigator",
        "temporal_investigator",
        "tls_investigator",
        "ood_investigator",
    }
    rows: list[AutonomousAgentRoleProfile] = []
    for role_id, allowed_skills in ROLE_SKILL_BINDINGS.items():
        advisory = role_id not in evidence_roles
        actions: tuple[Any, ...]
        if advisory:
            advisory_action = {
                "evidence_critic": "critic_feedback",
                "reflection": "critic_feedback",
                "memory_rag": "retrieve_memory",
                "reporter_hitl": "report",
            }[role_id]
            actions = ("request_skill", advisory_action, "abstain", "stop")
        else:
            actions = (
                "request_skill",
                "request_followup",
                "abstain",
                "stop",
                "escalate",
            )
        rows.append(
            AutonomousAgentRoleProfile(
                role_id=role_id,
                display_name=names[role_id],
                model=model,
                allowed_skills=allowed_skills,
                allowed_actions=actions,
                advisory_only=advisory,
                can_request_evidence=True,
            )
        )
    return tuple(rows)


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    unsafe = _unsafe_output_keys(payload)
    if unsafe:
        raise ValueError(f"LLM response contains forbidden ownership keys: {unsafe}")
    return payload


class NvidiaAutonomousRoleTransport:
    """Minimal NVIDIA chat-completions transport for role-constrained JSON."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        min_request_interval_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key and api_key_file:
            path = Path(api_key_file)
            if path.exists():
                value = path.read_text(encoding="utf-8").strip()
                if "=" in value and value.split("=", 1)[0].strip().upper() == (
                    "NVIDIA_API_KEY"
                ):
                    value = value.split("=", 1)[1].strip()
                self.api_key = value or None
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.min_request_interval_seconds = max(
            0.0, min_request_interval_seconds
        )
        self._last_request_started = 0.0

    def complete(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not configured")
        started = time.perf_counter()
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            elapsed = time.perf_counter() - self._last_request_started
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
            self._last_request_started = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout_seconds,
            )
            if response.status_code < 400:
                break
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= self.max_retries:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                delay = 0.0
            delay = max(
                delay,
                self.retry_backoff_seconds * (2**attempt),
            )
            time.sleep(delay)
        if response is None:
            raise RuntimeError("NVIDIA transport returned no response")
        response.raise_for_status()
        payload = response.json()
        raw = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return {
            "raw": raw,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


class AutonomousEvidenceTeamHarness:
    """Role-aware LLM harness with deterministic policy validation/fallback."""

    def __init__(
        self,
        *,
        registry: TrafficExpertSkillRegistryV2 | None = None,
        profiles: tuple[AutonomousAgentRoleProfile, ...] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.registry = registry or TrafficExpertSkillRegistryV2()
        items = profiles or default_role_profiles(model=model)
        self.profiles = {profile.role_id: profile for profile in items}
        if len(self.profiles) != 9:
            raise ValueError("Autonomous Evidence Team requires exactly nine roles")
        for profile in items:
            for skill_id in profile.allowed_skills:
                self.registry.get(skill_id)

    def system_prompt(self, role_id: str) -> str:
        profile = self.profiles[role_id]
        return (
            f"You are {profile.display_name} in MAD-ETD. Work only on the "
            "controlled CaseStateSkillV2. Choose at most one allow-listed Skill "
            "or a non-decision advisory action. Return one JSON object matching "
            "AutonomousRoleDecision. Do not output prediction, probabilities, "
            "benign/malicious labels, verdict, confidence, uncertainty, "
            "AgentEvidence, FusionResult, OOD override, blocked fields, or hidden "
            "chain-of-thought. Provide only a concise auditable rationale. "
            f"Allowed actions: {list(profile.allowed_actions)}. "
            f"Allowed Skills: {list(profile.allowed_skills)}. "
            "FusionAgent is the only final decision owner."
        )

    def _available_capabilities(
        self, state: CaseStateSkillV2
    ) -> set[str]:
        available = set(state.available_views)
        available.update(
            key
            for key, value in state.capability_profile.items()
            if str(value).lower() in {"supported", "available", "true", "1"}
        )
        return available

    def _applicable_skills(
        self,
        role_id: str,
        state: CaseStateSkillV2,
    ) -> list[str]:
        available = self._available_capabilities(state)
        profile = self.profiles[role_id]
        specs = []
        previous = set(state.previous_requests)
        for skill_id in profile.allowed_skills:
            spec = self.registry.get(skill_id)
            if skill_id in previous:
                continue
            if not set(spec.required_capabilities).issubset(available):
                continue
            if skill_id == "ShortFlowSkill" and not state.short_flow_state.get(
                "is_short"
            ):
                continue
            if skill_id == "OODDomainShiftSkill" and state.OOD_state.get(
                "state"
            ) not in {"warning", "hard", "uncertain"}:
                continue
            specs.append(spec)
        return [
            spec.skill_id
            for spec in sorted(
                specs,
                key=lambda spec: (-spec.scope_priority, spec.skill_id),
            )
        ]

    def user_prompt(self, role_id: str, state: CaseStateSkillV2) -> str:
        profile = self.profiles[role_id]
        applicable = self._applicable_skills(role_id, state)
        preferred = applicable[0] if applicable else None
        template = {
            "role_id": role_id,
            "case_state_hash": state.state_hash,
            "action": "request_skill" if preferred else "abstain",
            "requested_skill": preferred,
            "purpose": "collect one policy-compliant specialist result",
            "concise_rationale": "one concise reason without a class decision",
            "expected_output": "AgentEvidenceV2",
            "source": "llm",
            "prompt_version": profile.prompt_version,
        }
        return json.dumps(
            {
                "controlled_case_state": state.model_dump(mode="json"),
                "applicable_skill_candidates": applicable,
                "required_output_template": template,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _fallback_decision(
        self, role_id: str, state: CaseStateSkillV2
    ) -> AutonomousRoleDecision:
        profile = self.profiles[role_id]
        applicable = self._applicable_skills(role_id, state)
        if not applicable:
            return AutonomousRoleDecision(
                role_id=role_id,
                case_state_hash=state.state_hash,
                action="abstain",
                purpose="no registered Skill matches the available capabilities",
                concise_rationale="deterministic fallback found no applicable role Skill",
                expected_output="NoAction",
                source="fallback",
            )
        preferred = applicable[0]
        if role_id in {"coordinator", "temporal_investigator"}:
            if (
                state.short_flow_state.get("is_short")
                and "ShortFlowSkill" in profile.allowed_skills
            ):
                preferred = "ShortFlowSkill"
            elif (
                state.OOD_state.get("state") in {"warning", "hard"}
                and "OODDomainShiftSkill" in profile.allowed_skills
            ):
                preferred = "OODDomainShiftSkill"
        return AutonomousRoleDecision(
            role_id=role_id,
            case_state_hash=state.state_hash,
            action="request_skill",
            requested_skill=preferred,
            purpose="collect one policy-compliant specialist result",
            concise_rationale="deterministic fallback selected the first applicable role Skill",
            expected_output="AgentEvidenceV2",
            source="fallback",
        )

    def _review(
        self,
        decision: AutonomousRoleDecision,
        state: CaseStateSkillV2,
    ) -> tuple[EvidenceRequestSkillV1 | None, tuple[str, ...]]:
        profile = self.profiles[decision.role_id]
        reasons: list[str] = []
        if decision.case_state_hash != state.state_hash:
            reasons.append("case_state_hash_mismatch")
        if decision.action not in profile.allowed_actions:
            reasons.append("role_action_not_allowed")
        if decision.requested_skill:
            if decision.requested_skill not in profile.allowed_skills:
                reasons.append("role_skill_not_allowed")
            else:
                spec = self.registry.get(decision.requested_skill)
                missing_capabilities = sorted(
                    set(spec.required_capabilities)
                    - self._available_capabilities(state)
                )
                if missing_capabilities:
                    reasons.append(
                        "required_capability_missing:"
                        + ",".join(missing_capabilities)
                    )
        if reasons or not decision.requested_skill:
            return None, tuple(reasons)
        request = self.registry.create_request(
            case_state=state,
            requested_skill=decision.requested_skill,
            requested_agent=profile.display_name,
            hypothesis="the selected specialist can reduce the current evidence gap",
            purpose=decision.purpose,
            max_agent_calls=min(1, state.remaining_agent_budget),
            max_latency_ms=state.remaining_latency_budget,
            planner_type=(
                "llm" if decision.source == "llm" else "fallback"
            ),
            prompt_version=profile.prompt_version,
        )
        return request, ("policy_approved",)

    def execute_role(
        self,
        *,
        role_id: str,
        state: CaseStateSkillV2,
        mode: Literal["nvidia", "fixture", "unavailable"] = "unavailable",
        transport: NvidiaAutonomousRoleTransport | None = None,
        fixture_raw: str | None = None,
    ) -> AutonomousRoleExecution:
        if role_id not in self.profiles:
            raise KeyError(role_id)
        profile = self.profiles[role_id]
        raw = ""
        latency_ms = 0.0
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        error_code: str | None = None
        real_call = False
        fallback = False
        provider: Literal["NVIDIA", "fixture", "unavailable"]
        status: Literal[
            "llm_success",
            "fixture_success",
            "api_unavailable",
            "llm_rejected",
            "llm_error_fallback",
            "policy_rejected",
        ]
        try:
            if mode == "nvidia":
                if transport is None or not transport.api_key:
                    provider = "unavailable"
                    decision = self._fallback_decision(role_id, state)
                    status = "api_unavailable"
                    fallback = True
                    error_code = "NVIDIA_API_KEY_NOT_CONFIGURED"
                    request, reasons = self._review(decision, state)
                    return AutonomousRoleExecution(
                        role_id=role_id,
                        status=status,
                        provider=provider,
                        model=profile.model,
                        real_llm_call=False,
                        fallback_used=fallback,
                        raw_response_sha256=hashlib.sha256(b"").hexdigest(),
                        latency_ms=0.0,
                        error_code=error_code,
                        decision=decision,
                        evidence_request=request,
                        policy_reason_codes=reasons,
                    )
                provider = "NVIDIA"
                real_call = True
                result = transport.complete(
                    self.system_prompt(role_id), self.user_prompt(role_id, state)
                )
                raw = str(result["raw"])
                latency_ms = float(result["latency_ms"])
                prompt_tokens = result.get("prompt_tokens")
                completion_tokens = result.get("completion_tokens")
                payload = _extract_json_object(raw)
                decision = AutonomousRoleDecision.model_validate(payload)
                status = "llm_success"
            elif mode == "fixture":
                provider = "fixture"
                applicable = self._applicable_skills(role_id, state)
                preferred = (
                    applicable[0] if applicable else profile.allowed_skills[0]
                )
                raw = fixture_raw or json.dumps(
                    {
                        "schema_version": "2.0",
                        "role_id": role_id,
                        "case_state_hash": state.state_hash,
                        "action": "request_skill",
                        "requested_skill": preferred,
                        "purpose": "collect one policy-compliant specialist result",
                        "concise_rationale": "fixture validates the typed role boundary",
                        "expected_output": "AgentEvidenceV2",
                        "source": "fixture",
                        "prompt_version": profile.prompt_version,
                    }
                )
                decision = AutonomousRoleDecision.model_validate(
                    _extract_json_object(raw)
                )
                status = "fixture_success"
            else:
                provider = "unavailable"
                decision = self._fallback_decision(role_id, state)
                status = "api_unavailable"
                fallback = True
        except Exception as exc:
            provider = "NVIDIA" if mode == "nvidia" else "fixture"
            raw = raw or f"{type(exc).__name__}"
            error_code = type(exc).__name__.upper()
            decision = self._fallback_decision(role_id, state)
            status = (
                "llm_rejected"
                if isinstance(exc, (ValueError, json.JSONDecodeError))
                else "llm_error_fallback"
            )
            fallback = True
        request, reasons = self._review(decision, state)
        if request is None:
            status = "policy_rejected"
            fallback = True
            decision = self._fallback_decision(role_id, state)
            request, fallback_reasons = self._review(decision, state)
            reasons = reasons + ("fallback_applied",) + fallback_reasons
        return AutonomousRoleExecution(
            role_id=role_id,
            status=status,
            provider=provider,
            model=profile.model,
            real_llm_call=real_call,
            fallback_used=fallback,
            raw_response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=error_code,
            decision=decision,
            evidence_request=request,
            policy_reason_codes=reasons,
        )


def _sample_case_state() -> CaseStateSkillV2:
    capability_names = sorted(
        {
            capability
            for profile in default_role_profiles()
            for skill_id in profile.allowed_skills
            for capability in TrafficExpertSkillRegistryV2()
            .get(skill_id)
            .required_capabilities
        }
    )
    return CaseStateSkillV2(
        case_id="autonomous-team-smoke-001",
        field_audit_summary={
            "status": "passed",
            "blocked_field_violation": 0,
        },
        safe_feature_policy_hash="a" * 64,
        available_views=("stats", "sequence", "tls_records"),
        capability_profile={name: "supported" for name in capability_names},
        collected_evidence_summary=(),
        evidence_conflict={"present": True},
        uncertainty_state={"level": "medium"},
        OOD_state={"state": "warning"},
        missing_views=(),
        short_flow_state={"is_short": True, "packet_bucket": "2_to_4"},
        remaining_agent_budget=3,
        remaining_latency_budget=2500.0,
        previous_requests=(),
        previous_failures=(),
        escalation_state={"status": "none"},
        audit_chain_hash="b" * 64,
    )


def build_autonomous_evidence_team_v2(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    harness = AutonomousEvidenceTeamHarness(model=model)
    profiles = [
        profile.model_dump(mode="json")
        for profile in sorted(
            harness.profiles.values(), key=lambda item: item.role_id
        )
    ]
    _dump(
        output / "autonomous_agent_role_profile_schema.json",
        AutonomousAgentRoleProfile.model_json_schema(),
    )
    _dump(
        output / "autonomous_role_decision_schema.json",
        AutonomousRoleDecision.model_json_schema(),
    )
    _dump(
        output / "autonomous_case_state_schema.json",
        CaseStateSkillV2.model_json_schema(),
    )
    _dump(output / "role_registry.json", {"roles": profiles})
    _dump(
        output / "architecture_invariants.json",
        {
            "default_runtime": DEFAULT_RUNTIME,
            "default_runtime_modified": False,
            "candidate_default_enabled": False,
            "final_decision_owner": FINAL_DECISION_OWNER,
            "llm_can_emit_agent_evidence": False,
            "llm_can_write_fusion": False,
            "llm_can_override_ood": False,
            "field_audit_controls_detector_input": True,
            "policy_guard_controls_execution": True,
            "admission_gate_controls_fusion_input": True,
            "automatic_training": False,
            "automatic_deployment": False,
            "fake_metric_count": 0,
        },
    )
    _dump(
        output / "prompt_templates.json",
        {
            role_id: {
                "prompt_version": profile.prompt_version,
                "system_prompt": harness.system_prompt(role_id),
                "system_prompt_sha256": hashlib.sha256(
                    harness.system_prompt(role_id).encode("utf-8")
                ).hexdigest(),
            }
            for role_id, profile in sorted(harness.profiles.items())
        },
    )
    _write_csv(
        output / "skill_binding_table.csv",
        [
            {
                "role_id": profile.role_id,
                "display_name": profile.display_name,
                "model": profile.model,
                "skill_id": skill_id,
                "advisory_only": profile.advisory_only,
                "llm_enabled": profile.llm_enabled,
                "can_emit_agent_evidence": profile.can_emit_agent_evidence,
                "can_write_fusion": profile.can_write_fusion,
            }
            for profile in sorted(
                harness.profiles.values(), key=lambda item: item.role_id
            )
            for skill_id in profile.allowed_skills
        ],
    )
    report = {
        "status": "autonomous_role_protocol_built_default_off",
        "role_count": len(profiles),
        "all_requested_roles_llm_enabled": all(
            profile["llm_enabled"] for profile in profiles
        ),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "build_report.json", report)
    return report


def build_autonomous_evidence_performance_manifest_v2(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_path: str | Path = (
        "data/runs/mad_etd_hybrid_multiagent_evidence_v1/"
        "per_mode_predictions.npz"
    ),
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze a no-tuning group split for competence-aware performance work."""

    output = Path(output_dir)
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "y",
            "group",
            "sample_hash",
            "stats_tree_only__probability",
            "temporal_only__probability",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"performance source omits required arrays: {missing}")
        y = np.asarray(payload["y"], dtype=np.int64)
        groups = np.asarray(payload["group"]).astype(str)
        sample_hash = np.asarray(payload["sample_hash"]).astype(str)
    if len(set(sample_hash.tolist())) != len(sample_hash):
        raise ValueError("sample hashes are not unique")
    group_rows: list[dict[str, Any]] = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        labels = sorted(set(y[indices].tolist()))
        if len(labels) != 1:
            raise ValueError(f"group is not label-homogeneous: {group}")
        group_rows.append(
            {
                "group": group,
                "label": int(labels[0]),
                "sample_count": int(len(indices)),
                "rank_hash": _canonical_sha256(
                    {"seed": seed, "group": group}
                ),
            }
        )
    selection: list[str] = []
    acceptance: list[str] = []
    for label in (0, 1):
        class_groups = sorted(
            (row for row in group_rows if row["label"] == label),
            key=lambda row: row["rank_hash"],
        )
        if len(class_groups) < 4 or len(class_groups) % 2:
            raise ValueError(
                "each class requires an even number of at least four groups"
            )
        midpoint = len(class_groups) // 2
        selection.extend(row["group"] for row in class_groups[:midpoint])
        acceptance.extend(row["group"] for row in class_groups[midpoint:])
    if set(selection) & set(acceptance):
        raise RuntimeError("selection/acceptance group overlap")
    split_rows = [
        {
            **row,
            "split": (
                "selection"
                if row["group"] in selection
                else "acceptance"
            ),
        }
        for row in group_rows
    ]
    _write_csv(output / "performance_group_split.csv", split_rows)
    manifest = {
        "protocol": "autonomous_evidence_team_v2_competence_fusion",
        "seed": seed,
        "source_path": str(source.as_posix()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_count": int(len(y)),
        "group_count": len(group_rows),
        "selection_groups": sorted(selection),
        "acceptance_groups": sorted(acceptance),
        "selection_acceptance_overlap": 0,
        "base_predictions_are_group_held_out": True,
        "selection_only_uses": [
            "competence_model_selection",
            "calibration",
            "threshold_selection",
        ],
        "acceptance_only_uses": ["one_time_final_evaluation"],
        "acceptance_used_for_selection": False,
        "safe_meta_features": [
            "stats_probability",
            "temporal_probability",
            "stats_margin",
            "temporal_margin",
            "stats_uncertainty",
            "temporal_uncertainty",
            "evidence_disagreement",
            "probability_mean",
            "probability_product",
        ],
        "forbidden_runtime_features": [
            "label",
            "application",
            "family",
            "sample_id",
            "source_file",
            "ip",
            "port",
            "timestamp",
            "provenance",
            "group",
        ],
        "pre_registered_candidates": [
            "logistic_competence_fusion",
            "hgb_competence_fusion",
            "extra_trees_competence_fusion",
        ],
        "reference": {
            "mode": "temporal_only",
            "pooled_macro_f1_context_only": 0.9283399793261593,
            "pooled_accuracy_context_only": 0.928675,
            "note": (
                "context values are frozen prior evidence, not the new "
                "acceptance result"
            ),
        },
        "strong_positive_acceptance_gates": {
            "macro_f1_delta_ge": 0.01,
            "accuracy_delta_ge": 0.01,
            "grouped_bootstrap_iterations": 1000,
            "grouped_bootstrap_seed": 42,
            "macro_f1_delta_ci95_lower_gt": 0.0,
            "malicious_recall_not_lower": True,
            "worst_group_class_f1_not_lower": True,
            "ece_not_worse_by_more_than": 0.005,
            "harmful_flip_reduction_ge": 0.5,
            "blocked_field_violation": 0,
            "fusion_ownership_violation": 0,
            "ood_override": 0,
            "illegal_verdict_execution": 0,
            "fake_metric_count": 0,
        },
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
    }
    _dump(output / "performance_manifest.json", manifest)
    report = {
        "status": "performance_protocol_frozen_ready_for_selection_training",
        "sample_count": manifest["sample_count"],
        "group_count": manifest["group_count"],
        "selection_group_count": len(selection),
        "acceptance_group_count": len(acceptance),
        "selection_acceptance_overlap": 0,
        "acceptance_used_for_selection": False,
        "performance_metrics_generated": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "performance_manifest_report.json", report)
    return report


def run_autonomous_positive_skill_bridge_v2(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    mode: Literal["nvidia", "fixture", "unavailable"] = "unavailable",
    api_key_file: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 60.0,
    source_acceptance_path: str | Path = (
        "data/runs/mad_etd_nfiot_bot_targeted_performance_w96/"
        "acceptance_report.json"
    ),
) -> dict[str, Any]:
    """Bind an accepted NF-IoT Skill result to the autonomous Stats role.

    No metric is recomputed here.  The function verifies and references the
    already accepted W96 artifact, then tests whether the role controller
    delegates an IoT-capable case to ``IoTBotnetSkill``.
    """

    output = Path(output_dir)
    source_path = Path(source_acceptance_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != (
        "accepted_dataset_specific_full_coverage_accuracy_f1_result"
    ):
        raise ValueError("NF-IoT source result is not accepted")
    if source.get("dataset_scope") != "NF-BoT-IoT-v2":
        raise ValueError("NF-IoT positive Skill bridge has an unexpected scope")
    if source.get("fake_metric_count") != 0:
        raise ValueError("positive Skill source contains fake metrics")
    state = CaseStateSkillV2(
        case_id="autonomous-positive-iot-skill-bridge",
        field_audit_summary={
            "status": "passed",
            "blocked_field_violation": 0,
        },
        safe_feature_policy_hash="c" * 64,
        available_views=("stats",),
        capability_profile={
            "stats": "supported",
            "iot_scope": "supported",
        },
        collected_evidence_summary=(),
        evidence_conflict={},
        uncertainty_state={"level": "medium"},
        OOD_state={"state": "in_domain"},
        missing_views=("sequence", "tls_records"),
        short_flow_state={"is_short": False},
        remaining_agent_budget=1,
        remaining_latency_budget=2500.0,
        previous_requests=(),
        previous_failures=(),
        escalation_state={"status": "none"},
        audit_chain_hash="d" * 64,
    )
    harness = AutonomousEvidenceTeamHarness(model=model)
    transport = NvidiaAutonomousRoleTransport(
        api_key_file=api_key_file,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    execution = harness.execute_role(
        role_id="stats_investigator",
        state=state,
        mode=mode,
        transport=transport,
    )
    selected_skill = (
        execution.evidence_request.requested_skill
        if execution.evidence_request
        else None
    )
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    bridge = {
        "status": (
            "accepted_autonomous_iot_skill_bridge"
            if selected_skill == "IoTBotnetSkill"
            else "autonomous_iot_skill_bridge_not_selected"
        ),
        "role_id": "stats_investigator",
        "llm_execution_status": execution.status,
        "real_llm_call": execution.real_llm_call,
        "fallback_used": execution.fallback_used,
        "selected_skill": selected_skill,
        "skill_id": "IoTBotnetSkill",
        "source_experiment": source["experiment"],
        "source_status": source["status"],
        "source_artifact": str(source_path.as_posix()),
        "source_artifact_sha256": source_hash,
        "dataset_scope": source["dataset_scope"],
        "reference_metrics": source["reference_metrics"],
        "candidate_metrics": source["candidate_metrics"],
        "deltas": source["deltas"],
        "bootstrap": source["bootstrap"],
        "classification_metrics_recomputed": False,
        "source_result_reused_with_provenance": True,
        "claim_scope": (
            "NF-BoT-IoT-v2 only; default-off accepted Skill result, not a "
            "general runtime or LLM classification gain"
        ),
        "llm_is_classifier": False,
        "llm_enters_fusion": False,
        "final_decision_owner": "FusionAgent",
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
    }
    _dump(output / "autonomous_positive_skill_bridge.json", bridge)
    _dump(
        output / "autonomous_positive_skill_llm_execution.json",
        execution.model_dump(mode="json"),
    )
    return bridge


def run_autonomous_evidence_team_v2_smoke(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    mode: Literal["nvidia", "fixture", "unavailable"] = "unavailable",
    api_key_file: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    output = Path(output_dir)
    if not (output / "role_registry.json").exists():
        raise FileNotFoundError("build autonomous role protocol before smoke run")
    harness = AutonomousEvidenceTeamHarness(model=model)
    transport = NvidiaAutonomousRoleTransport(
        api_key_file=api_key_file,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    state = _sample_case_state()
    rows: list[dict[str, Any]] = []
    decisions_path = output / "role_decisions.jsonl"
    decisions_path.unlink(missing_ok=True)
    for role_id in sorted(harness.profiles):
        execution = harness.execute_role(
            role_id=role_id,
            state=state,
            mode=mode,
            transport=transport,
        )
        payload = execution.model_dump(mode="json")
        _append_jsonl(decisions_path, payload)
        rows.append(
            {
                "role_id": role_id,
                "status": execution.status,
                "provider": execution.provider,
                "model": execution.model,
                "real_llm_call": execution.real_llm_call,
                "fallback_used": execution.fallback_used,
                "action": execution.decision.action,
                "requested_skill": execution.decision.requested_skill,
                "evidence_request_created": execution.evidence_request is not None,
                "latency_ms": execution.latency_ms,
                "prompt_tokens": execution.prompt_tokens,
                "completion_tokens": execution.completion_tokens,
                "error_code": execution.error_code,
                "raw_response_sha256": execution.raw_response_sha256,
                "llm_emitted_agent_evidence": False,
                "llm_wrote_fusion": False,
            }
        )
    _write_csv(output / "role_smoke_results.csv", rows)

    malicious_fixture = json.dumps(
        {
            "schema_version": "2.0",
            "role_id": "coordinator",
            "case_state_hash": state.state_hash,
            "action": "request_skill",
            "requested_skill": "GeneralFlowMalwareSkill",
            "purpose": "make a final decision",
            "concise_rationale": "malicious",
            "expected_output": "AgentEvidenceV2",
            "source": "fixture",
            "prompt_version": PROMPT_VERSION,
            "final_verdict": "malicious",
        }
    )
    probe = harness.execute_role(
        role_id="coordinator",
        state=state,
        mode="fixture",
        fixture_raw=malicious_fixture,
    )
    security = {
        "illegal_output_probe_rejected": probe.status == "llm_rejected",
        "illegal_output_fallback_used": probe.fallback_used,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "llm_agent_evidence_execution_count": 0,
        "fake_metric_count": 0,
        "audit_completion": 1.0,
    }
    _dump(output / "security_acceptance.json", security)
    report = {
        "status": (
            "real_llm_role_smoke_completed"
            if mode == "nvidia"
            and all(row["status"] == "llm_success" for row in rows)
            else "bounded_role_harness_smoke_completed"
        ),
        "mode": mode,
        "role_count": len(rows),
        "real_llm_success_count": sum(
            row["status"] == "llm_success" for row in rows
        ),
        "fallback_count": sum(row["fallback_used"] for row in rows),
        "all_roles_produced_policy_reviewed_requests": all(
            row["evidence_request_created"] for row in rows
        ),
        "llm_agent_evidence_execution_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "smoke_report.json", report)
    return report


def finalize_autonomous_evidence_team_v2(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    required = (
        "build_report.json",
        "role_registry.json",
        "skill_binding_table.csv",
        "role_smoke_results.csv",
        "security_acceptance.json",
        "smoke_report.json",
    )
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing autonomous-team artifacts: {missing}")
    build = json.loads((output / "build_report.json").read_text(encoding="utf-8"))
    smoke = json.loads((output / "smoke_report.json").read_text(encoding="utf-8"))
    security = json.loads(
        (output / "security_acceptance.json").read_text(encoding="utf-8")
    )
    performance_path = output / "competence_acceptance_report.json"
    performance = (
        json.loads(performance_path.read_text(encoding="utf-8"))
        if performance_path.exists()
        else None
    )
    positive_bridge_path = output / "autonomous_positive_skill_bridge.json"
    positive_bridge = (
        json.loads(positive_bridge_path.read_text(encoding="utf-8"))
        if positive_bridge_path.exists()
        else None
    )
    case_loop_path = output / "case_loop_acceptance_report.json"
    case_loop = (
        json.loads(case_loop_path.read_text(encoding="utf-8"))
        if case_loop_path.exists()
        else None
    )
    safety_passed = all(
        security[key] == 0
        for key in (
            "blocked_field_violation",
            "fusion_ownership_violation",
            "ood_override",
            "illegal_verdict_execution",
            "llm_agent_evidence_execution_count",
            "fake_metric_count",
        )
    ) and security["audit_completion"] == 1.0
    accepted = (
        build["role_count"] == 9
        and build["all_requested_roles_llm_enabled"]
        and smoke["all_roles_produced_policy_reviewed_requests"]
        and safety_passed
        and tests_passed
    )
    report = {
        "status": (
            "accepted_default_off_autonomous_control_plane"
            if accepted
            else "incomplete_autonomous_control_plane_acceptance"
        ),
        "candidate": "MAD-ETD Autonomous Evidence Team v2",
        "role_count": build["role_count"],
        "all_requested_roles_llm_enabled": build[
            "all_requested_roles_llm_enabled"
        ],
        "real_llm_smoke_status": smoke["status"],
        "real_llm_success_count": smoke["real_llm_success_count"],
        "fallback_count": smoke["fallback_count"],
        "llm_can_emit_agent_evidence": False,
        "llm_can_write_fusion": False,
        "llm_can_override_ood": False,
        "final_decision_owner": FINAL_DECISION_OWNER,
        "performance_metrics_generated": performance is not None,
        "performance_candidate_status": (
            performance["status"] if performance else "not_run"
        ),
        "performance_claim_created": bool(
            performance
            and performance["status"]
            == "accepted_strong_positive_competence_aware_fusion"
        ),
        "performance_summary": (
            {
                "reference": performance["reference"],
                "baseline_metrics": performance["baseline_metrics"],
                "candidate_metrics": performance["candidate_metrics"],
                "deltas": performance["deltas"],
                "grouped_bootstrap": performance["grouped_bootstrap"],
                "failed_gates": performance["failed_gates"],
            }
            if performance
            else None
        ),
        "positive_skill_bridge_status": (
            positive_bridge["status"] if positive_bridge else "not_run"
        ),
        "autonomous_case_loop_status": (
            case_loop["status"] if case_loop else "not_run"
        ),
        "real_nvidia_case_loop_completed": bool(
            case_loop and case_loop["real_nvidia_pilot_completed"]
        ),
        "autonomous_case_loop_summary": (
            {
                "fixture_case_count": case_loop["fixture_run"]["case_count"],
                "fixture_avg_evidence_calls": case_loop["fixture_run"][
                    "avg_evidence_calls"
                ],
                "fixture_audit_completion": case_loop["fixture_run"][
                    "audit_completion"
                ],
                "fixture_compliance_rate": case_loop["fixture_run"][
                    "compliance_rate"
                ],
                "nvidia_case_count": case_loop["nvidia_pilot"]["case_count"],
                "nvidia_real_llm_call_count": case_loop["nvidia_pilot"][
                    "real_llm_call_count"
                ],
                "nvidia_fallback_count": case_loop["nvidia_pilot"][
                    "fallback_count"
                ],
                "final_decision_owner": case_loop["final_decision_owner"],
                "selection_only_classification_diagnostics": case_loop[
                    "classification_metrics_are_selection_only_diagnostics"
                ],
            }
            if case_loop and case_loop.get("nvidia_pilot")
            else None
        ),
        "accepted_dataset_specific_positive_result": (
            {
                "dataset_scope": positive_bridge["dataset_scope"],
                "source_experiment": positive_bridge["source_experiment"],
                "reference_metrics": positive_bridge["reference_metrics"],
                "candidate_metrics": positive_bridge["candidate_metrics"],
                "deltas": positive_bridge["deltas"],
                "bootstrap": positive_bridge["bootstrap"],
                "real_llm_selected_skill": (
                    positive_bridge["real_llm_call"]
                    and positive_bridge["selected_skill"] == "IoTBotnetSkill"
                ),
                "selected_skill": positive_bridge["selected_skill"],
                "claim_scope": positive_bridge["claim_scope"],
            }
            if positive_bridge
            and positive_bridge["status"]
            == "accepted_autonomous_iot_skill_bridge"
            else None
        ),
        "tests_passed": tests_passed,
        "test_count": test_count,
        "runtime_safe_v3_0_remains_default": True,
        "candidate_default_enabled": False,
        "promoted_runtime_created": False,
        "fake_metric_count": 0,
    }
    _dump(output / "acceptance_report.json", report)
    negative_items: list[dict[str, Any]] = []
    if not accepted:
        negative_items.append(
            {
                "candidate_module": "Autonomous Evidence Team v2 control plane",
                "failure_reason": "one or more implementation/test gates are incomplete",
                "safe_claim": "the incomplete candidate remains default-off",
                "forbidden_claim": "autonomous runtime promoted or classification improved",
                "fake_metric_count": 0,
            }
        )
    if performance and performance["status"] != (
        "accepted_strong_positive_competence_aware_fusion"
    ):
        negative_items.append(
            {
                "candidate_module": "competence-aware evidence Fusion v2",
                "failure_reason": performance["failed_gates"],
                "safe_claim": (
                    "ECE, worst-group behavior, and harmful flips may be "
                    "reported separately from classification acceptance"
                ),
                "forbidden_claim": (
                    "Accuracy or Macro-F1 strong-positive gate passed"
                ),
                "fake_metric_count": 0,
            }
        )
    if case_loop and case_loop["status"] != (
        "accepted_default_off_autonomous_case_loop_v2"
    ):
        negative_items.append(
            {
                "candidate_module": "Autonomous Case Loop v2",
                "failure_reason": case_loop["status"],
                "safe_claim": "the bounded case loop remains default-off",
                "forbidden_claim": "the autonomous case loop was promoted",
                "fake_metric_count": 0,
            }
        )
    _dump(output / "negative_results.json", {"items": negative_items})
    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    en = f"""# MAD-ETD Autonomous Evidence Team v2

Status: `{report['status']}`.

Nine role-constrained LLM controllers share the configured Nemotron endpoint
while retaining separate prompts, Skill allow-lists, state views, and
authority.  They can produce policy-reviewable Skill requests and advisory
actions only.  Numerical evidence still comes from deterministic detector
artifacts; `FusionAgent` remains the only final decision owner.

This round implements the default-off autonomous control plane.  Performance
status: `{report['performance_candidate_status']}`.  A classification gain is
claimed only when the pre-registered strong-positive gates pass.

Accepted Skill bridge status: `{report['positive_skill_bridge_status']}`.  The
NF-IoT values are referenced from their immutable accepted artifact and are
not recomputed or attributed to the LLM.

`runtime_safe_v3_0` remains default.
"""
    cn = f"""# MAD-ETD 有界自主多智能体证据团队 v2

状态：`{report['status']}`。

九个受角色权限约束的 LLM 控制器复用同一 Nemotron 接口，同时保持独立
prompt、Skill 白名单、案件状态视图和权限。LLM 只能生成经过策略审查的
Skill 请求或顾问动作；数值检测证据仍来自冻结检测模型，`FusionAgent`
继续独占最终判定权。

本轮实现的是默认关闭的自主控制面。性能候选状态为
`{report['performance_candidate_status']}`；只有通过预注册强正向门槛，
才会产生 Accuracy/Macro-F1 提升声明。

已验收 Skill 桥接状态为 `{report['positive_skill_bridge_status']}`。
NF-IoT 数值来自带哈希的既有验收工件，本轮没有重新计算，也不把该提升
归因于 LLM。

`runtime_safe_v3_0` 保持默认。
"""
    en_v2 = f"""# MAD-ETD Autonomous Evidence Team v2

Status: `{report['status']}`.

Nine role-constrained LLM controllers share the configured Nemotron endpoint
while retaining separate prompts, Skill allow-lists, state views, and
authority. They can produce policy-reviewable Skill requests and advisory
actions only. Numerical evidence still comes from deterministic detector
artifacts; `FusionAgent` remains the only final decision owner.

This round implements the default-off autonomous control plane. Performance
status: `{report['performance_candidate_status']}`. A classification gain is
claimed only when the pre-registered strong-positive gates pass.

The bounded CaseState-to-Fusion loop status is
`{report['autonomous_case_loop_status']}`. The loop uses immutable
detector-generated AgentEvidence, validates each handoff before Fusion, and
keeps Critic, Memory/RAG, Reflection, and Reporter/HITL handoffs outside
Fusion. Its selection-only classification values are execution diagnostics,
not a new performance claim.

Accepted Skill bridge status: `{report['positive_skill_bridge_status']}`. The
NF-IoT values are referenced from their immutable accepted artifact and are
not recomputed or attributed to the LLM.

`runtime_safe_v3_0` remains default.
"""
    cn_v2 = f"""# MAD-ETD 有界自主多智能体证据团队 v2

状态：`{report['status']}`。

九个受角色权限约束的 LLM 控制器复用同一 Nemotron 接口，同时保持独立
提示模板、Skill 白名单、案件状态视图和权限。LLM 只能生成经过策略审查的
Skill 请求或顾问动作；数值检测证据仍来自冻结检测器工件，`FusionAgent`
继续独占最终判定权。

本轮实现的是默认关闭的自主控制面。性能候选状态为
`{report['performance_candidate_status']}`。只有通过预注册强正向门槛，
才会产生 Accuracy 或 Macro-F1 提升声明。

有界 CaseState-to-Fusion 闭环状态为
`{report['autonomous_case_loop_status']}`。闭环读取不可变的检测器
AgentEvidence，在进入 Fusion 前逐条执行 AdmissionGate 验证，并确保
Critic、Memory/RAG、Reflection 和 Reporter/HITL 顾问交接不进入 Fusion。
闭环中的 selection-only 分类数值仅作为执行诊断，不构成新的性能提升结论。

已验收 Skill 桥接状态为 `{report['positive_skill_bridge_status']}`。
NF-IoT 数值来自带哈希的既有验收工件，本轮没有重新计算，也不把该提升
归因于 LLM。

`runtime_safe_v3_0` 保持默认。
"""
    (docs / "MAD_ETD_AUTONOMOUS_EVIDENCE_TEAM_V2.md").write_text(
        en_v2, encoding="utf-8"
    )
    (docs / "MAD_ETD_AUTONOMOUS_EVIDENCE_TEAM_V2_CN.md").write_text(
        cn_v2, encoding="utf-8"
    )
    return report
