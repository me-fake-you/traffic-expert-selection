"""Small, dependency-free lazy second-expert runtime contract.

This module is deliberately independent from the historical experiment
scripts.  It provides the boundary needed by E1's first-state-only pilot:
the routing state is constructed from first-expert information, while a
second provider is called only after an explicit acquisition decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Mapping, MutableMapping, Optional
from types import MappingProxyType
import math
import time


class StateLeakageError(ValueError):
    """Raised when a routing state contains an unapproved/future field."""


class AcquisitionNotPermitted(RuntimeError):
    """Raised when a new second-expert call lacks capability or budget."""


class SecondOutputUnavailable(RuntimeError):
    """Raised when code tries to read the second output before acquisition."""


class DuplicateFirstExecution(RuntimeError):
    """Raised when the first expert is executed more than once for a case."""


FIRST_STATE_FIELDS = frozenset(
    {
        "first_probability",
        "first_confidence",
        "visible_prefix_length",
        "quality_state",
        "ability_available",
        "budget_remaining",
    }
)

# These names are rejected even if a caller tries to smuggle them into an
# otherwise extensible mapping.  The check is intentionally conservative:
# labels, groups, future-flow statistics, and second-view information do not
# belong to a pre-acquisition online state.
FORBIDDEN_STATE_TOKENS = frozenset(
    {
        "second_probability",
        "second_confidence",
        "second_view",
        "future",
        "end_timestamp",
        "full_flow",
        "label",
        "group",
        "family",
        "sample_hash",
    }
)

QUALITY_STATUS_ENUM = frozenset({"ok", "degraded", "missing", "unavailable"})
QUALITY_FIELDS = frozenset({"missing_count", "status", "truncated"})


def _finite_number(value: Any, name: str, *, lower: Optional[float] = None, upper: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return result


def _normalise_quality(value: Any) -> Any:
    if isinstance(value, str):
        if value not in QUALITY_STATUS_ENUM:
            raise ValueError(f"quality_state status is outside the fixed enum: {value!r}")
        return value
    if not isinstance(value, Mapping) or not value:
        raise ValueError("quality_state must be a fixed status string or a non-empty mapping")
    unknown = set(value) - QUALITY_FIELDS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise StateLeakageError(f"quality_state contains unapproved fields: {names}")
    normalised: dict[str, Any] = {}
    for key, item in value.items():
        if key == "missing_count":
            if isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0:
                raise ValueError("quality_state.missing_count must be a non-negative integer")
            normalised[key] = int(item)
        elif key == "status":
            if not isinstance(item, str) or item not in QUALITY_STATUS_ENUM:
                raise ValueError("quality_state.status is outside the fixed enum")
            normalised[key] = item
        elif key == "truncated":
            if not isinstance(item, bool):
                raise ValueError("quality_state.truncated must be bool")
            normalised[key] = item
    return MappingProxyType(normalised)


@dataclass(frozen=True)
class PreAcquisitionState:
    """Immutable state available before asking for the second expert.

    ``quality_state`` and ``ability_available`` may be structured values, but
    their contents must be prepared before this object is built.  The object
    itself has no reference to either expert or to a second probability.
    """

    first_probability: float
    first_confidence: float
    visible_prefix_length: int
    quality_state: Any
    ability_available: bool
    budget_remaining: float

    def __post_init__(self) -> None:
        """Validate every construction path and freeze quality metadata."""

        probability = _finite_number(self.first_probability, "first_probability", lower=0.0, upper=1.0)
        confidence = _finite_number(self.first_confidence, "first_confidence", lower=0.0, upper=1.0)
        if isinstance(self.visible_prefix_length, bool) or not isinstance(self.visible_prefix_length, Integral):
            raise ValueError("visible_prefix_length must be a non-negative integer")
        prefix = int(self.visible_prefix_length)
        if prefix < 0:
            raise ValueError("visible_prefix_length must be a non-negative integer")
        if not isinstance(self.ability_available, bool):
            raise ValueError("ability_available must be bool")
        budget = _finite_number(self.budget_remaining, "budget_remaining", lower=0.0)
        quality = _normalise_quality(self.quality_state)
        object.__setattr__(self, "first_probability", probability)
        object.__setattr__(self, "first_confidence", confidence)
        object.__setattr__(self, "visible_prefix_length", prefix)
        object.__setattr__(self, "budget_remaining", budget)
        object.__setattr__(self, "quality_state", quality)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PreAcquisitionState":
        validate_pre_acquisition_mapping(values)
        keys = set(values)
        unexpected = keys - FIRST_STATE_FIELDS
        if unexpected:
            names = ", ".join(sorted(map(str, unexpected)))
            raise StateLeakageError(f"pre-acquisition state contains unapproved fields: {names}")
        missing = FIRST_STATE_FIELDS - keys
        if missing:
            names = ", ".join(sorted(missing))
            raise StateLeakageError(f"pre-acquisition state is incomplete: {names}")
        return cls(
            first_probability=values["first_probability"],
            first_confidence=values["first_confidence"],
            visible_prefix_length=values["visible_prefix_length"],
            quality_state=values["quality_state"],
            ability_available=values["ability_available"],
            budget_remaining=values["budget_remaining"],
        )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "first_probability": self.first_probability,
            "first_confidence": self.first_confidence,
            "visible_prefix_length": self.visible_prefix_length,
            "quality_state": dict(self.quality_state) if isinstance(self.quality_state, Mapping) else self.quality_state,
            "ability_available": self.ability_available,
            "budget_remaining": self.budget_remaining,
        }


def validate_pre_acquisition_mapping(values: Mapping[str, Any]) -> None:
    """Reject field names that could leak future, second-expert, or label data."""

    if not isinstance(values, Mapping):
        raise TypeError("pre-acquisition state must be a mapping")

    def check(value: Any, *, top_level: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                name = str(key).lower()
                if top_level and key in FIRST_STATE_FIELDS:
                    check(nested, top_level=False)
                    continue
                if any(token in name for token in FORBIDDEN_STATE_TOKENS):
                    raise StateLeakageError(f"forbidden pre-acquisition field: {key}")
                check(nested, top_level=False)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                check(nested, top_level=False)

    for key in values:
        name = str(key).lower()
        if key in FIRST_STATE_FIELDS:
            continue
        if any(token in name for token in FORBIDDEN_STATE_TOKENS):
            raise StateLeakageError(f"forbidden pre-acquisition field: {key}")
        raise StateLeakageError(f"unapproved pre-acquisition field: {key}")
    check(values, top_level=True)


@dataclass(frozen=True)
class RuntimeEvent:
    """One auditable event.  ``monotonic_ns`` is process-monotonic time."""

    sequence: int
    kind: str
    case_id: str
    expert: str
    monotonic_ns: int
    actual_call_count: int
    detail: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "case_id": self.case_id,
            "expert": self.expert,
            "monotonic_ns": self.monotonic_ns,
            "actual_call_count": self.actual_call_count,
            "detail": self.detail,
        }


class LazySecondExpert:
    """Request-gated, cache-aware wrapper for a second expert.

    ``provider`` receives only ``case_id``.  This keeps the second model from
    receiving the pre-acquisition state by accident and makes it impossible
    for this wrapper to inspect a second probability before the request is
    accepted.  A successful result is cached by case ID; cache hits emit an
    event but never increment ``actual_call_count``.
    """

    def __init__(
        self,
        provider: Callable[[str], Any],
        *,
        expert_name: str = "second",
        cache: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self._provider = provider
        self.expert_name = expert_name
        self._cache: MutableMapping[str, Any] = cache if cache is not None else {}
        self._outputs: dict[str, Any] = {}
        self.events: list[RuntimeEvent] = []
        self.actual_call_count = 0
        self.request_count = 0
        self.cache_hit_count = 0
        self._sequence = 0

    def _emit(self, kind: str, case_id: str, detail: Optional[str] = None) -> None:
        self._sequence += 1
        self.events.append(
            RuntimeEvent(
                sequence=self._sequence,
                kind=kind,
                case_id=str(case_id),
                expert=self.expert_name,
                monotonic_ns=time.monotonic_ns(),
                actual_call_count=self.actual_call_count,
                detail=detail,
            )
        )

    def request(
        self,
        case_id: str,
        state: PreAcquisitionState,
        *,
        should_execute: bool,
    ) -> Optional[Any]:
        """Record a request and execute only when ``should_execute`` is true.

        The state is validated for its type and never passed to ``provider``.
        A false decision returns ``None`` and does not materialize or read a
        second output.
        """

        if not isinstance(state, PreAcquisitionState):
            raise TypeError("request requires a PreAcquisitionState")
        case_id = str(case_id)
        self.request_count += 1
        self._emit("request", case_id, detail="accepted" if should_execute else "not_triggered")
        if not should_execute:
            return None
        if case_id in self._cache:
            self.cache_hit_count += 1
            self._outputs[case_id] = self._cache[case_id]
            self._emit("cache_hit", case_id, detail="no_new_call_budget_used")
            return self._outputs[case_id]

        if not state.ability_available:
            self._emit("failure", case_id, detail="AcquisitionNotPermitted: ability_unavailable")
            raise AcquisitionNotPermitted("second-expert ability is unavailable for a new call")
        if state.budget_remaining < 1.0:
            self._emit("failure", case_id, detail="AcquisitionNotPermitted: insufficient_new_call_budget")
            raise AcquisitionNotPermitted("budget_remaining must be at least 1 for a new second-expert call")

        self.actual_call_count += 1
        try:
            value = self._provider(case_id)
        except Exception as exc:
            self._emit("failure", case_id, detail=f"{type(exc).__name__}: {exc}")
            raise
        self._cache[case_id] = value
        self._outputs[case_id] = value
        self._emit("execution", case_id)
        return value

    def read(self, case_id: str) -> Any:
        """Read an acquired result; never returns an unrequested result."""

        case_id = str(case_id)
        if case_id not in self._outputs:
            raise SecondOutputUnavailable(f"second output unavailable for case {case_id}")
        return self._outputs[case_id]


class LazyCase:
    """Minimal first-then-second case runner used by the runtime tests."""

    def __init__(
        self,
        case_id: str,
        first_provider: Callable[[str], Any],
        second_expert: LazySecondExpert,
    ) -> None:
        self.case_id = str(case_id)
        self._first_provider = first_provider
        self.second_expert = second_expert
        self.first_output: Any = None
        self._first_executed = False

    def execute_first(self) -> Any:
        if self._first_executed:
            raise DuplicateFirstExecution(f"first expert already executed for case {self.case_id}")
        # Commit the phase transition only after the provider succeeds.  A
        # failed first call may be retried, but it can never open phase two.
        output = self._first_provider(self.case_id)
        self.first_output = output
        self._first_executed = True
        return self.first_output

    def request_second(self, state: PreAcquisitionState, *, should_execute: bool) -> Optional[Any]:
        if not self._first_executed:
            raise RuntimeError("first expert must execute before requesting second expert")
        return self.second_expert.request(self.case_id, state, should_execute=should_execute)
