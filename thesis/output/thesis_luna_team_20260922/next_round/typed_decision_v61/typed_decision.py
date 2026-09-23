"""Small typed probability-decision boundary for the Luna/Jev-inspired pilot.

This is a decision adapter, not a model, a classifier, a queue, or a Jev/RLCD
implementation.  It accepts only a three-outcome probability distribution and
returns a deterministic routing decision.  Final traffic labels are outside
this module and are never accepted as inputs or outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Callable, Iterable, Mapping, Optional
import math


RESULT_FIELDS = frozenset({"corrects_error", "introduces_error", "unchanged_correctness"})


class DecisionInputError(ValueError):
    """Raised when a distribution or deterministic routing input is invalid."""


def _finite(value: Any, name: str, *, lower: Optional[float] = None, upper: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DecisionInputError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise DecisionInputError(f"{name} must be finite")
    if lower is not None and result < lower:
        raise DecisionInputError(f"{name} must be >= {lower}")
    if upper is not None and result > upper:
        raise DecisionInputError(f"{name} must be <= {upper}")
    return result


@dataclass(frozen=True)
class DecisionDistribution:
    """Immutable, complete probability distribution for the three outcomes."""

    corrects_error: float
    introduces_error: float
    unchanged_correctness: float

    def __post_init__(self) -> None:
        values = {
            "corrects_error": _finite(self.corrects_error, "corrects_error", lower=0.0, upper=1.0),
            "introduces_error": _finite(self.introduces_error, "introduces_error", lower=0.0, upper=1.0),
            "unchanged_correctness": _finite(self.unchanged_correctness, "unchanged_correctness", lower=0.0, upper=1.0),
        }
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise DecisionInputError(f"probabilities must sum to 1.0; got {total!r}")
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionDistribution":
        if not isinstance(value, Mapping):
            raise DecisionInputError("decision output must be a mapping")
        keys = set(value)
        if keys != RESULT_FIELDS:
            unknown = sorted(map(str, keys - RESULT_FIELDS))
            missing = sorted(RESULT_FIELDS - keys)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise DecisionInputError("decision output must contain exactly the three result fields (" + ", ".join(details) + ")")
        return cls(
            corrects_error=value["corrects_error"],
            introduces_error=value["introduces_error"],
            unchanged_correctness=value["unchanged_correctness"],
        )

    @property
    def expected_delta(self) -> float:
        """Expected correctness change before subtracting an acquisition cost."""
        return self.corrects_error - self.introduces_error

    def as_mapping(self) -> dict[str, float]:
        return {
            "corrects_error": self.corrects_error,
            "introduces_error": self.introduces_error,
            "unchanged_correctness": self.unchanged_correctness,
        }


@dataclass(frozen=True)
class DecisionCandidate:
    """One independently scored case; no label or final classification is stored."""

    case_id: str
    distribution: DecisionDistribution
    cost: float
    ability_available: bool
    budget_remaining: float
    already_called: bool = False
    cache_available: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise DecisionInputError("case_id must be a non-empty string")
        if not isinstance(self.distribution, DecisionDistribution):
            raise DecisionInputError("distribution must be DecisionDistribution")
        object.__setattr__(self, "cost", _finite(self.cost, "cost", lower=0.0))
        object.__setattr__(self, "budget_remaining", _finite(self.budget_remaining, "budget_remaining", lower=0.0))
        for name in ("ability_available", "already_called", "cache_available"):
            if not isinstance(getattr(self, name), bool):
                raise DecisionInputError(f"{name} must be bool")

    @property
    def expected_delta(self) -> float:
        return self.distribution.expected_delta


@dataclass(frozen=True)
class RouteDecision:
    case_id: str
    action: str
    status: str
    expected_delta: Optional[float]
    net_value: Optional[float]
    reason: str
    rank: Optional[int] = None


@dataclass(frozen=True)
class FixedQuotaPlan:
    """New-call selections and cache reuses kept in separate ledgers."""

    selected_new_calls: tuple[RouteDecision, ...]
    cache_reuses: tuple[RouteDecision, ...]


@dataclass(frozen=True)
class TrustedCacheHandle:
    """Explicit view of the actual cache shared with a lazy runner.

    ``cache_available`` on a candidate is only an advisory flag.  A caller
    must pass this handle to prove the case key, pre-acquisition cache key,
    and actual mapping agree before cache reuse is allowed.
    """

    case_id: str
    cache_key: str
    state_cache_key: str
    store: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise DecisionInputError("trusted cache case_id must be non-empty")
        if not isinstance(self.cache_key, str) or not self.cache_key:
            raise DecisionInputError("trusted cache cache_key must be non-empty")
        if not isinstance(self.state_cache_key, str) or not self.state_cache_key:
            raise DecisionInputError("trusted cache state_cache_key must be non-empty")
        if not isinstance(self.store, Mapping):
            raise DecisionInputError("trusted cache store must be a mapping")

    def matches(self, case_id: str, pre_acquisition_state: Any) -> bool:
        return (
            self.case_id == str(case_id)
            and getattr(pre_acquisition_state, "cache_key", None) == self.state_cache_key
            and self.cache_key in self.store
        )

    def value(self) -> Any:
        return self.store[self.cache_key]


@dataclass(frozen=True)
class AdapterExecution:
    """Result of handing an accepted route to the existing lazy runner."""

    route: RouteDecision
    output: Any = None
    failure_type: Optional[str] = None


def decide_single(candidate: DecisionCandidate, lambda_cost: float) -> RouteDecision:
    """Apply deterministic cache/capability/budget/cost gates to one case."""
    lam = _finite(lambda_cost, "lambda_cost", lower=0.0)
    if candidate.cache_available:
        if candidate.expected_delta <= 0.0:
            return RouteDecision(candidate.case_id, "stop", "cache_harm_stop", candidate.expected_delta, candidate.expected_delta, "cached evidence is free to retrieve but predicted to harm correctness")
        return RouteDecision(candidate.case_id, "use_cache", "cache_hit", candidate.expected_delta, candidate.expected_delta, "cached evidence has no new call cost")
    if candidate.already_called:
        return RouteDecision(candidate.case_id, "stop", "already_called", candidate.expected_delta, candidate.expected_delta, "new call is forbidden after a non-cached call")
    if not candidate.ability_available:
        return RouteDecision(candidate.case_id, "stop", "ability_unavailable", candidate.expected_delta, candidate.expected_delta, "second expert is unavailable")
    if candidate.budget_remaining < candidate.cost:
        return RouteDecision(candidate.case_id, "stop", "budget_insufficient", candidate.expected_delta, candidate.expected_delta - lam * candidate.cost, "new call budget is below cost")
    net_value = candidate.expected_delta - lam * candidate.cost
    if net_value <= 0.0:
        return RouteDecision(candidate.case_id, "stop", "cost_threshold", candidate.expected_delta, net_value, "expected delta does not exceed lambda times cost")
    return RouteDecision(candidate.case_id, "execute", "approved", candidate.expected_delta, net_value, "expected delta exceeds lambda times cost")


def rank_fixed_quota(candidates: Iterable[DecisionCandidate], quota: int) -> FixedQuotaPlan:
    """Rank a batch once and select up to ``quota`` eligible candidates.

    This function intentionally owns no queue and makes no provider calls.  A
    caller supplies the complete comparison batch; ties are resolved by case
    ID, so equal probabilities are not treated as independent extra evidence.
    ``quota`` counts new calls only. Cached cases are returned separately in
    ``FixedQuotaPlan.cache_reuses`` and therefore never consume that quota.
    """
    if isinstance(quota, bool) or not isinstance(quota, Integral) or quota < 0:
        raise DecisionInputError("quota must be a non-negative integer")
    values = tuple(candidates)
    if len({candidate.case_id for candidate in values}) != len(values):
        raise DecisionInputError("fixed-quota batch contains duplicate case IDs")
    cached = [candidate for candidate in values if candidate.cache_available]
    cache_reuses = tuple(decide_single(candidate, lambda_cost=0.0) for candidate in cached)
    eligible = [
        candidate for candidate in values
        if not candidate.cache_available
        and candidate.ability_available
        and not candidate.already_called
        and candidate.budget_remaining >= candidate.cost
    ]
    ordered = sorted(eligible, key=lambda candidate: (-candidate.expected_delta, candidate.case_id))
    selected = ordered[: int(quota)]
    selected = tuple(
        replace(
            decide_single(candidate, lambda_cost=0.0),
            action="execute",
            status="fixed_quota_selected",
            reason="selected by deterministic fixed-quota ranking",
            rank=index,
        )
        for index, candidate in enumerate(selected, start=1)
    )
    return FixedQuotaPlan(selected_new_calls=selected, cache_reuses=cache_reuses)


def route_model_output(candidate: DecisionCandidate, model_output: Mapping[str, Any], *, lambda_cost: float) -> RouteDecision:
    """Parse exactly one model output; malformed output deterministically stops."""
    try:
        distribution = DecisionDistribution.from_mapping(model_output)
    except (DecisionInputError, TypeError, ValueError) as exc:
        return RouteDecision(candidate.case_id, "stop", "model_failure", None, None, f"invalid model decision: {type(exc).__name__}")
    scored = replace(candidate, distribution=distribution)
    return decide_single(scored, lambda_cost)


def execute_with_lazy_runner(
    candidate: DecisionCandidate,
    model_output: Mapping[str, Any],
    *,
    lambda_cost: float,
    lazy_runner: Any,
    pre_acquisition_state: Any,
    trusted_cache: Optional[TrustedCacheHandle] = None,
) -> AdapterExecution:
    """Adapt the typed route to the existing ``LazySecondExpert`` interface."""
    route = route_model_output(candidate, model_output, lambda_cost=lambda_cost)
    if route.action not in {"execute", "use_cache"}:
        return AdapterExecution(route=route)
    if route.action == "use_cache":
        # The default is deliberately absent: a candidate's boolean cache
        # claim cannot authorize a new lazy-runner request.
        if trusted_cache is None:
            guarded = replace(route, action="stop", status="cache_unverified", reason="candidate cache declaration has no trusted cache handle")
            return AdapterExecution(route=guarded)
        if not trusted_cache.matches(candidate.case_id, pre_acquisition_state):
            guarded = replace(route, action="stop", status="cache_miss_no_execution", reason="trusted cache key/state/case check failed")
            return AdapterExecution(route=guarded)
        # Read from the verified shared mapping directly.  This cannot fall
        # through to LazySecondExpert.request and cannot create a provider call
        # after a cache miss.
        try:
            cached_value = trusted_cache.value()
        except KeyError:
            guarded = replace(route, action="stop", status="cache_miss_no_execution", reason="verified cache disappeared before read")
            return AdapterExecution(route=guarded)
        return AdapterExecution(route=route, output=cached_value)
    try:
        output = lazy_runner.request(candidate.case_id, pre_acquisition_state, should_execute=True)
    except Exception as exc:  # provider failure is a fallback stop, never unchanged correctness
        failed = replace(route, action="stop", status="execution_failure", reason=f"lazy runner failed: {type(exc).__name__}")
        return AdapterExecution(route=failed, failure_type=type(exc).__name__)
    return AdapterExecution(route=route, output=output)
