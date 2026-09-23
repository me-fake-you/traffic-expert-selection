"""Contract tests for the first-state-only lazy second-expert pilot."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lazy_acquisition import (  # noqa: E402
    AcquisitionNotPermitted,
    LazyCase,
    LazySecondExpert,
    PreAcquisitionState,
    SecondOutputUnavailable,
    StateLeakageError,
)


def state(**overrides):
    values = {
        "first_probability": 0.42,
        "first_confidence": 0.58,
        "visible_prefix_length": 8,
        "quality_state": {"missing_count": 0},
        "ability_available": True,
        "budget_remaining": 1.0,
    }
    values.update(overrides)
    return PreAcquisitionState.from_mapping(values)


class LazyAcquisitionContractTests(unittest.TestCase):
    def test_direct_constructor_validates_and_freezes_quality_state(self):
        source_quality = {"status": "ok", "missing_count": 0, "truncated": False}
        constructed = PreAcquisitionState(
            first_probability=0.4,
            first_confidence=0.6,
            visible_prefix_length=8,
            quality_state=source_quality,
            ability_available=True,
            budget_remaining=1.0,
        )
        source_quality["missing_count"] = 9
        self.assertEqual(constructed.quality_state["missing_count"], 0)
        with self.assertRaises(TypeError):
            constructed.quality_state["status"] = "degraded"
        with self.assertRaises(StateLeakageError):
            PreAcquisitionState(0.4, 0.6, 8, {"pS": 0.8}, True, 1.0)

    def test_quality_state_only_allows_enumerated_fields_and_values(self):
        for invalid in (
            {"pS": 0.8},
            {"second_probability": 0.8},
            {"missing_count": -1},
            {"missing_count": True},
            {"status": "future"},
            {"truncated": 1},
        ):
            with self.assertRaises((StateLeakageError, ValueError)):
                PreAcquisitionState(0.4, 0.6, 8, invalid, True, 1.0)

    def test_numeric_and_boolean_boundaries_are_rejected(self):
        valid_quality = "ok"
        invalid_cases = (
            {"first_probability": True},
            {"first_probability": float("nan")},
            {"first_probability": 1.1},
            {"first_confidence": float("inf")},
            {"visible_prefix_length": True},
            {"visible_prefix_length": -1},
            {"ability_available": 1},
            {"budget_remaining": -0.1},
            {"budget_remaining": float("inf")},
        )
        for override in invalid_cases:
            values = {
                "first_probability": 0.4,
                "first_confidence": 0.6,
                "visible_prefix_length": 8,
                "quality_state": valid_quality,
                "ability_available": True,
                "budget_remaining": 1.0,
            }
            values.update(override)
            with self.assertRaises(ValueError):
                PreAcquisitionState(**values)

    def test_first_only_never_reads_or_invokes_second_probability(self):
        first_calls = []
        second_calls = []

        def first(case_id):
            first_calls.append(case_id)
            return 0.42

        def forbidden_second(case_id):
            second_calls.append(case_id)
            raise AssertionError("second provider was called on first-only path")

        second = LazySecondExpert(forbidden_second)
        case = LazyCase("first-only", first, second)
        self.assertEqual(case.execute_first(), 0.42)
        self.assertIsNone(case.request_second(state(), should_execute=False))
        with self.assertRaises(SecondOutputUnavailable):
            second.read("first-only")
        self.assertEqual(first_calls, ["first-only"])
        self.assertEqual(second_calls, [])
        self.assertEqual(second.actual_call_count, 0)
        self.assertEqual([event.kind for event in second.events], ["request"])
        self.assertEqual(second.events[0].detail, "not_triggered")

    def test_nested_second_probability_is_rejected_before_route(self):
        values = state().as_mapping()
        values["quality_state"] = {"second_probability": 0.99}
        with self.assertRaises(StateLeakageError):
            PreAcquisitionState.from_mapping(values)

        values = state().as_mapping()
        values["future_length"] = 100
        with self.assertRaises(StateLeakageError):
            PreAcquisitionState.from_mapping(values)

    def test_triggered_path_executes_once_and_records_monotonic_events(self):
        calls = []

        def provider(case_id):
            calls.append(case_id)
            return 0.91

        second = LazySecondExpert(provider)
        case = LazyCase("triggered", lambda _: 0.49, second)
        case.execute_first()
        self.assertEqual(case.request_second(state(), should_execute=True), 0.91)
        self.assertEqual(second.read("triggered"), 0.91)
        self.assertEqual(calls, ["triggered"])
        self.assertEqual(second.actual_call_count, 1)
        self.assertEqual([event.kind for event in second.events], ["request", "execution"])
        times = [event.monotonic_ns for event in second.events]
        self.assertEqual(times, sorted(times))
        self.assertEqual([event.actual_call_count for event in second.events], [0, 1])

    def test_cache_reuse_does_not_increment_actual_call_count(self):
        calls = []

        def provider(case_id):
            calls.append(case_id)
            return 0.73

        second = LazySecondExpert(provider)
        case = LazyCase("cached", lambda _: 0.51, second)
        case.execute_first()
        self.assertEqual(case.request_second(state(), should_execute=True), 0.73)
        self.assertEqual(case.request_second(state(), should_execute=True), 0.73)
        self.assertEqual(calls, ["cached"])
        self.assertEqual(second.request_count, 2)
        self.assertEqual(second.actual_call_count, 1)
        self.assertEqual(second.cache_hit_count, 1)
        self.assertEqual([event.kind for event in second.events], ["request", "execution", "request", "cache_hit"])
        self.assertEqual(second.events[-1].detail, "no_new_call_budget_used")

    def test_unavailable_or_under_budget_rejects_new_call_but_cache_hit_is_allowed(self):
        provider_calls = []

        def provider(case_id):
            provider_calls.append(case_id)
            return 0.88

        unavailable = LazySecondExpert(provider)
        with self.assertRaises(AcquisitionNotPermitted):
            unavailable.request(
                "no-ability",
                state(ability_available=False),
                should_execute=True,
            )
        under_budget = LazySecondExpert(provider)
        with self.assertRaises(AcquisitionNotPermitted):
            under_budget.request(
                "no-budget",
                state(budget_remaining=0.5),
                should_execute=True,
            )
        cached = LazySecondExpert(provider, cache={"replay": 0.77})
        self.assertEqual(
            cached.request("replay", state(ability_available=False, budget_remaining=0.0), should_execute=True),
            0.77,
        )
        self.assertEqual(provider_calls, [])
        self.assertEqual(cached.actual_call_count, 0)
        self.assertEqual(cached.cache_hit_count, 1)
        self.assertEqual(cached.events[-1].kind, "cache_hit")

    def test_failure_is_logged_and_failed_call_is_not_cached(self):
        attempts = []

        def flaky(case_id):
            attempts.append(case_id)
            if len(attempts) == 1:
                raise TimeoutError("synthetic timeout")
            return 0.64

        second = LazySecondExpert(flaky)
        case = LazyCase("failure", lambda _: 0.5, second)
        case.execute_first()
        with self.assertRaises(TimeoutError):
            case.request_second(state(), should_execute=True)
        self.assertEqual(case.request_second(state(), should_execute=True), 0.64)
        self.assertEqual(attempts, ["failure", "failure"])
        self.assertEqual(second.actual_call_count, 2)
        self.assertEqual(second.cache_hit_count, 0)
        self.assertEqual([event.kind for event in second.events], ["request", "failure", "request", "execution"])
        self.assertIn("TimeoutError", second.events[1].detail)

    def test_first_failure_does_not_open_second_phase_and_retry_is_explicit(self):
        first_attempts = []

        def flaky_first(case_id):
            first_attempts.append(case_id)
            if len(first_attempts) == 1:
                raise TimeoutError("first expert unavailable")
            return 0.41

        second_calls = []
        second = LazySecondExpert(lambda case_id: second_calls.append(case_id) or 0.9)
        case = LazyCase("first-failure", flaky_first, second)
        with self.assertRaises(TimeoutError):
            case.execute_first()
        with self.assertRaises(RuntimeError):
            case.request_second(state(), should_execute=True)
        self.assertEqual(case.execute_first(), 0.41)
        self.assertEqual(case.request_second(state(), should_execute=True), 0.9)
        self.assertEqual(first_attempts, ["first-failure", "first-failure"])
        self.assertEqual(second_calls, ["first-failure"])


if __name__ == "__main__":
    unittest.main()
