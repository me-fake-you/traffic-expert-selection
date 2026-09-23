import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNTIME = HERE.parents[1] / "runtime"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RUNTIME))

from typed_decision import (  # noqa: E402
    DecisionCandidate,
    DecisionDistribution,
    DecisionInputError,
    TrustedCacheHandle,
    decide_single,
    execute_with_lazy_runner,
    rank_fixed_quota,
    route_model_output,
)
from lazy_acquisition import LazySecondExpert, PreAcquisitionState  # noqa: E402


def dist(corrects=0.6, introduces=0.2, unchanged=0.2):
    return DecisionDistribution(corrects, introduces, unchanged)


def candidate(case_id="case-a", **kwargs):
    return DecisionCandidate(case_id, dist(), cost=kwargs.pop("cost", 1.0), ability_available=kwargs.pop("ability_available", True), budget_remaining=kwargs.pop("budget_remaining", 1.0), already_called=kwargs.pop("already_called", False), cache_available=kwargs.pop("cache_available", False))


class DistributionTests(unittest.TestCase):
    def test_exact_three_fields_and_expected_delta(self):
        value = DecisionDistribution.from_mapping({"corrects_error": 0.6, "introduces_error": 0.2, "unchanged_correctness": 0.2})
        self.assertAlmostEqual(value.expected_delta, 0.4)
        with self.assertRaises(DecisionInputError):
            DecisionDistribution.from_mapping({"corrects_error": 0.6, "introduces_error": 0.2, "unchanged_correctness": 0.2, "final_label": 1})

    def test_nan_infinity_bad_sum_and_boolean_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(DecisionInputError):
                DecisionDistribution(bad, 0.2, 0.8)
        with self.assertRaises(DecisionInputError):
            DecisionDistribution(0.6, 0.6, 0.1)
        with self.assertRaises(DecisionInputError):
            DecisionDistribution(True, 0.0, 0.0)

    def test_frozen_distribution_cannot_be_mutated(self):
        value = dist()
        with self.assertRaises(Exception):
            value.corrects_error = 0.9


class RoutingTests(unittest.TestCase):
    def test_lambda_cost_and_deterministic_gates(self):
        self.assertEqual(decide_single(candidate(), 0.3).status, "approved")
        self.assertEqual(decide_single(candidate(), 0.5).status, "cost_threshold")
        harmful = DecisionCandidate("harmful", dist(0.1, 0.7, 0.2), 1.0, True, 1.0)
        self.assertLess(harmful.expected_delta, 0.0)
        self.assertEqual(decide_single(harmful, 0.0).status, "cost_threshold")
        self.assertEqual(decide_single(candidate(budget_remaining=0.5), 0.0).status, "budget_insufficient")
        self.assertEqual(decide_single(candidate(ability_available=False), 0.0).status, "ability_unavailable")
        self.assertEqual(decide_single(candidate(already_called=True), 0.0).status, "already_called")
        self.assertEqual(decide_single(candidate(cache_available=True, budget_remaining=0.0, ability_available=False), 99.0).status, "cache_hit")

    def test_fixed_quota_is_batch_rank_with_stable_equal_ties(self):
        values = [candidate("b"), candidate("a"), candidate("c", ability_available=False)]
        selected = rank_fixed_quota(values, 2).selected_new_calls
        self.assertEqual([item.case_id for item in selected], ["a", "b"])
        self.assertEqual([item.rank for item in selected], [1, 2])
        self.assertTrue(all(abs(item.expected_delta - 0.4) < 1e-12 for item in selected))
        self.assertAlmostEqual(sum(item.expected_delta for item in selected), 0.8)
        with self.assertRaises(DecisionInputError):
            rank_fixed_quota([candidate("a"), candidate("a")], 1)

    def test_cache_is_separate_from_new_call_quota_and_harmful_cache_stops(self):
        cache = candidate("cache", cache_available=True)
        plan = rank_fixed_quota([cache, candidate("new")], 1)
        self.assertEqual([item.case_id for item in plan.selected_new_calls], ["new"])
        self.assertEqual([item.case_id for item in plan.cache_reuses], ["cache"])
        harmful = DecisionCandidate("bad-cache", dist(0.1, 0.7, 0.2), 0.0, True, 0.0, cache_available=True)
        self.assertEqual(decide_single(harmful, 0.0).status, "cache_harm_stop")

    def test_model_failure_falls_back_to_stop_without_label(self):
        route = route_model_output(candidate(), {"corrects_error": 0.8}, lambda_cost=0.0)
        self.assertEqual(route.status, "model_failure")
        self.assertEqual(route.action, "stop")


class LazyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.state = PreAcquisitionState.from_mapping({
            "first_probability": 0.5,
            "first_confidence": 0.5,
            "visible_prefix_length": 4,
            "quality_state": "ok",
            "ability_available": True,
            "budget_remaining": 1.0,
        })
        object.__setattr__(self.state, "cache_key", "state-input-v1")
        self.calls = []
        self.shared_cache = {}
        self.runner = LazySecondExpert(lambda case_id: self.calls.append(case_id) or {"evidence": 0.7}, cache=self.shared_cache)
        self.output = {"corrects_error": 0.8, "introduces_error": 0.1, "unchanged_correctness": 0.1}

    def test_adapter_executes_and_cache_reuses_existing_lazy_runner(self):
        first = execute_with_lazy_runner(candidate("a"), self.output, lambda_cost=0.0, lazy_runner=self.runner, pre_acquisition_state=self.state)
        cached = execute_with_lazy_runner(candidate("a", cache_available=True, budget_remaining=0.0, ability_available=False), self.output, lambda_cost=10.0, lazy_runner=self.runner, pre_acquisition_state=self.state, trusted_cache=TrustedCacheHandle("a", "a", "state-input-v1", self.shared_cache))
        self.assertEqual(first.route.status, "approved")
        self.assertEqual(cached.route.status, "cache_hit")
        self.assertEqual(self.runner.actual_call_count, 1)
        self.assertEqual(self.runner.cache_hit_count, 0)
        self.assertEqual(self.calls, ["a"])

    def test_forged_cache_flag_without_verified_hit_never_calls_provider(self):
        fake_cache = LazySecondExpert(lambda case_id: self.calls.append(case_id) or {"evidence": 0.7}, cache={})
        result = execute_with_lazy_runner(
            candidate("forged", cache_available=True, budget_remaining=0.0, ability_available=False),
            self.output,
            lambda_cost=0.0,
            lazy_runner=fake_cache,
            pre_acquisition_state=self.state,
            trusted_cache=TrustedCacheHandle("forged", "forged", "state-input-v1", {}),
        )
        self.assertEqual(result.route.status, "cache_miss_no_execution")
        self.assertEqual(fake_cache.actual_call_count, 0)
        self.assertEqual(self.calls, [])

    def test_forged_cache_flag_without_handle_never_calls_provider(self):
        result = execute_with_lazy_runner(
            candidate("forged-no-handle", cache_available=True, budget_remaining=0.0, ability_available=False),
            self.output,
            lambda_cost=0.0,
            lazy_runner=self.runner,
            pre_acquisition_state=self.state,
        )
        self.assertEqual(result.route.status, "cache_unverified")
        self.assertEqual(self.runner.actual_call_count, 0)
        self.assertEqual(self.calls, [])

    def test_cache_key_mismatch_never_calls_provider(self):
        cache = {"forged-key": {"evidence": 0.7}}
        runner = LazySecondExpert(lambda case_id: self.calls.append(case_id) or {"evidence": 0.7}, cache=cache)
        result = execute_with_lazy_runner(
            candidate("same-case", cache_available=True, budget_remaining=0.0, ability_available=False),
            self.output,
            lambda_cost=0.0,
            lazy_runner=runner,
            pre_acquisition_state=self.state,
            trusted_cache=TrustedCacheHandle("same-case", "forged-key", "different-state-input", cache),
        )
        self.assertEqual(result.route.status, "cache_miss_no_execution")
        self.assertEqual(runner.actual_call_count, 0)
        self.assertEqual(self.calls, [])

    def test_provider_failure_is_recorded_as_failure_not_no_change(self):
        failing = LazySecondExpert(lambda _: (_ for _ in ()).throw(TimeoutError("synthetic")))
        result = execute_with_lazy_runner(candidate("fail"), self.output, lambda_cost=0.0, lazy_runner=failing, pre_acquisition_state=self.state)
        self.assertEqual(result.route.status, "execution_failure")
        self.assertEqual(result.failure_type, "TimeoutError")
        self.assertIsNone(result.output)
        self.assertEqual(failing.actual_call_count, 1)

    def test_model_output_cannot_override_final_classification(self):
        result = execute_with_lazy_runner(candidate("label"), {**self.output, "final_classification": 1}, lambda_cost=0.0, lazy_runner=self.runner, pre_acquisition_state=self.state)
        self.assertEqual(result.route.status, "model_failure")
        self.assertEqual(self.runner.actual_call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
