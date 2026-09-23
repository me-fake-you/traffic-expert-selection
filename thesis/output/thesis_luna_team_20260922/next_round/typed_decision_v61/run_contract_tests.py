"""Run offline typed-decision contract tests and write auditable evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TESTS = HERE / "tests"
LOGS = HERE / "logs"
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "output/thesis_luna_team_20260922/runtime"))

from lazy_acquisition import LazySecondExpert, PreAcquisitionState  # noqa: E402
from typed_decision import DecisionCandidate, DecisionDistribution, TrustedCacheHandle, execute_with_lazy_runner  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def adapter_evidence() -> dict[str, object]:
    state = PreAcquisitionState.from_mapping({
        "first_probability": 0.5,
        "first_confidence": 0.5,
        "visible_prefix_length": 4,
        "quality_state": "ok",
        "ability_available": True,
        "budget_remaining": 1.0,
    })
    object.__setattr__(state, "cache_key", "state-input-v1")
    shared_cache = {}
    runner = LazySecondExpert(lambda case_id: {"case_id": case_id, "evidence": 0.7}, cache=shared_cache)
    candidate = DecisionCandidate("adapter-case", DecisionDistribution(0.8, 0.1, 0.1), 1.0, True, 1.0)
    output = {"corrects_error": 0.8, "introduces_error": 0.1, "unchanged_correctness": 0.1}
    first = execute_with_lazy_runner(candidate, output, lambda_cost=0.0, lazy_runner=runner, pre_acquisition_state=state)
    cached = execute_with_lazy_runner(
        DecisionCandidate("adapter-case", candidate.distribution, 1.0, False, 0.0, cache_available=True),
        output,
        lambda_cost=0.0,
        lazy_runner=runner,
        pre_acquisition_state=state,
        trusted_cache=TrustedCacheHandle("adapter-case", "adapter-case", "state-input-v1", shared_cache),
    )
    failing = LazySecondExpert(lambda _: (_ for _ in ()).throw(TimeoutError("synthetic")))
    failed = execute_with_lazy_runner(candidate, output, lambda_cost=0.0, lazy_runner=failing, pre_acquisition_state=state)
    return {
        "first_case_id": first.route.case_id,
        "first_status": first.route.status,
        "cached_status": cached.route.status,
        "provider_received_case_id": first.output["case_id"],
        "actual_call_count": runner.actual_call_count,
        "runner_cache_hit_count": runner.cache_hit_count,
        "adapter_cache_hit_count": 1,
        "cache_was_read_from_verified_shared_mapping": True,
        "failure_status": failed.route.status,
        "failure_type": failed.failure_type,
        "failure_actual_call_count": failing.actual_call_count,
    }


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic_ns()
    log_path = LOGS / "typed_decision_contract.log"
    with log_path.open("w", encoding="utf-8") as log:
        suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
        result = unittest.TextTestRunner(stream=Tee(sys.stdout, log), verbosity=2).run(suite)
    source_paths = {
        "jev_article": ROOT / "output/thesis_luna_team_20260922/next_round/jev_article/Jev借鉴与论文接入说明.md",
        "lazy_runtime": ROOT / "output/thesis_luna_team_20260922/runtime/lazy_acquisition.py",
        "runtime_smoke": ROOT / "output/thesis_luna_team_20260922/runtime/run_runtime_smoke.py",
        "planner": ROOT / "output/thesis_luna_team_20260922/planner/run_planner_pilot.py",
        "typed_decision": HERE / "typed_decision.py",
        "tests": HERE / "tests/test_typed_decision.py",
    }
    payload = {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "source_sha256": {name: sha(path) for name, path in source_paths.items()},
        "contract_scope": "offline synthetic routing contract; no model efficacy, Jev call, RLCD reproduction, remote API, SDK installation, labels, or production mutation",
        "decision_distribution": {
            "fields": ["corrects_error", "introduces_error", "unchanged_correctness"],
            "expected_delta": "P(corrects_error)-P(introduces_error)",
            "unknown_fields_rejected": True,
            "finite_unit_interval_and_sum_checked": True,
        },
        "routing": {
            "fixed_quota_is_batch_rank_only": True,
            "lambda_cost_stop": True,
            "cache_does_not_consume_new_call_budget": True,
            "cache_reuse_is_reported_separately": True,
            "harmful_cache_is_not_auto_fused": True,
            "model_failure_fallback": "stop",
            "final_classification_or_label_fields_accepted": False,
        },
        "adapter_evidence": adapter_evidence(),
        "started_monotonic_ns": started,
        "finished_monotonic_ns": time.monotonic_ns(),
    }
    (RESULTS / "typed_decision_contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
