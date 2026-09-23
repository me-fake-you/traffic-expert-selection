"""Run the small lazy-acquisition contract suite and write auditable results."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lazy_acquisition import LazyCase, LazySecondExpert, PreAcquisitionState  # noqa: E402


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


def make_state():
    return PreAcquisitionState.from_mapping(
        {
            "first_probability": 0.42,
            "first_confidence": 0.58,
            "visible_prefix_length": 8,
            "quality_state": {"missing_count": 0},
            "ability_available": True,
            "budget_remaining": 1.0,
        }
    )


def evidence_events():
    calls = []

    def provider(case_id):
        calls.append(case_id)
        return 0.73

    expert = LazySecondExpert(provider)
    case = LazyCase("evidence-cache", lambda _: 0.42, expert)
    case.execute_first()
    first_only_result = case.request_second(make_state(), should_execute=False)
    triggered_result = case.request_second(make_state(), should_execute=True)
    cached_result = case.request_second(make_state(), should_execute=True)
    return {
        "first_only_result": first_only_result,
        "triggered_result": triggered_result,
        "cached_result": cached_result,
        "provider_calls": calls,
        "actual_call_count": expert.actual_call_count,
        "request_count": expert.request_count,
        "cache_hit_count": expert.cache_hit_count,
        "events": [event.as_dict() for event in expert.events],
        "monotonic_non_decreasing": all(
            left <= right
            for left, right in zip(
                [event.monotonic_ns for event in expert.events],
                [event.monotonic_ns for event in expert.events][1:],
            )
        ),
    }


def failure_event_evidence():
    attempts = []

    def flaky(case_id):
        attempts.append(case_id)
        if len(attempts) == 1:
            raise TimeoutError("synthetic timeout")
        return 0.64

    expert = LazySecondExpert(flaky)
    case = LazyCase("evidence-failure", lambda _: 0.5, expert)
    case.execute_first()
    first_failure = None
    try:
        case.request_second(make_state(), should_execute=True)
    except TimeoutError as exc:
        first_failure = type(exc).__name__
    recovered = case.request_second(make_state(), should_execute=True)
    return {
        "first_failure_type": first_failure,
        "recovered_result": recovered,
        "provider_attempts": attempts,
        "actual_call_count": expert.actual_call_count,
        "events": [event.as_dict() for event in expert.events],
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "runtime_smoke.log"
    started = time.monotonic_ns()
    with log_path.open("w", encoding="utf-8") as log:
        runner = unittest.TextTestRunner(stream=Tee(sys.stdout, log), verbosity=2)
        suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
        result = runner.run(suite)
        evidence = evidence_events()
        failure_evidence = failure_event_evidence()
        summary = {
            "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "first_only_never_calls_second": evidence["first_only_result"] is None
            and evidence["provider_calls"] == ["evidence-cache"],
            "cache_reuse_keeps_actual_calls_at_one": evidence["actual_call_count"] == 1
            and evidence["cache_hit_count"] == 1,
            "monotonic_event_clock_non_decreasing": evidence["monotonic_non_decreasing"],
            "event_kinds": [event["kind"] for event in evidence["events"]],
            "failure_event_recorded": [event["kind"] for event in failure_evidence["events"]]
            == ["request", "failure", "request", "execution"],
            "strict_state_and_permission_tests_passed": result.wasSuccessful(),
            "first_failure_blocks_second_until_retry": result.wasSuccessful(),
            "runtime_scope": "cached provider contract only; no end-to-end latency or acceleration claim",
            "started_monotonic_ns": started,
            "finished_monotonic_ns": time.monotonic_ns(),
        }
        (RESULTS / "runtime_smoke_results.json").write_text(
            json.dumps(
                {"summary": summary, "evidence": evidence, "failure_evidence": failure_evidence},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
