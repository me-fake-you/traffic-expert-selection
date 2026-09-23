"""Small offline release entry point, not the historical all-experiments CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MAD-ETD offline synthetic demonstration")
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--input", type=Path, default=Path("examples/sample_flows.jsonl"))
    parser.add_argument("--output", type=Path, required=True,
                        help="A new output directory; existing directories are rejected")
    args = parser.parse_args()
    from .engine import build_default_engine
    from .io import load_flow_records

    flows = load_flow_records(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = build_default_engine(use_nvidia=False, force_rule_coordinator=True,
                                  cache_dir=None, max_workers=1)
    summaries = []
    for i, flow in enumerate(flows):
        report, _ = engine.analyze(flow, audit_path=args.output / f"synthetic_{i}.audit.jsonl")
        (args.output / f"synthetic_{i}.report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8")
        summaries.append(dict(case=i, verdict=report.verdict.value,
                              agents=report.participating_agents,
                              audit_events=report.audit_event_count))
    print(json.dumps(dict(scope="synthetic_rule_demo_not_model_evaluation",
                         network=False, cases=summaries), indent=2))


if __name__ == "__main__":
    main()
