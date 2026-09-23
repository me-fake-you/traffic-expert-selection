# Nemotron Ultra Planner Supplement Protocol

This is a feasibility supplement to the frozen 6,000-sample Rule/fallback
protocol. It does not replace the primary safety baseline and does not claim
that the LLM improves classification.

## Frozen configuration

- Model: `nvidia/nemotron-3-ultra-550b-a55b`
- Temperature: `0`
- Response format: `json_object`
- Thinking: disabled with `chat_template_kwargs.enable_thinking=false`
- Main subset: USTC test 200, CESNET 150, CipherSpectrum development 150
- Stability subset: 100 samples repeated three times
- CipherSpectrum locked_test: forbidden
- Pricing: unconfigured/free-trial; no fabricated cost estimate
- Detector, OOD v1/v2, RAG, Reporter and Fusion: frozen

The two-sample online smoke test passed with five real API decisions:

- valid plan rate: 1.0
- fallback rate: 0
- illegal verdict execution: 0
- blocked-field violation: 0
- Fusion ownership violation: 0
- OOD override: 0
- audit completion: 1.0

Smoke artifact:

`data/runs/future_framework/llm_planner_v1/smoke_nemotron_ultra_profiled/smoke_report.json`

## Long-running execution

The full supplementary run is resumable and writes checkpoints under:

`data/runs/future_framework/llm_planner_v1/nemotron_ultra_500/`

Observed smoke latency is roughly 115 seconds per sample, so the free endpoint
may require 20–30 hours for the main run plus stability repetitions. Partial
files and checkpoints are not reported as completed experiment metrics.

The first long-running attempt stopped after 350 completed samples because an
NVIDIA proposal requested more Agent steps than the remaining runtime budget.
The typed `ExecutionPlan` correctly rejected the mismatch, but the exception
occurred before `PlanPolicyGuard` could sanitize the proposal. The boundary was
fixed without relaxing the schema: proposal cost remains explicit, while
`PlanPolicyGuard` removes over-budget steps using the actual runtime budget.
The original traceback and a structured interruption report are preserved, and
the experiment resumes from the existing 350-row checkpoint.
