# MAD-ETD Architecture Governance v6.0

- status: `passed`
- default runtime: `runtime_safe_v3_0`
- runtime_safe_v3_0 remains default: `True`
- promoted runtime created: `False`
- fake metric count: `0`

## Architecture

```text
Raw Flow / Optional Sequence Input
    ↓
FieldAudit / AlignmentAudit
    ↓
Safe DetectorInput Builder
    ↓
Detector Registry
    ↓
Capability Profile
    ↓
Planner / Coordinator
    ↓
PolicyGuard
    ↓
Evidence Agents
    ↓
AgentEvidence v2 Bus
    ↓
FusionAgent
    ↓
OOD / Reliability Gate
    ↓
Reporter / AuditLogger
    ↓
PromotionGate Manager
    ↓
Negative Result Ledger / Claim Ledger
    ↓
Runtime Profile Manager
```

## Governance contribution

MAD-ETD is framed as a safety-constrained, capability-aware, evidence-governed, auditable, reproducibility-aware, negative-result-aware agentic encrypted traffic detection framework.

The only accepted performance-class positive result remains the NF-IoT dataset-specific optional calibration/runtime. It does not replace the default runtime.
