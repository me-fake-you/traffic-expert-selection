# MAD-ETD W71 Learned TLS Evidence Protocol

## Scope

This is a capture-group-isolated, dataset-specific experiment on official CIRA-CIC-DoHBrw-2020 benign DoH versus malicious DoH PCAPs.

- General default runtime: `runtime_safe_v3_0`.
- Any accepted profile is default-off and DoHBrw-only.
- Records-only TCN is the sole promotion candidate; handshake TCN is diagnostic.
- FusionAgent remains the only final verdict owner.
- LLM, RAG, Memory, HITL, Critic and Reflection remain advisory-only.

## Current protocol status

- `not_promoted_learned_tls_w71`.
- Fake metric count: `0`.
- Locked-test selective Macro-F1: records-only TCN `0.990058472588121` vs safe HGB `0.9998058309599143`.
- Failed promotion gates: `records_only_beats_hgb_by_0_03, bootstrap_ci_lower_above_zero, system_macro_f1_improves_0_02, audit_completion_is_one, frozen_artifacts_unchanged`.
- This document does not claim a generic TLS malware detector.
