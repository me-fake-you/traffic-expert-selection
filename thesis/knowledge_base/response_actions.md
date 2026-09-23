# Response actions

## Increase monitoring

Enhanced monitoring is appropriate when evidence is suspicious but not
decisive. Collect additional sessions, endpoint telemetry, DNS context, and
time-window correlations while preserving the original uncertainty.

## Analyst review

Human review is useful for conflicting evidence, severe distribution shift,
missing protocol context, or operationally important assets. Review is an
escalation for context, not proof of maliciousness.

## Retain and preserve

Retaining the flow record, audit trail, and relevant packet capture supports
reproducibility and later forensic analysis. Preservation should maintain
provenance without exposing blocked fields to detector Agents.

## Isolation and blocking

Isolation or blocking should follow existing deterministic response policy and
organizational authorization. Knowledge retrieval may explain an action that
the Reporter already selected, but it cannot add, remove, or upgrade actions.
