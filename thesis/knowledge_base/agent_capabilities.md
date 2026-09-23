# Agent capabilities

## Detection ownership

Specialist detector Agents produce bounded evidence from approved feature
views. FusionAgent alone owns the final verdict, confidence, uncertainty,
severity, and escalation decision. ReporterAgent can describe those results
but must not reclassify a flow.

## Explanation boundary

Knowledge retrieval is an explanatory-only reporting service. Retrieved text
is background context rather than sample evidence. It cannot change field
policy, reliability, OOD status, family attribution, or response actions.

## Audit expectation

Every retrieval records the knowledge-base version, content hash, query topics,
ranked chunks, scores, citations, validation status, and failure mode.
