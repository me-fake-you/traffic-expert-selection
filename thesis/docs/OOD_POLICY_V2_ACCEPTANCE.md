# OOD accuracy-coverage policy v2 acceptance

## Experiment

OOD policy v2 compares fixed rule combiners and an L2 logistic-regression
combiner using USTC-only nested meta-policy validation.

For each outer application/family fold, every inner OOD Gate excludes both the
outer and inner domain pairs. Candidate selection uses the remaining nine
inner held-domain results. CESNET is unavailable to the optimizer and is used
only after `selected_policy.json` has been frozen.

The policy is monotonic reject-only:

- it cannot flip benign to malicious or malicious to benign;
- warning can change a binary verdict only to suspicious;
- hard shift can change a non-unknown verdict only to unknown;
- suspicious and unknown can never be restored to a binary verdict.

RAG remains disabled throughout.

## Important methodological boundary

The nested component covers OOD Gate fitting and meta-policy selection. Base
detection outputs are frozen cross-fitted OOD-v1 held-domain predictions; OOD
policy v2 does not retrain or select detector models.

For an inner domain, its frozen detector excludes that inner domain but may
have trained on the current outer domain. Therefore this is a nested
meta-policy experiment, not a full nested retraining of the entire detector
stack. This limitation must be stated in the paper.

## Frozen USTC result

The final selected policy is:

- Combiner: `min(stats_shift, temporal_shift)`
- Temporal missing behavior: deterministic fallback to Stats
- Warning percentile: `0.99`
- Hard percentile: `0.999`
- Warning threshold: `0.9990737954`
- Hard threshold: `0.9990969505`
- Policy SHA-256:
  `d6ce019fb72e5a00cbdfaacf81230ef5c8fd16a44558c00e1c0af7bf11e81c20`

Nested outer pooled metrics:

| Metric | Result | Target | Status |
|---|---:|---:|---|
| Macro-F1 | 0.976950 | ≥ 0.95 | pass |
| Coverage | 0.770447 | ≥ 0.65 | pass |
| Minimum class coverage | 0.770292 | ≥ 0.50 | pass |
| Selective error | 0.023027 | ≤ 0.05 | pass |
| Benign false-positive rate | 0.024609 | ≤ 0.05 | pass |
| Nested OOD AUROC | 0.507874 | ≥ 0.97 | fail |

Compared with OOD v1 Hybrid, coverage improves from `0.140641` to `0.770447`
while selective error remains below 5%. However, held application/family
membership is not reliably separable as feature-distribution OOD under the
current Isolation Forest representation.

Outer folds selected `min` six times, `temporal_only` twice, and
`logistic_regression` twice. The weak and unstable inner OOD AUROC explains
why no combiner achieves the discrimination target.

## Frozen CESNET external result

The frozen policy was applied once to 499,997 unlabeled CESNET flows:

| Metric | Result | Target | Status |
|---|---:|---:|---|
| External OOD AUROC | 0.939533 | ≥ 0.97 | fail |
| Malicious rate | 0.280376 | ≤ 0.10 | fail |
| Alert rate | 0.841101 | ≤ 0.40 | fail |
| Unknown rate | 0.156577 | ≥ 0.50 | fail |

Supervised metrics remain unavailable because CESNET has no binary ground
truth. The failed external result was recorded without changing candidates,
thresholds, or policy.

This establishes the central trade-off:

- OOD v1 is externally safe but rejects too much USTC traffic.
- OOD v2 restores USTC coverage but loses external safety.
- A single percentile-based Isolation Forest rejection score is insufficient
  to satisfy both objectives.

## Reproducibility and audit

- Full test suite: `118 passed`
- Nested Gate pairs: 90
- USTC audit-chain completion: 100%
- Blocked-field violations: 0
- External data used for fitting or thresholds: false
- Logistic target: ID versus pseudo-OOD membership, never benign/malicious
- RAG enabled: false
- Frozen policy checksum verified before and after CESNET validation

Primary artifacts:

- `data/runs/ustc_tfc2016/ood_policy_v2/experiment_manifest.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/candidate_registry.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/inner_cv_results.csv`
- `data/runs/ustc_tfc2016/ood_policy_v2/selected_policy.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/outer_fold_metrics.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/outer_predictions.csv`
- `data/runs/ustc_tfc2016/ood_policy_v2/coverage_risk_curve.csv`
- `data/runs/ustc_tfc2016/ood_policy_v2/ood_roc.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/audit_summary.json`
- `data/runs/ustc_tfc2016/ood_policy_v2/external_validation/cesnet_external_validation.json`

## Research conclusion

Do not tune this policy again against CESNET. The next defensible research
step is to treat USTC unseen-domain robustness and cross-dataset OOD safety as
separate objectives, then develop a representation-level or conformal reject
mechanism using USTC-only validation and a newly reserved development
distribution. CESNET must remain a frozen final external test.
