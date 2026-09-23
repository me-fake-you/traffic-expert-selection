# MAD-ETD Thesis Final Experiment Closure v20

This round reuses frozen models, predictions, and paired perturbation records. It performs no detector training, does not reopen the sealed N-BaIoT acceptance set, and leaves `runtime_safe_v3_0` unchanged.

Key results:

- USTC: task-specific evidence policy Macro-F1 0.9491 versus 0.9283 for equal-probability averaging (+2.08 percentage points). The paired-sample bootstrap CI is positive; the application/family-grouped CI is [0.0000, 0.0591] and therefore does not establish strict grouped significance.
- N-BaIoT: a validation-selection diagnostic is positive (+3.67 points versus the best observed ordinary static ensemble), but it is not an independent acceptance claim. The sealed acceptance was not reopened.
- Paired perturbations: in the continuous diagnostic, harmful flips fall from 110/3,200 to 0/3,200 and selective error from 0.0394 to 0.0035, with coverage changing from 1.0000 to 0.8831.
- External context: five citable multi-agent systems have positive numerical references; two comparisons include official code in a local adaptation and three are paper-reported cross-protocol references. Faithful reproduction count remains zero.

All safety violation counters and `fake_metric_count` are zero. No runtime was promoted.
