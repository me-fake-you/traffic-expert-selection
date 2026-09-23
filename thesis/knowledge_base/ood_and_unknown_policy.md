# OOD and reject policy

## Suspicious versus unknown

Suspicious means directional risk evidence exists, but its strength,
reliability, or agreement is insufficient for a malicious verdict. Unknown
means evidence is too incomplete, unreliable, distribution-shifted, or
conflicting to support either benign or malicious direction.

## Severe distribution shift

Severe model-distribution shift indicates that detector behavior may not be
supported by the training distribution. Reliability discount or abstention is
a safety response. Knowledge text cannot change the measured OOD level,
threshold, coverage, confidence, or uncertainty.

## Unlabeled external evaluation

Accuracy, precision, recall, F1, PR-AUC, Brier score, and calibration error
require ground-truth labels. On unlabeled CESNET traffic these supervised
metrics must be reported as unavailable. Verdict proportions, unknown rate,
coverage, latency, and audit completeness may still be summarized, but they
are not substitutes for classification accuracy.

## Experimental integrity

External test data must not be used to train the retriever, select detector or
OOD thresholds, calibrate models, or rewrite acceptance conclusions.
