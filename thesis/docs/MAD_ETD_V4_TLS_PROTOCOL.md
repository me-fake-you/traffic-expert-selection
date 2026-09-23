# MAD-ETD v4 Supervised Learned TLS Protocol

## Scope

The supervised task is limited to benign DoH versus malicious DoH in CIRA-CIC-DoHBrw-2020. Non-DoH captures are unlabeled OOD only.

## Safety contract

- Final verdict ownership remains with FusionAgent.
- IP, port, SNI, JA3, tool, browser, resolver and source filename are excluded from detector input.
- Capture IDs group all flows from one PCAP into one split.
- Quad9 is a resolver holdout and locked test data never selects models, seeds, calibration or thresholds.
- `runtime_safe_v3_0` remains the general default.

## Fixed comparison

1. Rule TLS.
2. Aggregate-feature HistGradientBoosting.
3. Records-only residual TCN.
4. Records plus controlled-handshake residual TCN.

The records-only TCN is the sole promotion candidate. Handshake features cannot rescue a failed records-only candidate.
