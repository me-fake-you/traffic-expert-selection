# Release validation

This section records the original controlled-study release. The subsequent, separately scoped thesis-code extension is documented in [thesis/VALIDATION.md](thesis/VALIDATION.md); its checks do not replace or extend the original scientific evaluation.

Checked locally with Python 3.12.14 and NumPy 1.26.4 on 2026-09-23. No fresh model fitting, inference, data download, or evaluation-population change was performed for this release.

| Check | Observed result |
|---|---|
| Synthetic rule tests | 9 passed |
| Aggregate arithmetic | 127 recorded rows matched their confusion-count formulas |
| Portable replay versus historical outputs | All compared values agreed to absolute tolerance `1e-12` |
| Decomposition parity | 20 cases; 380 numeric cells |
| Simple selection parity | 15 choices; 60 numeric cells |
| Pooled direct-control parity | 9 policies; 99 numeric cells |
| Fold direct-control parity | 45 rows; 495 numeric cells |
| Candidate-support parity | 40 rows; 80 numeric cells |
| Archived sources and result snapshots | 39 copied files checked byte-for-byte against their recorded source hashes |
| Page behavior | Population selector displays 9 / 10 / 8 / 55 rows; command copying, figure loading and local links checked |
| Responsive layout | Desktop and 390-pixel mobile width inspected; no page-wide horizontal overflow |

The synthetic endpoint values in the mutation script are documentation examples (`192.0.2.1`, `198.51.100.2`), not observed endpoints. Whitelist staging and pattern checks found no raw-traffic files, checkpoints, private absolute user paths, or credential-format matches in the candidate.

These checks establish the stated implementation/packaging properties, not independent data-label certification, full training-chain replication, a formal privacy audit, or peer-review acceptance. Original settings, negative results, and evaluation limits remain in the manuscript and result files.
