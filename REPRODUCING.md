# Reproduction levels and exact input contract

## 1. Without research data

From this repository root, install `requirements.txt` in your own environment, then run:

```sh
python -m unittest discover -s tests -v
python scripts/check_results.py
```

The tests check binary metrics, corrected/introduced-error accounting, full-vector equivalence, tie ordering, the strict Fixed inequality, the explicit no-switch sentinel, invalid inputs, and the finite-set decomposition identity. Test arrays are synthetic; they are not experimental results.

The aggregate checker recomputes Macro-F1, malicious recall, and FPR from saved confusion counts. It checks arithmetic consistency, not the correctness of the raw labels or the original fitting isolation.

## 2. With legally held frozen predictions

Use the original candidate matrices produced by the historical candidate-analysis script. Forty `.npz` files are needed: two fitting-support conditions × two arbiters × five folds × selection/evaluation roles. The preserved historical naming is:

```text
{condition}_20260920_{arbiter}_fold{fold}_{role}.npz
condition: row100 | source_concentrated
arbiter: logistic | hgb
fold: 0 .. 4
role: selection | evaluation
```

Required arrays: `label` (binary), `predictions` (samples × 16 binary columns), `thresholds` (16 × 2), `first_probability`, `second_probability`, and `trigger`. `sample_hash` and `group`, when present, are used for alignment and cross-role disjointness checks. They are never copied to the replay output. The original full fitting-chain isolation must still be established from the original run records; these checks do not replace it.

The threshold order is the ascending Cartesian product of `[0.25, 0.5, 0.75, 1.01]`. Column 15 is no-switch and column 5 is 0.5/0.5. The four global candidates are columns `[0, 5, 10, 15]`, selected independently. The reference JSON has keys `"0"` through `"4"`, each with `reliability_temporal` and `reliability_stats` measured on the separate reliability role.

```sh
python -m traffic_selection.replay --matrices /path/to/candidate_matrices --references /path/to/reference_lock.json --output runs/replay_001
```

This reads frozen predictions only, never estimates a model, changes the candidate grid, or recomputes confidence intervals. It completes all selection choices before opening evaluation matrices. Original exposed-data status does not change. Outputs include selected indices, aggregate metrics, exact-vector support, post-hoc decomposition, and input checksums. An existing output directory is refused.

## 3. Historical training and execution code

The source archive preserves the original arbiter training, limited backend replacement, controlled subset analysis, input checks, raw-input execution, and external frozen transfer scripts. It is **not a portable end-to-end runner**: historical data adapters, configuration files, provenance records, upstream OOF/fitting artifacts and model files are not redistributed here. See `archive/README.md`.

Original numerical settings remain in the archived code and manuscript. The portable replay uses NumPy only; reproducing historical fitting requires the historical pinned environment, not arbitrary latest package versions. Do not load untrusted pickle/joblib checkpoints.

## Result interpretation

Macro-F1 is computed from pooled binary confusion counts, not an average of fold F1. `C` and `D` are absolute corrections/introduced errors relative to Temporal, so `C + D` equals switches; `C - D` is net correction and is not a Macro-F1 change. Candidate selections use malicious recall ≥ Temporal's recall (tolerance `1e-12`), then descending Macro-F1, recall, and fewer switches; exact remaining ties use the first ascending pair. No feasible candidate raises an error.

The hindsight gap is selected error minus the best evaluation error among **selection-feasible** candidates. It splits into the gap within the selected full-vector equivalence class and the gap between that class and all feasible candidates. Evaluation labels are used only for this retrospective calculation, never to replace the selected policy.
