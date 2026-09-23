# Evidence map

These are saved local results, not numbers copied from competing papers. All files below retain positive, zero and adverse cases within their recorded scope.

| Manuscript question | Recorded outputs | Historical execution source | Scope |
|---|---|---|---|
| What changes when the expert-selection policy changes? | `results/direct_controls/pooled.csv`, `fold_metrics.csv`, `conditional_intervals.csv` | `archive/original/direct_controls/delta_analysis.py` | Original 40,000-segment, frozen HGB-expert population; nine directly replayable policies |
| Can selection data distinguish the candidates? | `results/candidate_analysis/support.csv`, `mode_decomposition.csv`, `mode_members.csv` | `archive/original/candidate_analysis/candidate_information.py`; direct-controls script | Original 16 candidates, exact prediction vectors; full and concentrated arbiter fitting |
| Does more candidate coverage reliably improve selected-policy errors? | `results/coverage/selection_choices.csv`, `paired_contrasts.csv` | `archive/original/coverage/coverage_experiment.py` | Matched quotas, dependent subsets; no universal monotonicity claim |
| Is the observed selection behavior caused by the recall constraint? | `results/constraint/selection_diagnostics.csv`, `contrasts.csv` | `archive/original/constraint/constraint_ablation.py` | Zero choice changes across 90 dependent paired cases |
| Does the pattern survive one Stats-backend change? | `results/backend/pooled.csv`, `mode_decomposition.csv`, `selection_transfer.csv` | `archive/original/backend/backend_extratrees.py` | Same-source ExtraTrees sensitivity; not external or deep-model replication |
| Are prohibited input fields excluded on this path? | `results/input_checks/semantic_checks.csv`, `timing_stress.csv` | `archive/original/input_checks/field_and_timing.py` | 120,000 paired mutations and fixed timing stress; not proof that all acquisition shortcuts vanish |
| What was actually processed from raw input? | `results/raw_input/verified_results.csv`, `population_metrics.csv` | `archive/original/raw_input/raw_all.py` | 625,523 common eligible segments from 24 captures; one timing run; distinct from the 40k population |
| What happened under frozen external transfer? | `results/external_transfer/pooled.csv`, `bundle_range.csv` | `archive/original/external_transfer/frozen_transfer.py` | 1,159,418 matched IoT-23 author flows, seven dependent captures, five bundles × 11 policies; source and unit shift |

The portable replay entry point validates the first two rows' selected quantities against the saved originals. Archive availability is not evidence that all these experiments were rerun during release preparation. Full row-level predictions, original logs, and upstream fit records remain in the private research workspace.
