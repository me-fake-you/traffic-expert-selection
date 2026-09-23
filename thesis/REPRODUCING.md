# Reproduction levels and input requirements

## 1. Offline public checks

Use Python 3.11+ in a dedicated environment. From `thesis/`, install this package with `python -m pip install -e ".[dev]"`. The core dependencies are in `pyproject.toml`. Optional historical deep-model work additionally needs the `v2` extras and its own model/data preparation; these are not required for the demo.

```sh
python -m mad_etd.cli demo --output runs/synthetic_demo_001
python scripts/verify_release.py
python -m pytest tests -q
```

The small public CLI exports `demo` only, not every command from the original development workspace. The demo's output is explicitly synthetic and is not a table of research performance. Tests use temporary data; some build tiny synthetic model fixtures. Network access is not required. If running in a credential-bearing environment, remove API credentials from the test process. The release validation runs with outbound connections blocked.

Four neural-component checks additionally use optional PyTorch (`v2` extra); these skip explicitly when it is absent. Two integration checks need withheld private inputs and also skip explicitly. Report skips separately from passes.

`verify_release.py` checks copied artifact hashes, parses code/configuration files, and recomputes the prefix table's coverage, confusion-count Macro-F1 and all-request malicious recall. This is a saved-arithmetic check, not a rerun of the original PCAP pipeline or verification of original truth labels.

## 2. Re-running scientific experiments

| Experiment family | Entry point / location | Inputs not bundled |
|---|---|---|
| Feature/field tests | `src/mad_etd/feature_forensics_v1.py`, `feature_policy_effectiveness_v1.py` | Actual flow/feature inputs and fitted experts for performance comparisons |
| USTC factorial, cascade, N-BaIoT device holdout | Functions in `src/mad_etd/thesis_reviewer_closure_v24.py` | Exact licensed data versions, prepared feature caches, split manifests and fitted artifacts |
| Matched perturbation / rejection | `run_matched_perturbation_replay_v34`, `run_matched_rejection_replay_v34` in `src/mad_etd/thesis_method_closure_v34.py` | Frozen trace/evidence inputs and the original matching definitions |
| Later gain/risk development | `output/thesis_luna_team_20260922/next_round/` runners and locked configurations | Original grouped/OOF matrices and correctly isolated expert outputs |
| Frozen outer confirmation | `outer_confirmation_v62/run_outer_confirmation_v62.py` under that parent | The already fitted gain models, frozen selection records and outer-role inputs |
| Matched-prefix local replay | `prefix_runtime_v62/{train_prefix_models_v62,run_prefix_runtime_v62}.py` | Authorized PCAPs, parser availability, training-only groups, matched-prefix models |
| LLM development pilot | `llm_gain_v61/run_pilot.py` | Explicit external-service authorization and API account; private prompt/reply inputs are withheld |

Historical path defaults remain provenance clues and may need adaptation. The source manifest records all changes made for distribution. This package does not promise that every historical runner can execute from a fresh checkout with no other inputs. Missing prerequisites are not replaced by synthetic research results. Do not recreate OOF inputs by fitting on outer-role labels, choose thresholds by published test scores, or reuse correctness targets from a different expert.

## 3. Understanding saved evidence

Use `EXPERIMENT_INDEX.csv` to locate a family, read its protocol/configuration, then the aggregate result and scope statements. `SOURCE_MANIFEST.csv` preserves source and released SHA-256 values. Relative paths to withheld data may remain in reports; they document original inputs rather than functioning download links.

Archive acceptance/status fields are historical local checks, not a conference decision. Quoted literature metrics are not local reproductions. Directory counts include nested reports and repeated conditions; they are not independent experimental replications.

## Data and safety

Obtain data from its original provider under the appropriate terms; do not infer redistribution rights from this repository's MIT license. Never execute payloads extracted from traffic. Keep raw captures, private endpoints, row-level records, model files, API keys and personal review records out of public commits. The provided aggregate tables intentionally omit them. See `RELEASE_SCOPE.md`.
