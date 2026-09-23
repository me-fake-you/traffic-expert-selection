# Checks actually run for this release

Date: 2026-09-23. Existing Windows environment: Python 3.12.3, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.1, pydantic 2.13.4, pytest 7.4.4. No packages or system components were installed. Tests ran in a disposable copy, with outbound socket connections blocked and service API credentials removed from that process.

| Check | Observed result | Scope |
|---|---|---|
| Shipped thesis tests | **163 passed, 2 skipped; 35 warnings** | Software behavior and small synthetic fixtures, not new scientific training |
| Offline CLI example | 2 synthetic records completed; 18 and 20 audit events | Rule-based demo; no measured research accuracy |
| Copied artifact checksums | 1,291 source-linked files checked against release hashes | Source and released hashes distinguished |
| Prefix saved arithmetic | 64 checks across 8 configurations | Coverage, confusion-count F1, all-request recall and support counts |
| Source parsing | 194 Python files parsed | Syntax, not proof every historical runner has its inputs |
| Existing short-paper component | 9 tests passed; 127 aggregate rows matched formulas | Separate earlier component; no scores changed |
| Showcase | Catalog loaded; topic search and record-type filtering, including empty state, worked; original architecture loaded | Local browser check; no console errors in the observed session |
| Release scope | No new thesis PDF/LaTeX, raw traffic, checkpoints or private responses included | Whitelist and pattern checks plus targeted inspection, not a formal privacy certification |

## Two unavailable checks remain explicit

The first test attempt produced 160 passes and two failures from absent private inputs. These have **not** been converted into successful research results. The public test configuration now skips them when their named inputs are missing, and three tests were added for the public demo and overwrite protection.

- `test_performance_manifest_freezes_disjoint_groups_and_strong_gate` requires `data/runs/mad_etd_hybrid_multiagent_evidence_v1/per_mode_predictions.npz`, a withheld row-level prediction matrix.
- `test_v20_runtime_hash_and_safety_invariants_are_unchanged` requires the full historical completion bundle, including private source/artifact manifests and a bootstrap artifact. A historical `acceptance_report.json` alone cannot make that reconstruction test pass.

The original test assertions are retained; skips depend on input presence. Skipped tests are not counted as passed. Warnings were retained in the local validation logs; this is not a warning-free or cross-platform certification.

Four small neural-component tests use the optional PyTorch dependency already available in the checked environment. They are explicitly skipped on installations without PyTorch; that difference must be included when reporting test counts. No package installation is performed by the test suite.

## Provenance and privacy checks

Copied Python files parse; selected aggregate CSV/JSON files were screened for credentials, non-example endpoint literals, raw/row-level fields and private-author paths. Follow-up checks found only file-checksum references in aggregate human-pilot summaries and a count named `valid_responses`; neither contains actual personal responses. Synthetic private-range addresses were changed to documentation-range addresses. An API runner's local secret-file fallback was removed, and private input paths were parameterized. No result value was edited.

Full scientific reproduction still needs the withheld legal inputs, original role assignments and corresponding fitted artifacts. This release does not re-certify truth labels, independence of every historical study, robustness outside the stated protocols, or conference suitability.
