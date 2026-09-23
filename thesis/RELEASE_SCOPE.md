# Release scope

## Included

- Dependency-closed source modules for the thesis runtime and selected experiment implementations.
- Selected unit tests, two synthetic flow fixtures, field/feature policies and locked experiment configurations.
- Screened aggregate CSV/JSON files: metrics, contrasts, coverage, timing, failure and protocol summaries. Positive, zero, negative and incomplete results are retained on the same inclusion rules.
- A source-to-release SHA-256 ledger and searchable experiment-family index.
- Historical protocol notes. `acceptance`, `submission`, `blindreview`, `PASS`, or similar labels inside old filenames/reports describe local engineering workflow states, **not conference acceptance, completed submission or an independent scientific validation**.

## Not included

- The master's thesis PDF, LaTeX, chapter text or private manuscript editing history.
- Raw PCAP/flows/payloads, endpoint lists, row-level labels/predictions, sample identifiers, fitted checkpoints and cached feature matrices.
- API keys, actual prompts/replies, private browser/ChatGPT records, individual human-review responses or author submission records.
- Vendor libraries, downloaded research repositories or third-party model binaries. Dataset names and public capture-relative group labels in aggregate tables are provenance, not a redistribution of the underlying captures.
- Unfinished later experiments for which only preparation code exists. No result is manufactured to fill a missing path.

Some summaries contain hashes or relative filenames of withheld inputs. These permit local provenance comparison; they do **not** imply those inputs are publicly accessible. A few aggregate human-pilot files retain total counts and file checksums, but no individual responses or identities.

## Transformations

The original research workspace is unchanged. Public source paths are made relative or parameterized. Three synthetic-code/fixture files replace private-range example addresses with documentation-range examples; this changes no measured experimental result. A small offline demonstration CLI is added instead of exporting the original workspace-wide CLI. The manifest records released bytes separately from source bytes.

For historical scripts, local paths, optional dependencies, model inputs and protocol roles still need explicit setup. Read the runner before executing it. LLM-related runners are archived for inspection: running them requires a separate account, credential and permission to send the selected data; this release does not authorize such use or make any API calls.

## License

The author has authorized MIT publication of owned code and software documentation. This does not certify ownership of every upstream idea or relicense external data. No third-party source tree is vendored here. Python dependencies are obtained separately under their respective licenses. Scientific figures and the existing short-paper manuscript retain author rights; full-thesis rights are not granted by this release.

## Evaluation boundaries

Historical thesis protocols, later development extensions and the short-paper controlled study remain distinct. Check the source, split, group, window, label task, and denominator before comparing any values. Prefix quality on supported requests is not all-request recall. Repeated parses of the same captures are not independent datasets. Control-plane replay, local cached execution, and live end-to-end latency are different measurements. No new scientific experiments were run to prepare this release.
