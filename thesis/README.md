# MAD-ETD · thesis code and experimental record

[中文说明](README.zh-CN.md) · [Research map](RESEARCH_MAP.md) · [Experiment index](EXPERIMENT_INDEX.csv) · [Reproducing](REPRODUCING.md) · [Release scope](RELEASE_SCOPE.md)

**Haowen Liu · Lei Zhang — Capital Normal University**

This companion releases the master's research implementation and screened, aggregate experimental records. The thesis full text is **not** distributed. The [ICASSP-oriented controlled study](../README.md) remains a separate artifact: its populations, protocols, and metrics are not merged with the thesis record.

## What is here

The research system separates field use, expert execution, evidence admission, fusion ownership, and read-only advice. Its tests ask both whether these boundaries hold and when additional evidence helps detection. The archive includes outcomes that do **not** favor the full system or learned selection.

| Research area | Start with | Evidence |
|---|---|---|
| Field-use isolation and shortcut risk | `src/mad_etd/field_audit.py`, `field_contract.py`, `features.py` | FieldAudit pairing and feature-forensics records |
| Statistical, temporal, and conditional TLS evidence | `detectors.py`, `training.py`, `schemas.py` | Local baselines, specialist and feature-condition comparisons |
| Evidence admission, reliability, fusion and rejection | `evidence_admission.py`, `fusion.py`, `ood.py` | Fusion ablations and matched-coverage / rejection comparisons |
| Budgeted coordination and ownership | `engine.py`, `coordinator.py`, `guard.py` | Routing, branch coverage and control-plane timing |
| Shift and group-based evaluation | `splits.py`, `generalization.py` | Source / device / perturbation protocols, kept separate |
| Read-only advice and human feedback | `reasoning.py`, `knowledge.py`, `memory.py`, `hitl.py` | Bounded pilots, including cancellations and timeouts |
| Gain/risk selection and packet prefixes | `output/thesis_luna_team_20260922/next_round/` | Development, outer confirmation and matched-prefix results |

File locations in this table are relative to `thesis/`; short module names share `src/mad_etd/`. [RESEARCH_MAP.md](RESEARCH_MAP.md) provides specific result anchors and interpretation boundaries.

## Run an offline example

In a separate Python 3.11+ environment, from this directory:

```sh
python -m pip install -e ".[dev]"
python -m mad_etd.cli demo --output runs/synthetic_demo_001
python scripts/verify_release.py
python -m pytest tests -q
```

The two flow records in `examples/sample_flows.jsonl` are **synthetic fixtures**, with documentation-range addresses. The demo uses rule-based experts and a deterministic coordinator; it performs no model training, network request, paid API call, or real traffic evaluation. It refuses to overwrite an existing output directory.

The verification command checks public file hashes and selected saved arithmetic, not the original labels or full fitting chain. Tests may create small synthetic model fixtures in temporary directories. See [VALIDATION.md](VALIDATION.md) for the checks actually run on this release, including unavailable checks.

## Three different reproduction levels

1. **Runnable without research data:** the synthetic demo, shipped unit tests and aggregate checks.
2. **Executable with separately obtained inputs:** selected experiment functions and standalone runners. Obtain the appropriate licensed dataset, reproduce its preprocessing and group roles, and supply the exact model/configuration dependencies first.
3. **Historical evidence only in this distribution:** archived aggregate reports whose private row-level traces, models, or API replies are not bundled. Their presence is not a claim of a fresh independent reproduction.

The public entry point is intentionally smaller than the original workspace CLI. Historical revision suffixes identify provenance; they are not separate novel methods, independent repetitions, or publication-status claims.

## Rights and provenance

Author-owned code and software documentation use the repository's [MIT license](../LICENSE). No third-party dataset, checkpoint, malware binary, full thesis PDF/LaTeX, raw payload, private endpoint or personal review record is bundled. Third-party libraries and datasets retain their original terms. [SOURCE_MANIFEST.csv](SOURCE_MANIFEST.csv) records source and released hashes, including path sanitization and synthetic-fixture changes. [RELEASE_SCOPE.md](RELEASE_SCOPE.md) explains omissions.
