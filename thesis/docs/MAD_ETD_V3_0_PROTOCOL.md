# MAD-ETD v3.0 Fresh-Holdout Capability Protocol

MAD-ETD v3.0 evaluates backend capability-aware routing on 8,000 samples that
have zero sample-ID overlap with the v2.8 and v2.9 conformance manifests.

The candidate adds `DetectorCapabilityProfile` decisions with the statuses
`available`, `missing_view`, and `unsupported_schema`. The accepted rule TLS
backend may consume only protocol version, certificate validity, and handshake
completeness. TLS record sequences and handshake-count schemas require a
learned TLS artifact and are rejected when that artifact is absent.

The protocol compares `runtime_safe_v2_7` with `runtime_safe_v3_0`, performs
strict/legacy and TLS metadata-mutation replays, and runs six shadow modes.
Unsupported-view removal must preserve verdict and OOD state, increase no
confidence, decrease no uncertainty, and never promote an abstaining result to
a binary verdict.

```powershell
python -m mad_etd.cli build-v3-0-fresh-manifest
python -m mad_etd.cli run-v3-0-capability-acceptance
python -m mad_etd.cli finalize-v3-0 --test-count <pytest-count>
```

No Detector, OOD, Fusion, RAG, or model artifact is trained or modified.
Annotated-TLS remains unlabeled OOD data and is not used as a supervised TLS
classification dataset.
