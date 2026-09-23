# MAD-ETD v2.8 Field Contract

MAD-ETD v2.8 uses a versioned, fail-closed field contract before every
detector call.

## Roles

- `DETECTION_ALLOWED`: explicitly consumed by an accepted detector.
- `CONTEXT_ONLY`: retained for audit/reporting but removed from
  `DetectorInput`.
- `LABEL_ONLY`: supervision, family, application, or label-proxy metadata.
- `PROVENANCE`: source and capture metadata.
- `BLOCKED`: identifiers, disabled payload features, or direct label proxies.
- `UNKNOWN`: a populated field absent from the frozen contract.

Empty `None`, empty string, empty list, and empty dictionary values are
recorded separately as `ignored_empty_fields`. They are not treated as
populated unknown fields and never enter `DetectorInput`.

## Detection inputs

The accepted contract permits only:

- the eight frozen v1 flow-statistics inputs;
- packet lengths, directions, IATs, bursts, original count, and truncation;
- controlled TLS versions, record lengths, count encodings, ALPN,
  certificate validity, and handshake completeness.

Exact cipher values, transport/protocol names, SNI, IP addresses, ports,
application/family metadata, provenance, disabled payload tokens, and direct
threat-intelligence fingerprint matches do not enter detectors.

The machine-readable contract is generated as
`data/runs/mad_etd_v2_8/field_contract.json`. Its hash is frozen into the
v2.8 selection manifest before evaluation.
