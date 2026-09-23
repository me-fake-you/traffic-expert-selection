# TLS and QUIC anomaly notes

## Handshake completeness

Missing or incomplete TLS metadata can result from capture position,
truncation, resumed sessions, QUIC visibility, or parser limitations. A
protocol specialist should abstain when the metadata is insufficient.

## Fingerprints and certificates

Cipher suites, JA3 or JA4 fingerprints, certificate validity, SNI visibility,
and handshake structure provide context. Their rarity is environment-specific
and cannot independently establish maliciousness or malware family.

## Reliability

Protocol explanations must retain the original TLS reliability and uncertainty.
Knowledge retrieval cannot repair missing metadata or override an OOD warning.
