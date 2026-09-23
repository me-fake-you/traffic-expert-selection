# Malicious intent patterns

## Possible C2 beaconing

Potential beaconing indicators include periodic IAT values, stable packet
directions, repeated small exchanges, and recurring sessions. These indicators
describe behavior that may be compatible with C2; they are not a direct
malicious label and must remain subordinate to detector evidence.

## Possible data exfiltration

Potential exfiltration indicators include sustained outbound dominance,
bursty uploads, unusual transfer timing, and repeated staging behavior.
Legitimate cloud synchronization and backup activity can look similar, so the
recommended interpretation is conditional and uncertainty-aware.

## Intent language

Reports should use phrases such as "may be consistent with" or "possible
interpretation." They must not use retrieved knowledge to claim that an intent
or malware family has been confirmed.
