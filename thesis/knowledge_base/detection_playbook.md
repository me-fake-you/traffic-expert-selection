# Detection playbook

## Multi-view interpretation

Encrypted traffic should be interpreted across independent statistical,
temporal, and protocol views. A single unusual feature can have benign causes,
so agreement, reliability, missing evidence, and conflicts all matter.

## Beacon-like temporal behavior

Repeated connections or packets with regular inter-arrival timing, limited
timing variance, and stable direction patterns can be consistent with
command-and-control beaconing. Scheduled health checks, telemetry, and keepalive
traffic are common benign alternatives. Temporal patterns support a possible
interpretation; they do not prove malicious intent.

## Outbound and burst behavior

A high outbound byte ratio combined with unusual bursts can be consistent with
data staging or exfiltration. Backups, synchronization, media uploads, and
software distribution can create similar statistics. Host and time-window
correlation is required before operational escalation.
