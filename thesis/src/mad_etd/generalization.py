from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Iterator

import numpy as np

from .engine import build_default_engine
from .evaluation import (
    _average_precision,
    _expected_calibration_error,
    _safe_div,
)
from .features import feature_extractor, feature_names
from .fusion import FusionAgent
from .io import iter_flow_records, iter_split_records
from .schemas import (
    AgentEvidence,
    DetectorInput,
    FlowRecord,
    ReliabilityProfile,
    Verdict,
)
from .training import FeatureMatrix, _group_key, train_detector


DOMAIN_HOLDOUT_SCHEMA_VERSION = "1.0"
DOMAIN_HOLDOUT_PAIRS = (
    ("FTP", "Nsis-ay"),
    ("MySQL", "Htbot"),
    ("Weibo", "Tinba"),
    ("SMB", "Shifu"),
    ("Gmail", "Miuref"),
    ("WorldOfWarcraft", "Zeus"),
    ("Outlook", "Neris"),
    ("BitTorrent", "Geodo"),
    ("Skype", "Cridex"),
    ("Facetime", "Virut"),
)
GENERALIZATION_SYSTEMS = ("full", "stats_only", "temporal_only", "no_reject")


def _stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "big"
    )


def _domain(record: FlowRecord) -> tuple[str, str]:
    label = str(record.labels.get("binary", "")).lower()
    if label == "benign":
        return label, str(record.labels.get("application", "unknown"))
    if label == "malicious":
        return label, str(record.labels.get("family", "unknown"))
    return label, "unknown"


def _partition_bucket(record: FlowRecord, seed: int) -> int:
    group = _group_key(record, 300)
    return _stable_hash(f"{seed}|{group}") % 10


def _matrix_subset(
    matrix: FeatureMatrix,
    mask: np.ndarray,
) -> FeatureMatrix:
    ids = np.asarray(matrix.sample_ids, dtype=object)
    return FeatureMatrix(
        x=matrix.x[mask],
        y=matrix.y[mask],
        groups=matrix.groups[mask],
        sample_ids=list(ids[mask]),
    )


@dataclass(slots=True)
class CachedAgentFeatures:
    matrix: FeatureMatrix
    domains: np.ndarray
    buckets: np.ndarray
    partition_groups: np.ndarray

    def partitions(
        self,
        *,
        held_benign: str,
        held_malicious: str,
    ) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix]:
        held = (
            ((self.matrix.y == 0) & (self.domains == held_benign))
            | ((self.matrix.y == 1) & (self.domains == held_malicious))
        )
        remaining = ~held
        roles = _balanced_group_roles(
            self.matrix.y,
            self.partition_groups,
            remaining,
        )
        train = _matrix_subset(
            self.matrix, remaining & (roles == 2)
        )
        calibration = _matrix_subset(
            self.matrix, remaining & (roles == 0)
        )
        policy = _matrix_subset(
            self.matrix, remaining & (roles == 1)
        )
        for name, part in (
            ("train", train),
            ("calibration", calibration),
            ("policy", policy),
        ):
            if set(np.unique(part.y)) != {0, 1}:
                raise ValueError(
                    f"{name} partition lacks a binary class for "
                    f"{held_benign}/{held_malicious}"
                )
        return train, calibration, policy


def _balanced_group_roles(
    labels: np.ndarray,
    groups: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Assign whole hashed source/time groups to 80/10/10 partitions."""

    roles = np.full(len(labels), -1, dtype=np.int8)
    role_order = (2, 0, 1)  # train, calibration, policy
    ratios = {2: 0.8, 0: 0.1, 1: 0.1}
    for label in (0, 1):
        label_mask = eligible & (labels == label)
        unique, counts = np.unique(
            groups[label_mask], return_counts=True
        )
        if len(unique) < 3:
            raise ValueError(
                "at least three source/time groups per class are required"
            )
        ordered = sorted(
            zip(unique.tolist(), counts.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        total = int(sum(counts))
        targets = {role: total * ratios[role] for role in role_order}
        assigned = {role: 0 for role in role_order}
        group_roles: dict[int, int] = {}
        for index, (group, count) in enumerate(ordered):
            remaining_groups = len(ordered) - index
            empty_roles = [
                role for role in role_order if role not in group_roles.values()
            ]
            if remaining_groups == len(empty_roles):
                role = empty_roles[0]
            else:
                role = min(
                    role_order,
                    key=lambda candidate: (
                        sum(
                            (
                                assigned[item]
                                + (count if item == candidate else 0)
                                - targets[item]
                            )
                            ** 2
                            for item in role_order
                        ),
                        role_order.index(candidate),
                    ),
                )
            assigned[role] += count
            group_roles[int(group)] = role
        for group, role in group_roles.items():
            roles[label_mask & (groups == group)] = role
    return roles


@dataclass(slots=True)
class DomainFeatureCache:
    agents: dict[str, CachedAgentFeatures]
    domain_counts: dict[str, dict[str, int]]
    sequence_eligible_counts: dict[str, dict[str, int]]
    sample_count: int

    @classmethod
    def build(
        cls, flows: str | Path, *, seed: int
    ) -> "DomainFeatureCache":
        stats_rows: list[np.ndarray] = []
        stats_y: list[int] = []
        stats_groups: list[int] = []
        stats_ids: list[str] = []
        stats_domains: list[str] = []
        stats_buckets: list[int] = []
        stats_partition_groups: list[int] = []
        temporal_rows: list[np.ndarray] = []
        temporal_y: list[int] = []
        temporal_groups: list[int] = []
        temporal_ids: list[str] = []
        temporal_domains: list[str] = []
        temporal_buckets: list[int] = []
        temporal_partition_groups: list[int] = []
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        eligible: dict[str, Counter[str]] = defaultdict(Counter)
        stats_extract = feature_extractor("stats")
        temporal_extract = feature_extractor("temporal")
        total = 0

        for record in iter_flow_records(flows):
            label, domain = _domain(record)
            if label not in {"benign", "malicious"}:
                continue
            numeric_label = int(label == "malicious")
            detector_input = DetectorInput(
                stats=record.stats,
                sequence=record.sequence,
            )
            bucket = _partition_bucket(record, seed)
            partition_group = _stable_hash(
                f"{seed}|{_group_key(record, 300)}"
            )
            domain_group = _stable_hash(f"{label}:{domain}")
            stats_rows.append(stats_extract(detector_input))
            stats_y.append(numeric_label)
            stats_groups.append(domain_group)
            stats_ids.append(record.sample_id)
            stats_domains.append(domain)
            stats_buckets.append(bucket)
            stats_partition_groups.append(partition_group)
            counts[label][domain] += 1
            total += 1

            if len(record.sequence.packet_lengths) >= 4:
                temporal_rows.append(temporal_extract(detector_input))
                temporal_y.append(numeric_label)
                temporal_groups.append(domain_group)
                temporal_ids.append(record.sample_id)
                temporal_domains.append(domain)
                temporal_buckets.append(bucket)
                temporal_partition_groups.append(partition_group)
                eligible[label][domain] += 1

        def cached(
            rows: list[np.ndarray],
            labels: list[int],
            groups: list[int],
            ids: list[str],
            domains: list[str],
            buckets: list[int],
            partition_groups: list[int],
            width: int,
        ) -> CachedAgentFeatures:
            return CachedAgentFeatures(
                matrix=FeatureMatrix(
                    x=np.asarray(rows, dtype=np.float32).reshape(-1, width),
                    y=np.asarray(labels, dtype=np.int8),
                    groups=np.asarray(groups, dtype=np.uint64),
                    sample_ids=ids,
                ),
                domains=np.asarray(domains, dtype=object),
                buckets=np.asarray(buckets, dtype=np.int8),
                partition_groups=np.asarray(
                    partition_groups, dtype=np.uint64
                ),
            )

        return cls(
            agents={
                "stats": cached(
                    stats_rows,
                    stats_y,
                    stats_groups,
                    stats_ids,
                    stats_domains,
                    stats_buckets,
                    stats_partition_groups,
                    len(feature_names("stats")),
                ),
                "temporal": cached(
                    temporal_rows,
                    temporal_y,
                    temporal_groups,
                    temporal_ids,
                    temporal_domains,
                    temporal_buckets,
                    temporal_partition_groups,
                    len(feature_names("temporal")),
                ),
            },
            domain_counts={
                label: dict(values) for label, values in counts.items()
            },
            sequence_eligible_counts={
                label: dict(values) for label, values in eligible.items()
            },
            sample_count=total,
        )


def build_domain_fold_manifest(
    cache: DomainFeatureCache,
    *,
    seed: int = 42,
    folds: int = 10,
) -> dict[str, Any]:
    if folds < 1 or folds > len(DOMAIN_HOLDOUT_PAIRS):
        raise ValueError("folds must be between 1 and 10")
    fold_rows: list[dict[str, Any]] = []
    for index, (benign, malicious) in enumerate(DOMAIN_HOLDOUT_PAIRS[:folds]):
        fold_rows.append(
            {
                "fold": index,
                "held_out_benign_application": benign,
                "held_out_malware_family": malicious,
                "test_counts": {
                    "benign": cache.domain_counts["benign"][benign],
                    "malicious": cache.domain_counts["malicious"][malicious],
                },
                "temporal_eligible_counts": {
                    "benign": cache.sequence_eligible_counts["benign"].get(
                        benign, 0
                    ),
                    "malicious": cache.sequence_eligible_counts[
                        "malicious"
                    ].get(malicious, 0),
                },
            }
        )
    core = {
        "schema_version": DOMAIN_HOLDOUT_SCHEMA_VERSION,
        "dataset": "USTC-TFC2016",
        "seed": seed,
        "fold_count": folds,
        "partition_policy": {
            "test": "entire held-out benign application and malware family",
            "remaining_groups": "provenance source plus 300-second time block",
            "ratios": {
                "train": 0.8,
                "calibration": 0.1,
                "policy": 0.1,
            },
            "assignment": (
                "whole source/time groups sorted by sha256(seed|group), "
                "then greedily balanced to target ratios"
            ),
            "cross_validation_group": "binary label plus application/family domain",
        },
        "folds": fold_rows,
        "domain_counts": cache.domain_counts,
        "sequence_eligible_counts": cache.sequence_eligible_counts,
    }
    canonical = json.dumps(
        core, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    core["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return core


def iter_holdout_records(
    flows: str | Path,
    *,
    benign_application: str,
    malware_family: str,
) -> Iterator[FlowRecord]:
    for record in iter_flow_records(flows):
        label, domain = _domain(record)
        if (label == "benign" and domain == benign_application) or (
            label == "malicious" and domain == malware_family
        ):
            yield record


@dataclass(slots=True)
class MetricAccumulator:
    mode: str
    sample_count: int = 0
    labeled_count: int = 0
    covered_count: int = 0
    unknown_count: int = 0
    suspicious_count: int = 0
    total_calls: int = 0
    total_latency: float = 0.0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    labels: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def add(
        self,
        *,
        truth: str,
        verdict: str,
        risk_score: float,
        agent_calls: int,
        latency_ms: float,
    ) -> None:
        self.sample_count += 1
        covered = verdict in {"benign", "malicious"}
        self.covered_count += int(covered)
        self.unknown_count += int(verdict == "unknown")
        self.suspicious_count += int(verdict == "suspicious")
        self.total_calls += agent_calls
        self.total_latency += latency_ms
        if truth not in {"benign", "malicious"}:
            return
        self.labeled_count += 1
        label = int(truth == "malicious")
        self.labels.append(label)
        self.scores.append(risk_score)
        if not covered:
            return
        if label and verdict == "malicious":
            self.tp += 1
        elif not label and verdict == "malicious":
            self.fp += 1
        elif not label and verdict == "benign":
            self.tn += 1
        elif label and verdict == "benign":
            self.fn += 1

    def metrics(self) -> dict[str, Any]:
        covered_labeled = self.tp + self.fp + self.tn + self.fn
        if self.labeled_count:
            precision = _safe_div(self.tp, self.tp + self.fp)
            recall = _safe_div(self.tp, self.tp + self.fn)
            malicious_f1 = _safe_div(
                2 * precision * recall, precision + recall
            )
            benign_precision = _safe_div(self.tn, self.tn + self.fn)
            benign_recall = _safe_div(self.tn, self.tn + self.fp)
            benign_f1 = _safe_div(
                2 * benign_precision * benign_recall,
                benign_precision + benign_recall,
            )
            accuracy = _safe_div(self.tp + self.tn, covered_labeled)
            macro_f1 = (malicious_f1 + benign_f1) / 2
            brier = fmean(
                (score - label) ** 2
                for score, label in zip(
                    self.scores, self.labels, strict=True
                )
            )
            status = "computed"
        else:
            precision = recall = accuracy = macro_f1 = brier = None
            status = "skipped_no_labels"
        selective_error = (
            _safe_div(
                self.fp + self.fn,
                self.tp + self.fp + self.tn + self.fn,
            )
            if self.labeled_count
            else None
        )
        benign_fpr = (
            _safe_div(
                self.fp, sum(label == 0 for label in self.labels)
            )
            if self.labeled_count
            else None
        )
        malicious_detection = (
            _safe_div(
                self.tp, sum(label == 1 for label in self.labels)
            )
            if self.labeled_count
            else None
        )
        return {
            "schema_version": "1.0",
            "mode": self.mode,
            "metrics_status": status,
            "sample_count": self.sample_count,
            "labeled_sample_count": self.labeled_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "PR_AUC": _average_precision(self.labels, self.scores),
            "Brier_score": brier,
            "ECE": _expected_calibration_error(
                self.labels, self.scores
            ),
            "coverage": _safe_div(
                self.covered_count, self.sample_count
            ),
            "unknown_rate": _safe_div(
                self.unknown_count, self.sample_count
            ),
            "suspicious_rate": _safe_div(
                self.suspicious_count, self.sample_count
            ),
            "average_agent_calls": _safe_div(
                self.total_calls, self.sample_count
            ),
            "average_latency_ms": _safe_div(
                self.total_latency, self.sample_count
            ),
            "selective_error_rate": selective_error,
            "benign_false_positive_rate": benign_fpr,
            "malicious_detection_rate": malicious_detection,
            "confusion_on_covered": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
        }


@dataclass(slots=True)
class AuditAccumulator:
    event_counts: Counter[str] = field(default_factory=Counter)
    actor_counts: Counter[str] = field(default_factory=Counter)
    completed_chains: int = 0
    blocked_violations: int = 0
    policy_adjustments: int = 0
    sample_count: int = 0

    def add(self, audit: Any) -> None:
        self.sample_count += 1
        self.completed_chains += int(
            bool(audit.events)
            and audit.events[-1].event_type == "REPORT_GENERATED"
        )
        for event in audit.events:
            self.event_counts[event.event_type] += 1
            self.actor_counts[event.actor] += 1
            if event.event_type == "AGENT_EVIDENCE":
                self.blocked_violations += len(
                    event.input_summary.get(
                        "blocked_field_intersection", []
                    )
                )
            if event.event_type == "ROUTING_DECISION":
                self.policy_adjustments += len(
                    event.output_summary.get("policy_adjustments", [])
                )

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "sample_count": self.sample_count,
            "completed_chain_count": self.completed_chains,
            "audit_chain_completion_rate": _safe_div(
                self.completed_chains, self.sample_count
            ),
            "blocked_field_violation_count": self.blocked_violations,
            "policy_adjustment_count": self.policy_adjustments,
            "event_counts": dict(self.event_counts),
            "actor_counts": dict(self.actor_counts),
        }


def _fusion_risk(fusion: Any) -> float:
    total = fusion.benign_support + fusion.malicious_support
    return fusion.malicious_support / total if total else 0.5


def _audit_inputs(
    audit: Any,
) -> tuple[list[AgentEvidence], ReliabilityProfile]:
    evidence: list[AgentEvidence] = []
    reliability: ReliabilityProfile | None = None
    for event in audit.events:
        if event.event_type == "AGENT_EVIDENCE":
            evidence.append(
                AgentEvidence.model_validate(event.output_summary)
            )
        elif event.event_type in {
            "RELIABILITY_ASSESSED",
            "RELIABILITY_BYPASSED",
        }:
            reliability = ReliabilityProfile.model_validate(
                event.output_summary
            )
    if reliability is None:
        raise RuntimeError("audit chain lacks reliability profile")
    return evidence, reliability


def _system_fusions(
    audit: Any,
) -> tuple[dict[str, Any], bool]:
    evidence, reliability = _audit_inputs(audit)
    by_name = {item.agent_name: item for item in evidence}
    reject_fusion = FusionAgent(
        use_reliability_discount=True, allow_reject=True
    )
    no_reject_fusion = FusionAgent(
        use_reliability_discount=True, allow_reject=False
    )
    stats = by_name.get("StatsDetectorAgent")
    temporal = by_name.get("TemporalBehaviorAgent")
    return (
        {
            "stats_only": reject_fusion.fuse(
                [stats] if stats else [], reliability, final=True
            ),
            "temporal_only": reject_fusion.fuse(
                [temporal] if temporal else [], reliability, final=True
            ),
            "no_reject": no_reject_fusion.fuse(
                evidence, reliability, final=True
            ),
        },
        bool(temporal is None or temporal.abstained),
    )


def _prediction_fieldnames() -> list[str]:
    return [
        "schema_version",
        "fold",
        "held_out_benign_application",
        "held_out_malware_family",
        "trace_id",
        "sample_id",
        "true_label",
        "domain_type",
        "domain_name",
        "sequence_length",
        "temporal_abstained",
        "verdict",
        "covered",
        "risk_score",
        "confidence",
        "uncertainty",
        "conflict_score",
        "benign_support",
        "malicious_support",
        "agent_calls",
        "coordinator_rounds",
        "latency_ms",
        "stats_verdict",
        "stats_risk_score",
        "temporal_verdict",
        "temporal_risk_score",
        "no_reject_verdict",
        "audit_complete",
    ]


def evaluate_holdout_fold(
    *,
    flows: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    fold: int,
    held_benign: str,
    held_malicious: str,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    engine = build_default_engine(
        field_audit_mode="legacy",
        force_rule_coordinator=True,
        detector_backend="learned",
        model_dir=model_dir,
        max_workers=1,
    )
    accumulators = {
        system: MetricAccumulator(system)
        for system in GENERALIZATION_SYSTEMS
    }
    audit_acc = AuditAccumulator()
    temporal_abstained = 0
    predictions_path = target / "predictions.csv"
    with predictions_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_prediction_fieldnames()
        )
        writer.writeheader()
        for record in iter_holdout_records(
            flows,
            benign_application=held_benign,
            malware_family=held_malicious,
        ):
            started = time.perf_counter()
            report, audit = engine.analyze(record)
            latency = (time.perf_counter() - started) * 1000
            derived, temporal_abstain = _system_fusions(audit)
            temporal_abstained += int(temporal_abstain)
            truth, domain_name = _domain(record)
            domain_type = (
                "benign_application"
                if truth == "benign"
                else "malware_family"
            )
            full_risk = _safe_div(
                report.malicious_support,
                report.benign_support + report.malicious_support,
            ) if report.benign_support + report.malicious_support else 0.5
            systems = {
                "full": (
                    report.verdict.value,
                    full_risk,
                    len(report.participating_agents),
                    latency,
                ),
                "stats_only": (
                    derived["stats_only"].verdict.value,
                    _fusion_risk(derived["stats_only"]),
                    1,
                    0,
                ),
                "temporal_only": (
                    derived["temporal_only"].verdict.value,
                    _fusion_risk(derived["temporal_only"]),
                    1,
                    0,
                ),
                "no_reject": (
                    derived["no_reject"].verdict.value,
                    _fusion_risk(derived["no_reject"]),
                    len(report.participating_agents),
                    0,
                ),
            }
            for name, values in systems.items():
                accumulators[name].add(
                    truth=truth,
                    verdict=values[0],
                    risk_score=values[1],
                    agent_calls=values[2],
                    latency_ms=values[3],
                )
            audit_acc.add(audit)
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "fold": fold,
                    "held_out_benign_application": held_benign,
                    "held_out_malware_family": held_malicious,
                    "trace_id": record.trace_id,
                    "sample_id": record.sample_id,
                    "true_label": truth,
                    "domain_type": domain_type,
                    "domain_name": domain_name,
                    "sequence_length": len(
                        record.sequence.packet_lengths
                    ),
                    "temporal_abstained": temporal_abstain,
                    "verdict": report.verdict.value,
                    "covered": report.verdict
                    in {Verdict.BENIGN, Verdict.MALICIOUS},
                    "risk_score": full_risk,
                    "confidence": report.confidence,
                    "uncertainty": report.uncertainty,
                    "conflict_score": report.conflict_score,
                    "benign_support": report.benign_support,
                    "malicious_support": report.malicious_support,
                    "agent_calls": len(report.participating_agents),
                    "coordinator_rounds": len(
                        report.coordinator_decisions
                    ),
                    "latency_ms": latency,
                    "stats_verdict": derived[
                        "stats_only"
                    ].verdict.value,
                    "stats_risk_score": _fusion_risk(
                        derived["stats_only"]
                    ),
                    "temporal_verdict": derived[
                        "temporal_only"
                    ].verdict.value,
                    "temporal_risk_score": _fusion_risk(
                        derived["temporal_only"]
                    ),
                    "no_reject_verdict": derived[
                        "no_reject"
                    ].verdict.value,
                    "audit_complete": bool(audit.events)
                    and audit.events[-1].event_type
                    == "REPORT_GENERATED",
                }
            )
    metrics = {
        "schema_version": "1.0",
        "fold": fold,
        "holdout": {
            "benign_application": held_benign,
            "malware_family": held_malicious,
        },
        "systems": {
            name: accumulator.metrics()
            for name, accumulator in accumulators.items()
        },
        "temporal_abstain_rate": _safe_div(
            temporal_abstained, accumulators["full"].sample_count
        ),
        "audit_summary": audit_acc.summary(),
    }
    _write_json(target / "metrics.json", metrics)
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _aggregate_metric_dicts(
    fold_metrics: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    numeric_keys = (
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "PR_AUC",
        "Brier_score",
        "ECE",
        "coverage",
        "unknown_rate",
        "suspicious_rate",
        "average_agent_calls",
        "average_latency_ms",
    )
    for system in GENERALIZATION_SYSTEMS:
        per_fold = [
            item["systems"][system] for item in fold_metrics
        ]
        summary: dict[str, Any] = {}
        for key in numeric_keys:
            values = [
                float(item[key])
                for item in per_fold
                if item.get(key) is not None
            ]
            summary[key] = {
                "mean": fmean(values) if values else None,
                "std": pstdev(values) if len(values) >= 2 else 0.0
                if values
                else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        result[system] = summary
    return result


def _row_system_values(
    row: dict[str, str], system: str
) -> tuple[str, float, int, float]:
    if system == "full":
        return (
            row["verdict"],
            float(row["risk_score"]),
            int(row["agent_calls"]),
            float(row["latency_ms"]),
        )
    if system == "stats_only":
        return row["stats_verdict"], float(row["stats_risk_score"]), 1, 0
    if system == "temporal_only":
        return (
            row["temporal_verdict"],
            float(row["temporal_risk_score"]),
            1,
            0,
        )
    return (
        row["no_reject_verdict"],
        float(row["risk_score"]),
        int(row["agent_calls"]),
        0,
    )


def aggregate_holdout_results(
    *,
    output_dir: str | Path,
    fold_rows: list[dict[str, Any]],
    baseline_metrics_path: str | Path | None,
) -> dict[str, Any]:
    root = Path(output_dir)
    aggregate = root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    pooled = {
        system: MetricAccumulator(system)
        for system in GENERALIZATION_SYSTEMS
    }
    per_domain: dict[tuple[str, str, str], MetricAccumulator] = {}
    fold_metrics: list[dict[str, Any]] = []
    audit_total = AuditAccumulator()
    pooled_path = aggregate / "pooled_predictions.csv"
    with pooled_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as output:
        writer = csv.DictWriter(
            output, fieldnames=_prediction_fieldnames()
        )
        writer.writeheader()
        for fold_info in fold_rows:
            fold = int(fold_info["fold"])
            fold_dir = root / "folds" / f"fold-{fold:02d}"
            metrics = json.loads(
                (fold_dir / "metrics.json").read_text(encoding="utf-8")
            )
            fold_metrics.append(metrics)
            audit = metrics["audit_summary"]
            audit_total.sample_count += int(audit["sample_count"])
            audit_total.completed_chains += int(
                audit["completed_chain_count"]
            )
            audit_total.blocked_violations += int(
                audit["blocked_field_violation_count"]
            )
            audit_total.policy_adjustments += int(
                audit["policy_adjustment_count"]
            )
            audit_total.event_counts.update(audit["event_counts"])
            audit_total.actor_counts.update(audit["actor_counts"])
            with (fold_dir / "predictions.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    writer.writerow(row)
                    truth = row["true_label"]
                    for system in GENERALIZATION_SYSTEMS:
                        verdict, risk, calls, latency = (
                            _row_system_values(row, system)
                        )
                        pooled[system].add(
                            truth=truth,
                            verdict=verdict,
                            risk_score=risk,
                            agent_calls=calls,
                            latency_ms=latency,
                        )
                        key = (
                            system,
                            row["domain_type"],
                            row["domain_name"],
                        )
                        per_domain.setdefault(
                            key, MetricAccumulator(system)
                        ).add(
                            truth=truth,
                            verdict=verdict,
                            risk_score=risk,
                            agent_calls=calls,
                            latency_ms=latency,
                        )

    pooled_metrics = {
        system: accumulator.metrics()
        for system, accumulator in pooled.items()
    }
    aggregate_metrics = {
        "schema_version": "1.0",
        "fold_count": len(fold_rows),
        "pooled": pooled_metrics,
        "fold_summary": _aggregate_metric_dicts(fold_metrics),
        "folds": fold_metrics,
    }
    _write_json(aggregate / "aggregate_metrics.json", aggregate_metrics)

    with (aggregate / "per_domain_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "schema_version",
            "system",
            "domain_type",
            "domain_name",
            "sample_count",
            "coverage",
            "accuracy",
            "macro_f1",
            "precision",
            "recall",
            "PR_AUC",
            "Brier_score",
            "ECE",
            "unknown_rate",
            "suspicious_rate",
            "average_agent_calls",
            "selective_error_rate",
            "benign_false_positive_rate",
            "malicious_detection_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(per_domain):
            system, domain_type, domain_name = key
            metrics = per_domain[key].metrics()
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "system": system,
                    "domain_type": domain_type,
                    "domain_name": domain_name,
                    **{field: metrics.get(field) for field in fields[4:]},
                }
            )

    full = pooled["full"]
    confidence = np.abs(np.asarray(full.scores) - 0.5)
    order = np.argsort(-confidence)
    labels = np.asarray(full.labels, dtype=np.int8)[order]
    predictions = (
        np.asarray(full.scores, dtype=np.float64)[order] >= 0.5
    ).astype(np.int8)
    errors = np.cumsum(predictions != labels)
    curve: list[dict[str, Any]] = []
    for target in (0.01, 0.05, 0.10):
        risks = errors / np.arange(1, len(labels) + 1)
        eligible = np.flatnonzero(risks <= target)
        count = int(eligible[-1] + 1) if len(eligible) else 0
        curve.append(
            {
                "target_risk": target,
                "coverage": _safe_div(count, len(labels)),
                "observed_risk": float(risks[count - 1])
                if count
                else None,
            }
        )
    _write_json(
        aggregate / "coverage_risk.json",
        {"schema_version": "1.0", "system": "full", "points": curve},
    )

    gap: dict[str, Any] = {
        "schema_version": "1.0",
        "baseline_available": False,
    }
    if baseline_metrics_path and Path(baseline_metrics_path).exists():
        baseline = json.loads(
            Path(baseline_metrics_path).read_text(encoding="utf-8")
        )
        gap = {
            "schema_version": "1.0",
            "baseline_available": True,
            "baseline_path": str(Path(baseline_metrics_path).resolve()),
            "baseline": baseline,
            "domain_holdout": pooled_metrics["full"],
            "delta": {
                key: (
                    pooled_metrics["full"][key] - baseline[key]
                    if pooled_metrics["full"].get(key) is not None
                    and baseline.get(key) is not None
                    else None
                )
                for key in (
                    "accuracy",
                    "macro_f1",
                    "PR_AUC",
                    "Brier_score",
                    "ECE",
                    "coverage",
                    "unknown_rate",
                    "suspicious_rate",
                )
            },
        }
    _write_json(aggregate / "generalization_gap.json", gap)
    _write_json(
        aggregate / "audit_summary.json", audit_total.summary()
    )
    _write_json(
        aggregate / "cost_summary.json",
        {
            "schema_version": "1.0",
            "sample_count": pooled["full"].sample_count,
            "total_agent_calls": pooled["full"].total_calls,
            "average_agent_calls": pooled_metrics["full"][
                "average_agent_calls"
            ],
            "total_latency_ms": pooled["full"].total_latency,
            "average_latency_ms": pooled_metrics["full"][
                "average_latency_ms"
            ],
        },
    )
    return aggregate_metrics


@dataclass(slots=True)
class OODGroup:
    count: int = 0
    benign: int = 0
    malicious: int = 0
    suspicious: int = 0
    unknown: int = 0
    risk_sum: float = 0.0
    calls: int = 0

    def add(self, verdict: str, risk: float, calls: int) -> None:
        self.count += 1
        setattr(self, verdict, getattr(self, verdict) + 1)
        self.risk_sum += risk
        self.calls += calls

    def row(self) -> dict[str, Any]:
        return {
            "sample_count": self.count,
            "benign_rate": _safe_div(self.benign, self.count),
            "malicious_rate": _safe_div(self.malicious, self.count),
            "suspicious_rate": _safe_div(self.suspicious, self.count),
            "unknown_rate": _safe_div(self.unknown, self.count),
            "alert_rate": _safe_div(
                self.malicious + self.suspicious, self.count
            ),
            "average_risk_score": _safe_div(
                self.risk_sum, self.count
            ),
            "average_agent_calls": _safe_div(self.calls, self.count),
        }


def evaluate_cesnet_ood(
    *,
    cesnet_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset = Path(cesnet_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    engine = build_default_engine(
        field_audit_mode="legacy",
        force_rule_coordinator=True,
        detector_backend="learned",
        model_dir=model_dir,
        max_workers=1,
    )
    metrics = MetricAccumulator("cesnet_ood")
    audit_acc = AuditAccumulator()
    applications: dict[str, OODGroup] = defaultdict(OODGroup)
    categories: dict[str, OODGroup] = defaultdict(OODGroup)
    overall = OODGroup()
    confidences: list[float] = []
    uncertainties: list[float] = []
    risks: list[float] = []
    records: Iterable[FlowRecord] = iter_split_records(
        dataset / "flows",
        dataset / "splits" / "split-manifest.json",
        "test",
    )
    if limit is not None:
        import itertools

        records = itertools.islice(records, limit)
    prediction_fields = [
        "schema_version",
        "trace_id",
        "sample_id",
        "application",
        "category",
        "verdict",
        "risk_score",
        "confidence",
        "uncertainty",
        "agent_calls",
        "latency_ms",
    ]
    with (target / "cesnet_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for record in records:
            started = time.perf_counter()
            report, audit = engine.analyze(record)
            latency = (time.perf_counter() - started) * 1000
            risk = _safe_div(
                report.malicious_support,
                report.benign_support + report.malicious_support,
            ) if report.benign_support + report.malicious_support else 0.5
            verdict = report.verdict.value
            calls = len(report.participating_agents)
            application = str(
                record.labels.get("application", "unknown")
            )
            category = str(record.labels.get("category", "unknown"))
            metrics.add(
                truth="",
                verdict=verdict,
                risk_score=risk,
                agent_calls=calls,
                latency_ms=latency,
            )
            applications[application].add(verdict, risk, calls)
            categories[category].add(verdict, risk, calls)
            overall.add(verdict, risk, calls)
            confidences.append(report.confidence)
            uncertainties.append(report.uncertainty)
            risks.append(risk)
            audit_acc.add(audit)
            writer.writerow(
                {
                    "schema_version": "1.0",
                    "trace_id": record.trace_id,
                    "sample_id": record.sample_id,
                    "application": application,
                    "category": category,
                    "verdict": verdict,
                    "risk_score": risk,
                    "confidence": report.confidence,
                    "uncertainty": report.uncertainty,
                    "agent_calls": calls,
                    "latency_ms": latency,
                }
            )

    def write_groups(
        path: Path, label: str, groups: dict[str, OODGroup]
    ) -> None:
        fields = [
            "schema_version",
            label,
            "sample_count",
            "benign_rate",
            "malicious_rate",
            "suspicious_rate",
            "unknown_rate",
            "alert_rate",
            "average_risk_score",
            "average_agent_calls",
        ]
        with path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, group in sorted(
                groups.items(),
                key=lambda item: (-item[1].row()["alert_rate"], item[0]),
            ):
                writer.writerow(
                    {
                        "schema_version": "1.0",
                        label: name,
                        **group.row(),
                    }
                )

    write_groups(
        target / "cesnet_application_metrics.csv",
        "application",
        applications,
    )
    write_groups(
        target / "cesnet_category_metrics.csv", "category", categories
    )
    result = {
        "schema_version": "1.0",
        "dataset": "CESNET-TLS22-v3-sample-1m",
        "split": "test",
        "metrics": metrics.metrics(),
        "verdict_distribution": overall.row(),
        "score_distribution": {
            "confidence_quantiles": _quantiles(confidences),
            "uncertainty_quantiles": _quantiles(uncertainties),
            "risk_score_quantiles": _quantiles(risks),
        },
        "audit_summary": audit_acc.summary(),
        "high_alert_applications": [
            {"application": name, **group.row()}
            for name, group in sorted(
                applications.items(),
                key=lambda item: (
                    -item[1].row()["alert_rate"],
                    -item[1].count,
                ),
            )[:20]
        ],
    }
    _write_json(target / "cesnet_metrics.json", result)
    return result


def summarize_existing_cesnet_ood(
    output_dir: str | Path,
    *,
    audit_summary: dict[str, Any],
) -> dict[str, Any]:
    target = Path(output_dir)
    metrics = MetricAccumulator("cesnet_ood")
    applications: dict[str, OODGroup] = defaultdict(OODGroup)
    categories: dict[str, OODGroup] = defaultdict(OODGroup)
    overall = OODGroup()
    confidences: list[float] = []
    uncertainties: list[float] = []
    risks: list[float] = []
    with (target / "cesnet_predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            verdict = row["verdict"]
            risk = float(row["risk_score"])
            calls = int(row["agent_calls"])
            latency = float(row["latency_ms"])
            metrics.add(
                truth="",
                verdict=verdict,
                risk_score=risk,
                agent_calls=calls,
                latency_ms=latency,
            )
            applications[row["application"]].add(verdict, risk, calls)
            categories[row["category"]].add(verdict, risk, calls)
            overall.add(verdict, risk, calls)
            confidences.append(float(row["confidence"]))
            uncertainties.append(float(row["uncertainty"]))
            risks.append(risk)

    def write_groups(
        path: Path, label: str, groups: dict[str, OODGroup]
    ) -> None:
        fields = [
            "schema_version",
            label,
            "sample_count",
            "benign_rate",
            "malicious_rate",
            "suspicious_rate",
            "unknown_rate",
            "alert_rate",
            "average_risk_score",
            "average_agent_calls",
        ]
        with path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, group in sorted(
                groups.items(),
                key=lambda item: (-item[1].row()["alert_rate"], item[0]),
            ):
                writer.writerow(
                    {
                        "schema_version": "1.0",
                        label: name,
                        **group.row(),
                    }
                )

    write_groups(
        target / "cesnet_application_metrics.csv",
        "application",
        applications,
    )
    write_groups(
        target / "cesnet_category_metrics.csv", "category", categories
    )
    result = {
        "schema_version": "1.0",
        "dataset": "CESNET-TLS22-v3-sample-1m",
        "split": "test",
        "metrics": metrics.metrics(),
        "verdict_distribution": overall.row(),
        "score_distribution": {
            "confidence_quantiles": _quantiles(confidences),
            "uncertainty_quantiles": _quantiles(uncertainties),
            "risk_score_quantiles": _quantiles(risks),
        },
        "audit_summary": audit_summary,
        "high_alert_applications": [
            {"application": name, **group.row()}
            for name, group in sorted(
                applications.items(),
                key=lambda item: (
                    -item[1].row()["alert_rate"],
                    -item[1].count,
                ),
            )[:20]
        ],
    }
    _write_json(target / "cesnet_metrics.json", result)
    return result


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_is_complete(
    fold_dir: Path,
    manifest_path: Path,
    *,
    held_benign: str,
    held_malicious: str,
) -> bool:
    models_dir = fold_dir / "models"
    required = [
        fold_dir / "metrics.json",
        fold_dir / "predictions.csv",
        models_dir / "stats" / "model.joblib",
        models_dir / "temporal" / "model.joblib",
        models_dir / "stats" / "metadata.json",
        models_dir / "temporal" / "metadata.json",
    ]
    if not all(path.exists() for path in required):
        return False
    manifest_hash = _file_sha256(manifest_path)
    try:
        for agent in ("stats", "temporal"):
            metadata = json.loads(
                (models_dir / agent / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                metadata.get("split_manifest_sha256") != manifest_hash
                or metadata.get("held_out_benign_application")
                != held_benign
                or metadata.get("held_out_malware_family")
                != held_malicious
            ):
                return False
    except (OSError, json.JSONDecodeError):
        return False
    return True


def validate_generalization(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    folds: int = 10,
    seed: int = 42,
    cv_splits: int = 3,
    resume: bool = True,
    baseline_metrics_path: str | Path | None = None,
    cesnet_dir: str | Path | None = None,
    full_model_dir: str | Path | None = None,
    cesnet_limit: int | None = None,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    flows = dataset / "flows"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cache = DomainFeatureCache.build(flows, seed=seed)
    manifest = build_domain_fold_manifest(
        cache, seed=seed, folds=folds
    )
    manifest_path = target / "domain-fold-manifest.json"
    _write_json(manifest_path, manifest)
    extraction_manifest = (
        dataset / "manifests" / "extraction_manifest.json"
    )

    for fold_info in manifest["folds"]:
        fold = int(fold_info["fold"])
        held_benign = fold_info["held_out_benign_application"]
        held_malicious = fold_info["held_out_malware_family"]
        fold_dir = target / "folds" / f"fold-{fold:02d}"
        models_dir = fold_dir / "models"
        complete = _fold_is_complete(
            fold_dir,
            manifest_path,
            held_benign=held_benign,
            held_malicious=held_malicious,
        )
        if resume and complete:
            continue
        for agent in ("stats", "temporal"):
            train, calibration, policy = cache.agents[
                agent
            ].partitions(
                held_benign=held_benign,
                held_malicious=held_malicious,
            )
            train_detector(
                agent=agent,
                train=train,
                calibration=calibration,
                policy=policy,
                output_dir=models_dir / agent,
                seed=seed,
                cv_splits=cv_splits,
                split_manifest_path=manifest_path,
                extraction_manifest_path=extraction_manifest,
                extra_metadata={
                    "experiment": "domain_holdout_v1",
                    "fold": fold,
                    "held_out_benign_application": held_benign,
                    "held_out_malware_family": held_malicious,
                    "training_domains": {
                        "benign_applications": sorted(
                            set(cache.domain_counts["benign"])
                            - {held_benign}
                        ),
                        "malware_families": sorted(
                            set(cache.domain_counts["malicious"])
                            - {held_malicious}
                        ),
                    },
                    "test_sample_ids_stored": False,
                },
            )
        evaluate_holdout_fold(
            flows=flows,
            model_dir=models_dir,
            output_dir=fold_dir,
            fold=fold,
            held_benign=held_benign,
            held_malicious=held_malicious,
        )

    aggregate = aggregate_holdout_results(
        output_dir=target,
        fold_rows=manifest["folds"],
        baseline_metrics_path=baseline_metrics_path,
    )
    cesnet_result = None
    if cesnet_dir is not None and full_model_dir is not None:
        cesnet_metrics_path = target / "aggregate" / "cesnet_metrics.json"
        expected_count = (
            cesnet_limit
            if cesnet_limit is not None
            else 499_997
        )
        if resume and cesnet_metrics_path.exists():
            cached_ood = json.loads(
                cesnet_metrics_path.read_text(encoding="utf-8")
            )
            if (
                cached_ood.get("metrics", {}).get("sample_count")
                == expected_count
            ):
                if (
                    "verdict_distribution" in cached_ood
                    and cached_ood.get("metrics", {}).get(
                        "selective_error_rate"
                    )
                    is None
                ):
                    cesnet_result = cached_ood
                else:
                    cesnet_result = summarize_existing_cesnet_ood(
                        target / "aggregate",
                        audit_summary=cached_ood["audit_summary"],
                    )
        if cesnet_result is None:
            cesnet_result = evaluate_cesnet_ood(
                cesnet_dir=cesnet_dir,
                model_dir=full_model_dir,
                output_dir=target / "aggregate",
                limit=cesnet_limit,
            )
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset.resolve()),
        "output_dir": str(target.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _file_sha256(manifest_path),
        "fold_count": folds,
        "aggregate_metrics": aggregate,
        "cesnet_ood": cesnet_result,
    }
    _write_json(target / "validation_summary.json", summary)
    return summary
