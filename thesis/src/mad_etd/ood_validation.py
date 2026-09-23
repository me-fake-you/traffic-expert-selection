from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Iterator

import numpy as np

from .engine import build_default_engine
from .evaluation import _safe_div
from .fusion import FusionAgent
from .generalization import (
    AuditAccumulator,
    DOMAIN_HOLDOUT_PAIRS,
    DomainFeatureCache,
    MetricAccumulator,
    OODGroup,
    _audit_inputs,
    _domain,
    _file_sha256,
    _fusion_risk,
    _quantiles,
    _write_json,
    iter_holdout_records,
)
from .io import iter_split_records
from .ood import (
    OODAssessment,
    OODReliabilityGate,
    apply_ood_policy_to_evidence,
    train_ood_gate,
)
from .schemas import AgentEvidence, DetectorInput, FlowRecord


OOD_VALIDATION_STRATEGIES = (
    "off",
    "soft",
    "hybrid",
    "stats_gate_only",
    "temporal_gate_only",
)


def _policy_for_agent(strategy: str, agent_name: str) -> str:
    if strategy in {"off", "soft", "hybrid"}:
        return strategy
    if strategy == "stats_gate_only":
        return "hybrid" if agent_name == "StatsDetectorAgent" else "off"
    if strategy == "temporal_gate_only":
        return "hybrid" if agent_name == "TemporalBehaviorAgent" else "off"
    raise ValueError(f"unsupported OOD validation strategy: {strategy}")


def _attach_assessment(
    evidence: AgentEvidence,
    assessment: OODAssessment | None,
) -> AgentEvidence:
    if assessment is None:
        return evidence
    return evidence.model_copy(
        update={
            "distribution_shift_score": assessment.shift_score,
            "distribution_shift_raw_score": assessment.raw_score,
            "distribution_shift_level": assessment.level,
            "model_reliability": 1.0,
            "evidence": [
                *evidence.evidence,
                (
                    "model-distribution shift "
                    f"{assessment.level} "
                    f"(score={assessment.shift_score:.3f})"
                ),
            ],
        }
    )


def _strategy_evidence(
    evidence: list[AgentEvidence],
    strategy: str,
) -> list[AgentEvidence]:
    result: list[AgentEvidence] = []
    for item in evidence:
        if item.agent_name in {
            "StatsDetectorAgent",
            "TemporalBehaviorAgent",
        }:
            result.append(
                apply_ood_policy_to_evidence(
                    item, _policy_for_agent(strategy, item.agent_name)
                )
            )
        else:
            result.append(item)
    return result


def _prediction_fields() -> list[str]:
    fields = [
        "schema_version",
        "trace_id",
        "sample_id",
        "true_label",
        "domain_type",
        "domain_name",
        "application",
        "category",
        "stats_shift_score",
        "temporal_shift_score",
        "combined_shift_score",
        "stats_shift_level",
        "temporal_shift_level",
        "agent_calls",
        "latency_ms",
        "audit_complete",
    ]
    for strategy in OOD_VALIDATION_STRATEGIES:
        fields.extend(
            [
                f"{strategy}_verdict",
                f"{strategy}_risk_score",
                f"{strategy}_confidence",
                f"{strategy}_uncertainty",
                f"{strategy}_coverage",
            ]
        )
    return fields


@dataclass(slots=True)
class ReplayResult:
    metrics: dict[str, dict[str, Any]]
    audit_summary: dict[str, Any]
    stats_shift_scores: list[float]
    temporal_shift_scores: list[float]
    combined_shift_scores: list[float]
    error_labels: list[int]
    application_groups: dict[str, dict[str, OODGroup]]
    category_groups: dict[str, dict[str, OODGroup]]
    accumulators: dict[str, MetricAccumulator] = field(repr=False)


def _score_batch(
    records: list[FlowRecord],
    gates: dict[str, OODReliabilityGate],
) -> tuple[list[OODAssessment], dict[int, OODAssessment]]:
    stats_features = np.vstack(
        [
            gates["stats"].extract(
                DetectorInput(stats=record.stats, sequence=record.sequence)
            )
            for record in records
        ]
    )
    stats = gates["stats"].assess_feature_matrix(stats_features)
    temporal_indices = [
        index
        for index, record in enumerate(records)
        if len(record.sequence.packet_lengths) >= 4
    ]
    temporal: dict[int, OODAssessment] = {}
    if temporal_indices:
        temporal_features = np.vstack(
            [
                gates["temporal"].extract(
                    DetectorInput(
                        stats=records[index].stats,
                        sequence=records[index].sequence,
                    )
                )
                for index in temporal_indices
            ]
        )
        temporal_values = gates["temporal"].assess_feature_matrix(
            temporal_features
        )
        temporal = dict(
            zip(temporal_indices, temporal_values, strict=True)
        )
    return stats, temporal


def _evaluate_replay(
    records: Iterable[FlowRecord],
    *,
    model_dir: str | Path,
    gate_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 2048,
) -> ReplayResult:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    engine = build_default_engine(
        field_audit_mode="legacy",
        force_rule_coordinator=True,
        detector_backend="learned",
        model_dir=model_dir,
        max_workers=1,
        ood_policy="off",
    )
    gate_root = Path(gate_dir)
    gates = {
        agent: OODReliabilityGate(
            gate_root / agent, expected_agent=agent
        )
        for agent in ("stats", "temporal")
    }
    accumulators = {
        strategy: MetricAccumulator(strategy)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    audit_acc = AuditAccumulator()
    application_groups = {
        strategy: defaultdict(OODGroup)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    category_groups = {
        strategy: defaultdict(OODGroup)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    stats_scores: list[float] = []
    temporal_scores: list[float] = []
    combined_scores: list[float] = []
    error_labels: list[int] = []
    fields = _prediction_fields()
    buffer: list[FlowRecord] = []

    with (target / "predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        def process(items: list[FlowRecord]) -> None:
            if not items:
                return
            stats_assessments, temporal_assessments = _score_batch(
                items, gates
            )
            for index, record in enumerate(items):
                started = time.perf_counter()
                report, audit = engine.analyze(record)
                latency = (time.perf_counter() - started) * 1000
                base_evidence, reliability = _audit_inputs(audit)
                enriched: list[AgentEvidence] = []
                stats_assessment = stats_assessments[index]
                temporal_assessment = temporal_assessments.get(index)
                for item in base_evidence:
                    assessment = (
                        stats_assessment
                        if item.agent_name == "StatsDetectorAgent"
                        else temporal_assessment
                        if item.agent_name == "TemporalBehaviorAgent"
                        else None
                    )
                    enriched.append(_attach_assessment(item, assessment))
                temporal_score = (
                    temporal_assessment.shift_score
                    if temporal_assessment
                    else 0.0
                )
                combined_score = max(
                    stats_assessment.shift_score, temporal_score
                )
                stats_scores.append(stats_assessment.shift_score)
                if temporal_assessment:
                    temporal_scores.append(temporal_score)
                combined_scores.append(combined_score)
                truth, domain_name = _domain(record)
                domain_type = (
                    "benign_application"
                    if truth == "benign"
                    else "malware_family"
                    if truth == "malicious"
                    else ""
                )
                application = str(
                    record.labels.get("application", "")
                )
                category = str(record.labels.get("category", ""))
                row: dict[str, Any] = {
                    "schema_version": "1.0",
                    "trace_id": record.trace_id,
                    "sample_id": record.sample_id,
                    "true_label": truth
                    if truth in {"benign", "malicious"}
                    else "",
                    "domain_type": domain_type,
                    "domain_name": domain_name
                    if truth in {"benign", "malicious"}
                    else "",
                    "application": application,
                    "category": category,
                    "stats_shift_score": stats_assessment.shift_score,
                    "temporal_shift_score": (
                        temporal_assessment.shift_score
                        if temporal_assessment
                        else ""
                    ),
                    "combined_shift_score": combined_score,
                    "stats_shift_level": stats_assessment.level,
                    "temporal_shift_level": (
                        temporal_assessment.level
                        if temporal_assessment
                        else "unavailable"
                    ),
                    "agent_calls": len(report.participating_agents),
                    "latency_ms": latency,
                    "audit_complete": bool(audit.events)
                    and audit.events[-1].event_type
                    == "REPORT_GENERATED",
                }
                off_binary_error = 0
                for strategy in OOD_VALIDATION_STRATEGIES:
                    evidence = _strategy_evidence(enriched, strategy)
                    fusion = FusionAgent().fuse(
                        evidence, reliability, final=True
                    )
                    verdict = fusion.verdict.value
                    risk = _fusion_risk(fusion)
                    covered = verdict in {"benign", "malicious"}
                    accumulators[strategy].add(
                        truth=truth,
                        verdict=verdict,
                        risk_score=risk,
                        agent_calls=len(report.participating_agents),
                        latency_ms=latency,
                    )
                    if application:
                        application_groups[strategy][application].add(
                            verdict,
                            risk,
                            len(report.participating_agents),
                        )
                    if category:
                        category_groups[strategy][category].add(
                            verdict,
                            risk,
                            len(report.participating_agents),
                        )
                    row.update(
                        {
                            f"{strategy}_verdict": verdict,
                            f"{strategy}_risk_score": risk,
                            f"{strategy}_confidence": fusion.confidence,
                            f"{strategy}_uncertainty": fusion.uncertainty,
                            f"{strategy}_coverage": covered,
                        }
                    )
                    if strategy == "off" and truth in {
                        "benign",
                        "malicious",
                    }:
                        forced = (
                            "malicious" if risk >= 0.5 else "benign"
                        )
                        off_binary_error = int(forced != truth)
                if truth in {"benign", "malicious"}:
                    error_labels.append(off_binary_error)
                audit_acc.add(audit)
                writer.writerow(row)

        for record in records:
            buffer.append(record)
            if len(buffer) >= batch_size:
                process(buffer)
                buffer = []
        process(buffer)

    metrics = {
        strategy: accumulator.metrics()
        for strategy, accumulator in accumulators.items()
    }
    payload = {
        "schema_version": "1.0",
        "strategies": metrics,
        "audit_summary": audit_acc.summary(),
        "shift_distribution": {
            "stats": _quantiles(stats_scores),
            "temporal": _quantiles(temporal_scores),
            "combined": _quantiles(combined_scores),
        },
    }
    _write_json(target / "metrics.json", payload)
    return ReplayResult(
        metrics=metrics,
        audit_summary=audit_acc.summary(),
        stats_shift_scores=stats_scores,
        temporal_shift_scores=temporal_scores,
        combined_shift_scores=combined_scores,
        error_labels=error_labels,
        application_groups=application_groups,
        category_groups=category_groups,
        accumulators=accumulators,
    )


def _load_replay_result(output_dir: str | Path) -> ReplayResult:
    target = Path(output_dir)
    payload = json.loads(
        (target / "metrics.json").read_text(encoding="utf-8")
    )
    accumulators = {
        strategy: MetricAccumulator(strategy)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    application_groups = {
        strategy: defaultdict(OODGroup)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    category_groups = {
        strategy: defaultdict(OODGroup)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    stats_scores: list[float] = []
    temporal_scores: list[float] = []
    combined_scores: list[float] = []
    error_labels: list[int] = []
    with (target / "predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            truth = row["true_label"]
            stats_scores.append(float(row["stats_shift_score"]))
            if row["temporal_shift_score"]:
                temporal_scores.append(
                    float(row["temporal_shift_score"])
                )
            combined_scores.append(float(row["combined_shift_score"]))
            application = row["application"]
            category = row["category"]
            for strategy in OOD_VALIDATION_STRATEGIES:
                verdict = row[f"{strategy}_verdict"]
                risk = float(row[f"{strategy}_risk_score"])
                calls = int(row["agent_calls"])
                latency = float(row["latency_ms"])
                accumulators[strategy].add(
                    truth=truth,
                    verdict=verdict,
                    risk_score=risk,
                    agent_calls=calls,
                    latency_ms=latency,
                )
                if application:
                    application_groups[strategy][application].add(
                        verdict, risk, calls
                    )
                if category:
                    category_groups[strategy][category].add(
                        verdict, risk, calls
                    )
            if truth in {"benign", "malicious"}:
                forced = (
                    "malicious"
                    if float(row["off_risk_score"]) >= 0.5
                    else "benign"
                )
                error_labels.append(int(forced != truth))
    return ReplayResult(
        metrics={
            strategy: accumulator.metrics()
            for strategy, accumulator in accumulators.items()
        },
        audit_summary=payload["audit_summary"],
        stats_shift_scores=stats_scores,
        temporal_shift_scores=temporal_scores,
        combined_shift_scores=combined_scores,
        error_labels=error_labels,
        application_groups=application_groups,
        category_groups=category_groups,
        accumulators=accumulators,
    )


def _evaluate_or_load(
    records: Iterable[FlowRecord],
    *,
    model_dir: str | Path,
    gate_dir: str | Path,
    output_dir: str | Path,
    expected_count: int,
    batch_size: int,
) -> ReplayResult:
    target = Path(output_dir)
    metrics_path = target / "metrics.json"
    predictions_path = target / "predictions.csv"
    if metrics_path.exists() and predictions_path.exists():
        try:
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
            if (
                metrics.get("strategies", {})
                .get("off", {})
                .get("sample_count")
                == expected_count
            ):
                return _load_replay_result(target)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
    return _evaluate_replay(
        records,
        model_dir=model_dir,
        gate_dir=gate_dir,
        output_dir=target,
        batch_size=batch_size,
    )


def _merge_accumulator(
    target: MetricAccumulator, source: MetricAccumulator
) -> None:
    for name in (
        "sample_count",
        "labeled_count",
        "covered_count",
        "unknown_count",
        "suspicious_count",
        "total_calls",
        "total_latency",
        "tp",
        "fp",
        "tn",
        "fn",
    ):
        setattr(target, name, getattr(target, name) + getattr(source, name))
    target.labels.extend(source.labels)
    target.scores.extend(source.scores)


def _write_group_metrics(
    path: Path,
    *,
    group_label: str,
    groups: dict[str, dict[str, OODGroup]],
) -> None:
    fields = [
        "schema_version",
        "strategy",
        group_label,
        "sample_count",
        "benign_rate",
        "malicious_rate",
        "suspicious_rate",
        "unknown_rate",
        "alert_rate",
        "average_risk_score",
        "average_agent_calls",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for strategy in OOD_VALIDATION_STRATEGIES:
            for name, group in sorted(groups[strategy].items()):
                writer.writerow(
                    {
                        "schema_version": "1.0",
                        "strategy": strategy,
                        group_label: name,
                        **group.row(),
                    }
                )


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _ood_detection_report(
    *,
    id_result: ReplayResult,
    ood_result: ReplayResult,
    strict_result: ReplayResult,
) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for name, id_scores, ood_scores in (
        (
            "stats",
            id_result.stats_shift_scores,
            ood_result.stats_shift_scores,
        ),
        (
            "temporal",
            id_result.temporal_shift_scores,
            ood_result.temporal_shift_scores,
        ),
        (
            "combined",
            id_result.combined_shift_scores,
            ood_result.combined_shift_scores,
        ),
    ):
        threshold = float(np.quantile(id_scores, 0.95))
        labels = [0] * len(id_scores) + [1] * len(ood_scores)
        scores = [*id_scores, *ood_scores]
        views[name] = {
            "AUROC": _auroc(labels, scores),
            "threshold_at_95_percent_id_retention": threshold,
            "observed_id_rejection_rate": float(
                np.mean(np.asarray(id_scores) > threshold)
            ),
            "OOD_TPR_at_95_percent_id_retention": float(
                np.mean(np.asarray(ood_scores) > threshold)
            ),
            "id_score_quantiles": _quantiles(id_scores),
            "ood_score_quantiles": _quantiles(ood_scores),
        }
    views["shift_error_identification"] = {
        "AUROC": _auroc(
            strict_result.error_labels,
            strict_result.combined_shift_scores,
        ),
        "error_count": int(sum(strict_result.error_labels)),
        "sample_count": len(strict_result.error_labels),
    }
    return {"schema_version": "1.0", "views": views}


def _train_fold_gates(
    *,
    dataset_dir: Path,
    base_run_dir: Path,
    output_dir: Path,
    folds: int,
    seed: int,
) -> None:
    cache = DomainFeatureCache.build(dataset_dir / "flows", seed=seed)
    split_manifest = base_run_dir / "domain-fold-manifest.json"
    for fold, (held_benign, held_malicious) in enumerate(
        DOMAIN_HOLDOUT_PAIRS[:folds]
    ):
        gate_root = output_dir / "domain_holdout" / "folds" / f"fold-{fold:02d}" / "gates"
        model_root = base_run_dir / "folds" / f"fold-{fold:02d}" / "models"
        for agent in ("stats", "temporal"):
            metadata_path = gate_root / agent / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                if (
                    metadata.get("split_manifest_sha256")
                    == _file_sha256(split_manifest)
                    and metadata.get("held_out_benign_application")
                    == held_benign
                    and metadata.get("held_out_malware_family")
                    == held_malicious
                ):
                    continue
            train, _, policy = cache.agents[agent].partitions(
                held_benign=held_benign,
                held_malicious=held_malicious,
            )
            train_ood_gate(
                agent=agent,
                train=train,
                policy=policy,
                output_dir=gate_root / agent,
                seed=seed,
                split_manifest_path=split_manifest,
                base_model_dir=model_root,
                extra_metadata={
                    "experiment": "domain_ood_v1",
                    "fold": fold,
                    "held_out_benign_application": held_benign,
                    "held_out_malware_family": held_malicious,
                    "external_dataset_used": False,
                },
            )


def _fold_summary(
    fold_metrics: list[dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    keys = (
        "accuracy",
        "macro_f1",
        "PR_AUC",
        "Brier_score",
        "ECE",
        "coverage",
        "unknown_rate",
        "suspicious_rate",
        "selective_error_rate",
        "benign_false_positive_rate",
        "malicious_detection_rate",
    )
    result: dict[str, Any] = {}
    for strategy in OOD_VALIDATION_STRATEGIES:
        result[strategy] = {}
        for key in keys:
            values = [
                item[strategy][key]
                for item in fold_metrics
                if item[strategy].get(key) is not None
            ]
            result[strategy][key] = {
                "mean": fmean(values) if values else None,
                "std": pstdev(values) if len(values) >= 2 else 0.0
                if values
                else None,
            }
    return result


def validate_ood_gating(
    dataset_dir: str | Path,
    base_run_dir: str | Path,
    full_model_dir: str | Path,
    full_gate_dir: str | Path,
    cesnet_dir: str | Path,
    output_dir: str | Path,
    *,
    folds: int = 10,
    seed: int = 42,
    batch_size: int = 2048,
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    base_run = Path(base_run_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _train_fold_gates(
        dataset_dir=dataset,
        base_run_dir=base_run,
        output_dir=target,
        folds=folds,
        seed=seed,
    )

    split_manifest = json.loads(
        (dataset / "splits" / "split-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    time_block = _evaluate_or_load(
        iter_split_records(
            dataset / "flows",
            dataset / "splits" / "split-manifest.json",
            "test",
        ),
        model_dir=full_model_dir,
        gate_dir=full_gate_dir,
        output_dir=target / "time_block",
        expected_count=len(split_manifest["assignments"]["test"]),
        batch_size=batch_size,
    )

    pooled = {
        strategy: MetricAccumulator(strategy)
        for strategy in OOD_VALIDATION_STRATEGIES
    }
    strict_stats: list[float] = []
    strict_temporal: list[float] = []
    strict_combined: list[float] = []
    strict_errors: list[int] = []
    fold_metrics: list[dict[str, dict[str, Any]]] = []
    strict_audit = AuditAccumulator()
    base_manifest = json.loads(
        (base_run / "domain-fold-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    fold_expected_counts = {
        int(item["fold"]): int(sum(item["test_counts"].values()))
        for item in base_manifest["folds"]
    }
    for fold, (held_benign, held_malicious) in enumerate(
        DOMAIN_HOLDOUT_PAIRS[:folds]
    ):
        fold_target = (
            target / "domain_holdout" / "folds" / f"fold-{fold:02d}"
        )
        result = _evaluate_or_load(
            iter_holdout_records(
                dataset / "flows",
                benign_application=held_benign,
                malware_family=held_malicious,
            ),
            model_dir=base_run
            / "folds"
            / f"fold-{fold:02d}"
            / "models",
            gate_dir=fold_target / "gates",
            output_dir=fold_target,
            expected_count=fold_expected_counts[fold],
            batch_size=batch_size,
        )
        fold_metrics.append(result.metrics)
        for strategy in OOD_VALIDATION_STRATEGIES:
            _merge_accumulator(
                pooled[strategy], result.accumulators[strategy]
            )
        strict_stats.extend(result.stats_shift_scores)
        strict_temporal.extend(result.temporal_shift_scores)
        strict_combined.extend(result.combined_shift_scores)
        strict_errors.extend(result.error_labels)
        audit = result.audit_summary
        strict_audit.sample_count += audit["sample_count"]
        strict_audit.completed_chains += audit["completed_chain_count"]
        strict_audit.blocked_violations += audit[
            "blocked_field_violation_count"
        ]
        strict_audit.policy_adjustments += audit[
            "policy_adjustment_count"
        ]
        strict_audit.event_counts.update(audit["event_counts"])
        strict_audit.actor_counts.update(audit["actor_counts"])

    strict_result = ReplayResult(
        metrics={
            strategy: accumulator.metrics()
            for strategy, accumulator in pooled.items()
        },
        audit_summary=strict_audit.summary(),
        stats_shift_scores=strict_stats,
        temporal_shift_scores=strict_temporal,
        combined_shift_scores=strict_combined,
        error_labels=strict_errors,
        application_groups={},
        category_groups={},
        accumulators=pooled,
    )
    strict_payload = {
        "schema_version": "1.0",
        "fold_count": folds,
        "pooled": strict_result.metrics,
        "fold_summary": _fold_summary(fold_metrics),
        "folds": fold_metrics,
        "audit_summary": strict_result.audit_summary,
    }
    strict_aggregate = target / "domain_holdout" / "aggregate"
    strict_aggregate.mkdir(parents=True, exist_ok=True)
    _write_json(strict_aggregate / "metrics.json", strict_payload)

    cesnet = Path(cesnet_dir)
    cesnet_manifest = json.loads(
        (cesnet / "splits" / "split-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_ood_count = int(
        cesnet_manifest.get("counts", {}).get(
            "test",
            499_997,
        )
    )
    ood = _evaluate_or_load(
        iter_split_records(
            cesnet / "flows",
            cesnet / "splits" / "split-manifest.json",
            "test",
        ),
        model_dir=full_model_dir,
        gate_dir=full_gate_dir,
        output_dir=target / "cesnet",
        expected_count=expected_ood_count,
        batch_size=batch_size,
    )
    _write_group_metrics(
        target / "cesnet" / "application_metrics.csv",
        group_label="application",
        groups=ood.application_groups,
    )
    _write_group_metrics(
        target / "cesnet" / "category_metrics.csv",
        group_label="category",
        groups=ood.category_groups,
    )

    ood_detection = _ood_detection_report(
        id_result=time_block,
        ood_result=ood,
        strict_result=strict_result,
    )
    _write_json(target / "ood_detection.json", ood_detection)

    hybrid_strict = strict_result.metrics["hybrid"]
    hybrid_ood = ood.metrics["hybrid"]
    ood_alert_rate = (
        hybrid_ood["suspicious_rate"]
        + _safe_div(
            hybrid_ood["confusion_on_covered"]["tp"]
            + hybrid_ood["confusion_on_covered"]["fp"],
            hybrid_ood["sample_count"],
        )
    )
    # CESNET is unlabeled, so malicious verdict rate is obtained directly
    # from its prediction file rather than confusion counts.
    malicious_count = 0
    with (target / "cesnet" / "predictions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            malicious_count += int(
                row["hybrid_verdict"] == "malicious"
            )
    malicious_rate = _safe_div(
        malicious_count, hybrid_ood["sample_count"]
    )
    ood_alert_rate = malicious_rate + hybrid_ood["suspicious_rate"]
    acceptance = {
        "strict_macro_f1_at_least_0_95": (
            hybrid_strict["macro_f1"] is not None
            and hybrid_strict["macro_f1"] >= 0.95
        ),
        "strict_coverage_at_least_0_65": (
            hybrid_strict["coverage"] >= 0.65
        ),
        "strict_selective_error_at_most_0_05": (
            hybrid_strict["selective_error_rate"] is not None
            and hybrid_strict["selective_error_rate"] <= 0.05
        ),
        "combined_ood_auroc_at_least_0_97": (
            ood_detection["views"]["combined"]["AUROC"] is not None
            and ood_detection["views"]["combined"]["AUROC"] >= 0.97
        ),
        "cesnet_malicious_rate_at_most_0_10": malicious_rate <= 0.10,
        "cesnet_alert_rate_at_most_0_40": ood_alert_rate <= 0.40,
        "cesnet_unknown_rate_at_least_0_50": (
            hybrid_ood["unknown_rate"] >= 0.50
        ),
    }
    summary = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategies": list(OOD_VALIDATION_STRATEGIES),
        "time_block": time_block.metrics,
        "domain_holdout": strict_result.metrics,
        "cesnet": ood.metrics,
        "ood_detection": ood_detection,
        "acceptance": {
            **acceptance,
            "all_targets_met": all(acceptance.values()),
            "cesnet_hybrid_malicious_rate": malicious_rate,
            "cesnet_hybrid_alert_rate": ood_alert_rate,
        },
        "external_dataset_used_for_training_or_thresholds": False,
    }
    _write_json(target / "validation_summary.json", summary)
    return summary
