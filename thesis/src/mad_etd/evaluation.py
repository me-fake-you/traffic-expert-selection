from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .engine import DetectionEngine, build_default_engine
from .schemas import FlowRecord, FutureFeatureFlags, Verdict


EVALUATION_MODES = (
    "full_pipeline",
    "fixed_all_agents",
    "rule_coordinator",
    "nvidia_coordinator",
    "no_field_audit",
    "no_pseudo_guard",
    "no_reliability_discount",
    "no_reject",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_summary(
    detector_backend: str, model_dir: str | Path | None
) -> dict[str, Any] | None:
    if detector_backend not in {
        "learned",
        "deep_v2",
        "deep_v2_all_length",
        "deep_v2_2",
        "deep_v2_3",
        "deep_v2_6_continuous",
        "deep_v3_1_prefix",
    } or model_dir is None:
        return None
    root = Path(model_dir)
    result: dict[str, Any] = {}
    agents = ("stats", "temporal") if detector_backend == "learned" else (
        "stats",
        "temporal",
        "tls",
    )
    for agent in agents:
        metadata_path = root / agent / "metadata.json"
        model_path = root / agent / (
            "model.joblib"
            if detector_backend == "learned" or agent == "stats"
            else "model.pt"
        )
        if detector_backend in {"deep_v2_2", "deep_v2_3"} and agent == "temporal":
            metadata_path = root / agent / "metadata.json"
            model_path = root / agent / "short" / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            result[agent] = {"status": "not_available"}
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result[agent] = {
            "artifact_schema_version": metadata.get("schema_version"),
            "feature_schema_version": metadata.get("feature_schema_version"),
            "selected_candidate": metadata.get("selected_candidate"),
            "selected_family": metadata.get("selected_family"),
            "model_sha256": _sha256_file(model_path),
            "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        }
    return result


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not labels or not positives:
        return None
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _expected_calibration_error(
    labels: list[int], scores: list[float], bins: int = 10
) -> float | None:
    if not labels:
        return None
    total = len(labels)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (label, score)
            for label, score in zip(labels, scores, strict=True)
            if low <= score < high or (index == bins - 1 and score == 1)
        ]
        if not members:
            continue
        accuracy = fmean(label for label, _ in members)
        confidence = fmean(score for _, score in members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def build_engine_for_mode(
    mode: str,
    *,
    nvidia_api_key: str | None = None,
    nvidia_model: str | None = None,
    cache_dir: str | Path | None = None,
    detector_backend: str = "rule",
    model_dir: str | Path | None = None,
    ood_policy: str = "off",
    ood_gate_dir: str | Path | None = None,
    base_ood_policy: str = "off",
    base_ood_gate_dir: str | Path | None = None,
    enable_evidence_utility_v2: bool = False,
    enable_evidence_utility_v2_1: bool = False,
    utility_model_dir: str | Path | None = None,
    evidence_stability_policy: str = "off",
    orbit_reliability_dir: str | Path | None = None,
) -> DetectionEngine:
    if mode not in EVALUATION_MODES:
        raise ValueError(f"unsupported evaluation mode: {mode}")

    use_nvidia = mode in {
        "full_pipeline",
        "nvidia_coordinator",
        "no_field_audit",
        "no_pseudo_guard",
        "no_reliability_discount",
        "no_reject",
    }
    return build_default_engine(
        field_audit_mode="legacy",
        use_nvidia=use_nvidia,
        nvidia_api_key=nvidia_api_key,
        nvidia_model=nvidia_model,
        cache_dir=cache_dir,
        enable_field_audit=mode != "no_field_audit",
        enable_pseudo_guard=mode != "no_pseudo_guard",
        use_reliability_discount=mode != "no_reliability_discount",
        allow_reject=mode != "no_reject",
        execution_policy="fixed_all" if mode == "fixed_all_agents" else "dynamic",
        force_rule_coordinator=mode
        in {"rule_coordinator", "fixed_all_agents"},
        detector_backend=detector_backend,
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
        base_ood_policy=base_ood_policy,
        base_ood_gate_dir=base_ood_gate_dir,
        future_flags=FutureFeatureFlags(
            planner_executor=(
                enable_evidence_utility_v2
                or enable_evidence_utility_v2_1
            ),
            evidence_utility_v2=enable_evidence_utility_v2,
            evidence_utility_v2_1=enable_evidence_utility_v2_1,
        ),
        utility_model_dir=utility_model_dir,
        evidence_stability_policy=evidence_stability_policy,
        orbit_reliability_dir=orbit_reliability_dir,
    )


def evaluate_records(
    engine: DetectionEngine,
    records: list[FlowRecord],
    *,
    mode: str = "custom",
) -> dict[str, Any]:
    if engine.reporter.knowledge_service is not None:
        raise ValueError(
            "RAG must be disabled for classification and OOD metric evaluation"
        )
    rows: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    actor_counts: Counter[str] = Counter()
    blocked_field_violations = 0
    completed_chains = 0
    total_policy_adjustments = 0
    llm_call_count = 0
    llm_fallback_count = 0
    llm_replay_count = 0

    for record in records:
        started = time.perf_counter()
        report, audit = engine.analyze(record)
        latency_ms = (time.perf_counter() - started) * 1000
        truth_value = record.labels.get("binary")
        label_available = str(truth_value).lower() in {"benign", "malicious"}
        truth = str(truth_value).lower() if label_available else ""
        prediction = report.verdict.value
        covered = report.verdict in {Verdict.BENIGN, Verdict.MALICIOUS}
        mass_total = report.benign_support + report.malicious_support
        risk_score = (
            report.malicious_support / mass_total if mass_total else 0.5
        )
        sample_llm_calls = sum(
            bool(decision.planner_metadata.get("llm_call_attempted"))
            for decision in report.coordinator_decisions
        )
        sample_llm_fallbacks = sum(
            decision.source == "fallback"
            and bool(decision.planner_metadata.get("llm_call_attempted"))
            for decision in report.coordinator_decisions
        )
        sample_llm_replays = sum(
            decision.source == "replay"
            for decision in report.coordinator_decisions
        )
        llm_call_count += sample_llm_calls
        llm_fallback_count += sample_llm_fallbacks
        llm_replay_count += sample_llm_replays

        for event in audit.events:
            event_counts[event.event_type] += 1
            actor_counts[event.actor] += 1
            if event.event_type == "AGENT_EVIDENCE":
                blocked_field_violations += len(
                    event.input_summary.get("blocked_field_intersection", [])
                )
        chain_complete = bool(audit.events) and audit.events[-1].event_type == (
            "REPORT_GENERATED"
        )
        completed_chains += int(chain_complete)
        total_policy_adjustments += sum(
            len(decision.policy_adjustments)
            for decision in report.coordinator_decisions
        )

        rows.append(
            {
                "schema_version": "1.0",
                "trace_id": record.trace_id,
                "sample_id": record.sample_id,
                "label_available": label_available,
                "true_label": truth,
                "verdict": prediction,
                "covered": covered,
                "confidence": report.confidence,
                "uncertainty": report.uncertainty,
                "conflict_score": report.conflict_score,
                "benign_support": report.benign_support,
                "malicious_support": report.malicious_support,
                "distribution_shift_score": report.distribution_shift_score,
                "risk_score": risk_score,
                "severity": report.severity,
                "need_escalation": report.need_escalation,
                "family": report.family,
                "intent": report.intent,
                "agent_calls": len(report.participating_agents),
                "coordinator_rounds": len(report.coordinator_decisions),
                "latency_ms": latency_ms,
                "llm_call_count": sample_llm_calls,
                "llm_fallback_count": sample_llm_fallbacks,
                "audit_event_count": len(audit.events),
                "audit_complete": chain_complete,
            }
        )

    metrics = _compute_metrics(rows, mode=mode)
    metrics["llm_call_count"] = llm_call_count
    audit_summary = {
        "schema_version": "1.0",
        "mode": mode,
        "sample_count": len(rows),
        "completed_chain_count": completed_chains,
        "audit_chain_completion_rate": _safe_div(completed_chains, len(rows)),
        "event_counts": dict(event_counts),
        "actor_counts": dict(actor_counts),
        "blocked_field_violation_count": blocked_field_violations,
        "policy_adjustment_count": total_policy_adjustments,
        "required_event_presence": {
            "field_policy": bool(
                event_counts["FIELD_POLICY_APPLIED"]
                or event_counts["FIELD_POLICY_BYPASSED"]
            ),
            "reliability": bool(
                event_counts["RELIABILITY_ASSESSED"]
                or event_counts["RELIABILITY_BYPASSED"]
            ),
            "routing": event_counts["ROUTING_DECISION"] > 0,
            "policy_guard": event_counts["POLICY_GUARD_EVALUATION"] > 0,
            "agent_evidence": event_counts["AGENT_EVIDENCE"] > 0,
            "final_fusion": event_counts["FINAL_FUSION"] > 0,
            "reporter": event_counts["REPORT_GENERATED"] > 0,
        },
    }
    cost_summary = {
        "schema_version": "1.0",
        "mode": mode,
        "sample_count": len(rows),
        "total_agent_calls": sum(row["agent_calls"] for row in rows),
        "average_agent_calls": metrics["average_agent_calls"],
        "total_latency_ms": sum(row["latency_ms"] for row in rows),
        "average_latency_ms": metrics["average_latency_ms"],
        "llm_call_count": llm_call_count,
        "llm_fallback_count": llm_fallback_count,
        "llm_replay_count": llm_replay_count,
    }
    return {
        "schema_version": "1.0",
        "mode": mode,
        "metrics": metrics,
        "audit_summary": audit_summary,
        "cost_summary": cost_summary,
        "rows": rows,
    }


def _compute_metrics(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    covered_rows = [
        row for row in rows if row["covered"] and row["label_available"]
    ]
    labeled_rows = [row for row in rows if row["label_available"]]
    tp = sum(
        row["true_label"] == "malicious" and row["verdict"] == "malicious"
        for row in covered_rows
    )
    fp = sum(
        row["true_label"] == "benign" and row["verdict"] == "malicious"
        for row in covered_rows
    )
    tn = sum(
        row["true_label"] == "benign" and row["verdict"] == "benign"
        for row in covered_rows
    )
    fn = sum(
        row["true_label"] == "malicious" and row["verdict"] == "benign"
        for row in covered_rows
    )

    if labeled_rows:
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        malicious_f1 = _safe_div(2 * precision * recall, precision + recall)
        benign_precision = _safe_div(tn, tn + fn)
        benign_recall = _safe_div(tn, tn + fp)
        benign_f1 = _safe_div(
            2 * benign_precision * benign_recall,
            benign_precision + benign_recall,
        )
        accuracy = _safe_div(tp + tn, len(covered_rows))
        macro_f1 = (malicious_f1 + benign_f1) / 2
        labels = [
            int(row["true_label"] == "malicious") for row in labeled_rows
        ]
        scores = [row["risk_score"] for row in labeled_rows]
        brier = fmean(
            (score - label) ** 2
            for score, label in zip(scores, labels, strict=True)
        )
        status = "computed"
    else:
        precision = recall = accuracy = macro_f1 = brier = None
        labels = []
        scores = []
        status = "skipped_no_labels"

    return {
        "schema_version": "1.0",
        "mode": mode,
        "metrics_status": status,
        "classification_metric_scope": "covered_labeled_samples",
        "ranking_metric_scope": "all_labeled_samples",
        "sample_count": len(rows),
        "labeled_sample_count": len(labeled_rows),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "PR_AUC": _average_precision(labels, scores),
        "Brier_score": brier,
        "ECE": _expected_calibration_error(labels, scores),
        "coverage": _safe_div(
            sum(row["covered"] for row in rows), len(rows)
        ),
        "unknown_rate": _safe_div(
            sum(row["verdict"] == "unknown" for row in rows), len(rows)
        ),
        "suspicious_rate": _safe_div(
            sum(row["verdict"] == "suspicious" for row in rows), len(rows)
        ),
        "average_agent_calls": (
            fmean(row["agent_calls"] for row in rows) if rows else 0.0
        ),
        "average_latency_ms": (
            fmean(row["latency_ms"] for row in rows) if rows else 0.0
        ),
        "llm_call_count": 0,
        "confusion_on_covered": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_to_directory(
    records: Iterable[FlowRecord],
    *,
    input_path: str | Path,
    mode: str,
    output_dir: str | Path,
    nvidia_api_key: str | None = None,
    nvidia_model: str | None = None,
    detector_backend: str = "rule",
    model_dir: str | Path | None = None,
    ood_policy: str = "off",
    ood_gate_dir: str | Path | None = None,
    base_ood_policy: str = "off",
    base_ood_gate_dir: str | Path | None = None,
    enable_evidence_utility_v2: bool = False,
    enable_evidence_utility_v2_1: bool = False,
    utility_model_dir: str | Path | None = None,
    evidence_stability_policy: str = "off",
    orbit_reliability_dir: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    engine = build_engine_for_mode(
        mode,
        nvidia_api_key=nvidia_api_key,
        nvidia_model=nvidia_model,
        cache_dir=target / "coordinator-cache",
        detector_backend=detector_backend,
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
        base_ood_policy=base_ood_policy,
        base_ood_gate_dir=base_ood_gate_dir,
        enable_evidence_utility_v2=enable_evidence_utility_v2,
        enable_evidence_utility_v2_1=enable_evidence_utility_v2_1,
        utility_model_dir=utility_model_dir,
        evidence_stability_policy=evidence_stability_policy,
        orbit_reliability_dir=orbit_reliability_dir,
    )
    predictions_path = target / "predictions.csv"
    fieldnames = _prediction_fieldnames()
    accumulator = _StreamingAccumulator(mode)
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row, audit = _analyze_record(engine, record)
            writer.writerow(row)
            accumulator.add(row, audit)
    result = accumulator.result()
    run_config = {
        "schema_version": "1.0",
        "package_version": "0.1.0",
        "mode": mode,
        "input_path": str(Path(input_path).resolve()),
        "sample_count": result["metrics"]["sample_count"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "nvidia_model": nvidia_model
        or os.getenv("NVIDIA_MODEL")
        or "meta/llama-3.1-8b-instruct",
        "nvidia_api_key_configured": bool(
            nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        ),
        "field_audit_enabled": mode != "no_field_audit",
        "pseudo_guard_enabled": mode != "no_pseudo_guard",
        "reliability_discount_enabled": mode != "no_reliability_discount",
        "reject_option_enabled": mode != "no_reject",
        "execution_policy": (
            "fixed_all" if mode == "fixed_all_agents" else "dynamic"
        ),
        "coordinator_backend": (
            "rule"
            if mode in {"rule_coordinator", "fixed_all_agents"}
            else "nvidia_with_rule_fallback"
        ),
        "detector_backend": detector_backend,
        "model_dir": (
            str(Path(model_dir).resolve()) if model_dir is not None else None
        ),
        "model_artifacts": _model_artifact_summary(
            detector_backend, model_dir
        ),
        "ood_policy": ood_policy,
        "ood_gate_dir": (
            str(Path(ood_gate_dir).resolve())
            if ood_gate_dir is not None
            else None
        ),
        "base_ood_policy": base_ood_policy,
        "base_ood_gate_dir": (
            str(Path(base_ood_gate_dir).resolve())
            if base_ood_gate_dir is not None
            else None
        ),
        "evidence_utility_v2": enable_evidence_utility_v2,
        "evidence_utility_v2_1": enable_evidence_utility_v2_1,
        "utility_model_dir": (
            str(Path(utility_model_dir).resolve())
            if utility_model_dir is not None
            else None
        ),
        "evidence_stability_policy": evidence_stability_policy,
        "orbit_reliability_dir": (
            str(Path(orbit_reliability_dir).resolve())
            if orbit_reliability_dir is not None
            else None
        ),
        "rag_enabled": False,
        "knowledge_base_dir": None,
    }
    _write_json(target / "metrics.json", result["metrics"])
    _write_json(target / "run_config.json", run_config)
    _write_json(target / "audit_summary.json", result["audit_summary"])
    _write_json(target / "cost_summary.json", result["cost_summary"])
    return {**result, "run_config": run_config}


def _analyze_record(
    engine: DetectionEngine, record: FlowRecord
) -> tuple[dict[str, Any], Any]:
    started = time.perf_counter()
    report, audit = engine.analyze(record)
    latency_ms = (time.perf_counter() - started) * 1000
    truth_value = record.labels.get("binary")
    label_available = str(truth_value).lower() in {"benign", "malicious"}
    truth = str(truth_value).lower() if label_available else ""
    covered = report.verdict in {Verdict.BENIGN, Verdict.MALICIOUS}
    mass_total = report.benign_support + report.malicious_support
    risk_score = report.malicious_support / mass_total if mass_total else 0.5
    sample_llm_calls = sum(
        bool(decision.planner_metadata.get("llm_call_attempted"))
        for decision in report.coordinator_decisions
    )
    sample_llm_fallbacks = sum(
        decision.source == "fallback"
        and bool(decision.planner_metadata.get("llm_call_attempted"))
        for decision in report.coordinator_decisions
    )
    return (
        {
            "schema_version": "1.0",
            "trace_id": record.trace_id,
            "sample_id": record.sample_id,
            "label_available": label_available,
            "true_label": truth,
            "verdict": report.verdict.value,
            "covered": covered,
            "confidence": report.confidence,
            "uncertainty": report.uncertainty,
            "conflict_score": report.conflict_score,
            "benign_support": report.benign_support,
            "malicious_support": report.malicious_support,
            "distribution_shift_score": report.distribution_shift_score,
            "risk_score": risk_score,
            "severity": report.severity,
            "need_escalation": report.need_escalation,
            "family": report.family,
            "intent": report.intent,
            "agent_calls": len(report.participating_agents),
            "coordinator_rounds": len(report.coordinator_decisions),
            "latency_ms": latency_ms,
            "llm_call_count": sample_llm_calls,
            "llm_fallback_count": sample_llm_fallbacks,
            "audit_event_count": len(audit.events),
            "audit_complete": bool(audit.events)
            and audit.events[-1].event_type == "REPORT_GENERATED",
        },
        audit,
    )


class _StreamingAccumulator:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.sample_count = 0
        self.labeled_count = 0
        self.covered_count = 0
        self.unknown_count = 0
        self.suspicious_count = 0
        self.total_calls = 0
        self.total_latency = 0.0
        self.llm_calls = 0
        self.llm_fallbacks = 0
        self.llm_replays = 0
        self.tp = self.fp = self.tn = self.fn = 0
        self.labels: list[int] = []
        self.scores: list[float] = []
        self.event_counts: Counter[str] = Counter()
        self.actor_counts: Counter[str] = Counter()
        self.blocked_violations = 0
        self.completed_chains = 0
        self.policy_adjustments = 0

    def add(self, row: dict[str, Any], audit: Any) -> None:
        self.sample_count += 1
        self.covered_count += int(bool(row["covered"]))
        self.unknown_count += int(row["verdict"] == "unknown")
        self.suspicious_count += int(row["verdict"] == "suspicious")
        self.total_calls += int(row["agent_calls"])
        self.total_latency += float(row["latency_ms"])
        self.llm_calls += int(row["llm_call_count"])
        self.llm_fallbacks += int(row["llm_fallback_count"])
        self.completed_chains += int(bool(row["audit_complete"]))
        if row["label_available"]:
            self.labeled_count += 1
            label = int(row["true_label"] == "malicious")
            self.labels.append(label)
            self.scores.append(float(row["risk_score"]))
            if row["covered"]:
                if label and row["verdict"] == "malicious":
                    self.tp += 1
                elif not label and row["verdict"] == "malicious":
                    self.fp += 1
                elif not label and row["verdict"] == "benign":
                    self.tn += 1
                elif label and row["verdict"] == "benign":
                    self.fn += 1
        for event in audit.events:
            self.event_counts[event.event_type] += 1
            self.actor_counts[event.actor] += 1
            if event.event_type == "AGENT_EVIDENCE":
                self.blocked_violations += len(
                    event.input_summary.get("blocked_field_intersection", [])
                )
            if event.event_type == "ROUTING_DECISION":
                output = event.output_summary
                self.policy_adjustments += len(output.get("policy_adjustments", []))
                self.llm_replays += int(output.get("source") == "replay")

    def result(self) -> dict[str, Any]:
        covered_labeled = self.tp + self.fp + self.tn + self.fn
        if self.labeled_count:
            precision = _safe_div(self.tp, self.tp + self.fp)
            recall = _safe_div(self.tp, self.tp + self.fn)
            malicious_f1 = _safe_div(2 * precision * recall, precision + recall)
            benign_precision = _safe_div(self.tn, self.tn + self.fn)
            benign_recall = _safe_div(self.tn, self.tn + self.fp)
            benign_f1 = _safe_div(
                2 * benign_precision * benign_recall,
                benign_precision + benign_recall,
            )
            metrics_status = "computed"
            accuracy = _safe_div(self.tp + self.tn, covered_labeled)
            macro_f1 = (malicious_f1 + benign_f1) / 2
            brier = fmean(
                (score - label) ** 2
                for score, label in zip(self.scores, self.labels, strict=True)
            )
        else:
            precision = recall = accuracy = macro_f1 = brier = None
            metrics_status = "skipped_no_labels"
        metrics = {
            "schema_version": "1.0",
            "mode": self.mode,
            "metrics_status": metrics_status,
            "classification_metric_scope": "covered_labeled_samples",
            "ranking_metric_scope": "all_labeled_samples",
            "sample_count": self.sample_count,
            "labeled_sample_count": self.labeled_count,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": precision,
            "recall": recall,
            "PR_AUC": _average_precision(self.labels, self.scores),
            "Brier_score": brier,
            "ECE": _expected_calibration_error(self.labels, self.scores),
            "coverage": _safe_div(self.covered_count, self.sample_count),
            "unknown_rate": _safe_div(self.unknown_count, self.sample_count),
            "suspicious_rate": _safe_div(
                self.suspicious_count, self.sample_count
            ),
            "average_agent_calls": _safe_div(
                self.total_calls, self.sample_count
            ),
            "average_latency_ms": _safe_div(
                self.total_latency, self.sample_count
            ),
            "llm_call_count": self.llm_calls,
            "confusion_on_covered": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
        }
        audit_summary = {
            "schema_version": "1.0",
            "mode": self.mode,
            "sample_count": self.sample_count,
            "completed_chain_count": self.completed_chains,
            "audit_chain_completion_rate": _safe_div(
                self.completed_chains, self.sample_count
            ),
            "event_counts": dict(self.event_counts),
            "actor_counts": dict(self.actor_counts),
            "blocked_field_violation_count": self.blocked_violations,
            "policy_adjustment_count": self.policy_adjustments,
            "required_event_presence": {
                "field_policy": bool(
                    self.event_counts["FIELD_POLICY_APPLIED"]
                    or self.event_counts["FIELD_POLICY_BYPASSED"]
                ),
                "reliability": bool(
                    self.event_counts["RELIABILITY_ASSESSED"]
                    or self.event_counts["RELIABILITY_BYPASSED"]
                ),
                "routing": self.event_counts["ROUTING_DECISION"] > 0,
                "policy_guard": self.event_counts["POLICY_GUARD_EVALUATION"] > 0,
                "agent_evidence": self.event_counts["AGENT_EVIDENCE"] > 0,
                "final_fusion": self.event_counts["FINAL_FUSION"] > 0,
                "reporter": self.event_counts["REPORT_GENERATED"] > 0,
            },
        }
        cost_summary = {
            "schema_version": "1.0",
            "mode": self.mode,
            "sample_count": self.sample_count,
            "total_agent_calls": self.total_calls,
            "average_agent_calls": metrics["average_agent_calls"],
            "total_latency_ms": self.total_latency,
            "average_latency_ms": metrics["average_latency_ms"],
            "llm_call_count": self.llm_calls,
            "llm_fallback_count": self.llm_fallbacks,
            "llm_replay_count": self.llm_replays,
        }
        return {
            "schema_version": "1.0",
            "mode": self.mode,
            "metrics": metrics,
            "audit_summary": audit_summary,
            "cost_summary": cost_summary,
            "rows": [],
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _prediction_fieldnames()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_fieldnames() -> list[str]:
    return [
        "schema_version",
        "trace_id",
        "sample_id",
        "label_available",
        "true_label",
        "verdict",
        "covered",
        "confidence",
        "uncertainty",
        "conflict_score",
        "benign_support",
        "malicious_support",
        "distribution_shift_score",
        "risk_score",
        "severity",
        "need_escalation",
        "family",
        "intent",
        "agent_calls",
        "coordinator_rounds",
        "latency_ms",
        "llm_call_count",
        "llm_fallback_count",
        "audit_event_count",
        "audit_complete",
    ]
