from __future__ import annotations

import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .engine import DetectionEngine, build_default_engine
from .io import iter_split_records
from .perturb import apply_perturbation
from .schemas import DetectorInput, FlowRecord, SequenceFeatures, Verdict
from .short_flow import DeepV22TemporalModel
from .v2_evaluation import verify_frozen_v1_reference
from .v22_training import (
    audit_sequence_regimes,
    calibrate_conformal_v32,
    train_temporal_short_v22,
)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def freeze_v22_protocol_config(
    path: str | Path,
    config: dict[str, Any],
) -> None:
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "v2.2 resume configuration differs from the frozen run"
            )
        return
    _dump(target, config)


def _deterministic_subset(
    records: Iterable[FlowRecord],
    limit: int,
) -> list[FlowRecord]:
    heap: list[tuple[int, str, FlowRecord]] = []
    for record in records:
        score = int(
            hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:16],
            16,
        )
        item = (-score, record.sample_id, record)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    return [
        record
        for _, _, record in sorted(heap, key=lambda item: item[1])
    ]


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    covered = [row for row in rows if row["covered"]]
    tp = sum(
        row["truth"] == "malicious" and row["verdict"] == "malicious"
        for row in covered
    )
    fp = sum(
        row["truth"] == "benign" and row["verdict"] == "malicious"
        for row in covered
    )
    tn = sum(
        row["truth"] == "benign" and row["verdict"] == "benign"
        for row in covered
    )
    fn = sum(
        row["truth"] == "malicious" and row["verdict"] == "benign"
        for row in covered
    )
    malicious_precision = tp / (tp + fp) if tp + fp else 0.0
    malicious_recall = tp / (tp + fn) if tp + fn else 0.0
    malicious_f1 = (
        2
        * malicious_precision
        * malicious_recall
        / (malicious_precision + malicious_recall)
        if malicious_precision + malicious_recall
        else 0.0
    )
    benign_precision = tn / (tn + fn) if tn + fn else 0.0
    benign_recall = tn / (tn + fp) if tn + fp else 0.0
    benign_f1 = (
        2
        * benign_precision
        * benign_recall
        / (benign_precision + benign_recall)
        if benign_precision + benign_recall
        else 0.0
    )
    return (malicious_f1 + benign_f1) / 2


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    covered = [row for row in rows if row["covered"]]
    errors = sum(not row["correct"] for row in covered)
    return {
        "sample_count": count,
        "coverage": len(covered) / count if count else 0.0,
        "selective_error": (
            errors / len(covered) if covered else 1.0
        ),
        "selective_macro_f1": _macro_f1(rows),
        "unknown_rate": (
            sum(row["verdict"] == "unknown" for row in rows) / count
            if count
            else 0.0
        ),
        "suspicious_rate": (
            sum(row["verdict"] == "suspicious" for row in rows) / count
            if count
            else 0.0
        ),
        "average_agent_calls": (
            fmean(row["agent_calls"] for row in rows) if rows else 0.0
        ),
    }


def _evaluate_engine(
    engine: DetectionEngine,
    records: list[FlowRecord],
    output_dir: Path,
    *,
    mode_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    audit_complete = 0
    blocked = 0
    fusion_ownership = 0
    ood_override = 0
    for record in records:
        report, audit = engine.analyze(record)
        truth = str(record.labels.get("binary", "")).lower()
        covered = report.verdict in {Verdict.BENIGN, Verdict.MALICIOUS}
        packet_count = len(record.sequence.packet_lengths)
        bucket = str(packet_count) if packet_count < 4 else "4+"
        rows.append(
            {
                "sample_id": record.sample_id,
                "truth": truth,
                "packet_bucket": bucket,
                "verdict": report.verdict.value,
                "covered": covered,
                "correct": covered and report.verdict.value == truth,
                "uncertainty": report.uncertainty,
                "agent_calls": len(report.participating_agents),
            }
        )
        audit_complete += int(
            bool(audit.events)
            and audit.events[-1].event_type == "REPORT_GENERATED"
        )
        for event in audit.events:
            if event.event_type == "AGENT_EVIDENCE":
                blocked += len(
                    event.input_summary.get(
                        "blocked_field_intersection",
                        [],
                    )
                )
            if (
                event.event_type == "FINAL_FUSION"
                and event.actor != "FusionAgent"
            ):
                fusion_ownership += 1
            if event.event_type == "FINAL_FUSION":
                reasons = event.output_summary.get("reasons", [])
                ood_override += int(
                    any("override" in str(item).lower() for item in reasons)
                )
    by_bucket: dict[str, Any] = {}
    for bucket in ("1", "2", "3", "4+"):
        selected = [
            row for row in rows if row["packet_bucket"] == bucket
        ]
        by_bucket[bucket] = _summarize_rows(selected)
    result = {
        "schema_version": "1.0",
        "mode": mode_name,
        "overall": _summarize_rows(rows),
        "by_packet_bucket": by_bucket,
        "audit_completion": audit_complete / len(rows) if rows else 0.0,
        "blocked_field_violation_count": blocked,
        "fusion_ownership_violation_count": fusion_ownership,
        "ood_override_count": ood_override,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    _dump(output_dir / "metrics.json", result)
    return result


def _build_engine(
    *,
    backend: str,
    model_dir: Path,
    ood_policy: str,
    ood_gate_dir: Path,
    base_ood_gate_dir: Path,
) -> DetectionEngine:
    return build_default_engine(
        field_audit_mode="legacy",
        force_rule_coordinator=True,
        detector_backend=backend,
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
        base_ood_policy="hybrid",
        base_ood_gate_dir=base_ood_gate_dir,
        max_workers=1,
    )


def _boundary_consistency(
    records: list[FlowRecord],
    *,
    model_dir: Path,
    limit: int = 1000,
) -> dict[str, Any]:
    candidates = [record for record in records if len(record.sequence.packet_lengths) == 4]
    selected = _deterministic_subset(candidates, min(limit, len(candidates)))
    model = DeepV22TemporalModel(
        model_dir / "temporal",
        ood_policy="off",
    )
    agreements = 0
    probability_deltas: list[float] = []
    for record in selected:
        full_input = DetectorInput(
            sequence=record.sequence,
        )
        shortened = record.sequence.model_copy(deep=True)
        shortened.packet_lengths = shortened.packet_lengths[:3]
        shortened.directions = shortened.directions[:3]
        shortened.iats = shortened.iats[:2]
        shortened.original_packet_count = 4
        shortened.truncated = True
        short_prediction = model.short.predict(
            DetectorInput(sequence=shortened)
        )
        long_prediction = model.long.predict(full_input)
        agreements += int(
            (short_prediction.malicious_probability >= 0.5)
            == (long_prediction.malicious_probability >= 0.5)
        )
        probability_deltas.append(
            abs(
                short_prediction.malicious_probability
                - long_prediction.malicious_probability
            )
        )
    return {
        "sample_count": len(selected),
        "binary_probability_agreement": (
            agreements / len(selected) if selected else 0.0
        ),
        "mean_absolute_probability_delta": (
            fmean(probability_deltas) if probability_deltas else None
        ),
    }


def _perturbation_stability(
    records: list[FlowRecord],
    *,
    engines: dict[str, DetectionEngine],
    limit: int = 200,
) -> dict[str, Any]:
    eligible = [record for record in records if record.sequence.packet_lengths]
    selected = _deterministic_subset(eligible, min(limit, len(eligible)))
    kinds = (
        "padding",
        "dummy_packet",
        "iat_jitter",
        "sequence_truncation",
    )
    result: dict[str, Any] = {}
    for name, engine in engines.items():
        stable = 0
        total = 0
        by_kind: dict[str, float] = {}
        clean_verdicts = {
            record.sample_id: engine.analyze(record)[0].verdict.value
            for record in selected
        }
        for kind in kinds:
            kind_stable = 0
            for record in selected:
                perturbed = apply_perturbation(
                    record,
                    kind,
                    strength=0.2,
                    seed=42,
                )
                verdict = engine.analyze(perturbed)[0].verdict.value
                agreement = verdict == clean_verdicts[record.sample_id]
                stable += int(agreement)
                kind_stable += int(agreement)
                total += 1
            by_kind[kind] = (
                kind_stable / len(selected) if selected else 0.0
            )
        result[name] = {
            "sample_count": len(selected),
            "strength": 0.2,
            "seed": 42,
            "by_kind": by_kind,
            "mean_verdict_stability": stable / total if total else 0.0,
        }
    return result


def run_v22_selection(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    base_v2_model_dir: str | Path,
    v21_gate_root: str | Path,
    v22_model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    validation_limit: int = 6000,
    perturbation_limit: int = 200,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_dir)
    records = _deterministic_subset(
        iter_split_records(
            dataset_root / "flows",
            dataset_root / "splits" / "split-manifest.json",
            "validation",
        ),
        validation_limit,
    )
    engines = {
        "current_deep_v2": _build_engine(
            backend="deep_v2",
            model_dir=Path(base_v2_model_dir),
            ood_policy="conformal_v3_1",
            ood_gate_dir=Path(v21_gate_root),
            base_ood_gate_dir=Path(base_ood_gate_dir),
        ),
        "frozen_long_tcn_all_length": _build_engine(
            backend="deep_v2_all_length",
            model_dir=Path(base_v2_model_dir),
            ood_policy="conformal_v3_1",
            ood_gate_dir=Path(v21_gate_root),
            base_ood_gate_dir=Path(base_ood_gate_dir),
        ),
        "deep_v2_2": _build_engine(
            backend="deep_v2_2",
            model_dir=Path(v22_model_dir),
            ood_policy="conformal_v3_2",
            ood_gate_dir=Path(v22_model_dir),
            base_ood_gate_dir=Path(base_ood_gate_dir),
        ),
    }
    metrics = {
        name: _evaluate_engine(
            engine,
            records,
            output / "selection" / name,
            mode_name=name,
        )
        for name, engine in engines.items()
    }
    boundary = _boundary_consistency(
        records,
        model_dir=Path(v22_model_dir),
    )
    perturbation = _perturbation_stability(
        records,
        engines={
            "current_deep_v2": engines["current_deep_v2"],
            "deep_v2_2": engines["deep_v2_2"],
        },
        limit=perturbation_limit,
    )
    current = metrics["current_deep_v2"]["overall"]
    candidate = metrics["deep_v2_2"]["overall"]
    short_rows = []
    for bucket in ("1", "2", "3"):
        short_rows.append(
            metrics["deep_v2_2"]["by_packet_bucket"][bucket]
        )
    short_count = sum(item["sample_count"] for item in short_rows)
    short_covered = sum(
        item["coverage"] * item["sample_count"] for item in short_rows
    )
    short_errors = sum(
        item["selective_error"]
        * item["coverage"]
        * item["sample_count"]
        for item in short_rows
    )
    short_coverage = short_covered / short_count if short_count else 0.0
    short_selective_error = (
        short_errors / short_covered if short_covered else 1.0
    )
    checks = {
        "short_coverage_at_least_0_60": short_coverage >= 0.60,
        "short_selective_error_at_most_0_05": (
            short_selective_error <= 0.05
        ),
        "overall_coverage_gain_at_least_0_20": (
            candidate["coverage"] - current["coverage"] >= 0.20
        ),
        "macro_f1_drop_within_0_005": (
            candidate["selective_macro_f1"]
            - current["selective_macro_f1"]
            >= -0.005
        ),
        "boundary_consistency_at_least_0_98": (
            boundary["binary_probability_agreement"] >= 0.98
        ),
        "perturbation_stability_not_lower": (
            perturbation["deep_v2_2"]["mean_verdict_stability"]
            >= perturbation["current_deep_v2"]["mean_verdict_stability"]
        ),
        "audit_completion_is_one": (
            metrics["deep_v2_2"]["audit_completion"] == 1.0
        ),
        "blocked_field_violation_is_zero": (
            metrics["deep_v2_2"]["blocked_field_violation_count"] == 0
        ),
        "fusion_ownership_violation_is_zero": (
            metrics["deep_v2_2"]["fusion_ownership_violation_count"] == 0
        ),
        "ood_override_is_zero": (
            metrics["deep_v2_2"]["ood_override_count"] == 0
        ),
    }
    promoted = all(checks.values())
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_2_validation_selection",
        "selection_split": "USTC validation",
        "validation_limit": validation_limit,
        "test_used": False,
        "cesnet_used": False,
        "cipherspectrum_used": False,
        "cipherspectrum_locked_test_used": False,
        "modes": metrics,
        "short_1_to_3": {
            "sample_count": short_count,
            "coverage": short_coverage,
            "selective_error": short_selective_error,
        },
        "boundary_consistency": boundary,
        "perturbation": perturbation,
        "checks": checks,
        "promotion_status": "promoted" if promoted else "not_promoted",
        "test_evaluation_status": (
            "eligible_not_run" if promoted else "not_run_validation_failed"
        ),
        "default_backend": (
            "deep_v2_2" if promoted else "deep_v2"
        ),
        "fusion_modified": False,
        "automatic_deployment": False,
    }
    _dump(output / "selection_report.json", report)
    return report


def run_v22_protocol(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    base_v2_model_dir: str | Path,
    v21_gate_root: str | Path,
    v22_model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 12,
    batch_size: int = 1024,
    consistency_weight: float = 0.2,
    distillation_weight: float = 0.25,
    training_limit: int | None = None,
    validation_limit: int = 6000,
    perturbation_limit: int = 200,
) -> dict[str, Any]:
    output = Path(output_dir)
    seeds = list(seeds)
    protocol_config = {
        "schema_version": "1.0",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "base_v2_model_dir": str(Path(base_v2_model_dir).resolve()),
        "v21_gate_root": str(Path(v21_gate_root).resolve()),
        "v22_model_dir": str(Path(v22_model_dir).resolve()),
        "base_ood_gate_dir": str(Path(base_ood_gate_dir).resolve()),
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "consistency_weight": consistency_weight,
        "distillation_weight": distillation_weight,
        "training_limit": training_limit,
        "validation_limit": validation_limit,
        "perturbation_limit": perturbation_limit,
        "selection_split": "validation",
    }
    freeze_v22_protocol_config(
        output / "protocol_config.json",
        protocol_config,
    )
    audit_sequence_regimes(
        dataset_dir,
        output / "sequence_regime_audit.json",
    )
    training_summary_path = Path(v22_model_dir) / "training_summary.json"
    if not training_summary_path.exists():
        train_temporal_short_v22(
            dataset_dir,
            v22_model_dir,
            base_model_dir=base_v2_model_dir,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            consistency_weight=consistency_weight,
            distillation_weight=distillation_weight,
            limit=training_limit,
        )
    conformal_root = (
        Path(v22_model_dir)
        / "temporal"
        / "conformal_v3_2"
    )
    if not (conformal_root / "metadata.json").exists():
        calibrate_conformal_v32(
            dataset_dir,
            v22_model_dir,
            conformal_root,
        )
    return run_v22_selection(
        output,
        dataset_dir=dataset_dir,
        base_v2_model_dir=base_v2_model_dir,
        v21_gate_root=v21_gate_root,
        v22_model_dir=v22_model_dir,
        base_ood_gate_dir=base_ood_gate_dir,
        validation_limit=validation_limit,
        perturbation_limit=perturbation_limit,
    )


def finalize_v22(
    output_dir: str | Path,
    *,
    selection_report: str | Path,
    frozen_reference: str | Path,
    model_dir: str | Path,
    tests_passed: bool,
    test_count: int,
) -> dict[str, Any]:
    selection = json.loads(
        Path(selection_report).read_text(encoding="utf-8")
    )
    frozen = verify_frozen_v1_reference(frozen_reference)
    model_metadata = json.loads(
        (
            Path(model_dir) / "temporal" / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    copied_references_match = bool(
        model_metadata.get("source_hashes_match")
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "frozen_v1_hash_verification.json", frozen)
    promoted = selection["promotion_status"] == "promoted"
    safety = all(
        (
            tests_passed,
            frozen["all_match"],
            copied_references_match,
            selection["checks"]["audit_completion_is_one"],
            selection["checks"]["blocked_field_violation_is_zero"],
            selection["checks"]["fusion_ownership_violation_is_zero"],
            selection["checks"]["ood_override_is_zero"],
        )
    )
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_2",
        "acceptance_status": (
            "validation_promoted"
            if promoted and safety
            else "partial_complete"
        ),
        "promotion_status": selection["promotion_status"],
        "test_evaluation_status": selection["test_evaluation_status"],
        "tests_passed": tests_passed,
        "test_count": test_count,
        "frozen_v1_hashes_match": frozen["all_match"],
        "copied_stats_and_long_hashes_match_sources": (
            copied_references_match
        ),
        "fusion_modified": False,
        "ood_v1_v2_v3_v31_modified": False,
        "utility_retrained": False,
        "automatic_deployment": False,
        "cipherspectrum_locked_test_used": False,
    }
    _dump(output / "acceptance_report.json", report)
    _dump(
        output / "negative_results.json",
        {
            "schema_version": "1.0",
            "negative_results": (
                []
                if promoted
                else [
                    {
                        "item": "Temporal deep_v2_2 short-flow promotion",
                        "status": "not_promoted",
                        "failed_checks": [
                            name
                            for name, passed in selection["checks"].items()
                            if not passed
                        ],
                    }
                ]
            ),
        },
    )
    return report
