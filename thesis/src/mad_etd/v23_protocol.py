from __future__ import annotations

import csv
import hashlib
import heapq
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .engine import DetectionEngine, build_default_engine
from .io import iter_split_records
from .perturb import apply_perturbation
from .schemas import FlowRecord
from .v2_evaluation import verify_frozen_v1_reference
from .v22_protocol import (
    _deterministic_subset,
    _evaluate_engine,
)
from .v23_training import (
    PERTURBATION_KINDS,
    calibrate_conformal_v33,
    train_temporal_robust_v23,
)


ROBUSTNESS_BUCKETS = ("1", "2", "3", "4+")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def freeze_v23_protocol_config(
    path: str | Path,
    config: dict[str, Any],
) -> None:
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "v2.3 resume configuration differs from the frozen run"
            )
        return
    _dump(target, config)


def _bucket(record: FlowRecord) -> str:
    count = len(record.sequence.packet_lengths)
    return str(count) if count < 4 else "4+"


def _select_by_hash(
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


def build_v23_robustness_manifest(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    per_bucket: int = 200,
) -> dict[str, Any]:
    if per_bucket != 200:
        raise ValueError("v2.3 robustness manifest freezes 200 samples per bucket")
    root = Path(dataset_dir)
    records = list(
        iter_split_records(
            root / "flows",
            root / "splits" / "split-manifest.json",
            "validation",
        )
    )
    selections: dict[str, list[str]] = {}
    for bucket in ROBUSTNESS_BUCKETS:
        eligible = [record for record in records if _bucket(record) == bucket]
        if bucket == "4+":
            exact_four = [
                record
                for record in eligible
                if len(record.sequence.packet_lengths) == 4
            ]
            if len(exact_four) >= per_bucket:
                eligible = exact_four
        selected = _select_by_hash(eligible, per_bucket)
        if len(selected) != per_bucket:
            raise ValueError(
                f"insufficient validation records for packet bucket {bucket}"
            )
        selections[bucket] = [record.sample_id for record in selected]
    manifest = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_3_robustness_manifest",
        "dataset": "USTC-TFC2016",
        "split": "validation",
        "per_bucket": per_bucket,
        "total": per_bucket * len(ROBUSTNESS_BUCKETS),
        "buckets": selections,
        "four_plus_sampling": "exact_four_preferred",
        "perturbations": list(PERTURBATION_KINDS),
        "strength": 0.2,
        "seed": 42,
        "test_used": False,
        "cipherspectrum_locked_test_used": False,
    }
    _dump(Path(output_path), manifest)
    return manifest


def _load_manifest_records(
    dataset_dir: Path,
    manifest_path: Path,
) -> list[FlowRecord]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "validation":
        raise ValueError("v2.3 robustness evaluation is validation-only")
    selected = {
        sample_id
        for values in manifest["buckets"].values()
        for sample_id in values
    }
    records = [
        record
        for record in iter_split_records(
            dataset_dir / "flows",
            dataset_dir / "splits" / "split-manifest.json",
            "validation",
        )
        if record.sample_id in selected
    ]
    if len(records) != int(manifest["total"]):
        raise ValueError("v2.3 robustness manifest is incomplete")
    return sorted(records, key=lambda record: record.sample_id)


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


def _disable_boundary_guard(engine: DetectionEngine) -> None:
    temporal = engine.detectors["TemporalBehaviorAgent"]
    if not hasattr(temporal.deep_model, "boundary_guard"):
        raise ValueError("boundary guard ablation requires deep_v2_3")
    temporal.deep_model.boundary_guard = False


def _robustness_matrix(
    records: list[FlowRecord],
    *,
    engines: dict[str, DetectionEngine],
    output_dir: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, engine in engines.items():
        rows: list[dict[str, Any]] = []
        clean = {
            record.sample_id: engine.analyze(record)[0].verdict.value
            for record in records
        }
        for kind in PERTURBATION_KINDS:
            for record in records:
                original_count = len(record.sequence.packet_lengths)
                perturbed = apply_perturbation(
                    record,
                    kind,
                    strength=0.2,
                    seed=42,
                )
                perturbed_count = len(perturbed.sequence.packet_lengths)
                verdict = engine.analyze(perturbed)[0].verdict.value
                crossed = (
                    original_count <= 3 < perturbed_count
                    or perturbed_count <= 3 < original_count
                )
                rows.append(
                    {
                        "sample_id": record.sample_id,
                        "packet_bucket": _bucket(record),
                        "kind": kind,
                        "original_count": original_count,
                        "perturbed_count": perturbed_count,
                        "crossed_boundary": crossed,
                        "clean_verdict": clean[record.sample_id],
                        "perturbed_verdict": verdict,
                        "stable": verdict == clean[record.sample_id],
                    }
                )
        cells: dict[str, Any] = {}
        for bucket in ROBUSTNESS_BUCKETS:
            for kind in PERTURBATION_KINDS:
                selected = [
                    row
                    for row in rows
                    if row["packet_bucket"] == bucket
                    and row["kind"] == kind
                ]
                key = f"{bucket}:{kind}"
                cells[key] = {
                    "sample_count": len(selected),
                    "verdict_stability": (
                        sum(row["stable"] for row in selected) / len(selected)
                        if selected
                        else 0.0
                    ),
                }
        crossing = [row for row in rows if row["crossed_boundary"]]
        crossing_by_direction: dict[str, Any] = {}
        for direction, selected in (
            (
                "3_to_4",
                [
                    row
                    for row in crossing
                    if row["original_count"] == 3
                    and row["perturbed_count"] >= 4
                ],
            ),
            (
                "4_to_3",
                [
                    row
                    for row in crossing
                    if row["original_count"] >= 4
                    and row["perturbed_count"] <= 3
                ],
            ),
        ):
            crossing_by_direction[direction] = {
                "sample_count": len(selected),
                "verdict_stability": (
                    sum(row["stable"] for row in selected) / len(selected)
                    if selected
                    else 0.0
                ),
            }
        summary = {
            "sample_count": len(records),
            "evaluations": len(rows),
            "strength": 0.2,
            "seed": 42,
            "cells": cells,
            "balanced_mean_verdict_stability": fmean(
                item["verdict_stability"] for item in cells.values()
            ),
            "route_crossing": crossing_by_direction,
        }
        mode_root = output_dir / name
        mode_root.mkdir(parents=True, exist_ok=True)
        with (mode_root / "robustness_predictions.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _dump(mode_root / "robustness_metrics.json", summary)
        result[name] = summary
    return result


def _short_metrics(mode: dict[str, Any]) -> dict[str, float]:
    rows = [mode["by_packet_bucket"][bucket] for bucket in ("1", "2", "3")]
    count = sum(item["sample_count"] for item in rows)
    covered = sum(item["coverage"] * item["sample_count"] for item in rows)
    errors = sum(
        item["selective_error"] * item["coverage"] * item["sample_count"]
        for item in rows
    )
    return {
        "sample_count": count,
        "coverage": covered / count if count else 0.0,
        "selective_error": errors / covered if covered else 1.0,
    }


def _candidate_checks(
    *,
    name: str,
    metrics: dict[str, Any],
    robustness: dict[str, Any],
    v22_metrics: dict[str, Any],
    v22_robustness: dict[str, Any],
) -> dict[str, bool]:
    overall = metrics["overall"]
    short = _short_metrics(metrics)
    cells = robustness["cells"]
    route = robustness["route_crossing"]
    return {
        "clean_system_coverage_at_least_0_55": overall["coverage"] >= 0.55,
        "short_coverage_at_least_0_90": short["coverage"] >= 0.90,
        "selective_error_at_most_0_05": (
            overall["selective_error"] <= 0.05
        ),
        "macro_f1_drop_within_0_005": (
            overall["selective_macro_f1"]
            - v22_metrics["overall"]["selective_macro_f1"]
            >= -0.005
        ),
        "balanced_stability_at_least_0_80": (
            robustness["balanced_mean_verdict_stability"] >= 0.80
        ),
        "balanced_stability_gain_over_v22_at_least_0_10": (
            robustness["balanced_mean_verdict_stability"]
            - v22_robustness["balanced_mean_verdict_stability"]
            >= 0.10
        ),
        "two_packet_padding_at_least_0_80": (
            cells["2:padding"]["verdict_stability"] >= 0.80
        ),
        "two_packet_dummy_at_least_0_80": (
            cells["2:dummy_packet"]["verdict_stability"] >= 0.80
        ),
        "two_packet_truncation_at_least_0_80": (
            cells["2:sequence_truncation"]["verdict_stability"] >= 0.80
        ),
        "three_packet_dummy_at_least_0_80": (
            cells["3:dummy_packet"]["verdict_stability"] >= 0.80
        ),
        "three_packet_jitter_at_least_0_80": (
            cells["3:iat_jitter"]["verdict_stability"] >= 0.80
        ),
        "three_to_four_stability_at_least_0_80": (
            route["3_to_4"]["verdict_stability"] >= 0.80
        ),
        "four_to_three_stability_at_least_0_80": (
            route["4_to_3"]["verdict_stability"] >= 0.80
        ),
        "audit_completion_is_one": metrics["audit_completion"] == 1.0,
        "blocked_field_violation_is_zero": (
            metrics["blocked_field_violation_count"] == 0
        ),
        "fusion_ownership_violation_is_zero": (
            metrics["fusion_ownership_violation_count"] == 0
        ),
        "ood_override_is_zero": metrics["ood_override_count"] == 0,
        "candidate_name_is_registered": name
        in {"deep_v2_3_consistency_only", "deep_v2_3_boundary_guard"},
    }


def run_v23_selection(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    base_v2_model_dir: str | Path,
    v21_gate_root: str | Path,
    v22_model_dir: str | Path,
    v23_model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    robustness_manifest: str | Path,
    validation_limit: int = 6000,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_dir)
    clean_records = _deterministic_subset(
        iter_split_records(
            dataset_root / "flows",
            dataset_root / "splits" / "split-manifest.json",
            "validation",
        ),
        validation_limit,
    )
    robustness_records = _load_manifest_records(
        dataset_root,
        Path(robustness_manifest),
    )
    engines = {
        "current_deep_v2": _build_engine(
            backend="deep_v2",
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
        "deep_v2_3_consistency_only": _build_engine(
            backend="deep_v2_3",
            model_dir=Path(v23_model_dir),
            ood_policy="conformal_v3_3",
            ood_gate_dir=Path(v23_model_dir),
            base_ood_gate_dir=Path(base_ood_gate_dir),
        ),
        "deep_v2_3_boundary_guard": _build_engine(
            backend="deep_v2_3",
            model_dir=Path(v23_model_dir),
            ood_policy="conformal_v3_3",
            ood_gate_dir=Path(v23_model_dir),
            base_ood_gate_dir=Path(base_ood_gate_dir),
        ),
    }
    _disable_boundary_guard(engines["deep_v2_3_consistency_only"])
    metrics = {
        name: _evaluate_engine(
            engine,
            clean_records,
            output / "selection" / name,
            mode_name=name,
        )
        for name, engine in engines.items()
    }
    robustness = _robustness_matrix(
        robustness_records,
        engines=engines,
        output_dir=output / "robustness",
    )
    candidate_checks = {
        name: _candidate_checks(
            name=name,
            metrics=metrics[name],
            robustness=robustness[name],
            v22_metrics=metrics["deep_v2_2"],
            v22_robustness=robustness["deep_v2_2"],
        )
        for name in (
            "deep_v2_3_consistency_only",
            "deep_v2_3_boundary_guard",
        )
    }
    eligible = [
        name
        for name, checks in candidate_checks.items()
        if all(checks.values())
    ]
    selected = (
        max(
            eligible,
            key=lambda name: (
                robustness[name]["balanced_mean_verdict_stability"],
                metrics[name]["overall"]["coverage"],
                name == "deep_v2_3_boundary_guard",
            ),
        )
        if eligible
        else None
    )
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_3_validation_selection",
        "selection_split": "USTC validation",
        "validation_limit": validation_limit,
        "robustness_manifest": str(Path(robustness_manifest)),
        "test_used": False,
        "cesnet_used": False,
        "cipherspectrum_used": False,
        "cipherspectrum_locked_test_used": False,
        "modes": metrics,
        "robustness": robustness,
        "candidate_checks": candidate_checks,
        "promotion_status": "promoted" if selected else "not_promoted",
        "selected_candidate": selected,
        "test_evaluation_status": (
            "eligible_not_run"
            if selected
            else "not_run_validation_failed"
        ),
        "default_backend": selected or "deep_v2",
        "fusion_modified": False,
        "automatic_deployment": False,
    }
    _dump(output / "selection_report.json", report)
    return report


def run_v23_protocol(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    base_v2_model_dir: str | Path,
    v21_gate_root: str | Path,
    v22_model_dir: str | Path,
    v23_model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    seeds: Iterable[int] = (42, 43, 44),
    epochs: int = 6,
    batch_size: int = 1024,
    training_limit: int | None = None,
    validation_limit: int = 6000,
) -> dict[str, Any]:
    output = Path(output_dir)
    seeds = list(seeds)
    protocol_config = {
        "schema_version": "1.0",
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "base_v2_model_dir": str(Path(base_v2_model_dir).resolve()),
        "v21_gate_root": str(Path(v21_gate_root).resolve()),
        "v22_model_dir": str(Path(v22_model_dir).resolve()),
        "v23_model_dir": str(Path(v23_model_dir).resolve()),
        "base_ood_gate_dir": str(Path(base_ood_gate_dir).resolve()),
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "loss_weights": {
            "clean_bce": 1.0,
            "perturbed_bce": 0.5,
            "probability_consistency": 0.25,
            "embedding_consistency": 0.1,
            "boundary_consistency": 0.25,
        },
        "training_strengths": [0.1, 0.2, 0.3],
        "training_limit": training_limit,
        "validation_limit": validation_limit,
        "selection_split": "validation",
    }
    freeze_v23_protocol_config(
        output / "protocol_config.json",
        protocol_config,
    )
    manifest_path = output / "robustness_manifest.json"
    if not manifest_path.exists():
        build_v23_robustness_manifest(
            dataset_dir,
            manifest_path,
        )
    training_summary_path = Path(v23_model_dir) / "training_summary.json"
    if not training_summary_path.exists():
        train_temporal_robust_v23(
            dataset_dir,
            v23_model_dir,
            base_v22_model_dir=v22_model_dir,
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            limit=training_limit,
        )
    conformal_root = (
        Path(v23_model_dir) / "temporal" / "conformal_v3_3"
    )
    if not (conformal_root / "metadata.json").exists():
        calibrate_conformal_v33(
            dataset_dir,
            v23_model_dir,
            conformal_root,
        )
    return run_v23_selection(
        output,
        dataset_dir=dataset_dir,
        base_v2_model_dir=base_v2_model_dir,
        v21_gate_root=v21_gate_root,
        v22_model_dir=v22_model_dir,
        v23_model_dir=v23_model_dir,
        base_ood_gate_dir=base_ood_gate_dir,
        robustness_manifest=manifest_path,
        validation_limit=validation_limit,
    )


def finalize_v23(
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
    selected = selection.get("selected_candidate")
    selected_checks = (
        selection["candidate_checks"].get(selected, {})
        if selected
        else {}
    )
    safety = all(
        (
            tests_passed,
            frozen["all_match"],
            copied_references_match,
            bool(selected_checks)
            and selected_checks.get("audit_completion_is_one", False),
            bool(selected_checks)
            and selected_checks.get(
                "blocked_field_violation_is_zero",
                False,
            ),
            bool(selected_checks)
            and selected_checks.get(
                "fusion_ownership_violation_is_zero",
                False,
            ),
            bool(selected_checks)
            and selected_checks.get("ood_override_is_zero", False),
        )
    )
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_3",
        "acceptance_status": (
            "validation_promoted"
            if promoted and safety
            else "partial_complete"
        ),
        "promotion_status": selection["promotion_status"],
        "selected_candidate": selected,
        "test_evaluation_status": selection["test_evaluation_status"],
        "tests_passed": tests_passed,
        "test_count": test_count,
        "frozen_v1_hashes_match": frozen["all_match"],
        "copied_stats_and_long_hashes_match_sources": (
            copied_references_match
        ),
        "fusion_modified": False,
        "ood_v1_v2_v3_v31_v32_modified": False,
        "utility_retrained": False,
        "automatic_deployment": False,
        "cipherspectrum_locked_test_used": False,
    }
    _dump(output / "acceptance_report.json", report)
    failed = (
        [
            name
            for name, passed in selection["candidate_checks"][
                "deep_v2_3_boundary_guard"
            ].items()
            if not passed
        ]
        if not promoted
        else []
    )
    _dump(
        output / "negative_results.json",
        {
            "schema_version": "1.0",
            "negative_results": (
                []
                if promoted
                else [
                    {
                        "item": "Temporal deep_v2_3 robust short-flow promotion",
                        "status": "not_promoted",
                        "failed_checks": failed,
                    }
                ]
            ),
        },
    )
    return report
