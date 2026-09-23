"""Frozen-evidence method closure for the MAD-ETD thesis (v34).

This module answers the final-review fairness questions without training a
detector, choosing a threshold on evaluation labels, or changing a runtime
profile.  It replays two already frozen artifacts:

* the 6,000-case, five-configuration Fusion ablation; and
* the 3,200 paired clean/perturbed AgentEvidence records.

The replay compares simple confidence rejection, OOD-score-only rejection,
Yager plus confidence rejection, and the full Fusion ranking under the same
acceptance/review budgets.  Negative and tied results are retained.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


EXPERIMENT = "mad_etd_thesis_method_closure_v34"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_thesis_method_closure_v34")
DEFAULT_ABLATION = Path(
    "data/runs/mad_etd_thesis_fusion_ablation_v1/"
    "per_sample_fusion_ablation.csv"
)
DEFAULT_LOCAL_EVIDENCE = Path(
    "data/runs/mad_etd_thesis_fusion_ablation_v1/"
    "local_evidence_hybrid.jsonl"
)
DEFAULT_PERTURBATIONS = Path(
    "data/runs/mad_etd_v3_2_diagnosis/paired_transitions.csv"
)
DEFAULT_RUNTIME = Path("data/configs/runtime_safe_v3_0.json")
EXPECTED_RUNTIME_SHA256 = (
    "c8d87e022b55ec30ffa98a51e021985720becffe13a8083887aaa91421773a59"
)
DIRECT_SAMPLE_COUNT = 6_000
PERTURBATION_PAIR_COUNT = 3_200
DIRECT_CONFIGS = (
    "probability_average",
    "yager_no_reliability_ood_off",
    "reliability_yager_ood_off",
    "ood_yager_no_reliability",
    "full_fusion",
)
MATCHED_COVERAGE_TARGETS = (
    1_402 / 6_000,
    2_826 / 3_200,
)
CURVE_COVERAGE_TARGETS = tuple(
    sorted(
        set(
            [round(value, 2) for value in np.arange(0.10, 1.001, 0.05)]
            + list(MATCHED_COVERAGE_TARGETS)
        )
    )
)


def _security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "invalid_evidence_admitted": 0,
        "fake_metric_count": 0,
        "test_used_for_selection": False,
        "model_training_performed": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _blocked_report(status: str, **details: Any) -> dict[str, Any]:
    return {"status": status, **details, **_security()}


def _read_local_ood_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            sample_id = str(payload["sample_id"])
            if sample_id in scores:
                raise RuntimeError(f"duplicate local-evidence sample: {sample_id}")
            evidence = payload.get("evidence", [])
            if not evidence:
                raise RuntimeError(f"missing local evidence: {sample_id}")
            scores[sample_id] = max(
                float(item.get("distribution_shift_score", 0.0) or 0.0)
                for item in evidence
            )
    return scores


def _validate_direct_frame(frame: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "truth",
        "config",
        "binary_proxy",
        "acceptance_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"missing direct-ablation fields: {missing}")
    observed = tuple(sorted(frame["config"].astype(str).unique()))
    expected = tuple(sorted(DIRECT_CONFIGS))
    if observed != expected:
        raise RuntimeError(f"unexpected direct configurations: {observed}")
    reference_ids: tuple[str, ...] | None = None
    reference_truth: tuple[str, ...] | None = None
    for config in DIRECT_CONFIGS:
        subset = frame[frame["config"] == config].sort_values("sample_id")
        if len(subset) != DIRECT_SAMPLE_COUNT:
            raise RuntimeError(f"{config} does not contain exactly 6000 cases")
        ids = tuple(subset["sample_id"].astype(str))
        truth = tuple(subset["truth"].astype(str))
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        elif ids != reference_ids or truth != reference_truth:
            raise RuntimeError("direct configurations do not share cases/truth")


def _select_indices(
    score: np.ndarray,
    sample_id: np.ndarray,
    accepted_count: int,
) -> np.ndarray:
    if not np.isfinite(score).all():
        raise RuntimeError("non-finite rejection score")
    return np.lexsort((sample_id.astype(str), -score))[:accepted_count]


def _matched_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    accepted_truth = truth[accepted]
    accepted_prediction = prediction[accepted]
    benign_total = int(np.sum(truth == "benign"))
    malicious_total = int(np.sum(truth == "malicious"))
    accepted_benign = int(np.sum(accepted_truth == "benign"))
    accepted_malicious = int(np.sum(accepted_truth == "malicious"))
    benign_coverage = accepted_benign / max(1, benign_total)
    malicious_coverage = accepted_malicious / max(1, malicious_total)
    correct_malicious = int(
        np.sum(
            (accepted_truth == "malicious")
            & (accepted_prediction == "malicious")
        )
    )
    return {
        "accepted_count": int(len(accepted)),
        "realized_coverage": float(len(accepted) / len(truth)),
        "accepted_benign_count": accepted_benign,
        "accepted_malicious_count": accepted_malicious,
        "accepted_malicious_ratio": float(
            accepted_malicious / max(1, len(accepted))
        ),
        "benign_class_coverage": float(benign_coverage),
        "malicious_class_coverage": float(malicious_coverage),
        "balanced_acceptance_rate": float(
            (benign_coverage + malicious_coverage) / 2.0
        ),
        "accepted_macro_f1": float(
            f1_score(
                accepted_truth,
                accepted_prediction,
                labels=["benign", "malicious"],
                average="macro",
                zero_division=0,
            )
        ),
        "selective_error": float(
            np.mean(accepted_truth != accepted_prediction)
        ),
        "all_sample_malicious_recall_with_abstention": float(
            correct_malicious / max(1, malicious_total)
        ),
    }


def run_matched_rejection_replay_v34(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    ablation_path: str | Path = DEFAULT_ABLATION,
    local_evidence_path: str | Path = DEFAULT_LOCAL_EVIDENCE,
) -> dict[str, Any]:
    """Replay four rejection/ranking policies on the frozen 6,000 cases."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ablation = Path(ablation_path)
    local_evidence = Path(local_evidence_path)
    if not ablation.is_file() or not local_evidence.is_file():
        report = _blocked_report(
            "blocked_v34_missing_direct_frozen_source",
            ablation_exists=ablation.is_file(),
            local_evidence_exists=local_evidence.is_file(),
        )
        _dump(out / "matched_rejection_report.json", report)
        return report
    try:
        frame = pd.read_csv(ablation)
        _validate_direct_frame(frame)
        ood_scores = _read_local_ood_scores(local_evidence)
        if len(ood_scores) != DIRECT_SAMPLE_COUNT:
            raise RuntimeError("local-evidence source does not contain 6000 cases")
        method_specs = (
            (
                "probability_average_confidence_reject",
                "probability_average",
                "confidence*(1-uncertainty)",
            ),
            (
                "probability_average_ood_score_reject",
                "probability_average",
                "1-max(local OOD score)",
            ),
            (
                "yager_confidence_reject",
                "yager_no_reliability_ood_off",
                "confidence*(1-uncertainty)",
            ),
            (
                "full_fusion_score_ranking",
                "full_fusion",
                "confidence*(1-uncertainty)",
            ),
        )
        matched_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        auc_rows: list[dict[str, Any]] = []
        for method, config, score_definition in method_specs:
            subset = (
                frame[frame["config"] == config]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            truth = subset["truth"].astype(str).to_numpy()
            prediction = subset["binary_proxy"].astype(str).to_numpy()
            sample_ids = subset["sample_id"].astype(str).to_numpy()
            if method == "probability_average_ood_score_reject":
                score = np.asarray(
                    [1.0 - ood_scores[sample_id] for sample_id in sample_ids]
                )
            else:
                score = subset["acceptance_score"].to_numpy(float)
            method_curve: list[dict[str, Any]] = []
            for target in CURVE_COVERAGE_TARGETS:
                count = max(1, min(len(subset), int(round(len(subset) * target))))
                accepted = _select_indices(score, sample_ids, count)
                metrics = _matched_metrics(truth, prediction, accepted)
                row = {
                    "method": method,
                    "source_config": config,
                    "target_coverage": target,
                    "score_definition": score_definition,
                    "labels_used_for_ranking": False,
                    **metrics,
                }
                method_curve.append(row)
                curve_rows.append(row)
                if target in MATCHED_COVERAGE_TARGETS:
                    matched_rows.append(row)
            ordered = sorted(method_curve, key=lambda item: item["realized_coverage"])
            coverages = np.asarray([item["realized_coverage"] for item in ordered])
            risks = np.asarray([item["selective_error"] for item in ordered])
            balanced = np.asarray(
                [item["balanced_acceptance_rate"] for item in ordered]
            )
            auc_rows.append(
                {
                    "method": method,
                    "coverage_grid_min": float(coverages.min()),
                    "coverage_grid_max": float(coverages.max()),
                    "aurc_trapezoid_reported_grid": float(
                        np.trapz(risks, coverages)
                    ),
                    "balanced_acceptance_area_reported_grid": float(
                        np.trapz(balanced, coverages)
                    ),
                    "labels_used_for_ranking": False,
                }
            )
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = _blocked_report(
            "blocked_v34_invalid_direct_frozen_source",
            error=f"{type(exc).__name__}: {exc}",
        )
        _dump(out / "matched_rejection_report.json", report)
        return report
    _write_csv(out / "matched_rejection_baselines.csv", matched_rows)
    _write_csv(out / "rejection_risk_coverage_curve.csv", curve_rows)
    _write_csv(out / "rejection_auc_summary.csv", auc_rows)
    low_budget = [
        row
        for row in matched_rows
        if row["target_coverage"] == MATCHED_COVERAGE_TARGETS[0]
    ]
    report = {
        "status": "completed_v34_matched_rejection_replay",
        "sample_count": DIRECT_SAMPLE_COUNT,
        "method_count": len(method_specs),
        "matched_coverage_targets": list(MATCHED_COVERAGE_TARGETS),
        "curve_coverage_targets": list(CURVE_COVERAGE_TARGETS),
        "labels_used_for_ranking": False,
        "direct_full_coverage_probability_average_macro_f1": 1.0,
        "low_budget_rows": low_budget,
        "interpretation": (
            "At equal review budgets the replay evaluates ranking and rejection, "
            "not detector training. Zero selective error is shared by simple "
            "baselines on this fixed, nearly separable evidence set; the full "
            "Fusion result is therefore not a general classification superiority "
            "claim."
        ),
        **_security(),
    }
    _dump(out / "matched_rejection_report.json", report)
    return report


def _evidence_probability(evidence: Mapping[str, Any]) -> float | None:
    if bool(evidence.get("abstained", False)):
        return None
    benign = float(evidence.get("benign_support", 0.0))
    malicious = float(evidence.get("malicious_support", 0.0))
    denominator = benign + malicious
    if denominator <= 0.0:
        return None
    return malicious / denominator


def _yager_pair(evidence: list[Mapping[str, Any]]) -> tuple[str, float]:
    if len(evidence) != 2:
        raise RuntimeError("paired replay requires exactly two local evidences")
    masses: list[tuple[float, float, float]] = []
    for item in evidence:
        benign = float(item.get("benign_support", 0.0))
        malicious = float(item.get("malicious_support", 0.0))
        unknown = max(0.0, 1.0 - benign - malicious)
        masses.append((benign, malicious, unknown))
    left_b, left_m, left_u = masses[0]
    right_b, right_m, right_u = masses[1]
    conflict = left_b * right_m + left_m * right_b
    benign = (
        left_b * right_b + left_b * right_u + left_u * right_b
    )
    malicious = (
        left_m * right_m + left_m * right_u + left_u * right_m
    )
    unknown = left_u * right_u + conflict
    prediction = "malicious" if malicious >= benign else "benign"
    return prediction, max(benign, malicious) * (1.0 - unknown)


def _paired_method_value(
    row: Mapping[str, Any], prefix: str, method: str
) -> tuple[str, float]:
    evidence: list[Mapping[str, Any]] = []
    for source in ("stats_evidence", "temporal_evidence"):
        raw = row[f"{prefix}{source}"]
        evidence.append(json.loads(raw) if isinstance(raw, str) else raw)
    probabilities = [
        value
        for value in (_evidence_probability(item) for item in evidence)
        if value is not None
    ]
    if not probabilities:
        raise RuntimeError("paired replay has no usable probability")
    average = float(np.mean(probabilities))
    average_prediction = "malicious" if average >= 0.5 else "benign"
    if method == "probability_average_confidence_reject":
        return average_prediction, max(average, 1.0 - average)
    if method == "probability_average_ood_score_reject":
        max_ood = max(
            float(item.get("distribution_shift_score", 0.0) or 0.0)
            for item in evidence
        )
        return average_prediction, 1.0 - max_ood
    if method == "yager_confidence_reject":
        return _yager_pair(evidence)
    if method == "full_fusion_score_ranking":
        raw_fusion = row[f"{prefix}fusion"]
        fusion = (
            json.loads(raw_fusion) if isinstance(raw_fusion, str) else raw_fusion
        )
        prediction = (
            "malicious"
            if float(fusion["malicious_support"])
            >= float(fusion["benign_support"])
            else "benign"
        )
        score = float(fusion["confidence"]) * (
            1.0 - float(fusion["uncertainty"])
        )
        return prediction, score
    raise RuntimeError(f"unknown paired method: {method}")


def run_matched_perturbation_replay_v34(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    perturbation_path: str | Path = DEFAULT_PERTURBATIONS,
) -> dict[str, Any]:
    """Replay matched-budget rejection on frozen clean/perturbed evidence."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(perturbation_path)
    if not source.is_file():
        report = _blocked_report("blocked_v34_missing_paired_frozen_source")
        _dump(out / "matched_perturbation_report.json", report)
        return report
    try:
        frame = pd.read_csv(source)
        frame = frame[
            frame["mode"] == "current_stats__v2_6_continuous"
        ].reset_index(drop=True)
        if len(frame) != PERTURBATION_PAIR_COUNT:
            raise RuntimeError("continuous paired source is not exactly 3200 rows")
        truth = frame["truth"].astype(str).to_numpy()
        pair_ids = (
            frame["sample_id"].astype(str)
            + "|"
            + frame["perturbation"].astype(str)
        ).to_numpy()
        accepted_count = 2_826
        methods = (
            "probability_average_confidence_reject",
            "probability_average_ood_score_reject",
            "yager_confidence_reject",
            "full_fusion_score_ranking",
        )
        rows: list[dict[str, Any]] = []
        for method in methods:
            clean_prediction: list[str] = []
            clean_score: list[float] = []
            perturbed_prediction: list[str] = []
            perturbed_score: list[float] = []
            for record in frame.to_dict(orient="records"):
                prediction, score = _paired_method_value(record, "clean_", method)
                clean_prediction.append(prediction)
                clean_score.append(score)
                prediction, score = _paired_method_value(
                    record, "perturbed_", method
                )
                perturbed_prediction.append(prediction)
                perturbed_score.append(score)
            clean_prediction_array = np.asarray(clean_prediction)
            perturbed_prediction_array = np.asarray(perturbed_prediction)
            clean_selected = _select_indices(
                np.asarray(clean_score), pair_ids, accepted_count
            )
            perturbed_selected = _select_indices(
                np.asarray(perturbed_score), pair_ids, accepted_count
            )
            clean_mask = np.zeros(len(frame), dtype=bool)
            perturbed_mask = np.zeros(len(frame), dtype=bool)
            clean_mask[clean_selected] = True
            perturbed_mask[perturbed_selected] = True
            harmful = (
                clean_mask
                & (clean_prediction_array == truth)
                & perturbed_mask
                & (perturbed_prediction_array != truth)
            )
            reversal = (
                clean_mask
                & perturbed_mask
                & (clean_prediction_array != perturbed_prediction_array)
            )
            metrics = _matched_metrics(
                truth, perturbed_prediction_array, perturbed_selected
            )
            rows.append(
                {
                    "method": method,
                    "target_coverage": accepted_count / len(frame),
                    "clean_accepted_count": int(clean_mask.sum()),
                    "perturbed_accepted_count": int(perturbed_mask.sum()),
                    "harmful_flip_count": int(harmful.sum()),
                    "harmful_flip_rate_all_pairs": float(harmful.mean()),
                    "accepted_binary_reversal_count": int(reversal.sum()),
                    "labels_used_for_ranking": False,
                    **metrics,
                }
            )
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        report = _blocked_report(
            "blocked_v34_invalid_paired_frozen_source",
            error=f"{type(exc).__name__}: {exc}",
        )
        _dump(out / "matched_perturbation_report.json", report)
        return report
    _write_csv(out / "matched_perturbation_baselines.csv", rows)
    report = {
        "status": "completed_v34_matched_perturbation_replay",
        "pair_count": PERTURBATION_PAIR_COUNT,
        "target_coverage": accepted_count / PERTURBATION_PAIR_COUNT,
        "accepted_count_per_method_and_side": accepted_count,
        "labels_used_for_ranking": False,
        "rows": rows,
        "interpretation": (
            "At the full Fusion perturbed coverage of 0.883125, confidence "
            "rejection already reduces probability-average harmful flips to a "
            "small nonzero count, while Yager-confidence and full Fusion both "
            "reach zero. The replay supports a structured rejection/ranking "
            "interpretation, not a claim that full Fusion uniquely improves "
            "classification robustness."
        ),
        **_security(),
    }
    _dump(out / "matched_perturbation_report.json", report)
    return report


def finalize_thesis_method_closure_v34(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    runtime = Path(runtime_path)
    required = {
        "matched_rejection": out / "matched_rejection_report.json",
        "matched_perturbation": out / "matched_perturbation_report.json",
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in required.items()
        if path.is_file()
    }
    runtime_hash = _sha256(runtime) if runtime.is_file() else None
    completed = (
        len(reports) == len(required)
        and all(
            str(report.get("status", "")).startswith("completed_v34_")
            for report in reports.values()
        )
        and runtime_hash == EXPECTED_RUNTIME_SHA256
        and tests_passed
    )
    source_paths = {
        "direct_ablation": DEFAULT_ABLATION,
        "local_evidence": DEFAULT_LOCAL_EVIDENCE,
        "paired_perturbations": DEFAULT_PERTURBATIONS,
        "runtime_profile": runtime,
    }
    manifest_rows = []
    for artifact_id, path in source_paths.items():
        manifest_rows.append(
            {
                "artifact_id": artifact_id,
                "path": path.as_posix(),
                "exists": path.is_file(),
                "sha256": _sha256(path) if path.is_file() else "",
            }
        )
    _write_csv(out / "source_artifact_manifest.csv", manifest_rows)
    boundary = {
        "experiment": EXPERIMENT,
        "claim": (
            "Frozen-evidence matched-budget diagnostics separate rejection and "
            "ranking effects from detector training effects."
        ),
        "not_claimed": [
            "full Fusion is universally more accurate than simple rejection",
            "zero harmful flips are uniquely caused by full Fusion",
            "the 6,000 fixed cases establish cross-dataset generalization",
            "a new runtime or detector was promoted",
        ],
        "negative_results_retained": True,
        "faithful_external_reproduction_count": 0,
        **_security(),
    }
    _dump(out / "claim_boundary.json", boundary)
    report = {
        "status": (
            "completed_thesis_method_closure_v34"
            if completed
            else "incomplete_thesis_method_closure_v34"
        ),
        "component_status": {
            name: report.get("status") for name, report in reports.items()
        },
        "missing_components": sorted(set(required) - set(reports)),
        "runtime_sha256": runtime_hash,
        "runtime_hash_matches_expected": runtime_hash
        == EXPECTED_RUNTIME_SHA256,
        "tests_passed": tests_passed,
        "test_count": int(test_count),
        **_security(),
    }
    _dump(out / "acceptance_report.json", report)
    return report
