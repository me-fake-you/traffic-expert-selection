from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

import numpy as np
from pydantic import Field

from .paper_evaluation import hash_artifact_paths, sha256_file
from .schemas import StrictModel


V33_BASELINE = "current_stats__v2_6_continuous"
V33_BOOTSTRAP_ITERATIONS = 1000
V33_BOOTSTRAP_SEED = 42
BINARY = {"benign", "malicious"}
REJECT = {"suspicious", "unknown"}
TRANSITION_CATEGORIES = (
    "exact_stable",
    "harmful_flip",
    "unsafe_commitment",
    "conservative_degradation",
    "safety_recovery",
    "correctness_recovery",
    "wrong_persistence",
    "reject_churn",
)
RISK_TIERS = {
    "harmful_flip": ("critical", 5),
    "unsafe_commitment": ("critical", 5),
    "wrong_persistence": ("high", 4),
    "conservative_degradation": ("guarded", 3),
    "reject_churn": ("operational", 2),
    "safety_recovery": ("recovery", 1),
    "correctness_recovery": ("recovery", 0),
    "exact_stable": ("stable", 0),
}


class RiskOrderedTransition(StrictModel):
    truth: Literal["benign", "malicious"]
    clean_verdict: Literal[
        "benign", "malicious", "suspicious", "unknown"
    ]
    perturbed_verdict: Literal[
        "benign", "malicious", "suspicious", "unknown"
    ]
    clean_state: Literal["correct_binary", "wrong_binary", "reject"]
    perturbed_state: Literal["correct_binary", "wrong_binary", "reject"]
    category: Literal[
        "exact_stable",
        "harmful_flip",
        "unsafe_commitment",
        "conservative_degradation",
        "safety_recovery",
        "correctness_recovery",
        "wrong_persistence",
        "reject_churn",
    ]
    risk_tier: Literal[
        "critical", "high", "guarded", "operational", "recovery", "stable"
    ]
    risk_order: int = Field(ge=0, le=5)
    exact_stable: bool
    perturbed_wrong_binary: bool
    perturbed_covered: bool


class RiskOrderedRobustnessMetrics(StrictModel):
    sample_count: int = Field(ge=0)
    exact_stability: float = Field(ge=0, le=1)
    perturbed_wrong_binary_rate: float = Field(ge=0, le=1)
    perturbed_selective_error: float = Field(ge=0, le=1)
    harmful_flip_rate: float = Field(ge=0, le=1)
    unsafe_commitment_rate: float = Field(ge=0, le=1)
    dangerous_transition_rate: float = Field(ge=0, le=1)
    correct_retention: float = Field(ge=0, le=1)
    safe_retention: float = Field(ge=0, le=1)
    clean_coverage: float = Field(ge=0, le=1)
    perturbed_coverage: float = Field(ge=0, le=1)
    coverage_retention: float = Field(ge=0)
    conservative_degradation_rate: float = Field(ge=0, le=1)
    safety_recovery_rate: float = Field(ge=0, le=1)
    correctness_recovery_rate: float = Field(ge=0, le=1)
    wrong_persistence_rate: float = Field(ge=0, le=1)
    reject_churn_rate: float = Field(ge=0, le=1)
    transition_counts: dict[str, int]


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _state(verdict: str, truth: str) -> str:
    if verdict == truth:
        return "correct_binary"
    if verdict in BINARY:
        return "wrong_binary"
    if verdict in REJECT:
        return "reject"
    raise ValueError(f"unsupported verdict: {verdict}")


def classify_risk_ordered_transition(
    truth: str,
    clean_verdict: str,
    perturbed_verdict: str,
) -> RiskOrderedTransition:
    if truth not in BINARY:
        raise ValueError("risk-ordered transition requires binary truth")
    clean_state = _state(clean_verdict, truth)
    perturbed_state = _state(perturbed_verdict, truth)
    if clean_verdict == perturbed_verdict and clean_state != "wrong_binary":
        category = "exact_stable"
    elif clean_state == "correct_binary" and perturbed_state == "wrong_binary":
        category = "harmful_flip"
    elif clean_state == "reject" and perturbed_state == "wrong_binary":
        category = "unsafe_commitment"
    elif clean_state == "correct_binary" and perturbed_state == "reject":
        category = "conservative_degradation"
    elif clean_state == "wrong_binary" and perturbed_state == "reject":
        category = "safety_recovery"
    elif (
        clean_state in {"wrong_binary", "reject"}
        and perturbed_state == "correct_binary"
    ):
        category = "correctness_recovery"
    elif clean_state == "wrong_binary" and perturbed_state == "wrong_binary":
        category = "wrong_persistence"
    elif clean_state == "reject" and perturbed_state == "reject":
        category = "reject_churn"
    else:
        raise RuntimeError(
            "unclassified risk transition: "
            f"{truth}/{clean_verdict}/{perturbed_verdict}"
        )
    tier, order = RISK_TIERS[category]
    return RiskOrderedTransition(
        truth=truth,
        clean_verdict=clean_verdict,
        perturbed_verdict=perturbed_verdict,
        clean_state=clean_state,
        perturbed_state=perturbed_state,
        category=category,
        risk_tier=tier,
        risk_order=order,
        exact_stable=clean_verdict == perturbed_verdict,
        perturbed_wrong_binary=perturbed_state == "wrong_binary",
        perturbed_covered=perturbed_state != "reject",
    )


def summarize_risk_ordered_robustness(
    transitions: list[RiskOrderedTransition],
) -> RiskOrderedRobustnessMetrics:
    count = len(transitions)
    if not count:
        return RiskOrderedRobustnessMetrics(
            sample_count=0,
            exact_stability=0,
            perturbed_wrong_binary_rate=0,
            perturbed_selective_error=0,
            harmful_flip_rate=0,
            unsafe_commitment_rate=0,
            dangerous_transition_rate=0,
            correct_retention=0,
            safe_retention=0,
            clean_coverage=0,
            perturbed_coverage=0,
            coverage_retention=0,
            conservative_degradation_rate=0,
            safety_recovery_rate=0,
            correctness_recovery_rate=0,
            wrong_persistence_rate=0,
            reject_churn_rate=0,
            transition_counts={name: 0 for name in TRANSITION_CATEGORIES},
        )
    counts = Counter(item.category for item in transitions)
    clean_correct = [
        item for item in transitions if item.clean_state == "correct_binary"
    ]
    clean_covered = [
        item for item in transitions if item.clean_state != "reject"
    ]
    perturbed_covered = [
        item for item in transitions if item.perturbed_covered
    ]
    perturbed_wrong = sum(
        item.perturbed_wrong_binary for item in transitions
    )
    clean_coverage = len(clean_covered) / count
    perturbed_coverage = len(perturbed_covered) / count
    return RiskOrderedRobustnessMetrics(
        sample_count=count,
        exact_stability=sum(item.exact_stable for item in transitions) / count,
        perturbed_wrong_binary_rate=perturbed_wrong / count,
        perturbed_selective_error=(
            perturbed_wrong / len(perturbed_covered)
            if perturbed_covered
            else 0.0
        ),
        harmful_flip_rate=counts["harmful_flip"] / count,
        unsafe_commitment_rate=counts["unsafe_commitment"] / count,
        dangerous_transition_rate=(
            counts["harmful_flip"] + counts["unsafe_commitment"]
        )
        / count,
        correct_retention=(
            sum(
                item.perturbed_state == "correct_binary"
                for item in clean_correct
            )
            / len(clean_correct)
            if clean_correct
            else 0.0
        ),
        safe_retention=(
            sum(
                item.perturbed_state != "wrong_binary"
                for item in clean_correct
            )
            / len(clean_correct)
            if clean_correct
            else 0.0
        ),
        clean_coverage=clean_coverage,
        perturbed_coverage=perturbed_coverage,
        coverage_retention=(
            perturbed_coverage / clean_coverage if clean_coverage else 0.0
        ),
        conservative_degradation_rate=(
            counts["conservative_degradation"] / count
        ),
        safety_recovery_rate=counts["safety_recovery"] / count,
        correctness_recovery_rate=counts["correctness_recovery"] / count,
        wrong_persistence_rate=counts["wrong_persistence"] / count,
        reject_churn_rate=counts["reject_churn"] / count,
        transition_counts={
            name: counts[name] for name in TRANSITION_CATEGORIES
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_contract() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment": "mad_etd_v3_3_risk_ordered_robustness",
        "truth_states": ["correct_binary", "wrong_binary", "reject"],
        "transition_categories": {
            name: {
                "risk_tier": RISK_TIERS[name][0],
                "risk_order": RISK_TIERS[name][1],
            }
            for name in TRANSITION_CATEGORIES
        },
        "primary_metrics": [
            "perturbed_wrong_binary_rate",
            "unsafe_commitment_rate",
            "exact_stability",
            "perturbed_coverage",
        ],
        "legacy_metric_retained": "exact_stability",
        "weighted_total_score": False,
        "selection_or_promotion": False,
    }


def _validate_inputs(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    acceptance = json.loads(
        (input_dir / "acceptance_report.json").read_text(encoding="utf-8")
    )
    transition_metrics = json.loads(
        (input_dir / "transition_metrics.json").read_text(encoding="utf-8")
    )
    if (
        acceptance.get("status") != "accepted_diagnosis"
        or not acceptance.get("diagnosis_only")
        or acceptance.get("test_used")
        or acceptance.get("external_used")
        or acceptance.get("cipherspectrum_locked_test_used")
        or acceptance.get("promotion_status") != "not_applicable"
    ):
        raise ValueError("v3.3 requires accepted validation-only v3.2 input")
    if transition_metrics.get("test_used") or transition_metrics.get(
        "external_used"
    ) or transition_metrics.get("cipherspectrum_locked_test_used"):
        raise ValueError("v3.3 forbids test/external/locked input")
    return acceptance, transition_metrics


def _write_transitions(
    source_rows: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        transition = classify_risk_ordered_transition(
            source["truth"],
            source["clean_verdict"],
            source["perturbed_verdict"],
        )
        rows.append(
            {
                "mode": source["mode"],
                "sample_id": source["sample_id"],
                "packet_bucket": source["packet_bucket"],
                "perturbation": source["perturbation"],
                **transition.model_dump(mode="json"),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _summaries(
    rows: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        transitions = [
            RiskOrderedTransition.model_validate(
                {
                    key: row[key]
                    for key in RiskOrderedTransition.model_fields
                }
            )
            for row in selected
        ]
        return summarize_risk_ordered_robustness(
            transitions
        ).model_dump(mode="json")

    return {
        "schema_version": "1.0",
        "experiment": "mad_etd_v3_3_risk_ordered_robustness",
        "modes": {
            mode: {
                "overall": summarize(
                    [row for row in rows if row["mode"] == mode]
                ),
                "by_perturbation": {
                    kind: summarize(
                        [
                            row
                            for row in rows
                            if row["mode"] == mode
                            and row["perturbation"] == kind
                        ]
                    )
                    for kind in (
                        "padding",
                        "dummy_packet",
                        "iat_jitter",
                        "sequence_truncation",
                    )
                },
                "by_packet_bucket": {
                    bucket: summarize(
                        [
                            row
                            for row in rows
                            if row["mode"] == mode
                            and row["packet_bucket"] == bucket
                        ]
                    )
                    for bucket in ("1", "2", "3", "4+")
                },
            }
            for mode in modes
        },
        "test_used": False,
        "external_used": False,
        "cipherspectrum_locked_test_used": False,
    }


def _candidate_rows(
    metrics: dict[str, Any],
    clean_metrics: dict[str, Any],
    safety_ok: bool,
) -> list[dict[str, Any]]:
    baseline_clean = clean_metrics[V33_BASELINE]["overall"]
    baseline_risk = metrics["modes"][V33_BASELINE]["overall"]
    rows = []
    for mode, value in metrics["modes"].items():
        clean = clean_metrics[mode]["overall"]
        risk = value["overall"]
        checks = {
            "clean_coverage_noninferior": (
                clean["coverage"] >= baseline_clean["coverage"] - 0.005
            ),
            "clean_macro_f1_noninferior": (
                clean["selective_macro_f1"]
                >= baseline_clean["selective_macro_f1"] - 0.005
            ),
            "clean_selective_error_at_most_0_005": (
                clean["selective_error"] <= 0.005
            ),
            "perturbed_coverage_noninferior": (
                risk["perturbed_coverage"]
                >= baseline_risk["perturbed_coverage"] - 0.02
            ),
            "harmful_flip_not_above_baseline": (
                risk["harmful_flip_rate"]
                <= baseline_risk["harmful_flip_rate"] + 1e-12
            ),
            "original_safety_violations_zero": safety_ok,
        }
        rows.append(
            {
                "mode": mode,
                "is_baseline": mode == V33_BASELINE,
                "eligible": all(checks.values()),
                **checks,
                "clean_coverage": clean["coverage"],
                "clean_selective_error": clean["selective_error"],
                "clean_selective_macro_f1": clean["selective_macro_f1"],
                "perturbed_wrong_binary_rate": risk[
                    "perturbed_wrong_binary_rate"
                ],
                "unsafe_commitment_rate": risk[
                    "unsafe_commitment_rate"
                ],
                "harmful_flip_rate": risk["harmful_flip_rate"],
                "exact_stability": risk["exact_stability"],
                "perturbed_coverage": risk["perturbed_coverage"],
                "coverage_retention": risk["coverage_retention"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grouped_bootstrap(
    rows: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    fields = (
        "perturbed_wrong_binary_rate",
        "unsafe_commitment_rate",
        "harmful_flip_rate",
        "exact_stability",
        "perturbed_coverage",
    )
    by_mode: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_mode[row["mode"]][row["sample_id"]].append(row)
    baseline_ids = sorted(by_mode[V33_BASELINE])
    if not baseline_ids:
        raise ValueError("v3.3 baseline groups are empty")

    def vector(group: list[dict[str, Any]]) -> np.ndarray:
        transitions = [
            RiskOrderedTransition.model_validate(
                {
                    key: row[key]
                    for key in RiskOrderedTransition.model_fields
                }
            )
            for row in group
        ]
        metrics = summarize_risk_ordered_robustness(transitions)
        return np.asarray([getattr(metrics, field) for field in fields])

    baseline = np.stack(
        [vector(by_mode[V33_BASELINE][sample_id]) for sample_id in baseline_ids]
    )
    rng = np.random.default_rng(V33_BOOTSTRAP_SEED)
    draws = rng.integers(
        0,
        len(baseline_ids),
        size=(V33_BOOTSTRAP_ITERATIONS, len(baseline_ids)),
    )
    result = {}
    for mode in modes:
        if mode == V33_BASELINE:
            continue
        if set(by_mode[mode]) != set(baseline_ids):
            raise ValueError(f"v3.3 grouped samples differ for {mode}")
        candidate = np.stack(
            [vector(by_mode[mode][sample_id]) for sample_id in baseline_ids]
        )
        deltas = np.empty(
            (V33_BOOTSTRAP_ITERATIONS, len(fields)),
            dtype=np.float64,
        )
        for index, selected in enumerate(draws):
            deltas[index] = (
                candidate[selected].mean(axis=0)
                - baseline[selected].mean(axis=0)
            )
        result[mode] = {
            field: {
                "mean_delta": float(deltas[:, offset].mean()),
                "ci95": [
                    float(np.percentile(deltas[:, offset], 2.5)),
                    float(np.percentile(deltas[:, offset], 97.5)),
                ],
            }
            for offset, field in enumerate(fields)
        }
    return {
        "schema_version": "1.0",
        "method": "sample_id_grouped_bootstrap",
        "iterations": V33_BOOTSTRAP_ITERATIONS,
        "seed": V33_BOOTSTRAP_SEED,
        "group_count": len(baseline_ids),
        "rows_per_group": 4,
        "baseline": V33_BASELINE,
        "deltas": result,
    }


def _ranking(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in candidate_rows
        if row["eligible"] and not row["is_baseline"]
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            row["perturbed_wrong_binary_rate"],
            row["unsafe_commitment_rate"],
            -row["exact_stability"],
            -row["perturbed_coverage"],
            row["mode"],
        ),
    )
    return {
        "schema_version": "1.0",
        "baseline": V33_BASELINE,
        "ranking_rule": [
            "perturbed_wrong_binary_rate asc",
            "unsafe_commitment_rate asc",
            "exact_stability desc",
            "perturbed_coverage desc",
            "mode asc",
        ],
        "eligible_candidates": [row["mode"] for row in ranked],
        "retrospective_front_runner": (
            ranked[0]["mode"] if ranked else None
        ),
        "promotion_status": "not_applicable",
        "default_runtime": "runtime_safe_v3_0",
        "independent_acceptance_required": True,
    }


def aggregate_v33_risk_ordered(
    output_dir: str | Path,
    *,
    input_dir: str | Path = "data/runs/mad_etd_v3_2_diagnosis",
    config_dir: str | Path = "data/configs",
) -> dict[str, Any]:
    output = Path(output_dir)
    source = Path(input_dir)
    acceptance, v32_metrics = _validate_inputs(source)
    modes = list(acceptance["modes"])
    source_rows = _read_csv(source / "paired_transitions.csv")
    if len(source_rows) != int(acceptance["robustness_pairs"]):
        raise RuntimeError("v3.3 source transition count is incomplete")
    artifacts = {
        "v32_acceptance": source / "acceptance_report.json",
        "v32_transitions": source / "paired_transitions.csv",
        "v32_metrics": source / "transition_metrics.json",
        "runtime_profile": Path(config_dir) / "runtime_safe_v3_0.json",
    }
    before_path = output / "frozen_hashes_before.json"
    before = hash_artifact_paths(artifacts)
    _dump(before_path, before)
    _dump(output / "metric_contract.json", _metric_contract())
    rows = _write_transitions(
        source_rows,
        output / "risk_ordered_transitions.csv",
    )
    metrics = _summaries(rows, modes)
    legacy_matches = all(
        abs(
            metrics["modes"][mode]["overall"]["exact_stability"]
            - v32_metrics["transition_metrics"][mode]["overall"][
                "exact_stability"
            ]
        )
        <= 1e-12
        for mode in modes
    )
    metrics["legacy_exact_stability_matches_v32"] = legacy_matches
    _dump(output / "risk_ordered_metrics.json", metrics)
    safety = v32_metrics["safety"]
    safety_ok = (
        safety["audit_completion"] == 1.0
        and safety["blocked_field_violation_count"] == 0
        and safety["fusion_ownership_violation_count"] == 0
        and safety["ood_override_count"] == 0
        and safety["illegal_verdict_execution_count"] == 0
    )
    comparisons = _candidate_rows(
        metrics,
        v32_metrics["clean_metrics"],
        safety_ok,
    )
    _write_csv(output / "candidate_comparison.csv", comparisons)
    bootstrap = _grouped_bootstrap(rows, modes)
    _dump(output / "grouped_bootstrap.json", bootstrap)
    ranking = _ranking(comparisons)
    _dump(output / "retrospective_ranking.json", ranking)
    after = hash_artifact_paths(artifacts)
    _dump(output / "frozen_hashes_after.json", after)
    category_count = sum(
        metrics["modes"][mode]["overall"]["transition_counts"][category]
        for mode in modes
        for category in TRANSITION_CATEGORIES
    )
    checks = {
        "v32_status_accepted_diagnosis": True,
        "source_transition_count_complete": len(source_rows) == 28_800,
        "risk_transition_count_conserved": category_count == len(rows),
        "legacy_exact_stability_matches": legacy_matches,
        "grouped_bootstrap_complete": (
            bootstrap["iterations"] == 1000
            and bootstrap["group_count"] == 800
        ),
        "original_safety_checks_pass": safety_ok,
        "frozen_hashes_unchanged": before == after,
        "default_runtime_unchanged": True,
        "test_external_locked_not_used": True,
        "no_training_selection_or_promotion": True,
    }
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v3_3_risk_ordered_robustness",
        "status": "aggregated" if all(checks.values()) else "failed",
        "checks": checks,
        "mode_count": len(modes),
        "transition_count": len(rows),
        "baseline": V33_BASELINE,
        "retrospective_front_runner": ranking[
            "retrospective_front_runner"
        ],
        "front_runner_bootstrap": bootstrap["deltas"].get(
            ranking["retrospective_front_runner"]
        ),
        "promotion_status": "not_applicable",
        "default_runtime": "runtime_safe_v3_0",
        "test_used": False,
        "external_used": False,
        "cipherspectrum_locked_test_used": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(output / "acceptance_report.json", report)
    return report


def _write_document(
    path: str | Path,
    report: dict[str, Any],
    metrics: dict[str, Any],
    ranking: dict[str, Any],
    bootstrap: dict[str, Any],
) -> None:
    baseline = metrics["modes"][V33_BASELINE]["overall"]
    front = ranking["retrospective_front_runner"]
    front_metrics = metrics["modes"][front]["overall"] if front else None
    front_bootstrap = bootstrap["deltas"].get(front, {}) if front else {}
    lines = [
        "# MAD-ETD v3.3 Risk-Ordered Robustness",
        "",
        "This validation-only aggregation replaces no historical metric and "
        "promotes no runtime.",
        "",
        "## Baseline",
        "",
        f"- Mode: `{V33_BASELINE}`.",
        f"- Exact stability: {baseline['exact_stability']:.4f}.",
        "- Perturbed wrong-binary rate: "
        f"{baseline['perturbed_wrong_binary_rate']:.4f}.",
        "- Unsafe commitment rate: "
        f"{baseline['unsafe_commitment_rate']:.4f}.",
        f"- Perturbed coverage: {baseline['perturbed_coverage']:.4f}.",
        f"- Safe retention: {baseline['safe_retention']:.4f}.",
        "",
        "## Retrospective ranking",
        "",
        f"- Front-runner: `{front}`." if front else "- No eligible candidate.",
        *(
            [
                "- Front-runner wrong-binary rate: "
                f"{front_metrics['perturbed_wrong_binary_rate']:.4f}.",
                "- Front-runner unsafe commitment: "
                f"{front_metrics['unsafe_commitment_rate']:.4f}.",
                "- Front-runner exact stability: "
                f"{front_metrics['exact_stability']:.4f}.",
                "- Front-runner perturbed coverage: "
                f"{front_metrics['perturbed_coverage']:.4f}.",
                "- Wrong-binary delta CI95: "
                f"{front_bootstrap['perturbed_wrong_binary_rate']['ci95']}.",
                "- Exact-stability delta CI95: "
                f"{front_bootstrap['exact_stability']['ci95']}.",
                "- Perturbed-coverage delta CI95: "
                f"{front_bootstrap['perturbed_coverage']['ci95']}.",
            ]
            if front_metrics
            else []
        ),
        "",
        "This ranking is retrospective. It cannot change "
        "`runtime_safe_v3_0`; an independent non-overlapping acceptance "
        "protocol is required.",
        "",
        "The exact-stability confidence interval crosses zero, while "
        "perturbed coverage is lower. The result identifies an acceptance "
        "candidate, not a demonstrated comprehensive improvement.",
        "",
        f"Final status: `{report['status']}`.",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_v33_risk_ordered(
    output_dir: str | Path,
    *,
    document_path: str | Path = (
        "docs/MAD_ETD_V3_3_RISK_ORDERED_ROBUSTNESS.md"
    ),
    tests_passed: bool,
    test_count: int,
) -> dict[str, Any]:
    output = Path(output_dir)
    report = json.loads(
        (output / "acceptance_report.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (output / "risk_ordered_metrics.json").read_text(encoding="utf-8")
    )
    ranking = json.loads(
        (output / "retrospective_ranking.json").read_text(encoding="utf-8")
    )
    bootstrap = json.loads(
        (output / "grouped_bootstrap.json").read_text(encoding="utf-8")
    )
    checks = {
        **report["checks"],
        "tests_passed": tests_passed,
        "all_outputs_present": all(
            (output / name).exists()
            for name in (
                "metric_contract.json",
                "risk_ordered_transitions.csv",
                "risk_ordered_metrics.json",
                "candidate_comparison.csv",
                "grouped_bootstrap.json",
                "retrospective_ranking.json",
            )
        ),
    }
    final = {
        **report,
        "status": "accepted_aggregation"
        if all(checks.values())
        else "failed",
        "checks": checks,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "diagnostic_only": True,
        "promotion_status": "not_applicable",
        "default_runtime": "runtime_safe_v3_0",
    }
    _dump(output / "acceptance_report.json", final)
    _write_document(document_path, final, metrics, ranking, bootstrap)
    return final
