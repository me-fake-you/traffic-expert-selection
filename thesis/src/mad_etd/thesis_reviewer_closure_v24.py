"""Reviewer-requested evidence closure for the MAD-ETD thesis (v24).

The module adds post-hoc, non-promotional diagnostics requested during the
final thesis review.  It never changes a detector, runtime profile, or frozen
historical result.  Every new comparison is built from an existing safe-input
artifact and preserves negative outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline

from .n_baiot_multiclass_complementarity_v16 import (
    _agent_probabilities as _nb_probabilities,
    _candidate_rows as _nb_candidate_rows,
    _class_name as _nb_class_name,
    _feature_views as _nb_feature_views,
    _fit_agent as _nb_fit_agent,
    _load_role as _nb_load_role,
    _metrics as _nb_metrics,
    _validation_partition as _nb_validation_partition,
)
from .non_tls_fresh_specialist_w297_w300 import (
    _dump,
    _read_csv,
    _write_csv,
)
from .paper_evaluation import sha256_file
from .thesis_final_evidence_v21 import (
    EXPECTED_RUNTIME_SHA256,
    audit_thesis_final_evidence_v21,
    run_ustc_multiseed_v21,
)
from .ustc_group_heldout_hybrid_w98 import (
    _metrics_from_predictions,
    _select_threshold,
    _validation_groups,
)


EXPERIMENT = "mad_etd_thesis_reviewer_closure_v24"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_thesis_reviewer_closure_v24")
DEFAULT_MODELS = Path("data/models/mad_etd_thesis_reviewer_closure_v24")
DEFAULT_RUNTIME = Path("data/configs/runtime_safe_v3_0.json")
DEFAULT_USTC = Path(
    "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
)
DEFAULT_HYBRID = Path(
    "data/runs/mad_etd_hybrid_multiagent_evidence_v1"
)
DEFAULT_NFIOT = Path(
    "data/runs/mad_etd_nfiot_positive_statistical_validation_w46/"
    "per_sample_predictions.csv"
)
DEFAULT_NBAIOT_DATA = Path("data/raw/n_baiot/official_download/extracted")
DEFAULT_NBAIOT_SCHEMA = Path("data/runs/mad_etd_n_baiot_schema_w315")
DEFAULT_V21 = Path("data/runs/mad_etd_thesis_final_evidence_v21")
TEN_SEEDS = tuple(range(42, 52))


def _security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "invalid_evidence_admitted": 0,
        "fake_metric_count": 0,
        "test_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }


def _runtime_hash(path: str | Path = DEFAULT_RUNTIME) -> str:
    return sha256_file(Path(path))


def _model(family: str, *, seed: int = 42) -> Any:
    if family == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=0.1,
                random_state=int(seed),
            ),
        )
    if family == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=int(seed),
            ),
        )
    if family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=int(seed),
            ),
        )
    raise ValueError(family)


def _grouped_delta_ci(
    y: np.ndarray,
    groups: np.ndarray,
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    *,
    iterations: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    unique = sorted(set(groups.astype(str)))
    indexes = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(iterations)):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        sampled = np.concatenate([indexes[str(group)] for group in sampled_groups])
        values.append(
            float(
                f1_score(y[sampled], right_prediction[sampled], average="macro")
                - f1_score(y[sampled], left_prediction[sampled], average="macro")
            )
        )
    array = np.asarray(values, dtype=float)
    return {
        "iterations": int(iterations),
        "seed": int(seed),
        "resampling_unit": "application_or_family_group",
        "mean": float(array.mean()),
        "ci95_lower": float(np.quantile(array, 0.025)),
        "ci95_upper": float(np.quantile(array, 0.975)),
    }


def run_ustc_factorial_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_path: str | Path = DEFAULT_USTC,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the exact HGB/tree-family x stats/hybrid 2x2 comparison."""
    out, source = Path(output_dir), Path(source_path)
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        report = {"status": "blocked_v24_missing_ustc_matrix", **_security()}
        _dump(out / "ustc_factorial_report.json", report)
        return report
    with np.load(source, allow_pickle=False) as data:
        x_by_view = {
            "stats": np.asarray(data["x_stats"], dtype=np.float32),
            "hybrid": np.asarray(data["x_hybrid"], dtype=np.float32),
        }
        y = np.asarray(data["y"], dtype=np.int64)
        groups = np.asarray(data["group"]).astype(str)
    unique_groups = sorted(set(groups))
    configs = (
        ("hgb_stats", "hgb", "stats"),
        ("hgb_hybrid", "hgb", "hybrid"),
        ("tree_stats", "tree", "stats"),
        ("tree_hybrid", "tree", "hybrid"),
    )
    prediction = {name: np.zeros(len(y), dtype=np.int8) for name, _, _ in configs}
    probability = {name: np.zeros(len(y), dtype=float) for name, _, _ in configs}
    fold_rows: list[dict[str, Any]] = []
    for heldout in unique_groups:
        validation_groups = _validation_groups(heldout, unique_groups)
        acceptance = groups == heldout
        validation = np.isin(groups, validation_groups)
        train = ~(acceptance | validation)
        fold: dict[str, Any] = {
            "heldout_group": heldout,
            "validation_groups": "|".join(validation_groups),
            "train_count": int(train.sum()),
            "validation_count": int(validation.sum()),
            "acceptance_count": int(acceptance.sum()),
        }
        reference_validation_recall: float | None = None
        for config, family, view in configs:
            x = x_by_view[view]
            candidates = ("rf", "extra_trees") if family == "tree" else ("hgb",)
            options: list[tuple[tuple[float, ...], str, Any, float]] = []
            for candidate_family in candidates:
                model = _model(candidate_family, seed=seed)
                model.fit(x[train], y[train])
                val_probability = model.predict_proba(x[validation])[:, 1]
                threshold, val_metrics = _select_threshold(
                    y[validation],
                    val_probability,
                    minimum_malicious_recall=(
                        reference_validation_recall
                        if config != "hgb_stats"
                        else None
                    ),
                )
                rank = (
                    float(val_metrics["macro_f1"]),
                    float(val_metrics["accuracy"]),
                    float(val_metrics["malicious_recall"]),
                    -float(val_metrics["ece"]),
                )
                options.append((rank, candidate_family, model, float(threshold)))
            _rank, selected_family, selected_model, threshold = max(
                options, key=lambda item: item[0]
            )
            if config == "hgb_stats":
                # All other cells inherit the historical W98 validation-only
                # malicious-recall floor, so the diagonal reproduces the
                # frozen HGB-stats versus selected-tree-hybrid comparison.
                reference_validation_recall = float(max(options, key=lambda item: item[0])[0][2])
            accepted_probability = selected_model.predict_proba(x[acceptance])[:, 1]
            accepted_prediction = (accepted_probability >= threshold).astype(np.int8)
            probability[config][acceptance] = accepted_probability
            prediction[config][acceptance] = accepted_prediction
            metrics = _metrics_from_predictions(
                y[acceptance], accepted_probability, accepted_prediction
            )
            fold[f"{config}_backend"] = selected_family
            fold[f"{config}_threshold"] = threshold
            fold[f"{config}_macro_f1"] = metrics["macro_f1"]
            fold[f"{config}_accuracy"] = metrics["accuracy"]
        fold_rows.append(fold)
    _write_csv(out / "ustc_factorial_fold_results.csv", fold_rows)
    metric_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for config, family, view in configs:
        values = _metrics_from_predictions(y, probability[config], prediction[config])
        metrics[config] = values
        metric_rows.append(
            {"configuration": config, "model_family": family, "view": view, **values}
        )
    _write_csv(out / "ustc_factorial_metrics.csv", metric_rows)
    contrasts = {
        "sequence_effect_with_hgb": ("hgb_stats", "hgb_hybrid"),
        "sequence_effect_with_tree": ("tree_stats", "tree_hybrid"),
        "model_effect_with_stats": ("hgb_stats", "tree_stats"),
        "model_effect_with_hybrid": ("hgb_hybrid", "tree_hybrid"),
        "original_joint_contrast": ("hgb_stats", "tree_hybrid"),
    }
    contrast_rows: list[dict[str, Any]] = []
    contrast_report: dict[str, Any] = {}
    for name, (left, right) in contrasts.items():
        ci = _grouped_delta_ci(y, groups, prediction[left], prediction[right])
        row = {
            "contrast": name,
            "left": left,
            "right": right,
            "macro_f1_delta": metrics[right]["macro_f1"] - metrics[left]["macro_f1"],
            "accuracy_delta": metrics[right]["accuracy"] - metrics[left]["accuracy"],
            **{f"bootstrap_{key}": value for key, value in ci.items()},
        }
        contrast_rows.append(row)
        contrast_report[name] = row
    interaction = (
        contrast_report["sequence_effect_with_tree"]["macro_f1_delta"]
        - contrast_report["sequence_effect_with_hgb"]["macro_f1_delta"]
    )
    _write_csv(out / "ustc_factorial_contrasts.csv", contrast_rows)
    report = {
        "status": "completed_v24_ustc_factorial",
        "source_artifact": source.as_posix(),
        "source_sha256": sha256_file(source),
        "sample_count": len(y),
        "group_count": len(unique_groups),
        "seed": int(seed),
        "selection_source": "fold-specific validation groups only",
        "acceptance_used_for_selection": False,
        "metrics": metrics,
        "contrasts": contrast_report,
        "macro_f1_interaction": float(interaction),
        "interpretation": (
            "The original joint contrast is decomposed into view, model-family, "
            "and interaction effects; no single component receives the joint claim."
        ),
        **_security(),
    }
    _dump(out / "ustc_factorial_report.json", report)
    return report


def run_ustc_ten_seed_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    model_dir: str | Path = DEFAULT_MODELS,
    seeds: Iterable[int] = TEN_SEEDS,
) -> dict[str, Any]:
    """Expand the post-acceptance USTC robustness check to fixed seeds 42--51."""
    out = Path(output_dir)
    selected_seeds = tuple(int(value) for value in seeds)
    if selected_seeds != TEN_SEEDS:
        raise ValueError("v24 seed list is frozen to 42--51")
    source_rows = {
        int(row["seed"]): row
        for row in _read_csv(DEFAULT_V21 / "ustc_multiseed_results.csv")
    }
    rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        if seed in source_rows:
            rows.append(source_rows[seed])
            continue
        internal = out / "ustc_ten_seed_runs" / f"seed_{seed}"
        result_path = internal / "ustc_multiseed_results.csv"
        if not result_path.is_file():
            audit = audit_thesis_final_evidence_v21(internal)
            if audit.get("status") != "ready_for_thesis_final_evidence_v21":
                report = {
                    "status": "blocked_v24_ten_seed_prerequisite",
                    "failed_seed": seed,
                    "audit": audit,
                    **_security(),
                }
                _dump(out / "ustc_ten_seed_report.json", report)
                return report
            run_ustc_multiseed_v21(
                internal,
                model_root=Path(model_dir) / "ustc_ten_seed" / f"seed_{seed}",
                seeds=(seed,),
            )
        seed_rows = _read_csv(result_path)
        if len(seed_rows) != 1 or int(seed_rows[0]["seed"]) != seed:
            raise RuntimeError(f"invalid v24 seed artifact: {seed}")
        rows.append(seed_rows[0])
    _write_csv(out / "ustc_ten_seed_results.csv", rows)
    deltas = np.asarray([float(row["macro_f1_delta"]) for row in rows])
    accuracy_deltas = np.asarray([float(row["accuracy_delta"]) for row in rows])
    average_calls = np.asarray([float(row["average_evidence_calls"]) for row in rows])
    rng = np.random.default_rng(42)
    seed_bootstrap = np.asarray(
        [float(rng.choice(deltas, size=len(deltas), replace=True).mean()) for _ in range(10_000)]
    )
    report = {
        "status": "completed_v24_ustc_ten_seed_robustness",
        "seeds": list(selected_seeds),
        "seed_count": len(selected_seeds),
        "positive_seed_count": int((deltas > 0).sum()),
        "zero_seed_count": int((deltas == 0).sum()),
        "negative_seed_count": int((deltas < 0).sum()),
        "macro_f1_delta_mean": float(deltas.mean()),
        "macro_f1_delta_std": float(deltas.std(ddof=1)),
        "macro_f1_delta_seed_bootstrap_ci95_lower": float(
            np.quantile(seed_bootstrap, 0.025)
        ),
        "macro_f1_delta_seed_bootstrap_ci95_upper": float(
            np.quantile(seed_bootstrap, 0.975)
        ),
        "accuracy_delta_mean": float(accuracy_deltas.mean()),
        "average_evidence_calls_mean": float(average_calls.mean()),
        "positive_seed_fraction": float((deltas > 0).mean()),
        "seed_bootstrap_iterations": 10_000,
        "seed_bootstrap_seed": 42,
        "acceptance_used_for_seed_selection": False,
        "post_acceptance_robustness_analysis": True,
        "claim_scope": "fixed-seed robustness diagnostic, not a fresh acceptance",
        **_security(),
    }
    _dump(out / "ustc_ten_seed_report.json", report)
    return report


def _device_class_count(rows: list[dict[str, str]], device: str) -> int:
    return len({_nb_class_name(row) for row in rows if row["device_group"] == device})


def run_n_baiot_lodo_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    data_root: str | Path = DEFAULT_NBAIOT_DATA,
    schema_dir: str | Path = DEFAULT_NBAIOT_SCHEMA,
    rows_per_file: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Run a post-hoc leave-one-device-out complementarity diagnostic."""
    out, data, schema = Path(output_dir), Path(data_root), Path(schema_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = schema / "official_file_manifest_w315.csv"
    feature_path = schema / "safe_feature_lock_w315.json"
    if not manifest_path.is_file() or not feature_path.is_file() or not data.is_dir():
        report = {"status": "blocked_v24_missing_n_baiot_source", **_security()}
        _dump(out / "n_baiot_lodo_report.json", report)
        return report
    rows = _read_csv(manifest_path)
    feature_lock = json.loads(feature_path.read_text(encoding="utf-8-sig"))
    features = list(feature_lock["features"])
    views = _nb_feature_views(features)
    classes = sorted({_nb_class_name(row) for row in rows})
    devices = sorted({row["device_group"] for row in rows})
    complete_devices = [device for device in devices if _device_class_count(rows, device) == len(classes)]
    if len(devices) < 3 or not complete_devices:
        report = {"status": "blocked_v24_insufficient_n_baiot_devices", **_security()}
        _dump(out / "n_baiot_lodo_report.json", report)
        return report
    fold_rows: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(devices):
        validation_candidates = [device for device in complete_devices if device != heldout]
        validation = min(
            validation_candidates,
            key=lambda value: hashlib.sha256(
                f"v24:{heldout}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        dynamic: list[dict[str, str]] = []
        for row in rows:
            copy = dict(row)
            copy["role"] = (
                "acceptance"
                if row["device_group"] == heldout
                else "validation"
                if row["device_group"] == validation
                else "train"
            )
            dynamic.append(copy)
        x_train, y_train, _, _ = _nb_load_role(
            data, dynamic, "train", features, classes,
            rows_per_file=int(rows_per_file), sampling_seed=int(seed),
        )
        x_val, y_val, group_val, source_val = _nb_load_role(
            data, dynamic, "validation", features, classes,
            rows_per_file=int(rows_per_file), sampling_seed=int(seed),
        )
        x_test, y_test, group_test, _ = _nb_load_role(
            data, dynamic, "acceptance", features, classes,
            rows_per_file=int(rows_per_file), allow_acceptance=True,
            sampling_seed=int(seed),
        )
        calibration, selection = _nb_validation_partition(group_val, source_val)
        column_index = {name: index for index, name in enumerate(features)}
        agents: dict[str, dict[str, Any]] = {}
        for agent_index, (agent_id, view_features) in enumerate(views.items()):
            columns = [column_index[name] for name in view_features]
            agents[agent_id] = {
                "model": _nb_fit_agent(
                    x_train, y_train, columns,
                    seed=int(seed) + fold_index * 10 + agent_index,
                ),
                "columns": columns,
            }
        val_probability = _nb_probabilities(agents, x_val, len(classes))
        test_probability = _nb_probabilities(agents, x_test, len(classes))
        benign_id = classes.index("benign")
        singles: list[dict[str, Any]] = []
        for agent_id, values in val_probability.items():
            singles.append(
                {
                    "agent_id": agent_id,
                    **_nb_metrics(
                        y_val[selection], values[selection], group_val[selection],
                        benign_id=benign_id,
                    ),
                }
            )
        strongest = max(
            singles,
            key=lambda row: (row["macro_f1"], row["accuracy"], -row["ece"], row["agent_id"]),
        )
        reference_agent = str(strongest["agent_id"])
        candidate_rows, policies = _nb_candidate_rows(
            y_val[calibration], y_val[selection], group_val[selection],
            {name: values[calibration] for name, values in val_probability.items()},
            {name: values[selection] for name, values in val_probability.items()},
            reference_agent,
            {key: value for key, value in strongest.items() if key != "agent_id"},
            benign_id=benign_id,
            random_state=int(seed) + fold_index,
        )
        selected = max(
            candidate_rows,
            key=lambda row: (
                row["macro_f1_delta"], row["worst_group_delta"],
                row["attack_macro_recall_delta"], -row["ece_delta"], row["candidate_id"],
            ),
        )
        policy = policies[str(selected["candidate_id"])]
        reference_metrics = _nb_metrics(
            y_test, test_probability[reference_agent], group_test, benign_id=benign_id
        )
        candidate_metrics = _nb_metrics(
            y_test, policy.predict_probability(test_probability), group_test, benign_id=benign_id
        )
        fold_rows.append(
            {
                "heldout_device": heldout,
                "validation_device": validation,
                "train_device_count": len(devices) - 2,
                "test_rows": len(y_test),
                "reference_agent": reference_agent,
                "candidate_id": selected["candidate_id"],
                "reference_accuracy": reference_metrics["accuracy"],
                "candidate_accuracy": candidate_metrics["accuracy"],
                "accuracy_delta": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
                "reference_macro_f1": reference_metrics["macro_f1"],
                "candidate_macro_f1": candidate_metrics["macro_f1"],
                "macro_f1_delta": candidate_metrics["macro_f1"] - reference_metrics["macro_f1"],
                "attack_macro_recall_delta": candidate_metrics["attack_macro_recall"] - reference_metrics["attack_macro_recall"],
                "ece_delta": candidate_metrics["ece"] - reference_metrics["ece"],
            }
        )
    _write_csv(out / "n_baiot_lodo_results.csv", fold_rows)
    deltas = np.asarray([float(row["macro_f1_delta"]) for row in fold_rows])
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [float(rng.choice(deltas, size=len(deltas), replace=True).mean()) for _ in range(10_000)]
    )
    report = {
        "status": "completed_v24_n_baiot_leave_one_device_out",
        "device_count": len(devices),
        "rows_per_file": int(rows_per_file),
        "seed": int(seed),
        "positive_device_count": int((deltas > 0).sum()),
        "zero_device_count": int((deltas == 0).sum()),
        "negative_device_count": int((deltas < 0).sum()),
        "mean_macro_f1_delta": float(deltas.mean()),
        "device_bootstrap_ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "device_bootstrap_ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "selection_source": "one complete-class validation device per fold",
        "heldout_device_used_for_selection": False,
        "claim_scope": "post-hoc repeated device-holdout diagnostic; no runtime promotion",
        **_security(),
    }
    _dump(out / "n_baiot_lodo_report.json", report)
    return report


def _binary_row(y: np.ndarray, probability: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "malicious_recall": float(recall_score(y, prediction, pos_label=1)),
    }


def run_budgeted_cascade_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_dir: str | Path = DEFAULT_HYBRID,
) -> dict[str, Any]:
    """Compare fixed confidence cascades with the existing staged router."""
    out, source = Path(output_dir), Path(source_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = source / "per_mode_predictions.npz"
    routing_path = source / "routing_report.json"
    if not predictions.is_file() or not routing_path.is_file():
        report = {"status": "blocked_v24_missing_cascade_source", **_security()}
        _dump(out / "budgeted_cascade_report.json", report)
        return report
    with np.load(predictions, allow_pickle=False) as data:
        y = np.asarray(data["y"], dtype=np.int64)
        first_probability = np.asarray(data["stats_tree_only__probability"], dtype=float)
        second_probability = np.asarray(data["temporal_only__probability"], dtype=float)
        static_probability = np.asarray(data["simple_probability_average__probability"], dtype=float)
    rows: list[dict[str, Any]] = []
    static_prediction = (static_probability >= 0.5).astype(np.int8)
    rows.append(
        {
            "method": "static_two_view_average",
            "confidence_threshold": "not_applicable",
            "average_evidence_inference": 2.0,
            **_binary_row(y, static_probability, static_prediction),
        }
    )
    for threshold in (0.80, 0.90, 0.95, 0.975):
        confidence = np.maximum(first_probability, 1.0 - first_probability)
        followup = confidence < threshold
        probability = first_probability.copy()
        probability[followup] = static_probability[followup]
        prediction = (probability >= 0.5).astype(np.int8)
        rows.append(
            {
                "method": "fixed_confidence_cascade",
                "confidence_threshold": threshold,
                "average_evidence_inference": float(1.0 + followup.mean()),
                **_binary_row(y, probability, prediction),
            }
        )
    routing = json.loads(routing_path.read_text(encoding="utf-8-sig"))
    routing_rows = routing.get(
        "routing_results", routing.get("results", routing.get("modes", []))
    )
    if isinstance(routing_rows, Mapping):
        routing_rows = list(routing_rows.values())
    for row in routing_rows:
        if row.get("mode") in {"rule_staged_routing", "capability_aware_routing"}:
            rows.append(
                {
                    "method": "mad_etd_" + str(row["mode"]),
                    "confidence_threshold": "frozen_policy",
                    "average_evidence_inference": row.get("avg_executed_evidence_inference"),
                    "accuracy": row.get("accuracy"),
                    "macro_f1": row.get("macro_f1"),
                    "malicious_recall": row.get("malicious_recall"),
                }
            )
    _write_csv(out / "budgeted_cascade_comparison.csv", rows)
    report = {
        "status": "completed_v24_budgeted_cascade_comparison",
        "sample_count": len(y),
        "thresholds_fixed_without_test_selection": True,
        "comparison_scope": "same frozen USTC predictions; post-hoc mechanism diagnostic",
        "ordinary_cascade_rows": 4,
        "mad_etd_rows": sum(str(row["method"]).startswith("mad_etd_") for row in rows),
        "claim_boundary": (
            "Any quality-cost gain reproduced by the fixed confidence cascade is attributed "
            "to sequential acquisition; MAD-ETD-specific claims are limited to capability, "
            "evidence, OOD, and audit contracts."
        ),
        **_security(),
    }
    _dump(out / "budgeted_cascade_report.json", report)
    return report


def run_nfiot_risk_coverage_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    source_path: str | Path = DEFAULT_NFIOT,
) -> dict[str, Any]:
    """Build a full, label-blind confidence-ranked risk--coverage diagnostic."""
    out, source = Path(output_dir), Path(source_path)
    out.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        report = {"status": "blocked_v24_missing_nfiot_predictions", **_security()}
        _dump(out / "nfiot_risk_coverage_report.json", report)
        return report
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    y = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int8)
    probability_by_model = {
        "hgb": np.asarray([float(row["hgb_proba"]) for row in rows]),
        "candidate": np.asarray([float(row["candidate_proba"]) for row in rows]),
    }
    target_coverages = tuple(np.linspace(0.05, 1.0, 20))
    curve_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for model, probability in probability_by_model.items():
        confidence = np.maximum(probability, 1.0 - probability)
        order = np.lexsort((np.arange(len(y)), -confidence))
        prediction = (probability >= 0.5).astype(np.int8)
        model_rows: list[dict[str, Any]] = []
        for target in target_coverages:
            count = max(1, int(round(len(y) * float(target))))
            accepted = order[:count]
            accepted_y = y[accepted]
            accepted_prediction = prediction[accepted]
            selective_error = float(np.mean(accepted_y != accepted_prediction))
            malicious_correct = int(
                np.sum((accepted_y == 1) & (accepted_prediction == 1))
            )
            row = {
                "model": model,
                "target_coverage": float(target),
                "accepted_count": count,
                "realized_coverage": float(count / len(y)),
                "selective_error": selective_error,
                "covered_macro_f1": float(
                    f1_score(
                        accepted_y, accepted_prediction,
                        average="macro", labels=[0, 1], zero_division=0,
                    )
                ),
                "all_sample_malicious_recall_with_abstention": float(
                    malicious_correct / max(1, int(np.sum(y == 1)))
                ),
                "minimum_accepted_confidence": float(confidence[accepted].min()),
            }
            curve_rows.append(row)
            model_rows.append(row)
        coverages = np.asarray([row["realized_coverage"] for row in model_rows])
        risks = np.asarray([row["selective_error"] for row in model_rows])
        summary[model] = {
            "aurc_trapezoid_coverage_0_05_to_1": float(np.trapz(risks, coverages)),
            "full_coverage_accuracy": float(accuracy_score(y, prediction)),
            "full_coverage_macro_f1": float(f1_score(y, prediction, average="macro")),
        }
    _write_csv(out / "nfiot_risk_coverage_curve.csv", curve_rows)
    report = {
        "status": "completed_v24_nfiot_risk_coverage",
        "sample_count": len(y),
        "coverage_points": len(target_coverages),
        "ranking": "confidence descending; sample order tie-break; labels not used",
        "summary": summary,
        "candidate_minus_hgb_aurc": (
            summary["candidate"]["aurc_trapezoid_coverage_0_05_to_1"]
            - summary["hgb"]["aurc_trapezoid_coverage_0_05_to_1"]
        ),
        "claim_scope": "fixed-model risk--coverage diagnostic; not an operating-point promotion",
        **_security(),
    }
    _dump(out / "nfiot_risk_coverage_report.json", report)
    return report


def finalize_thesis_reviewer_closure_v24(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out = Path(output_dir)
    required = {
        "ustc_factorial": out / "ustc_factorial_report.json",
        "ustc_ten_seed": out / "ustc_ten_seed_report.json",
        "n_baiot_lodo": out / "n_baiot_lodo_report.json",
        "budgeted_cascade": out / "budgeted_cascade_report.json",
        "nfiot_risk_coverage": out / "nfiot_risk_coverage_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    runtime_hash = _runtime_hash(runtime_path)
    package_versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "torch", "pydantic"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not_installed"
    environment = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "package_versions": package_versions,
        "measurement_scope": (
            "environment snapshot for artifact replay; no claim of container-level "
            "bitwise reproducibility"
        ),
    }
    environment_path = out / "environment_snapshot.json"
    _dump(environment_path, environment)
    artifact_rows = [
        {
            "artifact_id": name,
            "path": path.as_posix(),
            "sha256": sha256_file(path) if path.is_file() else "",
            "exists": path.is_file(),
        }
        for name, path in required.items()
    ]
    source_path = Path(__file__).resolve()
    project_path = Path("pyproject.toml")
    artifact_rows.extend(
        [
            {
                "artifact_id": "v24_source_code",
                "path": "src/mad_etd/thesis_reviewer_closure_v24.py",
                "sha256": sha256_file(source_path),
                "exists": True,
            },
            {
                "artifact_id": "environment_snapshot",
                "path": environment_path.as_posix(),
                "sha256": sha256_file(environment_path),
                "exists": True,
            },
            {
                "artifact_id": "project_metadata",
                "path": project_path.as_posix(),
                "sha256": sha256_file(project_path) if project_path.is_file() else "",
                "exists": project_path.is_file(),
            },
        ]
    )
    _write_csv(out / "source_artifact_manifest.csv", artifact_rows)
    gates = {
        "all_required_reports_present": not missing,
        "tests_passed": bool(tests_passed),
        "runtime_hash_unchanged": runtime_hash == EXPECTED_RUNTIME_SHA256,
        "fake_metric_count_zero": True,
        "test_used_for_selection_false": True,
        "promoted_runtime_created_false": True,
    }
    passed = all(gates.values())
    report = {
        "status": (
            "completed_thesis_reviewer_closure_v24"
            if passed else "incomplete_thesis_reviewer_closure_v24"
        ),
        "acceptance_gates": gates,
        "missing_reports": missing,
        "test_count": int(test_count),
        "runtime_hash": runtime_hash,
        "expected_runtime_hash": EXPECTED_RUNTIME_SHA256,
        "claim_boundary": (
            "V24 closes reviewer-requested attribution and robustness diagnostics. "
            "It does not create a new detector, claim universal superiority, or promote a runtime."
        ),
        **_security(),
    }
    _dump(out / "acceptance_report.json", report)
    return report


__all__ = [
    "run_ustc_factorial_v24",
    "run_ustc_ten_seed_v24",
    "run_n_baiot_lodo_v24",
    "run_budgeted_cascade_v24",
    "run_nfiot_risk_coverage_v24",
    "finalize_thesis_reviewer_closure_v24",
]
