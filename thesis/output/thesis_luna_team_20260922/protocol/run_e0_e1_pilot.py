"""Pre-registered fold-0 E0/E1 development diagnostic.

This runner deliberately never indexes evaluation rows or their labels.  It reuses
the frozen seed-42 fold-0 OOF predictions for fitting one fixed gain regressor,
then reports matched-budget diagnostics on the independent selection groups.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREREG = OUT / "preregistration.json"
RUN_DIR = OUT / "pilot_004"
PREVIOUS_RUN_DIR = OUT / "pilot_003"
HIST = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
PROVENANCE = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/raw_audit_001/selected_provenance.csv"
SPLIT_MANIFEST = HIST / "split_manifest.json"
FOLD_DIR = HIST / "seed_42_fold_0"
OOF = FOLD_DIR / "oof_predictions.npz"
OOF_DEPS = FOLD_DIR / "oof_dependencies.json"
TEMPORAL_MODEL = FOLD_DIR / "models/full_temporal.joblib"
STATS_MODEL = FOLD_DIR / "models/full_stats.joblib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def state_features(p_t: np.ndarray) -> np.ndarray:
    p_t = np.asarray(p_t, dtype=float)
    if not np.isfinite(p_t).all() or (p_t < 0.0).any() or (p_t > 1.0).any():
        raise ValueError("pT must be finite probabilities in [0,1]")
    return np.column_stack((p_t, np.abs(p_t - 0.5), entropy(p_t)))


def delta_values(y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=int)
    h0 = (np.asarray(p_t) >= 0.5).astype(int)
    h1 = (((np.asarray(p_t) + np.asarray(p_s)) / 2.0) >= 0.5).astype(int)
    loss0 = (h0 != y).astype(int)
    loss1 = (h1 != y).astype(int)
    delta = loss0 - loss1
    assert set(np.unique(delta)).issubset({-1, 0, 1})
    return {"h0": h0, "h1": h1, "delta": delta, "loss0": loss0, "loss1": loss1}


def confusion_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    return {
        "macro_f1": float(f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0)),
        "malicious_recall": float(tp / max(tp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def group_diagnostics(partition: str, sample_hash: np.ndarray, group: np.ndarray, y: np.ndarray, values: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in sorted(set(group.astype(str))):
        mask = group.astype(str) == name
        d = values["delta"][mask]
        rows.append({
            "partition": partition,
            "group": name,
            "rows": int(mask.sum()),
            "delta_-1": int((d == -1).sum()),
            "delta_0": int((d == 0).sum()),
            "delta_+1": int((d == 1).sum()),
            "corrected_C": int(((values["h0"][mask] != y[mask]) & (values["h1"][mask] == y[mask])).sum()),
            "introduced_D": int(((values["h0"][mask] == y[mask]) & (values["h1"][mask] != y[mask])).sum()),
            "net_potential_gain_C_minus_D": int(d.sum()),
            "oracle_correctable_C": int((d == 1).sum()),
            "sample_hash_first": str(sorted(sample_hash[mask].astype(str))[0]),
        })
    return pd.DataFrame(rows)


def error_overlap(partition: str, group: np.ndarray, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray) -> pd.DataFrame:
    h_t = (np.asarray(p_t) >= 0.5).astype(int)
    h_s = (np.asarray(p_s) >= 0.5).astype(int)
    rows = []
    for name in sorted(set(group.astype(str))):
        mask = group.astype(str) == name
        t_correct = h_t[mask] == y[mask]
        s_correct = h_s[mask] == y[mask]
        rows.append({
            "partition": partition,
            "group": name,
            "rows": int(mask.sum()),
            "both_correct": int((t_correct & s_correct).sum()),
            "only_temporal_correct": int((t_correct & ~s_correct).sum()),
            "only_stats_correct": int((~t_correct & s_correct).sum()),
            "both_wrong": int((~t_correct & ~s_correct).sum()),
            "disagreement": int((h_t[mask] != h_s[mask]).sum()),
            "temporal_to_stats_oracle_correctable": int((~t_correct & s_correct).sum()),
            "stats_to_temporal_oracle_correctable": int((t_correct & ~s_correct).sum()),
        })
    return pd.DataFrame(rows)


def stable_hash_order(sample_hash: np.ndarray) -> np.ndarray:
    return np.array(sorted(range(len(sample_hash)), key=lambda i: hashlib.sha256(str(sample_hash[i]).encode("utf-8")).hexdigest()), dtype=int)


def quota_indices(n: int, budget: float) -> int:
    return int(np.floor((budget - 1.0) * n + 0.5))


def selected_indices(name: str, budget: float, p_t: np.ndarray, predicted_delta: np.ndarray, sample_hash: np.ndarray) -> np.ndarray:
    k = quota_indices(len(p_t), budget)
    if k <= 0:
        return np.array([], dtype=int)
    if k >= len(p_t):
        return np.arange(len(p_t), dtype=int)
    hash_order = stable_hash_order(sample_hash)
    if name == "confidence_budget_gate":
        order = np.array(sorted(range(len(p_t)), key=lambda i: (abs(float(p_t[i]) - 0.5), str(sample_hash[i]))), dtype=int)
    elif name == "hash_random":
        order = hash_order
    elif name == "predicted_delta":
        order = np.array(sorted(range(len(p_t)), key=lambda i: (-float(predicted_delta[i]), str(sample_hash[i]))), dtype=int)
    else:
        raise ValueError(f"budgeted policy not supported: {name}")
    return order[:k]


def policy_metrics(name: str, budget: float, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, predicted_delta: np.ndarray, sample_hash: np.ndarray) -> dict[str, float | int | str]:
    n = len(y)
    h0 = (p_t >= 0.5).astype(int)
    h1 = (((p_t + p_s) / 2.0) >= 0.5).astype(int)
    if name == "first_only":
        acquired = np.array([], dtype=int)
    elif name == "all_average":
        acquired = np.arange(n, dtype=int)
    else:
        acquired = selected_indices(name, budget, p_t, predicted_delta, sample_hash)
    prediction = h0.copy()
    prediction[acquired] = h1[acquired]
    m = confusion_metrics(y, prediction)
    first_metrics = confusion_metrics(y, h0)
    corrected = int(((h0 == 0) & (y == 1) & (prediction == 1)).sum() + ((h0 == 1) & (y == 0) & (prediction == 0)).sum())
    introduced = int(((h0 == 1) & (y == 1) & (prediction == 0)).sum() + ((h0 == 0) & (y == 0) & (prediction == 1)).sum())
    return {
        "policy": name,
        "target_budget": float(budget),
        "rows": n,
        "second_calls": int(len(acquired)),
        "effective_budget": float(1.0 + len(acquired) / max(n, 1)),
        "macro_f1": m["macro_f1"],
        "malicious_recall": m["malicious_recall"],
        "false_positive_rate": m["false_positive_rate"],
        "tn": m["tn"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
        "corrected_C": corrected,
        "introduced_D": introduced,
        "net_C_minus_D": corrected - introduced,
        "first_only_macro_f1": first_metrics["macro_f1"],
        "diagnostic_scope": "fold0 selection development; fixed equal average after acquisition; no latency claim",
    }


def run_internal_tests() -> dict[str, str]:
    assert list(inspect.signature(state_features).parameters) == ["p_t"]
    p_t = np.array([0.25, 0.5, 0.75])
    p_s = np.array([0.2, 0.8, 0.3])
    y = np.array([0, 1, 1])
    expected_entropy = np.array([-(0.25 * np.log2(0.25) + 0.75 * np.log2(0.75)), 1.0, -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))])
    expected_state = np.column_stack((p_t, np.abs(p_t - 0.5), expected_entropy))
    assert np.allclose(state_features(p_t), expected_state, atol=1e-12)
    for invalid in (np.array([np.nan]), np.array([np.inf]), np.array([-0.01]), np.array([1.01])):
        try:
            state_features(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite/out-of-range pT was accepted")
    vals = delta_values(y, p_t, p_s)
    assert set(np.unique(vals["delta"])).issubset({-1, 0, 1})
    hashes = np.array([f"sample-{i}" for i in range(len(y))])
    for policy in ("confidence_budget_gate", "hash_random", "predicted_delta"):
        assert len(selected_indices(policy, 1.5, p_t, np.array([0.2, 0.8, 0.1, -0.1]), hashes)) == 2
    selected = selected_indices("hash_random", 1.5, p_t, np.zeros(len(y)), hashes)
    h0 = vals["h0"]
    prediction = h0.copy()
    prediction[selected] = vals["h1"][selected]
    assert np.flatnonzero(prediction != h0).tolist() == [i for i in selected if vals["h1"][i] != h0[i]]
    return {
        "state_signature_is_first_probability_only": "PASS",
        "state_matches_independent_manual_transform": "PASS",
        "state_rejects_nonfinite_or_out_of_range_probability": "PASS",
        "delta_domain_is_minus1_zero_plus1": "PASS",
        "matched_budget_quota_and_paired_switch_count": "PASS",
    }


def read_selection_labels_strict(path: Path, selection_hashes: set[str], selection_order: np.ndarray) -> np.ndarray:
    """Decode only rows whose sample_hash is preselected; never retain other labels."""
    found: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames and "sample_hash" in reader.fieldnames and "label" in reader.fieldnames
        for row in reader:
            sample = row["sample_hash"]
            if sample in selection_hashes:
                assert sample not in found, f"duplicate selection sample hash: {sample}"
                found[sample] = int(row["label"])
    assert set(found) == selection_hashes, "not all preselected sample hashes had a decoded label"
    values = np.array([found[str(sample)] for sample in selection_order], dtype=int)
    assert set(np.unique(values)).issubset({0, 1})
    return values


def assert_same_frame(current: pd.DataFrame, previous_path: Path, *, compressed: bool = False) -> None:
    previous = pd.read_csv(previous_path, compression="gzip" if compressed else None)
    current = current.reset_index(drop=True)
    previous = previous.reset_index(drop=True)
    assert list(current.columns) == list(previous.columns)
    assert len(current) == len(previous)
    for column in current.columns:
        left = current[column]
        right = previous[column]
        if pd.api.types.is_numeric_dtype(left):
            assert np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0.0, atol=1e-12, equal_nan=True), column
        else:
            assert left.astype(str).equals(right.astype(str)), column


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {RUN_DIR}")
    started = time.perf_counter()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))[0]
    roles = split["groups"]
    role_names = ("train", "calibration", "selection", "evaluation")
    assert set(roles) == set(role_names)
    log = []

    # Provenance is loaded without labels.  This supports role/capture/session checks
    # while ensuring evaluation labels are never materialized from the CSV.
    prov_cols = ["sample_hash", "partition", "group", "capture", "capture_sha256", "session_signature"]
    provenance = pd.read_csv(PROVENANCE, usecols=prov_cols)
    provenance = provenance[provenance["partition"] == "development40k"].copy()
    assert provenance["sample_hash"].is_unique

    with np.load(SOURCE, allow_pickle=False) as source:
        x_stats = source["x_stats"].copy()
        x_hybrid = source["x_hybrid"].copy()
        source_group = source["group"].astype(str).copy()
        sample_hash = source["sample_hash"].astype(str).copy()
    assert len(sample_hash) == 40000
    assert len(provenance) == len(sample_hash)
    assert set(provenance["sample_hash"]) == set(sample_hash)
    provenance = provenance.set_index("sample_hash").loc[sample_hash].reset_index()
    assert np.array_equal(provenance["group"].astype(str).to_numpy(), source_group)

    role_sets = {}
    for role in role_names:
        group_set = set(roles[role])
        role_mask = np.isin(source_group, list(group_set))
        role_rows = provenance.loc[role_mask]
        role_sets[role] = {
            "groups": sorted(group_set),
            "rows": int(role_mask.sum()),
            "captures": sorted(set(role_rows["capture"].astype(str))),
            "capture_sha256": sorted(set(role_rows["capture_sha256"].astype(str))),
            "sessions": sorted(set(role_rows["session_signature"].astype(str))),
        }
    for left_i, left in enumerate(role_names):
        for right in role_names[left_i + 1:]:
            for unit in ("groups", "captures", "capture_sha256", "sessions"):
                assert not (set(role_sets[left][unit]) & set(role_sets[right][unit])), f"role overlap: {left}/{right}/{unit}"

    with np.load(OOF, allow_pickle=False) as oof_file:
        oof = {key: oof_file[key].copy() for key in oof_file.files}
    required_oof = {"indices", "y", "stats", "temporal", "reliability_stats", "reliability_temporal"}
    assert required_oof.issubset(oof)
    oof_idx = oof["indices"].astype(int)
    assert len(oof_idx) == len(set(oof_idx.tolist()))
    assert np.isin(oof_idx, np.flatnonzero(np.isin(source_group, roles["train"]))).all()
    assert np.isfinite(oof["temporal"]).all() and np.isfinite(oof["stats"]).all()
    deps = json.loads(OOF_DEPS.read_text(encoding="utf-8"))
    for dep in deps:
        assert not (set(dep["predict_groups"]) & set(dep["fit_groups"]))
        assert not (set(dep["predict_groups"]) & set(dep["calibration_groups"]))

    # Fold 0 train-role genuine OOF labels and predictions are the only fit data.
    train_p_t = oof["temporal"].astype(float)
    train_p_s = oof["stats"].astype(float)
    train_y = oof["y"].astype(int)
    train_hash = sample_hash[oof_idx]
    train_group = source_group[oof_idx]
    train_values = delta_values(train_y, train_p_t, train_p_s)
    train_x = state_features(train_p_t)
    assert train_x.shape[1] == 3
    assert prereg["router_state"]["features"] == ["pT", "abs(pT - 0.5)", "binary_entropy(pT)"]

    # This is the single pre-registered model; no selection result enters its fit.
    with threadpool_limits(limits=1):
        gain_model = HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=20260922,
            early_stopping=False,
        )
        gain_model.fit(train_x, train_values["delta"])
    train_predicted_delta = gain_model.predict(train_x)

    selection_idx = np.flatnonzero(np.isin(source_group, roles["selection"]))
    assert len(selection_idx) > 0
    # This is the only use of selection labels.  No evaluation index is formed;
    # labels are decoded strictly by preselected sample_hash from provenance.
    selection_y = read_selection_labels_strict(PROVENANCE, set(sample_hash[selection_idx]), sample_hash[selection_idx])
    with threadpool_limits(limits=1):
        temporal_model = joblib.load(TEMPORAL_MODEL)
        stats_model = joblib.load(STATS_MODEL)
        selection_p_t = temporal_model.predict_proba(x_hybrid[selection_idx, 8:])[:, 1]
        selection_p_s = stats_model.predict_proba(x_stats[selection_idx])[:, 1]
    selection_hash = sample_hash[selection_idx]
    selection_group = source_group[selection_idx]
    selection_values = delta_values(selection_y, selection_p_t, selection_p_s)
    selection_predicted_delta = gain_model.predict(state_features(selection_p_t))

    # No evaluation labels, features, or sample hashes are indexed in this run.
    eval_group_set = set(roles["evaluation"])
    assert not set(selection_group) & eval_group_set
    assert not set(train_group) & eval_group_set
    log.append("evaluation labels/features/sample hashes were not indexed")

    train_frame = pd.DataFrame({
        "partition": "train_oof",
        "sample_hash": train_hash,
        "group": train_group,
        "y": train_y,
        "pT": train_p_t,
        "pS_offline_only": train_p_s,
        "p_first": train_p_t,
        "p_second": train_p_s,
        "h0_first_only": train_values["h0"],
        "h1_fixed_average": train_values["h1"],
        "delta": train_values["delta"],
        "predicted_delta": train_predicted_delta,
        "state_abs_margin": np.abs(train_p_t - 0.5),
        "state_entropy": entropy(train_p_t),
    })
    selection_frame = pd.DataFrame({
        "partition": "selection",
        "sample_hash": selection_hash,
        "group": selection_group,
        "y": selection_y,
        "pT": selection_p_t,
        "pS_offline_only": selection_p_s,
        "p_first": selection_p_t,
        "p_second": selection_p_s,
        "h0_first_only": selection_values["h0"],
        "h1_fixed_average": selection_values["h1"],
        "delta": selection_values["delta"],
        "predicted_delta": selection_predicted_delta,
        "state_abs_margin": np.abs(selection_p_t - 0.5),
        "state_entropy": entropy(selection_p_t),
    })
    assert list(train_frame.columns) == list(selection_frame.columns)
    group_frame = pd.concat([
        group_diagnostics("train_oof", train_hash, train_group, train_y, train_values),
        group_diagnostics("selection", selection_hash, selection_group, selection_y, selection_values),
    ], ignore_index=True)
    e0_overlap_frame = pd.concat([
        error_overlap("train_oof", train_group, train_y, train_p_t, train_p_s),
        error_overlap("selection", selection_group, selection_y, selection_p_t, selection_p_s),
    ], ignore_index=True)

    policies = ["first_only", "all_average", "confidence_budget_gate", "hash_random", "predicted_delta"]
    metric_rows = []
    for budget in prereg["budgets"]:
        for policy in policies:
            metric_rows.append(policy_metrics(policy, float(budget), selection_y, selection_p_t, selection_p_s, selection_predicted_delta, selection_hash))
    metric_frame = pd.DataFrame(metric_rows)
    for budget in prereg["budgets"]:
        expected = quota_indices(len(selection_y), float(budget))
        actual = metric_frame[(metric_frame.target_budget == budget) & metric_frame.policy.isin(["confidence_budget_gate", "hash_random", "predicted_delta"])]
        assert (actual.second_calls.to_numpy() == expected).all()
    assert (metric_frame["effective_budget"] >= 1.0).all()

    tests = run_internal_tests()
    role_record = {
        "outer_fold": 0,
        "role_groups": roles,
        "role_summary": role_sets,
        "oof_source": "train role only; inner OOF predict groups are checked against fit and calibration groups",
        "selection_source": "selection groups only; labels used only for development diagnostics",
        "evaluation": {"groups": roles["evaluation"], "labels_read": False, "rows_scored": 0},
    }

    tracked = [PREREG, Path(__file__), SOURCE, PROVENANCE, SPLIT_MANIFEST, OOF, OOF_DEPS, TEMPORAL_MODEL, STATS_MODEL]
    hashes = {str(path.resolve()): sha256(path) for path in tracked}
    if PREVIOUS_RUN_DIR.exists():
        assert_same_frame(train_frame, PREVIOUS_RUN_DIR / "train_oof_predictions.csv.gz", compressed=True)
        assert_same_frame(selection_frame, PREVIOUS_RUN_DIR / "selection_predictions.csv.gz", compressed=True)
        assert_same_frame(group_frame, PREVIOUS_RUN_DIR / "group_diagnostics.csv")
        assert_same_frame(e0_overlap_frame, PREVIOUS_RUN_DIR / "e0_error_overlap.csv")
        assert_same_frame(metric_frame, PREVIOUS_RUN_DIR / "selection_policy_metrics.csv")
        previous_summary = json.loads((PREVIOUS_RUN_DIR / "summary.json").read_text(encoding="utf-8"))
        assert previous_summary["train_delta_support"] == {str(v): int((train_values["delta"] == v).sum()) for v in (-1, 0, 1)}
        assert previous_summary["selection_delta_support"] == {str(v): int((selection_values["delta"] == v).sum()) for v in (-1, 0, 1)}
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    train_frame.to_csv(RUN_DIR / "train_oof_predictions.csv.gz", index=False, compression="gzip")
    selection_frame.to_csv(RUN_DIR / "selection_predictions.csv.gz", index=False, compression="gzip")
    group_frame.to_csv(RUN_DIR / "group_diagnostics.csv", index=False)
    e0_overlap_frame.to_csv(RUN_DIR / "e0_error_overlap.csv", index=False)
    metric_frame.to_csv(RUN_DIR / "selection_policy_metrics.csv", index=False)
    write_json(RUN_DIR / "split_roles.json", role_record)
    write_json(RUN_DIR / "input_hashes.json", hashes)
    write_json(RUN_DIR / "internal_tests.json", tests)
    joblib.dump(gain_model, RUN_DIR / "gain_model.joblib")
    summary = {
        "status": "COMPLETED_DEVELOPMENT_DIAGNOSTIC",
        "protocol_id": prereg["protocol_id"],
        "outer_fold": 0,
        "train_oof_rows": int(len(train_frame)),
        "selection_rows": int(len(selection_frame)),
        "evaluation_rows_scored": 0,
        "evaluation_labels_materialized": False,
        "evaluation_labels_used": False,
        "regressor_fit_once_on_train_oof": True,
        "router_state_columns": prereg["router_state"]["features"],
        "second_view_used_only_for_offline_delta_and_selection_reporting": True,
        "train_delta_support": {str(v): int((train_values["delta"] == v).sum()) for v in (-1, 0, 1)},
        "selection_delta_support": {str(v): int((selection_values["delta"] == v).sum()) for v in (-1, 0, 1)},
        "e0_train_error_overlap": {column: int(e0_overlap_frame.loc[e0_overlap_frame.partition == "train_oof", column].sum()) for column in ("both_correct", "only_temporal_correct", "only_stats_correct", "both_wrong", "disagreement")},
        "e0_selection_error_overlap": {column: int(e0_overlap_frame.loc[e0_overlap_frame.partition == "selection", column].sum()) for column in ("both_correct", "only_temporal_correct", "only_stats_correct", "both_wrong", "disagreement")},
        "selection_groups": sorted(set(selection_group)),
        "selection_budget_rows": int(quota_indices(len(selection_y), 1.1)),
        "input_hashes_saved": True,
        "role_checks_passed": True,
        "matched_budget_checks_passed": True,
        "pilot_003_row_probability_prediction_metric_equality": bool(PREVIOUS_RUN_DIR.exists()),
        "internal_tests": tests,
        "fresh_blind_test": False,
        "real_end_to_end_latency_measured": False,
        "final_generalization_claim": False,
        "negative_damage_zero_is_not_a_safety_claim": True,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(RUN_DIR / "summary.json", summary)
    log.extend([
        f"train OOF rows: {len(train_frame)}",
        f"selection rows: {len(selection_frame)}",
        f"train delta support: {summary['train_delta_support']}",
        f"selection delta support: {summary['selection_delta_support']}",
        "regressor: HistGradientBoostingRegressor fixed preregistered parameters",
        "selection policy metrics computed for all prespecified policies and budgets; no policy was promoted to a final method",
    ])
    (RUN_DIR / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
