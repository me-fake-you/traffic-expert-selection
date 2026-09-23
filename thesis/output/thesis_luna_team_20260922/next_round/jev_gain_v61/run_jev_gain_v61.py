"""Fixed five-fold three-result delta probability diagnostic.

This is a development extension of v60.  It deliberately consumes the
saved v60 train/selection rows and the feature-only SOURCE arrays; it never
loads SOURCE['y'] or any evaluation prediction/label.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
LOGS = OUT / "logs"
MODELS = OUT / "models"
V60 = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade"
PREREG = OUT / "preregistration.json"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
FOLDS = [0, 1, 2, 3, 4]
SEED = 42
BUDGETS = [1.0, 1.1, 1.25, 1.5, 2.0]
CLASSES = np.array([-1, 0, 1], dtype=int)
LAMBDA_VALUES = [0.0, 0.01, 0.05]
REPRO_PYTHON = r"USER_HOME\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

PROTOCOL_DIR = ROOT / "output/thesis_luna_team_20260922/protocol"
sys.path.insert(0, str(PROTOCOL_DIR))
from run_e0_e1_pilot import confusion_metrics, entropy, quota_indices, stable_hash_order  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics_with_changes(y: np.ndarray, pred: np.ndarray, first: np.ndarray) -> dict[str, float | int]:
    out = confusion_metrics(y, pred)
    corrected = int(((first == 0) & (y == 1) & (pred == 1)).sum() + ((first == 1) & (y == 0) & (pred == 0)).sum())
    introduced = int(((first == 1) & (y == 1) & (pred == 0)).sum() + ((first == 0) & (y == 0) & (pred == 1)).sum())
    out.update({"C": corrected, "D": introduced, "C_minus_D": corrected - introduced})
    return out


def hash_order(sample_hash: np.ndarray) -> np.ndarray:
    return stable_hash_order(sample_hash)


def exact_indices(kind: str, budget: float, p_t: np.ndarray, score: np.ndarray, sample_hash: np.ndarray) -> np.ndarray:
    k = quota_indices(len(p_t), budget)
    if k <= 0:
        return np.array([], dtype=int)
    if k >= len(p_t):
        return np.arange(len(p_t), dtype=int)
    if kind == "confidence_budget":
        order = np.array(sorted(range(len(p_t)), key=lambda i: (abs(float(p_t[i]) - 0.5), str(sample_hash[i]))), dtype=int)
    elif kind == "hashrandom_budget":
        order = hash_order(sample_hash)
    elif kind in {"probability_only_gain_regression", "post_first_temporal_gain_regression", "probability_only_gain_classifier", "post_first_temporal_gain_classifier"}:
        order = np.array(sorted(range(len(p_t)), key=lambda i: (-float(score[i]), str(sample_hash[i]))), dtype=int)
    else:
        raise ValueError(kind)
    return order[:k]


def align_proba(model: HistGradientBoostingClassifier, proba: np.ndarray) -> np.ndarray:
    out = np.zeros((len(proba), len(CLASSES)), dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        out[:, int(np.where(CLASSES == cls)[0][0])] = proba[:, j]
    return np.clip(out, 1e-15, 1.0)


def probability_metrics(y_delta: np.ndarray, proba: np.ndarray) -> dict[str, object]:
    y_idx = np.array([int(np.where(CLASSES == y)[0][0]) for y in y_delta], dtype=int)
    p_true = proba[np.arange(len(y_idx)), y_idx]
    brier = float(np.mean(np.sum((proba - np.eye(3)[y_idx]) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0))))
    out: dict[str, object] = {"multiclass_brier": brier, "nll": nll, "rows": int(len(y_delta))}
    for j, cls in enumerate(CLASSES):
        event = (y_delta == cls).astype(float)
        bins = []
        ece = 0.0
        for b in range(10):
            lo, hi = b / 10.0, (b + 1) / 10.0
            mask = (proba[:, j] >= lo) & ((proba[:, j] < hi) if b < 9 else (proba[:, j] <= hi))
            count = int(mask.sum())
            if count:
                mean_p = float(proba[mask, j].mean())
                freq = float(event[mask].mean())
                ece += count / len(event) * abs(mean_p - freq)
            else:
                mean_p, freq = None, None
            bins.append({"bin": b, "count": count, "mean_probability": mean_p, "event_rate": freq})
        out[f"class_{cls}_positive_count"] = int(event.sum())
        out[f"class_{cls}_mean_probability"] = float(proba[:, j].mean())
        out[f"class_{cls}_ece"] = float(ece)
        out[f"class_{cls}_bins"] = bins
    return out


def policy_row(fold: int, policy: str, budget: float | None, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, pred: np.ndarray, sample_hash: np.ndarray, acquired: np.ndarray, representation: str) -> tuple[dict[str, object], np.ndarray]:
    first = (p_t >= 0.5).astype(int)
    avg = (((p_t + p_s) / 2.0) >= 0.5).astype(int)
    final = first.copy()
    final[acquired] = avg[acquired]
    m = metrics_with_changes(y, final, first)
    calls = int(len(acquired))
    out = {
        "fold": fold, "seed": SEED, "policy": policy, "representation": representation,
        "target_budget": budget, "budget": budget, "rows": len(y),
        "simulated_second_calls": calls, "effective_budget": 1.0 + calls / max(len(y), 1),
        "call_rate": calls / max(len(y), 1), "mean_score_acquired": float(np.mean(pred[acquired])) if calls else 0.0,
        "C_D_reference": "first_temporal", "budget_semantics": "exact_quota" if budget is not None else "positive_score_stop",
        **m,
    }
    for lam in LAMBDA_VALUES:
        out[f"mean_delta"] = m["C_minus_D"] / max(len(y), 1)
        out[f"utility_lambda_{str(lam).replace('.', '_')}"] = out["mean_delta"] - lam * out["call_rate"]
    return out, final


def add_policy_set(rows: list[dict[str, object]], group_rows: list[dict[str, object]], fold: int, df: pd.DataFrame, p_t: np.ndarray, p_s: np.ndarray, sample_hash: np.ndarray, scores: dict[str, np.ndarray], representation: str) -> None:
    y = df["y"].to_numpy(dtype=int)
    first = (p_t >= 0.5).astype(int)
    group = df["group"].astype(str).to_numpy()
    for budget in BUDGETS:
        specs = {
            "confidence_budget": (np.array([], dtype=int), "rule"),
            "hashrandom_budget": (np.array([], dtype=int), "rule"),
        }
        for name, score in scores.items():
            specs[name] = (exact_indices(name, budget, p_t, score, sample_hash), representation)
        for name, (acquired, rep) in specs.items():
            if name.endswith("budget") and name in {"confidence_budget", "hashrandom_budget"}:
                acquired = exact_indices(name, budget, p_t, np.zeros(len(y)), sample_hash)
            row, final = policy_row(fold, name, budget, y, p_t, p_s, scores.get(name, np.zeros(len(y))), sample_hash, acquired, rep)
            row["representation"] = "v60_rule" if name in {"confidence_budget", "hashrandom_budget"} else representation
            rows.append(row)
            for g in sorted(set(group)):
                mask = group == g
                global_positions = np.flatnonzero(mask)
                local_acquired = np.flatnonzero(np.isin(global_positions, acquired))
                gm, _ = policy_row(fold, name, budget, y[mask], p_t[mask], p_s[mask], scores.get(name, np.zeros(len(y)))[mask], sample_hash[mask], local_acquired, rep)
                gm["representation"] = "v60_rule" if name in {"confidence_budget", "hashrandom_budget"} else representation
                gm.update({"group": g})
                group_rows.append(gm)
    for name, score in scores.items():
        acquired = np.flatnonzero(score > 0.0)
        row, final = policy_row(fold, name + "_positive_stop", None, y, p_t, p_s, score, sample_hash, acquired, representation)
        rows.append(row)
        for g in sorted(set(group)):
            mask = group == g
            local_global = np.flatnonzero(mask)
            local_acquired = np.flatnonzero(score[mask] > 0.0)
            gm, _ = policy_row(fold, name + "_positive_stop", None, y[mask], p_t[mask], p_s[mask], score[mask], sample_hash[mask], local_acquired, representation)
            gm.update({"group": g})
            group_rows.append(gm)


def main() -> None:
    started = time.perf_counter()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["seed"] == SEED and prereg["outer_folds"] == FOLDS
    train = pd.read_csv(V60 / "train_predictions.csv.gz")
    selection = pd.read_csv(V60 / "selection_predictions.csv.gz")
    assert set(train["partition"]) == {"train_oof"} and set(selection["partition"]) == {"selection"}
    assert len(train) == 120000 and len(selection) == 20000
    assert all((train.groupby("fold").size() == 24000).to_numpy())
    assert all((selection.groupby("fold").size() == 4000).to_numpy())
    with np.load(SOURCE, allow_pickle=False) as z:
        # Deliberately only these feature-free contract keys are loaded.
        x_hybrid = z["x_hybrid"].copy()
        source_hash = z["sample_hash"].astype(str).copy()
        source_group = z["group"].astype(str).copy()
    assert len(source_hash) == len(source_group) == len(x_hybrid) == 40000
    assert len(set(source_hash.tolist())) == len(source_hash)
    hash_to_idx = {h: i for i, h in enumerate(source_hash)}
    for frame in (train, selection):
        assert set(frame["sample_hash"].astype(str)).issubset(hash_to_idx)
        source_group_by_hash = frame["sample_hash"].astype(str).map(dict(zip(source_hash, source_group))).to_numpy()
        assert np.array_equal(source_group_by_hash, frame["group"].astype(str).to_numpy())

    policy_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    train_prob_frames: list[pd.DataFrame] = []
    selection_prob_frames: list[pd.DataFrame] = []
    logs = []
    for fold in FOLDS:
        tr = train[train.fold == fold].copy().reset_index(drop=True)
        se = selection[selection.fold == fold].copy().reset_index(drop=True)
        tr_idx = np.array([hash_to_idx[h] for h in tr.sample_hash.astype(str)], dtype=int)
        se_idx = np.array([hash_to_idx[h] for h in se.sample_hash.astype(str)], dtype=int)
        x3_train = np.column_stack((tr.p_temporal, np.abs(tr.p_temporal - 0.5), entropy(tr.p_temporal.to_numpy(dtype=float))))
        x3_selection = np.column_stack((se.p_temporal, np.abs(se.p_temporal - 0.5), entropy(se.p_temporal.to_numpy(dtype=float))))
        x43_train = np.column_stack((x3_train, x_hybrid[tr_idx, 8:]))
        x43_selection = np.column_stack((x3_selection, x_hybrid[se_idx, 8:]))
        y_train = tr.delta.to_numpy(dtype=int)
        y_selection = se.delta.to_numpy(dtype=int)
        assert set(np.unique(y_train)) == {-1, 0, 1}
        priors = np.array([(y_train == cls).mean() for cls in CLASSES], dtype=float)
        prior_proba = np.tile(priors, (len(se), 1))
        probability_rows.append({"fold": fold, "representation": "train_prior", **probability_metrics(y_selection, prior_proba)})
        model_scores: dict[str, np.ndarray] = {}
        train_prob_dict: dict[str, np.ndarray] = {}
        selection_prob_dict: dict[str, np.ndarray] = {}
        for variant, xtr, xse in (("probability_only_3d", x3_train, x3_selection), ("post_first_temporal_43d", x43_train, x43_selection)):
            t0 = time.perf_counter()
            with threadpool_limits(limits=1):
                model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=2.0, random_state=20260922, early_stopping=False)
                model.fit(xtr, y_train)
            fit_seconds = time.perf_counter() - t0
            path = MODELS / f"fold_{fold}_{variant}.joblib"
            joblib.dump(model, path)
            p_train = align_proba(model, model.predict_proba(xtr))
            p_selection = align_proba(model, model.predict_proba(xse))
            train_prob_dict[variant] = p_train
            selection_prob_dict[variant] = p_selection
            score = p_selection[:, 2] - p_selection[:, 0]
            model_scores[variant] = score
            probability_rows.append({"fold": fold, "representation": variant, **probability_metrics(y_selection, p_selection)})
            model_rows.append({"fold": fold, "representation": variant, "fit_rows": len(xtr), "features": xtr.shape[1], "fit_seconds": fit_seconds, "n_iter": int(getattr(model, "n_iter_", -1)), "n_trees_per_iteration": int(getattr(model, "n_trees_per_iteration_", -1)), "model_bytes": path.stat().st_size, "classes": model.classes_.astype(int).tolist()})
        # Full train/selection probabilities are kept for audit and later analysis.
        def prob_frame(frame: pd.DataFrame, p3: np.ndarray, p43: np.ndarray) -> pd.DataFrame:
            out = frame[["fold", "partition", "sample_hash", "group", "delta"]].copy()
            for prefix, p in (("probability_only_3d", p3), ("post_first_temporal_43d", p43)):
                for j, cls in enumerate(CLASSES):
                    out[f"{prefix}_p_delta_{cls:+d}"] = p[:, j]
                out[f"{prefix}_score_plus_minus"] = p[:, 2] - p[:, 0]
            return out
        train_prob_frames.append(prob_frame(tr, train_prob_dict["probability_only_3d"], train_prob_dict["post_first_temporal_43d"]))
        selection_prob_frames.append(prob_frame(se, selection_prob_dict["probability_only_3d"], selection_prob_dict["post_first_temporal_43d"]))
        scores = {
            "probability_only_gain_classifier": model_scores["probability_only_3d"],
            "post_first_temporal_gain_classifier": model_scores["post_first_temporal_43d"],
            "probability_only_gain_regression": se.predicted_delta.to_numpy(dtype=float),
            "post_first_temporal_gain_regression": se.predicted_delta_post_first_temporal.to_numpy(dtype=float),
        }
        # Rules and v60 regressors are recomputed on the same saved selection rows.
        add_policy_set(policy_rows, group_rows, fold, se, se.p_temporal.to_numpy(float), se.p_stats_offline_target_only.to_numpy(float), se.sample_hash.astype(str).to_numpy(), scores, "v60_rule_or_gain")
        logs.append(f"fold={fold} train=24000 selection=4000 model_fit_rows=24000")

    # Add explicit v60 baselines to the aligned panel once per fold/budget.
    for fold in FOLDS:
        se = selection[selection.fold == fold].copy().reset_index(drop=True)
        p_t, p_s, sh, y = se.p_temporal.to_numpy(float), se.p_stats_offline_target_only.to_numpy(float), se.sample_hash.astype(str).to_numpy(), se.y.to_numpy(int)
        for budget in BUDGETS:
            for name in ("first_temporal", "first_stats", "all_average"):
                if name == "first_temporal": acquired = np.array([], dtype=int); pred = np.zeros(len(se))
                elif name == "first_stats": acquired = np.array([], dtype=int); pred = np.zeros(len(se))
                else: acquired = np.arange(len(se)); pred = np.zeros(len(se))
                final = (p_t >= 0.5).astype(int) if name == "first_temporal" else ((p_s >= 0.5).astype(int) if name == "first_stats" else (((p_t + p_s) / 2) >= 0.5).astype(int))
                m = metrics_with_changes(y, final, (p_t >= 0.5).astype(int)); calls = len(acquired)
                row = {"fold": fold, "seed": SEED, "policy": name, "representation": "v60_baseline", "target_budget": budget, "budget": budget, "rows": len(se), "simulated_second_calls": calls, "effective_budget": 1 + calls / len(se), "call_rate": calls / len(se), "mean_score_acquired": 0.0, "C_D_reference": "first_temporal", "budget_semantics": "repeated_reference_panel", **m}
                row["mean_delta"] = m["C_minus_D"] / len(se)
                for lam in LAMBDA_VALUES: row[f"utility_lambda_{str(lam).replace('.', '_')}"] = row["mean_delta"] - lam * row["call_rate"]
                policy_rows.append(row)
    train_prob = pd.concat(train_prob_frames, ignore_index=True)
    selection_prob = pd.concat(selection_prob_frames, ignore_index=True)
    train_prob.to_csv(RESULTS / "train_classifier_probabilities.csv.gz", index=False, compression="gzip")
    selection_prob.to_csv(RESULTS / "selection_classifier_probabilities.csv.gz", index=False, compression="gzip")
    pd.DataFrame(policy_rows).sort_values(["fold", "policy", "target_budget"], na_position="last").to_csv(RESULTS / "policy_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(RESULTS / "group_policy_metrics.csv", index=False)
    pd.DataFrame(probability_rows).to_json(RESULTS / "probability_metrics.json", orient="records", force_ascii=False, indent=2)
    pd.DataFrame(model_rows).to_csv(RESULTS / "model_training_metrics.csv", index=False)
    dump(RESULTS / "model_training_metrics.json", model_rows)
    input_paths = [PREREG, Path(__file__), V60 / "train_predictions.csv.gz", V60 / "selection_predictions.csv.gz", V60 / "policy_metrics.csv", V60 / "input_hashes.json", SOURCE]
    dump(OUT / "input_hashes.json", {str(p.resolve()): sha256(p) for p in input_paths})
    exact_quota_ok = True
    for fold in FOLDS:
        se_check = selection[selection.fold == fold]
        p_check = se_check.p_temporal.to_numpy(dtype=float)
        h_check = se_check.sample_hash.astype(str).to_numpy()
        for budget in BUDGETS:
            expected = quota_indices(len(se_check), budget)
            exact_quota_ok = exact_quota_ok and len(exact_indices("confidence_budget", budget, p_check, np.zeros(len(se_check)), h_check)) == expected
            exact_quota_ok = exact_quota_ok and len(exact_indices("hashrandom_budget", budget, p_check, np.zeros(len(se_check)), h_check)) == expected
            exact_quota_ok = exact_quota_ok and len(exact_indices("probability_only_gain_classifier", budget, p_check, np.zeros(len(se_check)), h_check)) == expected
    boundary = {
        "source_y_not_loaded": True,
        "evaluation_not_read_or_scored": True,
        "train_rows_each_fold_24000": True,
        "selection_rows_each_fold_4000": True,
        "exact_quota_all_budgeted_policies": exact_quota_ok,
        "three_classes_present_each_train_fold": True,
        "no_probability_calibrator_fit": True,
        "models_count_10": len(model_rows) == 10,
        "classifier_state_3d_or_43d_only": True,
    }
    dump(OUT / "boundary_tests.json", boundary)
    assert all(boundary.values())
    dump(RESULTS / "summary.json", {"status": "COMPLETED_FIXED_FIVEFOLD_THREE_RESULT_PROBABILITY_DIAGNOSTIC", "seed": SEED, "folds": FOLDS, "train_rows": len(train), "selection_rows": len(selection), "evaluation_read": False, "models": len(model_rows), "budgets": BUDGETS, "lambdas": LAMBDA_VALUES, "v60_results_exposed": True, "not_jev_or_rlcd_reproduction": True, "duration_seconds": time.perf_counter() - started, "reproduction_command": f'"{REPRO_PYTHON}" "{Path(__file__).resolve()}"'})
    (LOGS / "run_log.txt").write_text("\n".join(logs) + "\n" + json.dumps({"duration_seconds": time.perf_counter() - started, "evaluation_read": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "models": len(model_rows), "train_rows": len(train), "selection_rows": len(selection), "evaluation_read": False, "duration_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
