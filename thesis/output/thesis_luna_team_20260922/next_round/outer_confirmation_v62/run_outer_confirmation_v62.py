"""Execute only after root locks outer_confirmation_v62 preregistration.

The default invocation is fail-closed.  Use --execute after review.  No model
is fitted in this script; it only loads frozen gain models and saved outer
prediction files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
LOGS = OUT / "logs"
MODELS_OUT = OUT / "models"
PREREG = OUT / "preregistration.json"
ROOT_LOCK = OUT / "root_lock.json"
OUTER = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
PROVENANCE = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/raw_audit_001/selected_provenance.csv"
V60_MODELS = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade/models"
V61_MODELS = ROOT / "output/thesis_luna_team_20260922/next_round/jev_gain_v61/models"
FOLDS = [0, 1, 2, 3, 4]
BUDGETS = [1.0, 1.1, 1.25, 1.5, 2.0]
LAMBDAS = [0.0, 0.01, 0.05]
CLASSES = np.array([-1, 0, 1], dtype=int)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def quota(n: int, budget: float) -> int:
    return int(np.floor((budget - 1.0) * n + 0.5))


def hash_order(sample_hash: np.ndarray) -> np.ndarray:
    return np.array(sorted(range(len(sample_hash)), key=lambda i: hashlib.sha256(str(sample_hash[i]).encode("utf-8")).hexdigest()), dtype=int)


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def metrics(y: np.ndarray, pred: np.ndarray, first: np.ndarray) -> dict[str, object]:
    y = y.astype(int); pred = pred.astype(int); first = first.astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    c = int(((first == 0) & (y == 1) & (pred == 1)).sum() + ((first == 1) & (y == 0) & (pred == 0)).sum())
    d = int(((first == 1) & (y == 1) & (pred == 0)).sum() + ((first == 0) & (y == 0) & (pred == 1)).sum())
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)), "malicious_recall": (float(tp / (tp + fn)) if (tp + fn) else None), "false_positive_rate": (float(fp / (fp + tn)) if (fp + tn) else None), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "C": c, "D": d, "C_minus_D": c - d}


def acquire_indices(name: str, budget: float, p_t: np.ndarray, score: np.ndarray, sample_hash: np.ndarray) -> np.ndarray:
    k = quota(len(p_t), budget)
    if k <= 0: return np.array([], dtype=int)
    if k >= len(p_t): return np.arange(len(p_t), dtype=int)
    if name == "confidence_budget": order = np.array(sorted(range(len(p_t)), key=lambda i: (abs(float(p_t[i]) - .5), str(sample_hash[i]))), dtype=int)
    elif name == "hashrandom_budget": order = hash_order(sample_hash)
    else: order = np.array(sorted(range(len(p_t)), key=lambda i: (-float(score[i]), str(sample_hash[i]))), dtype=int)
    return order[:k]


def aligned_classifier_proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        out[:, int(np.where(CLASSES == cls)[0][0])] = proba[:, j]
    return out


def policy_row(fold: int, policy: str, representation: str, budget: float | None, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, sample_hash: np.ndarray, acquired: np.ndarray, scope: str, base: str = "temporal") -> tuple[dict[str, object], np.ndarray]:
    first_temporal = (p_t >= .5).astype(int)
    first = (p_s >= .5).astype(int) if base == "stats" else first_temporal.copy()
    avg = (((p_t + p_s) / 2) >= .5).astype(int)
    pred = first.copy(); pred[acquired] = avg[acquired]
    m = metrics(y, pred, first_temporal); calls = len(acquired); rate = calls / len(y); mean_delta = m["C_minus_D"] / len(y)
    row = {"fold": fold, "policy": policy, "representation": representation, "target_budget": budget, "budget": budget, "rows": len(y), "simulated_second_calls": calls, "effective_budget": 1 + rate, "call_rate": rate, "C_D_reference": "first_temporal", "budget_semantics": "exact_quota" if budget is not None else "positive_score_stop", "scope": scope, **m, "mean_delta": mean_delta}
    for lam in LAMBDAS: row[f"utility_lambda_{str(lam).replace('.', '_')}"] = mean_delta - lam * rate
    return row, pred


def add_group_rows(rows: list[dict[str, object]], fold: int, policy: str, rep: str, budget: float | None, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, h: np.ndarray, acquired: np.ndarray, groups: np.ndarray, scope: str, base: str = "temporal") -> None:
    for group in sorted(set(groups.astype(str))):
        mask = groups.astype(str) == group
        global_pos = np.flatnonzero(mask)
        local_acquired = np.flatnonzero(np.isin(global_pos, acquired))
        row, _ = policy_row(fold, policy, rep, budget, y[mask], p_t[mask], p_s[mask], h[mask], local_acquired, scope, base=base)
        row["group"] = group
        rows.append(row)


def safe_ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Protocol is pending root lock. Re-run with --execute only after preregistration review.")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["status"] == "locked_by_root_review"
    started = time.perf_counter()
    RESULTS.mkdir(exist_ok=True); LOGS.mkdir(exist_ok=True); MODELS_OUT.mkdir(exist_ok=True)
    model_paths = []
    for fold in FOLDS:
        model_paths.extend([V60_MODELS / f"fold_{fold}_probability_only_gain.joblib", V60_MODELS / f"fold_{fold}_post_first_temporal_gain.joblib", V61_MODELS / f"fold_{fold}_probability_only_3d.joblib", V61_MODELS / f"fold_{fold}_post_first_temporal_43d.joblib"])
    eval_paths = [OUTER / f"seed_42_fold_{fold}" / "temporal_stats_predictions.csv" for fold in FOLDS]
    lock_paths = [SOURCE, PROVENANCE, OUTER / "run_manifest.json", OUTER / "split_manifest.json", PREREG, Path(__file__), *eval_paths, *model_paths]
    lock = json.loads(ROOT_LOCK.read_text(encoding="utf-8"))
    assert lock["status"] == "locked_by_root" and lock["protocol_id"] == prereg["protocol_id"]
    expected_hashes = {str(p.resolve()): sha256(p) for p in lock_paths}
    assert lock["hashes"] == expected_hashes
    split_manifest = json.loads((OUTER / "split_manifest.json").read_text(encoding="utf-8"))
    assert len(split_manifest) == 5
    evaluation_groups_by_fold = {}
    role_groups_by_fold = {}
    for entry in split_manifest:
        role_groups = entry["groups"]
        eval_groups = set(role_groups["evaluation"])
        assert eval_groups and eval_groups.isdisjoint(role_groups["train"]) and eval_groups.isdisjoint(role_groups["calibration"]) and eval_groups.isdisjoint(role_groups["selection"])
        evaluation_groups_by_fold[int(entry["outer_fold"])] = eval_groups
        role_groups_by_fold[int(entry["outer_fold"])] = role_groups
    with np.load(SOURCE, allow_pickle=False) as z:
        x_hybrid = z["x_hybrid"].copy()
        source_hash = z["sample_hash"].astype(str).copy()
        source_group = z["group"].astype(str).copy()
    assert len(source_hash) == len(set(source_hash.tolist())) == len(source_group) == len(x_hybrid)
    source_idx = {h: i for i, h in enumerate(source_hash)}
    model_manifest = []
    for fold in FOLDS:
        for root, names in ((V60_MODELS, [f"fold_{fold}_probability_only_gain.joblib", f"fold_{fold}_post_first_temporal_gain.joblib"]), (V61_MODELS, [f"fold_{fold}_probability_only_3d.joblib", f"fold_{fold}_post_first_temporal_43d.joblib"])):
            for name in names:
                p = root / name; assert p.exists()
                model_manifest.append({"fold": fold, "path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size})
    rows: list[dict[str, object]] = []; group_rows: list[dict[str, object]] = []; outer_frames = []; logs = []
    evaluation_role_checks = []
    probability_checks = []
    for fold in FOLDS:
        path = OUTER / f"seed_42_fold_{fold}" / "temporal_stats_predictions.csv"
        df = pd.read_csv(path)
        required = {"sample_hash", "group", "label", "first_probability", "second_probability", "first_only", "second_only", "equal_average"}
        assert required.issubset(df.columns) and len(df) == 8000
        h = df.sample_hash.astype(str).to_numpy(); groups = df.group.astype(str).to_numpy(); y = df.label.to_numpy(int)
        assert set(h).issubset(source_idx)
        assert np.array_equal(np.array([source_group[source_idx[x]] for x in h]), groups)
        evaluation_role_checks.append(set(groups) == evaluation_groups_by_fold[fold] and set(groups).isdisjoint(set(role_groups_by_fold[fold]["train"])) and set(groups).isdisjoint(set(role_groups_by_fold[fold]["calibration"])) and set(groups).isdisjoint(set(role_groups_by_fold[fold]["selection"])))
        idx = np.array([source_idx[x] for x in h], dtype=int)
        p_t = df.first_probability.to_numpy(float); p_s = df.second_probability.to_numpy(float)
        first_saved = df.first_only.to_numpy(int); second_saved = df.second_only.to_numpy(int); average_saved = df.equal_average.to_numpy(int)
        probability_checks.append(bool(np.isfinite(p_t).all() and np.isfinite(p_s).all() and (p_t >= 0).all() and (p_t <= 1).all() and (p_s >= 0).all() and (p_s <= 1).all() and set(np.unique(y)).issubset({0, 1}) and np.array_equal(first_saved, (p_t >= .5).astype(int)) and np.array_equal(second_saved, (p_s >= .5).astype(int)) and np.array_equal(average_saved, (((p_t + p_s) / 2) >= .5).astype(int))))
        x3 = np.column_stack((p_t, np.abs(p_t - .5), entropy(p_t)))
        x43 = np.column_stack((x3, x_hybrid[idx, 8:]))
        classifier_probabilities = {
            "probability_only_gain_classifier": aligned_classifier_proba(joblib.load(V61_MODELS / f"fold_{fold}_probability_only_3d.joblib"), x3),
            "post_first_temporal_gain_classifier": aligned_classifier_proba(joblib.load(V61_MODELS / f"fold_{fold}_post_first_temporal_43d.joblib"), x43),
        }
        scores = {
            "probability_only_gain_regression": joblib.load(V60_MODELS / f"fold_{fold}_probability_only_gain.joblib").predict(x3),
            "post_first_temporal_gain_regression": joblib.load(V60_MODELS / f"fold_{fold}_post_first_temporal_gain.joblib").predict(x43),
            "probability_only_gain_classifier": classifier_probabilities["probability_only_gain_classifier"][:, 2] - classifier_probabilities["probability_only_gain_classifier"][:, 0],
            "post_first_temporal_gain_classifier": classifier_probabilities["post_first_temporal_gain_classifier"][:, 2] - classifier_probabilities["post_first_temporal_gain_classifier"][:, 0],
        }
        for name, score in scores.items():
            for budget in BUDGETS:
                acquired = acquire_indices(name, budget, p_t, score, h)
                row, pred = policy_row(fold, name, "frozen_gain_model", budget, y, p_t, p_s, h, acquired, "outer_role_re-evaluation; no refit")
                rows.append(row); add_group_rows(group_rows, fold, name, "frozen_gain_model", budget, y, p_t, p_s, h, acquired, groups, "outer_role_re-evaluation; no refit")
            acquired = np.flatnonzero(score > 0)
            row, pred = policy_row(fold, name + "_positive_stop", "frozen_gain_model", None, y, p_t, p_s, h, acquired, "outer_role_re-evaluation; no refit")
            rows.append(row); add_group_rows(group_rows, fold, name + "_positive_stop", "frozen_gain_model", None, y, p_t, p_s, h, acquired, groups, "outer_role_re-evaluation; no refit")
        for name in ["first_temporal", "first_stats", "all_average", "confidence_budget", "hashrandom_budget"]:
            for budget in BUDGETS:
                if name == "first_temporal": acquired = np.array([], dtype=int)
                elif name == "first_stats": acquired = np.array([], dtype=int)
                elif name == "all_average": acquired = np.arange(len(df))
                else: acquired = acquire_indices(name, budget, p_t, np.zeros(len(df)), h)
                base = "stats" if name == "first_stats" else "temporal"
                row, _ = policy_row(fold, name, "frozen_rule", budget, y, p_t, p_s, h, acquired, "outer_role_re-evaluation; no refit", base=base)
                rows.append(row); add_group_rows(group_rows, fold, name, "frozen_rule", budget, y, p_t, p_s, h, acquired, groups, "outer_role_re-evaluation; no refit", base=base)
        h_temporal = (p_t >= .5).astype(int); h_average = (((p_t + p_s) / 2) >= .5).astype(int)
        out = df[["sample_hash", "group", "label"]].copy(); out["fold"] = fold; out["p_temporal"] = p_t; out["p_stats"] = p_s; out["delta"] = (h_temporal != y).astype(int) - (h_average != y).astype(int)
        out["score_probability_only_gain_regression"] = scores["probability_only_gain_regression"]; out["score_post_first_temporal_gain_regression"] = scores["post_first_temporal_gain_regression"]
        for name, proba in classifier_probabilities.items():
            out[f"{name}_p_delta_-1"] = proba[:, 0]; out[f"{name}_p_delta_0"] = proba[:, 1]; out[f"{name}_p_delta_+1"] = proba[:, 2]; out[f"{name}_score_plus_minus"] = proba[:, 2] - proba[:, 0]
        outer_frames.append(out)
        logs.append(f"fold={fold} rows={len(df)} unique_hash={df.sample_hash.nunique()} groups={sorted(df.group.unique().tolist())}")
    policy = pd.DataFrame(rows); groups = pd.DataFrame(group_rows); outer = pd.concat(outer_frames, ignore_index=True)
    policy.to_csv(RESULTS / "policy_metrics.csv", index=False); groups.to_csv(RESULTS / "group_policy_metrics.csv", index=False); outer.to_csv(RESULTS / "outer_predictions.csv.gz", index=False, compression="gzip")
    pooled = policy.groupby(["policy", "representation", "target_budget", "budget"], dropna=False, as_index=False).agg({"rows": "sum", "simulated_second_calls": "sum", "C": "sum", "D": "sum", "C_minus_D": "sum", "tn": "sum", "fp": "sum", "fn": "sum", "tp": "sum"})
    pooled["effective_budget"] = 1 + pooled.simulated_second_calls / pooled.rows; pooled["call_rate"] = pooled.simulated_second_calls / pooled.rows; pooled["mean_delta"] = pooled.C_minus_D / pooled.rows
    pooled["macro_f1"] = [float(((2 * r.tn / max(2 * r.tn + r.fp + r.fn, 1)) + (2 * r.tp / max(2 * r.tp + r.fp + r.fn, 1))) / 2) for _, r in pooled.iterrows()]
    pooled["malicious_recall"] = [safe_ratio(r.tp, r.tp + r.fn) for _, r in pooled.iterrows()]; pooled["false_positive_rate"] = [safe_ratio(r.fp, r.fp + r.tn) for _, r in pooled.iterrows()]
    for lam in LAMBDAS: pooled[f"utility_lambda_{str(lam).replace('.', '_')}"] = pooled.mean_delta - lam * pooled.call_rate
    pooled.to_csv(RESULTS / "pooled_policy_metrics.csv", index=False)
    pooled_groups = groups.groupby(["group", "policy", "representation", "target_budget", "budget"], dropna=False, as_index=False).agg({"rows": "sum", "simulated_second_calls": "sum", "C": "sum", "D": "sum", "C_minus_D": "sum", "tn": "sum", "fp": "sum", "fn": "sum", "tp": "sum"})
    pooled_groups["effective_budget"] = 1 + pooled_groups.simulated_second_calls / pooled_groups.rows; pooled_groups["call_rate"] = pooled_groups.simulated_second_calls / pooled_groups.rows; pooled_groups["mean_delta"] = pooled_groups.C_minus_D / pooled_groups.rows
    pooled_groups["malicious_recall"] = [safe_ratio(r.tp, r.tp + r.fn) for _, r in pooled_groups.iterrows()]; pooled_groups["false_positive_rate"] = [safe_ratio(r.fp, r.fp + r.tn) for _, r in pooled_groups.iterrows()]
    pooled_groups.to_csv(RESULTS / "pooled_group_policy_metrics.csv", index=False)
    tracked = [PREREG, Path(__file__), SOURCE, PROVENANCE, OUTER / "run_manifest.json", OUTER / "split_manifest.json"] + [OUTER / f"seed_42_fold_{f}" / "temporal_stats_predictions.csv" for f in FOLDS] + [Path(x["path"]) for x in model_manifest]
    dump(OUT / "input_hashes.json", {str(p.resolve()): sha256(p) for p in tracked})
    dump(MODELS_OUT / "model_manifest.json", model_manifest)
    boundary = {"outer_rows_40000": len(outer) == 40000, "outer_hash_unique": outer.sample_hash.nunique() == 40000, "fold_rows_8000": all((outer.groupby("fold").size() == 8000).to_numpy()), "no_refit": True, "no_source_y_access": True, "models_loaded_20": len(model_manifest) == 20, "evaluation_role_matches_manifest": all(evaluation_role_checks), "evaluation_probabilities_finite_binary_and_saved_predictions_consistent": all(probability_checks)}
    dump(OUT / "boundary_tests.json", boundary); assert all(boundary.values())
    dump(RESULTS / "summary.json", {"status": "COMPLETED_OUTER_ROLE_REEVALUATION_V62", "seed": 42, "folds": FOLDS, "rows": len(outer), "unique_hash": int(outer.sample_hash.nunique()), "not_blind": True, "not_independent_confirmation": True, "no_refit": True, "duration_seconds": time.perf_counter() - started})
    (LOGS / "run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "rows": len(outer), "unique_hash": int(outer.sample_hash.nunique())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
