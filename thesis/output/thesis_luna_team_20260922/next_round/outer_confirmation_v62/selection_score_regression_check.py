"""Recompute frozen v60/v61 selection scores only; never opens outer evaluation files."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[4]
V60 = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade"
V61 = ROOT / "output/thesis_luna_team_20260922/next_round/jev_gain_v61"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
OUT = Path(__file__).resolve().parent / "selection_score_regression_check.json"


def entropy_log2(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def aligned_proba(model, x: np.ndarray) -> np.ndarray:
    z = np.zeros((len(x), 3), dtype=float)
    for j, cls in enumerate(model.classes_.astype(int)):
        z[:, int(np.where(np.array([-1, 0, 1]) == cls)[0][0])] = model.predict_proba(x)[:, j]
    return z


def main() -> None:
    train = pd.read_csv(V60 / "train_predictions.csv.gz")
    selection = pd.read_csv(V60 / "selection_predictions.csv.gz")
    saved = pd.read_csv(V61 / "results/selection_classifier_probabilities.csv.gz")
    with np.load(SOURCE, allow_pickle=False) as z:
        x_hybrid = z["x_hybrid"].copy()
        source_hash = z["sample_hash"].astype(str).copy()
        source_group = z["group"].astype(str).copy()
    idx = {h: i for i, h in enumerate(source_hash)}
    errors = []
    fold_rows = []
    for fold in range(5):
        se = selection[selection.fold == fold].copy().reset_index(drop=True)
        sv = saved[saved.fold == fold].copy().set_index("sample_hash").loc[se.sample_hash.astype(str)]
        positions = np.array([idx[h] for h in se.sample_hash.astype(str)], dtype=int)
        p = se.p_temporal.to_numpy(float)
        x3 = np.column_stack((p, np.abs(p - .5), entropy_log2(p)))
        x43 = np.column_stack((x3, x_hybrid[positions, 8:]))
        with threadpool_limits(limits=1):
            r3 = joblib.load(V60 / "models" / f"fold_{fold}_probability_only_gain.joblib").predict(x3)
            r43 = joblib.load(V60 / "models" / f"fold_{fold}_post_first_temporal_gain.joblib").predict(x43)
            c3 = aligned_proba(joblib.load(V61 / "models" / f"fold_{fold}_probability_only_3d.joblib"), x3)
            c43 = aligned_proba(joblib.load(V61 / "models" / f"fold_{fold}_post_first_temporal_43d.joblib"), x43)
        comparisons = {
            "v60_regression_3d": (r3, se.predicted_delta.to_numpy(float)),
            "v60_regression_43d": (r43, se.predicted_delta_post_first_temporal.to_numpy(float)),
            "v61_classifier_3d_score": (c3[:, 2] - c3[:, 0], sv.probability_only_3d_score_plus_minus.to_numpy(float)),
            "v61_classifier_43d_score": (c43[:, 2] - c43[:, 0], sv.post_first_temporal_43d_score_plus_minus.to_numpy(float)),
            "v61_classifier_3d_p_minus": (c3[:, 0], sv["probability_only_3d_p_delta_-1"].to_numpy(float)),
            "v61_classifier_3d_p_zero": (c3[:, 1], sv["probability_only_3d_p_delta_+0"].to_numpy(float)),
            "v61_classifier_3d_p_plus": (c3[:, 2], sv["probability_only_3d_p_delta_+1"].to_numpy(float)),
            "v61_classifier_43d_p_minus": (c43[:, 0], sv["post_first_temporal_43d_p_delta_-1"].to_numpy(float)),
            "v61_classifier_43d_p_zero": (c43[:, 1], sv["post_first_temporal_43d_p_delta_+0"].to_numpy(float)),
            "v61_classifier_43d_p_plus": (c43[:, 2], sv["post_first_temporal_43d_p_delta_+1"].to_numpy(float)),
        }
        fold_detail = {"fold": fold, "rows": len(se), "max_abs_error": {}}
        for name, (got, expected) in comparisons.items():
            err = float(np.max(np.abs(got - expected)))
            errors.append(err); fold_detail["max_abs_error"][name] = err
        fold_rows.append(fold_detail)
    report = {"status": "PASS" if max(errors, default=0.0) <= 1e-12 else "FAIL", "scope": "selection only; no evaluation files opened", "rows": len(selection), "folds": fold_rows, "max_abs_error": max(errors, default=0.0), "entropy": "log2", "models": 20}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
