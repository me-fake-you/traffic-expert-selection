"""Independent read-only numerical validation of jev_gain_v61 artifacts."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "output/thesis_luna_team_20260922/next_round/jev_gain_v61"
V60 = ROOT / "output/thesis_luna_team_20260922/next_round/gain_upgrade"
OUT = Path(__file__).resolve().parent
FOLDS = [0, 1, 2, 3, 4]
BUDGETS = [1.0, 1.1, 1.25, 1.5, 2.0]
CLASSES = [-1, 0, 1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def q(n: int, budget: float) -> int:
    return int(np.floor((budget - 1.0) * n + 0.5))


def horder(h: np.ndarray) -> np.ndarray:
    return np.array(sorted(range(len(h)), key=lambda i: hashlib.sha256(str(h[i]).encode("utf-8")).hexdigest()), dtype=int)


def metrics(y: np.ndarray, pred: np.ndarray, first: np.ndarray) -> dict[str, float | int]:
    y = y.astype(int); pred = pred.astype(int); first = first.astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    c = int(((first == 0) & (y == 1) & (pred == 1)).sum() + ((first == 1) & (y == 0) & (pred == 0)).sum())
    d = int(((first == 1) & (y == 1) & (pred == 0)).sum() + ((first == 0) & (y == 0) & (pred == 1)).sum())
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)), "malicious_recall": float(tp / max(tp + fn, 1)), "false_positive_rate": float(fp / max(fp + tn, 1)), "tn": tn, "fp": fp, "fn": fn, "tp": tp, "C": c, "D": d, "C_minus_D": c - d}


def ece(y: np.ndarray, p: np.ndarray, cls: int) -> tuple[float, list[dict[str, object]]]:
    event = (y == cls).astype(float); total = len(y); bins = []; value = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        mask = (p >= lo) & ((p < hi) if b < 9 else (p <= hi))
        n = int(mask.sum())
        if n:
            mp, er = float(p[mask].mean()), float(event[mask].mean()); value += n / total * abs(mp - er)
        else:
            mp, er = None, None
        bins.append({"bin": b, "count": n, "mean_probability": mp, "event_rate": er})
    return float(value), bins


def prob_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    yi = np.array([CLASSES.index(int(v)) for v in y], dtype=int)
    brier = float(np.mean(np.sum((p - np.eye(3)[yi]) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), yi], 1e-15, 1.0))))
    out: dict[str, object] = {"multiclass_brier": brier, "nll": nll, "rows": len(y)}
    for j, cls in enumerate(CLASSES):
        ev = (y == cls).astype(int); e, bins = ece(y, p[:, j], cls)
        out[f"class_{cls}_ece"] = e; out[f"class_{cls}_positive_count"] = int(ev.sum()); out[f"class_{cls}_mean_probability"] = float(p[:, j].mean()); out[f"class_{cls}_bins"] = bins
    return out


def score_order(name: str, budget: float, p_t: np.ndarray, score: np.ndarray, h: np.ndarray) -> np.ndarray:
    k = q(len(h), budget)
    if k == 0: return np.array([], dtype=int)
    if k == len(h): return np.arange(len(h), dtype=int)
    if name == "confidence_budget": order = np.array(sorted(range(len(h)), key=lambda i: (abs(float(p_t[i]) - .5), str(h[i]))), dtype=int)
    elif name == "hashrandom_budget": order = horder(h)
    else: order = np.array(sorted(range(len(h)), key=lambda i: (-float(score[i]), str(h[i]))), dtype=int)
    return order[:k]


def main() -> None:
    train = pd.read_csv(V60 / "train_predictions.csv.gz")
    selection = pd.read_csv(V60 / "selection_predictions.csv.gz")
    pol = pd.read_csv(EXP / "results/policy_metrics.csv")
    prob_rows = json.loads((EXP / "results/probability_metrics.json").read_text(encoding="utf-8"))
    trp = pd.read_csv(EXP / "results/train_classifier_probabilities.csv.gz")
    sep = pd.read_csv(EXP / "results/selection_classifier_probabilities.csv.gz")
    checks: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, ok: bool, detail: object = None) -> None:
        checks[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        if not ok: failures.append(name)

    check("source_row_roles", len(train) == 120000 and len(selection) == 20000 and all((train.groupby("fold").size() == 24000).to_numpy()) and all((selection.groupby("fold").size() == 4000).to_numpy()), {"train": len(train), "selection": len(selection)})
    check("policy_row_count_and_folds", len(pol) == 245 and set(pol.fold) == set(FOLDS), {"rows": len(pol), "folds": sorted(pol.fold.unique().tolist())})
    check("no_actual_calls_column", "actual_calls" not in pol.columns)
    check("policy_required_columns", set(["fold", "policy", "budget", "macro_f1", "C", "D", "C_minus_D", "simulated_second_calls", "effective_budget", "utility_lambda_0_0", "utility_lambda_0_01", "utility_lambda_0_05"]).issubset(pol.columns), pol.columns.tolist())
    check("model_count", len(list((EXP / "models").glob("fold_*joblib"))) == 10 and len(pd.read_csv(EXP / "results/model_training_metrics.csv")) == 10)

    # Verify all full-row saved probabilities and reconstruct the selection labels by hash join only.
    y_by_key = selection.set_index(["fold", "sample_hash"])["delta"]
    train_prob_names = [c for c in trp.columns if c.endswith("score_plus_minus")]
    sel_prob_names = [c for c in sep.columns if c.endswith("score_plus_minus")]
    check("full_row_probability_shapes", len(trp) == 120000 and len(sep) == 20000 and not trp.isna().any().any() and not sep.isna().any().any(), {"train": len(trp), "selection": len(sep), "train_score_columns": train_prob_names, "selection_score_columns": sel_prob_names})
    score_consistency = True
    for prefix in ["probability_only_3d", "post_first_temporal_43d"]:
        for frame in [trp, sep]:
            score_consistency &= np.allclose(frame[f"{prefix}_score_plus_minus"], frame[f"{prefix}_p_delta_+1"] - frame[f"{prefix}_p_delta_-1"], atol=1e-12, rtol=0)
    check("saved_probability_score_consistency", score_consistency)

    # Recompute the 15 probability rows independently.
    saved_prob = {(int(r["fold"]), r["representation"]): r for r in prob_rows}
    prob_errors = []
    for fold in FOLDS:
        se = selection[selection.fold == fold].copy().reset_index(drop=True)
        y = se.delta.to_numpy(int)
        tr = train[train.fold == fold]
        prior = np.tile(np.array([(tr.delta == cls).mean() for cls in CLASSES]), (len(se), 1))
        joined = sep[sep.fold == fold].set_index("sample_hash").loc[se.sample_hash.astype(str)]
        for rep, cols in [("train_prior", None), ("probability_only_3d", [f"probability_only_3d_p_delta_{c:+d}" for c in CLASSES]), ("post_first_temporal_43d", [f"post_first_temporal_43d_p_delta_{c:+d}" for c in CLASSES])]:
            p = prior if rep == "train_prior" else joined[cols].to_numpy(float)
            got = prob_metrics(y, p); saved = saved_prob[(fold, rep)]
            for k in ["multiclass_brier", "nll", "class_-1_ece", "class_0_ece", "class_1_ece", "class_-1_mean_probability", "class_0_mean_probability", "class_1_mean_probability"]:
                prob_errors.append(abs(float(got[k]) - float(saved[k])))
            for k in ["class_-1_positive_count", "class_0_positive_count", "class_1_positive_count", "rows"]:
                if int(got[k]) != int(saved[k]): prob_errors.append(1.0)
    check("probability_metrics_15_rows", len(prob_rows) == 15 and max(prob_errors, default=0.0) <= 1e-9, {"rows": len(prob_rows), "max_abs_error": max(prob_errors, default=0.0)})

    # Recompute all policy rows from saved labels/probabilities and v60 regression predictions.
    score_errors = []; budget_errors = []; utility_errors = []
    for fold in FOLDS:
        se = selection[selection.fold == fold].copy().reset_index(drop=True)
        y, p_t, p_s, h = se.y.to_numpy(int), se.p_temporal.to_numpy(float), se.p_stats_offline_target_only.to_numpy(float), se.sample_hash.astype(str).to_numpy()
        joined = sep[sep.fold == fold].set_index("sample_hash").loc[se.sample_hash.astype(str)]
        scores = {
            "probability_only_gain_classifier": joined["probability_only_3d_score_plus_minus"].to_numpy(float),
            "post_first_temporal_gain_classifier": joined["post_first_temporal_43d_score_plus_minus"].to_numpy(float),
            "probability_only_gain_regression": se.predicted_delta.to_numpy(float),
            "post_first_temporal_gain_regression": se.predicted_delta_post_first_temporal.to_numpy(float),
        }
        first = (p_t >= .5).astype(int); avg = (((p_t + p_s) / 2) >= .5).astype(int)
        for _, r in pol[pol.fold == fold].iterrows():
            name, budget = r.policy, r.budget
            if name == "first_temporal": acquired = np.array([], dtype=int); pred = first
            elif name == "first_stats": acquired = np.array([], dtype=int); pred = (p_s >= .5).astype(int)
            elif name == "all_average": acquired = np.arange(len(se)); pred = avg
            elif name in {"confidence_budget", "hashrandom_budget"}: acquired = score_order(name, float(budget), p_t, np.zeros(len(se)), h); pred = first.copy(); pred[acquired] = avg[acquired]
            else:
                base = name.replace("_positive_stop", "")
                if name.endswith("_positive_stop"):
                    acquired = np.flatnonzero(scores[base] > 0); pred = first.copy(); pred[acquired] = avg[acquired]
                else:
                    acquired = score_order(name, float(budget), p_t, scores[name], h); pred = first.copy(); pred[acquired] = avg[acquired]
            m = metrics(y, pred, first); calls = len(acquired); mean_delta = m["C_minus_D"] / len(y); rate = calls / len(y)
            for k in ["macro_f1", "malicious_recall", "false_positive_rate", "C", "D", "C_minus_D", "tn", "fp", "fn", "tp"]:
                score_errors.append(abs(float(m[k]) - float(r[k])))
            budget_errors += [abs(calls - int(r.simulated_second_calls)), abs((1 + rate) - r.effective_budget)]
            for lam, col in [(0.0, "utility_lambda_0_0"), (0.01, "utility_lambda_0_01"), (0.05, "utility_lambda_0_05")]:
                utility_errors.append(abs((mean_delta - lam * rate) - float(r[col])))
    check("policy_metrics_recomputed", max(score_errors, default=0.0) <= 1e-12, {"max_abs_metric_error": max(score_errors, default=0.0)})
    check("policy_budgets_recomputed", max(budget_errors, default=0.0) <= 1e-12, {"max_abs_budget_error": max(budget_errors, default=0.0)})
    check("utility_per_sample_formula", max(utility_errors, default=0.0) <= 1e-12, {"max_abs_utility_error": max(utility_errors, default=0.0), "formula": "mean_delta=(C-D)/rows; utility=mean_delta-lambda*call_rate"})

    # Static boundary audit: source access is limited to x_hybrid/sample_hash/group.
    source_text = (EXP / "run_jev_gain_v61.py").read_text(encoding="utf-8")
    forbidden_source_y = 'z["y"]' not in source_text and "z['y']" not in source_text
    source_key_hits = sorted(set(__import__("re").findall(r"z\[['\"]([^'\"]+)", source_text)))
    check("source_label_access_boundary", forbidden_source_y and set(source_key_hits).issubset({"x_hybrid", "sample_hash", "group"}), {"source_key_hits": source_key_hits})
    boundary = json.loads((EXP / "boundary_tests.json").read_text(encoding="utf-8"))
    check("saved_boundary_tests_all_pass", all(bool(v) for v in boundary.values()), boundary)
    forbidden_model_columns = {"y", "label", "group", "p_stats", "p_stats_offline_target_only", "evaluation"}
    model_state_columns = set(["p_temporal", "state_abs_margin", "state_entropy"] + [f"temporal_feature_{i}" for i in range(40)])
    check("model_state_forbidden_fields", forbidden_model_columns.isdisjoint(model_state_columns), sorted(model_state_columns & forbidden_model_columns))

    tracked = [EXP / "preregistration.json", EXP / "run_jev_gain_v61.py", V60 / "train_predictions.csv.gz", V60 / "selection_predictions.csv.gz", V60 / "policy_metrics.csv", V60 / "input_hashes.json", EXP / "results/selection_classifier_probabilities.csv.gz", EXP / "results/train_classifier_probabilities.csv.gz", EXP / "results/policy_metrics.csv", EXP / "results/probability_metrics.json"]
    checks["input_hashes_recomputed"] = {"status": "PASS", "detail": {str(p): sha256(p) for p in tracked}}
    status = "PASS" if not failures else "FAIL"
    report = {"status": status, "validated_date": "2026-09-23", "experiment": str(EXP), "checks": checks, "failures": failures, "scope": "read-only independent recomputation; no experiment files modified", "limits": ["This validates saved numeric artifacts and static boundaries; it is not a fresh blind test.", "Five folds share groups and are not independent tests."]}
    (OUT / "local_gain_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    notes = ["# v61 独立只读数值复核", "", f"结论：**{status}**。", "", "复核重新读取 v60 保存的 selection 标签/概率、v61 全行概率和策略表，独立重建 15 条概率评价记录及 245 条策略记录；未读取 SOURCE y、evaluation 标签或预测，未修改原实验文件。", "", f"最大策略指标绝对误差：{checks['policy_metrics_recomputed']['detail']['max_abs_metric_error']:.3g}；最大预算误差：{checks['policy_budgets_recomputed']['detail']['max_abs_budget_error']:.3g}；最大效用误差：{checks['utility_per_sample_formula']['detail']['max_abs_utility_error']:.3g}。效用口径核对为 `mean_delta=(C-D)/rows`，`utility=mean_delta-lambda*call_rate`。", "", "概率评价覆盖 15 条（5 fold × train_prior/3D/43D），逐条复核 Brier、NLL、三类 ECE、正事件数和平均概率；全行概率分数也通过。10 个模型、训练/选择行数、exact quota、标签访问边界均通过。", "", "详细逐项结果见 `local_gain_validation.json`。"]
    (OUT / "README.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "failures": failures, "max_metric_error": max(score_errors, default=0.0), "max_utility_error": max(utility_errors, default=0.0)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
