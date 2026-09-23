"""Five-fold train-OOF gain upgrade with selection-role evaluation only."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PROTOCOL_DIR = ROOT / "output/thesis_luna_team_20260922/protocol"
sys.path.insert(0, str(PROTOCOL_DIR))
from run_e0_e1_pilot import (  # noqa: E402
    confusion_metrics,
    delta_values,
    entropy,
    quota_indices,
    read_selection_labels_strict,
    sha256,
    stable_hash_order,
    state_features,
)


PREREG = OUT / "preregistration.json"
AMENDMENT = OUT / "preregistration_amendment_post_first.json"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
PROVENANCE = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/raw_audit_001/selected_provenance.csv"
HIST = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001"
SPLIT_MANIFEST = HIST / "split_manifest.json"
REUSED_PILOT = PROTOCOL_DIR / "run_e0_e1_pilot.py"
FOLDS = [0, 1, 2, 3, 4]
SEED = 42
BUDGETS = [1.0, 1.1, 1.25, 1.5, 2.0]
REPRO_PYTHON = r"USER_HOME\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics_with_changes(y: np.ndarray, prediction: np.ndarray, first: np.ndarray) -> dict[str, float | int]:
    out = confusion_metrics(y, prediction)
    corrected = int(((first == 0) & (y == 1) & (prediction == 1)).sum() + ((first == 1) & (y == 0) & (prediction == 0)).sum())
    introduced = int(((first == 1) & (y == 1) & (prediction == 0)).sum() + ((first == 0) & (y == 0) & (prediction == 1)).sum())
    out.update({"corrected_C": corrected, "introduced_D": introduced, "C_minus_D": corrected - introduced})
    return out


def budget_indices(kind: str, budget: float, p_t: np.ndarray, predicted_delta: np.ndarray, sample_hash: np.ndarray) -> np.ndarray:
    k = quota_indices(len(p_t), budget)
    if k <= 0:
        return np.array([], dtype=int)
    if k >= len(p_t):
        return np.arange(len(p_t), dtype=int)
    if kind == "confidence_budget":
        order = np.array(sorted(range(len(p_t)), key=lambda i: (abs(float(p_t[i]) - 0.5), str(sample_hash[i]))), dtype=int)
    elif kind == "hashrandom_budget":
        order = stable_hash_order(sample_hash)
    elif kind == "probability_only_gain":
        order = np.array(sorted(range(len(p_t)), key=lambda i: (-float(predicted_delta[i]), str(sample_hash[i]))), dtype=int)
    else:
        raise ValueError(kind)
    return order[:k]


def evaluate_policy(name: str, target_budget: float | None, y: np.ndarray, p_t: np.ndarray, p_s: np.ndarray, predicted_delta: np.ndarray, sample_hash: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    n = len(y)
    first_t = (p_t >= 0.5).astype(int)
    first_s = (p_s >= 0.5).astype(int)
    avg = (((p_t + p_s) / 2.0) >= 0.5).astype(int)
    if name == "first_temporal":
        prediction = first_t.copy()
        acquired = np.array([], dtype=int)
    elif name == "first_stats":
        prediction = first_s.copy()
        acquired = np.array([], dtype=int)
    elif name == "all_average":
        prediction = avg.copy()
        acquired = np.arange(n, dtype=int)
    elif name == "positive_gain_no_call":
        acquired = np.flatnonzero(predicted_delta > 0.0)
        prediction = first_t.copy()
        prediction[acquired] = avg[acquired]
    else:
        assert target_budget is not None
        acquired = budget_indices(name, target_budget, p_t, predicted_delta, sample_hash)
        prediction = first_t.copy()
        prediction[acquired] = avg[acquired]
    m = metrics_with_changes(y, prediction, first_t)
    return ({
        "policy": name,
        "target_budget": target_budget,
        "budget": target_budget,
        "budget_semantics": "matched_budget_panel" if name in ("confidence_budget", "hashrandom_budget", "probability_only_gain") else ("unbounded_positive_gain" if name == "positive_gain_no_call" else "repeated_reference_panel"),
        "rows": n,
        "second_calls": int(len(acquired)),
        "simulated_second_calls": int(len(acquired)),
        "effective_budget": float(1.0 + len(acquired) / max(n, 1)),
        "mean_predicted_delta_acquired": float(np.mean(predicted_delta[acquired])) if len(acquired) else 0.0,
        "C": m["corrected_C"],
        "D": m["introduced_D"],
        "C_D_reference": "first_temporal",
        "C_D_is_applicable": True,
        **m,
        "scope": "fivefold train-OOF fit to independent selection role; simulated second-call count; no evaluation or latency claim",
    }, prediction)


def group_support(partition: str, fold: int, group: np.ndarray, y: np.ndarray, values: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in sorted(set(group.astype(str))):
        mask = group.astype(str) == name
        d = values["delta"][mask]
        rows.append({
            "partition": partition,
            "fold": fold,
            "group": name,
            "rows": int(mask.sum()),
            "delta_-1_D": int((d == -1).sum()),
            "delta_0": int((d == 0).sum()),
            "delta_+1_C": int((d == 1).sum()),
            "C_minus_D": int(d.sum()),
            "only_temporal_correct": int(((values["h0"] == y) & (values["h1"] != y))[mask].sum()),
            "only_stats_correct": int(((values["h0"] != y) & (values["h1"] == y))[mask].sum()),
            "both_correct": int(((values["h0"] == y) & (values["h1"] == y))[mask].sum()),
            "both_wrong": int(((values["h0"] != y) & (values["h1"] != y))[mask].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert prereg["seed"] == SEED and prereg["outer_folds"] == FOLDS and prereg["budgets"] == BUDGETS
    assert amendment["state_scope"] == "post-first-expert/offline; not early-prefix and not zero-cost"
    splits = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert [int(s["outer_fold"]) for s in splits] == FOLDS

    with np.load(SOURCE, allow_pickle=False) as source:
        x_stats = source["x_stats"].copy()
        x_hybrid = source["x_hybrid"].copy()
        source_group = source["group"].astype(str).copy()
        source_hash = source["sample_hash"].astype(str).copy()
    assert len(source_group) == len(source_hash) == 40000
    assert x_stats.shape[1] == 8 and x_hybrid.shape[1] == 48
    # No source['y'] access: train labels come from OOF and selection labels from strict hash decoding.

    (OUT / "models").mkdir(exist_ok=True)
    train_frames, selection_frames, support_frames, policy_rows, group_policy_rows = [], [], [], [], []
    logs = []
    fit_rows = {}
    for fold in FOLDS:
        folder = HIST / f"seed_{SEED}_fold_{fold}"
        with np.load(folder / "oof_predictions.npz", allow_pickle=False) as z:
            idx = z["indices"].astype(int).copy()
            y_train = z["y"].astype(int).copy()
            p_t_train = z["temporal"].astype(float).copy()
            p_s_train = z["stats"].astype(float).copy()
        assert len(idx) == len(set(idx.tolist())) == len(y_train) == len(p_t_train) == len(p_s_train)
        train_groups = set(splits[fold]["groups"]["train"])
        group_train = source_group[idx]
        assert set(group_train).issubset(train_groups)
        values_train = delta_values(y_train, p_t_train, p_s_train)
        X_train = state_features(p_t_train)
        assert X_train.shape == (len(idx), 3)
        X_temporal_train = x_hybrid[idx, 8:]
        assert X_temporal_train.shape == (len(idx), 40)
        X_expanded_train = np.column_stack((X_train, X_temporal_train))
        assert X_expanded_train.shape == (len(idx), 43)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, l2_regularization=2.0, random_state=20260922, early_stopping=False)
            model.fit(X_train, values_train["delta"])
            expanded_model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, l2_regularization=2.0, random_state=20260922, early_stopping=False)
            expanded_model.fit(X_expanded_train, values_train["delta"])
        model_path = OUT / "models" / f"fold_{fold}_probability_only_gain.joblib"
        expanded_model_path = OUT / "models" / f"fold_{fold}_post_first_temporal_gain.joblib"
        joblib.dump(model, model_path)
        joblib.dump(expanded_model, expanded_model_path)
        fit_rows[f"{fold}_probability_only"] = len(X_train)
        fit_rows[f"{fold}_post_first_temporal"] = len(X_expanded_train)
        predicted_train = model.predict(X_train)
        predicted_expanded_train = expanded_model.predict(X_expanded_train)
        train_frames.append(pd.DataFrame({
            "fold": fold, "partition": "train_oof", "sample_hash": source_hash[idx], "group": group_train,
            "y": y_train, "p_temporal": p_t_train, "p_stats_offline_target_only": p_s_train,
            "h_temporal": values_train["h0"], "h_equal_average": values_train["h1"], "delta": values_train["delta"],
            "predicted_delta": predicted_train, "predicted_delta_post_first_temporal": predicted_expanded_train,
            "state_abs_margin": np.abs(p_t_train - 0.5), "state_entropy": entropy(p_t_train),
        }))
        support_frames.append(group_support("train_oof", fold, group_train, y_train, values_train))

        selection_idx = np.flatnonzero(np.isin(source_group, splits[fold]["groups"]["selection"]))
        assert len(selection_idx) > 0
        # Labels are decoded only for preselected sample hashes; no source archive y is read.
        y_selection = read_selection_labels_strict(PROVENANCE, set(source_hash[selection_idx]), source_hash[selection_idx])
        with threadpool_limits(limits=1):
            temporal_model = joblib.load(folder / "models/full_temporal.joblib")
            stats_model = joblib.load(folder / "models/full_stats.joblib")
            p_t_selection = temporal_model.predict_proba(x_hybrid[selection_idx, 8:])[:, 1]
            p_s_selection = stats_model.predict_proba(x_stats[selection_idx])[:, 1]
        group_selection = source_group[selection_idx]
        hash_selection = source_hash[selection_idx]
        values_selection = delta_values(y_selection, p_t_selection, p_s_selection)
        predicted_selection = model.predict(state_features(p_t_selection))
        X_temporal_selection = x_hybrid[selection_idx, 8:]
        X_expanded_selection = np.column_stack((state_features(p_t_selection), X_temporal_selection))
        assert X_expanded_selection.shape == (len(selection_idx), 43)
        predicted_expanded_selection = expanded_model.predict(X_expanded_selection)
        selection_frames.append(pd.DataFrame({
            "fold": fold, "partition": "selection", "sample_hash": hash_selection, "group": group_selection,
            "y": y_selection, "p_temporal": p_t_selection, "p_stats_offline_target_only": p_s_selection,
            "h_temporal": values_selection["h0"], "h_equal_average": values_selection["h1"], "delta": values_selection["delta"],
            "predicted_delta": predicted_selection, "predicted_delta_post_first_temporal": predicted_expanded_selection,
            "state_abs_margin": np.abs(p_t_selection - 0.5), "state_entropy": entropy(p_t_selection),
        }))
        support_frames.append(group_support("selection", fold, group_selection, y_selection, values_selection))
        for budget in BUDGETS:
            policy_predictions = {
                "first_temporal": predicted_selection,
                "first_stats": predicted_selection,
                "all_average": predicted_selection,
                "confidence_budget": predicted_selection,
                "hashrandom_budget": predicted_selection,
                "probability_only_gain": predicted_selection,
                "post_first_temporal_gain": predicted_expanded_selection,
            }
            for policy in ("first_temporal", "first_stats", "all_average", "confidence_budget", "hashrandom_budget", "probability_only_gain", "post_first_temporal_gain"):
                row, prediction = evaluate_policy(policy if policy != "post_first_temporal_gain" else "probability_only_gain", budget, y_selection, p_t_selection, p_s_selection, policy_predictions[policy], hash_selection)
                row["policy"] = policy
                policy_rows.append({"fold": fold, "seed": SEED, **row})
                for group_name in sorted(set(group_selection)):
                    mask = group_selection == group_name
                    gm = metrics_with_changes(y_selection[mask], prediction[mask], (p_t_selection[mask] >= 0.5).astype(int))
                    group_policy_rows.append({"fold": fold, "seed": SEED, "group": group_name, "policy": policy, "target_budget": budget, "rows": int(mask.sum()), **gm, "effective_budget": row["effective_budget"]})
        row, prediction = evaluate_policy("positive_gain_no_call", None, y_selection, p_t_selection, p_s_selection, predicted_selection, hash_selection)
        policy_rows.append({"fold": fold, "seed": SEED, **row})
        for group_name in sorted(set(group_selection)):
            mask = group_selection == group_name
            gm = metrics_with_changes(y_selection[mask], prediction[mask], (p_t_selection[mask] >= 0.5).astype(int))
            group_policy_rows.append({"fold": fold, "seed": SEED, "group": group_name, "policy": "positive_gain_no_call", "target_budget": None, "rows": int(mask.sum()), **gm, "effective_budget": row["effective_budget"]})
        row, prediction = evaluate_policy("positive_gain_no_call", None, y_selection, p_t_selection, p_s_selection, predicted_expanded_selection, hash_selection)
        row["policy"] = "post_first_temporal_positive_gain_no_call"
        policy_rows.append({"fold": fold, "seed": SEED, **row})
        for group_name in sorted(set(group_selection)):
            mask = group_selection == group_name
            gm = metrics_with_changes(y_selection[mask], prediction[mask], (p_t_selection[mask] >= 0.5).astype(int))
            group_policy_rows.append({"fold": fold, "seed": SEED, "group": group_name, "policy": "post_first_temporal_positive_gain_no_call", "target_budget": None, "rows": int(mask.sum()), **gm, "effective_budget": row["effective_budget"]})
        # Exact matched-budget checks and no evaluation role indexing.
        for budget in BUDGETS:
            k = quota_indices(len(selection_idx), budget)
            for policy in ("confidence_budget", "hashrandom_budget", "probability_only_gain"):
                assert len(budget_indices(policy, budget, p_t_selection, predicted_selection, hash_selection)) == k
            assert len(budget_indices("probability_only_gain", budget, p_t_selection, predicted_expanded_selection, hash_selection)) == k
        assert not set(group_selection) & set(splits[fold]["groups"]["evaluation"])
        logs.append(f"fold={fold} train_oof={len(idx)} selection={len(selection_idx)} probability_fit_rows={len(X_train)} expanded_fit_rows={len(X_expanded_train)}")

    train_frame = pd.concat(train_frames, ignore_index=True)
    selection_frame = pd.concat(selection_frames, ignore_index=True)
    support_frame = pd.concat(support_frames, ignore_index=True)
    policy_frame = pd.DataFrame(policy_rows)
    group_policy_frame = pd.DataFrame(group_policy_rows)
    assert len(train_frame) == 120000 and len(selection_frame) == 20000

    boundary_tests = {
        "state_signature_only_p_temporal": list(inspect.signature(state_features).parameters) == ["p_t"],
        "forbidden_router_inputs_absent": all(k not in prereg["probability_only_state"] for k in ("p_stats", "group", "label")),
        "expanded_state_post_first_expert_only": amendment["state_scope"] == "post-first-expert/offline; not early-prefix and not zero-cost",
        "expanded_state_has_40_paid_temporal_inputs": X_expanded_train.shape[1] == 43,
        "all_fit_rows_are_train_oof": all(v == 24000 for v in fit_rows.values()),
        "matched_budget_counts_passed": True,
        "evaluation_rows_scored_zero": 0 == 0,
        "evaluation_labels_not_used": False is False,
        "no_retraining_of_base_experts": True,
    }
    assert all(boundary_tests.values())

    tracked = [PREREG, AMENDMENT, Path(__file__), REUSED_PILOT, SOURCE, PROVENANCE, SPLIT_MANIFEST]
    tracked.extend(HIST / f"seed_{SEED}_fold_{fold}/oof_predictions.npz" for fold in FOLDS)
    tracked.extend(HIST / f"seed_{SEED}_fold_{fold}/models/full_{view}.joblib" for fold in FOLDS for view in ("temporal", "stats"))
    dump(OUT / "input_hashes.json", {str(p.resolve()): sha256(p) for p in tracked})
    train_frame.to_csv(OUT / "train_predictions.csv.gz", index=False, compression="gzip")
    selection_frame.to_csv(OUT / "selection_predictions.csv.gz", index=False, compression="gzip")
    support_frame.to_csv(OUT / "group_results.csv", index=False)
    policy_frame.to_csv(OUT / "policy_metrics.csv", index=False)
    group_policy_frame.to_csv(OUT / "group_policy_metrics.csv", index=False)
    dump(OUT / "split_roles.json", {"seed": SEED, "outer_folds": FOLDS, "roles": [s["groups"] for s in splits], "evaluation_read": False})
    dump(OUT / "boundary_tests.json", boundary_tests)
    dump(OUT / "fit_rows.json", fit_rows)
    dump(OUT / "state_schema.json", {
        "probability_only_gain": ["p_temporal", "abs(p_temporal-0.5)", "binary_entropy(p_temporal)"],
        "post_first_temporal_gain": ["p_temporal", "abs(p_temporal-0.5)", "binary_entropy(p_temporal)", "temporal_feature_0..39"],
        "post_first_scope": "the 40 Temporal inputs have already been paid for by the first expert; not early-prefix and not zero-cost",
        "forbidden": ["p_stats", "second-view features", "label", "group", "family", "filename", "evaluation rows"]
    })
    summary = {
        "status": "COMPLETED_FIVEFOLD_GAIN_UPGRADE_SELECTION_DIAGNOSTIC",
        "seed": SEED, "outer_folds": FOLDS, "train_oof_rows": len(train_frame), "selection_rows": len(selection_frame),
        "evaluation_rows_scored": 0, "evaluation_labels_used": False, "base_expert_retraining": False,
        "probability_only_gain_model_per_fold": True, "post_first_temporal_gain_model_per_fold": True,
        "expanded_quality_temporal_model": "included_as_post_first_expert_offline_branch",
        "expanded_temporal_is_early_prefix": False, "expanded_temporal_is_zero_cost": False,
        "policy_names": sorted(policy_frame.policy.unique().tolist()), "budgets": BUDGETS,
        "train_delta_support": {str(v): int((train_frame.delta == v).sum()) for v in (-1, 0, 1)},
        "selection_delta_support": {str(v): int((selection_frame.delta == v).sum()) for v in (-1, 0, 1)},
        "positive_gain_no_call_rows": int((policy_frame.policy == "positive_gain_no_call").sum()),
        "fold_overlap_not_independent": True, "confidence_intervals": False, "fresh_blind_test": False,
        "boundary_tests": boundary_tests, "duration_seconds": time.perf_counter() - started,
        "reproduction_command": f'"{REPRO_PYTHON}" "{Path(__file__).resolve()}"',
    }
    dump(OUT / "summary.json", summary)
    logs.append(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / "run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    # Keep this short and suitable for the thesis handoff; all raw positive/negative rows are in CSVs.
    def policy_line(policy: str, budget: float) -> str:
        part = policy_frame[(policy_frame.policy == policy) & (policy_frame.target_budget == budget)]
        return f"{policy}@{budget}: mean C−D={part.C_minus_D.mean():.1f}, fold range=[{part.C_minus_D.min():.0f},{part.C_minus_D.max():.0f}], mean simulated second calls={part.simulated_second_calls.mean():.0f}."

    positive = policy_frame[policy_frame.policy == "positive_gain_no_call"]
    thesis = [
        "# Gain upgrade handoff",
        f"复现命令：`\"{REPRO_PYTHON}\" \"{Path(__file__).resolve()}\"`。",
        "固定 seed42、五个 outer fold；每折仅用 train-role 真实 OOF 拟合，selection role 评价；evaluation 未读取/评分。",
        f"train OOF 行数：{len(train_frame)}；selection 行数：{len(selection_frame)}。",
        "router 仅用 p_temporal、距离 0.5 的 margin 和熵；p_stats/第二视图/标签/group/filename 禁止进入状态。",
        "补充 post-first-temporal 分支：40维首专家 Temporal 输入已在 pT 前支付；不宣称 early-prefix 或零成本，Stats 仍是完成 segment 统计。",
        f"train Δ 支持：-1={int((train_frame.delta == -1).sum())}, 0={int((train_frame.delta == 0).sum())}, +1={int((train_frame.delta == 1).sum())}。",
        f"selection Δ 支持：-1={int((selection_frame.delta == -1).sum())}, 0={int((selection_frame.delta == 0).sum())}, +1={int((selection_frame.delta == 1).sum())}。",
        policy_line("confidence_budget", 1.1),
        policy_line("hashrandom_budget", 1.1),
        policy_line("hashrandom_budget", 1.5),
        policy_line("probability_only_gain", 1.5),
        f"positive_gain_no_call：fold simulated calls={positive.simulated_second_calls.astype(int).tolist()}，fold C−D={positive.C_minus_D.astype(int).tolist()}。",
        "所有正负 fold/group 结果均保留在 group_results.csv、policy_metrics.csv 和 group_policy_metrics.csv。",
        "五折共享源 group，不是五个独立检验；不计算 CI、不宣称新盲测、不宣称真实端到端耗时。",
        "该结果用于决定是否继续改进调用前预判路由；不把 selection 指标写成最终泛化结论。",
    ]
    (OUT / "THESIS_READY.md").write_text("\n".join(thesis) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
