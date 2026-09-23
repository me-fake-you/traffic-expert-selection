"""Five-fold train-OOF error-support diagnostic; no evaluation access."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
PREREG = OUT / "preregistration.json"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
HIST = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001"
SPLIT_MANIFEST = HIST / "split_manifest.json"
FOLDS = [0, 1, 2, 3, 4]
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    assert prereg["seed"] == SEED and prereg["outer_folds"] == FOLDS
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert [int(row["outer_fold"]) for row in split_manifest] == FOLDS

    # Only label-free source arrays are loaded. OOF y is used because every OOF
    # row is a train-role prediction; source y and all evaluation arrays remain untouched.
    with np.load(SOURCE, allow_pickle=False) as source:
        source_group = source["group"].astype(str).copy()
        source_hash = source["sample_hash"].astype(str).copy()
    assert len(source_group) == len(source_hash) == 40000
    assert len(set(source_hash)) == len(source_hash)

    rows = []
    fold_rows = []
    logs = []
    oof_paths = []
    for fold in FOLDS:
        folder = HIST / f"seed_{SEED}_fold_{fold}"
        oof_path = folder / "oof_predictions.npz"
        oof_paths.append(oof_path)
        with np.load(oof_path, allow_pickle=False) as oof_file:
            required = {"indices", "y", "stats", "temporal"}
            assert required.issubset(set(oof_file.files))
            indices = oof_file["indices"].astype(int).copy()
            y = oof_file["y"].astype(int).copy()
            p_t = oof_file["temporal"].astype(float).copy()
            p_s = oof_file["stats"].astype(float).copy()
        assert len(indices) == len(set(indices.tolist()))
        assert len(indices) == len(y) == len(p_t) == len(p_s)
        assert np.isfinite(p_t).all() and np.isfinite(p_s).all()
        groups = source_group[indices]
        train_groups = set(split_manifest[fold]["groups"]["train"])
        assert set(groups).issubset(train_groups), f"fold {fold}: OOF row outside train groups"
        h_t = (p_t >= 0.5).astype(int)
        h_s = (p_s >= 0.5).astype(int)
        h_avg = (((p_t + p_s) / 2.0) >= 0.5).astype(int)
        t_correct = h_t == y
        s_correct = h_s == y
        avg_correct = h_avg == y
        delta = (h_t != y).astype(int) - (h_avg != y).astype(int)
        assert set(np.unique(delta)).issubset({-1, 0, 1})
        sample_hash = source_hash[indices]
        assert len(set(sample_hash)) == len(sample_hash)

        for group in sorted(set(groups)):
            mask = groups == group
            d = delta[mask]
            rows.append({
                "fold": fold,
                "seed": SEED,
                "group": group,
                "rows": int(mask.sum()),
                "both_correct": int((t_correct[mask] & s_correct[mask]).sum()),
                "only_temporal_correct": int((t_correct[mask] & ~s_correct[mask]).sum()),
                "only_stats_correct": int((~t_correct[mask] & s_correct[mask]).sum()),
                "both_wrong": int((~t_correct[mask] & ~s_correct[mask]).sum()),
                "expert_disagreement": int((h_t[mask] != h_s[mask]).sum()),
                "equal_average_switches": int((h_t[mask] != h_avg[mask]).sum()),
                "delta_-1_D": int((d == -1).sum()),
                "delta_0": int((d == 0).sum()),
                "delta_+1_C": int((d == 1).sum()),
                "C_minus_D": int(d.sum()),
                "oracle_temporal_to_stats": int((~t_correct[mask] & s_correct[mask]).sum()),
                "oracle_stats_to_temporal": int((t_correct[mask] & ~s_correct[mask]).sum()),
            })
        fold_rows.append({
            "fold": fold,
            "seed": SEED,
            "rows": len(y),
            "train_group_count": len(set(groups)),
            "both_correct": int((t_correct & s_correct).sum()),
            "only_temporal_correct": int((t_correct & ~s_correct).sum()),
            "only_stats_correct": int((~t_correct & s_correct).sum()),
            "both_wrong": int((~t_correct & ~s_correct).sum()),
            "expert_disagreement": int((h_t != h_s).sum()),
            "equal_average_switches": int((h_t != h_avg).sum()),
            "delta_-1_D": int((delta == -1).sum()),
            "delta_0": int((delta == 0).sum()),
            "delta_+1_C": int((delta == 1).sum()),
            "C_minus_D": int(delta.sum()),
        })
        logs.append(f"fold={fold} rows={len(y)} train_groups={len(set(groups))} C={int((delta == 1).sum())} D={int((delta == -1).sum())} zero={int((delta == 0).sum())}")

    group_frame = pd.DataFrame(rows).sort_values(["fold", "group"]).reset_index(drop=True)
    fold_frame = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    pooled = group_frame.groupby("group", as_index=False).agg(
        folds_present=("fold", "nunique"),
        rows=("rows", "sum"),
        both_correct=("both_correct", "sum"),
        only_temporal_correct=("only_temporal_correct", "sum"),
        only_stats_correct=("only_stats_correct", "sum"),
        both_wrong=("both_wrong", "sum"),
        expert_disagreement=("expert_disagreement", "sum"),
        equal_average_switches=("equal_average_switches", "sum"),
        delta_minus1_D=("delta_-1_D", "sum"),
        delta_0=("delta_0", "sum"),
        delta_plus1_C=("delta_+1_C", "sum"),
        C_minus_D=("C_minus_D", "sum"),
        oracle_temporal_to_stats=("oracle_temporal_to_stats", "sum"),
        oracle_stats_to_temporal=("oracle_stats_to_temporal", "sum"),
    ).sort_values("group").reset_index(drop=True)

    totals = {column: int(group_frame[column].sum()) for column in (
        "rows", "both_correct", "only_temporal_correct", "only_stats_correct", "both_wrong",
        "expert_disagreement", "equal_average_switches", "delta_-1_D", "delta_0", "delta_+1_C", "C_minus_D"
    )}
    fold_positive = int((fold_frame["C_minus_D"] > 0).sum())
    fold_negative = int((fold_frame["C_minus_D"] < 0).sum())
    fold_zero = int((fold_frame["C_minus_D"] == 0).sum())
    group_positive = int((pooled["C_minus_D"] > 0).sum())
    group_negative = int((pooled["C_minus_D"] < 0).sum())
    group_zero = int((pooled["C_minus_D"] == 0).sum())

    if totals["C_minus_D"] > 0 and fold_positive >= 3:
        direction = "固定等权组合在五折训练OOF上有净纠错信号，但仍需路由节省调用的独立验证。"
    elif totals["C_minus_D"] < 0 and fold_negative >= 3:
        direction = "固定等权组合在多数训练OOF折有净损害，优先改进调用前预判路由并保留组合负结果。"
    else:
        direction = "固定等权组合的净效应跨折混合，优先做调用前预判路由与逐组条件化，而不是直接强化组合。"
    judgment = [
        "五折 seed42 train-role OOF 支持诊断（不是五个独立检验）。",
        f"总行数 {totals['rows']}；两者都对 {totals['both_correct']}，仅 Temporal 对 {totals['only_temporal_correct']}，仅 Stats 对 {totals['only_stats_correct']}，两者都错 {totals['both_wrong']}。",
        f"专家分歧 {totals['expert_disagreement']}，等权平均切换 {totals['equal_average_switches']}。",
        f"Δ 支持：D(-1)={totals['delta_-1_D']}、0={totals['delta_0']}、C(+1)={totals['delta_+1_C']}，C−D={totals['C_minus_D']}。",
        f"C−D 为正/负/零的 fold 数：{fold_positive}/{fold_negative}/{fold_zero}；group 数：{group_positive}/{group_negative}/{group_zero}。",
        direction,
        "仅 Stats 正确的行是 Temporal→Stats 的组合上限；仅 Temporal 正确的行是 Stats→Temporal 的替代上限。",
        "这些是共享数据上的训练OOF统计，不能替代 evaluation 或新来源确认，也不提供安全保证。",
    ]
    assert len(judgment) <= 15

    tracked = [PREREG, Path(__file__), SOURCE, SPLIT_MANIFEST, *oof_paths]
    input_hashes = {str(path.resolve()): sha256(path) for path in tracked}
    group_frame.to_csv(OUT / "group_support.csv", index=False)
    fold_frame.to_csv(OUT / "fold_summary.csv", index=False)
    pooled.to_csv(OUT / "pooled_group_support.csv", index=False)
    dump(OUT / "input_hashes.json", input_hashes)
    summary = {
        "status": "COMPLETED_FIVEFOLD_TRAIN_OOF_SUPPORT_DIAGNOSTIC",
        "seed": SEED,
        "outer_folds": FOLDS,
        "folds_processed": len(fold_frame),
        "evaluation_predictions_read": False,
        "evaluation_labels_read": False,
        "retraining": False,
        "fold_selection": False,
        "train_oof_rows_total": totals["rows"],
        "totals": totals,
        "positive_negative_zero_fold_counts": [fold_positive, fold_negative, fold_zero],
        "positive_negative_zero_group_counts": [group_positive, group_negative, group_zero],
        "shared_source_groups_across_folds": True,
        "independence_claim": False,
        "judgment_direction": direction,
        "duration_seconds": time.perf_counter() - start,
    }
    dump(OUT / "summary.json", summary)
    (OUT / "judgment.md").write_text("\n".join(judgment) + "\n", encoding="utf-8")
    (OUT / "run_log.txt").write_text("\n".join(logs) + "\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
