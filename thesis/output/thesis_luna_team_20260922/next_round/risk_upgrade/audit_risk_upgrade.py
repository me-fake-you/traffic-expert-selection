"""Five-fold offline error-risk audit for the fixed pavg rule.

This runner uses only train-role OOF arrays for fitting and the predeclared
selection groups for evaluation.  It never indexes sealed archive ``y`` and
never forms an evaluation-role mask.  The two-view predictions are already
available inputs; this is a risk-ranking study, not a base-classifier retrain.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import struct
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HIST = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/nested_controls_001"
SOURCE = ROOT / "data/runs/mad_etd_ustc_group_heldout_hybrid_w98/sealed_group_cv.npz"
PROVENANCE = ROOT / "output/mad_etd_icassp2027_v55_r1/runs/raw_audit_001/selected_provenance.csv"
SPLIT_MANIFEST = HIST / "split_manifest.json"
PILOT_SCRIPT = ROOT / "output/thesis_luna_team_20260922/protocol/run_e0_e1_pilot.py"
PREREG = OUT / "PREREGISTRATION.md"
MODEL_DIR = OUT / "risk_models"
QUOTAS = (0.50, 0.70, 0.80, 0.90, 0.95, 1.00)
METHODS = ("risk_pavg", "risk_disagreement", "risk_hgb")
FEATURE_NAMES = ("pT", "pS", "abs(pT-pS)", "entropy_mean", "margin_mean")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(value: object) -> str:
    return digest_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def digest_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return digest_bytes(struct.pack("<Q", arr.size) + str(arr.dtype).encode("ascii") + arr.tobytes())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def model_features(p_t: np.ndarray, p_s: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            p_t,
            p_s,
            np.abs(p_t - p_s),
            (entropy(p_t) + entropy(p_s)) / 2.0,
            (np.abs(p_t - 0.5) + np.abs(p_s - 0.5)) / 2.0,
        )
    )


def load_source_without_y() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Deliberately request only non-label arrays from the sealed archive.
    with np.load(SOURCE, allow_pickle=False) as source:
        x_stats = source["x_stats"].copy()
        x_hybrid = source["x_hybrid"].copy()
        group = source["group"].astype(str).copy()
        sample_hash = source["sample_hash"].astype(str).copy()
    return x_stats, x_hybrid, group, sample_hash


def strict_selection_labels(selection_hashes: np.ndarray) -> np.ndarray:
    """Decode only preselected hashes, following final pilot_004's safe path."""
    wanted = set(selection_hashes.astype(str).tolist())
    found: dict[str, int] = {}
    with PROVENANCE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_hash", "partition", "label"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise RuntimeError("selection provenance lacks required safe-decoder fields")
        for row in reader:
            sample = str(row["sample_hash"])
            if sample not in wanted:
                continue
            if row["partition"] != "development40k":
                raise RuntimeError(f"selection hash is outside development40k: {sample}")
            if sample in found:
                raise RuntimeError(f"duplicate selection label: {sample}")
            found[sample] = int(row["label"])
    if set(found) != wanted or not set(found.values()).issubset({0, 1}):
        raise RuntimeError("safe selection label decode did not recover exactly the requested hashes")
    return np.asarray([found[str(v)] for v in selection_hashes], dtype=int)


def stable_order(scores: np.ndarray, sample_hash: np.ndarray) -> np.ndarray:
    # Lower risk is accepted first.  sample_hash is the predeclared tie key.
    return np.asarray(sorted(range(len(scores)), key=lambda i: (float(scores[i]), str(sample_hash[i]))), dtype=int)


def tie_stats(scores: np.ndarray) -> tuple[int, int, int]:
    counts: dict[float, int] = {}
    for value in scores.astype(float):
        counts[float(value)] = counts.get(float(value), 0) + 1
    tied = [n for n in counts.values() if n > 1]
    return len(tied), (max(tied) if tied else 1), sum(tied) if tied else 0


def aurc(scores: np.ndarray, errors: np.ndarray, sample_hash: np.ndarray) -> tuple[float, int, int, int]:
    order = stable_order(scores, sample_hash)
    prefixes = np.cumsum(errors[order], dtype=float) / np.arange(1, len(order) + 1, dtype=float)
    tie_count, max_tie, tied_rows = tie_stats(scores)
    return float(prefixes.mean()), tie_count, max_tie, tied_rows


def quota_count(n: int, coverage: float) -> int:
    return int(np.floor(float(coverage) * n + 0.5))


def coverage_row(
    fold: int,
    group_name: str,
    method: str,
    coverage: float,
    scores: np.ndarray,
    sample_hash: np.ndarray,
    y: np.ndarray,
    base_pred: np.ndarray,
    base_error: np.ndarray,
) -> dict[str, object]:
    order = stable_order(scores, sample_hash)
    k = quota_count(len(y), coverage)
    accepted = np.zeros(len(y), dtype=bool)
    accepted[order[:k]] = True
    rejected = ~accepted
    accepted_errors = int(base_error[accepted].sum())
    malicious = y == 1
    correct_detected = int((accepted & malicious & (base_pred == 1)).sum())
    missed = int(malicious.sum()) - correct_detected
    return {
        "fold": fold,
        "group": group_name,
        "method": method,
        "coverage_target": float(coverage),
        "accepted_rows": int(k),
        "total_rows": int(len(y)),
        "coverage": float(k / max(len(y), 1)),
        "accepted_error_count": accepted_errors,
        "accepted_error": float(accepted_errors / max(k, 1)),
        "malicious_correct_detected": correct_detected,
        "malicious_missed": missed,
        "malicious_refused": int((rejected & malicious).sum()),
        "normal_refused": int((rejected & (y == 0)).sum()),
        "base_pavg_malicious_correct_full": int((malicious & (base_pred == 1)).sum()),
        "base_pavg_malicious_missed_full": int((malicious & (base_pred == 0)).sum()),
    }


def make_fold_rows(
    fold: int,
    hashes: np.ndarray,
    groups: np.ndarray,
    p_t: np.ndarray,
    p_s: np.ndarray,
    y: np.ndarray,
    hgb_score: np.ndarray,
) -> pd.DataFrame:
    p_avg = (p_t + p_s) / 2.0
    base_pred = (p_avg >= 0.5).astype(int)
    base_error = (base_pred != y).astype(int)
    feats = model_features(p_t, p_s)
    row_hashes = [
        digest_json({"sample_hash": str(h), "features": [float(v) for v in row]})
        for h, row in zip(hashes, feats)
    ]
    return pd.DataFrame(
        {
            "fold": fold,
            "partition": "selection",
            "sample_hash": hashes,
            "group": groups,
            "y_selection_safe": y,
            "pT": p_t,
            "pS": p_s,
            "pavg": p_avg,
            "base_pavg_prediction": base_pred,
            "base_pavg_error": base_error,
            "entropy_mean": feats[:, 3],
            "margin_mean": feats[:, 4],
            "risk_pavg": 1.0 - np.maximum(p_avg, 1.0 - p_avg),
            "risk_disagreement": np.abs(p_t - p_s),
            "risk_hgb": hgb_score,
            "input_sha256": row_hashes,
        }
    )


def metric_rows(frame: pd.DataFrame, fold: int, group_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = []
    quotas = []
    y = frame["y_selection_safe"].to_numpy(dtype=int)
    errors = frame["base_pavg_error"].to_numpy(dtype=int)
    base_pred = frame["base_pavg_prediction"].to_numpy(dtype=int)
    hashes = frame["sample_hash"].astype(str).to_numpy()
    for method in METHODS:
        scores = frame[method].to_numpy(dtype=float)
        a, tie_count, max_tie, tied_rows = aurc(scores, errors, hashes)
        methods.append(
            {
                "fold": fold,
                "group": group_name if group_name is not None else "__ALL__",
                "method": method,
                "rows": int(len(frame)),
                "base_pavg_error_rate": float(errors.mean()),
                "aurc": a,
                "tie_count": tie_count,
                "max_tie_size": max_tie,
                "tied_rows": tied_rows,
                "base_malicious_correct_full": int(((y == 1) & (base_pred == 1)).sum()),
                "base_malicious_missed_full": int(((y == 1) & (base_pred == 0)).sum()),
            }
        )
        for coverage in QUOTAS:
            quotas.append(coverage_row(fold, group_name or "__ALL__", method, coverage, scores, hashes, y, base_pred, errors))
    return pd.DataFrame(methods), pd.DataFrame(quotas)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if len(split) != 5:
        raise RuntimeError(f"expected five split records, found {len(split)}")
    x_stats, x_hybrid, source_group, sample_hash = load_source_without_y()
    if len(source_group) != 40000 or len(sample_hash) != 40000:
        raise RuntimeError("unexpected development source size")
    all_rows: list[pd.DataFrame] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_coverage: list[pd.DataFrame] = []
    input_hashes: dict[str, object] = {
        "source_selected_arrays": {},
        "provenance_selected_labels": {},
        "fold_train_oof": {},
        "fold_models": {},
    }
    run_log: list[str] = []

    for fold_record in split:
        fold = int(fold_record["outer_fold"])
        role_groups = fold_record["groups"]
        fold_dir = HIST / f"seed_42_fold_{fold}"
        oof_path = fold_dir / "oof_predictions.npz"
        deps_path = fold_dir / "oof_dependencies.json"
        temporal_path = fold_dir / "models/full_temporal.joblib"
        stats_path = fold_dir / "models/full_stats.joblib"
        if not all(p.exists() for p in (oof_path, deps_path, temporal_path, stats_path)):
            raise FileNotFoundError(f"missing fold {fold} artifact")

        selection_idx = np.flatnonzero(np.isin(source_group, np.asarray(role_groups["selection"], dtype=str)))
        train_mask = np.isin(source_group, np.asarray(role_groups["train"], dtype=str))
        if len(selection_idx) != 4000:
            raise RuntimeError(f"fold {fold}: expected 4000 selection rows, got {len(selection_idx)}")
        with np.load(oof_path, allow_pickle=False) as oof_file:
            required = {"indices", "y", "stats", "temporal"}
            if not required.issubset(set(oof_file.files)):
                raise RuntimeError(f"fold {fold}: missing OOF fields")
            oof_idx = oof_file["indices"].astype(int).copy()
            train_y = oof_file["y"].astype(int).copy()
            train_p_s = oof_file["stats"].astype(float).copy()
            train_p_t = oof_file["temporal"].astype(float).copy()
        if len(oof_idx) != 24000 or len(set(oof_idx.tolist())) != 24000:
            raise RuntimeError(f"fold {fold}: expected 24000 unique train OOF rows")
        if not np.isin(oof_idx, np.flatnonzero(train_mask)).all():
            raise RuntimeError(f"fold {fold}: OOF index escaped train role")
        if set(np.unique(train_y)) - {0, 1}:
            raise RuntimeError(f"fold {fold}: invalid train OOF labels")
        deps = json.loads(deps_path.read_text(encoding="utf-8"))
        for dep in deps:
            if set(dep["predict_groups"]) & set(dep["fit_groups"]):
                raise RuntimeError(f"fold {fold}: predict/fit overlap")
            if set(dep["predict_groups"]) & set(dep["calibration_groups"]):
                raise RuntimeError(f"fold {fold}: predict/calibration overlap")

        selection_hash = sample_hash[selection_idx]
        selection_y = strict_selection_labels(selection_hash)
        # Full models are used only to obtain the pre-existing second-view probabilities
        # for selection rows; they are never retrained here.
        with threadpool_limits(limits=1):
            temporal_model = joblib.load(temporal_path)
            stats_model = joblib.load(stats_path)
            selection_p_t = temporal_model.predict_proba(x_hybrid[selection_idx, 8:])[:, 1]
            selection_p_s = stats_model.predict_proba(x_stats[selection_idx])[:, 1]
        train_features = model_features(train_p_t, train_p_s)
        train_pavg = (train_p_t + train_p_s) / 2.0
        train_target = ((train_pavg >= 0.5).astype(int) != train_y).astype(int)
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=20260922,
            early_stopping=False,
        )
        with threadpool_limits(limits=1):
            hgb.fit(train_features, train_target)
            selection_score = hgb.predict_proba(model_features(selection_p_t, selection_p_s))[:, 1]
        model_path = MODEL_DIR / f"risk_hgb_seed20260922_fold_{fold}.joblib"
        joblib.dump(hgb, model_path, compress=3)
        input_hashes["fold_train_oof"][str(fold)] = {
            "oof_path": str(oof_path.resolve()),
            "oof_payload_sha256": digest_json(
                {
                    "indices": digest_array(oof_idx),
                    "y": digest_array(train_y),
                    "pT": digest_array(train_p_t),
                    "pS": digest_array(train_p_s),
                }
            ),
            "rows": int(len(oof_idx)),
            "target_error_support": {str(v): int((train_target == v).sum()) for v in (0, 1)},
        }
        input_hashes["source_selected_arrays"][str(fold)] = digest_json(
            {
                "sample_hash": digest_array(selection_hash.astype("U")),
                "group": digest_array(source_group[selection_idx].astype("U")),
                "x_stats": digest_array(x_stats[selection_idx]),
                "x_hybrid": digest_array(x_hybrid[selection_idx]),
            }
        )
        input_hashes["provenance_selected_labels"][str(fold)] = digest_json(
            {"sample_hash": digest_array(selection_hash.astype("U")), "label": digest_array(selection_y)}
        )
        input_hashes["fold_models"][str(fold)] = {
            "risk_model_path": str(model_path.resolve()),
            "risk_model_sha256": sha256_file(model_path),
            "base_temporal_model_path": str(temporal_path.resolve()),
            "base_stats_model_path": str(stats_path.resolve()),
        }
        rows = make_fold_rows(fold, selection_hash, source_group[selection_idx], selection_p_t, selection_p_s, selection_y, selection_score)
        all_rows.append(rows)
        fold_metric, fold_coverage = metric_rows(rows, fold)
        all_fold_metrics.append(fold_metric)
        all_coverage.append(fold_coverage)
        for group_name, group_frame in rows.groupby("group", sort=True):
            gm, gc = metric_rows(group_frame.reset_index(drop=True), fold, str(group_name))
            all_fold_metrics.append(gm)
            all_coverage.append(gc)
        run_log.append(f"fold={fold} train_oof=24000 selection=4000 groups={sorted(set(rows['group']))} target_error_support={input_hashes['fold_train_oof'][str(fold)]['target_error_support']}")

    row_frame = pd.concat(all_rows, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    coverage_metrics = pd.concat(all_coverage, ignore_index=True)
    row_frame.to_csv(OUT / "selection_rows_all_folds.csv.gz", index=False, compression="gzip")
    fold_metrics.to_csv(OUT / "fold_method_metrics.csv", index=False)
    coverage_metrics.to_csv(OUT / "fold_group_coverage_metrics.csv", index=False)
    fold_metrics[fold_metrics["group"] == "__ALL__"][['fold', 'method', 'aurc']].to_csv(
        OUT / "fold_method_aurc.csv", index=False
    )
    coverage_metrics[coverage_metrics["group"] == "__ALL__"][["fold", "method", "coverage", "accepted_error"]].rename(
        columns={"accepted_error": "selective_error"}
    ).to_csv(OUT / "fold_method_coverage_selective_error.csv", index=False)
    pooled = fold_metrics[fold_metrics["group"] == "__ALL__"].groupby("method", as_index=False).agg(
        folds=("fold", "count"),
        mean_aurc=("aurc", "mean"),
        std_aurc=("aurc", "std"),
        mean_base_error_rate=("base_pavg_error_rate", "mean"),
        mean_rows=("rows", "mean"),
        mean_tie_count=("tie_count", "mean"),
        max_tie_size_seen=("max_tie_size", "max"),
    )
    pooled.insert(1, "aggregation_scope", "descriptive_mean_across_folds_no_CI")
    pooled.to_csv(OUT / "aggregate_descriptive_metrics.csv", index=False)
    write_json(OUT / "input_hashes.json", input_hashes)
    run_log.extend(
        [
            "evaluation rows scored: 0 (evaluation role intentionally excluded)",
            "sealed archive y materialized by runner: false",
            "OOD ablation: not executed; no verified true-OOF OOD field was used",
            "base classifier retraining: false",
            "risk model inputs: " + ",".join(FEATURE_NAMES),
            "AURC: mean(prefix_error), ascending risk, sample_hash tie-break",
        ]
    )
    (OUT / "run.log").write_text("\n".join(run_log) + "\n", encoding="utf-8")
    manifest = {
        "status": "COMPLETED_SELECTION_ONLY_RISK_AUDIT",
        "protocol": "risk_upgrade_v1",
        "folds": 5,
        "seed": 42,
        "train_oof_rows_per_fold": 24000,
        "selection_rows_per_fold": 4000,
        "evaluation_rows_scored": 0,
        "sealed_archive_y_used": False,
        "base_classifier_retrained": False,
        "single_thread": True,
        "risk_model": {
            "class": "HistGradientBoostingClassifier",
            "max_iter": 100,
            "max_leaf_nodes": 7,
            "l2_regularization": 2.0,
            "random_state": 20260922,
            "early_stopping": False,
            "features": list(FEATURE_NAMES),
            "target": "pavg_error=(pavg>=0.5)!=y",
        },
        "base_rule": "pavg=(pT+pS)/2; prediction=(pavg>=0.5)",
        "methods": list(METHODS),
        "coverage_quotas": list(QUOTAS),
        "selection_label_decoder": "run_e0_e1_pilot.py final pilot_004 strict selected_hash provenance path",
        "selection_only_scope": True,
        "ood_status": "not_executed_no_verified_true_oof_ood_field",
        "aggregation": "descriptive only; no confirmatory CI",
        "features_are_existing_two_view_outputs": True,
        "no_new_semantic_information": True,
    }
    manifest["script_sha256"] = sha256_file(Path(__file__))
    write_json(OUT / "run_manifest.json", manifest)
    summary = {
        **manifest,
        "row_output": str((OUT / "selection_rows_all_folds.csv.gz").resolve()),
        "fold_metrics_output": str((OUT / "fold_method_metrics.csv").resolve()),
        "group_coverage_output": str((OUT / "fold_group_coverage_metrics.csv").resolve()),
        "aggregate_output": str((OUT / "aggregate_descriptive_metrics.csv").resolve()),
        "methods_summary": pooled.to_dict(orient="records"),
    }
    write_json(OUT / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
