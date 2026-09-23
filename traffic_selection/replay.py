"""Replay recorded matrices; seal all selection choices before outer scoring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import GLOBAL_IDS, THRESHOLDS, choose, decomposition, fixed_matrix, metrics, pattern_support


def load(root, condition, kind, fold, role):
    path = root / f"{condition}_20260920_{kind}_fold{fold}_{role}.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    for key in ("label", "predictions", "thresholds", "first_probability", "second_probability", "trigger"):
        if key not in data:
            raise ValueError(f"Missing array {key} in {path.name}")
    if not np.array_equal(data["thresholds"], THRESHOLDS):
        raise ValueError("Not the original 16-column threshold grid")
    if not np.array_equal(data["predictions"][:, 15], data["first_probability"] >= .5):
        raise ValueError("No-switch candidate does not equal Temporal")
    return data, dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(matrix_dir, reference_file, out):
    refs = json.loads(reference_file.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    selected, inputs, support, cache, controls = {}, [], [], {}, []
    # No evaluation files are loaded until every selection has been saved.
    for fold in range(5):
        for condition in ("row100", "source_concentrated"):
            for kind in ("logistic", "hgb"):
                s, receipt = load(matrix_dir, condition, kind, fold, "selection")
                inputs.append(receipt)
                cache[fold, condition, kind] = s
                first = s["first_probability"] >= .5
                best, feasible, _, _ = choose(s["label"], first, s["predictions"])
                selected[fold, condition, kind] = best
                for name, ids in (("all", range(16)), ("dev_feasible", feasible)):
                    support.append(dict(fold_id=fold, condition=condition, arbiter=kind,
                                        candidate_set=name, **pattern_support(s["predictions"][:, ids])))
                if condition == "row100":
                    if kind == "logistic":
                        fm = fixed_matrix(s["first_probability"], s["second_probability"], s["trigger"],
                                          refs[str(fold)]["reliability_temporal"], refs[str(fold)]["reliability_stats"])
                        j, f, opt, rows = choose(s["label"], first, fm)
                        selected[fold, "Fixed-Dev16"] = j
                        controls.append(dict(fold_id=fold, method="Fixed-Dev16", candidate=j,
                                             candidate_budget=16, feasible_candidates=len(f),
                                             best_numeric_candidates=len(opt)))
                    j, f, opt, _ = choose(s["label"], first, s["predictions"], GLOBAL_IDS)
                    name = ("Logistic" if kind == "logistic" else "HGB") + "-Global4"
                    selected[fold, name] = j
                    controls.append(dict(fold_id=fold, method=name, candidate=j, candidate_budget=4,
                                         feasible_candidates=len(f), best_numeric_candidates=len(opt)))
    write_csv(out/"selection_records.csv", controls)
    write_csv(out/"candidate_support.csv", support)
    gate = dict(selection_choices=[dict(key=list(k), candidate=v) for k, v in selected.items()],
                inputs=inputs, evaluation_opened=False,
                reference_sha256=hashlib.sha256(reference_file.read_bytes()).hexdigest())
    dump(out/"selection_gate.json", gate)
    decompositions, fold_rows, pooled = [], [], {}
    all_y, all_first = [], []
    for fold in range(5):
        evaluation = {}
        for condition in ("row100", "source_concentrated"):
            for kind in ("logistic", "hgb"):
                s = cache[fold, condition, kind]
                e, receipt = load(matrix_dir, condition, kind, fold, "evaluation")
                inputs.append(receipt)
                if "sample_hash" in s and "sample_hash" in e:
                    if np.intersect1d(s["sample_hash"], e["sample_hash"]).size:
                        raise ValueError("Selection/evaluation sample overlap")
                if "group" in s and "group" in e and np.intersect1d(s["group"], e["group"]).size:
                    raise ValueError("Selection/evaluation group overlap")
                record = decomposition(s["label"], s["first_probability"] >= .5, s["predictions"],
                                       e["label"], e["first_probability"] >= .5, e["predictions"])
                if record["selected_candidate"] != selected[fold, condition, kind]:
                    raise AssertionError("Selection changed during scoring")
                decompositions.append(dict(fold_id=fold, condition=condition, arbiter=kind, **record))
                if condition == "row100":
                    evaluation[kind] = e
        d, h = evaluation["logistic"], evaluation["hgb"]
        for key in ("label", "first_probability", "second_probability", "trigger", "sample_hash"):
            if key in d and not np.array_equal(d[key], h[key]):
                raise ValueError(f"Arbiters do not share aligned {key}")
        first = d["first_probability"] >= .5
        fm = fixed_matrix(d["first_probability"], d["second_probability"], d["trigger"],
                          refs[str(fold)]["reliability_temporal"], refs[str(fold)]["reliability_stats"])
        predictions = {"Temporal": first, "Fixed": fm[:, 0],
                       "Fixed-Dev16": fm[:, selected[fold, "Fixed-Dev16"]]}
        for kind, name in (("logistic", "Logistic"), ("hgb", "HGB")):
            p = evaluation[kind]["predictions"]
            predictions[name+"-0.5"] = p[:, 5]
            predictions[name+"-Dev16"] = p[:, selected[fold, "row100", kind]]
            predictions[name+"-Global4"] = p[:, selected[fold, name+"-Global4"]]
        all_y.append(d["label"])
        all_first.append(first)
        for name, p in predictions.items():
            pooled.setdefault(name, []).append(p)
            fold_rows.append(dict(fold_id=fold, method=name, **metrics(d["label"], p, first)))
    write_csv(out/"mode_decomposition.csv", decompositions)
    write_csv(out/"direct_fold_results.csv", fold_rows)
    y, first = np.concatenate(all_y), np.concatenate(all_first)
    write_csv(out/"direct_results.csv", [dict(method=k, **metrics(y, np.concatenate(v), first))
                                        for k, v in pooled.items()])
    dump(out/"replay_receipt.json", dict(status="COMPLETED", training=False, model_inference=False,
         purpose="Arithmetic parity on already-exposed frozen predictions, not new research evidence",
         samples=int(len(y)), decomposition_cases=len(decompositions), simple_control_choices=len(controls),
         inputs=inputs, selection_gate_sha256=hashlib.sha256((out/"selection_gate.json").read_bytes()).hexdigest()))
    print(f"Replayed {len(decompositions)} cases and {len(controls)} control choices on {len(y):,} segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    args = parser.parse_args()
    run(args.matrices.resolve(), args.references.resolve(), args.output.resolve())
