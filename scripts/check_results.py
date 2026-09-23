"""Check included aggregate arithmetic, not labels, fitting or scientific validity."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = ["results/direct_controls/pooled.csv", "results/direct_controls/fold_metrics.csv",
             "results/backend/pooled.csv", "results/external_transfer/pooled.csv",
             "results/raw_input/verified_results.csv"]
    checked = 0
    for name in files:
        with (ROOT/name).open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                tn, fp, fn, tp = [int(r[k]) for k in ("tn", "fp", "fn", "tp")]
                assert min(tn, fp, fn, tp) >= 0, name
                assert tn + fp + fn + tp == int(r["n"]), name
                expected = dict(macro_f1=tp/max(2*tp+fp+fn, 1)+tn/max(2*tn+fp+fn, 1),
                                malicious_recall=tp/max(tp+fn, 1),
                                false_positive_rate=fp/max(tn+fp, 1))
                for k, value in expected.items():
                    assert abs(value-float(r[k])) < 1e-12, (name, r.get("method"), k)
                if all(k in r for k in ("C", "D", "switches")):
                    assert int(r["C"])+int(r["D"]) == int(r["switches"])
                if all(k in r for k in ("corrected", "introduced", "switches")):
                    assert int(r["corrected"])+int(r["introduced"]) == int(r["switches"])
                checked += 1
    print(json.dumps(dict(status="PASS", aggregate_rows_checked=checked,
                         scope="Arithmetic only; not raw-label or fitting-chain certification")))


if __name__ == "__main__":
    main()
