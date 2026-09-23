"""Build the static page data from included recorded CSVs; no model execution."""
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = dict(controlled="results/direct_controls/pooled.csv", backend="results/backend/pooled.csv",
               raw="results/raw_input/verified_results.csv", external="results/external_transfer/pooled.csv")
downloads = ROOT/"docs/downloads"
downloads.mkdir(exist_ok=True)
output = {}
for key, source in SOURCES.items():
    with (ROOT/source).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output[key] = []
    for r in rows:
        name = r["method"] + (f" · bundle {r['bundle']}" if key == "external" else "")
        output[key].append(dict(method=name, **{k:float(r[k]) for k in ("macro_f1", "malicious_recall", "false_positive_rate")},
                               C=int(r.get("C", r.get("C_vs_Temporal", r.get("corrected", 0)))),
                               D=int(r.get("D", r.get("D_vs_Temporal", r.get("introduced", 0))))))
    shutil.copy2(ROOT/source, downloads/f"{key}.csv")
(ROOT/"docs/assets/results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
(ROOT/"docs/assets/results-data.js").write_text("const RECORDED_RESULTS = " + json.dumps(output, indent=2) + ";\n", encoding="utf-8")
for a, b in (("REPRODUCING.md", "reproducing.md"), ("EVIDENCE_MAP.md", "evidence-map.md"), ("DATA.md", "data.md")):
    shutil.copy2(ROOT/a, downloads/b)
print(json.dumps({key:len(value) for key,value in output.items()}))
