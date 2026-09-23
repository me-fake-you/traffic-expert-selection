"use strict";
const notes = {
  controlled: "Original HGB experts · 40,000 exposed USTC-TFC2016 segments · five outer folds pooled. These nine policies are covered by the portable direct-control replay.",
  backend: "Stats ExtraTrees sensitivity · same 40,000-segment protocol; Temporal remains frozen, Stats OOF/reliability and arbiters are refitted. This is not an independent external validation.",
  raw: "625,523 common eligible segments from 24 same-source captures, at natural frequencies. This is a different population from the 40k experiment; the larger F1 is not a new-model gain. One timing run does not establish stable speedup.",
  external: "Frozen transfer: five USTC bundles × 11 policies on 1,159,418 IoT-23 author flows from seven dependent captures. No refitting or reselection; both the source and sample unit change. All 55 configurations are retained below."
};
let records;
function renderResults(key) {
  const body = document.getElementById("result-rows");
  body.replaceChildren();
  for (const r of records[key]) {
    const row = document.createElement("tr");
    const values = [r.method, ...[r.macro_f1, r.malicious_recall, r.false_positive_rate].map(v => (100*v).toFixed(3)+"%"), r.C.toLocaleString("en-US"), r.D.toLocaleString("en-US")];
    for (const value of values) { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); }
    body.append(row);
  }
  document.getElementById("population-note").textContent = notes[key];
  document.getElementById("download-results").href = `downloads/${key}.csv`;
}
if (typeof RECORDED_RESULTS !== "undefined") {
  records = RECORDED_RESULTS;
  renderResults("controlled");
  document.getElementById("population").addEventListener("change", e => renderResults(e.target.value));
} else {
  document.getElementById("result-rows").replaceChildren();
  document.getElementById("population-note").textContent = "The local results asset is missing. Rebuild it using scripts/build_page_data.py; the CSV download remains available.";
}
document.getElementById("copy-command").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById("commands").textContent);
    document.getElementById("copy-status").textContent = "Copied the three local commands.";
  } catch {
    document.getElementById("copy-status").textContent = "Clipboard access unavailable. Select and copy the commands above.";
  }
});
