"use strict";
const catalog = window.THESIS_CATALOG;
const list = document.getElementById("catalog-list");
const search = document.getElementById("experiment-search");
const scope = document.getElementById("experiment-scope");
const status = document.getElementById("catalog-status");
const repo = "https://github.com/me-fake-you/traffic-expert-selection/tree/main/";
function renderCatalog() {
  const query = search.value.trim().toLowerCase();
  const rows = catalog.families.filter(row => (scope.value === "all" || scope.value === row.scope) && (row.id + " " + row.source_family).toLowerCase().includes(query));
  list.replaceChildren();
  rows.forEach(row => {
    const card = document.createElement("article"); card.className = "catalog-card";
    const badge = document.createElement("span"); badge.className = "catalog-badge";
    badge.textContent = row.scope === "historical_thesis_evidence" ? "THESIS EVIDENCE" : "DEVELOPMENT EXTENSION";
    const heading = document.createElement("h3");
    const link = document.createElement("a"); link.href = repo + row.path.split("/").map(encodeURIComponent).join("/");
    link.textContent = row.id.replace(/^thesis_/, "").replaceAll("_", " ") + " ↗";
    heading.append(link);
    const count = document.createElement("p"); count.textContent = row.aggregate_files + " aggregate files · historical record";
    card.append(badge, heading, count); list.append(card);
  });
  if (!rows.length) { const empty=document.createElement("p"); empty.className="catalog-empty"; empty.textContent="No matching records. Try a shorter topic or choose all records."; list.append(empty); }
  status.textContent = rows.length + " of " + catalog.families.length + " evidence folders shown. Folder counts are not independent experiments.";
}
if (catalog) {
  document.getElementById("code-count").textContent = catalog.counts.code;
  document.getElementById("family-count").textContent = catalog.family_count;
  search.addEventListener("input", renderCatalog); scope.addEventListener("change", renderCatalog); renderCatalog();
} else { status.textContent="The catalog did not load. Use the complete CSV index below."; }
document.getElementById("copy-thesis-command").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(document.getElementById("thesis-command").textContent); document.getElementById("thesis-copy-status").textContent="Commands copied."; }
  catch { document.getElementById("thesis-copy-status").textContent="Select the commands above to copy them."; }
});
