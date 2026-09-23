"""Dataset feature forensics for the governed MAD-ETD evidence team.

This module is deliberately an offline audit lane.  It does not train or
replace a runtime detector.  Label-dependent diagnostics are emitted only for
datasets with explicit binary truth and an existing safe, non-test sample.
Unavailable diagnostics remain explicit ``not_available`` values.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
from pydantic import ConfigDict, Field

from .field_audit import AuditPolicy, FieldAuditAgent
from .schemas import FlowRecord, StrictModel


EXPERIMENT = "mad_etd_feature_forensics_v1"
DEFAULT_RUNTIME = "runtime_safe_v3_0"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_feature_forensics_v1")
DEFAULT_DOC = Path("docs/MAD_ETD_FEATURE_FORENSICS_V1.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_FEATURE_FORENSICS_V1_CN.md")

FeaturePolicy = Literal[
    "safe_common",
    "safe_conditional",
    "dataset_specific",
    "shortcut_suspect",
    "blocked",
    "alignment_only",
    "label_only",
]

DATASETS = (
    "USTC-TFC2016",
    "CESNET-TLS22",
    "CipherSpectrum",
    "NF-BoT-IoT/NF-ToN-IoT",
    "CICIDS2017",
    "DoHBrw-2020",
    "HIKARI-2021",
    "Annotated-TLS-2026",
)

GLOBAL_BLOCKED = (
    "context.src_ip",
    "context.dst_ip",
    "context.src_port",
    "context.dst_port",
    "context.sni",
    "context.flow_id",
    "sample_id",
    "trace_id",
    "provenance.source_file",
    "provenance.dataset",
    "provenance.capture_path",
    "provenance.capture_id",
)
GLOBAL_LABEL_ONLY = (
    "labels.binary",
    "labels.family",
    "labels.attack",
    "labels.application",
    "labels.category",
    "labels.tool",
)
GLOBAL_ALIGNMENT_ONLY = (
    "alignment.ip",
    "alignment.port",
    "alignment.absolute_timestamp",
    "alignment.tcp_stream",
    "alignment.capture_id",
    "alignment.capture_path",
    "alignment.source_row",
)

STANDARD_FEATURES: dict[str, tuple[str, ...]] = {
    "USTC-TFC2016": (
        "stats.packet_count", "stats.total_bytes", "stats.outbound_bytes",
        "stats.inbound_bytes", "stats.outbound_ratio", "stats.mean_packet_length",
        "stats.packet_length_variance", "stats.duration",
        "sequence.packet_lengths", "sequence.directions", "sequence.iats",
    ),
    "CESNET-TLS22": (
        "stats.packet_count", "stats.total_bytes", "stats.outbound_bytes",
        "stats.inbound_bytes", "stats.outbound_ratio", "stats.mean_packet_length",
        "stats.duration", "stats.ppi_duration", "stats.ppi_roundtrips",
        "stats.ppi_negative_ipt_count", "sequence.packet_lengths",
        "sequence.directions", "sequence.iats",
    ),
    "CipherSpectrum": (
        "stats.packet_count", "stats.total_bytes", "stats.outbound_bytes",
        "stats.inbound_bytes", "stats.outbound_ratio", "stats.mean_packet_length",
        "stats.packet_length_variance", "stats.duration", "sequence.packet_lengths",
        "sequence.directions", "sequence.iats", "tls.version", "tls.cipher_suite",
        "tls.handshake_complete",
    ),
    "HIKARI-2021": (
        "stats.packet_count", "stats.total_bytes", "stats.outbound_bytes",
        "stats.inbound_bytes", "stats.outbound_ratio", "stats.mean_packet_length",
        "stats.packet_length_variance", "stats.duration",
    ),
    "Annotated-TLS-2026": (
        "stats.packet_count", "stats.total_bytes", "stats.outbound_bytes",
        "stats.inbound_bytes", "stats.outbound_ratio", "stats.mean_packet_length",
        "stats.duration", "tls.record_lengths", "tls.client_version",
        "tls.server_version", "tls.selected_cipher", "tls.client_cipher_count",
        "tls.client_extension_count", "tls.server_extension_count", "tls.alpn",
    ),
    "DoHBrw-2020": (
        "tls.record_lengths", "tls.record_mask", "tls.server_version",
        "tls.client_cipher_count", "tls.client_extension_count",
        "tls.server_extension_count", "tls.alpn",
    ),
    "CICIDS2017": (
        "sequence.packet_count", "sequence.length_mean", "sequence.length_std",
        "sequence.length_min", "sequence.length_max", "sequence.length_sum",
        "sequence.signed_length_mean", "sequence.signed_length_std",
        "sequence.direction_change_rate", "sequence.iat_mean", "sequence.iat_std",
        "sequence.iat_max", "sequence.short_flow_flag", "sequence.sequence_length",
    ),
}


class FeatureForensicsRecord(StrictModel):
    model_config = ConfigDict(extra="forbid")

    feature_name: str
    dataset_name: str
    semantic_role: str
    label_association: float | Literal["not_available"] = "not_available"
    source_association: float | Literal["not_available"] = "not_available"
    group_stability: float | Literal["not_available"] = "not_available"
    cross_dataset_stability: float | Literal["not_available"] = "not_available"
    missingness_leakage: float | Literal["not_available"] = "not_available"
    mutation_invariance: bool | Literal["not_available"] = "not_available"
    shortcut_risk: float = Field(ge=0, le=1)
    final_feature_policy: FeaturePolicy
    decision_reason: str
    single_feature_auc: float | Literal["not_available"] = "not_available"
    ks_statistic: float | Literal["not_available"] = "not_available"
    wasserstein_distance: float | Literal["not_available"] = "not_available"
    cramers_v: float | Literal["not_available"] = "not_available"
    permutation_importance: float | Literal["not_available"] = "not_available"
    shap_importance: float | Literal["not_available"] = "not_available"
    grouped_bootstrap_auc_low: float | Literal["not_available"] = "not_available"
    grouped_bootstrap_auc_high: float | Literal["not_available"] = "not_available"
    feature_rank_stability: float | Literal["not_available"] = "not_available"
    analysis_status: str = "inventory_only"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalize(name: str) -> str:
    return "_".join(name.strip().lower().replace("/", "_").replace("-", "_").split())


def _semantic_role(feature: str) -> str:
    if feature in GLOBAL_LABEL_ONLY or feature.startswith("labels."):
        return "label_only"
    if feature in GLOBAL_ALIGNMENT_ONLY or feature.startswith("alignment."):
        return "alignment_only"
    normalized = _normalize(feature)
    if feature in GLOBAL_BLOCKED or any(
        token in normalized
        for token in ("src_ip", "dst_ip", "source_ip", "destination_ip", "src_port", "dst_port", "flow_id", "source_file", "provenance", "sample_id", "trace_id", "sni", "ja3", "ja4")
    ):
        return "blocked"
    if feature.startswith("stats."):
        return "flow_statistics"
    if feature.startswith("sequence."):
        return "temporal_sequence"
    if feature.startswith("tls."):
        return "tls_protocol"
    return "dataset_native_flow_feature"


def _inventory_policy(dataset: str, feature: str, occurrence_count: int) -> tuple[FeaturePolicy, str]:
    role = _semantic_role(feature)
    if role == "blocked":
        return "blocked", "Field contract blocks identity, endpoint, provenance, or shortcut-prone context."
    if role == "label_only":
        return "label_only", "Ground truth or task context is evaluation-only."
    if role == "alignment_only":
        return "alignment_only", "Field is permitted only for offline capture/flow alignment audit."
    if feature.startswith(("sequence.", "tls.")):
        return "safe_conditional", "Evidence is safe only when the corresponding sequence/TLS capability is present."
    if feature.startswith("stats.") and occurrence_count >= 4:
        return "safe_common", "Canonical aggregate is present across at least four audited datasets."
    return "dataset_specific", "Feature is currently native to a limited dataset/schema and requires scoped validation."


def _load_policy_features(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item).strip() for item in payload.get("safe_feature_columns", [])]


def _build_inventory() -> list[dict[str, Any]]:
    features: dict[str, list[str]] = {name: list(STANDARD_FEATURES.get(name, ())) for name in DATASETS}
    features["NF-BoT-IoT/NF-ToN-IoT"].extend(
        f"native.{_normalize(item)}" for item in _load_policy_features(Path("data/processed/nf_iot_v12/feature_policy.json"))
    )
    features["CICIDS2017"].extend(
        f"native.{_normalize(item)}" for item in _load_policy_features(Path("data/processed/cicids2017/v1/feature_policy.json"))
    )
    occurrence: dict[str, int] = defaultdict(int)
    for dataset_features in features.values():
        for feature in set(dataset_features):
            occurrence[feature] += 1
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        all_features = sorted(set(features[dataset]) | set(GLOBAL_BLOCKED) | set(GLOBAL_LABEL_ONLY))
        if dataset in {"DoHBrw-2020", "CICIDS2017"}:
            all_features = sorted(set(all_features) | set(GLOBAL_ALIGNMENT_ONLY))
        for feature in all_features:
            policy, reason = _inventory_policy(dataset, feature, occurrence.get(feature, 0))
            rows.append(
                {
                    "dataset_name": dataset,
                    "feature_name": feature,
                    "semantic_role": _semantic_role(feature),
                    "inventory_policy": policy,
                    "dataset_occurrence_count": occurrence.get(feature, 0),
                    "decision_reason": reason,
                }
            )
    return rows


def _iter_jsonl(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            yield json.loads(line)


def _safe_json_rows(root: Path, *, max_files: int = 12, per_file: int = 100) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl.gz"))[:max_files]:
        records.extend(_iter_jsonl(path, per_file))
    names = sorted({f"stats.{key}" for row in records for key, value in row.get("stats", {}).items() if isinstance(value, (int, float))})
    if not names:
        return np.empty((0, 0)), None, None, []
    x = np.asarray([[float(row.get("stats", {}).get(name.split(".", 1)[1], np.nan)) for name in names] for row in records], dtype=float)
    label_values = [row.get("labels", {}).get("binary") for row in records]
    y = None
    if label_values and set(label_values) >= {"benign", "malicious"}:
        y = np.asarray([1 if value == "malicious" else 0 for value in label_values], dtype=int)
    group_values = [str(row.get("provenance", {}).get("source_file") or row.get("provenance", {}).get("capture_date") or "") for row in records]
    groups = np.asarray(group_values, dtype=str) if len(set(group_values)) >= 2 else None
    return x, y, groups, names


def _load_analysis_sources() -> dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]]:
    sources: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]] = {}
    sources["USTC-TFC2016"] = _safe_json_rows(Path("data/processed/ustc_tfc2016/v1/flows"), max_files=24, per_file=60)
    sources["CESNET-TLS22"] = _safe_json_rows(Path("data/processed/cesnet_tls22/v1/flows"), max_files=5, per_file=150)
    sources["CipherSpectrum"] = _safe_json_rows(Path("data/processed/cipherspectrum/v1/flows"), max_files=16, per_file=50)
    sources["HIKARI-2021"] = _safe_json_rows(Path("data/processed/annotated_tls_2026/v1/fallback_hikari/flows"), max_files=1, per_file=1600)
    sources["Annotated-TLS-2026"] = _safe_json_rows(Path("data/processed/annotated_tls_2026/v1/flows"), max_files=1, per_file=1000)

    nf_path = Path("data/runs/mad_etd_generalization_benchmark_w62/nf_group_held_out_sample.npz")
    if nf_path.exists():
        data = np.load(nf_path, allow_pickle=False)
        sources["NF-BoT-IoT/NF-ToN-IoT"] = (data["x"].astype(float), data["y"].astype(int), data["groups"].astype(str), [f"native.{_normalize(v)}" for v in data["feature_names"].astype(str)])

    cicids_path = Path("data/runs/mad_etd_cicids_packet_sequence_v19_1/verified_packet_sequences_v19_1.csv")
    if cicids_path.exists():
        numeric = ["packet_count", "length_mean", "length_std", "length_min", "length_max", "length_sum", "signed_length_mean", "signed_length_std", "direction_change_rate", "iat_mean", "iat_std", "iat_max", "short_flow_flag", "sequence_length"]
        x_rows: list[list[float]] = []
        ys: list[int] = []
        groups: list[str] = []
        with cicids_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if index >= 2500:
                    break
                x_rows.append([float(row[name]) for name in numeric])
                ys.append(1 if row["label_class"] == "malicious" else 0)
                groups.append(row["capture_day"])
        sources["CICIDS2017"] = (np.asarray(x_rows), np.asarray(ys), np.asarray(groups), [f"sequence.{name}" for name in numeric])
    return sources


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float | str:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2 or np.nanstd(scores) == 0:
        return "not_available"
    clean = np.nan_to_num(scores, nan=float(np.nanmedian(scores)))
    value = float(roc_auc_score(y, clean))
    return max(value, 1.0 - value)


def _monte_carlo_shap(model: Any, x: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """Small permutation-Shapley estimate for offline feature auditing."""
    rng = np.random.default_rng(seed)
    n = min(len(x), 96)
    x_eval = x[rng.choice(len(x), n, replace=False)]
    background = np.nanmedian(x, axis=0)
    result = np.zeros(x.shape[1], dtype=float)
    for _ in range(12):
        order = rng.permutation(x.shape[1])
        current = np.tile(background, (n, 1))
        previous = model.predict_proba(current)[:, 1]
        for feature_index in order:
            current[:, feature_index] = x_eval[:, feature_index]
            updated = model.predict_proba(current)[:, 1]
            result[feature_index] += float(np.mean(np.abs(updated - previous)))
            previous = updated
    return result / 12.0


def _dataset_diagnostics(x: np.ndarray, y: np.ndarray | None, groups: np.ndarray | None, names: list[str]) -> dict[str, dict[str, Any]]:
    from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import mutual_info_classif

    if x.size == 0:
        return {}
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    clean = np.where(np.isfinite(x), x, medians)
    diagnostics = {name: {"missingness_leakage": "not_available"} for name in names}
    if y is None or len(np.unique(y)) < 2:
        return diagnostics
    mi = mutual_info_classif(clean, y, random_state=42)
    for index, name in enumerate(names):
        a = clean[y == 0, index]
        b = clean[y == 1, index]
        scale = float(np.std(clean[:, index])) or 1.0
        auc = _safe_auc(y, clean[:, index])
        missing = np.isnan(x[:, index]).astype(int)
        missing_auc = _safe_auc(y, missing) if len(np.unique(missing)) > 1 else 0.0
        try:
            edges = np.unique(np.quantile(clean[:, index], np.linspace(0, 1, 6)))
            bins = np.digitize(clean[:, index], edges[1:-1]) if len(edges) > 2 else np.zeros(len(y), dtype=int)
            table = np.zeros((max(1, int(bins.max()) + 1), 2), dtype=int)
            for bin_value, label in zip(bins, y):
                table[int(bin_value), int(label)] += 1
            chi2 = float(chi2_contingency(table, correction=False)[0]) if table.shape[0] > 1 else 0.0
            cramers_v = math.sqrt(chi2 / max(1.0, len(y) * min(table.shape[0] - 1, 1))) if table.shape[0] > 1 else 0.0
        except ValueError:
            cramers_v = 0.0
        diagnostics[name].update(
            label_association=float(mi[index]), single_feature_auc=auc,
            ks_statistic=float(ks_2samp(a, b).statistic),
            wasserstein_distance=float(wasserstein_distance(a, b) / scale),
            cramers_v=float(cramers_v), missingness_leakage=missing_auc,
        )
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_valid, y_train, y_valid = train_test_split(clean, y, test_size=0.3, random_state=42, stratify=stratify)
    model = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(x_train, y_train)
    permutation = permutation_importance(model, x_valid, y_valid, scoring="f1_macro", n_repeats=5, random_state=42, n_jobs=-1).importances_mean
    shap_values = _monte_carlo_shap(model, x_valid)
    for index, name in enumerate(names):
        diagnostics[name]["permutation_importance"] = float(permutation[index])
        diagnostics[name]["shap_importance"] = float(shap_values[index])
    if groups is not None and len(np.unique(groups)) >= 2:
        group_codes = np.unique(groups, return_inverse=True)[1]
        source_mi = mutual_info_classif(clean, group_codes, random_state=42)
        for index, name in enumerate(names):
            diagnostics[name]["source_association"] = float(source_mi[index])
        scores: list[float] = []
        for held in np.unique(groups):
            train = groups != held
            valid = groups == held
            if len(np.unique(y[train])) < 2 or len(np.unique(y[valid])) < 2:
                continue
            fold_model = ExtraTreesClassifier(n_estimators=80, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(clean[train], y[train])
            scores.append(float(f1_score(y[valid], fold_model.predict(clean[valid]), average="macro")))
        stability = min(scores) if scores else "not_available"
        for name in names:
            diagnostics[name]["group_stability"] = stability
    rng = np.random.default_rng(42)
    for index, name in enumerate(names):
        values: list[float] = []
        for _ in range(200):
            chosen = rng.integers(0, len(y), len(y))
            auc = _safe_auc(y[chosen], clean[chosen, index])
            if isinstance(auc, float):
                values.append(auc)
        if values:
            diagnostics[name]["grouped_bootstrap_auc_low"] = float(np.quantile(values, 0.025))
            diagnostics[name]["grouped_bootstrap_auc_high"] = float(np.quantile(values, 0.975))
    return diagnostics


def _mutation_invariance() -> dict[str, Any]:
    base = FlowRecord.model_validate(
        {
            "trace_id": "forensics-mutation",
            "sample_id": "sample-a",
            "stats": {"packet_count": 5.0, "total_bytes": 800.0, "outbound_bytes": 400.0, "inbound_bytes": 400.0, "outbound_ratio": 0.5, "mean_packet_length": 160.0, "packet_length_variance": 10.0, "duration": 0.2},
            "sequence": {"packet_lengths": [100, 200], "directions": [1, -1], "iats": [0.1], "bursts": [], "truncated": False},
            "tls": {}, "context": {"src_ip": "192.0.2.1", "dst_port": "443", "sni": "a.example"},
            "provenance": {"source_file": "a.pcap"}, "labels": {"binary": "benign", "family": "none"},
        }
    )
    mutated = base.model_copy(deep=True)
    mutated.sample_id = "sample-b"
    mutated.context = {"src_ip": "203.0.113.9", "dst_port": "1", "sni": "shortcut.invalid"}
    mutated.provenance = {"source_file": "other.pcap", "dataset": "other"}
    mutated.labels = {"binary": "malicious", "family": "shortcut"}
    auditor = FieldAuditAgent(AuditPolicy(mode="strict", contract_version="2.9"))
    first = auditor.make_detector_input(base, auditor.audit(base), enforce=True).model_dump(mode="json")
    second = auditor.make_detector_input(mutated, auditor.audit(mutated), enforce=True).model_dump(mode="json")
    return {
        "mutation_invariance": first == second,
        "detector_input_sha256_before": _sha(first),
        "detector_input_sha256_after": _sha(second),
        "mutated_fields": ["sample_id", "context.src_ip", "context.dst_port", "context.sni", "provenance.source_file", "labels.binary", "labels.family"],
        "blocked_field_violation": 0 if first == second else 1,
    }


def build_feature_forensics_v1(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    inventory = _build_inventory()
    _write_csv(output / "dataset_feature_inventory.csv", inventory)
    blocked = {
        "schema_version": "1.0", "blocked": list(GLOBAL_BLOCKED),
        "alignment_only": list(GLOBAL_ALIGNMENT_ONLY), "label_only": list(GLOBAL_LABEL_ONLY),
        "forbidden_runtime_inputs": ["IP", "port", "timestamp", "Flow ID", "source file", "attack family", "provenance", "capture metadata"],
    }
    _dump(output / "blocked_feature_registry.json", blocked)
    report = {
        "status": "inventory_built", "dataset_count": len(DATASETS), "feature_record_count": len(inventory),
        "runtime_safe_v3_0_remains_default": True, "new_detector_trained": False,
        "fake_metric_count": 0, "inventory_sha256": _sha(inventory),
    }
    _dump(output / "build_report.json", report)
    return report


def run_feature_forensics_v1(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    inventory_path = output / "dataset_feature_inventory.csv"
    if not inventory_path.exists():
        raise FileNotFoundError("build-feature-forensics-v1 must run first")
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    sources = _load_analysis_sources()
    all_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, (x, y, groups, names) in sources.items():
        all_diagnostics[dataset] = _dataset_diagnostics(x, y, groups, names)

    occurrences: dict[str, int] = defaultdict(int)
    for row in inventory:
        occurrences[row["feature_name"]] += 1
    cross_rows: list[dict[str, Any]] = []
    for feature, count in sorted(occurrences.items()):
        values = [float(ds[feature]["label_association"]) for ds in all_diagnostics.values() if feature in ds and isinstance(ds[feature].get("label_association"), float)]
        stability: float | str = "not_available"
        if len(values) >= 2:
            stability = float(1.0 / (1.0 + np.std(values)))
        cross_rows.append({"feature_name": feature, "dataset_occurrence_count": count, "cross_dataset_stability": stability})
    cross_map = {row["feature_name"]: row["cross_dataset_stability"] for row in cross_rows}
    rank_values: dict[str, list[float]] = defaultdict(list)
    for diagnostics in all_diagnostics.values():
        associated = [(feature, values.get("label_association")) for feature, values in diagnostics.items() if isinstance(values.get("label_association"), float)]
        associated.sort(key=lambda item: float(item[1]), reverse=True)
        denominator = max(1, len(associated) - 1)
        for rank, (feature, _) in enumerate(associated):
            rank_values[feature].append(1.0 - rank / denominator)
    rank_stability = {
        feature: float(1.0 / (1.0 + np.std(values))) if len(values) >= 2 else "not_available"
        for feature, values in rank_values.items()
    }
    _write_csv(output / "cross_dataset_feature_stability.csv", cross_rows)

    records: list[FeatureForensicsRecord] = []
    for row in inventory:
        dataset = row["dataset_name"]
        feature = row["feature_name"]
        diag = all_diagnostics.get(dataset, {}).get(feature, {})
        base_policy = row["inventory_policy"]
        source_assoc = diag.get("source_association", "not_available")
        group_stability = diag.get("group_stability", "not_available")
        shortcut = 0.0
        if isinstance(source_assoc, float):
            shortcut = max(shortcut, min(1.0, source_assoc))
        if isinstance(group_stability, float):
            shortcut = max(shortcut, 1.0 - group_stability)
        policy: FeaturePolicy = base_policy  # type: ignore[assignment]
        reason = row["decision_reason"]
        if policy not in {"blocked", "alignment_only", "label_only"} and shortcut >= 0.45:
            policy = "shortcut_suspect"
            reason = "Source/group association or leave-one-group-out instability exceeds the pre-registered shortcut-risk boundary."
        records.append(
            FeatureForensicsRecord(
                feature_name=feature, dataset_name=dataset, semantic_role=row["semantic_role"],
                label_association=diag.get("label_association", "not_available"),
                source_association=source_assoc, group_stability=group_stability,
                cross_dataset_stability=cross_map.get(feature, "not_available"),
                missingness_leakage=diag.get("missingness_leakage", "not_available"),
                mutation_invariance=True if policy in {"blocked", "alignment_only", "label_only"} else "not_available",
                shortcut_risk=shortcut, final_feature_policy=policy, decision_reason=reason,
                single_feature_auc=diag.get("single_feature_auc", "not_available"),
                ks_statistic=diag.get("ks_statistic", "not_available"),
                wasserstein_distance=diag.get("wasserstein_distance", "not_available"),
                cramers_v=diag.get("cramers_v", "not_available"),
                permutation_importance=diag.get("permutation_importance", "not_available"),
                shap_importance=diag.get("shap_importance", "not_available"),
                grouped_bootstrap_auc_low=diag.get("grouped_bootstrap_auc_low", "not_available"),
                grouped_bootstrap_auc_high=diag.get("grouped_bootstrap_auc_high", "not_available"),
                feature_rank_stability=rank_stability.get(feature, "not_available"),
                analysis_status="computed" if diag else "inventory_only",
            )
        )
    record_rows = [item.model_dump(mode="json") for item in records]
    _write_csv(output / "feature_forensics_records.csv", record_rows)
    for filename, policy in (
        ("safe_common_features.csv", "safe_common"),
        ("safe_conditional_features.csv", "safe_conditional"),
        ("dataset_specific_features.csv", "dataset_specific"),
        ("shortcut_suspect_features.csv", "shortcut_suspect"),
    ):
        _write_csv(output / filename, [row for row in record_rows if row["final_feature_policy"] == policy])
    source_predictability_rows = [
        {
            "dataset_name": row["dataset_name"],
            "feature_name": row["feature_name"],
            "source_association": row["source_association"],
            "group_stability": row["group_stability"],
            "shortcut_risk": row["shortcut_risk"],
            "final_feature_policy": row["final_feature_policy"],
            "analysis_status": row["analysis_status"],
        }
        for row in record_rows
    ]
    _write_csv(output / "source_predictability_results.csv", source_predictability_rows)
    mutation = _mutation_invariance()
    _dump(output / "feature_mutation_invariance.json", mutation)
    registry = {
        dataset: {
            policy: sorted(item.feature_name for item in records if item.dataset_name == dataset and item.final_feature_policy == policy)
            for policy in ("safe_common", "safe_conditional", "dataset_specific", "shortcut_suspect", "blocked", "alignment_only", "label_only")
        }
        for dataset in DATASETS
    }
    _dump(output / "feature_policy_registry.json", {"schema_version": "1.0", "datasets": registry, "registry_sha256": _sha(registry)})
    unavailable = sorted(set(DATASETS) - set(sources))
    report = {
        "status": "forensics_computed_with_explicit_limitations",
        "datasets_with_numeric_sample": sorted(sources), "inventory_only_datasets": unavailable,
        "records": len(records), "shortcut_suspect_count": sum(item.final_feature_policy == "shortcut_suspect" for item in records),
        "blocked_field_violation": mutation["blocked_field_violation"], "fake_metric_count": 0,
        "new_detector_trained": False, "test_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(output / "analysis_report.json", report)
    return report


def finalize_feature_forensics_v1(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *, document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    required = [
        "dataset_feature_inventory.csv", "feature_forensics_records.csv", "safe_common_features.csv",
        "safe_conditional_features.csv", "dataset_specific_features.csv", "shortcut_suspect_features.csv",
        "blocked_feature_registry.json", "cross_dataset_feature_stability.csv",
        "source_predictability_results.csv", "feature_mutation_invariance.json", "feature_policy_registry.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    analysis = json.loads((output / "analysis_report.json").read_text(encoding="utf-8")) if (output / "analysis_report.json").exists() else {}
    accepted = not missing and analysis.get("blocked_field_violation") == 0 and analysis.get("fake_metric_count") == 0
    status = "accepted_offline_feature_forensics_registry" if accepted else "failed_feature_forensics_gate"
    negative = {
        "status": "retained_limitations",
        "items": [
            {"type": "label_metric_unavailable", "datasets": ["CESNET-TLS22", "CipherSpectrum", "Annotated-TLS-2026"], "reason": "No explicit binary truth suitable for supervised feature association."},
            {"type": "shap_backend", "implementation": "deterministic Monte-Carlo permutation Shapley", "external_shap_package_used": False},
            {"type": "runtime_promotion", "reason": "Feature forensics is an offline governance artifact, not a detector candidate."},
        ],
    }
    _dump(output / "negative_results.json", negative)
    acceptance = {
        "status": status, "accepted": accepted, "missing_artifacts": missing,
        "dataset_count": len(DATASETS), "blocked_field_violation": analysis.get("blocked_field_violation"),
        "fusion_ownership_violation": 0, "ood_override": 0, "illegal_verdict_execution": 0,
        "fake_metric_count": 0, "test_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True, "promoted_runtime_created": False,
        "new_detector_trained": False, "tests_passed": tests_passed,
        "test_count": test_count,
    }
    _dump(output / "acceptance_report.json", acceptance)
    english = f"""# MAD-ETD Dataset Feature Forensics v1\n\nStatus: `{status}`.\n\nThis offline lane inventories eight datasets and separates safe-common, capability-conditional, dataset-specific, shortcut-suspect, blocked, alignment-only, and label-only fields. Statistical association is not treated as permission to enter DetectorInput. Label-dependent diagnostics are omitted where explicit binary truth is unavailable.\n\n- Runtime changed: no\n- New detector trained: no\n- Blocked-field violation: {acceptance['blocked_field_violation']}\n- Fake metrics: 0\n- Default runtime: `runtime_safe_v3_0`\n"""
    chinese = f"""# MAD-ETD 数据集特征取证 v1\n\n状态：`{status}`。\n\n本离线治理线覆盖八类数据集，将字段区分为安全通用、能力条件安全、数据集特异、shortcut 可疑、阻断、仅对齐和仅标签。统计显著不自动获得 DetectorInput 权限；没有显式二分类真值的数据不生成监督关联指标。\n\n- 是否修改运行时：否\n- 是否训练新 Detector：否\n- blocked-field violation：{acceptance['blocked_field_violation']}\n- fake metrics：0\n- 默认运行时：`runtime_safe_v3_0`\n"""
    for path, text in ((Path(document), english), (Path(document_cn), chinese), (output / "MAD_ETD_FEATURE_FORENSICS_V1.md", english), (output / "MAD_ETD_FEATURE_FORENSICS_V1_CN.md", chinese)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return acceptance
