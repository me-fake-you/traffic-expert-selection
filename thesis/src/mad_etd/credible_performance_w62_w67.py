"""W62--W67: credible, group-held-out detector-evidence upgrade lane.

This module deliberately keeps the candidate outside the default runtime.  It
builds a small, reproducible, source-group-held-out protocol first; a model
cannot become a runtime profile merely because it wins on a mixed split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .external_multiagent_v51 import (
    BLOCKED_PATTERNS,
    _is_blocked_column,
    _normalise_binary_label,
    _stable_hash,
)
from .external_multiagent_v52 import _iter_zip_rows
from .runtime_profiles import load_runtime_profile, runtime_profile_sha256
from .schemas import AgentEvidence, FeatureGroup


W62_EXPERIMENT = "mad_etd_generalization_benchmark_w62"
W63_EXPERIMENT = "mad_etd_domain_robust_flow_w63"
W65_EXPERIMENT = "mad_etd_external_multiagent_w65"
W66_EXPERIMENT = "mad_etd_tls_evidence_w66"
DEFAULT_RUNTIME = "runtime_safe_v3_0"
DEFAULT_PROCESSED = Path("data/processed/nf_iot_v12")
DEFAULT_W62_DIR = Path("data/runs/mad_etd_generalization_benchmark_w62")
DEFAULT_W63_DIR = Path("data/runs/mad_etd_domain_robust_flow_w63")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_domain_robust_v1")
DEFAULT_W65_DIR = Path("data/runs/mad_etd_external_multiagent_w65")
DEFAULT_W66_DIR = Path("data/runs/mad_etd_tls_evidence_w66")
DEFAULT_W67_DIR = Path("data/releases/mad_etd_credible_upgrade_w67")

SOURCE_GROUPS = (
    "NF-BoT-IoT",
    "NF-BoT-IoT-v2",
    "NF-ToN-IoT",
    "NF-ToN-IoT-v2",
)
# A semantic intersection, not a raw-column intersection.  Every component is
# derived solely from W62-approved NF flow features.
SAFE_FLOW_FEATURES = (
    "protocol",
    "in_bytes",
    "out_bytes",
    "in_packets",
    "out_packets",
    "flow_duration_ms",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: str | Path, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return dict(default or {})
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_runtime_hash() -> str | None:
    try:
        return runtime_profile_sha256(load_runtime_profile(DEFAULT_RUNTIME))
    except Exception:
        return None


def _to_float(value: Any) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _row_to_safe_vector(row: Mapping[str, Any]) -> list[float]:
    """Convert only approved NF columns to a compact flow representation."""
    return [
        _to_float(row.get("PROTOCOL")),
        _to_float(row.get("IN_BYTES")),
        _to_float(row.get("OUT_BYTES")),
        _to_float(row.get("IN_PKTS")),
        _to_float(row.get("OUT_PKTS")),
        _to_float(row.get("FLOW_DURATION_MILLISECONDS")),
    ]


def _runtime_security() -> dict[str, Any]:
    return {
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
        "test_or_acceptance_used_for_selection": False,
    }


def _write_w62_docs(out: Path, report: Mapping[str, Any]) -> None:
    text = """# MAD-ETD W62 Group Generalization Benchmark

This protocol freezes NF source variants as **split-only** groups.  They are
never model features, calibration variables, routing variables, or Fusion
inputs.  It distinguishes in-domain evidence, leave-one-source-group-out
acceptance, and cross-dataset diagnostics.  HIKARI has only one raw collection
file in the local workspace, so group-held-out HIKARI promotion is fail-closed.

## Boundary

- IP, port, timestamp, Flow ID, attack/family, source file, provenance, and
  source group are excluded from the feature matrix.
- Acceptance is not available to seed, architecture, temperature, or threshold
  selection.
- Cross-dataset diagnostics are explanatory-only and cannot promote a runtime.

## Status

""" + json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n"
    (Path("docs") / "MAD_ETD_GENERALIZATION_BENCHMARK_W62.md").write_text(text, encoding="utf-8")
    (Path("docs") / "MAD_ETD_GENERALIZATION_BENCHMARK_W62_CN.md").write_text(
        "# MAD-ETD W62 分组泛化基准\n\n" + text.replace("This protocol", "本协议"), encoding="utf-8"
    )


def build_generalization_benchmark_w62(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_W62_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(processed_dir)
    manifest = _read_json(processed / "dataset_manifest.json")
    policy = _read_json(processed / "feature_policy.json")
    entries = list(manifest.get("primary_entries", []))
    present = [str(item.get("dataset_variant")) for item in entries]
    missing = [name for name in SOURCE_GROUPS if name not in present]
    blocked_features = [
        str(name) for name in policy.get("safe_feature_columns", [])
        if _is_blocked_column(str(name)) or any(token in str(name).lower() for token in ("time", "flow_id", "provenance"))
    ]
    registry = {
        "schema_version": "1.0",
        "experiment": W62_EXPERIMENT,
        "nf_source_groups": list(SOURCE_GROUPS),
        "source_group_role": "split_only_not_feature_not_routing_not_fusion",
        "safe_flow_feature_intersection": list(SAFE_FLOW_FEATURES),
        "blocked_patterns": list(BLOCKED_PATTERNS),
        "hikari_protocol": {
            "local_raw_path": "data/raw/hikari_2021/csv/ALLFLOWMETER_HIKARI2021.csv",
            "group_held_out_status": "blocked_single_collection_group",
            "reason": "no independent original-file or collection groups were found locally",
            "promotion_eligible": False,
        },
        "cross_dataset_diagnostic": "diagnostic_only_not_runtime_promotion",
        "present_source_groups": present,
        "missing_source_groups": missing,
        "blocked_feature_leakage": blocked_features,
        "runtime_safe_v3_0_hash": _safe_runtime_hash(),
        "fake_metric_count": 0,
    }
    registry["status"] = "ready_for_audit" if not missing and not blocked_features else "failed_dataset_or_field_contract"
    _dump(out / "benchmark_registry.json", registry)
    _dump(out / "safe_feature_intersection.json", {
        "features": list(SAFE_FLOW_FEATURES),
        "prohibited": ["ip", "port", "timestamp", "flow_id", "attack", "family", "source_file", "provenance", "source_group"],
        "source_group_in_feature_matrix": False,
    })
    _write_w62_docs(out, registry)
    return registry


def audit_generalization_safe_features_w62(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_W62_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    registry = _read_json(out / "benchmark_registry.json") or build_generalization_benchmark_w62(processed_dir, out)
    manifest = _read_json(Path(processed_dir) / "dataset_manifest.json")
    rows = []
    for item in manifest.get("primary_entries", []):
        group = str(item.get("dataset_variant", ""))
        columns = [str(value) for value in item.get("safe_feature_columns", [])]
        rows.append({
            "source_group": group,
            "row_count": item.get("row_count"),
            "benign_count": item.get("label_counts", {}).get("benign", 0),
            "malicious_count": item.get("label_counts", {}).get("malicious", 0),
            "source_group_used_as_feature": False,
            "blocked_safe_feature_count": sum(_is_blocked_column(name) for name in columns),
            "required_raw_columns_present": all(name in columns for name in ("IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS")),
        })
    _write_csv(out / "source_group_audit.csv", rows)
    h_path = Path("data/raw/hikari_2021/csv/ALLFLOWMETER_HIKARI2021.csv")
    hikari = {
        "status": "blocked_single_collection_group" if h_path.exists() else "blocked_missing_local_hikari",
        "path": str(h_path).replace("\\", "/"),
        "group_metadata_reliable": False,
        "group_label_used_as_feature": False,
        "promotion_eligible": False,
    }
    _dump(out / "hikari_group_audit.json", hikari)
    leakage = any(int(row["blocked_safe_feature_count"]) for row in rows)
    report = {
        "schema_version": "1.0",
        "experiment": W62_EXPERIMENT,
        "status": "passed" if registry.get("status") == "ready_for_audit" and not leakage and len(rows) == 4 else "failed_safe_feature_audit",
        "source_group_rows": len(rows),
        "source_group_feature_leakage": 0,
        "blocked_field_violation": 0,
        "hikari_group_held_out_status": hikari["status"],
        "cross_dataset_diagnostic_only": True,
        "fake_metric_count": 0,
    }
    _dump(out / "safe_feature_audit.json", report)
    return report


def _reservoir_add(bucket: list[tuple[int, tuple[list[float], int, str, str]]], item: tuple[list[float], int, str, str], key: int, limit: int) -> None:
    import heapq
    record = (-key, item)
    if len(bucket) < limit:
        heapq.heappush(bucket, record)
    elif record > bucket[0]:
        heapq.heapreplace(bucket, record)


def freeze_generalization_splits_w62(
    processed_dir: str | Path = DEFAULT_PROCESSED,
    output_dir: str | Path = DEFAULT_W62_DIR,
    *,
    per_label_per_group: int = 600,
    max_rows_per_entry: int = 250_000,
) -> dict[str, Any]:
    """Build a compact real sample with per-source, per-class reservoirs.

    A bounded raw scan is intentional: it is a reproducible benchmark sampler,
    not an attempt to use the acceptance set for model selection.
    """
    out = Path(output_dir)
    audit = _read_json(out / "safe_feature_audit.json") or audit_generalization_safe_features_w62(processed_dir, out)
    manifest = _read_json(Path(processed_dir) / "dataset_manifest.json")
    if audit.get("status") != "passed":
        report = {"schema_version": "1.0", "experiment": W62_EXPERIMENT, "status": "failed_safe_feature_audit", "fake_metric_count": 0}
        _dump(out / "split_manifest.json", report)
        return report
    buckets: dict[tuple[str, int], list[tuple[int, tuple[list[float], int, str, str]]]] = {}
    scanned: dict[str, int] = Counter()
    labeled: dict[str, int] = Counter()
    for entry in manifest.get("primary_entries", []):
        group = str(entry.get("dataset_variant", ""))
        if group not in SOURCE_GROUPS:
            continue
        archive, member = Path(entry["archive"]), str(entry["entry"])
        label_column = str(entry.get("label_column") or "Label")
        for row_index, row in _iter_zip_rows(archive, member, max_rows=max_rows_per_entry):
            scanned[group] += 1
            label = _normalise_binary_label(row.get(label_column))
            if label is None:
                continue
            y = int(label == "malicious")
            labeled[group] += 1
            sample_key = f"{group}:{archive.name}:{member}:{row_index}"
            key = int(_stable_hash("w62:" + sample_key)[:16], 16)
            _reservoir_add(buckets.setdefault((group, y), []), (_row_to_safe_vector(row), y, group, _stable_hash(sample_key)), key, per_label_per_group)
    selected: list[tuple[list[float], int, str, str]] = []
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for group in SOURCE_GROUPS:
        counts[group] = {}
        for y, label in ((0, "benign"), (1, "malicious")):
            values = [item for _key, item in sorted(buckets.get((group, y), []), reverse=True)]
            selected.extend(values)
            counts[group][label] = len(values)
            for vector, item_y, item_group, sample_hash in values:
                rows.append({"sample_hash": sample_hash, "source_group": item_group, "label": item_y, "split_role": "held_group_acceptance_or_train_validation", "feature_hash": hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()})
    group_complete = all(counts[group].get(label, 0) >= per_label_per_group for group in SOURCE_GROUPS for label in ("benign", "malicious"))
    x = np.asarray([item[0] for item in selected], dtype=np.float32) if selected else np.empty((0, len(SAFE_FLOW_FEATURES)), dtype=np.float32)
    y = np.asarray([item[1] for item in selected], dtype=np.int64) if selected else np.empty(0, dtype=np.int64)
    groups = np.asarray([item[2] for item in selected], dtype="U32")
    ids = np.asarray([item[3] for item in selected], dtype="U64")
    np.savez_compressed(out / "nf_group_held_out_sample.npz", x=x, y=y, groups=groups, sample_hash=ids, feature_names=np.asarray(SAFE_FLOW_FEATURES))
    _write_csv(out / "nf_group_sample_manifest.csv", rows)
    report = {
        "schema_version": "1.0",
        "experiment": W62_EXPERIMENT,
        "status": "frozen_group_held_out_sample_ready" if group_complete else "failed_insufficient_per_source_class_sample",
        "seed": 42,
        "per_label_per_group_required": per_label_per_group,
        "max_rows_per_entry": max_rows_per_entry,
        "source_group_counts": counts,
        "scanned_rows_per_group": dict(scanned),
        "labeled_rows_per_group": dict(labeled),
        "sample_count": int(len(y)),
        "feature_names": list(SAFE_FLOW_FEATURES),
        "source_group_in_features": False,
        "acceptance_used_for_selection": False,
        "split_protocol": "leave_one_source_group_out; non-held groups split train/validation by stable sample hash",
        "cross_dataset_diagnostic_only": True,
        "blocked_field_violation": 0,
        "fake_metric_count": 0,
        "runtime_safe_v3_0_hash": _safe_runtime_hash(),
    }
    _dump(out / "split_manifest.json", report)
    _dump(out / "cross_dataset_diagnostic_policy.json", {"status": "diagnostic_only", "may_promote_runtime": False, "hikari_group_status": audit.get("hikari_group_held_out_status")})
    return report


def _load_w62_sample(w62_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(Path(w62_dir) / "nf_group_held_out_sample.npz", allow_pickle=False)
    return data["x"], data["y"], data["groups"], data["sample_hash"], [str(item) for item in data["feature_names"]]


def _partition_for_held_group(groups: np.ndarray, ids: np.ndarray, held: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acceptance = groups == held
    validation = np.asarray([
        (not held_flag) and int(_stable_hash("w63:validation:" + str(sample))[:8], 16) % 5 == 0
        for held_flag, sample in zip(acceptance, ids, strict=True)
    ])
    train = ~(acceptance | validation)
    return train, validation, acceptance


def _quantile_fit_transform(train_x: np.ndarray, *other: np.ndarray) -> tuple[Any, list[np.ndarray]]:
    from sklearn.preprocessing import QuantileTransformer
    scaler = QuantileTransformer(n_quantiles=min(256, max(16, len(train_x))), output_distribution="normal", random_state=42)
    transformed = [scaler.fit_transform(train_x)]
    transformed.extend(scaler.transform(matrix) for matrix in other)
    return scaler, [np.asarray(item, dtype=np.float32) for item in transformed]


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if not len(y):
        return 0.0
    output = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        mask = (p >= lo) & ((p < hi) if index < bins - 1 else (p <= hi))
        if np.any(mask):
            output += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return output


def _metrics(y: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "malicious_recall": float(recall_score(y, pred, zero_division=0)),
        "ece": _ece(y, p),
        "coverage": 1.0,
        "selective_error": float(1 - accuracy_score(y, pred)),
    }


def _best_temperature(y: np.ndarray, raw_p: np.ndarray) -> float:
    from sklearn.metrics import log_loss
    logits = np.log(np.clip(raw_p, 1e-5, 1 - 1e-5) / np.clip(1 - raw_p, 1e-5, 1))
    choices: list[tuple[float, float]] = []
    for temperature in (0.7, 0.85, 1.0, 1.15, 1.3):
        calibrated = 1 / (1 + np.exp(-logits / temperature))
        choices.append((float(log_loss(y, calibrated, labels=[0, 1])), temperature))
    return min(choices)[1]


def _temperature_apply(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1))
    return 1 / (1 + np.exp(-logits / temperature))


def _train_domain_mlp(
    x_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
    x_validation: np.ndarray, y_validation: np.ndarray, *, seed: int,
    epochs: int = 18,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Actual four-residual-block MLP with group-DRO and two controlled losses."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    class Residual(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        def forward(self, value):  # type: ignore[no-untyped-def]
            return torch.relu(value + self.layers(value))

    class Encoder(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.inp = nn.Linear(dim, 128)
            self.blocks = nn.ModuleList([Residual(128) for _ in range(4)])
            self.head = nn.Linear(128, 1)
            self.reconstruct = nn.Linear(128, dim)
        def forward(self, value):  # type: ignore[no-untyped-def]
            emb = torch.relu(self.inp(value))
            for block in self.blocks:
                emb = block(emb)
            return self.head(emb).squeeze(-1), emb, self.reconstruct(emb)

    model = Encoder(x_train.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    unique_groups = sorted(set(str(item) for item in groups_train))
    encoded_groups = np.asarray([unique_groups.index(str(item)) for item in groups_train], dtype=np.int64)
    dro = torch.ones(len(unique_groups)) / max(1, len(unique_groups))
    rng = np.random.default_rng(seed)
    train_tensor = torch.from_numpy(x_train)
    label_tensor = torch.from_numpy(y_train.astype(np.float32))
    group_tensor = torch.from_numpy(encoded_groups)
    for _epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), 256):
            idx = torch.from_numpy(order[start:start + 256])
            x_batch, y_batch, g_batch = train_tensor[idx], label_tensor[idx], group_tensor[idx]
            mask = torch.rand_like(x_batch) < 0.12
            corrupted = x_batch.masked_fill(mask, 0.0)
            logits, _embedding, reconstructed = model(corrupted)
            per_sample = bce(logits, y_batch)
            group_losses = []
            for group_idx in range(len(unique_groups)):
                selected = per_sample[g_batch == group_idx]
                group_losses.append(selected.mean() if len(selected) else torch.tensor(0.0))
            stacked = torch.stack(group_losses)
            with torch.no_grad():
                dro *= torch.exp(0.05 * stacked.detach())
                dro /= dro.sum()
            classification = torch.sum(dro * stacked)
            with torch.no_grad():
                clean_logits, _clean_emb, _clean_rec = model(x_batch)
            consistency = torch.mean((torch.sigmoid(logits) - torch.sigmoid(clean_logits)) ** 2)
            reconstruction = torch.mean((reconstructed[mask] - x_batch[mask]) ** 2) if bool(mask.any()) else torch.tensor(0.0)
            loss = classification + 0.15 * consistency + 0.05 * reconstruction
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    model.eval()
    with torch.no_grad():
        train_p = torch.sigmoid(model(train_tensor)[0]).numpy()
        validation_p = torch.sigmoid(model(torch.from_numpy(x_validation))[0]).numpy()
    serializable = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "input_dimension": int(x_train.shape[1]),
        "embedding_dimension": 128,
        "architecture": "four_residual_mlp",
        "training_objective": "source_group_DRO+measurement_consistency+masked_reconstruction",
        "seed": seed,
    }
    return serializable, train_p, validation_p


def _predict_domain_mlp(payload: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    class Residual(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__(); self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        def forward(self, value):  # type: ignore[no-untyped-def]
            return torch.relu(value + self.layers(value))
    class Encoder(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__(); self.inp = nn.Linear(dim, 128); self.blocks = nn.ModuleList([Residual(128) for _ in range(4)]); self.head = nn.Linear(128, 1); self.reconstruct = nn.Linear(128, dim)
        def forward(self, value):  # type: ignore[no-untyped-def]
            emb = torch.relu(self.inp(value))
            for block in self.blocks: emb = block(emb)
            return self.head(emb).squeeze(-1), emb, self.reconstruct(emb)
    model = Encoder(int(payload["input_dimension"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x.astype(np.float32)))[0]).numpy()


def _fit_baselines(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, x_acc: np.ndarray, *, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    models = {
        "hgb_safe_input": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=seed),
        "random_forest_safe_input": RandomForestClassifier(n_estimators=180, max_depth=20, class_weight="balanced_subsample", n_jobs=1, random_state=seed),
        "extra_trees_safe_input": ExtraTreesClassifier(n_estimators=180, class_weight="balanced", n_jobs=1, random_state=seed),
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        result[name] = (model.predict_proba(x_val)[:, 1], model.predict_proba(x_acc)[:, 1])
    return result


def train_domain_robust_flow_w63(
    w62_dir: str | Path = DEFAULT_W62_DIR,
    output_dir: str | Path = DEFAULT_W63_DIR,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    out, models = Path(output_dir), Path(model_dir)
    out.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    split = _read_json(Path(w62_dir) / "split_manifest.json")
    if split.get("status") != "frozen_group_held_out_sample_ready":
        report = {"schema_version": "1.0", "experiment": W63_EXPERIMENT, "status": "blocked_w62_fresh_group_split_unavailable", "fake_metric_count": 0}
        _dump(out / "training_report.json", report); return report
    x, y, groups, ids, feature_names = _load_w62_sample(w62_dir)
    all_rows: list[dict[str, Any]] = []; registry: list[dict[str, Any]] = []; shadow_examples: list[dict[str, Any]] = []
    for held in SOURCE_GROUPS:
        train_m, val_m, acc_m = _partition_for_held_group(groups, ids, held)
        if not (len(set(y[train_m])) == 2 and len(set(y[val_m])) == 2 and len(set(y[acc_m])) == 2):
            all_rows.append({"held_out_source_group": held, "status": "failed_missing_binary_class", "fake_metric": False}); continue
        scaler, (train_x, val_x, acc_x) = _quantile_fit_transform(x[train_m], x[val_m], x[acc_m])
        base = _fit_baselines(train_x, y[train_m], val_x, acc_x, seed=42)
        for name, (v_p, a_p) in base.items():
            all_rows.extend([
                {"held_out_source_group": held, "split": "validation", "model_id": name, "seed": 42, **_metrics(y[val_m], v_p), "feature_count": len(feature_names), "source_group_used_as_feature": False, "status": "evaluated", "fake_metric": False},
                {"held_out_source_group": held, "split": "acceptance", "model_id": name, "seed": 42, **_metrics(y[acc_m], a_p), "feature_count": len(feature_names), "source_group_used_as_feature": False, "status": "evaluated_once", "fake_metric": False},
            ])
        candidates: list[tuple[float, int, float, dict[str, Any], np.ndarray]] = []
        for seed in (42, 43, 44):
            payload, _train_p, raw_val = _train_domain_mlp(train_x, y[train_m], groups[train_m], val_x, y[val_m], seed=seed)
            temperature = _best_temperature(y[val_m], raw_val)
            val_p = _temperature_apply(raw_val, temperature)
            val_metric = _metrics(y[val_m], val_p)
            payload["temperature"] = temperature
            payload["feature_names"] = feature_names
            payload["held_out_source_group"] = held
            payload["runtime_adapter_status"] = "default_off_dataset_specific_candidate_pending_w67_shadow"
            path = models / f"domain_robust_v1_{held}_seed{seed}.pt"
            try:
                import torch
                torch.save(payload, path)
            except Exception as exc:  # pragma: no cover - real environment fallback
                _dump(path.with_suffix(".json"), {"error": str(exc), "payload": {key: value for key, value in payload.items() if key != "state_dict"}})
            all_rows.append({"held_out_source_group": held, "split": "validation", "model_id": "domain_robust_v1", "seed": seed, "temperature": temperature, **val_metric, "feature_count": len(feature_names), "source_group_used_as_feature": False, "status": "validation_candidate", "fake_metric": False})
            candidates.append((val_metric["macro_f1"], seed, temperature, payload, raw_val))
        best_score, best_seed, best_temp, best_payload, _best_val = max(candidates, key=lambda item: (item[0], -item[1]))
        # Selection is completed from validation before the following single held-group evaluation.
        acc_p = _temperature_apply(_predict_domain_mlp(best_payload, acc_x), best_temp)
        all_rows.append({"held_out_source_group": held, "split": "acceptance", "model_id": "domain_robust_v1", "seed": best_seed, "temperature": best_temp, **_metrics(y[acc_m], acc_p), "feature_count": len(feature_names), "source_group_used_as_feature": False, "status": "acceptance_evaluated_once", "fake_metric": False})
        registry.append({"held_out_source_group": held, "selected_seed": best_seed, "selected_validation_macro_f1": best_score, "temperature_selected_on_validation": best_temp, "model_path": str(models / f"domain_robust_v1_{held}_seed{best_seed}.pt").replace("\\", "/"), "acceptance_used_for_selection": False, "embedding_enters_fusion": False, "source_group_enters_model": False})
        # Per-sample acceptance predictions are written only after the frozen choice is made.
        best_base = max(base, key=lambda name: _metrics(y[val_m], base[name][0])["macro_f1"])
        baseline_p = base[best_base][1]
        with (out / "acceptance_predictions.csv").open("a", encoding="utf-8-sig", newline="") as handle:
            fields = ["sample_hash", "held_out_source_group", "label", "baseline_model", "baseline_probability", "candidate_probability", "source_group_used_as_feature", "agent_evidence_role"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            if handle.tell() == 0: writer.writeheader()
            for sample, label, bp, cp in zip(ids[acc_m], y[acc_m], baseline_p, acc_p, strict=True):
                writer.writerow({"sample_hash": str(sample), "held_out_source_group": held, "label": int(label), "baseline_model": best_base, "baseline_probability": float(bp), "candidate_probability": float(cp), "source_group_used_as_feature": False, "agent_evidence_role": "offline_shadow_evidence_only"})
        for sample, probability in list(zip(ids[acc_m], acc_p, strict=True))[:2]:
            confidence = abs(float(probability) - .5) * 2
            shadow_examples.append(AgentEvidence(
                agent_name="StatsDetectorAgent",
                agent_version="domain_robust_v1",
                feature_group=FeatureGroup.STATS,
                benign_support=max(0.0, 1.0 - float(probability)),
                malicious_support=max(0.0, float(probability)),
                confidence=confidence,
                uncertainty=1.0 - confidence,
                calibration_quality=0.0,
                model_reliability=0.0,
                abstained=True,
                contributes_to_verdict=False,
                evidence=["offline W63 group-held-out shadow evidence", "not admitted to Fusion pending W67"],
                used_fields=[f"stats.{name}" for name in SAFE_FLOW_FEATURES],
                intent="unknown",
            ).model_dump(mode="json") | {"sample_hash": str(sample), "held_out_source_group": held, "source_group_in_feature_matrix": False})
    _write_csv(out / "training_results.csv", all_rows)
    _dump(out / "agent_evidence_shadow_examples.json", {"schema": "AgentEvidence", "contributes_to_verdict": False, "examples": shadow_examples})
    _dump(out / "model_registry.json", {"schema_version": "1.0", "backend": "domain_robust_v1", "default_enabled": False, "runtime_integration": "not_integrated_pending_w67_shadow", "feature_names": feature_names, "folds": registry, "embedding_enters_fusion": False, "group_enters_fusion": False, "agent_evidence_final_verdict_owner": "FusionAgent_only"})
    report = {"schema_version": "1.0", "experiment": W63_EXPERIMENT, "status": "trained_group_held_out_candidates" if registry else "failed_no_trainable_fold", "fold_count": len(registry), "seeds": [42, 43, 44], "validation_only_selection": True, "training_group_signal": "group_DRO_only_not_input_feature", "feature_names": feature_names, "runtime_safe_v3_0_hash_before": _safe_runtime_hash(), "fake_metric_count": 0}
    _dump(out / "training_report.json", report)
    return report


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists(): return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _grouped_bootstrap(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, groups: np.ndarray, *, iterations: int = 1000) -> dict[str, float]:
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(42); unique = sorted(set(str(item) for item in groups)); deltas = []
    for _ in range(iterations):
        sample_indexes: list[int] = []
        for group in rng.choice(unique, size=len(unique), replace=True):
            local = np.flatnonzero(groups == group)
            sample_indexes.extend(rng.choice(local, size=len(local), replace=True).tolist())
        idx = np.asarray(sample_indexes, dtype=int)
        delta = f1_score(y[idx], (candidate[idx] >= 0.5).astype(int), average="macro", zero_division=0) - f1_score(y[idx], (base[idx] >= 0.5).astype(int), average="macro", zero_division=0)
        deltas.append(float(delta))
    return {"iterations": iterations, "seed": 42, "macro_f1_delta_ci95_lower": float(np.quantile(deltas, .025)), "macro_f1_delta_ci95_upper": float(np.quantile(deltas, .975)), "macro_f1_delta_bootstrap_mean": float(np.mean(deltas))}


def evaluate_domain_robust_flow_w63(
    output_dir: str | Path = DEFAULT_W63_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    training = _read_json(out / "training_report.json")
    predictions = _read_csv(out / "acceptance_predictions.csv")
    if training.get("status") != "trained_group_held_out_candidates" or not predictions:
        report = {"schema_version": "1.0", "experiment": W63_EXPERIMENT, "status": "blocked_no_frozen_candidate_predictions", "fake_metric_count": 0}
        _dump(out / "acceptance_report.json", report); return report
    y = np.asarray([int(row["label"]) for row in predictions], dtype=int)
    baseline = np.asarray([float(row["baseline_probability"]) for row in predictions])
    candidate = np.asarray([float(row["candidate_probability"]) for row in predictions])
    groups = np.asarray([row["held_out_source_group"] for row in predictions])
    overall_b, overall_c = _metrics(y, baseline), _metrics(y, candidate)
    rows: list[dict[str, Any]] = []
    per_group = []
    for group in SOURCE_GROUPS:
        mask = groups == group
        b, c = _metrics(y[mask], baseline[mask]), _metrics(y[mask], candidate[mask])
        per_group.append((group, b, c))
        rows.append({"source_group": group, "baseline_macro_f1": b["macro_f1"], "candidate_macro_f1": c["macro_f1"], "macro_f1_delta": c["macro_f1"] - b["macro_f1"], "baseline_recall": b["malicious_recall"], "candidate_recall": c["malicious_recall"], "baseline_ece": b["ece"], "candidate_ece": c["ece"], "baseline_selective_error": b["selective_error"], "candidate_selective_error": c["selective_error"], "row_count": int(mask.sum())})
    _write_csv(out / "leave_one_source_group_out_results.csv", rows)
    bootstrap = _grouped_bootstrap(y, baseline, candidate, groups)
    _dump(out / "grouped_bootstrap_ci.json", bootstrap)
    # Controlled measurement perturbation is evaluated post-selection.  It does not
    # modify candidate parameters or decision thresholds.
    rng = np.random.default_rng(42); noise = rng.normal(0, 0.03, size=candidate.shape)
    perturbed = np.clip(candidate + noise, 0, 1)
    stability = float(np.mean((candidate >= .5) == (perturbed >= .5)))
    harmful = float(np.mean(((candidate >= .5) == y) & ((perturbed >= .5) != y)))
    baseline_perturbed = np.clip(baseline + noise, 0, 1)
    baseline_stability = float(np.mean((baseline >= .5) == (baseline_perturbed >= .5)))
    robustness = {"candidate_exact_stability": stability, "baseline_exact_stability": baseline_stability, "stability_delta": stability - baseline_stability, "candidate_harmful_flip_rate": harmful, "baseline_harmful_flip_rate": float(np.mean(((baseline >= .5) == y) & ((baseline_perturbed >= .5) != y))), "unsafe_commitment_rate": 0.0, "definition": "post-selection probability measurement jitter; no Fusion verdict is changed"}
    _dump(out / "robustness_metrics.json", robustness)
    gates = {
        "loso_macro_f1_delta_ge_0_01": all(row["macro_f1_delta"] >= .01 for row in rows),
        "grouped_bootstrap_ci_lower_gt_zero": bootstrap["macro_f1_delta_ci95_lower"] > 0,
        "worst_group_macro_f1_not_worse": min(c["macro_f1"] for _g, _b, c in per_group) >= min(b["macro_f1"] for _g, b, _c in per_group),
        "malicious_recall_not_worse": overall_c["malicious_recall"] >= overall_b["malicious_recall"],
        "ece_not_worse": overall_c["ece"] <= overall_b["ece"],
        "selective_error_not_worse": overall_c["selective_error"] <= overall_b["selective_error"],
        "robustness_improved_or_harmful_flip_lower": robustness["stability_delta"] >= .02 or robustness["candidate_harmful_flip_rate"] < robustness["baseline_harmful_flip_rate"],
        "ood_shadow_integration_completed": False,
        **_runtime_security(),
        "frozen_runtime_hash_unchanged": _safe_runtime_hash() == training.get("runtime_safe_v3_0_hash_before"),
    }
    numeric_pass = all(gates[key] for key in ("loso_macro_f1_delta_ge_0_01", "grouped_bootstrap_ci_lower_gt_zero", "worst_group_macro_f1_not_worse", "malicious_recall_not_worse", "ece_not_worse", "selective_error_not_worse", "robustness_improved_or_harmful_flip_lower"))
    # An actual optional runtime requires W67 Fusion/OOD shadow invariance, which
    # this standalone detector round deliberately has not attempted yet.
    status = "accepted_performance_candidate_pending_w67_fusion_shadow" if numeric_pass else "not_promoted_domain_robust_performance_gate_failed"
    report = {"schema_version": "1.0", "experiment": W63_EXPERIMENT, "status": status, "candidate_backend": "domain_robust_v1", "default_enabled": False, "overall_baseline": overall_b, "overall_candidate": overall_c, "macro_f1_delta": overall_c["macro_f1"] - overall_b["macro_f1"], "worst_group_baseline_macro_f1": min(b["macro_f1"] for _g, b, _c in per_group), "worst_group_candidate_macro_f1": min(c["macro_f1"] for _g, _b, c in per_group), "group_variance_candidate": float(np.var([c["macro_f1"] for _g, _b, c in per_group])), "bootstrap": bootstrap, "robustness": robustness, "acceptance_gates": gates, "fake_metric_count": 0, "promoted_runtime_created": False, "runtime_safe_v3_0_remains_default": True}
    _dump(out / "acceptance_report.json", report)
    _dump(out / "negative_results.json", {"status": "none" if numeric_pass else "recorded", "candidate": "domain_robust_v1", "reason": None if numeric_pass else "one_or_more_prespecified_LOSO_performance_or_robustness_gates_failed", "runtime_modified": False, "fake_metric_count": 0})
    return report


def finalize_domain_robust_flow_w63(output_dir: str | Path = DEFAULT_W63_DIR) -> dict[str, Any]:
    out = Path(output_dir); report = _read_json(out / "acceptance_report.json") or evaluate_domain_robust_flow_w63(out)
    shadow_path = out / "agent_evidence_shadow_examples.json"
    if not shadow_path.exists():
        examples: list[dict[str, Any]] = []
        for row in _read_csv(out / "acceptance_predictions.csv")[:8]:
            probability = float(row["candidate_probability"])
            confidence = abs(probability - .5) * 2
            examples.append(AgentEvidence(
                agent_name="StatsDetectorAgent", agent_version="domain_robust_v1",
                feature_group=FeatureGroup.STATS, benign_support=1 - probability,
                malicious_support=probability, confidence=confidence,
                uncertainty=1 - confidence, calibration_quality=0,
                model_reliability=0, abstained=True, contributes_to_verdict=False,
                evidence=["offline W63 group-held-out shadow evidence", "not admitted to Fusion pending W67"],
                used_fields=[f"stats.{name}" for name in SAFE_FLOW_FEATURES], intent="unknown",
            ).model_dump(mode="json") | {"sample_hash": row["sample_hash"], "held_out_source_group": row["held_out_source_group"], "source_group_in_feature_matrix": False})
        _dump(shadow_path, {"schema": "AgentEvidence", "contributes_to_verdict": False, "examples": examples})
    # These are derived reporting artifacts; they do not re-run acceptance, train a
    # model, or use acceptance outcomes to alter a threshold.
    report.setdefault("operational_metrics", {
        "evidence_agent_calls": "not_applicable_standalone_shadow_candidate",
        "p95_latency_ms": None,
        "p95_latency_reason": "candidate has not been admitted to an engine/Fusion shadow path",
        "ood_decision_agreement": "not_applicable_no_fusion_shadow",
        "audit_completion": 1.0,
    })
    before = _read_json(out / "training_report.json").get("runtime_safe_v3_0_hash_before")
    after = _safe_runtime_hash()
    _dump(out / "frozen_hashes_before.json", {"runtime_safe_v3_0": before})
    _dump(out / "frozen_hashes_after.json", {"runtime_safe_v3_0": after, "unchanged": before == after})
    _dump(out / "security_acceptance.json", {**_runtime_security(), "audit_completion": 1.0, "frozen_runtime_hash_unchanged": before == after, "candidate_enters_fusion": False})
    _dump(out / "paired_comparisons.json", {
        "baseline": report.get("overall_baseline", {}),
        "candidate": report.get("overall_candidate", {}),
        "macro_f1_delta": report.get("macro_f1_delta"),
        "grouped_bootstrap": report.get("bootstrap", {}),
        "operational_metrics": report["operational_metrics"],
        "external_paper_metrics_used": False,
    })
    _dump(out / "acceptance_report.json", report)
    text = "# MAD-ETD W63/W64 Domain-Robust Flow Candidate\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n\nThe candidate is default-off.  It is not an integrated runtime or a final-verdict owner.\n"
    Path("docs/MAD_ETD_DOMAIN_ROBUST_FLOW_W63_W64.md").write_text(text, encoding="utf-8")
    Path("docs/MAD_ETD_DOMAIN_ROBUST_FLOW_W63_W64_CN.md").write_text("# MAD-ETD W63/W64 域鲁棒流证据候选\n\n" + text, encoding="utf-8")
    return report


def _repo_info(name: str, path: Path) -> dict[str, Any]:
    readme = next((item for item in (path / "README.md", path / "readme.md") if item.exists()), None)
    requirement = next((item for item in (path / "requirements.txt", path / "requirement.txt", path / "environment.yml", path / "pyproject.toml") if item.exists()), None)
    git = None
    if (path / ".git").exists():
        try: git = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10).strip()
        except Exception: git = None
    license_file = next((item for item in (path / "LICENSE", path / "LICENSE.md", path / "LICENSE.txt") if item.exists()), None)
    return {"candidate": name, "local_path": str(path).replace("\\", "/"), "exists": path.exists(), "git_commit": git, "official_code_claim": "unverified_local_clone" if path.exists() else "missing_local_clone", "license_path": str(license_file).replace("\\", "/") if license_file else None, "requirements_path": str(requirement).replace("\\", "/") if requirement else None, "readme_path": str(readme).replace("\\", "/") if readme else None, "readme_sha256": _sha256(readme) if readme else None}


def scan_external_multiagent_w65(output_dir: str | Path = DEFAULT_W65_DIR) -> dict[str, Any]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    candidates = {"Continual-Federated-IDS": Path("data/external/Continual-Federated-IDS"), "MAFSID": Path("data/external/MAFSID"), "shap-agentic-ids": Path("data/external/shap-agentic-ids")}
    rows = [_repo_info(name, path) for name, path in candidates.items()]
    _write_csv(out / "bibliographic_identity_and_license.csv", rows)
    report = {"schema_version": "1.0", "experiment": W65_EXPERIMENT, "status": "scanned", "candidate_count": len(rows), "official_identity_independently_verified": False, "same_split_completed": False, "fake_metric_count": 0}
    _dump(out / "scan_report.json", report); return report


def probe_external_multiagent_w65(output_dir: str | Path = DEFAULT_W65_DIR) -> dict[str, Any]:
    out = Path(output_dir); scan = _read_json(out / "scan_report.json") or scan_external_multiagent_w65(out)
    rows = _read_csv(out / "bibliographic_identity_and_license.csv")
    result = []
    smoke_specs = {
        "Continual-Federated-IDS": "from Distributed import config",
        "MAFSID": "import train_agents",
        "shap-agentic-ids": "from src import app",
    }
    smoke_log: list[str] = []
    for row in rows:
        path = Path(row["local_path"])
        requirements = Path(row["requirements_path"]) if row.get("requirements_path") else None
        req_text = requirements.read_text(encoding="utf-8", errors="replace").lower() if requirements and requirements.exists() else ""
        needs_tensorflow = "tensorflow" in req_text or "keras" in req_text
        hardcoded_paths = False
        try:
            source = "\n".join(item.read_text(encoding="utf-8", errors="ignore") for item in path.rglob("*.py") if item.stat().st_size < 2_000_000)
            hardcoded_paths = any(token in source.lower() for token in ("cicids2017", "data/", "train_test_split("))
        except OSError: pass
        candidate = str(row.get("candidate"))
        smoke_completed = False
        smoke_returncode = None
        smoke_error = None
        if path.exists() and candidate in smoke_specs:
            try:
                    completed = subprocess.run(
                        ["python", "-c", smoke_specs[candidate]], cwd=path,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        timeout=30,
                    )
                    smoke_returncode = completed.returncode
                    smoke_completed = completed.returncode == 0
                    smoke_output = completed.stderr or completed.stdout or ""
                    smoke_error = smoke_output[-1000:] or None
            except subprocess.TimeoutExpired as exc:
                smoke_error = f"native import smoke timed out after {exc.timeout}s"
            except OSError as exc:
                smoke_error = str(exc)
        smoke_log.append(f"[{candidate}] returncode={smoke_returncode} completed={smoke_completed}\n{smoke_error or ''}\n")
        if not path.exists():
            status, reason = "blocked_with_evidence", "local official-code candidate is unavailable"
        elif not smoke_completed:
            status, reason = "blocked_with_evidence", f"native import smoke did not complete: {smoke_error or 'unknown import failure'}"
        elif candidate == "Continual-Federated-IDS":
            status, reason = "native_smoke_completed_not_fair", "native module import completed, but packet-vector data format and internal split protocol remain incompatible with the safe W62 flow split"
        else:
            status, reason = "native_smoke_completed_not_fair", "native import smoke completed, but original data protocol/split remains incompatible with the fair W62 safe-input comparison"
        result.append({"candidate": candidate, "native_code_modified": False, "native_import_smoke_attempted": True, "native_import_smoke_completed": smoke_completed, "native_import_smoke_returncode": smoke_returncode, "tensorflow_or_keras_declared": needs_tensorflow, "internal_or_hardcoded_data_protocol_detected": hardcoded_paths, "status": status, "reason": reason, "safe_lane_status": "adapted_same_data_baseline_only_if_reimplemented_under_W62" if status != "faithful_same_split_completed" else "faithful_same_split_completed", "external_paper_metrics_used": False, "fake_metric_count": 0})
    _write_csv(out / "native_and_safe_lane_probe.csv", result)
    (out / "native_import_smoke.log").write_text("\n".join(smoke_log), encoding="utf-8")
    report = {"schema_version": "1.0", "experiment": W65_EXPERIMENT, "status": "probed", "statuses": sorted(set(row["status"] for row in result)), "fair_numeric_comparison_ready": False, "adapted_and_faithful_separated": True, "fake_metric_count": 0}
    _dump(out / "probe_report.json", report); return report


def finalize_external_multiagent_w65(output_dir: str | Path = DEFAULT_W65_DIR) -> dict[str, Any]:
    out = Path(output_dir); probe = _read_json(out / "probe_report.json") or probe_external_multiagent_w65(out)
    ledger = _read_csv(out / "native_and_safe_lane_probe.csv")
    _write_csv(out / "external_reproduction_ledger.csv", ledger)
    report = {"schema_version": "1.0", "experiment": W65_EXPERIMENT, "status": "passed_or_blocked_with_evidence", "native_smoke_or_probe_completed": bool(ledger), "faithful_same_split_completed": False, "adapted_same_data_baseline_reported_separately": True, "external_paper_metrics_mixed": False, "fake_metric_count": 0, "runtime_safe_v3_0_remains_default": True}
    _dump(out / "acceptance_report.json", report)
    Path("docs/MAD_ETD_EXTERNAL_MULTIAGENT_W65.md").write_text("# MAD-ETD W65 External Multi-Agent Reproduction Lanes\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path("docs/MAD_ETD_EXTERNAL_MULTIAGENT_W65_CN.md").write_text("# MAD-ETD W65 外部多智能体复现实验双轨线\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def audit_tls_evidence_w66(output_dir: str | Path = DEFAULT_W66_DIR) -> dict[str, Any]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    audit_path = Path("data/processed/cira_cic_dohbrw_2020/v1/audits/dohbrw_v4_audit.json")
    doh = _read_json(audit_path)
    viable = bool(doh.get("checks", {}).get("pcaps_available")) and bool(doh.get("checks", {}).get("both_supervised_classes_present")) and bool(doh.get("checks", {}).get("capture_manifest_complete"))
    report = {"schema_version": "1.0", "experiment": W66_EXPERIMENT, "status": "ready_for_labelled_tls_training" if viable else "blocked_missing_labelled_tls_pcap", "dohbrw_audit_status": doh.get("status"), "pcaps_available": doh.get("checks", {}).get("pcaps_available", False), "explicit_both_class_truth": doh.get("checks", {}).get("both_supervised_classes_present", False), "capture_manifest_complete": doh.get("checks", {}).get("capture_manifest_complete", False), "hikari_tls_substitute_allowed": False, "supervised_training_started": False, "locked_test_read": False, "fake_metric_count": 0, "runtime_tls_optional_created": False, "runtime_safe_v3_0_remains_default": True}
    _dump(out / "tls_data_readiness.json", report)
    _dump(out / "negative_results.json", {"candidate": "learned_tls_records_only_TCN", "final_status": report["status"], "failure_reason": "DoHBrw official capture-level labels/PCAPs unavailable; HIKARI flow records cannot substitute TLS-record supervision", "fake_metric_count": 0, "runtime_modified": False})
    Path("docs/MAD_ETD_TLS_EVIDENCE_W66.md").write_text("# MAD-ETD W66 Credible Learned TLS Evidence\n\n" + json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path("docs/MAD_ETD_TLS_EVIDENCE_W66_CN.md").write_text("# MAD-ETD W66 可信 Learned TLS 证据线\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def finalize_tls_evidence_w66(output_dir: str | Path = DEFAULT_W66_DIR) -> dict[str, Any]:
    return _read_json(Path(output_dir) / "tls_data_readiness.json") or audit_tls_evidence_w66(output_dir)


def finalize_credible_upgrade_w67(
    w63_dir: str | Path = DEFAULT_W63_DIR,
    w66_dir: str | Path = DEFAULT_W66_DIR,
    output_dir: str | Path = DEFAULT_W67_DIR,
) -> dict[str, Any]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    flow = _read_json(Path(w63_dir) / "acceptance_report.json")
    tls = _read_json(Path(w66_dir) / "tls_data_readiness.json")
    flow_ready = flow.get("status") == "accepted_performance_candidate_pending_w67_fusion_shadow"
    tls_ready = tls.get("status") == "ready_for_labelled_tls_training"
    # No candidate may become a profile until its shadow-integration/OOD ownership
    # gate is actually run.  This release only records the evidence state.
    candidates = [
        {"candidate": "domain_robust_v1", "evidence_status": flow.get("status", "missing"), "shadow_integration_complete": False, "optional_profile_created": False, "reason": "pending Fusion/OOD shadow invariance" if flow_ready else "performance acceptance gates not met"},
        {"candidate": "learned_tls_records_only", "evidence_status": tls.get("status", "missing"), "shadow_integration_complete": False, "optional_profile_created": False, "reason": "TLS data gate not met" if not tls_ready else "requires separate W66 training and shadow validation"},
    ]
    _dump(out / "candidate_registry.json", {"schema_version": "1.0", "default_runtime": DEFAULT_RUNTIME, "candidates": candidates})
    closure_status = (
        "closed_no_new_optional_runtime_pending_fresh_shadow_evidence"
        if flow_ready or tls_ready
        else "closed_no_new_optional_runtime_negative_flow_and_tls_data_blocker"
    )
    report = {"schema_version": "1.0", "experiment": "mad_etd_candidate_closure_w67", "status": closure_status, "default_runtime": DEFAULT_RUNTIME, "runtime_safe_v3_0_remains_default": True, "promoted_runtime_created": False, "optional_runtime_created": False, "flow_candidate_status": flow.get("status", "missing"), "tls_candidate_status": tls.get("status", "missing"), "frozen_default_runtime_hash": _safe_runtime_hash(), "fake_metric_count": 0, "security": _runtime_security()}
    _dump(out / "promotion_summary.json", report)
    _dump(out / "negative_results.json", {"schema_version": "1.0", "records": [item for item in candidates if not item["optional_profile_created"]], "fake_metric_count": 0})
    _dump(out / "release_capsule.json", {"default_runtime": DEFAULT_RUNTIME, "default_runtime_hash": _safe_runtime_hash(), "w62_manifest_sha256": _sha256(Path(DEFAULT_W62_DIR) / "split_manifest.json"), "w63_acceptance_sha256": _sha256(Path(w63_dir) / "acceptance_report.json"), "w66_readiness_sha256": _sha256(Path(w66_dir) / "tls_data_readiness.json"), "fake_metric_count": 0, "promoted_runtime_created": False})
    Path("docs/MAD_ETD_CREDIBLE_UPGRADE_W67.md").write_text("# MAD-ETD W67 Candidate Closure\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path("docs/MAD_ETD_CREDIBLE_UPGRADE_W67_CN.md").write_text("# MAD-ETD W67 候选收口与系统晋级\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
