from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import joblib
import numpy as np

from .deep_models import DeepDetectorModel
from .detectors import TLSProtocolAgent
from .engine import build_default_engine
from .io import iter_flow_records
from .paper_evaluation import hash_artifact_paths
from .runtime_profiles import load_runtime_profile, profile_engine_kwargs
from .schemas import DetectorInput, FlowRecord
from .tls_v4_training import HGB_FEATURE_NAMES, hgb_tls_features
from .v2_training import _ece
from .v28_protocol import _audit_safety


TLS_V4_MODELS = ("rule", "hgb", "records_only", "records_handshake")
TLS_V4_PERTURBATIONS = (
    "record_padding",
    "dummy_record",
    "sequence_truncation",
)
TLS_V4_BOOTSTRAP_ITERATIONS = 1000
TLS_V4_SEED = 42
BINARY = {"benign", "malicious"}


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_input(record: FlowRecord) -> DetectorInput:
    return DetectorInput(
        tls={
            key: value
            for key, value in record.tls.items()
            if key
            in {
                "record_lengths",
                "tls_record_lengths",
                "server_version",
                "client_cipher_count",
                "client_extension_count",
                "server_extension_count",
                "alpn",
            }
        }
    )


class TLSV4HGBModel:
    def __init__(self, model_dir: str | Path) -> None:
        root = Path(model_dir)
        self.metadata = json.loads(
            (root / "metadata.json").read_text(encoding="utf-8")
        )
        if self.metadata.get("backend") != "hgb_tls_v4":
            raise ValueError("invalid TLS v4 HGB artifact")
        if tuple(self.metadata.get("feature_names", [])) != HGB_FEATURE_NAMES:
            raise ValueError("TLS v4 HGB feature schema mismatch")
        if _sha256(root / "model.joblib") != self.metadata["model_sha256"]:
            raise ValueError("TLS v4 HGB checksum mismatch")
        payload = joblib.load(root / "model.joblib")
        self.estimator = payload["estimator"]
        self.calibrator = payload["calibrator"]

    def predict(self, detector_input: DetectorInput) -> dict[str, Any]:
        features = hgb_tls_features(detector_input).reshape(1, -1)
        raw = self.estimator.predict_proba(features)[:, 1]
        probability = float(self.calibrator.predict(raw)[0])
        policy = self.metadata["decision_policy"]
        accepted = (
            "benign"
            if probability <= float(policy["benign_max_probability"])
            else "malicious"
            if probability >= float(policy["malicious_min_probability"])
            else None
        )
        return {
            "probability": probability,
            "accepted_class": accepted,
            "uncertainty": 1 - abs(probability - 0.5) * 2,
        }


def _predictors(model_dir: Path) -> dict[str, Callable[[FlowRecord], dict[str, Any]]]:
    rule = TLSProtocolAgent(backend="rule", contract_native_fields=True)
    hgb = TLSV4HGBModel(model_dir / "hgb")
    records = DeepDetectorModel(
        model_dir / "records_only",
        expected_agent="tls",
    )
    handshake = DeepDetectorModel(
        model_dir / "records_handshake",
        expected_agent="tls",
    )

    def rule_predict(record: FlowRecord) -> dict[str, Any]:
        evidence = rule.analyze(_safe_input(record))
        total = evidence.benign_support + evidence.malicious_support
        probability = (
            evidence.malicious_support / total if total > 0 else 0.5
        )
        accepted = (
            None
            if evidence.abstained or total == 0
            else "malicious"
            if evidence.malicious_support > evidence.benign_support
            else "benign"
        )
        return {
            "probability": probability,
            "accepted_class": accepted,
            "uncertainty": evidence.uncertainty,
        }

    def deep_predict(model: DeepDetectorModel, record: FlowRecord) -> dict[str, Any]:
        prediction = model.predict(_safe_input(record))
        return {
            "probability": prediction.malicious_probability,
            "accepted_class": prediction.accepted_class,
            "uncertainty": prediction.uncertainty,
        }

    return {
        "rule": rule_predict,
        "hgb": lambda record: hgb.predict(_safe_input(record)),
        "records_only": lambda record: deep_predict(records, record),
        "records_handshake": lambda record: deep_predict(handshake, record),
    }


def _load_test_records(dataset_dir: Path, manifest: dict[str, Any]) -> list[FlowRecord]:
    selected = set(manifest["assignments"]["test"])
    return [
        record
        for record in iter_flow_records(dataset_dir / "flows")
        if record.sample_id in selected
        and record.labels.get("binary") in BINARY
    ]


PREDICTION_FIELDS = [
    "model",
    "sample_id",
    "capture_id",
    "truth",
    "generator",
    "resolver",
    "probability",
    "prediction",
    "covered",
    "correct",
    "uncertainty",
]


def _completed(path: Path, keys: tuple[str, ...]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            tuple(row[key] for key in keys)
            for row in csv.DictReader(handle)
        }


def _open_writer(path: Path, fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open(
        "a" if exists else "w",
        encoding="utf-8-sig",
        newline="",
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not exists:
        writer.writeheader()
    return handle, writer


def _run_predictions(
    records: list[FlowRecord],
    predictors: dict[str, Callable[[FlowRecord], dict[str, Any]]],
    path: Path,
) -> None:
    completed = _completed(path, ("model", "sample_id"))
    handle, writer = _open_writer(path, PREDICTION_FIELDS)
    try:
        for model, predict in predictors.items():
            for record in records:
                key = (model, record.sample_id)
                if key in completed:
                    continue
                result = predict(record)
                prediction = result["accepted_class"] or "unknown"
                truth = str(record.labels["binary"])
                writer.writerow(
                    {
                        "model": model,
                        "sample_id": record.sample_id,
                        "capture_id": record.provenance["capture_id"],
                        "truth": truth,
                        "generator": record.labels.get("generator") or "",
                        "resolver": record.labels.get("resolver") or "",
                        "probability": result["probability"],
                        "prediction": prediction,
                        "covered": int(prediction in BINARY),
                        "correct": int(prediction == truth),
                        "uncertainty": result["uncertainty"],
                    }
                )
                handle.flush()
    finally:
        handle.close()


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    covered = [row for row in rows if row["prediction"] in BINARY]
    scores = []
    for label in ("benign", "malicious"):
        tp = sum(
            row["truth"] == label and row["prediction"] == label
            for row in covered
        )
        fp = sum(
            row["truth"] != label and row["prediction"] == label
            for row in covered
        )
        fn = sum(
            row["truth"] == label and row["prediction"] != label
            for row in covered
        )
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    probabilities = np.asarray(
        [float(row["probability"]) for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(row["truth"] == "malicious") for row in rows],
        dtype=np.int8,
    )
    tool_recall = {}
    for tool in ("dns2tcp", "dnscat2", "iodine"):
        local = [row for row in rows if row["generator"] == tool]
        tool_recall[tool] = (
            sum(row["prediction"] == "malicious" for row in local)
            / len(local)
            if local
            else None
        )
    quad9 = [row for row in rows if row["resolver"] == "quad9"]
    return {
        "sample_count": len(rows),
        "coverage": len(covered) / max(1, len(rows)),
        "selective_macro_f1": float(fmean(scores)),
        "selective_error": (
            sum(row["prediction"] != row["truth"] for row in covered)
            / max(1, len(covered))
        ),
        "ece": _ece(probabilities, labels),
        "per_malicious_tool_recall": tool_recall,
        "quad9": (
            _metrics_without_nested(quad9)
            if quad9
            else {"sample_count": 0, "coverage": 0.0, "selective_macro_f1": 0.0}
        ),
    }


def _metrics_without_nested(rows: list[dict[str, str]]) -> dict[str, Any]:
    covered = [row for row in rows if row["prediction"] in BINARY]
    scores = []
    for label in ("benign", "malicious"):
        tp = sum(
            row["truth"] == label and row["prediction"] == label
            for row in covered
        )
        fp = sum(
            row["truth"] != label and row["prediction"] == label
            for row in covered
        )
        fn = sum(
            row["truth"] == label and row["prediction"] != label
            for row in covered
        )
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {
        "sample_count": len(rows),
        "coverage": len(covered) / max(1, len(rows)),
        "selective_macro_f1": float(fmean(scores)),
    }


def _perturb(record: FlowRecord, kind: str, *, seed: int = 42) -> FlowRecord:
    result = record.model_copy(deep=True)
    records = list(result.tls.get("record_lengths") or [])
    rng = np.random.default_rng(
        int(
            hashlib.sha256(
                f"{seed}:{kind}:{record.sample_id}".encode("utf-8")
            ).hexdigest()[:16],
            16,
        )
    )
    if kind == "record_padding":
        records = [
            (1 if value > 0 else -1)
            * (abs(value) + int(rng.integers(1, 129)))
            for value in records
        ]
    elif kind == "dummy_record":
        additions = max(1, round(len(records) * 0.20))
        for _ in range(additions):
            index = int(rng.integers(0, len(records) + 1))
            sign = 1 if rng.random() >= 0.5 else -1
            records.insert(index, sign * int(rng.integers(16, 257)))
    elif kind == "sequence_truncation":
        keep = max(2, math.floor(len(records) * 0.80))
        records = records[:keep]
    else:
        raise ValueError(f"unsupported TLS v4 perturbation: {kind}")
    result.tls["record_lengths"] = records[:64]
    return result


ROBUSTNESS_FIELDS = [
    "model",
    "sample_id",
    "capture_id",
    "truth",
    "perturbation",
    "clean_prediction",
    "perturbed_prediction",
    "clean_covered",
    "perturbed_covered",
    "perturbed_wrong_binary",
    "harmful_flip",
]


def _run_robustness(
    records: list[FlowRecord],
    predictors: dict[str, Callable[[FlowRecord], dict[str, Any]]],
    path: Path,
) -> None:
    completed = _completed(
        path, ("model", "sample_id", "perturbation")
    )
    handle, writer = _open_writer(path, ROBUSTNESS_FIELDS)
    try:
        for model in ("hgb", "records_only"):
            predict = predictors[model]
            for record in records:
                clean = predict(record)["accepted_class"] or "unknown"
                truth = str(record.labels["binary"])
                for kind in TLS_V4_PERTURBATIONS:
                    key = (model, record.sample_id, kind)
                    if key in completed:
                        continue
                    perturbed = (
                        predict(_perturb(record, kind))["accepted_class"]
                        or "unknown"
                    )
                    writer.writerow(
                        {
                            "model": model,
                            "sample_id": record.sample_id,
                            "capture_id": record.provenance["capture_id"],
                            "truth": truth,
                            "perturbation": kind,
                            "clean_prediction": clean,
                            "perturbed_prediction": perturbed,
                            "clean_covered": int(clean in BINARY),
                            "perturbed_covered": int(perturbed in BINARY),
                            "perturbed_wrong_binary": int(
                                perturbed in BINARY and perturbed != truth
                            ),
                            "harmful_flip": int(
                                clean == truth
                                and perturbed in BINARY
                                and perturbed != truth
                            ),
                        }
                    )
                    handle.flush()
    finally:
        handle.close()


def _robustness_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "perturbed_coverage": sum(int(row["perturbed_covered"]) for row in rows)
        / max(1, len(rows)),
        "perturbed_wrong_binary_rate": sum(
            int(row["perturbed_wrong_binary"]) for row in rows
        )
        / max(1, len(rows)),
        "harmful_flip_rate": sum(int(row["harmful_flip"]) for row in rows)
        / max(1, len(rows)),
    }


def _grouped_bootstrap(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_model[row["model"]][row["capture_id"]].append(row)
    captures = sorted(by_model["hgb"])
    if set(captures) != set(by_model["records_only"]):
        raise ValueError("TLS v4 paired capture groups differ")
    rng = np.random.default_rng(TLS_V4_SEED)
    deltas = []
    for _ in range(TLS_V4_BOOTSTRAP_ITERATIONS):
        selected = rng.choice(captures, size=len(captures), replace=True)
        hgb_rows = [row for capture in selected for row in by_model["hgb"][capture]]
        tcn_rows = [
            row for capture in selected for row in by_model["records_only"][capture]
        ]
        deltas.append(
            _metrics_without_nested(tcn_rows)["selective_macro_f1"]
            - _metrics_without_nested(hgb_rows)["selective_macro_f1"]
        )
    return {
        "method": "capture_id_grouped_bootstrap",
        "iterations": TLS_V4_BOOTSTRAP_ITERATIONS,
        "seed": TLS_V4_SEED,
        "group_count": len(captures),
        "macro_f1_delta_mean": float(np.mean(deltas)),
        "macro_f1_delta_ci95": [
            float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
        ],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _system_rows(
    records: list[FlowRecord],
    model_dir: Path,
    path: Path,
) -> dict[str, Any]:
    baseline = build_default_engine(
        field_audit_mode="strict",
        field_contract_version="2.9",
        routing_policy="capability_v3_0",
        enrichment_policy="off",
        force_rule_coordinator=True,
        detector_backend="learned",
        model_dir="data/models/ustc_tfc2016/v1",
        ood_policy="hybrid",
        ood_gate_dir="data/models/ustc_tfc2016/ood_v1",
        tls_backend="rule",
        max_workers=1,
    )
    candidate = build_default_engine(
        field_audit_mode="strict",
        field_contract_version="2.9",
        routing_policy="capability_v3_0",
        enrichment_policy="off",
        force_rule_coordinator=True,
        detector_backend="learned",
        model_dir="data/models/ustc_tfc2016/v1",
        ood_policy="hybrid",
        ood_gate_dir="data/models/ustc_tfc2016/ood_v1",
        tls_backend="deep_tls_v4",
        tls_model_dir=model_dir / "records_only",
        max_workers=1,
    )
    fields = [
        "mode",
        "sample_id",
        "truth",
        "prediction",
        "covered",
        "correct",
        "audit_complete",
        "blocked_field_violation",
        "fusion_ownership_violation",
        "ood_override",
        "illegal_verdict_execution",
    ]
    completed = _completed(path, ("mode", "sample_id"))
    handle, writer = _open_writer(path, fields)
    try:
        for mode, engine in (("rule_tls", baseline), ("learned_tls_v4", candidate)):
            for record in records:
                if (mode, record.sample_id) in completed:
                    continue
                report, audit = engine.analyze(record)
                safety = _audit_safety(report, audit)
                prediction = report.verdict.value
                truth = str(record.labels["binary"])
                writer.writerow(
                    {
                        "mode": mode,
                        "sample_id": record.sample_id,
                        "truth": truth,
                        "prediction": prediction,
                        "covered": int(prediction in BINARY),
                        "correct": int(prediction == truth),
                        **{
                            key: int(value)
                            for key, value in safety.items()
                            if key
                            in {
                                "audit_complete",
                                "blocked_field_violation",
                                "fusion_ownership_violation",
                                "ood_override",
                                "illegal_verdict_execution",
                            }
                        },
                    }
                )
                handle.flush()
    finally:
        handle.close()
    rows = _read_csv(path)
    safety_rows = rows
    return {
        "metrics": {
            mode: _metrics_without_nested(
                [row for row in rows if row["mode"] == mode]
            )
            for mode in ("rule_tls", "learned_tls_v4")
        },
        "safety": {
            "audit_completion": sum(
                int(row["audit_complete"]) for row in safety_rows
            )
            / max(1, len(safety_rows)),
            "blocked_field_violation_count": sum(
                int(row["blocked_field_violation"]) for row in safety_rows
            ),
            "fusion_ownership_violation_count": sum(
                int(row["fusion_ownership_violation"]) for row in safety_rows
            ),
            "ood_override_count": sum(
                int(row["ood_override"]) for row in safety_rows
            ),
            "illegal_verdict_execution_count": sum(
                int(row["illegal_verdict_execution"]) for row in safety_rows
            ),
        },
        "candidate_engine": candidate,
    }


def _blocked_mutation_invariance(
    records: list[FlowRecord],
    engine,
) -> float:
    agreements = 0
    selected = sorted(records, key=lambda item: item.sample_id)[:200]
    for record in selected:
        baseline, _ = engine.analyze(record)
        mutated = record.model_copy(deep=True)
        mutated.sample_id = "mutated-" + record.sample_id
        mutated.trace_id = "mutated-" + record.trace_id
        mutated.labels["binary"] = (
            "malicious"
            if record.labels["binary"] == "benign"
            else "benign"
        )
        mutated.provenance.update(
            {
                "source_file": "mutated.pcap",
                "capture_id": "mutated",
                "generator": "mutated",
                "resolver": "mutated",
            }
        )
        mutated.tls.update(
            {
                "sni": "mutated.example",
                "ja3": "mutated",
                "known_bad_fingerprint": True,
            }
        )
        changed, _ = engine.analyze(mutated)
        agreements += baseline.verdict == changed.verdict
    return agreements / max(1, len(selected))


def _frozen_artifacts() -> dict[str, Path]:
    paths = {
        "default_runtime": Path("data/configs/runtime_safe_v3_0.json"),
        "default_models": Path("data/models/ustc_tfc2016/v1"),
        "default_ood": Path("data/models/ustc_tfc2016/ood_v1"),
        "fusion": Path("src/mad_etd/fusion.py"),
        "field_audit": Path("src/mad_etd/field_audit.py"),
        "rag": Path("src/mad_etd/knowledge.py"),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TLS v4 frozen artifacts: {missing}")
    return paths


def evaluate_tls_v4(
    dataset_dir: str | Path = "data/processed/cira_cic_dohbrw_2020/v1",
    model_dir: str | Path = "data/models/mad_etd_v4/tls",
    output_dir: str | Path = "data/runs/mad_etd_v4",
) -> dict[str, Any]:
    dataset = Path(dataset_dir)
    models = Path(model_dir)
    output = Path(output_dir)
    manifest_path = dataset / "splits" / "split-manifest.json"
    training_path = models / "training_summary.json"
    if not manifest_path.exists() or not training_path.exists():
        raise FileNotFoundError("TLS v4 split and training artifacts are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_path = output / "locked_test_access.json"
    access_contract = {
        "split_manifest_sha256": _sha256(manifest_path),
        "model_hashes": {
            name: _sha256(models / name / "metadata.json")
            for name in ("hgb", "records_only", "records_handshake")
        },
        "split": "test",
        "selection_or_tuning": False,
    }
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != access_contract:
            raise RuntimeError("locked test access contract changed")
    else:
        _dump(marker_path, access_contract)
    before_path = output / "frozen_hashes_before.json"
    if not before_path.exists():
        _dump(before_path, hash_artifact_paths(_frozen_artifacts()))
    records = _load_test_records(dataset, manifest)
    expected = len(set(manifest["assignments"]["test"]))
    if len(records) != expected:
        raise RuntimeError(f"locked test incomplete: {len(records)}/{expected}")
    predictors = _predictors(models)
    _run_predictions(records, predictors, output / "locked_test_predictions.csv")
    robustness_records = sorted(records, key=lambda item: item.sample_id)[:800]
    _run_robustness(
        robustness_records,
        predictors,
        output / "robustness_predictions.csv",
    )
    prediction_rows = _read_csv(output / "locked_test_predictions.csv")
    robustness_rows = _read_csv(output / "robustness_predictions.csv")
    metrics = {
        model: _metrics(
            [row for row in prediction_rows if row["model"] == model]
        )
        for model in TLS_V4_MODELS
    }
    robustness = {
        model: _robustness_metrics(
            [row for row in robustness_rows if row["model"] == model]
        )
        for model in ("hgb", "records_only")
    }
    bootstrap = _grouped_bootstrap(
        [
            row
            for row in prediction_rows
            if row["model"] in {"hgb", "records_only"}
        ]
    )
    system = _system_rows(
        records,
        models,
        output / "system_predictions.csv",
    )
    mutation_invariance = _blocked_mutation_invariance(
        records,
        system.pop("candidate_engine"),
    )
    after = hash_artifact_paths(_frozen_artifacts())
    _dump(output / "frozen_hashes_after.json", after)
    before = json.loads(before_path.read_text(encoding="utf-8"))
    records_only = metrics["records_only"]
    hgb = metrics["hgb"]
    records_robust = robustness["records_only"]
    hgb_robust = robustness["hgb"]
    system_rule = system["metrics"]["rule_tls"]
    system_candidate = system["metrics"]["learned_tls_v4"]
    tool_recalls = records_only["per_malicious_tool_recall"].values()
    checks = {
        "records_only_coverage_at_least_0_80": records_only["coverage"] >= 0.80,
        "records_only_macro_f1_at_least_0_85": (
            records_only["selective_macro_f1"] >= 0.85
        ),
        "records_only_ece_at_most_0_05": records_only["ece"] <= 0.05,
        "records_only_beats_hgb_by_0_03": (
            records_only["selective_macro_f1"]
            >= hgb["selective_macro_f1"] + 0.03
        ),
        "bootstrap_ci_lower_above_zero": (
            bootstrap["macro_f1_delta_ci95"][0] > 0
        ),
        "quad9_macro_f1_at_least_0_75": (
            records_only["quad9"]["selective_macro_f1"] >= 0.75
        ),
        "each_malicious_tool_recall_at_least_0_70": all(
            value is not None and value >= 0.70 for value in tool_recalls
        ),
        "perturbed_wrong_binary_not_worse_than_hgb": (
            records_robust["perturbed_wrong_binary_rate"]
            <= hgb_robust["perturbed_wrong_binary_rate"]
        ),
        "perturbed_coverage_noninferior_0_02": (
            records_robust["perturbed_coverage"]
            >= hgb_robust["perturbed_coverage"] - 0.02
        ),
        "system_macro_f1_improves_0_02": (
            system_candidate["selective_macro_f1"]
            >= system_rule["selective_macro_f1"] + 0.02
        ),
        "system_coverage_noninferior_0_01": (
            system_candidate["coverage"] >= system_rule["coverage"] - 0.01
        ),
        "blocked_context_mutation_invariance_is_one": mutation_invariance == 1.0,
        "audit_completion_is_one": system["safety"]["audit_completion"] == 1.0,
        "blocked_field_violation_is_zero": (
            system["safety"]["blocked_field_violation_count"] == 0
        ),
        "fusion_ownership_violation_is_zero": (
            system["safety"]["fusion_ownership_violation_count"] == 0
        ),
        "ood_override_is_zero": system["safety"]["ood_override_count"] == 0,
        "illegal_verdict_execution_is_zero": (
            system["safety"]["illegal_verdict_execution_count"] == 0
        ),
        "frozen_artifacts_unchanged": before == after,
        "locked_test_not_used_for_selection": (
            not manifest.get("test_used_for_selection")
        ),
    }
    passed = all(checks.values())
    aggregate = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v4_tls_evaluation",
        "scope": "benign_doh_vs_malicious_doh",
        "models": metrics,
        "robustness": robustness,
        "grouped_bootstrap": bootstrap,
        "system": system,
        "blocked_context_mutation_invariance": mutation_invariance,
    }
    _dump(output / "aggregate_metrics.json", aggregate)
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v4_tls_acceptance",
        "status": "accepted_optional_profile" if passed else "not_promoted",
        "scope": "known_doh_monitoring_only",
        "checks": checks,
        "create_optional_runtime_profile": passed,
        "replace_runtime_safe_v3_0": False,
        "locked_test_used_once": True,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(output / "acceptance_report.json", report)
    return report
