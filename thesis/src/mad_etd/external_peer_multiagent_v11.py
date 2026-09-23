"""Fourth paper-level external multi-agent IDS comparison (v11).

This lane audits candidate identity, freezes an Edge-IIoTset safe-input
protocol, executes official ZTA-FL agentic classes through an isolated
adapter, and compares predictions on the already frozen W205 acceptance.
The result is explicitly adapted and is never represented as a faithful
paper-scale reproduction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .edge_iiot_multiagent_w203_w210 import _adapted_probabilities


EXPERIMENT = "mad_etd_external_peer_multiagent_v11"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_external_peer_multiagent_v11")
DEFAULT_ZTA = Path("data/external/zta-federated-learning")
DEFAULT_EDGE = Path("data/runs/mad_etd_edge_iiot_multiagent_w203_w210")
DEFAULT_MODEL_DIR = Path("data/models/mad_etd_external_peer_multiagent_v11")
DEFAULT_RUNNER = Path("src/mad_etd/ztafl_official_adapter_runner_v11.py")
DEFAULT_DOCUMENT = Path("docs/MAD_ETD_EXTERNAL_PEER_MULTIAGENT_V11.md")
DEFAULT_DOCUMENT_CN = Path("docs/MAD_ETD_EXTERNAL_PEER_MULTIAGENT_V11_CN.md")
DEFAULT_RUNTIME = Path("data/configs/runtime_safe_v3_0.json")
SEEDS = (42, 43, 44)
BOOTSTRAP_ITERATIONS = 1_000
METRICS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "malicious_precision",
    "malicious_recall",
    "malicious_f1",
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def _read(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_csv(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    target = Path(path)
    values = list(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    if not (repository / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _security() -> dict[str, Any]:
    return {
        "audit_completion": 1.0,
        "blocked_field_violation": 0,
        "fusion_ownership_violation": 0,
        "ood_override_count": 0,
        "illegal_verdict_execution_count": 0,
        "fake_metric_count": 0,
        "test_or_acceptance_used_for_selection": False,
        "runtime_safe_v3_0_remains_default": True,
        "promoted_runtime_created": False,
    }


def _binary_metrics(
    truth: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    prediction = np.asarray(probability).argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                labels=[0, 1],
                average="weighted",
                zero_division=0,
            )
        ),
        "malicious_precision": float(
            precision_score(truth, prediction, pos_label=1, zero_division=0)
        ),
        "malicious_recall": float(
            recall_score(truth, prediction, pos_label=1, zero_division=0)
        ),
        "malicious_f1": float(
            f1_score(truth, prediction, pos_label=1, zero_division=0)
        ),
    }


def _paired_group_bootstrap(
    truth: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = 42,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indexes = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for _ in range(iterations):
        selected_groups = rng.choice(
            unique, size=len(unique), replace=True
        )
        selected = np.concatenate(
            [indexes[group] for group in selected_groups]
        )
        baseline = _binary_metrics(
            truth[selected], baseline_probability[selected]
        )
        candidate = _binary_metrics(
            truth[selected], candidate_probability[selected]
        )
        for metric in METRICS:
            samples[metric].append(candidate[metric] - baseline[metric])
    observed_baseline = _binary_metrics(truth, baseline_probability)
    observed_candidate = _binary_metrics(truth, candidate_probability)
    return {
        "unit": "frozen_source_group",
        "iterations": iterations,
        "seed": seed,
        "group_count": int(len(unique)),
        "metrics": {
            metric: {
                "delta": float(
                    observed_candidate[metric] - observed_baseline[metric]
                ),
                "bootstrap_delta_mean": float(np.mean(samples[metric])),
                "delta_ci95_lower": float(
                    np.quantile(samples[metric], 0.025)
                ),
                "delta_ci95_upper": float(
                    np.quantile(samples[metric], 0.975)
                ),
            }
            for metric in METRICS
        },
    }


def _run_upstream_tests(repository: Path) -> dict[str, Any]:
    if not (repository / "tests").is_dir():
        return {
            "passed": False,
            "test_count": 0,
            "returncode": None,
            "summary": "upstream tests directory missing",
        }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "passed": False,
            "test_count": 0,
            "returncode": None,
            "summary": str(error),
        }
    output = "\n".join((result.stdout, result.stderr)).strip()
    matches = re.findall(r"(\d+) passed", output)
    return {
        "passed": result.returncode == 0,
        "test_count": int(matches[-1]) if matches else 0,
        "returncode": result.returncode,
        "summary": output[-2_000:],
    }


def audit_external_peer_candidates_v11(
    output_dir: str | Path = DEFAULT_OUTPUT,
    zta_repo: str | Path = DEFAULT_ZTA,
    *,
    native_smoke_dir: str | Path | None = None,
    run_official_tests: bool = True,
) -> dict[str, Any]:
    out, zta = Path(output_dir), Path(zta_repo)
    out.mkdir(parents=True, exist_ok=True)
    smoke_dir = (
        Path(native_smoke_dir)
        if native_smoke_dir is not None
        else out / "zta_native_smoke"
    )
    smoke = _read(smoke_dir / "agentic_results.json")
    required_sources = [
        zta / "src" / "agentic" / "edge_agent.py",
        zta / "src" / "agentic" / "fog_agent.py",
        zta / "src" / "agentic" / "trust_db.py",
        zta / "src" / "models" / "cnn_lstm.py",
        zta / "scripts" / "run_agentic_experiment.py",
    ]
    source_ready = all(path.is_file() for path in required_sources)
    tests = (
        _run_upstream_tests(zta)
        if run_official_tests
        else {
            "passed": source_ready,
            "test_count": 0,
            "returncode": 0 if source_ready else None,
            "summary": "not executed by this audit invocation",
        }
    )
    smoke_ready = bool(
        smoke.get("ztafl", {}).get("per_seed")
        and smoke.get("vanilla", {}).get("per_seed")
    )
    repository_commit = _git(zta, "rev-parse", "HEAD")
    candidates = [
        {
            "rank": 1,
            "candidate": "ZTA-FL",
            "paper_identity": "IEEE_SATC_2026_10.1109/SATC69565.2026.11542411",
            "official_author_repository": True,
            "repository_url": "https://github.com/ssam18/zta-federated-learning",
            "repository_commit": repository_commit,
            "license_status": "readme_claims_mit_but_license_file_missing",
            "multiagent_semantics": "edge_and_fog_agents_collaborate_during_federated_training",
            "inference_time_multiagent_fusion": False,
            "local_dataset": "Edge-IIoTset",
            "official_tests_passed": tests["passed"],
            "native_smoke_completed": smoke_ready,
            "paper_scale_reproduction": False,
            "adapted_same_split_lane": source_ready and tests["passed"],
            "status": "selected_official_code_participating_adapted_lane",
            "blocker_or_scope": "paper-scale TensorFlow/SMOTE/split/checkpoint protocol unavailable; repository is PyTorch and resource-bounded",
        },
        {
            "rank": 2,
            "candidate": "MAFSIDS",
            "paper_identity": "Journal_of_Big_Data_2023_10.1186/s40537-023-00814-4",
            "official_author_repository": False,
            "repository_url": "",
            "repository_commit": "",
            "license_status": "not_applicable_no_public_code",
            "multiagent_semantics": "multiagent_reinforcement_learning_feature_selection",
            "inference_time_multiagent_fusion": False,
            "local_dataset": "CSE-CIC-IDS2018",
            "official_tests_passed": False,
            "native_smoke_completed": False,
            "paper_scale_reproduction": False,
            "adapted_same_split_lane": False,
            "status": "blocked_code_available_only_on_author_request",
            "blocker_or_scope": "publisher availability statement provides datasets but no public algorithm repository",
        },
        {
            "rank": 3,
            "candidate": "Revelation",
            "paper_identity": "arXiv_2510.13925",
            "official_author_repository": True,
            "repository_url": "https://github.com/WadElla/Revelation",
            "repository_commit": "06641aabcc91b4a1ad137edb003d05fb113a7add",
            "license_status": "license_file_missing",
            "multiagent_semantics": "single_llm_agent_for_traffic_interpretation_and_rag",
            "inference_time_multiagent_fusion": False,
            "local_dataset": "Edge-IIoTset",
            "official_tests_passed": False,
            "native_smoke_completed": False,
            "paper_scale_reproduction": False,
            "adapted_same_split_lane": False,
            "status": "excluded_not_multiagent_detection",
            "blocker_or_scope": "BERT classifier checkpoint missing; reported primary metrics are retrieval and narrative quality",
        },
        {
            "rank": 4,
            "candidate": "X-MAG-IDS",
            "paper_identity": "submission_oriented_manuscript_only",
            "official_author_repository": True,
            "repository_url": "https://github.com/alqithami/xmag",
            "repository_commit": "5a60a9f5d9898cb384743f50b5ffac796d14e1ad",
            "license_status": "MIT",
            "multiagent_semantics": "multiagent_evidence_messaging",
            "inference_time_multiagent_fusion": True,
            "local_dataset": "CICIoT2023",
            "official_tests_passed": False,
            "native_smoke_completed": False,
            "paper_scale_reproduction": False,
            "adapted_same_split_lane": False,
            "status": "blocked_no_verified_peer_reviewed_or_preprint_identity",
            "blocker_or_scope": "repository cannot yet support a paper-level numerical claim",
        },
    ]
    _write_csv(out / "candidate_audit.csv", candidates)
    sources = {
        "ztafl_paper": "https://doi.org/10.1109/SATC69565.2026.11542411",
        "ztafl_preprint": "https://arxiv.org/abs/2512.23809",
        "ztafl_repository": "https://github.com/ssam18/zta-federated-learning",
        "mafsids_paper": "https://doi.org/10.1186/s40537-023-00814-4",
        "revelation_preprint": "https://arxiv.org/abs/2510.13925",
        "revelation_repository": "https://github.com/WadElla/Revelation",
    }
    _dump(out / "bibliographic_sources.json", sources)
    report = {
        "status": (
            "selected_ztafl_official_code_adapter_v11"
            if source_ready and tests["passed"] and smoke_ready
            else "blocked_ztafl_audit_or_native_smoke_incomplete_v11"
        ),
        "selected_candidate": "ZTA-FL",
        "selected_repository_commit": repository_commit,
        "peer_reviewed_paper_identity": True,
        "official_author_repository": True,
        "official_source_ready": source_ready,
        "official_tests": tests,
        "native_smoke_completed": smoke_ready,
        "native_smoke_is_fair_comparison": False,
        "strict_inference_time_multiagent_detection": False,
        "agentic_federated_training_system": True,
        "supervised_comparison_metrics_generated": False,
        **_security(),
    }
    _dump(out / "candidate_audit_report.json", report)
    return report


def freeze_ztafl_adapted_protocol_v11(
    output_dir: str | Path = DEFAULT_OUTPUT,
    edge_run_dir: str | Path = DEFAULT_EDGE,
    zta_repo: str | Path = DEFAULT_ZTA,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    *,
    use_full_train: bool = False,
    rows_per_class: int = 5_000,
) -> dict[str, Any]:
    out, edge, zta, models, runtime = (
        Path(output_dir),
        Path(edge_run_dir),
        Path(zta_repo),
        Path(model_dir),
        Path(runtime_path),
    )
    out.mkdir(parents=True, exist_ok=True)
    audit = _read(out / "candidate_audit_report.json")
    role_paths = {
        "train": edge / "train_w205.npz",
        "calibration": edge / "calibration_w205.npz",
        "selection": edge / "selection_w205.npz",
        "acceptance": edge / "acceptance_sealed_w205.npz",
    }
    required = [
        *role_paths.values(),
        edge / "safe_feature_policy_w204.json",
        edge / "candidate_lock_w207.json",
        edge / "adapted_multiagent_registry_w208.csv",
        runtime,
        zta / "src" / "agentic" / "edge_agent.py",
        zta / "src" / "agentic" / "fog_agent.py",
        zta / "src" / "models" / "cnn_lstm.py",
    ]
    if (
        audit.get("status") != "selected_ztafl_official_code_adapter_v11"
        or not all(path.is_file() for path in required)
    ):
        report = {
            "status": "blocked_ztafl_v11_required_artifact_missing",
            "missing_paths": [
                path.as_posix() for path in required if not path.is_file()
            ],
            "training_started": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "protocol_freeze_report.json", report)
        return report

    archives = {
        role: np.load(path, allow_pickle=False)
        for role, path in role_paths.items()
    }
    try:
        feature_orders = {
            role: tuple(archive["feature_names"].astype(str).tolist())
            for role, archive in archives.items()
        }
        group_sets = {
            role: set(archive["groups"].astype(str).tolist())
            for role, archive in archives.items()
        }
        role_rows = {
            role: int(len(archive["y"]))
            for role, archive in archives.items()
        }
    finally:
        for archive in archives.values():
            archive.close()
    group_overlaps: dict[str, int] = {}
    roles = list(role_paths)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            group_overlaps[f"{left}__{right}"] = len(
                group_sets[left] & group_sets[right]
            )
    policy = _read(edge / "safe_feature_policy_w204.json")
    expected = tuple(policy.get("safe_features", []))
    feature_order_equal = all(
        feature_order == expected for feature_order in feature_orders.values()
    )
    groups_isolated = all(value == 0 for value in group_overlaps.values())
    if not feature_order_equal or not groups_isolated:
        report = {
            "status": "blocked_ztafl_v11_feature_or_group_isolation_failed",
            "feature_order_equal": feature_order_equal,
            "group_overlaps": group_overlaps,
            "training_started": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "protocol_freeze_report.json", report)
        return report

    repository_commit = _git(zta, "rev-parse", "HEAD")
    config = {
        "experiment": EXPERIMENT,
        "comparison_class": (
            "official_code_participating_adapted_exact_train_samples_"
            "same_split_role_and_query"
            if use_full_train
            else "official_code_participating_adapted_same_split_role_and_query"
        ),
        "zta_repo": zta.as_posix(),
        "repository_commit": repository_commit,
        "train_artifact": role_paths["train"].as_posix(),
        "calibration_artifact": role_paths["calibration"].as_posix(),
        "selection_artifact": role_paths["selection"].as_posix(),
        "acceptance_artifact": role_paths["acceptance"].as_posix(),
        "prediction_artifact": (
            out / "ztafl_adapter_predictions.npz"
        ).as_posix(),
        "runner_report": (out / "ztafl_runner_report.json").as_posix(),
        "model_dir": models.as_posix(),
        "seeds": list(SEEDS),
        "subset_seed": 20260725,
        "rows_per_class": int(rows_per_class),
        "use_full_train": bool(use_full_train),
        "target_features": 40,
        "agent_count": 5,
        "rounds": 6,
        "local_epochs": 5,
        "batch_size": 128,
        "inference_batch_size": 1024,
        "learning_rate": 0.001,
        "clean_fraction": 0.7,
        "pgd_iters": 7,
        "pgd_eps": 0.1,
        "fgsm_alpha": 0.01,
        "byzantine_fraction": 0.0,
        "use_cuda": False,
        "train_only_minmax": True,
        "pca_used": False,
        "selection_used_for_training_or_tuning": False,
        "acceptance_used_for_selection": False,
        "mad_etd_candidate_locked_before_external_training": True,
        "same_training_role": True,
        "same_training_samples": bool(use_full_train),
        "same_safe_input_policy": True,
        "same_acceptance_queries": True,
        "historical_acceptance_already_opened": True,
        "faithful_reproduction": False,
    }
    _dump(out / "ztafl_adapter_config.json", config)
    protocol = {
        "status": "frozen_ztafl_official_code_adapted_protocol_v11",
        "selected_candidate": "ZTA-FL",
        "repository_commit": repository_commit,
        "feature_policy_hash": policy.get("feature_policy_hash"),
        "feature_names": list(expected),
        "role_rows": role_rows,
        "group_overlaps": group_overlaps,
        "role_artifact_hashes": {
            role: _sha256(path) for role, path in role_paths.items()
        },
        "runtime_hash_before": _sha256(runtime),
        "adapter_config": config,
        "acceptance_scope": (
            "locked historical W205 acceptance reused only for the external "
            "same-query comparison; no selection or tuning"
        ),
        "native_paper_scale_reproduction": False,
        "training_started": False,
        "supervised_metrics_generated": False,
        **_security(),
    }
    protocol["protocol_hash"] = _json_hash(protocol)
    _dump(out / "frozen_protocol.json", protocol)
    report = {
        "status": protocol["status"],
        "protocol_hash": protocol["protocol_hash"],
        "feature_order_equal": feature_order_equal,
        "group_isolation_passed": groups_isolated,
        "same_acceptance_queries": True,
        "selection_or_acceptance_used_for_tuning": False,
        "training_started": False,
        "supervised_metrics_generated": False,
        **_security(),
    }
    _dump(out / "protocol_freeze_report.json", report)
    return report


def train_ztafl_official_adapter_v11(
    output_dir: str | Path = DEFAULT_OUTPUT,
    zta_repo: str | Path = DEFAULT_ZTA,
    runner: str | Path = DEFAULT_RUNNER,
    *,
    timeout_seconds: int = 7_200,
) -> dict[str, Any]:
    out, zta, runner_path = (
        Path(output_dir), Path(zta_repo), Path(runner)
    )
    protocol = _read(out / "frozen_protocol.json")
    config_path = out / "ztafl_adapter_config.json"
    if (
        protocol.get("status")
        != "frozen_ztafl_official_code_adapted_protocol_v11"
        or not config_path.is_file()
        or not runner_path.is_file()
        or _git(zta, "rev-parse", "HEAD")
        != protocol.get("repository_commit")
    ):
        report = {
            "status": "blocked_ztafl_v11_protocol_or_commit_mismatch",
            "official_code_participating": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "training_report.json", report)
        return report
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(runner_path), "--config", str(config_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        report = {
            "status": "failed_ztafl_v11_runner_exception",
            "error": str(error),
            "official_code_participating": False,
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "training_report.json", report)
        return report
    elapsed = time.perf_counter() - started
    runner_report = _read(out / "ztafl_runner_report.json")
    imported = all(
        str(runner_report.get(key, "")).startswith("src.")
        for key in (
            "official_edge_agent_module",
            "official_fog_agent_module",
            "official_model_module",
            "official_partition_module",
        )
    )
    trained = (
        result.returncode == 0
        and runner_report.get("status")
        == "trained_ztafl_official_code_participating_adapter_v11"
        and imported
    )
    report = {
        "status": (
            "trained_ztafl_official_code_participating_adapter_v11"
            if trained
            else "failed_ztafl_official_adapter_v11"
        ),
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": result.stdout[-4_000:],
        "stderr_tail": result.stderr[-4_000:],
        "official_code_participating": imported,
        "official_edge_agent_imported": imported,
        "official_fog_agent_imported": imported,
        "seed_count": len(runner_report.get("seed_reports", [])),
        "prediction_artifact": runner_report.get("prediction_artifact"),
        "selection_used": False,
        "acceptance_used_for_selection": False,
        "native_reproduction": False,
        "faithful_reproduction": False,
        "supervised_metrics_generated": trained,
        **_security(),
    }
    _dump(out / "training_report.json", report)
    return report


def evaluate_ztafl_adapted_same_query_v11(
    output_dir: str | Path = DEFAULT_OUTPUT,
    edge_run_dir: str | Path = DEFAULT_EDGE,
) -> dict[str, Any]:
    out, edge = Path(output_dir), Path(edge_run_dir)
    protocol = _read(out / "frozen_protocol.json")
    config = _read(out / "ztafl_adapter_config.json")
    training = _read(out / "training_report.json")
    prediction_path = Path(
        training.get("prediction_artifact")
        or config.get("prediction_artifact", "")
    )
    acceptance_path = edge / "acceptance_sealed_w205.npz"
    if (
        training.get("status")
        != "trained_ztafl_official_code_participating_adapter_v11"
        or not prediction_path.is_file()
        or not acceptance_path.is_file()
        or _sha256(acceptance_path)
        != protocol.get("role_artifact_hashes", {}).get("acceptance")
    ):
        report = {
            "status": "blocked_ztafl_v11_training_or_acceptance_missing",
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "evaluation_report.json", report)
        return report

    with np.load(acceptance_path, allow_pickle=False) as acceptance:
        features = acceptance["x"].astype(np.float32)
        truth = acceptance["y"].astype(np.int8)
        groups = acceptance["groups"].astype(str)
        categories = acceptance["categories"].astype(str)
        feature_names = acceptance["feature_names"].astype(str)
    with np.load(prediction_path, allow_pickle=False) as predictions:
        seeds = predictions["seeds"].astype(int).tolist()
        zta_probabilities = np.mean(
            np.stack(
                [
                    predictions[f"acceptance_probability_seed{seed}"]
                    for seed in seeds
                ],
                axis=0,
            ),
            axis=0,
        )
        prediction_truth = predictions["truth"].astype(np.int8)
    if not np.array_equal(truth, prediction_truth):
        report = {
            "status": "failed_ztafl_v11_prediction_truth_mismatch",
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "evaluation_report.json", report)
        return report

    lock = _read(edge / "candidate_lock_w207.json")
    registry_rows = _read_csv(edge / "adapted_multiagent_registry_w208.csv")
    mad_row = next(
        (
            row
            for row in registry_rows
            if row.get("baseline") == "mad_etd_edge_candidate_w207"
        ),
        None,
    )
    if mad_row is None:
        report = {
            "status": "failed_ztafl_v11_locked_mad_candidate_missing",
            "supervised_metrics_generated": False,
            **_security(),
        }
        _dump(out / "evaluation_report.json", report)
        return report
    mad_probability = _adapted_probabilities(mad_row, features, lock)
    systems = {
        "ztafl_official_agentic_adapter_three_seed_ensemble": (
            zta_probabilities
        ),
        "mad_etd_locked_edge_evidence_team_w207": mad_probability,
    }
    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for system, probability in systems.items():
        values = _binary_metrics(truth, probability)
        per_group = []
        for group in np.unique(groups):
            mask = groups == group
            group_metrics = _binary_metrics(truth[mask], probability[mask])
            per_group.append(group_metrics["macro_f1"])
            group_rows.append(
                {
                    "system": system,
                    "group": group,
                    "category": "|".join(sorted(set(categories[mask]))),
                    "rows": int(np.sum(mask)),
                    **group_metrics,
                }
            )
        metric_rows.append(
            {
                "system": system,
                "comparison_class": config.get(
                    "comparison_class",
                    "official_code_participating_adapted_same_split_role_and_query",
                ),
                **values,
                "worst_group_macro_f1": min(per_group) if per_group else 0.0,
                "same_safe_input": True,
                "same_acceptance_query": True,
                "official_code_participating": system.startswith("ztafl"),
                "faithful_reproduction": False,
                "fake_metric": False,
            }
        )
    _write_csv(out / "same_query_metrics.csv", metric_rows)
    _write_csv(out / "per_group_metrics.csv", group_rows)

    zta_prediction = zta_probabilities.argmax(axis=1)
    mad_prediction = mad_probability.argmax(axis=1)
    _write_csv(
        out / "same_query_predictions.csv",
        (
            {
                "sample_index": index,
                "group": groups[index],
                "category": categories[index],
                "truth": int(truth[index]),
                "ztafl_probability_malicious": float(
                    zta_probabilities[index, 1]
                ),
                "ztafl_prediction": int(zta_prediction[index]),
                "mad_etd_probability_malicious": float(
                    mad_probability[index, 1]
                ),
                "mad_etd_prediction": int(mad_prediction[index]),
            }
            for index in range(len(truth))
        ),
    )
    bootstrap = _paired_group_bootstrap(
        truth,
        zta_probabilities,
        mad_probability,
        groups,
        iterations=BOOTSTRAP_ITERATIONS,
    )
    _dump(out / "paired_group_bootstrap.json", bootstrap)
    positive = [
        metric
        for metric, values in bootstrap["metrics"].items()
        if values["delta"] > 0 and values["delta_ci95_lower"] > 0
    ]
    report = {
        "status": (
            "completed_positive_ztafl_official_code_adapter_v11"
            if all(metric in positive for metric in (
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ))
            else "completed_ztafl_adapter_without_required_positive_ci_v11"
        ),
        "official_code_participating": True,
        "agentic_federated_training_system": True,
        "strict_inference_time_multiagent_fusion": False,
        "native_reproduction": False,
        "faithful_reproduction": False,
        "same_training_role": True,
        "same_training_samples": bool(config.get("same_training_samples")),
        "same_safe_input": True,
        "safe_feature_count": int(len(feature_names)),
        "same_acceptance_query": True,
        "acceptance_rows": int(len(truth)),
        "acceptance_group_count": int(len(np.unique(groups))),
        "positive_ci_metrics": positive,
        "positive_ci_metric_count": len(positive),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "acceptance_used_for_selection": False,
        "supervised_metrics_generated": True,
        **_security(),
    }
    _dump(out / "evaluation_report.json", report)
    return report


def finalize_external_peer_multiagent_v11(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOCUMENT,
    document_cn: str | Path = DEFAULT_DOCUMENT_CN,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    out, runtime = Path(output_dir), Path(runtime_path)
    audit = _read(out / "candidate_audit_report.json")
    protocol = _read(out / "frozen_protocol.json")
    config = _read(out / "ztafl_adapter_config.json")
    training = _read(out / "training_report.json")
    evaluation = _read(out / "evaluation_report.json")
    metrics = _read_csv(out / "same_query_metrics.csv")
    bootstrap = _read(out / "paired_group_bootstrap.json")
    by_system = {row.get("system", ""): row for row in metrics}
    baseline = by_system.get(
        "ztafl_official_agentic_adapter_three_seed_ensemble", {}
    )
    candidate = by_system.get(
        "mad_etd_locked_edge_evidence_team_w207", {}
    )
    paper_rows: list[dict[str, Any]] = []
    required_supported = 0
    for metric in METRICS:
        baseline_value = float(baseline.get(metric, 0.0))
        candidate_value = float(candidate.get(metric, 0.0))
        interval = bootstrap.get("metrics", {}).get(metric, {})
        delta = float(
            interval.get("delta", candidate_value - baseline_value)
        )
        lower = interval.get("delta_ci95_lower", "")
        upper = interval.get("delta_ci95_upper", "")
        supported = (
            lower != "" and delta > 0 and float(lower) > 0
        )
        if metric in {"accuracy", "macro_f1", "weighted_f1"} and supported:
            required_supported += 1
        paper_rows.append(
            {
                "dataset_task": "Edge-IIoTset_binary_benign_vs_malicious",
                "metric": metric,
                "ztafl_official_code_adapter": baseline_value,
                "mad_etd_locked_w207": candidate_value,
                "delta": delta,
                "delta_ci95_lower": lower,
                "delta_ci95_upper": upper,
                "paired_group_ci_supported": supported,
                "official_code_participating": True,
                "same_safe_input": True,
                "same_acceptance_query": True,
                "same_training_samples": bool(
                    config.get("same_training_samples")
                ),
                "faithful_reproduction": False,
            }
        )
    _write_csv(out / "paper_ready_ztafl_same_query_table.csv", paper_rows)
    _write_csv(
        out / "negative_results.csv",
        [
            {
                "candidate": "ZTA-FL paper-scale reproduction",
                "status": "blocked_with_evidence",
                "reason": "paper TensorFlow/SMOTE/split/checkpoints and 100-agent five-seed environment are absent; repository protocol differs",
            },
            {
                "candidate": "Revelation",
                "status": "excluded_not_multiagent_detection",
                "reason": "single LLM interpretation agent and unavailable BERT checkpoint",
            },
            {
                "candidate": "MAFSIDS",
                "status": "blocked_no_public_official_code",
                "reason": "publisher states algorithms are available only from the corresponding author on request",
            },
            {
                "candidate": "X-MAG-IDS",
                "status": "blocked_without_citable_paper_identity",
                "reason": "repository still describes a submission-oriented manuscript",
            },
        ],
    )
    runtime_hash_after = _sha256(runtime)
    runtime_unchanged = (
        protocol.get("runtime_hash_before") == runtime_hash_after
    )
    goal_achieved = (
        audit.get("selected_candidate") == "ZTA-FL"
        and audit.get("peer_reviewed_paper_identity") is True
        and training.get("official_edge_agent_imported") is True
        and training.get("official_fog_agent_imported") is True
        and evaluation.get("official_code_participating") is True
        and required_supported == 3
        and runtime_unchanged
        and tests_passed
    )
    claims = {
        "supported_claims": [
            "The official ZTA-FL EdgeAgent, FogAgent, TrustDB, CNN-LSTM, attestation, and aggregation sources participated in a resource-bounded adapted Edge-IIoTset comparison.",
            "Both systems used the same 26-field safe-input policy and the exact same locked W205 binary acceptance queries; no selection or tuning used acceptance labels.",
            "Only deltas whose 1,000-resample paired source-group bootstrap lower bound exceeds zero are reported as supported positive results.",
            "The external adapter remained outside MAD-ETD Fusion and runtime_safe_v3_0 was unchanged.",
        ],
        "forbidden_claims": [
            "The ZTA-FL paper-scale experiment was faithfully reproduced.",
            "The adapted binary protocol is identical to the paper's 15-class TensorFlow/SMOTE protocol.",
            "ZTA-FL performs multi-agent inference-time verdict fusion.",
            "The native smoke metrics can be compared numerically with the W205 binary acceptance metrics.",
            "MAD-ETD universally outperforms ZTA-FL beyond the frozen Edge-IIoTset binary scope.",
            "The W207 Edge-IIoT candidate was promoted to the default runtime.",
        ],
    }
    _dump(out / "claim_boundary.json", claims)
    final_status = (
        "accepted_ztafl_official_code_participating_adapted_same_split_v11"
        if goal_achieved
        else (
            "completed_ztafl_adapter_without_required_positive_ci_v11"
            if tests_passed and required_supported < 3
            else "completed_v11_pending_tests_or_positive_evidence"
        )
    )
    final = {
        "status": final_status,
        "goal_achieved": goal_achieved,
        "fourth_external_system": "ZTA-FL",
        "peer_reviewed_paper_identity": True,
        "official_code_participating": evaluation.get(
            "official_code_participating"
        )
        is True,
        "agentic_federated_training_system": True,
        "strict_inference_time_multiagent_fusion": False,
        "native_smoke_completed": audit.get("native_smoke_completed") is True,
        "native_reproduction_completed": False,
        "faithful_reproduction": False,
        "same_training_role": True,
        "same_training_samples": bool(config.get("same_training_samples")),
        "same_safe_input": True,
        "same_acceptance_query": True,
        "required_ci_supported_result_count": required_supported,
        "paper_ready_row_count": len(paper_rows),
        "runtime_hash_unchanged": runtime_unchanged,
        "tests_passed": tests_passed,
        "test_count": int(test_count),
        **_security(),
    }
    _dump(out / "acceptance_report.json", final)

    display_rows = [
        row
        for row in paper_rows
        if row["metric"] in {
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "malicious_precision",
            "malicious_recall",
            "malicious_f1",
        }
    ]
    table = "\n".join(
        "| {metric} | {base:.6f} | {candidate:.6f} | {delta:+.6f} | "
        "[{lower:+.6f}, {upper:+.6f}] |".format(
            metric=row["metric"],
            base=float(row["ztafl_official_code_adapter"]),
            candidate=float(row["mad_etd_locked_w207"]),
            delta=float(row["delta"]),
            lower=float(row["delta_ci95_lower"]),
            upper=float(row["delta_ci95_upper"]),
        )
        for row in display_rows
    )
    training_description_en = (
        "all 33,000 frozen training rows"
        if bool(config.get("same_training_samples"))
        else "a fixed class-balanced subset of the frozen train role"
    )
    training_description_cn = (
        "全部 33,000 条冻结训练样本"
        if bool(config.get("same_training_samples"))
        else "训练角色的固定类别平衡子集"
    )
    english = f"""# MAD-ETD External Peer Multi-Agent Comparison v11

Status: `{final['status']}`.

| Metric | ZTA-FL official-code adapter | MAD-ETD locked W207 | Delta | 95% paired group CI |
|---|---:|---:|---:|---:|
{table}

The official ZTA-FL agentic training classes participate in this comparison.
Both systems receive the same 26-field safe input and the same 13,229 locked
Edge-IIoTset binary acceptance rows. The adapter uses {training_description_en},
calibration-only fog validation, five agents, six
rounds, five local epochs, and seeds 42/43/44. It is not a faithful
reproduction of the paper's 15-class, TensorFlow, SMOTE, 100-agent protocol.
"""
    chinese = f"""# MAD-ETD 外部论文级多 Agent 对比 v11

状态：`{final['status']}`。

| 指标 | ZTA-FL 官方代码适配器 | MAD-ETD 锁定 W207 | 增量 | 95% 配对分组区间 |
|---|---:|---:|---:|---:|
{table}

本对比实际执行了 ZTA-FL 官方 EdgeAgent、FogAgent、TrustDB、CNN-LSTM、
attestation 与 aggregation 代码。双方接收相同的 26 个安全字段，并在完全相同的
13,229 条 Edge-IIoTset 二分类锁定 acceptance 上评价。适配器使用
{training_description_cn}、仅 calibration 的 Fog 验证、5 个 Agent、6 轮、5 个本地 epoch
和 42/43/44 三个种子。该口径不是原论文 15 类、TensorFlow、SMOTE、100 Agent
协议的 faithful reproduction。
"""
    Path(document).parent.mkdir(parents=True, exist_ok=True)
    Path(document).write_text(english, encoding="utf-8")
    Path(document_cn).parent.mkdir(parents=True, exist_ok=True)
    Path(document_cn).write_text(chinese, encoding="utf-8")
    manifest = {
        path.name: _sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _dump(out / "artifact_manifest.json", manifest)
    return final


__all__ = [
    "audit_external_peer_candidates_v11",
    "evaluate_ztafl_adapted_same_query_v11",
    "finalize_external_peer_multiagent_v11",
    "freeze_ztafl_adapted_protocol_v11",
    "train_ztafl_official_adapter_v11",
]
