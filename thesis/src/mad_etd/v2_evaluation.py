from __future__ import annotations

import csv
import hashlib
import heapq
import json
from pathlib import Path
from typing import Any, Iterable

from .coordinator import RuleCoordinator
from .detectors import StatsDetectorAgent, TemporalBehaviorAgent
from .engine import DetectionEngine
from .evaluation import evaluate_to_directory
from .field_audit import AuditPolicy, FieldAuditAgent
from .guard import PseudoFeatureGuardAgent
from .fusion import FusionAgent
from .io import iter_flow_records, iter_split_records
from .schemas import FlowRecord
from .utility import UTILITY_FEATURE_NAMES, fit_evidence_utility_policy


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_utility_replay_rows(
    records: Iterable[FlowRecord],
    stats_engine: DetectionEngine,
    candidate_engine: DetectionEngine,
    *,
    candidate_agent: str,
    limit: int | None = None,
) -> list[dict[str, float]]:
    """Build offline counterfactual targets; labels never become runtime features."""

    auditor = FieldAuditAgent(AuditPolicy(mode="legacy"))
    guard = PseudoFeatureGuardAgent()
    rows: list[dict[str, float]] = []
    for record in records:
        base, _ = stats_engine.analyze(record)
        candidate, _ = candidate_engine.analyze(record)
        audit = auditor.audit(record)
        safe = auditor.make_safe_flow(record, audit)
        reliability = guard.assess(safe)
        label = str(record.labels.get("binary", "")).lower()
        base_correct = float(base.verdict.value == label)
        candidate_correct = float(candidate.verdict.value == label)
        observed = (
            (base.uncertainty - candidate.uncertainty)
            + (base.conflict_score - candidate.conflict_score)
            + 0.5 * (candidate_correct - base_correct)
            - 0.05
        )
        row = {
            "remaining_budget": 3.0,
            "round_no": 1.0,
            "evidence_count": float(len(base.agent_results)),
            "called_stats": 1.0,
            "called_temporal": 0.0,
            "called_tls": 0.0,
            "uncertainty": base.uncertainty,
            "conflict": base.conflict_score,
            "distribution_shift": base.distribution_shift_score,
            "stats_reliability": reliability.stats_reliability,
            "sequence_reliability": reliability.sequence_reliability,
            "tls_reliability": reliability.tls_reliability,
            "input_completeness": reliability.input_completeness,
            "sequence_missing": float(not bool(safe.sequence.packet_lengths)),
            "tls_missing": float(not bool(safe.tls)),
            "candidate_temporal": float(
                candidate_agent == "TemporalBehaviorAgent"
            ),
            "candidate_tls": float(candidate_agent == "TLSProtocolAgent"),
            "observed_utility": observed,
        }
        if set(row) != set(UTILITY_FEATURE_NAMES) | {"observed_utility"}:
            raise RuntimeError("utility replay row does not match frozen schema")
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def train_utility_from_replays(
    replay_files: Iterable[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in replay_files:
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    metadata = fit_evidence_utility_policy(rows, output_dir, seed=seed)
    metadata["replay_sources"] = [str(Path(item)) for item in replay_files]
    _dump(Path(output_dir) / "training_summary.json", metadata)
    return metadata


def generate_utility_replay_v2(
    ustc_dataset_dir: str | Path,
    output_path: str | Path,
    *,
    base_model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    v2_model_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_root = Path(ustc_dataset_dir)
    base_root = Path(base_model_dir)
    gate_root = Path(base_ood_gate_dir)
    v2_root = Path(v2_model_dir)
    stats = StatsDetectorAgent(
        backend="learned",
        model_dir=base_root / "stats",
        ood_policy="hybrid",
        ood_gate_dir=gate_root / "stats",
    )
    base_engine = DetectionEngine(
        coordinator=RuleCoordinator(),
        detectors=[stats],
        execution_policy="fixed_all",
        fusion_agent=FusionAgent(),
        max_workers=1,
    )
    candidate_engine = DetectionEngine(
        coordinator=RuleCoordinator(),
        detectors=[
            StatsDetectorAgent(
                backend="learned",
                model_dir=base_root / "stats",
                ood_policy="hybrid",
                ood_gate_dir=gate_root / "stats",
            ),
            TemporalBehaviorAgent(
                backend="deep_v2",
                model_dir=v2_root / "temporal",
                ood_policy="conformal_v3",
                ood_gate_dir=v2_root / "temporal" / "conformal_v3",
            ),
        ],
        execution_policy="fixed_all",
        fusion_agent=FusionAgent(),
        max_workers=1,
    )
    records = iter_split_records(
        dataset_root / "flows",
        dataset_root / "splits" / "split-manifest.json",
        "validation",
    )
    rows = build_utility_replay_rows(
        records,
        base_engine,
        candidate_engine,
        candidate_agent="TemporalBehaviorAgent",
        limit=limit,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0",
        "experiment": "evidence_utility_v2_replay",
        "source": "USTC validation",
        "row_count": len(rows),
        "candidate_agent": "TemporalBehaviorAgent",
        "target_uses_labels": True,
        "runtime_features_use_labels": False,
        "annotated_tls_validation_status": "unavailable_no_explicit_binary_truth",
        "cipherspectrum_locked_test_used": False,
        "output_sha256": _sha256(target),
    }
    _dump(target.with_suffix(".manifest.json"), manifest)
    return manifest


def evaluate_v2_split(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    split_manifest: str | Path | None = None,
    split: str | None = None,
    selection_manifest: str | Path | None = None,
    base_ood_policy: str = "hybrid",
    base_ood_gate_dir: str | Path | None = None,
    ood_policy: str = "conformal_v3",
    ood_gate_dir: str | Path | None = None,
    utility_model_dir: str | Path | None = None,
    utility_policy_version: str = "off",
    limit: int | None = None,
) -> dict[str, Any]:
    if utility_policy_version not in {"off", "v2", "v2_1"}:
        raise ValueError(
            f"unsupported utility policy version: {utility_policy_version}"
        )
    if utility_policy_version != "off" and utility_model_dir is None:
        raise ValueError(
            "utility_model_dir is required when a utility policy is enabled"
        )
    resolved_utility_version = (
        "v2"
        if utility_model_dir is not None and utility_policy_version == "off"
        else utility_policy_version
    )
    if selection_manifest is not None:
        selection = json.loads(
            Path(selection_manifest).read_text(encoding="utf-8")
        )
        selected = set(selection["sample_ids"])
        records = (
            record
            for record in iter_flow_records(input_path)
            if record.sample_id in selected
        )
    else:
        records = (
            iter_split_records(input_path, split_manifest, split)
            if split_manifest is not None and split is not None
            else iter_flow_records(input_path)
        )
    if limit is not None:
        records = _deterministic_limit(records, limit)
    return evaluate_to_directory(
        records,
        input_path=input_path,
        mode="rule_coordinator",
        output_dir=output_dir,
        detector_backend="deep_v2",
        model_dir=model_dir,
        ood_policy=ood_policy,
        ood_gate_dir=ood_gate_dir,
        base_ood_policy=base_ood_policy,
        base_ood_gate_dir=base_ood_gate_dir,
        enable_evidence_utility_v2=resolved_utility_version == "v2",
        enable_evidence_utility_v2_1=(
            resolved_utility_version == "v2_1"
        ),
        utility_model_dir=utility_model_dir,
    )


def aggregate_utility_v2_comparison(
    current_dir: str | Path,
    utility_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    current_root = Path(current_dir)
    utility_root = Path(utility_dir)
    output_root = Path(output_dir)
    current_metrics = _read_optional(current_root / "metrics.json")
    utility_metrics = _read_optional(utility_root / "metrics.json")
    current_audit = _read_optional(current_root / "audit_summary.json")
    utility_audit = _read_optional(utility_root / "audit_summary.json")
    if any(
        value is None
        for value in (
            current_metrics,
            utility_metrics,
            current_audit,
            utility_audit,
        )
    ):
        raise FileNotFoundError(
            "both runs require metrics.json and audit_summary.json"
        )

    def load_predictions(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return {
                row["sample_id"]: row for row in csv.DictReader(handle)
            }

    current_predictions = load_predictions(
        current_root / "predictions.csv"
    )
    utility_predictions = load_predictions(
        utility_root / "predictions.csv"
    )
    if current_predictions.keys() != utility_predictions.keys():
        raise RuntimeError(
            "utility comparison requires identical paired sample IDs"
        )

    paired_path = output_root / "paired_comparisons.csv"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_agreements = 0
    with paired_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "true_label",
            "current_verdict",
            "utility_verdict",
            "verdict_agreement",
            "current_covered",
            "utility_covered",
            "current_agent_calls",
            "utility_agent_calls",
            "agent_call_delta",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(current_predictions):
            current = current_predictions[sample_id]
            utility = utility_predictions[sample_id]
            agreement = current["verdict"] == utility["verdict"]
            verdict_agreements += int(agreement)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "true_label": current["true_label"],
                    "current_verdict": current["verdict"],
                    "utility_verdict": utility["verdict"],
                    "verdict_agreement": agreement,
                    "current_covered": current["covered"],
                    "utility_covered": utility["covered"],
                    "current_agent_calls": current["agent_calls"],
                    "utility_agent_calls": utility["agent_calls"],
                    "agent_call_delta": (
                        int(utility["agent_calls"])
                        - int(current["agent_calls"])
                    ),
                }
            )

    current_calls = float(current_metrics["average_agent_calls"])
    utility_calls = float(utility_metrics["average_agent_calls"])
    call_reduction = (
        (current_calls - utility_calls) / current_calls
        if current_calls > 0
        else 0.0
    )
    coverage_delta = (
        float(utility_metrics["coverage"])
        - float(current_metrics["coverage"])
    )
    macro_f1_delta = (
        float(utility_metrics["macro_f1"])
        - float(current_metrics["macro_f1"])
    )
    checks = {
        "agent_calls_reduction_at_least_10pct": call_reduction >= 0.10,
        "coverage_drop_within_0_005": coverage_delta >= -0.005,
        "selective_macro_f1_drop_within_0_005": (
            macro_f1_delta >= -0.005
        ),
        "paired_sample_ids_identical": True,
        "audit_completion_is_one": (
            current_audit["audit_chain_completion_rate"] == 1.0
            and utility_audit["audit_chain_completion_rate"] == 1.0
        ),
        "blocked_field_violation_is_zero": (
            current_audit["blocked_field_violation_count"] == 0
            and utility_audit["blocked_field_violation_count"] == 0
        ),
        "external_verdict_agreement_at_least_0_98": None,
    }
    promoted = all(value is True for value in checks.values())
    sample_count = len(current_predictions)
    report = {
        "schema_version": "1.0",
        "experiment": "evidence_utility_policy_v2",
        "comparison_scope": "paired_ustc_selection",
        "sample_count": sample_count,
        "current": current_metrics,
        "utility_v2": utility_metrics,
        "deltas": {
            "average_agent_calls": utility_calls - current_calls,
            "agent_calls_reduction_fraction": call_reduction,
            "coverage": coverage_delta,
            "selective_macro_f1": macro_f1_delta,
            "paired_verdict_agreement": (
                verdict_agreements / sample_count if sample_count else None
            ),
        },
        "acceptance_checks": checks,
        "promotion_status": (
            "promoted" if promoted else "not_promoted"
        ),
        "negative_result": not promoted,
        "reason": (
            "Utility v2 reduced calls but failed the frozen coverage "
            "non-inferiority requirement; no post-test threshold tuning was "
            "performed."
            if coverage_delta < -0.005
            else "Full promotion requirements were not all evaluated or met."
        ),
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(output_root / "aggregate_metrics.json", report)
    return report


def _iter_named_assignment(
    input_path: str | Path,
    manifest_path: str | Path,
    split_name: str,
):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = set(manifest["assignments"][split_name])
    for record in iter_flow_records(input_path):
        if record.sample_id in selected:
            yield record


def _deterministic_limit(records, limit: int):
    heap: list[tuple[int, str, FlowRecord]] = []
    for record in records:
        score = int(
            hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest()[:16],
            16,
        )
        item = (-score, record.sample_id, record)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
    for _, _, record in sorted(heap, key=lambda item: item[1]):
        yield record


def run_v2_core_protocol(
    output_dir: str | Path,
    *,
    model_dir: str | Path,
    base_ood_gate_dir: str | Path,
    ustc_dir: str | Path,
    hikari_dir: str | Path,
    annotated_tls_flow_path: str | Path,
    cesnet_dir: str | Path,
    cipherspectrum_dir: str | Path,
    utility_model_dir: str | Path | None = None,
    limit_ustc: int | None = None,
    limit_hikari: int | None = None,
    limit_annotated_tls: int | None = None,
    limit_cesnet: int | None = None,
    limit_cipherspectrum: int | None = None,
) -> dict[str, Any]:
    import itertools

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, Any]] = {}

    def run_split(
        name: str,
        input_path: Path,
        manifest: Path | None,
        split: str | None,
        limit: int | None,
        *,
        named_assignment: bool = False,
    ) -> None:
        target = root / name
        existing = _read_optional(target / "metrics.json")
        if existing is not None:
            runs[name] = existing
            return
        if named_assignment:
            records = _iter_named_assignment(input_path, manifest, split)
        elif manifest is not None and split is not None:
            records = iter_split_records(input_path, manifest, split)
        else:
            records = iter_flow_records(input_path)
        if limit is not None:
            records = _deterministic_limit(records, limit)
        result = evaluate_to_directory(
            records,
            input_path=input_path,
            mode="rule_coordinator",
            output_dir=target,
            detector_backend="deep_v2",
            model_dir=model_dir,
            ood_policy="conformal_v3",
            base_ood_policy="hybrid",
            base_ood_gate_dir=base_ood_gate_dir,
            enable_evidence_utility_v2=utility_model_dir is not None,
            utility_model_dir=utility_model_dir,
        )
        runs[name] = result["metrics"]

    ustc_root = Path(ustc_dir)
    run_split(
        "ustc_time_block",
        ustc_root / "flows",
        ustc_root / "splits" / "split-manifest.json",
        "test",
        limit_ustc,
    )
    hikari_root = Path(hikari_dir)
    run_split(
        "hikari_external",
        hikari_root / "flows",
        hikari_root / "splits" / "split-manifest.json",
        "test",
        limit_hikari,
    )
    run_split(
        "annotated_tls_unlabeled",
        Path(annotated_tls_flow_path),
        None,
        None,
        limit_annotated_tls,
    )
    cesnet_root = Path(cesnet_dir)
    run_split(
        "cesnet_external",
        cesnet_root / "flows",
        cesnet_root / "splits" / "split-manifest.json",
        "test",
        limit_cesnet,
    )
    cipher_root = Path(cipherspectrum_dir)
    run_split(
        "cipherspectrum_validation",
        cipher_root / "flows",
        cipher_root / "splits" / "domain-split-manifest.json",
        "validation",
        limit_cipherspectrum,
        named_assignment=True,
    )
    supervised_external = runs["hikari_external"]
    if runs["cesnet_external"]["metrics_status"] != "skipped_no_labels":
        raise RuntimeError("CESNET supervised metrics must remain disabled")
    if runs["cipherspectrum_validation"]["metrics_status"] != "skipped_no_labels":
        raise RuntimeError("CipherSpectrum supervised metrics must remain disabled")
    summary = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_core_protocol",
        "runs": runs,
        "limits": {
            "ustc": limit_ustc,
            "hikari": limit_hikari,
            "annotated_tls": limit_annotated_tls,
            "cesnet": limit_cesnet,
            "cipherspectrum": limit_cipherspectrum,
        },
        "hikari_supervised_metrics_available": (
            supervised_external["metrics_status"] == "computed"
        ),
        "cesnet_supervised_metrics_available": False,
        "cipherspectrum_supervised_metrics_available": False,
        "cipherspectrum_locked_test_used": False,
        "fusion_modified": False,
        "automatic_retraining": False,
    }
    _dump(root / "core_protocol_summary.json", summary)
    return summary


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_v1_reference(
    reference_path: str | Path,
) -> dict[str, Any]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for key in ("detector_models", "ood_v1", "knowledge_base"):
        group = reference.get(key, {})
        root = Path(group.get("path", ""))
        for relative, expected in group.get("files", {}).items():
            path = root / relative
            actual = _sha256(path) if path.exists() else None
            checks.append(
                {
                    "artifact": f"{key}/{relative}",
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "match": actual == expected,
                }
            )
    direct = {
        "ood_v2_policy": Path(
            reference.get("ood_v2_policy", {}).get("path", "")
        ),
        "fusion_module": Path("src/mad_etd/fusion.py"),
        "rag_module": Path("src/mad_etd/knowledge.py"),
        "reporter_module": Path("src/mad_etd/reporter.py"),
    }
    for key, path in direct.items():
        expected = reference.get(key, {}).get("sha256")
        actual = _sha256(path) if path.exists() else None
        checks.append(
            {
                "artifact": key,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    return {
        "schema_version": "1.0",
        "reference": str(reference_path),
        "all_match": all(item["match"] for item in checks),
        "checks": checks,
    }


def finalize_v2_acceptance(
    output_dir: str | Path,
    *,
    dataset_dir: str | Path,
    model_dir: str | Path,
    run_dir: str | Path,
    tests_passed: bool,
    test_count: int,
    frozen_reference: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    dataset_root = Path(dataset_dir)
    model_root = Path(model_dir)
    run_root = Path(run_dir)
    supervision = _read_optional(
        dataset_root / "audits" / "supervision_decision.json"
    )
    training = _read_optional(model_root / "training_summary.json")
    required_runs = {
        "ustc_time_block": _read_optional(
            run_root / "ustc_time_block" / "metrics.json"
        ),
        "ustc_domain_holdout": _read_optional(
            run_root / "ustc_domain_holdout" / "aggregate_metrics.json"
        ),
        "hikari_external": _read_optional(
            run_root / "hikari_external" / "metrics.json"
        ),
        "cesnet_external": _read_optional(
            run_root / "cesnet_external" / "metrics.json"
        ),
        "cipherspectrum_validation": _read_optional(
            run_root / "cipherspectrum_validation" / "metrics.json"
        ),
        "perturbation": _read_optional(
            run_root / "perturbation" / "aggregate_metrics.json"
        ),
        "utility": _read_optional(
            run_root / "utility_policy" / "aggregate_metrics.json"
        ),
    }
    utility_result = required_runs["utility"]
    core_smoke = _read_optional(
        run_root / "core_smoke_200" / "core_protocol_summary.json"
    )
    missing = [name for name, value in required_runs.items() if value is None]
    frozen_verification = (
        verify_frozen_v1_reference(frozen_reference)
        if frozen_reference is not None
        else None
    )
    if frozen_verification is not None:
        _dump(output / "frozen_v1_hash_verification.json", frozen_verification)
    training_agents = (training or {}).get("agents", {})
    temporal_status = training_agents.get("temporal", {}).get("status")
    tls_status = training_agents.get("tls", {}).get("status")
    failures = []
    if not tests_passed:
        failures.append("full pytest suite did not pass")
    if temporal_status != "trained":
        failures.append("Temporal deep_v2 was not trained")
    if tls_status != "trained":
        failures.append("TLS deep_v2 was not trained")
    if missing:
        failures.append("required v2 experiments are incomplete")
    if frozen_verification is not None and not frozen_verification["all_match"]:
        failures.append("one or more frozen v1 artifacts changed")
    acceptance_status = (
        "passed"
        if not failures
        else "protocol_ready"
        if tests_passed and temporal_status in {"trained", None}
        else "incomplete"
    )
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2",
        "acceptance_status": acceptance_status,
        "failures": failures,
        "missing_experiments": missing,
        "tests_passed": tests_passed,
        "test_count": test_count,
        "annotated_tls_supervised_status": (
            supervision or {}
        ).get("supervised_use_status"),
        "annotated_tls_binary_labels_inferred": False,
        "fallback_dataset": "HIKARI-2021",
        "temporal_deep_v2_status": temporal_status,
        "tls_deep_v2_status": tls_status,
        "utility_v2_promotion_status": (
            utility_result or {}
        ).get("promotion_status"),
        "utility_v2_negative_result": (
            utility_result or {}
        ).get("negative_result"),
        "core_smoke_completed": core_smoke is not None,
        "core_smoke_counts_as_full_acceptance": False,
        "fusion_modified": False,
        "ood_v1_v2_modified": False,
        "llm_or_memory_is_classifier": False,
        "automatic_training": False,
        "automatic_deployment": False,
        "cipherspectrum_locked_test_used_for_selection": False,
        "frozen_core_hashes_match_reference": (
            frozen_verification["all_match"]
            if frozen_verification is not None
            else None
        ),
    }
    _dump(output / "acceptance_report.json", report)
    negative = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2",
        "negative_or_incomplete_results": [
            {
                "item": "Annotated TLS 2026 supervised binary use",
                "status": (supervision or {}).get(
                    "supervised_use_status", "not_checked"
                ),
                "reason": (
                    "No explicit binary truth was found; null malware family "
                    "and system-service metadata were not converted into labels."
                ),
            },
            {
                "item": "TLSProtocolAgent deep_v2",
                "status": tls_status or "not_run",
                "reason": (
                    "Training requires explicitly labeled TLS record sequences; "
                    "the fallback HIKARI table has aggregate flow features only."
                ),
            },
            {
                "item": "EvidenceUtilityPolicyV2",
                "status": (
                    utility_result or {}
                ).get("promotion_status", "not_run"),
                "reason": (
                    (utility_result or {}).get(
                        "reason",
                        "The registered paired comparison is incomplete.",
                    )
                ),
                "deltas": (utility_result or {}).get("deltas"),
            },
            {
                "item": "Conformal v3 cross-domain smoke",
                "status": (
                    "completed_not_promoted"
                    if core_smoke is not None
                    else "not_run"
                ),
                "reason": (
                    "The limited integration run is not a substitute for the "
                    "registered full OOD comparison and cannot promote v3."
                ),
            },
            {
                "item": "Full v2 experiment matrix",
                "status": "complete" if not missing else "pending",
                "missing": missing,
            },
        ],
    }
    _dump(output / "negative_results.json", negative)
    artifact_files = [
        item
        for root in (dataset_root, model_root, run_root, output)
        if root.exists()
        for item in root.rglob("*")
        if item.is_file()
        and "nvidia_api_key" not in item.name.lower()
        and "flows" not in {part.lower() for part in item.parts}
        and "predictions" not in item.name.lower()
        and "replay" not in item.name.lower()
        and item.stat().st_size < 100 * 1024 * 1024
    ]
    _dump(
        output / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "files": [
                {
                    "path": str(item),
                    "sha256": _sha256(item),
                    "size_bytes": item.stat().st_size,
                }
                for item in sorted(set(artifact_files))
            ],
        },
    )
    return report
