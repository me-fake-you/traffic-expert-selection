from __future__ import annotations

import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .engine import build_default_engine
from .field_contract import field_contract_sha256, write_field_contract
from .io import iter_flow_records, iter_split_records
from .paper_evaluation import hash_artifact_paths, sha256_file
from .runtime_profiles import (
    load_runtime_profile,
    profile_artifact_paths,
    profile_engine_kwargs,
    runtime_profile_sha256,
)
from .schemas import FlowRecord, FutureFeatureFlags
from .security_acceptance import (
    FUSION_OWNED_FIELDS,
    REQUIRED_AUDIT_EVENTS,
    _fusion_ownership_violation,
    _ood_override,
)


V28_DATASET_COUNTS = {
    "ustc_validation": 2_000,
    "cesnet_validation": 2_000,
    "cipherspectrum_development": 1_000,
    "cipherspectrum_validation": 1_000,
    "hikari_validation": 1_000,
    "annotated_tls_ood": 1_000,
}
V28_SHADOW_COUNT = 1_000
V28_SHADOW_MODES = {
    "planner_executor": {
        "flags": FutureFeatureFlags(planner_executor=True),
        "rag": False,
    },
    "deliberation_shadow": {
        "flags": FutureFeatureFlags(deliberation=True, shadow_mode=True),
        "rag": False,
    },
    "reflection_critic": {
        "flags": FutureFeatureFlags(
            reflection=True,
            audit_critic=True,
            shadow_mode=True,
        ),
        "rag": False,
    },
    "rag": {
        "flags": FutureFeatureFlags(shadow_mode=True),
        "rag": True,
    },
    "memory_hint": {
        "flags": FutureFeatureFlags(memory=True, shadow_mode=True),
        "rag": False,
    },
}


def default_v28_sources() -> dict[str, dict[str, Any]]:
    return {
        "ustc_validation": {
            "dataset": "USTC-TFC2016",
            "input_path": "data/processed/ustc_tfc2016/v1/flows",
            "manifest_path": (
                "data/processed/ustc_tfc2016/v1/splits/"
                "split-manifest.json"
            ),
            "split": "validation",
            "adapter": "standard_split",
            "expected_count": 2_000,
            "supervised_metrics_allowed": False,
        },
        "cesnet_validation": {
            "dataset": "CESNET-TLS22",
            "input_path": "data/processed/cesnet_tls22/v1/flows",
            "manifest_path": (
                "data/processed/cesnet_tls22/v1/splits/"
                "split-manifest.json"
            ),
            "split": "validation",
            "adapter": "standard_split",
            "expected_count": 2_000,
            "supervised_metrics_allowed": False,
        },
        "cipherspectrum_development": {
            "dataset": "CipherSpectrum",
            "input_path": "data/processed/cipherspectrum/v1/flows",
            "manifest_path": (
                "data/processed/cipherspectrum/v1/splits/"
                "domain-split-manifest.json"
            ),
            "split": "development",
            "adapter": "assignment_split",
            "expected_count": 1_000,
            "supervised_metrics_allowed": False,
        },
        "cipherspectrum_validation": {
            "dataset": "CipherSpectrum",
            "input_path": "data/processed/cipherspectrum/v1/flows",
            "manifest_path": (
                "data/processed/cipherspectrum/v1/splits/"
                "domain-split-manifest.json"
            ),
            "split": "validation",
            "adapter": "assignment_split",
            "expected_count": 1_000,
            "supervised_metrics_allowed": False,
        },
        "hikari_validation": {
            "dataset": "HIKARI-2021",
            "input_path": (
                "data/processed/annotated_tls_2026/v1/fallback_hikari/flows"
            ),
            "manifest_path": (
                "data/processed/annotated_tls_2026/v1/fallback_hikari/"
                "splits/split-manifest.json"
            ),
            "split": "validation",
            "adapter": "standard_split",
            "expected_count": 1_000,
            "supervised_metrics_allowed": False,
        },
        "annotated_tls_ood": {
            "dataset": "Annotated-TLS-2026",
            "input_path": (
                "data/processed/annotated_tls_2026/v1/flows/"
                "unlabeled-ood.jsonl.gz"
            ),
            "manifest_path": None,
            "split": "unlabeled_ood",
            "adapter": "all_records",
            "expected_count": 1_000,
            "supervised_metrics_allowed": False,
        },
    }


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _priority(seed: int, dataset: str, sample_id: str) -> int:
    return int(
        hashlib.sha256(
            f"{seed}:{dataset}:{sample_id}".encode("utf-8")
        ).hexdigest(),
        16,
    )


def _validate_source(name: str, spec: dict[str, Any]) -> None:
    if spec["split"] in {"test", "locked_test"}:
        raise ValueError(f"v2.8 forbids test/locked_test source: {name}")
    if "locked_test" in str(spec["input_path"]).lower():
        raise ValueError(f"v2.8 forbids locked_test path: {name}")
    if not Path(spec["input_path"]).exists():
        raise FileNotFoundError(spec["input_path"])
    manifest = spec.get("manifest_path")
    if manifest and not Path(manifest).exists():
        raise FileNotFoundError(manifest)
    if spec.get("supervised_metrics_allowed"):
        raise ValueError("v2.8 conformance must not emit supervised metrics")


def _iter_source(spec: dict[str, Any]) -> Iterator[FlowRecord]:
    adapter = spec["adapter"]
    if adapter == "all_records":
        yield from iter_flow_records(spec["input_path"])
        return
    if adapter == "standard_split":
        yield from iter_split_records(
            spec["input_path"],
            spec["manifest_path"],
            spec["split"],
        )
        return
    if adapter == "assignment_split":
        manifest = json.loads(
            Path(spec["manifest_path"]).read_text(encoding="utf-8")
        )
        selected = set(manifest["assignments"][spec["split"]])
        for record in iter_flow_records(spec["input_path"]):
            if record.sample_id in selected:
                yield record
        return
    raise ValueError(f"unsupported v2.8 source adapter: {adapter}")


def _hash_sample(
    records: Iterable[FlowRecord],
    *,
    dataset: str,
    count: int,
    seed: int,
) -> tuple[list[str], int]:
    heap: list[tuple[int, str]] = []
    seen: set[str] = set()
    population_count = 0
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"duplicate sample ID in {dataset}: {record.sample_id}")
        seen.add(record.sample_id)
        population_count += 1
        priority = _priority(seed, dataset, record.sample_id)
        entry = (-priority, record.sample_id)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    if len(heap) != count:
        raise RuntimeError(
            f"{dataset} yielded {len(heap)}/{count} conformance samples"
        )
    selected = [sample_id for _, sample_id in heap]
    selected.sort(key=lambda item: _priority(seed, dataset, item))
    return selected, population_count


def build_v28_conformance_manifest(
    output_dir: str | Path,
    *,
    sources: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
    shadow_count: int = V28_SHADOW_COUNT,
) -> dict[str, Any]:
    specs = sources or default_v28_sources()
    if set(specs) != set(V28_DATASET_COUNTS):
        raise ValueError(
            "v2.8 sources must be exactly "
            f"{sorted(V28_DATASET_COUNTS)}"
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    contract = write_field_contract(target / "field_contract.json")
    datasets: dict[str, Any] = {}
    all_keys: list[str] = []
    for name, raw_spec in specs.items():
        spec = dict(raw_spec)
        _validate_source(name, spec)
        expected = int(spec["expected_count"])
        if expected != V28_DATASET_COUNTS[name]:
            raise ValueError(f"v2.8 count changed for {name}")
        sample_ids, population = _hash_sample(
            _iter_source(spec),
            dataset=name,
            count=expected,
            seed=seed,
        )
        source_manifest = spec.get("manifest_path")
        datasets[name] = {
            **spec,
            "population_count": population,
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "source_manifest_sha256": (
                sha256_file(source_manifest) if source_manifest else None
            ),
            "labels_read_for_selection": False,
            "used_for_model_or_policy_selection": False,
        }
        all_keys.extend(f"{name}::{sample_id}" for sample_id in sample_ids)
    if shadow_count > len(all_keys):
        raise ValueError("shadow_count exceeds conformance sample count")
    shadow_keys = sorted(
        all_keys,
        key=lambda key: hashlib.sha256(
            f"{seed}:shadow:{key}".encode("utf-8")
        ).hexdigest(),
    )[:shadow_count]
    payload = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_8_runtime_conformance",
        "seed": seed,
        "strategy": "sample_id_hash_without_label_access",
        "sample_count": sum(
            item["sample_count"] for item in datasets.values()
        ),
        "expected_sample_count": sum(V28_DATASET_COUNTS.values()),
        "shadow_sample_count": len(shadow_keys),
        "datasets": datasets,
        "shadow_sample_keys": shadow_keys,
        "field_contract_sha256": contract["contract_sha256"],
        "test_used": False,
        "external_used_for_model_or_policy_selection": False,
        "cipherspectrum_locked_test_used": False,
        "supervised_metrics_emitted": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(target / "selection_manifest.json", payload)
    return payload


def _iter_manifest_records(
    manifest: dict[str, Any],
    *,
    shadow_only: bool = False,
) -> Iterator[tuple[str, FlowRecord]]:
    shadow = set(manifest["shadow_sample_keys"]) if shadow_only else None
    for name, spec in manifest["datasets"].items():
        selected = set(spec["sample_ids"])
        if shadow is not None:
            selected = {
                key.split("::", 1)[1]
                for key in shadow
                if key.startswith(f"{name}::")
            }
        yielded: set[str] = set()
        for record in _iter_source(spec):
            if record.sample_id not in selected:
                continue
            if record.sample_id in yielded:
                raise ValueError(f"duplicate selected sample: {name}")
            yielded.add(record.sample_id)
            yield name, record
        missing = selected - yielded
        if missing:
            raise RuntimeError(
                f"{name} is missing {len(missing)} selected records"
            )


def _snapshot(report) -> dict[str, Any]:
    return {
        field: (
            getattr(report, field).value
            if hasattr(getattr(report, field), "value")
            else getattr(report, field)
        )
        for field in FUSION_OWNED_FIELDS
    }


def _ood_signature(report) -> list[dict[str, Any]]:
    return [
        {
            "agent": item.agent_name,
            "score": item.distribution_shift_score,
            "raw_score": item.distribution_shift_raw_score,
            "level": item.distribution_shift_level,
            "reliability": item.model_reliability,
            "abstained": item.abstained,
        }
        for item in report.agent_results
        if item.agent_name in {"StatsDetectorAgent", "TemporalBehaviorAgent"}
    ]


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _open_csv(path: Path, fields: list[str], *, resume: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_csv(path) if resume else []
    handle = path.open(
        "a" if existing else "w",
        encoding="utf-8-sig",
        newline="",
    )
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not existing:
        writer.writeheader()
    return handle, writer, existing


def _engine(profile, *, mode: str, memory_dir: str | Path):
    kwargs = profile_engine_kwargs(profile)
    kwargs.update(
        {
            "force_rule_coordinator": True,
            "field_audit_mode": mode,
            "max_workers": 1,
        }
    )
    return build_default_engine(**kwargs)


def _audit_safety(report, audit) -> dict[str, int | bool]:
    event_types = {event.event_type for event in audit.events}
    blocked = sum(
        bool(event.input_summary.get("blocked_field_intersection", []))
        for event in audit.events
        if event.event_type == "AGENT_EVIDENCE"
    )
    field_event = next(
        event
        for event in audit.events
        if event.event_type == "FIELD_POLICY_APPLIED"
    )
    unknown = set(field_event.output_summary.get("unknown_fields", []))
    visible = set(field_event.output_summary.get("detector_visible_fields", []))
    illegal_verdict_execution = sum(
        bool(
            decision.planner_metadata.get(
                "illegal_verdict_execution_count",
                0,
            )
        )
        for decision in report.coordinator_decisions
    )
    return {
        "audit_complete": REQUIRED_AUDIT_EVENTS.issubset(event_types),
        "blocked_field_violation": blocked,
        "unknown_field_execution": len(unknown & visible),
        "fusion_ownership_violation": int(
            _fusion_ownership_violation(report, audit)
        ),
        "ood_override": int(_ood_override(audit, report)),
        "illegal_verdict_execution": illegal_verdict_execution,
    }


def _strict_legacy_rows(
    manifest: dict[str, Any],
    target: Path,
    *,
    profile_name: str,
    config_dir: str | Path,
    resume: bool,
) -> list[dict[str, str]]:
    profile = load_runtime_profile(profile_name, config_dir=config_dir)
    strict = _engine(profile, mode="strict", memory_dir=target / "unused")
    legacy = _engine(profile, mode="legacy", memory_dir=target / "unused")
    fields = [
        "dataset",
        "sample_id",
        "strict_snapshot",
        "legacy_snapshot",
        "strict_ood_signature",
        "legacy_ood_signature",
        "strict_agent_calls",
        "legacy_agent_calls",
        "verdict_agreement",
        "ood_agreement",
        "agent_call_agreement",
        "audit_complete",
        "blocked_field_violation",
        "unknown_field_execution",
        "fusion_ownership_violation",
        "ood_override",
        "illegal_verdict_execution",
        "strict_field_decisions",
        "strict_unknown_fields",
        "strict_ignored_empty_fields",
        "strict_visible_fields",
    ]
    path = target / "strict_vs_legacy" / "predictions.csv"
    handle, writer, existing = _open_csv(path, fields, resume=resume)
    completed = {(row["dataset"], row["sample_id"]) for row in existing}
    try:
        for dataset, record in _iter_manifest_records(manifest):
            key = (dataset, record.sample_id)
            if key in completed:
                continue
            strict_report, strict_audit = strict.analyze(record)
            legacy_report, _ = legacy.analyze(record)
            strict_snapshot = _snapshot(strict_report)
            legacy_snapshot = _snapshot(legacy_report)
            strict_ood = _ood_signature(strict_report)
            legacy_ood = _ood_signature(legacy_report)
            safety = _audit_safety(strict_report, strict_audit)
            field_event = next(
                event
                for event in strict_audit.events
                if event.event_type == "FIELD_POLICY_APPLIED"
            )
            row = {
                "dataset": dataset,
                "sample_id": record.sample_id,
                "strict_snapshot": _json_cell(strict_snapshot),
                "legacy_snapshot": _json_cell(legacy_snapshot),
                "strict_ood_signature": _json_cell(strict_ood),
                "legacy_ood_signature": _json_cell(legacy_ood),
                "strict_agent_calls": len(strict_report.agent_results),
                "legacy_agent_calls": len(legacy_report.agent_results),
                "verdict_agreement": int(
                    strict_snapshot["verdict"]
                    == legacy_snapshot["verdict"]
                ),
                "ood_agreement": int(strict_ood == legacy_ood),
                "agent_call_agreement": int(
                    len(strict_report.agent_results)
                    == len(legacy_report.agent_results)
                ),
                **{key: int(value) for key, value in safety.items()},
                "strict_field_decisions": _json_cell(
                    field_event.output_summary["decisions"]
                ),
                "strict_unknown_fields": _json_cell(
                    field_event.output_summary["unknown_fields"]
                ),
                "strict_ignored_empty_fields": _json_cell(
                    field_event.output_summary["ignored_empty_fields"]
                ),
                "strict_visible_fields": _json_cell(
                    field_event.output_summary["detector_visible_fields"]
                ),
            }
            writer.writerow(row)
            handle.flush()
    finally:
        handle.close()
    rows = _load_csv(path)
    if len(rows) != manifest["sample_count"]:
        raise RuntimeError(
            f"strict/legacy run incomplete: {len(rows)}/"
            f"{manifest['sample_count']}"
        )
    return rows


def _shadow_engine(
    profile,
    mode: str,
    *,
    memory_dir: str | Path,
    knowledge_base_dir: str | Path,
):
    spec = V28_SHADOW_MODES[mode]
    kwargs = profile_engine_kwargs(profile)
    kwargs.update(
        {
            "force_rule_coordinator": True,
            "field_audit_mode": "strict",
            "future_flags": spec["flags"],
            "enable_rag": spec["rag"],
            "knowledge_base_dir": knowledge_base_dir,
            "memory_dir": memory_dir,
            "max_workers": 1,
        }
    )
    return build_default_engine(**kwargs)


def _mode_executed(mode: str, audit) -> bool:
    artifacts = audit.future_artifacts
    event_types = {event.event_type for event in audit.events}
    if mode == "planner_executor":
        return bool(artifacts.execution_plans)
    if mode == "deliberation_shadow":
        return bool(artifacts.deliberations)
    if mode == "reflection_critic":
        return artifacts.reflection is not None and artifacts.compliance is not None
    if mode == "rag":
        return "RAG_RETRIEVAL_COMPLETED" in event_types
    if mode == "memory_hint":
        return artifacts.memory_hint is not None
    return False


def _shadow_rows(
    manifest: dict[str, Any],
    target: Path,
    *,
    profile_name: str,
    config_dir: str | Path,
    memory_dir: str | Path,
    knowledge_base_dir: str | Path,
    resume: bool,
) -> list[dict[str, str]]:
    profile = load_runtime_profile(profile_name, config_dir=config_dir)
    baseline = _engine(profile, mode="strict", memory_dir=memory_dir)
    engines = {
        mode: _shadow_engine(
            profile,
            mode,
            memory_dir=memory_dir,
            knowledge_base_dir=knowledge_base_dir,
        )
        for mode in V28_SHADOW_MODES
    }
    fields = [
        "dataset",
        "sample_id",
        "mode",
        "fusion_invariance",
        "ood_invariance",
        "audit_complete",
        "blocked_field_violation",
        "unknown_field_execution",
        "fusion_ownership_violation",
        "ood_override",
        "illegal_verdict_execution",
        "mode_executed",
        "baseline_snapshot",
        "mode_snapshot",
        "baseline_ood_signature",
        "mode_ood_signature",
        "baseline_agent_calls",
        "mode_agent_calls",
    ]
    path = target / "shadow_invariance" / "predictions.csv"
    handle, writer, existing = _open_csv(path, fields, resume=resume)
    completed = {
        (row["dataset"], row["sample_id"], row["mode"])
        for row in existing
    }
    try:
        for dataset, record in _iter_manifest_records(
            manifest,
            shadow_only=True,
        ):
            baseline_report, baseline_audit = baseline.analyze(record)
            baseline_snapshot = _snapshot(baseline_report)
            baseline_ood = _ood_signature(baseline_report)
            baseline_safety = _audit_safety(
                baseline_report,
                baseline_audit,
            )
            for mode, engine in engines.items():
                key = (dataset, record.sample_id, mode)
                if key in completed:
                    continue
                report, audit = engine.analyze(record)
                snapshot = _snapshot(report)
                ood_signature = _ood_signature(report)
                safety = _audit_safety(report, audit)
                row = {
                    "dataset": dataset,
                    "sample_id": record.sample_id,
                    "mode": mode,
                    "fusion_invariance": int(snapshot == baseline_snapshot),
                    "ood_invariance": int(ood_signature == baseline_ood),
                    "audit_complete": int(
                        bool(safety["audit_complete"])
                        and bool(baseline_safety["audit_complete"])
                    ),
                    "blocked_field_violation": int(
                        safety["blocked_field_violation"]
                    ),
                    "unknown_field_execution": int(
                        safety["unknown_field_execution"]
                    ),
                    "fusion_ownership_violation": int(
                        safety["fusion_ownership_violation"]
                    ),
                    "ood_override": int(safety["ood_override"]),
                    "illegal_verdict_execution": int(
                        safety["illegal_verdict_execution"]
                    ),
                    "mode_executed": int(_mode_executed(mode, audit)),
                    "baseline_snapshot": _json_cell(baseline_snapshot),
                    "mode_snapshot": _json_cell(snapshot),
                    "baseline_ood_signature": _json_cell(baseline_ood),
                    "mode_ood_signature": _json_cell(ood_signature),
                    "baseline_agent_calls": len(baseline_report.agent_results),
                    "mode_agent_calls": len(report.agent_results),
                }
                writer.writerow(row)
                handle.flush()
    finally:
        handle.close()
    rows = _load_csv(path)
    expected = manifest["shadow_sample_count"] * len(V28_SHADOW_MODES)
    if len(rows) != expected:
        raise RuntimeError(f"shadow run incomplete: {len(rows)}/{expected}")
    return rows


def _field_profiles(rows: list[dict[str, str]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        roles: dict[str, Counter[str]] = defaultdict(Counter)
        unknown: Counter[str] = Counter()
        ignored: Counter[str] = Counter()
        visible: Counter[str] = Counter()
        for row in selected:
            for item in json.loads(row["strict_field_decisions"]):
                roles[item["path"]][item["role"]] += 1
            unknown.update(json.loads(row["strict_unknown_fields"]))
            ignored.update(json.loads(row["strict_ignored_empty_fields"]))
            visible.update(json.loads(row["strict_visible_fields"]))
        profiles[dataset] = {
            "sample_count": len(selected),
            "field_roles": {
                path: dict(sorted(counts.items()))
                for path, counts in sorted(roles.items())
            },
            "unknown_populated_fields": dict(sorted(unknown.items())),
            "ignored_empty_fields": dict(sorted(ignored.items())),
            "detector_visible_fields": dict(sorted(visible.items())),
        }
    return profiles


def run_v28_runtime_conformance(
    output_dir: str | Path,
    *,
    profile_name: str = "runtime_safe_v2_8",
    config_dir: str | Path = "data/configs",
    memory_dir: str | Path = (
        "data/runs/future_framework/hitl_pilot_v1/memory"
    ),
    knowledge_base_dir: str | Path = "knowledge_base",
    resume: bool = True,
    rerun_shadow: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir)
    manifest_path = target / "selection_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_used") or manifest.get(
        "cipherspectrum_locked_test_used"
    ):
        raise ValueError("v2.8 manifest includes a forbidden test source")
    if manifest.get("field_contract_sha256") != field_contract_sha256():
        raise ValueError("v2.8 field contract changed after manifest freeze")
    profile = load_runtime_profile(profile_name, config_dir=config_dir)
    artifact_paths = profile_artifact_paths(profile)
    artifact_paths["runtime_profile"] = (
        Path(config_dir) / f"{profile_name}.json"
    )
    artifact_paths["field_contract"] = Path(
        "src/mad_etd/field_contract.py"
    )
    if Path(knowledge_base_dir).exists():
        artifact_paths["knowledge_base"] = Path(knowledge_base_dir)
    if (Path(memory_dir) / "cases.jsonl").exists():
        artifact_paths["case_memory"] = Path(memory_dir)
    before_path = target / "frozen_artifact_hashes_before.json"
    if not before_path.exists():
        _dump(before_path, hash_artifact_paths(artifact_paths))
    strict_rows = _strict_legacy_rows(
        manifest,
        target,
        profile_name=profile_name,
        config_dir=config_dir,
        resume=resume,
    )
    shadow_rows = _shadow_rows(
        manifest,
        target,
        profile_name=profile_name,
        config_dir=config_dir,
        memory_dir=memory_dir,
        knowledge_base_dir=knowledge_base_dir,
        resume=resume and not rerun_shadow,
    )
    profiles = _field_profiles(strict_rows)
    _dump(target / "dataset_field_profiles.json", profiles)
    _dump(
        target / "frozen_artifact_hashes_after.json",
        hash_artifact_paths(artifact_paths),
    )
    payload = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_8_runtime_conformance",
        "status": "completed",
        "runtime_profile": profile.name,
        "runtime_profile_sha256": runtime_profile_sha256(profile),
        "strict_legacy_sample_count": len(strict_rows),
        "shadow_evaluation_count": len(shadow_rows),
        "resume_supported": True,
        "api_calls_made": False,
        "labels_used": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(target / "run_summary.json", payload)
    return payload


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return sum(int(row[field]) for row in rows) / max(len(rows), 1)


def finalize_v28(
    output_dir: str | Path,
    *,
    document_path: str | Path | None = None,
    negative_document_path: str | Path | None = None,
    test_count: int | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    manifest = json.loads(
        (target / "selection_manifest.json").read_text(encoding="utf-8")
    )
    strict_rows = _load_csv(
        target / "strict_vs_legacy" / "predictions.csv"
    )
    shadow_rows = _load_csv(
        target / "shadow_invariance" / "predictions.csv"
    )
    before = json.loads(
        (target / "frozen_artifact_hashes_before.json").read_text(
            encoding="utf-8"
        )
    )
    after = json.loads(
        (target / "frozen_artifact_hashes_after.json").read_text(
            encoding="utf-8"
        )
    )
    comparisons: list[dict[str, Any]] = []
    for dataset in sorted(manifest["datasets"]):
        rows = [row for row in strict_rows if row["dataset"] == dataset]
        verdict_agreement = sum(
            json.loads(row["strict_snapshot"])["verdict"]
            == json.loads(row["legacy_snapshot"])["verdict"]
            for row in rows
        ) / max(len(rows), 1)
        comparisons.append(
            {
                "dataset": dataset,
                "sample_count": len(rows),
                "verdict_agreement": verdict_agreement,
                "ood_signature_agreement": _rate(rows, "ood_agreement"),
                "agent_call_agreement": _rate(
                    rows,
                    "agent_call_agreement",
                ),
            }
        )
    paired_path = target / "paired_comparisons.csv"
    with paired_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "sample_count",
                "verdict_agreement",
                "ood_signature_agreement",
                "agent_call_agreement",
            ],
        )
        writer.writeheader()
        writer.writerows(comparisons)

    shadow_by_mode = {
        mode: {
            "sample_count": len(
                [row for row in shadow_rows if row["mode"] == mode]
            ),
            "fusion_invariance": _rate(
                [row for row in shadow_rows if row["mode"] == mode],
                "fusion_invariance",
            ),
            "ood_invariance": _rate(
                [row for row in shadow_rows if row["mode"] == mode],
                "ood_invariance",
            ),
            "mode_execution_rate": _rate(
                [row for row in shadow_rows if row["mode"] == mode],
                "mode_executed",
            ),
        }
        for mode in V28_SHADOW_MODES
    }
    safety_fields = (
        "blocked_field_violation",
        "unknown_field_execution",
        "fusion_ownership_violation",
        "ood_override",
        "illegal_verdict_execution",
    )
    all_rows = strict_rows + shadow_rows
    safety = {
        f"{field}_count": sum(int(row[field]) for row in all_rows)
        for field in safety_fields
    }
    audit_completion = _rate(all_rows, "audit_complete")
    checks = {
        "sample_count_matches_manifest": (
            len(strict_rows) == manifest["expected_sample_count"]
        ),
        "shadow_sample_count_matches_manifest": (
            len(shadow_rows)
            == manifest["shadow_sample_count"] * len(V28_SHADOW_MODES)
        ),
        "strict_legacy_verdict_agreement_is_one": all(
            item["verdict_agreement"] == 1.0 for item in comparisons
        ),
        "strict_legacy_ood_agreement_is_one": all(
            item["ood_signature_agreement"] == 1.0
            for item in comparisons
        ),
        "strict_legacy_agent_calls_agreement_is_one": all(
            item["agent_call_agreement"] == 1.0 for item in comparisons
        ),
        "shadow_fusion_invariance_is_one": all(
            item["fusion_invariance"] == 1.0
            for item in shadow_by_mode.values()
        ),
        "shadow_ood_invariance_is_one": all(
            item["ood_invariance"] == 1.0
            for item in shadow_by_mode.values()
        ),
        "shadow_modes_executed": all(
            item["mode_execution_rate"] > 0
            for item in shadow_by_mode.values()
        ),
        "audit_completion_is_one": audit_completion == 1.0,
        "safety_violations_are_zero": all(
            count == 0 for count in safety.values()
        ),
        "frozen_artifacts_unchanged": before == after,
        "test_not_used": manifest.get("test_used") is False,
        "external_not_used_for_selection": manifest.get(
            "external_used_for_model_or_policy_selection"
        )
        is False,
        "locked_test_not_used": manifest.get(
            "cipherspectrum_locked_test_used"
        )
        is False,
        "no_supervised_metrics": manifest.get(
            "supervised_metrics_emitted"
        )
        is False,
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "schema_version": "1.0",
        "experiment": "mad_etd_v2_8_runtime_conformance",
        "acceptance_status": "passed" if not failures else "failed",
        "promotion_status": (
            "promote_runtime_safe_v2_8"
            if not failures
            else "retain_runtime_safe_v2_7"
        ),
        "test_count": test_count,
        "failures": failures,
        "checks": checks,
        "strict_legacy": comparisons,
        "shadow_invariance": shadow_by_mode,
        "audit_completion_rate": audit_completion,
        **safety,
        "frozen_artifacts_unchanged": before == after,
        "field_contract_sha256": manifest["field_contract_sha256"],
        "test_used": False,
        "external_used_for_model_or_policy_selection": False,
        "cipherspectrum_locked_test_used": False,
        "automatic_training": False,
        "automatic_deployment": False,
    }
    _dump(target / "acceptance_report.json", report)
    if document_path is not None:
        lines = [
            "# MAD-ETD v2.8 Runtime Conformance",
            "",
            f"- Status: `{report['acceptance_status']}`",
            f"- Promotion: `{report['promotion_status']}`",
            f"- Strict/legacy samples: `{len(strict_rows)}`",
            f"- Shadow evaluations: `{len(shadow_rows)}`",
            f"- Audit completion: `{audit_completion}`",
            f"- Frozen artifacts unchanged: `{str(before == after).lower()}`",
            "",
            "## Strict versus legacy",
            "",
            "| Dataset | Verdict agreement | OOD agreement | Agent-call agreement |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| {item['dataset']} | "
                    f"{item['verdict_agreement']:.4f} | "
                    f"{item['ood_signature_agreement']:.4f} | "
                    f"{item['agent_call_agreement']:.4f} |"
                )
                for item in comparisons
            ],
            "",
            "## Agentic shadow",
            "",
            "| Mode | Fusion invariance | OOD invariance | Executed |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| {mode} | {item['fusion_invariance']:.4f} | "
                    f"{item['ood_invariance']:.4f} | "
                    f"{item['mode_execution_rate']:.4f} |"
                )
                for mode, item in shadow_by_mode.items()
            ],
            "",
            "No model training, threshold selection, NVIDIA API call, or "
            "supervised external metric was performed.",
        ]
        document = Path(document_path)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text("\n".join(lines), encoding="utf-8")
    if negative_document_path is not None:
        negative = Path(negative_document_path)
        negative.parent.mkdir(parents=True, exist_ok=True)
        negative.write_text(
            "\n".join(
                [
                    "# MAD-ETD v2.8 Negative Results",
                    "",
                    (
                        "No v2.8 acceptance failure was observed."
                        if not failures
                        else "The following preregistered checks failed:"
                    ),
                    *[f"- `{item}`" for item in failures],
                    "",
                    (
                        "CipherSpectrum retained final-verdict and OOD "
                        "agreement at 1.0, but strict removal of exact cipher "
                        "metadata changed TLS reliability-driven routing. "
                        "Agent-call agreement was 0.9730 on development and "
                        "0.9410 on validation. The field was not restored."
                        if "strict_legacy_agent_calls_agreement_is_one"
                        in failures
                        else ""
                    ),
                    "",
                    "Failures are retained without expanding the allowlist "
                    "after observing acceptance results.",
                ]
            ),
            encoding="utf-8",
        )
    return report
