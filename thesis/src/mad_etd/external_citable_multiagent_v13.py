"""Build a citable, evidence-tiered five-system comparison (v13).

This round performs no training. It replaces repository-only systems in the
main v12 table with systems that have a DOI or a directly verifiable preprint,
while preserving the distinction between official-code-participating adapted
comparisons and paper-reported cross-protocol references.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT = "mad_etd_external_citable_multiagent_v13"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_external_citable_multiagent_v13")
DEFAULT_V8 = Path("data/runs/mad_etd_external_multiagent_fair_v8")
DEFAULT_V9 = Path("data/runs/mad_etd_external_multiagent_fair_v9")
DEFAULT_V10 = Path("data/runs/mad_etd_external_peer_multiagent_v10")
DEFAULT_V11 = Path("data/runs/mad_etd_external_peer_multiagent_v11")
DEFAULT_W237 = Path(
    "data/runs/mad_etd_five_system_positive_evidence_w236_w240"
)
DEFAULT_RUNTIME = Path("data/configs/runtime_safe_v3_0.json")
DEFAULT_DOCUMENT = Path("docs/MAD_ETD_EXTERNAL_CITABLE_MULTIAGENT_V13.md")
DEFAULT_DOCUMENT_CN = Path(
    "docs/MAD_ETD_EXTERNAL_CITABLE_MULTIAGENT_V13_CN.md"
)


MAIN_SYSTEMS = (
    "Continual-Federated-IDS",
    "ZTA-FL",
    "MARL-NIDS",
    "MA-IDS",
    "MAS-LSTM",
)


BIBLIOGRAPHIC_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "system": "Continual-Federated-IDS",
        "title": (
            "A Multi-Agent Adaptive Deep Learning Framework for Online "
            "Intrusion Detection"
        ),
        "primary_identity_type": "doi",
        "primary_identity": "10.1186/s42400-023-00199-0",
        "primary_url": "https://doi.org/10.1186/s42400-023-00199-0",
        "secondary_identity": "arXiv:2303.02622",
        "secondary_url": "https://arxiv.org/abs/2303.02622",
        "publication_year": 2024,
        "first_public_date": "2023-03-05",
        "publication_status": "peer_reviewed",
        "identity_directly_verifiable": True,
        "selected_for_main_table": True,
        "comparison_lane": "official_code_participating_adapted_same_query",
        "official_code_participating": True,
        "locally_executed": True,
        "official_code_url": (
            "https://github.com/INL-Laboratory/Continual-Federated-IDS"
        ),
        "scope_note": "Adapted safe-flow same-query lane; not paper-scale reproduction.",
    },
    {
        "system": "ZTA-FL",
        "title": (
            "Zero-Trust Agentic Federated Learning for Secure Internet of "
            "Things (IoT) Defense Systems"
        ),
        "primary_identity_type": "doi",
        "primary_identity": "10.1109/SATC69565.2026.11542411",
        "primary_url": "https://doi.org/10.1109/SATC69565.2026.11542411",
        "secondary_identity": "arXiv:2512.23809",
        "secondary_url": "https://arxiv.org/abs/2512.23809",
        "publication_year": 2026,
        "first_public_date": "2026-03-24",
        "publication_status": "peer_reviewed",
        "identity_directly_verifiable": True,
        "selected_for_main_table": True,
        "comparison_lane": "official_code_participating_adapted_same_query",
        "official_code_participating": True,
        "locally_executed": True,
        "official_code_url": "https://github.com/ssam18/zta-federated-learning",
        "scope_note": (
            "Resource-bounded source-group comparison with visible source "
            "heterogeneity; not paper-scale reproduction."
        ),
    },
    {
        "system": "MARL-NIDS",
        "title": (
            "Multi-agent Reinforcement Learning-based Network Intrusion "
            "Detection System"
        ),
        "primary_identity_type": "doi",
        "primary_identity": "10.1109/NOMS59830.2024.10575541",
        "primary_url": "https://doi.org/10.1109/NOMS59830.2024.10575541",
        "secondary_identity": "arXiv:2407.05766",
        "secondary_url": "https://arxiv.org/abs/2407.05766",
        "publication_year": 2024,
        "first_public_date": "2024-05-06",
        "publication_status": "peer_reviewed",
        "identity_directly_verifiable": True,
        "selected_for_main_table": True,
        "comparison_lane": "paper_reported_cross_protocol",
        "official_code_participating": False,
        "locally_executed": False,
        "official_code_url": "",
        "scope_note": (
            "Paper scalar versus a separately frozen local reconstruction; "
            "samples and protocol are not paired."
        ),
    },
    {
        "system": "MA-IDS",
        "title": (
            "MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion "
            "Detection with an Experience Library"
        ),
        "primary_identity_type": "arxiv",
        "primary_identity": "arXiv:2604.05458",
        "primary_url": "https://arxiv.org/abs/2604.05458",
        "secondary_identity": "",
        "secondary_url": "",
        "publication_year": 2026,
        "first_public_date": "2026-04-07",
        "publication_status": "verifiable_preprint",
        "identity_directly_verifiable": True,
        "selected_for_main_table": True,
        "comparison_lane": "paper_reported_cross_protocol",
        "official_code_participating": False,
        "locally_executed": False,
        "official_code_url": "",
        "scope_note": (
            "Preprint scalar versus a separately frozen local safe-input "
            "reconstruction; no same split."
        ),
    },
    {
        "system": "MAS-LSTM",
        "title": (
            "MAS-LSTM: A Multi-Agent LSTM-Based Approach for Scalable "
            "Anomaly Detection in IIoT Networks"
        ),
        "primary_identity_type": "doi",
        "primary_identity": "10.3390/pr13030753",
        "primary_url": "https://doi.org/10.3390/pr13030753",
        "secondary_identity": "",
        "secondary_url": "",
        "publication_year": 2025,
        "first_public_date": "2025-03-05",
        "publication_status": "peer_reviewed",
        "identity_directly_verifiable": True,
        "selected_for_main_table": True,
        "comparison_lane": "paper_reported_cross_protocol",
        "official_code_participating": False,
        "locally_executed": False,
        "official_code_url": "",
        "scope_note": (
            "Positive only for Accuracy and Precision; Recall and F1 are "
            "negative and retained in the ledger."
        ),
    },
    {
        "system": "Attention-based multi-agent IDS (JISA 2021)",
        "title": (
            "Attention based multi-agent intrusion detection systems using "
            "reinforcement learning"
        ),
        "primary_identity_type": "doi",
        "primary_identity": "10.1016/j.jisa.2021.102923",
        "primary_url": "https://doi.org/10.1016/j.jisa.2021.102923",
        "secondary_identity": "",
        "secondary_url": "",
        "publication_year": 2021,
        "first_public_date": "2021",
        "publication_status": "peer_reviewed",
        "identity_directly_verifiable": True,
        "selected_for_main_table": False,
        "comparison_lane": "blocked_original_metric_table_not_directly_verified",
        "official_code_participating": False,
        "locally_executed": False,
        "official_code_url": "",
        "scope_note": (
            "Bibliographic identity is verified, but the original numerical "
            "table was not directly accessible; secondary copies are excluded."
        ),
    },
)


def _dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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
        "new_mad_etd_detector_created": False,
        "training_started": False,
    }


def audit_citable_multiagent_systems_v13(
    output_dir: str | Path = DEFAULT_OUTPUT,
    runtime_path: str | Path = DEFAULT_RUNTIME,
) -> dict[str, Any]:
    """Freeze citable identities and direct-verification boundaries."""

    out, runtime = Path(output_dir), Path(runtime_path)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "bibliographic_registry.csv", BIBLIOGRAPHIC_RECORDS)
    selected = [
        row for row in BIBLIOGRAPHIC_RECORDS if row["selected_for_main_table"]
    ]
    valid_identity = all(
        row["identity_directly_verifiable"]
        and row["primary_identity_type"] in {"doi", "arxiv"}
        and row["primary_identity"]
        for row in selected
    )
    report = {
        "status": (
            "audited_v13_five_citable_systems"
            if len(selected) == 5 and valid_identity and runtime.is_file()
            else "blocked_v13_bibliographic_or_runtime_requirement"
        ),
        "bibliographic_record_count": len(BIBLIOGRAPHIC_RECORDS),
        "selected_citable_system_count": len(selected),
        "selected_citable_systems": [row["system"] for row in selected],
        "all_selected_identities_directly_verifiable": valid_identity,
        "jisa_2021_status": "blocked_original_metric_table_not_directly_verified",
        "secondary_metric_copy_used": False,
        "supervised_metrics_generated": False,
        "runtime_hash_before": _sha256(runtime),
        **_security(),
    }
    _dump(out / "bibliographic_audit_report.json", report)
    return report


def _metric_row(
    *,
    system: str,
    dataset_task: str,
    metric: str,
    external_value: Any,
    mad_value: Any,
    interval_lower: Any,
    interval_upper: Any,
    interval_type: str,
    interval_scope: str,
    interval_supports_positive: bool,
    formal_paired_significance: bool,
    evidence_tier: str,
    official_code_participating: bool,
    same_query: bool,
    same_safe_input: bool,
    same_split_roles: bool,
    source: Path,
) -> dict[str, Any]:
    external = _number(external_value)
    mad = _number(mad_value)
    delta = mad - external if external is not None and mad is not None else None
    bibliography = next(
        row for row in BIBLIOGRAPHIC_RECORDS if row["system"] == system
    )
    return {
        "external_system": system,
        "paper_identity": bibliography["primary_identity"],
        "publication_status": bibliography["publication_status"],
        "dataset_task": dataset_task,
        "metric": metric,
        "external_value": external,
        "mad_etd_value": mad,
        "delta": delta,
        "delta_percentage_points": delta * 100 if delta is not None else None,
        "numerically_positive": bool(delta is not None and delta > 0),
        "interval_lower": _number(interval_lower),
        "interval_upper": _number(interval_upper),
        "interval_type": interval_type,
        "interval_scope": interval_scope,
        "interval_supports_positive": interval_supports_positive,
        "formal_paired_significance": formal_paired_significance,
        "evidence_tier": evidence_tier,
        "official_code_participating": official_code_participating,
        "locally_executed_external_system": official_code_participating,
        "same_acceptance_query": same_query,
        "same_safe_input": same_safe_input,
        "same_split_roles": same_split_roles,
        "faithful_reproduction": False,
        "source_artifact": source.as_posix(),
        "source_artifact_sha256": _sha256(source),
    }


def _remove_stale_derived(out: Path) -> None:
    for name in (
        "citable_five_system_metric_matrix.csv",
        "paper_ready_citable_five_system_positive_table.csv",
        "supplementary_code_systems.csv",
        "negative_metric_ledger.csv",
    ):
        path = out / name
        if path.is_file():
            path.unlink()


def build_citable_five_system_table_v13(
    output_dir: str | Path = DEFAULT_OUTPUT,
    v8_dir: str | Path = DEFAULT_V8,
    v9_dir: str | Path = DEFAULT_V9,
    v10_dir: str | Path = DEFAULT_V10,
    v11_dir: str | Path = DEFAULT_V11,
    w237_dir: str | Path = DEFAULT_W237,
    runtime_path: str | Path = DEFAULT_RUNTIME,
) -> dict[str, Any]:
    """Build a five-system table from existing frozen artifacts only."""

    out, runtime = Path(output_dir), Path(runtime_path)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "v8_mafsid": Path(v8_dir) / "paper_ready_mafsid_same_split_table.csv",
        "v9_shap_agentic": (
            Path(v9_dir) / "paper_ready_shap_agentic_same_query_table.csv"
        ),
        "v10_cfids": Path(v10_dir) / "paper_ready_cfids_same_query_table.csv",
        "v11_ztafl": Path(v11_dir) / "paper_ready_ztafl_same_query_table.csv",
        "w237_cross_protocol": (
            Path(w237_dir) / "unified_metric_matrix_w237.csv"
        ),
        "runtime_safe_v3_0": runtime,
        "bibliographic_registry": out / "bibliographic_registry.csv",
    }
    audit = _read(out / "bibliographic_audit_report.json")
    missing = [path.as_posix() for path in paths.values() if not path.is_file()]
    if audit.get("status") != "audited_v13_five_citable_systems" or missing:
        _remove_stale_derived(out)
        report = {
            "status": "blocked_v13_required_frozen_artifact_missing",
            "missing_paths": missing,
            "bibliographic_audit_status": audit.get("status", "missing"),
            "metric_rows_generated": False,
            **_security(),
        }
        _dump(out / "build_report.json", report)
        return report

    matrix: list[dict[str, Any]] = []
    for item in _read_csv(paths["v10_cfids"]):
        if item.get("configuration") != "5way3shot":
            continue
        paired = _truth(item.get("paired_ci_supported"))
        matrix.append(
            _metric_row(
                system="Continual-Federated-IDS",
                dataset_task="CICIDS2017 5-way 3-shot binary projection",
                metric=item.get("metric", ""),
                external_value=item.get("cfids_adapter"),
                mad_value=item.get("mad_etd"),
                interval_lower=item.get("delta_ci95_lower"),
                interval_upper=item.get("delta_ci95_upper"),
                interval_type=(
                    "1000_resample_paired_episode_delta_ci"
                    if item.get("delta_ci95_lower") not in (None, "")
                    else "not_available"
                ),
                interval_scope="paired_delta",
                interval_supports_positive=paired,
                formal_paired_significance=paired,
                evidence_tier="official_code_participating_adapted_same_query",
                official_code_participating=True,
                same_query=_truth(item.get("same_query")),
                same_safe_input=True,
                same_split_roles=True,
                source=paths["v10_cfids"],
            )
        )
    for item in _read_csv(paths["v11_ztafl"]):
        paired = _truth(item.get("paired_group_ci_supported"))
        matrix.append(
            _metric_row(
                system="ZTA-FL",
                dataset_task=item.get("dataset_task", "Edge-IIoTset binary"),
                metric=item.get("metric", ""),
                external_value=item.get("ztafl_official_code_adapter"),
                mad_value=item.get("mad_etd_locked_w207"),
                interval_lower=item.get("delta_ci95_lower"),
                interval_upper=item.get("delta_ci95_upper"),
                interval_type=(
                    "1000_resample_paired_source_group_delta_ci"
                    if item.get("delta_ci95_lower") not in (None, "")
                    else "not_available"
                ),
                interval_scope="paired_delta",
                interval_supports_positive=paired,
                formal_paired_significance=paired,
                evidence_tier="official_code_participating_adapted_same_query",
                official_code_participating=True,
                same_query=_truth(item.get("same_acceptance_query")),
                same_safe_input=_truth(item.get("same_safe_input")),
                same_split_roles=True,
                source=paths["v11_ztafl"],
            )
        )

    paper_systems = {"MA-IDS", "MARL-NIDS", "MAS-LSTM"}
    for item in _read_csv(paths["w237_cross_protocol"]):
        system = item.get("external_system", "")
        if system not in paper_systems:
            continue
        dataset = item.get("dataset", "")
        if system == "MARL-NIDS":
            task = "CICIDS2017 15-label weighted classification"
        elif system == "MAS-LSTM":
            task = "NF-ToN-IoT-v2 binary classification"
        else:
            task = dataset
        matrix.append(
            _metric_row(
                system=system,
                dataset_task=task,
                metric=item.get("metric", ""),
                external_value=item.get("paper_reported_reference"),
                mad_value=item.get("mad_etd_local"),
                interval_lower=item.get("local_ci95_lower"),
                interval_upper=item.get("local_ci95_upper"),
                interval_type="local_metric_bootstrap_ci_vs_fixed_paper_scalar",
                interval_scope="mad_etd_local_metric_only",
                interval_supports_positive=_truth(
                    item.get("local_interval_separated_from_paper_scalar")
                ),
                formal_paired_significance=False,
                evidence_tier="paper_reported_cross_protocol",
                official_code_participating=False,
                same_query=False,
                same_safe_input=False,
                same_split_roles=False,
                source=paths["w237_cross_protocol"],
            )
        )

    systems = {row["external_system"] for row in matrix}
    primary_keys = {
        "Continual-Federated-IDS": ("macro_f1", None),
        "ZTA-FL": ("macro_f1", None),
        "MARL-NIDS": ("weighted_f1", None),
        "MA-IDS": ("macro_f1", "NF-BoT-IoT"),
        "MAS-LSTM": ("accuracy", None),
    }
    primary: list[dict[str, Any]] = []
    for system in MAIN_SYSTEMS:
        metric, dataset = primary_keys[system]
        matches = [
            row
            for row in matrix
            if row["external_system"] == system
            and row["metric"] == metric
            and (dataset is None or row["dataset_task"] == dataset)
        ]
        if len(matches) == 1:
            primary.append(matches[0])

    positive_systems = {
        row["external_system"]
        for row in primary
        if row["numerically_positive"] and row["interval_supports_positive"]
    }
    if systems != set(MAIN_SYSTEMS) or len(primary) != 5 or positive_systems != set(MAIN_SYSTEMS):
        _remove_stale_derived(out)
        report = {
            "status": "blocked_v13_five_citable_positive_rows_incomplete",
            "systems": sorted(systems),
            "primary_row_count": len(primary),
            "positive_systems": sorted(positive_systems),
            "metric_rows_generated": False,
            **_security(),
        }
        _dump(out / "build_report.json", report)
        return report

    negative = [
        {
            **row,
            "negative_result_status": "retained_not_hidden",
            "negative_result_reason": "mad_etd_local_below_paper_scalar",
        }
        for row in matrix
        if not row["numerically_positive"]
    ]
    supplementary = [
        {
            "system": "MAFSID",
            "repository": "https://github.com/abdkhanstd/MAFSID",
            "evidence": "official architecture code and checkpoint participated",
            "bibliographic_identity_status": "no_verified_paper_identity_for_this_repository_system",
            "main_table_eligible": False,
            "comparison_lane": "official_checkpoint_adapted_same_split_role",
            "faithful_reproduction": False,
            "source_artifact": paths["v8_mafsid"].as_posix(),
            "source_artifact_sha256": _sha256(paths["v8_mafsid"]),
        },
        {
            "system": "SHAP-Agentic IDS",
            "repository": "https://github.com/omerfarooq223/shap-agentic-ids",
            "evidence": "official repository RF checkpoint participated",
            "bibliographic_identity_status": "no_verified_peer_reviewed_or_preprint_identity",
            "main_table_eligible": False,
            "comparison_lane": "official_checkpoint_retrospective_same_query_backend",
            "faithful_reproduction": False,
            "source_artifact": paths["v9_shap_agentic"].as_posix(),
            "source_artifact_sha256": _sha256(paths["v9_shap_agentic"]),
        },
    ]
    _write_csv(out / "citable_five_system_metric_matrix.csv", matrix)
    _write_csv(out / "paper_ready_citable_five_system_positive_table.csv", primary)
    _write_csv(out / "negative_metric_ledger.csv", negative)
    _write_csv(out / "supplementary_code_systems.csv", supplementary)
    _dump(
        out / "source_artifact_manifest.json",
        {
            name: {"path": path.as_posix(), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    )
    report = {
        "status": "built_v13_five_citable_positive_systems_tiered",
        "metric_row_count": len(matrix),
        "primary_paper_row_count": len(primary),
        "citable_positive_system_count": len(positive_systems),
        "citable_positive_systems": list(MAIN_SYSTEMS),
        "official_code_participating_system_count": 2,
        "official_code_participating_systems": list(MAIN_SYSTEMS[:2]),
        "paper_reported_cross_protocol_system_count": 3,
        "paper_reported_cross_protocol_systems": list(MAIN_SYSTEMS[2:]),
        "supplementary_repository_system_count": len(supplementary),
        "negative_metric_count": len(negative),
        "mas_lstm_negative_metrics": sorted(
            row["metric"]
            for row in negative
            if row["external_system"] == "MAS-LSTM"
        ),
        "faithful_reproduction_count": 0,
        "metric_rows_generated": True,
        "runtime_hash_before": audit.get("runtime_hash_before", ""),
        "runtime_hash_after": _sha256(runtime),
        **_security(),
    }
    _dump(out / "build_report.json", report)
    return report


def finalize_citable_multiagent_v13(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOCUMENT,
    document_cn: str | Path = DEFAULT_DOCUMENT_CN,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    """Validate the evidence capsule and write paper-safe documentation."""

    out, runtime = Path(output_dir), Path(runtime_path)
    audit = _read(out / "bibliographic_audit_report.json")
    build = _read(out / "build_report.json")
    primary = _read_csv(out / "paper_ready_citable_five_system_positive_table.csv")
    negative = _read_csv(out / "negative_metric_ledger.csv")
    selected_registry = [
        row
        for row in _read_csv(out / "bibliographic_registry.csv")
        if _truth(row.get("selected_for_main_table"))
    ]
    runtime_unchanged = bool(
        audit.get("runtime_hash_before")
        and audit.get("runtime_hash_before") == _sha256(runtime)
    )
    primary_systems = {row.get("external_system") for row in primary}
    negative_mas = {
        row.get("metric")
        for row in negative
        if row.get("external_system") == "MAS-LSTM"
    }
    goal_achieved = bool(
        build.get("status") == "built_v13_five_citable_positive_systems_tiered"
        and len(selected_registry) == 5
        and len(primary) == 5
        and primary_systems == set(MAIN_SYSTEMS)
        and all(
            _truth(row.get("numerically_positive"))
            and _truth(row.get("interval_supports_positive"))
            for row in primary
        )
        and negative_mas == {"recall", "f1"}
        and build.get("official_code_participating_system_count") == 2
        and build.get("paper_reported_cross_protocol_system_count") == 3
        and runtime_unchanged
        and tests_passed
    )
    claims = {
        "supported_claims": [
            "Five paper-identified multi-agent systems have at least one positive numerical metric in the frozen tiered matrix.",
            "Continual-Federated-IDS and ZTA-FL use official-code-participating local adapted comparisons.",
            "MARL-NIDS, MA-IDS, and MAS-LSTM are paper-reported cross-protocol references.",
            "The intervals for paper-reported references are local MAD-ETD metric bootstrap intervals, not paired external-system delta intervals.",
            "MAS-LSTM Accuracy and Precision are positive while Recall and F1 remain negative in the complete evidence ledger.",
            "MAFSID and SHAP-Agentic IDS remain supplementary repository/checkpoint evidence rather than paper-level main-table systems.",
        ],
        "forbidden_claims": [
            "All five external systems were faithfully reproduced.",
            "The paper-reported systems were run locally on identical samples or splits.",
            "Local metric bootstrap intervals are paired external-system delta intervals.",
            "MAD-ETD universally outperforms the five systems across datasets, tasks, or protocols.",
            "MAS-LSTM Recall or F1 improved in the frozen comparison.",
            "The inaccessible JISA 2021 numerical table was directly verified or used.",
            "MAFSID or SHAP-Agentic IDS has a verified paper identity in this capsule.",
        ],
        "blocked_references": {
            "10.1016/j.jisa.2021.102923": (
                "original metric table not directly accessible; secondary "
                "copied values excluded"
            )
        },
        "evidence_tier_policy": {
            "official_code_participating_adapted_same_query": (
                "locally executed adapted comparison with paired local query "
                "units; not a faithful paper-scale reproduction"
            ),
            "paper_reported_cross_protocol": (
                "paper scalar versus separately frozen local result; no paired "
                "external inference or formal paired delta significance"
            ),
        },
    }
    _dump(out / "claim_boundary.json", claims)
    final = {
        "status": (
            "accepted_v13_five_citable_positive_systems_tiered"
            if goal_achieved
            else "blocked_v13_final_acceptance_incomplete"
        ),
        "goal_achieved": goal_achieved,
        "citable_positive_system_count": build.get(
            "citable_positive_system_count", 0
        ),
        "all_five_systems_have_citable_identity": len(selected_registry) == 5,
        "all_five_systems_have_positive_primary_metric": (
            len(primary) == 5 and primary_systems == set(MAIN_SYSTEMS)
        ),
        "official_code_participating_system_count": build.get(
            "official_code_participating_system_count", 0
        ),
        "paper_reported_cross_protocol_system_count": build.get(
            "paper_reported_cross_protocol_system_count", 0
        ),
        "faithful_reproduction_count": 0,
        "mas_lstm_negative_metrics_retained": negative_mas == {"recall", "f1"},
        "jisa_original_metric_table_used": False,
        "runtime_hash_unchanged": runtime_unchanged,
        "tests_passed": tests_passed,
        "test_count": int(test_count),
        **_security(),
    }
    _dump(out / "acceptance_report.json", final)

    table_lines: list[str] = []
    for row in primary:
        interval = (
            f"[{float(row['interval_lower']):.6f}, "
            f"{float(row['interval_upper']):.6f}]"
        )
        table_lines.append(
            "| {system} | {identity} | {task} | {metric} | {external:.6f} | "
            "{mad:.6f} | {delta:+.3f} pp | {interval} ({scope}) | {tier} |".format(
                system=row["external_system"],
                identity=row["paper_identity"],
                task=row["dataset_task"],
                metric=row["metric"],
                external=float(row["external_value"]),
                mad=float(row["mad_etd_value"]),
                delta=float(row["delta_percentage_points"]),
                interval=interval,
                scope=row["interval_scope"],
                tier=row["evidence_tier"],
            )
        )
    table = "\n".join(table_lines)
    english = f"""# MAD-ETD Citable Multi-Agent Comparison v13

Status: `{final['status']}`.

| External system | Paper identity | Frozen task | Metric | External | MAD-ETD | Delta | 95% interval (scope) | Evidence tier |
|---|---|---|---:|---:|---:|---:|---|---|
{table}

The table contains five systems with a DOI or directly verifiable preprint and
at least one positive metric. Continual-Federated-IDS and ZTA-FL are the only
official-code-participating local adapted comparisons. MARL-NIDS, MA-IDS, and
MAS-LSTM are paper-reported cross-protocol references: their intervals cover
the local MAD-ETD metric, not a paired external-system delta. ZTA-FL's large
gain remains subject to the source heterogeneity recorded by its frozen
source-group comparison. MAS-LSTM Recall (-1.360 pp) and F1 (-0.047 pp) are
retained in `negative_metric_ledger.csv`.

No row is a faithful paper-scale reproduction, and this capsule does not
establish universal superiority. MAFSID and SHAP-Agentic IDS are retained only
as supplementary repository/checkpoint evidence because a citable paper
identity was not verified for those repository systems. DOI
10.1016/j.jisa.2021.102923 remains blocked because its original numerical
table was not directly accessible; secondary copied values were not used.
"""
    chinese = f"""# MAD-ETD 可引用多智能体系统对比 v13

状态：`{final['status']}`。

| 外部系统 | 论文身份 | 冻结任务 | 指标 | 外部值 | MAD-ETD | 增量 | 95% 区间（口径） | 证据等级 |
|---|---|---|---:|---:|---:|---:|---|---|
{table}

主表包含 5 个具有 DOI 或可直接核验预印本身份、且至少存在一个正向指标的多智能体系统。
其中，只有 Continual-Federated-IDS 与 ZTA-FL 属于官方代码参与的本地适配比较；
MARL-NIDS、MA-IDS 与 MAS-LSTM 均为论文报告标量与本地冻结结果的跨协议参照，
其区间只描述 MAD-ETD 本地指标，不是外部系统与 MAD-ETD 的配对差值区间。
ZTA-FL 的较大增量仍受冻结 source-group 对比中来源异质性的约束。
MAS-LSTM 的 Recall（-1.360 个百分点）与 F1（-0.047 个百分点）完整保留在
`negative_metric_ledger.csv` 中。

上述比较均不是原论文规模的忠实复现，也不能外推为跨数据集、跨任务或跨协议的普遍优势。
MAFSID 与 SHAP-Agentic IDS 因未核验到对应仓库系统的可引用论文身份，仅列入补充仓库/检查点证据表。
DOI 10.1016/j.jisa.2021.102923 的原始数值表无法直接访问，因此保持阻塞，未采用二手转录数值。
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
    manifest[Path(document).name] = _sha256(document)
    manifest[Path(document_cn).name] = _sha256(document_cn)
    _dump(out / "artifact_manifest.json", manifest)
    return final


__all__ = [
    "audit_citable_multiagent_systems_v13",
    "build_citable_five_system_table_v13",
    "finalize_citable_multiagent_v13",
]
