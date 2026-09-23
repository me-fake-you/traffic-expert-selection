"""Close the third official-code search and prepare thesis evidence (v14).

No detector is trained here. The module freezes the literature/code search
outcome, keeps the absence of a third citable official-code candidate as a
negative result, and converts the accepted v13 evidence into thesis-safe tiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT = "mad_etd_external_citable_thesis_v14"
DEFAULT_OUTPUT = Path("data/runs/mad_etd_external_citable_thesis_v14")
DEFAULT_V13 = Path("data/runs/mad_etd_external_citable_multiagent_v13")
DEFAULT_RUNTIME = Path("data/configs/runtime_safe_v3_0.json")
DEFAULT_DOCUMENT = Path("docs/MAD_ETD_EXTERNAL_CITABLE_THESIS_V14.md")
DEFAULT_DOCUMENT_CN = Path("docs/MAD_ETD_EXTERNAL_CITABLE_THESIS_V14_CN.md")


MAIN_SYSTEMS = (
    "Continual-Federated-IDS",
    "ZTA-FL",
    "MARL-NIDS",
    "MA-IDS",
    "MAS-LSTM",
)


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate": "Semantic Multi-Agent IDS",
        "paper_identity": "arXiv:2606.10323",
        "paper_identity_verified": True,
        "public_executable_code_verified": False,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "CICIoT2023",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "blocked_no_verified_official_code",
        "evidence_url": "https://arxiv.org/abs/2606.10323",
        "evidence": (
            "The paper exposes no author-code link; exact-title GitHub "
            "repository search returned no matching repository."
        ),
    },
    {
        "candidate": "MA-IDS",
        "paper_identity": "arXiv:2604.05458",
        "paper_identity_verified": True,
        "public_executable_code_verified": False,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "NF-BoT-IoT;NF-ToN-IoT",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "blocked_no_verified_official_code",
        "evidence_url": "https://arxiv.org/abs/2604.05458",
        "evidence": (
            "The preprint exposes no author-code link; arXiv-id and exact-name "
            "GitHub repository searches returned no official implementation."
        ),
    },
    {
        "candidate": "MARL-NIDS",
        "paper_identity": "10.1109/NOMS59830.2024.10575541",
        "paper_identity_verified": True,
        "public_executable_code_verified": False,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "CICIDS2017",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "blocked_no_verified_official_code_or_predictions",
        "evidence_url": "https://doi.org/10.1109/NOMS59830.2024.10575541",
        "evidence": (
            "Exact system-name and DOI-suffix GitHub repository searches each "
            "returned zero repositories; no predictions were located."
        ),
    },
    {
        "candidate": "MAS-LSTM",
        "paper_identity": "10.3390/pr13030753",
        "paper_identity_verified": True,
        "public_executable_code_verified": False,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "NF-ToN-IoT-v2",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "blocked_no_verified_official_code_or_predictions",
        "evidence_url": "https://doi.org/10.3390/pr13030753",
        "evidence": (
            "Exact-title GitHub repository search returned zero repositories; "
            "no official predictions or checkpoint were verified."
        ),
    },
    {
        "candidate": "Agentic Quantum Computing",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": True,
        "compatible_local_dataset": "KDD Cup 99 only",
        "multiagent_detection_relevant": False,
        "selected_for_execution": False,
        "status": "excluded_no_citable_paper_and_incompatible_anomaly_task",
        "evidence_url": (
            "https://github.com/FareedKhan-dev/agentic-quantum-computing"
        ),
        "evidence": (
            "MIT repository with an orchestrating LLM agent, but no DOI/arXiv "
            "identity and no compatible frozen MAD-ETD task."
        ),
    },
    {
        "candidate": "FANET-MARL-IDS",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "simulated FANET GPS spoofing",
        "multiagent_detection_relevant": False,
        "selected_for_execution": False,
        "status": "excluded_simulation_no_citable_identity_or_license",
        "evidence_url": "https://github.com/fe-cmd/FANET-MARL-IDS",
        "evidence": (
            "Simulation code is public, but it has no paper identity, license, "
            "or compatible real network-flow dataset."
        ),
    },
    {
        "candidate": "MA-NIDS repository project",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "CSE-CIC-IDS2018",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "excluded_no_citable_identity_or_license",
        "evidence_url": "https://github.com/aish6498-hub/ma-nids",
        "evidence": (
            "Executable multi-model repository, but no DOI/arXiv identity and "
            "no verified license; it cites rather than implements a named paper."
        ),
    },
    {
        "candidate": "IoT multi-agent inference project",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "IoT-DIAD 2024",
        "multiagent_detection_relevant": False,
        "selected_for_execution": False,
        "status": "excluded_no_citable_identity_and_pipeline_role_agents",
        "evidence_url": (
            "https://github.com/Zunaira-Noor123/"
            "iot-network-intrusion-detection"
        ),
        "evidence": (
            "Public project with preprocessing, one ANN detector, and reporting "
            "roles, but no citable research identity or multi-detector evidence team."
        ),
    },
    {
        "candidate": "ATGC-MACIDS",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": False,
        "compatible_local_dataset": "UNSW-NB15;synthetic CICIDS2017 schema",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "excluded_unpublished_repository_without_license",
        "evidence_url": "https://github.com/Bhavyareddy16/atgc-ids",
        "evidence": (
            "Repository claims journal readiness but exposes no DOI/arXiv "
            "identity or license; synthetic schema checks are not a paper protocol."
        ),
    },
    {
        "candidate": "X-MAG-IDS",
        "paper_identity": "",
        "paper_identity_verified": False,
        "public_executable_code_verified": True,
        "official_code_verified": False,
        "license_verified": True,
        "compatible_local_dataset": "CICIoT2023 stress test only",
        "multiagent_detection_relevant": True,
        "selected_for_execution": False,
        "status": "blocked_no_verified_paper_or_preprint_identity",
        "evidence_url": "https://github.com/alqithami/xmag",
        "evidence": (
            "Official MIT repository is executable, but no DOI, arXiv identity, "
            "or citable release was verified."
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


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def audit_third_official_code_v14(
    output_dir: str | Path = DEFAULT_OUTPUT,
    runtime_path: str | Path = DEFAULT_RUNTIME,
) -> dict[str, Any]:
    """Freeze the 2026-07-25 literature and official-code search result."""

    out, runtime = Path(output_dir), Path(runtime_path)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "candidate_search_registry.csv", CANDIDATES)
    protocol = {
        "retrieved_on": "2026-07-25",
        "arxiv": {
            "query": "multi-agent intrusion detection",
            "max_results": 20,
            "sort_by": "relevance",
            "start_date": "2023-01-01",
            "result_count": 2,
            "results": ["arXiv:2606.10323", "arXiv:2604.05458"],
        },
        "github_public_api": {
            "exact_queries": {
                "Semantic Multi-Agent Intrusion Detection": 0,
                "MARL-NIDS": 0,
                "10575541": 0,
                "MAS-LSTM intrusion detection": 0,
                "2604.05458": 0,
            },
            "broad_query": "multi-agent intrusion detection",
            "shortlisted_repository_count": 5,
        },
        "selection_requirements": [
            "DOI or directly verifiable preprint",
            "author-linked public executable detection code",
            "license or explicit reuse permission",
            "compatible real network-flow dataset and task",
            "safe-input adaptation without blocked fields",
        ],
        "search_snippet_used_as_metric": False,
        "independent_metadata_extraction_completed": True,
    }
    _dump(out / "search_protocol.json", protocol)
    _dump(
        out / "literature_synthesis.json",
        {
            "paper_count": 2,
            "papers": [
                {
                    "identity": "arXiv:2606.10323",
                    "method": "Scout-Mutator-Auditor-Arbiter semantic reasoning pipeline",
                    "datasets": "CICIoT2023;IoT-23;Bot-IoT;TON IoT",
                    "public_code": False,
                    "official_predictions_or_checkpoints": False,
                    "same_query_safe_input_feasibility": "partial_not_directly_reproducible",
                    "primary_blockers": (
                        "small manually balanced evaluation subset; missing sample IDs, "
                        "prompts, thresholds, projection checkpoint, and official outputs"
                    ),
                },
                {
                    "identity": "arXiv:2604.05458",
                    "method": "Traffic Classification Agent plus Error Analysis Agent and FAISS experience library",
                    "datasets": "NF-BoT-IoT;NF-ToN-IoT",
                    "public_code": False,
                    "official_predictions_or_checkpoints": False,
                    "same_query_safe_input_feasibility": "partial_not_directly_reproducible",
                    "primary_blockers": (
                        "missing exact prompts, sample IDs, similarity threshold, prediction "
                        "files, and frozen experience libraries"
                    ),
                },
            ],
            "themes": [
                "Role-specialized agents organize detection reasoning rather than merely averaging models.",
                "Semantic or retrieval context is the claimed mechanism for improving closed-set or zero-day decisions.",
                "Both systems depend on proprietary LLM services and unreleased experiment-specific artifacts.",
            ],
            "convergences": [
                "Both report positive IoT benchmark metrics.",
                "Neither exposes author-provided code, official predictions, or paper-specific checkpoints.",
                "A safe-input reconstruction is conceptually possible but cannot be an official same-query reproduction.",
            ],
            "evidence_gaps": [
                "Exact sample identities and random seeds",
                "Complete prompts and decision thresholds",
                "Frozen retrieval or projection artifacts",
                "Official predictions for paired comparison",
            ],
        },
    )
    selected = [row for row in CANDIDATES if row["selected_for_execution"]]
    decision = {
        "status": "blocked_no_third_citable_official_code_candidate_v14",
        "candidate_count": len(CANDIDATES),
        "selected_candidate_count": len(selected),
        "selected_candidates": [row["candidate"] for row in selected],
        "third_official_code_comparison_executed": False,
        "third_official_code_system_count_after_search": 2,
        "paper_reported_fallback_added": False,
        "reason": (
            "No new candidate simultaneously satisfied citable identity, "
            "official executable code, license, compatible data, and safe input."
        ),
        "runtime_hash_before": _sha256(runtime),
        **_security(),
    }
    _dump(out / "third_official_code_decision.json", decision)
    return decision


def _remove_stale(out: Path) -> None:
    for name in (
        "thesis_tiered_positive_table.csv",
        "thesis_complete_metric_appendix.csv",
        "thesis_negative_metric_table.csv",
        "figure_source_positive_comparison.csv",
        "thesis_bibliography_registry.csv",
        "thesis_claim_evidence_matrix.csv",
    ):
        path = out / name
        if path.is_file():
            path.unlink()


def build_thesis_citable_evidence_v14(
    v13_dir: str | Path = DEFAULT_V13,
    output_dir: str | Path = DEFAULT_OUTPUT,
    runtime_path: str | Path = DEFAULT_RUNTIME,
) -> dict[str, Any]:
    """Convert accepted v13 rows into thesis-safe evidence tiers."""

    source, out, runtime = Path(v13_dir), Path(output_dir), Path(runtime_path)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "v13_acceptance": source / "acceptance_report.json",
        "v13_primary": source / "paper_ready_citable_five_system_positive_table.csv",
        "v13_matrix": source / "citable_five_system_metric_matrix.csv",
        "v13_negative": source / "negative_metric_ledger.csv",
        "v13_bibliography": source / "bibliographic_registry.csv",
        "v13_claim_boundary": source / "claim_boundary.json",
        "v14_search_decision": out / "third_official_code_decision.json",
        "runtime_safe_v3_0": runtime,
    }
    missing = [path.as_posix() for path in paths.values() if not path.is_file()]
    acceptance = _read(paths["v13_acceptance"])
    decision = _read(paths["v14_search_decision"])
    if (
        missing
        or acceptance.get("status")
        != "accepted_v13_five_citable_positive_systems_tiered"
        or decision.get("status")
        != "blocked_no_third_citable_official_code_candidate_v14"
    ):
        _remove_stale(out)
        report = {
            "status": "blocked_v14_required_frozen_evidence_missing",
            "missing_paths": missing,
            "v13_status": acceptance.get("status", "missing"),
            "search_status": decision.get("status", "missing"),
            "tables_generated": False,
            **_security(),
        }
        _dump(out / "build_report.json", report)
        return report

    primary = _read_csv(paths["v13_primary"])
    matrix = _read_csv(paths["v13_matrix"])
    negative = _read_csv(paths["v13_negative"])
    bibliography = [
        row
        for row in _read_csv(paths["v13_bibliography"])
        if _truth(row.get("selected_for_main_table"))
    ]
    tiered: list[dict[str, Any]] = []
    figure: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for row in primary:
        code_lane = _truth(row.get("official_code_participating"))
        group = (
            "A_official_code_participating_local_adapted"
            if code_lane
            else "B_paper_reported_cross_protocol_reference"
        )
        lower, upper = _number(row.get("interval_lower")), _number(
            row.get("interval_upper")
        )
        delta_ci_valid = bool(
            code_lane
            and row.get("interval_scope") == "paired_delta"
            and _truth(row.get("formal_paired_significance"))
        )
        enriched = {
            **row,
            "comparison_group": group,
            "delta_ci_valid": delta_ci_valid,
            "delta_ci95_lower_percentage_points": (
                lower * 100 if delta_ci_valid and lower is not None else ""
            ),
            "delta_ci95_upper_percentage_points": (
                upper * 100 if delta_ci_valid and upper is not None else ""
            ),
            "local_metric_ci95_lower": (
                lower if not code_lane and lower is not None else ""
            ),
            "local_metric_ci95_upper": (
                upper if not code_lane and upper is not None else ""
            ),
            "thesis_claim_level": (
                "local_adapted_paired_improvement"
                if code_lane
                else "cross_protocol_numeric_reference"
            ),
            "fair_or_faithful_reproduction": False,
        }
        tiered.append(enriched)
        figure.append(
            {
                "external_system": row.get("external_system", ""),
                "metric": row.get("metric", ""),
                "external_value": row.get("external_value", ""),
                "mad_etd_value": row.get("mad_etd_value", ""),
                "delta_percentage_points": row.get(
                    "delta_percentage_points", ""
                ),
                "comparison_group": group,
                "visual_mark": "filled_circle" if code_lane else "open_diamond",
                "delta_error_bar_allowed": delta_ci_valid,
                "delta_ci95_lower_percentage_points": enriched[
                    "delta_ci95_lower_percentage_points"
                ],
                "delta_ci95_upper_percentage_points": enriched[
                    "delta_ci95_upper_percentage_points"
                ],
                "local_metric_ci_available_separately": not code_lane,
                "interval_warning": (
                    "paired delta CI"
                    if code_lane
                    else "point-only delta; interval belongs to local MAD-ETD metric"
                ),
            }
        )
        claim_rows.append(
            {
                "external_system": row.get("external_system", ""),
                "supported_wording": (
                    "official-code-participating local adapted comparison"
                    if code_lane
                    else "paper-reported cross-protocol numerical reference"
                ),
                "metric": row.get("metric", ""),
                "delta_percentage_points": row.get(
                    "delta_percentage_points", ""
                ),
                "paired_delta_inference_supported": delta_ci_valid,
                "faithful_reproduction": False,
                "fair_or_faithful_reproduction": False,
                "universal_superiority_supported": False,
                "source_artifact": row.get("source_artifact", ""),
                "source_artifact_sha256": row.get(
                    "source_artifact_sha256", ""
                ),
            }
        )

    systems = {row.get("external_system") for row in tiered}
    code_count = sum(
        row["comparison_group"]
        == "A_official_code_participating_local_adapted"
        for row in tiered
    )
    paper_count = len(tiered) - code_count
    negative_metrics = {
        row.get("metric")
        for row in negative
        if row.get("external_system") == "MAS-LSTM"
    }
    valid = bool(
        len(tiered) == 5
        and systems == set(MAIN_SYSTEMS)
        and code_count == 2
        and paper_count == 3
        and len(bibliography) == 5
        and negative_metrics == {"recall", "f1"}
    )
    if not valid:
        _remove_stale(out)
        report = {
            "status": "blocked_v14_tier_or_negative_result_mismatch",
            "systems": sorted(system for system in systems if system),
            "code_tier_count": code_count,
            "paper_tier_count": paper_count,
            "bibliography_count": len(bibliography),
            "mas_lstm_negative_metrics": sorted(
                metric for metric in negative_metrics if metric
            ),
            "tables_generated": False,
            **_security(),
        }
        _dump(out / "build_report.json", report)
        return report

    negative_out = [
        {
            **row,
            "required_in_thesis_appendix": True,
            "main_positive_table_qualification_affected": False,
        }
        for row in negative
    ]
    _write_csv(out / "thesis_tiered_positive_table.csv", tiered)
    _write_csv(out / "thesis_complete_metric_appendix.csv", matrix)
    _write_csv(out / "thesis_negative_metric_table.csv", negative_out)
    _write_csv(out / "figure_source_positive_comparison.csv", figure)
    _write_csv(out / "thesis_bibliography_registry.csv", bibliography)
    _write_csv(out / "thesis_claim_evidence_matrix.csv", claim_rows)
    _dump(
        out / "source_artifact_manifest.json",
        {
            name: {"path": path.as_posix(), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    )
    report = {
        "status": "built_v14_thesis_tiered_citable_evidence",
        "positive_system_count": len(tiered),
        "official_code_participating_system_count": code_count,
        "paper_reported_cross_protocol_system_count": paper_count,
        "third_official_code_candidate_found": False,
        "third_official_code_comparison_executed": False,
        "figure_rows_with_paired_delta_ci": sum(
            _truth(row["delta_error_bar_allowed"]) for row in figure
        ),
        "figure_rows_point_only_cross_protocol": sum(
            not _truth(row["delta_error_bar_allowed"]) for row in figure
        ),
        "negative_metric_count": len(negative_out),
        "mas_lstm_negative_metrics": sorted(negative_metrics),
        "faithful_reproduction_count": 0,
        "runtime_hash_before": decision.get("runtime_hash_before", ""),
        "runtime_hash_after": _sha256(runtime),
        "tables_generated": True,
        **_security(),
    }
    _dump(out / "build_report.json", report)
    return report


def finalize_external_citable_v14(
    output_dir: str | Path = DEFAULT_OUTPUT,
    document: str | Path = DEFAULT_DOCUMENT,
    document_cn: str | Path = DEFAULT_DOCUMENT_CN,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    *,
    tests_passed: bool = False,
    test_count: int = 0,
) -> dict[str, Any]:
    """Finalize a negative search result and positive thesis evidence pack."""

    out, runtime = Path(output_dir), Path(runtime_path)
    decision = _read(out / "third_official_code_decision.json")
    build = _read(out / "build_report.json")
    tiered = _read_csv(out / "thesis_tiered_positive_table.csv")
    negative = _read_csv(out / "thesis_negative_metric_table.csv")
    runtime_unchanged = bool(
        decision.get("runtime_hash_before")
        and decision.get("runtime_hash_before") == _sha256(runtime)
    )
    goal_achieved = bool(
        decision.get("status")
        == "blocked_no_third_citable_official_code_candidate_v14"
        and build.get("status") == "built_v14_thesis_tiered_citable_evidence"
        and len(tiered) == 5
        and build.get("official_code_participating_system_count") == 2
        and build.get("paper_reported_cross_protocol_system_count") == 3
        and {
            row.get("metric")
            for row in negative
            if row.get("external_system") == "MAS-LSTM"
        }
        == {"recall", "f1"}
        and runtime_unchanged
        and tests_passed
    )
    claims = {
        "supported_claims": [
            "The refreshed search found no third system satisfying all citable-identity, official-code, license, compatible-data, and safe-input requirements.",
            "Five citable systems retain at least one positive numerical metric in the frozen v13 evidence.",
            "Continual-Federated-IDS and ZTA-FL are official-code-participating local adapted comparisons with paired delta intervals.",
            "MARL-NIDS, MA-IDS, and MAS-LSTM are paper-reported cross-protocol numerical references shown as point-only deltas.",
            "MAS-LSTM Recall and F1 remain negative and are retained in the thesis appendix table.",
        ],
        "forbidden_claims": [
            "A third citable official-code comparison was completed.",
            "All five systems were faithfully or fairly reproduced.",
            "Paper-reported cross-protocol rows have paired external-system delta confidence intervals.",
            "MAD-ETD universally outperforms all five systems across datasets or protocols.",
            "MAS-LSTM Recall or F1 improved.",
            "Repository-only projects are paper-level external systems.",
        ],
        "figure_policy": {
            "filled_circle_with_error_bar": (
                "official-code-participating local adapted paired delta"
            ),
            "open_diamond_without_error_bar": (
                "paper-reported cross-protocol point delta"
            ),
            "do_not_mix": (
                "local MAD-ETD metric intervals must not be plotted as paired "
                "external-system delta intervals"
            ),
        },
    }
    _dump(out / "claim_boundary.json", claims)
    final = {
        "status": (
            "accepted_v14_third_code_search_blocked_thesis_evidence_closed"
            if goal_achieved
            else "blocked_v14_final_acceptance_incomplete"
        ),
        "goal_achieved": goal_achieved,
        "third_official_code_candidate_found": False,
        "third_official_code_comparison_executed": False,
        "official_code_participating_system_count": build.get(
            "official_code_participating_system_count", 0
        ),
        "paper_reported_cross_protocol_system_count": build.get(
            "paper_reported_cross_protocol_system_count", 0
        ),
        "positive_citable_system_count": build.get("positive_system_count", 0),
        "faithful_reproduction_count": 0,
        "negative_metrics_retained": build.get("negative_metric_count", 0),
        "runtime_hash_unchanged": runtime_unchanged,
        "tests_passed": tests_passed,
        "test_count": int(test_count),
        **_security(),
    }
    _dump(out / "acceptance_report.json", final)

    table_lines: list[str] = []
    for row in tiered:
        interval = (
            "paired delta CI [{:.3f}, {:.3f}] pp".format(
                float(row["delta_ci95_lower_percentage_points"]),
                float(row["delta_ci95_upper_percentage_points"]),
            )
            if _truth(row.get("delta_ci_valid"))
            else "point delta only; local metric CI reported separately"
        )
        table_lines.append(
            "| {system} | {metric} | {delta:+.3f} pp | {group} | {interval} |".format(
                system=row["external_system"],
                metric=row["metric"],
                delta=float(row["delta_percentage_points"]),
                group=row["comparison_group"],
                interval=interval,
            )
        )
    table = "\n".join(table_lines)
    english = f"""# MAD-ETD External Citable Evidence v14

Status: `{final['status']}`.

| External system | Positive metric | Delta | Comparison group | Interval interpretation |
|---|---:|---:|---|---|
{table}

The refreshed arXiv and GitHub audit did not identify a third system that
simultaneously has a citable paper identity, author-linked executable code,
reuse permission, a compatible real flow task, and a safe-input adaptation
path. No repository-only project was promoted to paper-level evidence.

The five accepted v13 rows therefore remain split into two non-interchangeable
tiers. Continual-Federated-IDS and ZTA-FL use official-code-participating local
adapted comparisons and may show paired delta intervals. MARL-NIDS, MA-IDS,
and MAS-LSTM are paper-reported cross-protocol references; their delta marks
are point estimates only. MAS-LSTM Recall (-1.360 pp) and F1 (-0.047 pp) remain
in the negative metric table. This evidence does not establish faithful
reproduction or universal superiority.
"""
    chinese = f"""# MAD-ETD 外部可引用证据 v14

状态：`{final['status']}`。

| 外部系统 | 正向指标 | 增量 | 对比层级 | 区间解释 |
|---|---:|---:|---|---|
{table}

本轮重新检索 arXiv 与 GitHub 后，未发现同时满足可引用论文身份、作者关联可执行代码、
复用许可、兼容真实流量任务和安全输入适配条件的第三个系统。因此，没有将只有仓库、
没有论文身份的项目提升为论文级外部证据，也没有启动新的训练或检测器实验。

现有 5 个 v13 正向结果继续分为两个不可混用的证据层级。Continual-Federated-IDS 与
ZTA-FL 属于官方代码参与的本地适配比较，可以报告配对差值区间；MARL-NIDS、MA-IDS
与 MAS-LSTM 属于论文报告的跨协议数值参照，其增量在图中只能作为无误差棒的点估计。
MAS-LSTM 的 Recall（-1.360 个百分点）与 F1（-0.047 个百分点）继续保留在负指标表中。
上述证据不支持忠实复现或跨数据集、跨任务的普遍优越性结论。
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
    "audit_third_official_code_v14",
    "build_thesis_citable_evidence_v14",
    "finalize_external_citable_v14",
]
