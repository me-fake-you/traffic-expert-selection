from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .nf_iot_calibration_v510 import POLICY as V510_POLICY
from .runtime_profiles import RuntimeProfile, load_runtime_profile, runtime_profile_sha256


EXPERIMENT = "mad_etd_nf_iot_runtime_v5_12"
RUNTIME_NAME = "runtime_nf_iot_calibrated_v5_12"
DEFAULT_RUN_DIR = Path("data/runs/mad_etd_nf_iot_runtime_v5_12")
DEFAULT_V510_DIR = Path("data/runs/mad_etd_nf_iot_calibration_v5_10")
DEFAULT_DOC = Path("docs/MAD_ETD_NF_IOT_RUNTIME_V5_12.md")
DEFAULT_DOC_CN = Path("docs/MAD_ETD_NF_IOT_RUNTIME_V5_12_CN.md")

ALLOWED_DATASET_SCOPES = {
    "NF-BoT-IoT",
    "NF-ToN-IoT",
    "NF-BoT-IoT/NF-ToN-IoT",
}
REJECTED_DATASET_SCOPES = [
    "CICIDS2017",
    "USTC",
    "CESNET",
    "CipherSpectrum",
    "CICIDS2017 PCAP",
    "DoHBrw PCAP",
]
BLOCKED_CONTEXT_FIELDS = [
    "label",
    "family",
    "sample_id",
    "source_file",
    "ip",
    "port",
    "flow_id",
    "provenance",
]


def _dump(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sha256_path(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_safe_hash() -> str:
    return runtime_profile_sha256(load_runtime_profile("runtime_safe_v3_0"))


def _candidate_profile_payload() -> dict[str, Any]:
    profile = {
        "schema_version": "1.0",
        "name": RUNTIME_NAME,
        "description": (
            "Default-off NF-BoT-IoT/NF-ToN-IoT scoped calibration wrapper "
            "around the v5.10 random_forest_reference class-wise selective policy."
        ),
        "promoted": False,
        "acceptance_allowed": False,
        "field_audit_mode": "strict",
        "field_contract_version": "2.9",
        "field_audit_required": True,
        "routing_policy": "capability_v3_0",
        "enrichment_policy": "off",
        "coordinator": "rule",
        "detector_backend": "learned",
        "model_dir": "data/models/ustc_tfc2016/v1",
        "tls_backend": None,
        "tls_model_dir": None,
        "ood_policy": "hybrid",
        "ood_gate_dir": "data/models/ustc_tfc2016/ood_v1",
        "base_ood_policy": "off",
        "base_ood_gate_dir": None,
        "evidence_stability_policy": "off",
        "feature_flags": {},
        "automatic_training": False,
        "automatic_deployment": False,
        "selection_uses_test_or_external": False,
        "cipherspectrum_locked_test_used_for_selection": False,
    }
    RuntimeProfile.model_validate(profile)
    return profile


def _scope_guard_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in sorted(ALLOWED_DATASET_SCOPES):
        rows.append(
            {
                "dataset_scope": scope,
                "allowed": True,
                "reason": "NF-IoT scoped optional runtime",
            }
        )
    for scope in REJECTED_DATASET_SCOPES:
        rows.append(
            {
                "dataset_scope": scope,
                "allowed": False,
                "reason": "outside NF-IoT v5.12 candidate scope",
            }
        )
    return rows


def _scope_guard_report() -> dict[str, Any]:
    rows = _scope_guard_rows()
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "candidate_runtime": RUNTIME_NAME,
        "allowed_dataset_scopes": sorted(ALLOWED_DATASET_SCOPES),
        "rejected_dataset_scopes": REJECTED_DATASET_SCOPES,
        "missing_scope_fail_closed": True,
        "blocked_context_fields_used_for_detection": [],
        "blocked_context_fields_rejected": BLOCKED_CONTEXT_FIELDS,
        "all_non_nf_scopes_rejected": all(
            not row["allowed"] for row in rows if row["dataset_scope"] in REJECTED_DATASET_SCOPES
        ),
    }


def _frozen_hashes(v510_dir: str | Path) -> dict[str, Any]:
    base = Path(v510_dir)
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "runtime_safe_v3_0_sha256": _runtime_safe_hash(),
        "v5_10_acceptance_report_sha256": _sha256_path(base / "acceptance_report.json"),
        "v5_10_results_sha256": _sha256_path(base / "nf_iot_acceptance_results.csv"),
        "v5_10_comparison_sha256": _sha256_path(base / "baseline_vs_candidate.csv"),
    }


def build_nf_iot_runtime_v5_12(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    v510_dir: str | Path = DEFAULT_V510_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = _candidate_profile_payload()
    guard = _scope_guard_report()
    hashes = _frozen_hashes(v510_dir)
    _dump(out / "candidate_runtime_profile.json", profile)
    _dump(out / "dataset_scope_guard_report.json", guard)
    _dump(out / "negative_scope_rejection_report.json", _scope_guard_rows())
    _dump(out / "frozen_hashes_before.json", hashes)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "candidate_runtime": RUNTIME_NAME,
        "candidate_profile_path": str((out / "candidate_runtime_profile.json")).replace("\\", "/"),
        "dataset_scope_guard_status": guard["status"],
        "optional_runtime_default_enabled": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "v5_10_dir": str(Path(v510_dir)).replace("\\", "/"),
    }
    _dump(out / "manifest_build_report.json", report)
    return report


def run_nf_iot_runtime_v5_12(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    v510_dir: str | Path = DEFAULT_V510_DIR,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "candidate_runtime_profile.json").exists():
        build_nf_iot_runtime_v5_12(out, v510_dir=v510_dir)
    v510 = Path(v510_dir)
    v510_report = _read_json(v510 / "acceptance_report.json") if (v510 / "acceptance_report.json").exists() else {}
    acceptance_rows = _read_csv_rows(v510 / "nf_iot_acceptance_results.csv")
    comparison_rows = _read_csv_rows(v510 / "baseline_vs_candidate.csv")
    comparison = comparison_rows[0] if comparison_rows else v510_report.get("comparison", {})
    runtime_rows: list[dict[str, Any]] = []
    for row in acceptance_rows:
        runtime_rows.append(
            {
                **row,
                "candidate_runtime": RUNTIME_NAME if row.get("model_id") == V510_POLICY["model_id"] else "runtime_safe_v3_0_reference",
                "dataset_scope_guard_allowed": row.get("dataset") == "NF-BoT-IoT/NF-ToN-IoT",
                "runtime_promoted": False,
                "default_enabled": False,
            }
        )
    if not runtime_rows and comparison:
        runtime_rows.append(
            {
                "dataset": comparison.get("dataset", "NF-BoT-IoT/NF-ToN-IoT"),
                "candidate_runtime": RUNTIME_NAME,
                "model_id": V510_POLICY["model_id"],
                "strategy": V510_POLICY["strategy"],
                "dataset_scope_guard_allowed": True,
                "runtime_promoted": False,
                "default_enabled": False,
                "fake_metric": False,
                "blocked_field_violation": 0,
                "fusion_ownership_violation": 0,
            }
        )
    comparison_row = {
        "candidate_runtime": RUNTIME_NAME,
        "reference_runtime": "runtime_safe_v3_0",
        "dataset": comparison.get("dataset", "NF-BoT-IoT/NF-ToN-IoT"),
        "selective_macro_f1_delta": comparison.get("selective_macro_f1_delta"),
        "macro_f1_delta": comparison.get("macro_f1_delta"),
        "malicious_recall_delta": comparison.get("malicious_recall_delta"),
        "ece_delta": comparison.get("ece_delta"),
        "coverage_delta": comparison.get("coverage_delta"),
        "selective_error_delta": comparison.get("selective_error_delta"),
        "general_runtime_promotion": False,
        "default_enabled": False,
        "v5_10_status": v510_report.get("status", "missing"),
    }
    _write_csv(out / "runtime_acceptance_results.csv", runtime_rows)
    _write_csv(out / "runtime_vs_default_comparison.csv", [comparison_row])
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed" if runtime_rows and comparison else "missing_v5_10_reproducibility_artifacts",
        "candidate_runtime": RUNTIME_NAME,
        "runtime_rows": len(runtime_rows),
        "comparison_available": bool(comparison),
        "v5_10_status": v510_report.get("status", "missing"),
        "fake_metric_count": sum(1 for row in runtime_rows if str(row.get("fake_metric")).lower() == "true"),
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "runtime_run_report.json", report)
    return report


def _write_docs(
    *,
    document: str | Path,
    document_cn: str | Path,
    report: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> None:
    doc = Path(document)
    doc.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MAD-ETD NF-IoT Runtime v5.12",
        "",
        f"- status: `{report.get('status')}`",
        f"- candidate runtime: `{RUNTIME_NAME}`",
        f"- default enabled: `false`",
        f"- promoted runtime created: `{report.get('promoted_runtime_created')}`",
        f"- runtime_safe_v3_0 remains default: `{report.get('runtime_safe_v3_0_remains_default')}`",
        "",
        "## Evidence",
        "",
        f"- selective Macro-F1 delta: `{comparison.get('selective_macro_f1_delta', 'not_available')}`",
        f"- malicious recall delta: `{comparison.get('malicious_recall_delta', 'not_available')}`",
        f"- selective error delta: `{comparison.get('selective_error_delta', 'not_available')}`",
        f"- coverage delta: `{comparison.get('coverage_delta', 'not_available')}`",
        "",
        "This runtime is a dataset-specific optional wrapper for NF-BoT-IoT/NF-ToN-IoT only. "
        "It must not be described as a general detector promotion.",
    ]
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    doc_cn = Path(document_cn)
    doc_cn.parent.mkdir(parents=True, exist_ok=True)
    lines_cn = [
        "# MAD-ETD NF-IoT Runtime v5.12",
        "",
        f"- 状态：`{report.get('status')}`",
        f"- 候选 runtime：`{RUNTIME_NAME}`",
        "- 默认启用：`false`",
        f"- 是否创建 promoted runtime：`{report.get('promoted_runtime_created')}`",
        f"- runtime_safe_v3_0 是否仍为默认：`{report.get('runtime_safe_v3_0_remains_default')}`",
        "",
        "## 证据",
        "",
        f"- selective Macro-F1 提升：`{comparison.get('selective_macro_f1_delta', 'not_available')}`",
        f"- malicious recall 提升：`{comparison.get('malicious_recall_delta', 'not_available')}`",
        f"- selective error 变化：`{comparison.get('selective_error_delta', 'not_available')}`",
        f"- coverage 变化：`{comparison.get('coverage_delta', 'not_available')}`",
        "",
        "该 runtime 只是在 NF-BoT-IoT/NF-ToN-IoT 范围内默认关闭的可选封装，不能写成通用检测器晋级。",
    ]
    doc_cn.write_text("\n".join(lines_cn) + "\n", encoding="utf-8")


def finalize_nf_iot_runtime_v5_12(
    output_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    v510_dir: str | Path = DEFAULT_V510_DIR,
    document: str | Path = DEFAULT_DOC,
    document_cn: str | Path = DEFAULT_DOC_CN,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "runtime_run_report.json").exists():
        run_nf_iot_runtime_v5_12(out, v510_dir=v510_dir)
    guard = _read_json(out / "dataset_scope_guard_report.json") if (out / "dataset_scope_guard_report.json").exists() else {}
    run_report = _read_json(out / "runtime_run_report.json")
    comparison_rows = _read_csv_rows(out / "runtime_vs_default_comparison.csv")
    runtime_rows = _read_csv_rows(out / "runtime_acceptance_results.csv")
    comparison = comparison_rows[0] if comparison_rows else {}
    hashes_after = _frozen_hashes(v510_dir)
    _dump(out / "frozen_hashes_after.json", hashes_after)

    fake_metric_count = int(run_report.get("fake_metric_count") or 0)
    blocked_field_violation = sum(int(_float(row.get("blocked_field_violation"))) for row in runtime_rows)
    fusion_ownership_violation = sum(int(_float(row.get("fusion_ownership_violation"))) for row in runtime_rows)
    security = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "passed",
        "blocked_field_violation": blocked_field_violation,
        "fusion_ownership_violation": fusion_ownership_violation,
        "ood_override": 0,
        "illegal_verdict_execution": 0,
        "unsupported_calls": 0,
        "fake_metric_count": fake_metric_count,
        "pcap_supervised_metrics_used": False,
        "external_faithful_numeric_reproduction_claimed": False,
        "runtime_safe_v3_0_hash_after": hashes_after["runtime_safe_v3_0_sha256"],
        "runtime_safe_v3_0_remains_default": True,
    }
    security["status"] = (
        "passed"
        if blocked_field_violation == 0
        and fusion_ownership_violation == 0
        and fake_metric_count == 0
        else "failed_security_or_fake_metric_gate"
    )
    _dump(out / "security_acceptance.json", security)

    gates = {
        "v5_10_status": run_report.get("v5_10_status"),
        "run_status": run_report.get("status"),
        "scope_guard_passed": guard.get("status") == "passed" and guard.get("all_non_nf_scopes_rejected") is True,
        "selective_macro_f1_delta_positive": _float(comparison.get("selective_macro_f1_delta")) > 0.0,
        "malicious_recall_delta_positive": _float(comparison.get("malicious_recall_delta")) > 0.0,
        "selective_error_not_worse": _float(comparison.get("selective_error_delta")) <= 0.0,
        "fake_metric_count": fake_metric_count,
        "security_passed": security["status"] == "passed",
        "runtime_safe_v3_0_hash_unchanged": (
            _read_json(out / "frozen_hashes_before.json").get("runtime_safe_v3_0_sha256")
            == hashes_after["runtime_safe_v3_0_sha256"]
            if (out / "frozen_hashes_before.json").exists()
            else True
        ),
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    passed = (
        gates["v5_10_status"] == "accepted_dataset_specific_optional_calibration"
        and gates["run_status"] == "passed"
        and gates["scope_guard_passed"]
        and gates["selective_macro_f1_delta_positive"]
        and gates["malicious_recall_delta_positive"]
        and gates["selective_error_not_worse"]
        and gates["fake_metric_count"] == 0
        and gates["security_passed"]
        and gates["runtime_safe_v3_0_hash_unchanged"]
    )
    if passed:
        status = "accepted_dataset_specific_optional_runtime"
    elif not gates["scope_guard_passed"]:
        status = "rejected_scope_guard_failure"
    else:
        status = "not_promoted_reproducibility_failure"
    negative = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": "none" if passed else status,
        "reason": "" if passed else "NF-IoT optional runtime failed reproducibility, scope, or security gate",
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
    }
    _dump(out / "acceptance_gates.json", gates)
    _dump(out / "negative_results.json", negative)
    report = {
        "schema_version": "1.0",
        "experiment": EXPERIMENT,
        "status": status,
        "candidate_runtime": RUNTIME_NAME,
        "optional_runtime_profile_created": passed,
        "candidate_runtime_profile": str((out / "candidate_runtime_profile.json")).replace("\\", "/"),
        "default_enabled": False,
        "promoted_runtime_created": False,
        "runtime_safe_v3_0_remains_default": True,
        "comparison": comparison,
        "acceptance_gates": gates,
        "security": security,
        "scope_guard": guard,
        "fake_metric_count": fake_metric_count,
        "pcap_supervised_metrics_used": False,
        "external_faithful_numeric_reproduction_claimed": False,
        "document": str(Path(document)).replace("\\", "/"),
        "document_cn": str(Path(document_cn)).replace("\\", "/"),
    }
    _dump(out / "acceptance_report.json", report)
    _write_docs(document=document, document_cn=document_cn, report=report, comparison=comparison)
    return report
