from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import FutureFeatureFlags


RUNTIME_PROFILE_NAMES = (
    "runtime_safe_v3_0",
    "runtime_safe_v2_9",
    "runtime_safe_v2_8",
    "runtime_safe_v2_7",
    "paper_v1_reproduction",
    "deep_v2_experimental",
    "runtime_tls_doh_v4",
    "runtime_tls_doh_w71_optional",
    "runtime_robust_temporal_v1_optional",
)


class RuntimeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    name: str
    description: str
    promoted: bool
    acceptance_allowed: bool
    field_audit_mode: Literal["strict", "legacy"]
    field_contract_version: Literal["2.8", "2.9"] = "2.8"
    field_audit_required: bool = True
    routing_policy: Literal[
        "legacy", "contract_v2_9", "capability_v3_0"
    ] = "legacy"
    enrichment_policy: Literal[
        "legacy_inline", "off", "post_fusion"
    ] = "legacy_inline"
    coordinator: Literal["rule", "nvidia"] = "rule"
    detector_backend: str
    model_dir: str | None = None
    tls_backend: str | None = None
    tls_model_dir: str | None = None
    ood_policy: str = "off"
    ood_gate_dir: str | None = None
    base_ood_policy: str = "off"
    base_ood_gate_dir: str | None = None
    evidence_stability_policy: str = "off"
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    automatic_training: bool = False
    automatic_deployment: bool = False
    selection_uses_test_or_external: bool = False
    cipherspectrum_locked_test_used_for_selection: bool = False

    @model_validator(mode="after")
    def validate_safety_contract(self) -> "RuntimeProfile":
        if self.acceptance_allowed and not self.promoted:
            raise ValueError("only promoted profiles may be acceptance eligible")
        if self.acceptance_allowed and (
            not self.field_audit_required or self.field_audit_mode != "strict"
        ):
            raise ValueError(
                "acceptance profiles require non-bypassable strict FieldAudit"
            )
        if self.automatic_training or self.automatic_deployment:
            raise ValueError("runtime profiles cannot train or deploy automatically")
        FutureFeatureFlags.model_validate(self.feature_flags)
        return self


def _profile_path(
    name_or_path: str | Path,
    *,
    config_dir: str | Path = "data/configs",
) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix == ".json" or candidate.parent != Path("."):
        return candidate
    if candidate.name not in RUNTIME_PROFILE_NAMES:
        raise ValueError(f"unknown runtime profile: {candidate.name}")
    return Path(config_dir) / f"{candidate.name}.json"


def load_runtime_profile(
    name_or_path: str | Path,
    *,
    config_dir: str | Path = "data/configs",
) -> RuntimeProfile:
    path = _profile_path(name_or_path, config_dir=config_dir)
    if not path.exists():
        raise FileNotFoundError(f"runtime profile not found: {path}")
    profile = RuntimeProfile.model_validate_json(path.read_text(encoding="utf-8"))
    if path.stem in RUNTIME_PROFILE_NAMES and profile.name != path.stem:
        raise ValueError("runtime profile name does not match its file name")
    return profile


def runtime_profile_sha256(profile: RuntimeProfile) -> str:
    payload = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_engine_kwargs(profile: RuntimeProfile) -> dict[str, Any]:
    return {
        "use_nvidia": profile.coordinator == "nvidia",
        "detector_backend": profile.detector_backend,
        "model_dir": profile.model_dir,
        "tls_backend": profile.tls_backend,
        "tls_model_dir": profile.tls_model_dir,
        "ood_policy": profile.ood_policy,
        "ood_gate_dir": profile.ood_gate_dir,
        "base_ood_policy": profile.base_ood_policy,
        "base_ood_gate_dir": profile.base_ood_gate_dir,
        "enable_field_audit": profile.field_audit_required,
        "field_audit_mode": profile.field_audit_mode,
        "field_contract_version": profile.field_contract_version,
        "routing_policy": profile.routing_policy,
        "enrichment_policy": profile.enrichment_policy,
        "evidence_stability_policy": profile.evidence_stability_policy,
        "future_flags": FutureFeatureFlags.model_validate(
            profile.feature_flags
        ),
    }


def profile_artifact_paths(profile: RuntimeProfile) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "field_audit_source": Path("src/mad_etd/field_audit.py"),
        "field_contract_source": Path("src/mad_etd/field_contract.py"),
        "view_contract_source": Path("src/mad_etd/view_contract.py"),
        "routing_source": Path("src/mad_etd/coordinator.py"),
        "reliability_source": Path("src/mad_etd/guard.py"),
        "detector_source": Path("src/mad_etd/detectors.py"),
        "fusion_source": Path("src/mad_etd/fusion.py"),
        "rag_source": Path("src/mad_etd/knowledge.py"),
        "reporter_source": Path("src/mad_etd/reporter.py"),
    }
    if profile.model_dir:
        paths["detector_models"] = Path(profile.model_dir)
    if profile.ood_gate_dir:
        paths["ood_artifacts"] = Path(profile.ood_gate_dir)
    if profile.tls_model_dir:
        paths["tls_models"] = Path(profile.tls_model_dir)
    return paths
