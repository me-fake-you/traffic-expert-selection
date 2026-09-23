"""Default-off domain-robust StatsDetectorAgent backend contract.

This module defines architecture and future-training interfaces only.  It does
not train, fit, evaluate, or promote a model.  Missing artifacts fail closed and
produce non-contributing AgentEvidenceV2 records for the shadow evidence bus.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .schemas import AgentEvidenceV2, DetectorCapabilityProfile, DetectorInput


BACKEND_NAME = "domain_robust_v2"
CANDIDATE_ID = "stats_domain_robust_v2"
IMPLEMENTATION_STATUS = "implementation_ready_not_evaluated"
PROMOTION_STATUS = "not_evaluated"

_BLOCKED_TOKENS = frozenset(
    {
        "ip",
        "ipv4",
        "ipv6",
        "port",
        "timestamp",
        "time",
        "flowid",
        "flow_id",
        "attack",
        "family",
        "label",
        "sampleid",
        "sample_id",
        "sourcefile",
        "source_file",
        "provenance",
        "dataset",
        "sourcegroup",
        "source_group",
        "split",
    }
)
_REQUIRED_ARTIFACTS = (
    "candidate_config.json",
    "quantile_transform.json",
    "model.pt",
)


def _tokens(path: str) -> set[str]:
    normalized = path.lower().replace("-", "_")
    parts = set(filter(None, re.split(r"[._/]+", normalized)))
    parts.add(normalized.rsplit(".", 1)[-1])
    parts.add(normalized.replace(".", "_"))
    return parts


def validate_safe_stats_feature_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Return a deterministic safe policy or reject shortcut-prone fields."""

    safe: list[str] = []
    for path in paths:
        normalized = str(path).strip()
        if not normalized.startswith("stats."):
            raise ValueError(f"domain_robust_v2 accepts only stats.* fields: {path}")
        if _tokens(normalized) & _BLOCKED_TOKENS:
            raise ValueError(f"blocked/context-only inference feature: {path}")
        safe.append(normalized)
    return tuple(sorted(set(safe)))


def canonical_feature_policy_hash(paths: Sequence[str]) -> str:
    safe = validate_safe_stats_feature_paths(paths)
    payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DomainRobustStatsV2Config:
    backend: str = BACKEND_NAME
    hidden_width: int = 256
    residual_blocks: int = 4
    embedding_dim: int = 128
    dropout: float = 0.1
    quantile_normalization: str = "train_only"
    group_objective: str = "group_dro_train_only"
    reconstruction_weight: float = 0.10
    consistency_weight: float = 0.10
    source_group_inference_feature: bool = False
    embedding_enters_fusion: bool = False
    default_enabled: bool = False
    implementation_status: str = IMPLEMENTATION_STATUS
    promotion_status: str = PROMOTION_STATUS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_domain_robust_residual_mlp(
    input_dim: int,
    config: DomainRobustStatsV2Config | None = None,
) -> Any:
    """Build the untrained architecture lazily; no optimizer or data is used."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    cfg = config or DomainRobustStatsV2Config()
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
            )

        def forward(self, value: Any) -> Any:
            return value + self.net(value)

    class DomainRobustResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, cfg.hidden_width)
            self.blocks = nn.ModuleList(
                ResidualBlock(cfg.hidden_width, cfg.dropout)
                for _ in range(cfg.residual_blocks)
            )
            self.embedding_head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_width),
                nn.Linear(cfg.hidden_width, cfg.embedding_dim),
            )
            self.classification_head = nn.Linear(cfg.embedding_dim, 2)
            self.reconstruction_head = nn.Linear(cfg.embedding_dim, input_dim)

        def forward(self, value: Any) -> dict[str, Any]:
            hidden = self.input_projection(value)
            for block in self.blocks:
                hidden = block(hidden)
            embedding = self.embedding_head(hidden)
            return {
                "logits": self.classification_head(embedding),
                "embedding": embedding,
                "reconstruction": self.reconstruction_head(embedding),
            }

    return DomainRobustResidualMLP()


def domain_robust_training_loss(
    *,
    logits: Any,
    labels: Any,
    group_ids: Any,
    reconstruction: Any | None = None,
    original_features: Any | None = None,
    consistency_logits: Any | None = None,
    config: DomainRobustStatsV2Config | None = None,
) -> dict[str, Any]:
    """Future train-only objective; group IDs never enter model inference."""

    cfg = config or DomainRobustStatsV2Config()
    import torch
    import torch.nn.functional as functional

    per_sample = functional.cross_entropy(logits, labels, reduction="none")
    unique_groups = torch.unique(group_ids)
    if unique_groups.numel() == 0:
        raise ValueError("group_ids must contain at least one train-time group")
    group_losses = torch.stack(
        [per_sample[group_ids == group].mean() for group in unique_groups]
    )
    group_dro = group_losses.max()
    reconstruction_loss = torch.zeros((), device=logits.device)
    if reconstruction is not None or original_features is not None:
        if reconstruction is None or original_features is None:
            raise ValueError("reconstruction and original_features must be paired")
        reconstruction_loss = functional.mse_loss(
            reconstruction, original_features
        )
    consistency_loss = torch.zeros((), device=logits.device)
    if consistency_logits is not None:
        consistency_loss = functional.mse_loss(
            torch.softmax(logits, dim=-1),
            torch.softmax(consistency_logits, dim=-1),
        )
    total = (
        group_dro
        + cfg.reconstruction_weight * reconstruction_loss
        + cfg.consistency_weight * consistency_loss
    )
    return {
        "total": total,
        "group_dro": group_dro,
        "group_losses": group_losses,
        "reconstruction": reconstruction_loss,
        "consistency": consistency_loss,
    }


class DomainRobustStatsV2Backend:
    """Untrained candidate contract exposed only to the default-off shadow lane."""

    backend = BACKEND_NAME
    candidate_id = CANDIDATE_ID
    default_enabled = False
    implementation_status = IMPLEMENTATION_STATUS
    promotion_status = PROMOTION_STATUS

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.missing_artifacts = [
            name
            for name in _REQUIRED_ARTIFACTS
            if self.model_dir is None or not (self.model_dir / name).is_file()
        ]
        self.artifact_ready = not self.missing_artifacts

    def capability_profile(
        self, detector_input: DetectorInput
    ) -> DetectorCapabilityProfile:
        fields = validate_safe_stats_feature_paths(
            [f"stats.{key}" for key in detector_input.stats]
        )
        if not fields:
            status = "missing_view"
            reasons = ["STATS_VIEW_MISSING"]
        elif not self.artifact_ready:
            status = "unsupported_schema"
            reasons = ["DOMAIN_ROBUST_V2_ARTIFACT_MISSING_DEFAULT_OFF"]
        else:
            status = "available"
            reasons = ["DOMAIN_ROBUST_V2_SAFE_STATS_AVAILABLE"]
        return DetectorCapabilityProfile(
            agent_name="StatsDetectorAgent",
            backend=self.backend,
            status=status,
            consumed_fields=list(fields),
            available_fields=list(fields),
            observed_fields=list(fields),
            reason_codes=reasons,
        )

    def analyze_v2(
        self,
        detector_input: DetectorInput,
        *,
        feature_policy_hash: str | None = None,
    ) -> AgentEvidenceV2:
        fields = validate_safe_stats_feature_paths(
            [f"stats.{key}" for key in detector_input.stats]
        )
        reason = (
            "candidate model artifact is missing; inference failed closed"
            if not self.artifact_ready
            else "candidate artifact exists but formal inference is disabled until a future training/evaluation protocol"
        )
        return AgentEvidenceV2(
            agent_name="StatsDetectorAgent",
            agent_type="stats:domain_robust_v2",
            input_feature_policy_hash=feature_policy_hash
            or canonical_feature_policy_hash(fields),
            artifact_hash="missing_untrained_domain_robust_v2",
            prediction="abstain",
            probabilities={"benign": 0.0, "malicious": 0.0},
            confidence=0.0,
            uncertainty=1.0,
            reliability=0.0,
            calibration_quality=0.0,
            contributes_to_verdict=False,
            applicability="unsupported",
            unsupported_reason=reason,
            reason_codes=["CANDIDATE_NOT_TRAINED", "FAIL_CLOSED"],
            calibration_status="not_fitted",
            supported_capabilities=list(fields),
            safety_flags=["DEFAULT_OFF", "NO_RUNTIME_AUTHORITY"],
            dataset_scope="future_group_held_out_safe_flow_only",
            promotion_status="not_evaluated",
        )


def candidate_contract() -> dict[str, Any]:
    return {
        **DomainRobustStatsV2Config().to_dict(),
        "candidate_id": CANDIDATE_ID,
        "agent": "StatsDetectorAgent",
        "required_artifacts": list(_REQUIRED_ARTIFACTS),
        "inference_input": "FieldAudit-approved stats.* only",
        "train_only_group_signal": "source_group",
        "output_schema": "AgentEvidenceV2",
        "enters_fusion": False,
        "final_verdict_owner": "FusionAgent",
        "no_performance_claim": True,
    }
