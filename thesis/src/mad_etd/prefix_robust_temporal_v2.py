"""Default-off prefix/truncation-aware TemporalBehaviorAgent contract.

Only architecture, safe representation, future loss, and fail-closed shadow
inference contracts are defined here.  No model is trained or evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import AgentEvidenceV2, DetectorCapabilityProfile, DetectorInput


BACKEND_NAME = "prefix_robust_v2"
CANDIDATE_ID = "temporal_prefix_robust_v2"
IMPLEMENTATION_STATUS = "implementation_ready_not_evaluated"
PROMOTION_STATUS = "not_evaluated"
SAFE_SEQUENCE_FIELDS = (
    "sequence.packet_lengths",
    "sequence.directions",
    "sequence.iats",
)
_REQUIRED_ARTIFACTS = (
    "candidate_config.json",
    "model.pt",
    "short_flow_calibration.json",
)


def sequence_feature_policy_hash() -> str:
    payload = json.dumps(
        SAFE_SEQUENCE_FIELDS, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PrefixRobustTemporalV2Config:
    backend: str = BACKEND_NAME
    max_packets: int = 64
    input_channels: int = 4
    hidden_width: int = 128
    embedding_dim: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    prefix_augmentation: bool = True
    truncation_augmentation: bool = True
    consistency_regularization: bool = True
    short_flow_uncertainty_calibration: bool = True
    consistency_weight: float = 0.20
    short_flow_uncertainty_weight: float = 0.10
    embedding_enters_fusion: bool = False
    default_enabled: bool = False
    implementation_status: str = IMPLEMENTATION_STATUS
    promotion_status: str = PROMOTION_STATUS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dilations"] = list(self.dilations)
        return payload


def encode_prefix_sequence(
    detector_input: DetectorInput,
    config: PrefixRobustTemporalV2Config | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode 1--64 safe packet events and a derived mask."""

    cfg = config or PrefixRobustTemporalV2Config()
    sequence = detector_input.sequence
    count = min(len(sequence.packet_lengths), cfg.max_packets)
    if count < 1:
        raise ValueError("prefix_robust_v2 requires at least one packet event")
    if len(sequence.directions) < count:
        raise ValueError("packet direction is required for every packet event")
    if sequence.iats and len(sequence.iats) < count:
        raise ValueError("IAT sequence must be empty or aligned to packet events")
    values = np.zeros((cfg.max_packets, cfg.input_channels), dtype=np.float32)
    mask = np.zeros((cfg.max_packets,), dtype=np.float32)
    for index in range(count):
        length = float(sequence.packet_lengths[index])
        direction = float(sequence.directions[index])
        iat = float(sequence.iats[index]) if sequence.iats else 0.0
        signed_log_length = direction * math.log1p(length)
        values[index] = (
            signed_log_length,
            direction,
            math.log1p(max(iat, 0.0)),
            1.0,
        )
        mask[index] = 1.0
    return values, mask


def build_prefix_robust_tcn(
    config: PrefixRobustTemporalV2Config | None = None,
) -> Any:
    """Build the untrained residual TCN lazily without touching data."""

    cfg = config or PrefixRobustTemporalV2Config()
    import torch
    from torch import nn

    class TemporalResidualBlock(nn.Module):
        def __init__(self, width: int, dilation: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.GroupNorm(8, width),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                ),
            )

        def forward(self, value: Any) -> Any:
            return value + self.net(value)

    class PrefixRobustTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(
                cfg.input_channels, cfg.hidden_width, kernel_size=1
            )
            self.blocks = nn.ModuleList(
                TemporalResidualBlock(cfg.hidden_width, dilation)
                for dilation in cfg.dilations
            )
            self.embedding = nn.Linear(
                cfg.hidden_width * 2, cfg.embedding_dim
            )
            self.classifier = nn.Linear(cfg.embedding_dim, 2)
            self.uncertainty_head = nn.Sequential(
                nn.Linear(cfg.embedding_dim, 1), nn.Sigmoid()
            )

        def forward(self, value: Any, mask: Any) -> dict[str, Any]:
            hidden = self.input_projection(value.transpose(1, 2))
            for block in self.blocks:
                hidden = block(hidden)
            expanded_mask = mask.unsqueeze(1).to(hidden.dtype)
            denominator = expanded_mask.sum(dim=-1).clamp_min(1.0)
            mean_pool = (hidden * expanded_mask).sum(dim=-1) / denominator
            negative = torch.finfo(hidden.dtype).min
            max_pool = hidden.masked_fill(expanded_mask == 0, negative).max(
                dim=-1
            ).values
            embedding = self.embedding(torch.cat([mean_pool, max_pool], dim=-1))
            return {
                "logits": self.classifier(embedding),
                "embedding": embedding,
                "uncertainty": self.uncertainty_head(embedding).squeeze(-1),
            }

    return PrefixRobustTCN()


def prefix_robust_training_loss(
    *,
    full_logits: Any,
    prefix_logits: Any,
    labels: Any,
    predicted_uncertainty: Any,
    prefix_lengths: Any,
    config: PrefixRobustTemporalV2Config | None = None,
) -> dict[str, Any]:
    """Future train-only prefix consistency and short-flow uncertainty loss."""

    cfg = config or PrefixRobustTemporalV2Config()
    import torch
    import torch.nn.functional as functional

    classification = functional.cross_entropy(full_logits, labels)
    prefix_classification = functional.cross_entropy(prefix_logits, labels)
    consistency = functional.mse_loss(
        torch.softmax(prefix_logits, dim=-1),
        torch.softmax(full_logits.detach(), dim=-1),
    )
    short_target = (prefix_lengths <= 2).to(predicted_uncertainty.dtype)
    short_uncertainty = functional.binary_cross_entropy(
        predicted_uncertainty.clamp(1e-6, 1 - 1e-6), short_target
    )
    total = (
        classification
        + prefix_classification
        + cfg.consistency_weight * consistency
        + cfg.short_flow_uncertainty_weight * short_uncertainty
    )
    return {
        "total": total,
        "classification": classification,
        "prefix_classification": prefix_classification,
        "consistency": consistency,
        "short_flow_uncertainty": short_uncertainty,
    }


class PrefixRobustTemporalV2Backend:
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

    @staticmethod
    def _sequence_schema_valid(detector_input: DetectorInput) -> bool:
        sequence = detector_input.sequence
        count = min(len(sequence.packet_lengths), 64)
        return bool(count) and len(sequence.directions) >= count and (
            not sequence.iats or len(sequence.iats) >= count
        )

    def capability_profile(
        self, detector_input: DetectorInput
    ) -> DetectorCapabilityProfile:
        count = len(detector_input.sequence.packet_lengths)
        if count < 1:
            status = "missing_view"
            reasons = ["SEQUENCE_VIEW_MISSING"]
        elif not self._sequence_schema_valid(detector_input):
            status = "unsupported_schema"
            reasons = ["PREFIX_SEQUENCE_ALIGNMENT_INVALID"]
        elif not self.artifact_ready:
            status = "unsupported_schema"
            reasons = ["PREFIX_ROBUST_V2_ARTIFACT_MISSING_DEFAULT_OFF"]
        else:
            status = "available"
            reasons = ["PREFIX_ROBUST_V2_SEQUENCE_AVAILABLE"]
        observed = [
            field
            for field, present in (
                ("sequence.packet_lengths", bool(count)),
                ("sequence.directions", bool(detector_input.sequence.directions)),
                ("sequence.iats", bool(detector_input.sequence.iats)),
            )
            if present
        ]
        return DetectorCapabilityProfile(
            agent_name="TemporalBehaviorAgent",
            backend=self.backend,
            status=status,
            consumed_fields=list(SAFE_SEQUENCE_FIELDS),
            available_fields=observed,
            observed_fields=observed,
            unsupported_fields=sorted(set(SAFE_SEQUENCE_FIELDS) - set(observed)),
            reason_codes=reasons,
        )

    def analyze_v2(self, detector_input: DetectorInput) -> AgentEvidenceV2:
        schema_valid = self._sequence_schema_valid(detector_input)
        if schema_valid:
            _values, _mask = encode_prefix_sequence(detector_input)
        reason = (
            "safe packet sequence schema is unavailable or misaligned"
            if not schema_valid
            else "candidate model artifact is missing; inference failed closed"
            if not self.artifact_ready
            else "candidate artifact exists but formal inference is disabled until a future training/evaluation protocol"
        )
        return AgentEvidenceV2(
            agent_name="TemporalBehaviorAgent",
            agent_type="sequence:prefix_robust_v2",
            input_feature_policy_hash=sequence_feature_policy_hash(),
            artifact_hash="missing_untrained_prefix_robust_v2",
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
            supported_capabilities=list(SAFE_SEQUENCE_FIELDS),
            safety_flags=["DEFAULT_OFF", "NO_RUNTIME_AUTHORITY"],
            dataset_scope="future_prefix_truncation_safe_sequence_only",
            promotion_status="not_evaluated",
        )


def candidate_contract() -> dict[str, Any]:
    return {
        **PrefixRobustTemporalV2Config().to_dict(),
        "candidate_id": CANDIDATE_ID,
        "agent": "TemporalBehaviorAgent",
        "required_artifacts": list(_REQUIRED_ARTIFACTS),
        "inference_input": list(SAFE_SEQUENCE_FIELDS) + ["derived_mask"],
        "handles_packet_counts": "1-64",
        "risk_cells": [
            "1_2_packet_short_flow",
            "3_4_packet_boundary",
            "4_to_3_truncation",
            "general_sequence_truncation",
        ],
        "output_schema": "AgentEvidenceV2",
        "enters_fusion": False,
        "final_verdict_owner": "FusionAgent",
        "no_performance_claim": True,
    }

