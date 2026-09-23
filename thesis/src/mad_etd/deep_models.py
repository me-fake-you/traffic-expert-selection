from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ood_v3 import ConformalReliabilityGate
from .ood_v31 import ConformalReliabilityGateV31
from .ood_prefix import ConformalPrefixGate
from .schemas import ConformalAssessment, DetectorInput


DEEP_V2_ARTIFACT_SCHEMA_VERSION = "1.0"
MAX_SEQUENCE_LENGTH = 64


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "deep_v2 requires PyTorch; install the optional 'v2' dependencies"
        ) from exc
    return torch, nn


@dataclass(slots=True)
class TCNConfig:
    input_channels: int
    static_dim: int = 0
    hidden_channels: int = 64
    embedding_dim: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1


def build_tcn(config: TCNConfig):
    torch, nn = _torch()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation * (config.kernel_size - 1) // 2
            self.net = nn.Sequential(
                nn.Conv1d(
                    channels,
                    channels,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Conv1d(
                    channels,
                    channels,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )

        def forward(self, values):
            return values + self.net(values)

    class LightweightTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Conv1d(
                config.input_channels,
                config.hidden_channels,
                kernel_size=1,
            )
            self.blocks = nn.Sequential(
                *[
                    ResidualBlock(config.hidden_channels, dilation)
                    for dilation in config.dilations
                ]
            )
            pooled_dim = config.hidden_channels * 2 + config.static_dim
            self.embedding = nn.Sequential(
                nn.Linear(pooled_dim, config.embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.classifier = nn.Linear(config.embedding_dim, 1)

        def forward(self, sequence, mask, static=None, *, return_embedding=False):
            hidden = self.blocks(self.input_projection(sequence))
            expanded = mask.unsqueeze(1)
            denominator = expanded.sum(dim=2).clamp_min(1.0)
            mean_pool = (hidden * expanded).sum(dim=2) / denominator
            minimum = torch.finfo(hidden.dtype).min
            max_pool = hidden.masked_fill(expanded == 0, minimum).max(dim=2).values
            max_pool = torch.where(
                torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool)
            )
            pieces = [mean_pool, max_pool]
            if config.static_dim:
                if static is None:
                    raise ValueError("static features are required by this TCN")
                pieces.append(static)
            embedding = self.embedding(torch.cat(pieces, dim=1))
            logits = self.classifier(embedding).squeeze(1)
            return (logits, embedding) if return_embedding else logits

    return LightweightTCN()


def encode_temporal_input(
    detector_input: DetectorInput,
    *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence = detector_input.sequence
    count = min(len(sequence.packet_lengths), max_length)
    values = np.zeros((4, max_length), dtype=np.float32)
    mask = np.zeros(max_length, dtype=np.float32)
    for index in range(count):
        length = max(0, int(sequence.packet_lengths[index]))
        direction = (
            int(sequence.directions[index])
            if index < len(sequence.directions)
            else 1
        )
        iat = (
            max(0.0, float(sequence.iats[index]))
            if index < len(sequence.iats)
            else 0.0
        )
        values[0, index] = direction * math.log1p(length)
        values[1, index] = float(direction)
        values[2, index] = math.log1p(iat)
        values[3, index] = 1.0
        mask[index] = 1.0
    return values, mask, np.empty(0, dtype=np.float32)


def _version_code(value: Any) -> float:
    normalized = str(value or "").lower().replace("v", "")
    mapping = {
        "ssl3": -1.0,
        "tls1.0": -0.6,
        "tls1": -0.6,
        "tls1.1": -0.2,
        "tls1.2": 0.4,
        "tls1.3": 1.0,
        "quic": 1.0,
    }
    return mapping.get(normalized, 0.0)


def encode_tls_input(
    detector_input: DetectorInput,
    *,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tls = detector_input.tls
    raw_records = tls.get("record_lengths") or tls.get("tls_record_lengths") or []
    records = [int(item) for item in raw_records if int(item) != 0][:max_length]
    values = np.zeros((2, max_length), dtype=np.float32)
    mask = np.zeros(max_length, dtype=np.float32)
    for index, signed_length in enumerate(records):
        direction = 1 if signed_length > 0 else -1
        values[0, index] = direction * math.log1p(abs(signed_length))
        values[1, index] = 1.0
        mask[index] = 1.0
    alpn = [str(item).lower() for item in tls.get("alpn", [])]
    static = np.asarray(
        [
            _version_code(tls.get("server_version") or tls.get("version")),
            math.log1p(max(0, int(tls.get("client_cipher_count", 0)))),
            math.log1p(
                max(
                    0,
                    int(tls.get("client_extension_count", 0))
                    + int(tls.get("server_extension_count", 0)),
                )
            ),
            float(any(item in {"h2", "http/1.1", "h3"} for item in alpn)),
        ],
        dtype=np.float32,
    )
    return values, mask, static


@dataclass(slots=True)
class DeepDetectorPrediction:
    malicious_probability: float
    uncertainty: float
    calibration_quality: float
    accepted_class: str | None
    embedding: np.ndarray
    conformal: ConformalAssessment | None = None
    regime: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeepDetectorModel:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        expected_agent: str,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        torch, _ = _torch()
        root = Path(model_dir)
        metadata_path = root / "metadata.json"
        model_path = root / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            raise ValueError(f"incomplete deep_v2 artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != DEEP_V2_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported deep_v2 artifact schema")
        if self.metadata.get("agent") != expected_agent:
            raise ValueError("deep_v2 artifact agent mismatch")
        if _sha256(model_path) != self.metadata.get("model_sha256"):
            raise ValueError("deep_v2 model checksum mismatch")
        config_data = dict(self.metadata["tcn_config"])
        config_data["dilations"] = tuple(config_data["dilations"])
        self.config = TCNConfig(**config_data)
        self.model = build_tcn(self.config)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.agent = expected_agent
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.calibration_quality = float(
            self.metadata.get("calibration_quality", 0.8)
        )
        self.ood_policy = ood_policy
        if ood_policy not in {
            "off",
            "conformal_v3",
            "conformal_v3_1",
            "conformal_v3_1_prefix",
        }:
            raise ValueError(f"unsupported deep_v2 OOD policy: {ood_policy}")
        self.conformal_gate = None
        if ood_policy == "conformal_v3" and ood_gate_dir is not None:
            self.conformal_gate = ConformalReliabilityGate(ood_gate_dir)
        elif ood_policy == "conformal_v3_1" and ood_gate_dir is not None:
            self.conformal_gate = ConformalReliabilityGateV31(ood_gate_dir)
        elif (
            ood_policy == "conformal_v3_1_prefix"
            and ood_gate_dir is not None
        ):
            self.conformal_gate = ConformalPrefixGate(ood_gate_dir)
        if ood_policy != "off" and self.conformal_gate is None:
            raise ValueError(f"{ood_policy} requires ood_gate_dir")

    def predict(self, detector_input: DetectorInput) -> DeepDetectorPrediction:
        torch, _ = _torch()
        encode = (
            encode_temporal_input
            if self.agent == "temporal"
            else encode_tls_input
        )
        sequence, mask, static = encode(detector_input)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        static_tensor = (
            torch.from_numpy(static).unsqueeze(0).to(self.device)
            if len(static)
            else None
        )
        with torch.inference_mode():
            logits, embedding = self.model(
                sequence_tensor,
                mask_tensor,
                static_tensor,
                return_embedding=True,
            )
            probability = float(
                torch.sigmoid(logits / max(self.temperature, 1e-6)).item()
            )
        thresholds = self.metadata["decision_policy"]
        benign_max = float(thresholds["benign_max_probability"])
        malicious_min = float(thresholds["malicious_min_probability"])
        accepted = (
            "benign"
            if probability <= benign_max
            else "malicious"
            if probability >= malicious_min
            else None
        )
        entropy = -(
            probability * math.log2(max(probability, 1e-9))
            + (1 - probability) * math.log2(max(1 - probability, 1e-9))
        )
        embedding_array = embedding[0].detach().cpu().numpy().astype(np.float32)
        if isinstance(
            self.conformal_gate,
            (ConformalReliabilityGateV31, ConformalPrefixGate),
        ):
            conformal = self.conformal_gate.assess(
                embedding_array,
                probability,
            )
        else:
            conformal = (
                self.conformal_gate.assess(embedding_array)
                if self.conformal_gate is not None
                else None
            )
        return DeepDetectorPrediction(
            malicious_probability=probability,
            uncertainty=max(0.08, min(0.95, entropy)),
            calibration_quality=self.calibration_quality,
            accepted_class=accepted,
            embedding=embedding_array,
            conformal=conformal,
        )


def config_as_json(config: TCNConfig) -> dict[str, Any]:
    value = asdict(config)
    value["dilations"] = list(config.dilations)
    return value
