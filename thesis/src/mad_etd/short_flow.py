from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deep_models import DeepDetectorModel, DeepDetectorPrediction, _torch
from .ood_v32 import RegimeConformalGateV32
from .schemas import DetectorInput


SHORT_FLOW_SCHEMA_VERSION = "1.0"
SHORT_FLOW_MAX_PACKETS = 4


@dataclass(slots=True)
class ShortFlowConfig:
    input_channels: int = 4
    max_packets: int = SHORT_FLOW_MAX_PACKETS
    hidden_dim: int = 128
    embedding_dim: int = 128
    dropout: float = 0.1


def build_short_flow_encoder(config: ShortFlowConfig):
    torch, nn = _torch()

    class ResidualMLPBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.norm = nn.LayerNorm(config.hidden_dim)

        def forward(self, values):
            return self.norm(values + self.net(values))

    class ShortFlowEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            input_dim = config.input_channels * config.max_packets
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.GELU(),
            )
            self.blocks = nn.Sequential(
                ResidualMLPBlock(),
                ResidualMLPBlock(),
            )
            self.embedding = nn.Sequential(
                nn.Linear(config.hidden_dim, config.embedding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.classifier = nn.Linear(config.embedding_dim, 1)

        def forward(
            self,
            sequence,
            mask,
            static=None,
            *,
            return_embedding=False,
        ):
            del static
            values = sequence[:, :, : config.max_packets].clone()
            visible = mask[:, : config.max_packets].unsqueeze(1)
            values = values * visible
            hidden = self.blocks(
                self.input_projection(values.flatten(start_dim=1))
            )
            embedding = self.embedding(hidden)
            logits = self.classifier(embedding).squeeze(1)
            return (logits, embedding) if return_embedding else logits

    return ShortFlowEncoder()


def config_as_json(config: ShortFlowConfig) -> dict[str, Any]:
    return asdict(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShortFlowModel:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        conformal_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        torch, _ = _torch()
        root = Path(model_dir)
        metadata_path = root / "metadata.json"
        model_path = root / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            raise ValueError(f"incomplete short-flow artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != SHORT_FLOW_SCHEMA_VERSION:
            raise ValueError("unsupported short-flow artifact schema")
        if self.metadata.get("backend") != "deep_v2_2_short":
            raise ValueError("short-flow backend metadata mismatch")
        if _sha256(model_path) != self.metadata.get("model_sha256"):
            raise ValueError("short-flow model checksum mismatch")
        self.config = ShortFlowConfig(**self.metadata["config"])
        self.model = build_short_flow_encoder(self.config)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        self.model.to(self.device).eval()
        self.temperature = float(self.metadata.get("temperature", 1.0))
        self.calibration_quality = float(
            self.metadata.get("calibration_quality", 0.8)
        )
        self.conformal = (
            RegimeConformalGateV32(conformal_dir, regime="short")
            if conformal_dir is not None
            else None
        )

    def predict(self, detector_input: DetectorInput) -> DeepDetectorPrediction:
        from .deep_models import encode_temporal_input

        torch, _ = _torch()
        sequence, mask, _ = encode_temporal_input(detector_input)
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits, embedding = self.model(
                sequence_tensor,
                mask_tensor,
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
        assessment = (
            self.conformal.assess(embedding_array)
            if self.conformal is not None
            else None
        )
        return DeepDetectorPrediction(
            malicious_probability=probability,
            uncertainty=max(0.08, min(0.95, entropy)),
            calibration_quality=self.calibration_quality,
            accepted_class=accepted,
            embedding=embedding_array,
            conformal=assessment,
            regime="short",
        )


class DeepV22TemporalModel:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        root = Path(model_dir)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"incomplete deep_v2_2 temporal artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("backend") != "deep_v2_2":
            raise ValueError("deep_v2_2 temporal metadata mismatch")
        if ood_policy not in {"off", "conformal_v3_2"}:
            raise ValueError(
                "deep_v2_2 supports only off or conformal_v3_2"
            )
        conformal_root = (
            Path(ood_gate_dir) if ood_gate_dir is not None else None
        )
        if ood_policy == "conformal_v3_2" and conformal_root is None:
            raise ValueError("conformal_v3_2 requires ood_gate_dir")
        self.short = ShortFlowModel(
            root / "short",
            conformal_dir=conformal_root,
            device=device,
        )
        self.long = DeepDetectorModel(
            root / "long",
            expected_agent="temporal",
            ood_policy="off",
            device=device,
        )
        self.long_conformal = (
            RegimeConformalGateV32(conformal_root, regime="long")
            if conformal_root is not None
            else None
        )

    def predict(self, detector_input: DetectorInput) -> DeepDetectorPrediction:
        count = len(detector_input.sequence.packet_lengths)
        if count == 0:
            raise ValueError("deep_v2_2 cannot predict an empty sequence")
        if count <= 3:
            return self.short.predict(detector_input)
        prediction = self.long.predict(detector_input)
        prediction.regime = "long"
        if self.long_conformal is not None:
            prediction.conformal = self.long_conformal.assess(
                prediction.embedding
            )
        return prediction
