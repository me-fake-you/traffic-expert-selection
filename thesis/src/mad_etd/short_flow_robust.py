from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .deep_models import DeepDetectorModel, DeepDetectorPrediction, _torch
from .ood_v33 import RegimeConformalGateV33
from .ood_v34_orbit import OrbitConformalGateV34
from .schemas import DetectorInput
from .short_flow import (
    SHORT_FLOW_SCHEMA_VERSION,
    ShortFlowConfig,
    build_short_flow_encoder,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RobustShortFlowModel:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        conformal_dir: str | Path | None = None,
        conformal_policy: str = "conformal_v3_3",
        device: str | None = None,
    ) -> None:
        torch, _ = _torch()
        root = Path(model_dir)
        metadata_path = root / "metadata.json"
        model_path = root / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            raise ValueError(f"incomplete robust short-flow artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != SHORT_FLOW_SCHEMA_VERSION:
            raise ValueError("unsupported robust short-flow artifact schema")
        if self.metadata.get("backend") != "deep_v2_3_short":
            raise ValueError("robust short-flow backend metadata mismatch")
        if _sha256(model_path) != self.metadata.get("model_sha256"):
            raise ValueError("robust short-flow model checksum mismatch")
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
        self.conformal = None
        if conformal_dir is not None:
            self.conformal = (
                OrbitConformalGateV34(conformal_dir, regime="short")
                if conformal_policy == "conformal_v3_4_orbit"
                else RegimeConformalGateV33(conformal_dir, regime="short")
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
            regime="short_robust",
        )


class DeepV23TemporalModel:
    def __init__(
        self,
        model_dir: str | Path,
        *,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
        device: str | None = None,
        boundary_guard: bool = True,
    ) -> None:
        root = Path(model_dir)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"incomplete deep_v2_3 temporal artifact: {root}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("backend") != "deep_v2_3":
            raise ValueError("deep_v2_3 temporal metadata mismatch")
        if ood_policy not in {
            "off",
            "conformal_v3_3",
            "conformal_v3_4_orbit",
        }:
            raise ValueError(
                "deep_v2_3 supports off, conformal_v3_3, "
                "or conformal_v3_4_orbit"
            )
        conformal_root = (
            Path(ood_gate_dir) if ood_gate_dir is not None else None
        )
        if ood_policy != "off" and conformal_root is None:
            raise ValueError(f"{ood_policy} requires ood_gate_dir")
        self.short = RobustShortFlowModel(
            root / "short",
            conformal_dir=conformal_root,
            conformal_policy=ood_policy,
            device=device,
        )
        self.long = DeepDetectorModel(
            root / "long",
            expected_agent="temporal",
            ood_policy="off",
            device=device,
        )
        self.long_conformal = None
        if conformal_root is not None:
            self.long_conformal = (
                OrbitConformalGateV34(conformal_root, regime="long")
                if ood_policy == "conformal_v3_4_orbit"
                else RegimeConformalGateV33(conformal_root, regime="long")
            )
        self.boundary_guard = boundary_guard
        self.secondary_confidence_threshold = float(
            self.metadata.get("secondary_confidence_threshold", 0.9)
        )

    def _long_prediction(
        self,
        detector_input: DetectorInput,
    ) -> DeepDetectorPrediction:
        prediction = self.long.predict(detector_input)
        prediction.regime = "long"
        if self.long_conformal is not None:
            prediction.conformal = self.long_conformal.assess(
                prediction.embedding
            )
        return prediction

    def _secondary_is_reliable(
        self,
        prediction: DeepDetectorPrediction,
    ) -> bool:
        probability = prediction.malicious_probability
        high_confidence = (
            probability <= 1 - self.secondary_confidence_threshold
            or probability >= self.secondary_confidence_threshold
        )
        if not high_confidence or prediction.accepted_class is None:
            return False
        if prediction.conformal is None:
            return True
        return (
            prediction.conformal.level == "in_domain"
            and prediction.conformal.prediction_set
            == [prediction.accepted_class]
        )

    def _guard(
        self,
        primary: DeepDetectorPrediction,
        secondary: DeepDetectorPrediction,
    ) -> DeepDetectorPrediction:
        if (
            self.boundary_guard
            and primary.accepted_class is not None
            and self._secondary_is_reliable(secondary)
            and primary.accepted_class != secondary.accepted_class
        ):
            primary.accepted_class = None
            primary.uncertainty = 1.0
            primary.regime = "boundary_conflict"
        return primary

    def predict(self, detector_input: DetectorInput) -> DeepDetectorPrediction:
        count = len(detector_input.sequence.packet_lengths)
        if count == 0:
            raise ValueError("deep_v2_3 cannot predict an empty sequence")
        if count <= 2:
            return self.short.predict(detector_input)
        if count == 3:
            primary = self.short.predict(detector_input)
            secondary = self._long_prediction(detector_input)
            return self._guard(primary, secondary)
        if count == 4:
            primary = self._long_prediction(detector_input)
            secondary = self.short.predict(detector_input)
            return self._guard(primary, secondary)
        return self._long_prediction(detector_input)
