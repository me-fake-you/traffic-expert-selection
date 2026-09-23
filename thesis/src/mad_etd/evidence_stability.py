from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .base import DetectorAgent
from .perturb import apply_perturbation
from .schemas import AgentEvidence, DetectorInput, FeatureGroup, FlowRecord


EVIDENCE_STABILITY_POLICIES = (
    "off",
    "temporal_envelope",
    "primary_view_envelope",
    "temporal_orbit_envelope",
    "temporal_orbit_calibrated",
)
COUNTERFACTUAL_KINDS = (
    "padding",
    "dummy_packet",
    "iat_jitter",
    "sequence_truncation",
)


@dataclass(slots=True)
class EvidenceEnvelopeResult:
    evidence: AgentEvidence
    audit_metadata: dict[str, Any]


def _dominant_class(item: AgentEvidence) -> str | None:
    if item.abstained or item.confidence <= 0:
        return None
    if item.benign_support > item.malicious_support:
        return "benign"
    if item.malicious_support > item.benign_support:
        return "malicious"
    return None


def _shift_level(items: list[AgentEvidence]) -> str:
    ranks = {"off": 0, "in_domain": 1, "warning": 2, "hard": 3}
    return max(
        (item.distribution_shift_level for item in items),
        key=lambda value: ranks[value],
    )


class CounterfactualEvidenceEnvelope:
    name = "CounterfactualEvidenceEnvelope"
    version = "1.0"

    def __init__(
        self,
        policy: str = "off",
        *,
        strength: float = 0.2,
        seed: int = 42,
        support_span_limit: float = 0.15,
        orbit_reliability_dir: str | None = None,
    ) -> None:
        if policy not in EVIDENCE_STABILITY_POLICIES:
            raise ValueError(f"unsupported evidence stability policy: {policy}")
        if strength != 0.2 or seed != 42 or support_span_limit != 0.15:
            raise ValueError("v2.4 evidence envelope parameters are frozen")
        self.policy = policy
        self.strength = strength
        self.seed = seed
        self.support_span_limit = support_span_limit
        self.orbit_reliability = None
        if policy == "temporal_orbit_calibrated":
            if orbit_reliability_dir is None:
                raise ValueError(
                    "temporal_orbit_calibrated requires orbit_reliability_dir"
                )
            from .orbit_reliability import OrbitReliabilityCalibrator

            self.orbit_reliability = OrbitReliabilityCalibrator(
                orbit_reliability_dir
            )

    def targets(self, agent_name: str) -> bool:
        if self.policy == "temporal_envelope":
            return agent_name == "TemporalBehaviorAgent"
        if self.policy in {
            "temporal_orbit_envelope",
            "temporal_orbit_calibrated",
        }:
            return agent_name == "TemporalBehaviorAgent"
        if self.policy == "primary_view_envelope":
            return agent_name in {
                "StatsDetectorAgent",
                "TemporalBehaviorAgent",
            }
        return False

    def analyze(
        self,
        agent: DetectorAgent,
        detector_input: DetectorInput,
    ) -> EvidenceEnvelopeResult:
        packet_count = len(detector_input.sequence.packet_lengths)
        if (
            self.policy == "off"
            or not self.targets(agent.name)
            or packet_count < 1
            or packet_count > 4
        ):
            evidence = agent.analyze(detector_input.model_copy(deep=True))
            return EvidenceEnvelopeResult(
                evidence=evidence,
                audit_metadata={
                    "policy": self.policy,
                    "applied": False,
                    "reason": (
                        "policy_off"
                        if self.policy == "off"
                        else "agent_not_targeted"
                        if not self.targets(agent.name)
                        else "packet_regime_outside_1_to_4"
                    ),
                    "probe_count": 1,
                },
            )

        started = time.perf_counter()
        sanitized_context = {
            key: value
            for key, value in detector_input.context.items()
            if key != "synthetic_perturbation"
        }
        inputs = [
            DetectorInput(
                stats=detector_input.stats,
                sequence=detector_input.sequence,
                tls=detector_input.tls,
                payload_tokens=detector_input.payload_tokens,
                context=sanitized_context,
            )
        ]
        probe_names = ["clean"]
        base = FlowRecord(
            trace_id="counterfactual-envelope",
            sample_id="counterfactual-envelope",
            stats=dict(detector_input.stats),
            sequence=detector_input.sequence.model_copy(deep=True),
            tls=dict(detector_input.tls),
            payload_tokens=(
                list(detector_input.payload_tokens)
                if detector_input.payload_tokens
                else None
            ),
            context=sanitized_context,
            provenance={},
            labels={},
        )
        for kind in COUNTERFACTUAL_KINDS:
            perturbed = apply_perturbation(
                base,
                kind,
                strength=self.strength,
                seed=self.seed,
            )
            inputs.append(
                DetectorInput(
                    stats=perturbed.stats,
                    sequence=perturbed.sequence,
                    tls=detector_input.tls,
                    payload_tokens=detector_input.payload_tokens,
                    context=sanitized_context,
                )
            )
            probe_names.append(kind)

        orbit_mode = self.policy in {
            "temporal_orbit_envelope",
            "temporal_orbit_calibrated",
        }
        orbit_predictions = None
        if orbit_mode:
            from .detectors import _deep_prediction_evidence

            if not hasattr(agent, "deep_model") or agent.deep_model is None:
                raise ValueError(
                    "orbit evidence policies require a deep temporal agent"
                )
            orbit_predictions = [
                agent.deep_model.predict(value) for value in inputs
            ]
            probes = [
                _deep_prediction_evidence(
                    prediction,
                    agent_name=agent.name,
                    agent_version="3.4",
                    feature_group=FeatureGroup.SEQUENCE,
                    used_fields=[
                        "sequence.packet_lengths",
                        "sequence.directions",
                        "sequence.iats",
                    ],
                    latency_ms=0,
                )
                for prediction in orbit_predictions
            ]
        else:
            probes = [agent.analyze(value) for value in inputs]
        classes = [_dominant_class(item) for item in probes]
        unanimous = (
            classes[0]
            if classes[0] is not None
            and all(value == classes[0] for value in classes)
            else None
        )
        hard_or_empty = any(
            item.abstained
            or item.distribution_shift_level == "hard"
            for item in probes
        )
        primary_supports = (
            [
                (
                    item.benign_support
                    if unanimous == "benign"
                    else item.malicious_support
                )
                for item in probes
            ]
            if unanimous is not None
            else []
        )
        support_span = (
            max(primary_supports) - min(primary_supports)
            if primary_supports
            else 1.0
        )
        action = (
            "abstain_probe_rejection"
            if hard_or_empty
            else "abstain_class_disagreement"
            if unanimous is None
            else "conservative_merge"
            if orbit_mode
            else "abstain_support_span"
            if support_span > self.support_span_limit
            else "conservative_merge"
        )
        common = {
            "agent_name": agent.name,
            "agent_version": probes[0].agent_version,
            "feature_group": probes[0].feature_group,
            "calibration_quality": min(
                item.calibration_quality for item in probes
            ),
            "distribution_shift_score": max(
                item.distribution_shift_score for item in probes
            ),
            "distribution_shift_raw_score": max(
                (
                    item.distribution_shift_raw_score
                    for item in probes
                    if item.distribution_shift_raw_score is not None
                ),
                default=None,
            ),
            "distribution_shift_level": _shift_level(probes),
            "model_reliability": min(
                item.model_reliability for item in probes
            ),
            "used_fields": sorted(
                {
                    field
                    for item in probes
                    for field in item.used_fields
                }
            ),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
        calibrated_lower = None
        if (
            action == "conservative_merge"
            and self.policy == "temporal_orbit_calibrated"
            and orbit_predictions is not None
        ):
            from .orbit_reliability import orbit_feature_vector

            feature_rows = []
            for prediction, value in zip(
                orbit_predictions,
                inputs,
                strict=True,
            ):
                assessment = prediction.conformal
                regime = (
                    "short"
                    if len(value.sequence.packet_lengths) <= 3
                    else "long"
                )
                feature_rows.append(
                    {
                        "probability": prediction.malicious_probability,
                        "accepted_class": prediction.accepted_class,
                        "uncertainty": prediction.uncertainty,
                        "regime": regime,
                        "benign_p_value": (
                            assessment.benign_p_value
                            if assessment is not None
                            else 1.0
                        ),
                        "malicious_p_value": (
                            assessment.malicious_p_value
                            if assessment is not None
                            else 1.0
                        ),
                        "prediction_set": (
                            assessment.prediction_set
                            if assessment is not None
                            else ["benign", "malicious"]
                        ),
                        "hard_ood": (
                            assessment is not None
                            and assessment.level == "hard"
                        ),
                    }
                )
            features, calibrated_class, _ = orbit_feature_vector(feature_rows)
            if calibrated_class != unanimous:
                action = "abstain_class_disagreement"
            else:
                calibrated_lower = self.orbit_reliability.lower_support(
                    features
                )
                if calibrated_lower <= 0:
                    action = "abstain_calibrated_zero_support"

        if action != "conservative_merge":
            result = AgentEvidence(
                **common,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                evidence=[
                    "counterfactual evidence stability envelope",
                    action,
                ],
            )
        else:
            benign = min(item.benign_support for item in probes)
            malicious = min(item.malicious_support for item in probes)
            if calibrated_lower is not None:
                if unanimous == "benign":
                    benign = min(probes[0].benign_support, calibrated_lower)
                else:
                    malicious = min(
                        probes[0].malicious_support,
                        calibrated_lower,
                    )
            ignorance = max(0.0, 1 - benign - malicious)
            uncertainty = max(
                ignorance,
                max(item.uncertainty for item in probes),
            )
            result = AgentEvidence(
                **common,
                benign_support=benign,
                malicious_support=malicious,
                confidence=max(benign, malicious),
                uncertainty=uncertainty,
                abstained=False,
                evidence=[
                    "counterfactual evidence stability envelope",
                    "conservative minimum support across five probes",
                ],
            )
        metadata = {
            "policy": self.policy,
            "applied": True,
            "probe_count": len(probes),
            "probe_names": probe_names,
            "strength": self.strength,
            "seed": self.seed,
            "class_consistent": unanimous is not None,
            "dominant_classes": classes,
            "support_span": support_span,
            "support_span_limit": self.support_span_limit,
            "action": action,
            "orbit_conformal": orbit_mode,
            "calibrated_lower_support": calibrated_lower,
            "hard_ood_count": (
                sum(
                    prediction.conformal is not None
                    and prediction.conformal.level == "hard"
                    for prediction in orbit_predictions
                )
                if orbit_predictions is not None
                else sum(
                    item.distribution_shift_level == "hard"
                    for item in probes
                )
            ),
            "synthetic_context_visible": False,
            "labels_visible": False,
            "identifiers_visible": False,
            "provenance_visible": False,
        }
        return EvidenceEnvelopeResult(
            evidence=result,
            audit_metadata=metadata,
        )
