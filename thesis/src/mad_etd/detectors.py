from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

from .base import DetectorAgent
from .models import LearnedDetectorModel
from .ood import (
    OOD_POLICIES,
    OODReliabilityGate,
    apply_ood_policy_to_evidence,
)
from .schemas import AgentEvidence, DetectorInput, FeatureGroup


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _finalize_support(risk: float, uncertainty: float) -> tuple[float, float, float]:
    certainty_mass = max(0.0, min(1.0, 1 - uncertainty))
    malicious = max(0.0, min(certainty_mass, risk * certainty_mass))
    benign = max(0.0, min(certainty_mass - malicious, (1 - risk) * certainty_mass))
    confidence = max(malicious, benign)
    return benign, malicious, confidence


def _deep_prediction_evidence(
    prediction,
    *,
    agent_name: str,
    agent_version: str,
    feature_group: FeatureGroup,
    used_fields: list[str],
    latency_ms: float,
) -> AgentEvidence:
    assessment = prediction.conformal
    if prediction.regime == "boundary_conflict":
        return AgentEvidence(
            agent_name=agent_name,
            agent_version=agent_version,
            feature_group=feature_group,
            benign_support=0,
            malicious_support=0,
            confidence=0,
            uncertainty=1,
            calibration_quality=prediction.calibration_quality,
            distribution_shift_score=(
                assessment.shift_score if assessment is not None else 0
            ),
            distribution_shift_level=(
                assessment.level if assessment is not None else "off"
            ),
            model_reliability=0,
            abstained=True,
            evidence=[
                "deep_v2_3 boundary consistency guard",
                "high-confidence short/long disagreement; temporal evidence abstained",
            ],
            used_fields=used_fields,
            latency_ms=latency_ms,
        )
    if assessment is not None and (
        assessment.level == "hard" or not assessment.prediction_set
    ):
        return AgentEvidence(
            agent_name=agent_name,
            agent_version=agent_version,
            feature_group=feature_group,
            benign_support=0,
            malicious_support=0,
            confidence=0,
            uncertainty=1,
            calibration_quality=prediction.calibration_quality,
            distribution_shift_score=assessment.shift_score,
            distribution_shift_level=(
                "hard" if assessment.level == "hard" else "warning"
            ),
            model_reliability=0,
            abstained=True,
            evidence=[
                "deep temporal sequence model",
                "conformal reliability gate found no supported class",
            ],
            used_fields=used_fields,
            latency_ms=latency_ms,
        )
    uncertainty = prediction.uncertainty
    model_reliability = 1.0
    evidence = [
        (
            "deep_v2_3 robust short-flow MLP with temperature calibration"
            if prediction.regime == "short_robust"
            else "deep_v2_2 short-flow MLP with temperature calibration"
            if prediction.regime == "short"
            else "deep temporal TCN with temperature calibration"
        )
    ]
    shift_score = 0.0
    shift_level = "off"
    if assessment is not None:
        shift_score = assessment.shift_score
        shift_level = assessment.level
        if len(assessment.prediction_set) == 2:
            uncertainty = max(uncertainty, 0.65)
            model_reliability = 0.5
            evidence.append("conformal prediction set is ambiguous")
        else:
            evidence.append(
                "conformal singleton prediction set: "
                + assessment.prediction_set[0]
            )
    benign, malicious, confidence = _finalize_support(
        prediction.malicious_probability,
        uncertainty,
    )
    return AgentEvidence(
        agent_name=agent_name,
        agent_version=agent_version,
        feature_group=feature_group,
        benign_support=benign,
        malicious_support=malicious,
        confidence=confidence,
        uncertainty=uncertainty,
        calibration_quality=prediction.calibration_quality,
        distribution_shift_score=shift_score,
        distribution_shift_level=shift_level,
        model_reliability=model_reliability,
        evidence=evidence,
        used_fields=used_fields,
        latency_ms=latency_ms,
    )


class StatsDetectorAgent(DetectorAgent):
    name = "StatsDetectorAgent"
    version = "2.1"

    def __init__(
        self,
        *,
        backend: str = "rule",
        model_dir: str | Path | None = None,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
    ) -> None:
        if backend not in {
            "rule",
            "learned",
            "learned_robust_v2_6",
            "domain_robust_v1",
            "domain_robust_v2",
            "safe_flow_ensemble_v1",
        }:
            raise ValueError(f"unsupported detector backend: {backend}")
        if backend in {"learned", "learned_robust_v2_6"} and model_dir is None:
            raise ValueError("model_dir is required for learned StatsDetectorAgent")
        if ood_policy not in OOD_POLICIES:
            raise ValueError(f"unsupported OOD policy: {ood_policy}")
        if ood_policy != "off" and ood_gate_dir is None:
            raise ValueError("ood_gate_dir is required when OOD policy is enabled")
        self.backend = backend
        self.ood_policy = ood_policy
        if backend == "domain_robust_v2":
            from .domain_robust_stats_v2 import DomainRobustStatsV2Backend

            self.candidate_backend = DomainRobustStatsV2Backend(model_dir)
        else:
            self.candidate_backend = None
        self.model = (
            LearnedDetectorModel(model_dir, expected_agent="stats")
            if backend in {"learned", "learned_robust_v2_6"}
            else None
        )
        self.ood_gate = (
            OODReliabilityGate(ood_gate_dir, expected_agent="stats")
            if ood_gate_dir is not None
            else None
        )

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        started = time.perf_counter()
        stats = detector_input.stats
        if not stats:
            return AgentEvidence(
                agent_name=self.name,
                feature_group=FeatureGroup.STATS,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                evidence=["flow statistics unavailable"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # W63's residual group-DRO candidate is deliberately registered but not
        # runtime-enabled.  Its NF source-specific feature contract has not passed
        # W67 Fusion/OOD shadow integration, so fail closed instead of silently
        # projecting arbitrary runtime statistics into its training schema.
        if self.backend in {
            "domain_robust_v1",
            "domain_robust_v2",
            "safe_flow_ensemble_v1",
        }:
            candidate_reason = (
                "safe_flow_ensemble_v1 failed W75 calibration/Brier promotion gates"
                if self.backend == "safe_flow_ensemble_v1"
                else "domain_robust_v2 is untrained, default-off, and not evaluated"
                if self.backend == "domain_robust_v2"
                else "domain_robust_v1 is a default-off dataset-specific candidate"
            )
            return AgentEvidence(
                agent_name=self.name,
                agent_version=self.version,
                feature_group=FeatureGroup.STATS,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                contributes_to_verdict=False,
                evidence=[
                    candidate_reason,
                    "Fusion/OOD shadow integration has not been accepted",
                ],
                used_fields=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.model is not None:
            prediction = self.model.predict(detector_input)
            benign, malicious, confidence = _finalize_support(
                prediction.malicious_probability,
                prediction.uncertainty,
            )
            policy_text = (
                f"calibrated {prediction.accepted_class} region"
                if prediction.accepted_class
                else "calibrated reject region"
            )
            assessment = (
                self.ood_gate.assess(detector_input)
                if self.ood_gate is not None
                else None
            )
            base_evidence = AgentEvidence(
                agent_name=self.name,
                agent_version=self.version,
                feature_group=FeatureGroup.STATS,
                benign_support=benign,
                malicious_support=malicious,
                confidence=confidence,
                uncertainty=prediction.uncertainty,
                calibration_quality=prediction.calibration_quality,
                distribution_shift_score=(
                    assessment.shift_score if assessment else 0
                ),
                distribution_shift_raw_score=(
                    assessment.raw_score if assessment else None
                ),
                distribution_shift_level=(
                    assessment.level if assessment else "off"
                ),
                model_reliability=1,
                evidence=[
                    (
                        "robust learned statistical model with Platt calibration"
                        if self.backend == "learned_robust_v2_6"
                        else "learned statistical model with Platt calibration"
                    ),
                    policy_text,
                    *(
                        [
                            (
                                "model-distribution shift "
                                f"{assessment.level} "
                                f"(score={assessment.shift_score:.3f})"
                            )
                        ]
                        if assessment
                        else []
                    ),
                ],
                used_fields=sorted(f"stats.{key}" for key in stats),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return apply_ood_policy_to_evidence(
                base_evidence, self.ood_policy
            )

        packet_count = stats.get("packet_count", 0)
        total_bytes = stats.get("total_bytes", 0)
        duration = max(stats.get("duration", 0), 1e-6)
        outbound_ratio = stats.get(
            "outbound_ratio",
            stats.get("outbound_bytes", 0) / max(total_bytes, 1),
        )
        mean_length = stats.get(
            "mean_packet_length", total_bytes / max(packet_count, 1)
        )
        bytes_per_second = total_bytes / duration

        score = -1.1
        reasons: list[str] = []
        if outbound_ratio >= 0.85:
            score += 1.2
            reasons.append("strong outbound byte asymmetry")
        elif outbound_ratio <= 0.15:
            score += 0.55
            reasons.append("strong inbound byte asymmetry")
        if 5 <= packet_count <= 40 and duration >= 20:
            score += 0.75
            reasons.append("small long-lived flow")
        if packet_count >= 500 and mean_length <= 80:
            score += 0.9
            reasons.append("many small packets")
        if bytes_per_second >= 2_000_000:
            score += 0.55
            reasons.append("unusually high transfer rate")
        if 40 <= mean_length <= 1400 and 10 <= packet_count <= 300:
            score -= 0.25
            reasons.append("volume profile within common interactive range")

        risk = _sigmoid(score)
        missing = sum(
            key not in stats
            for key in ("packet_count", "total_bytes", "duration")
        )
        uncertainty = min(0.65, 0.15 + missing * 0.12)
        benign, malicious, confidence = _finalize_support(risk, uncertainty)
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.STATS,
            benign_support=benign,
            malicious_support=malicious,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=reasons or ["no strong statistical anomaly"],
            used_fields=sorted(f"stats.{key}" for key in stats),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class TemporalBehaviorAgent(DetectorAgent):
    name = "TemporalBehaviorAgent"
    version = "2.1"

    def __init__(
        self,
        *,
        backend: str = "rule",
        model_dir: str | Path | None = None,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
    ) -> None:
        if backend not in {
            "rule",
            "learned",
            "deep_v2",
            "deep_v2_all_length",
            "deep_v2_2",
            "deep_v2_3",
            "deep_v2_6_continuous",
            "deep_v3_1_prefix",
            "prefix_robust_v2",
        }:
            raise ValueError(f"unsupported detector backend: {backend}")
        if backend in {
            "learned",
            "deep_v2",
            "deep_v2_all_length",
            "deep_v2_2",
            "deep_v2_3",
            "deep_v2_6_continuous",
            "deep_v3_1_prefix",
        } and model_dir is None:
            raise ValueError(
                "model_dir is required for learned TemporalBehaviorAgent"
            )
        allowed_ood = set(OOD_POLICIES) | {
            "conformal_v3",
            "conformal_v3_1",
            "conformal_v3_2",
            "conformal_v3_3",
            "conformal_v3_4_orbit",
            "conformal_v3_1_prefix",
        }
        if ood_policy not in allowed_ood:
            raise ValueError(f"unsupported OOD policy: {ood_policy}")
        if ood_policy != "off" and ood_gate_dir is None:
            raise ValueError("ood_gate_dir is required when OOD policy is enabled")
        if backend in {"deep_v2", "deep_v2_all_length"} and ood_policy not in {
            "off",
            "conformal_v3",
            "conformal_v3_1",
        }:
            raise ValueError(
                "deep_v2 supports only off, conformal_v3, or conformal_v3_1"
            )
        if backend == "deep_v2_2" and ood_policy not in {
            "off",
            "conformal_v3_2",
        }:
            raise ValueError(
                "deep_v2_2 supports only off or conformal_v3_2"
            )
        if backend == "deep_v2_3" and ood_policy not in {
            "off",
            "conformal_v3_3",
            "conformal_v3_4_orbit",
        }:
            raise ValueError(
                "deep_v2_3 supports off, conformal_v3_3, "
                "or conformal_v3_4_orbit"
            )
        if backend == "deep_v2_6_continuous" and ood_policy != "off":
            raise ValueError(
                "deep_v2_6_continuous selection is preregistered with OOD off"
            )
        if backend == "deep_v3_1_prefix" and ood_policy not in {
            "off",
            "conformal_v3_1_prefix",
        }:
            raise ValueError(
                "deep_v3_1_prefix supports off or conformal_v3_1_prefix"
            )
        if backend not in {
            "deep_v2",
            "deep_v2_all_length",
            "deep_v2_2",
            "deep_v2_3",
            "deep_v2_6_continuous",
            "deep_v3_1_prefix",
        } and ood_policy in {
            "conformal_v3",
            "conformal_v3_1",
            "conformal_v3_2",
            "conformal_v3_3",
            "conformal_v3_4_orbit",
            "conformal_v3_1_prefix",
        }:
            raise ValueError(f"{ood_policy} requires a deep temporal backend")
        self.backend = backend
        self.ood_policy = ood_policy
        if backend == "prefix_robust_v2":
            from .prefix_robust_temporal_v2 import PrefixRobustTemporalV2Backend

            self.candidate_backend = PrefixRobustTemporalV2Backend(model_dir)
        else:
            self.candidate_backend = None
        self.model = (
            LearnedDetectorModel(model_dir, expected_agent="temporal")
            if backend == "learned"
            else None
        )
        if backend in {"deep_v2", "deep_v2_all_length"}:
            from .deep_models import DeepDetectorModel

            self.deep_model = DeepDetectorModel(
                model_dir,
                expected_agent="temporal",
                ood_policy=ood_policy,
                ood_gate_dir=ood_gate_dir,
            )
        else:
            self.deep_model = None
        if backend == "deep_v2_2":
            from .short_flow import DeepV22TemporalModel

            self.deep_model = DeepV22TemporalModel(
                model_dir,
                ood_policy=ood_policy,
                ood_gate_dir=ood_gate_dir,
            )
        if backend == "deep_v2_3":
            from .short_flow_robust import DeepV23TemporalModel

            self.deep_model = DeepV23TemporalModel(
                model_dir,
                ood_policy=ood_policy,
                ood_gate_dir=ood_gate_dir,
            )
        if backend == "deep_v2_6_continuous":
            from .deep_models import DeepDetectorModel

            self.deep_model = DeepDetectorModel(
                model_dir,
                expected_agent="temporal",
                ood_policy="off",
            )
        if backend == "deep_v3_1_prefix":
            from .deep_models import DeepDetectorModel

            self.deep_model = DeepDetectorModel(
                model_dir,
                expected_agent="temporal",
                ood_policy=ood_policy,
                ood_gate_dir=ood_gate_dir,
            )
        self.ood_gate = (
            OODReliabilityGate(ood_gate_dir, expected_agent="temporal")
            if backend == "learned" and ood_gate_dir is not None
            else None
        )

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        started = time.perf_counter()
        seq = detector_input.sequence
        minimum_packets = (
            1
            if self.backend in {
                "deep_v2_all_length",
                "deep_v2_2",
                "deep_v2_3",
                "deep_v2_6_continuous",
                "deep_v3_1_prefix",
                "prefix_robust_v2",
            }
            else 4
        )
        if len(seq.packet_lengths) < minimum_packets:
            return AgentEvidence(
                agent_name=self.name,
                feature_group=FeatureGroup.SEQUENCE,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                evidence=[
                    (
                        "packet sequence unavailable"
                        if not seq.packet_lengths
                        else "short packet sequence unsupported by backend"
                    )
                ],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.backend == "prefix_robust_v2":
            reason = self.candidate_backend.analyze_v2(detector_input)
            return AgentEvidence(
                agent_name=self.name,
                agent_version="4.2-candidate",
                feature_group=FeatureGroup.SEQUENCE,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                calibration_quality=0,
                model_reliability=0,
                abstained=True,
                contributes_to_verdict=False,
                evidence=[reason.unsupported_reason or "candidate unavailable"],
                used_fields=[],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.deep_model is not None:
            prediction = self.deep_model.predict(detector_input)
            return _deep_prediction_evidence(
                prediction,
                agent_name=self.name,
                agent_version=(
                    "3.3"
                    if self.backend == "deep_v2_3"
                    else "4.1"
                    if self.backend == "deep_v3_1_prefix"
                    else "3.6"
                    if self.backend == "deep_v2_6_continuous"
                    else "3.2"
                    if self.backend == "deep_v2_2"
                    else "3.0"
                ),
                feature_group=FeatureGroup.SEQUENCE,
                used_fields=[
                    "sequence.packet_lengths",
                    "sequence.directions",
                    "sequence.iats",
                ],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.model is not None:
            prediction = self.model.predict(detector_input)
            benign, malicious, confidence = _finalize_support(
                prediction.malicious_probability,
                prediction.uncertainty,
            )
            policy_text = (
                f"calibrated {prediction.accepted_class} region"
                if prediction.accepted_class
                else "calibrated reject region"
            )
            assessment = (
                self.ood_gate.assess(detector_input)
                if self.ood_gate is not None
                else None
            )
            base_evidence = AgentEvidence(
                agent_name=self.name,
                agent_version=self.version,
                feature_group=FeatureGroup.SEQUENCE,
                benign_support=benign,
                malicious_support=malicious,
                confidence=confidence,
                uncertainty=prediction.uncertainty,
                calibration_quality=prediction.calibration_quality,
                distribution_shift_score=(
                    assessment.shift_score if assessment else 0
                ),
                distribution_shift_raw_score=(
                    assessment.raw_score if assessment else None
                ),
                distribution_shift_level=(
                    assessment.level if assessment else "off"
                ),
                model_reliability=1,
                evidence=[
                    "learned temporal model with Platt calibration",
                    policy_text,
                    *(
                        [
                            (
                                "model-distribution shift "
                                f"{assessment.level} "
                                f"(score={assessment.shift_score:.3f})"
                            )
                        ]
                        if assessment
                        else []
                    ),
                ],
                used_fields=[
                    "sequence.packet_lengths",
                    "sequence.directions",
                    "sequence.iats",
                    "sequence.bursts",
                    "sequence.original_packet_count",
                    "sequence.truncated",
                ],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return apply_ood_policy_to_evidence(
                base_evidence, self.ood_policy
            )

        score = -1.0
        reasons: list[str] = []
        iats = seq.iats
        directions = seq.directions
        lengths = seq.packet_lengths

        if len(iats) >= 5 and statistics.fmean(iats) > 0:
            cv = statistics.pstdev(iats) / statistics.fmean(iats)
            if cv <= 0.15 and statistics.fmean(iats) >= 1:
                score += 1.8
                reasons.append("stable periodic inter-arrival pattern")
            elif cv >= 2.5:
                score += 0.5
                reasons.append("highly bursty timing pattern")
        if directions:
            switches = sum(
                left != right for left, right in zip(directions, directions[1:])
            )
            switch_ratio = switches / max(len(directions) - 1, 1)
            if switch_ratio <= 0.1:
                score += 0.65
                reasons.append("long unidirectional packet run")
            elif switch_ratio >= 0.75:
                score -= 0.25
                reasons.append("frequent request-response direction changes")
        repeated = max(lengths.count(value) for value in set(lengths)) / len(lengths)
        if len(lengths) >= 8 and repeated >= 0.65:
            score += 0.7
            reasons.append("repeated packet-size template")

        risk = _sigmoid(score)
        uncertainty = 0.18 if len(lengths) >= 8 else 0.35
        benign, malicious, confidence = _finalize_support(risk, uncertainty)
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.SEQUENCE,
            benign_support=benign,
            malicious_support=malicious,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=reasons or ["no strong temporal anomaly"],
            used_fields=[
                "sequence.packet_lengths",
                "sequence.directions",
                "sequence.iats",
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class TLSProtocolAgent(DetectorAgent):
    name = "TLSProtocolAgent"

    def __init__(
        self,
        *,
        backend: str = "rule",
        model_dir: str | Path | None = None,
        ood_policy: str = "off",
        ood_gate_dir: str | Path | None = None,
        contract_native_fields: bool = False,
    ) -> None:
        if backend not in {"rule", "deep_v2", "deep_tls_v4"}:
            raise ValueError(f"unsupported TLS backend: {backend}")
        if backend in {"deep_v2", "deep_tls_v4"} and model_dir is None:
            raise ValueError("model_dir is required for learned TLSProtocolAgent")
        if backend == "deep_v2" and ood_policy not in {
            "off",
            "conformal_v3",
            "conformal_v3_1",
        }:
            raise ValueError(
                "deep_v2 TLS supports only off, conformal_v3, or "
                "conformal_v3_1"
            )
        if backend == "deep_tls_v4" and ood_policy != "off":
            raise ValueError("deep_tls_v4 is preregistered with OOD off")
        if ood_policy in {"conformal_v3", "conformal_v3_1"} and (
            ood_gate_dir is None
        ):
            raise ValueError(f"{ood_policy} requires ood_gate_dir")
        self.backend = backend
        self.contract_native_fields = contract_native_fields
        if backend in {"deep_v2", "deep_tls_v4"}:
            from .deep_models import DeepDetectorModel

            self.deep_model = DeepDetectorModel(
                model_dir,
                expected_agent="tls",
                ood_policy=ood_policy,
                ood_gate_dir=ood_gate_dir,
            )
        else:
            self.deep_model = None

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        started = time.perf_counter()
        tls = detector_input.tls
        if not tls:
            return AgentEvidence(
                agent_name=self.name,
                feature_group=FeatureGroup.TLS,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                evidence=["TLS/QUIC metadata unavailable"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if self.deep_model is not None:
            records = tls.get("record_lengths") or tls.get(
                "tls_record_lengths"
            ) or []
            if len(records) < 2:
                return AgentEvidence(
                    agent_name=self.name,
                    agent_version="3.0",
                    feature_group=FeatureGroup.TLS,
                    benign_support=0,
                    malicious_support=0,
                    confidence=0,
                    uncertainty=1,
                    abstained=True,
                    evidence=["insufficient TLS record sequence"],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            prediction = self.deep_model.predict(detector_input)
            used_fields = ["tls.record_lengths"]
            if self.deep_model.metadata.get("variant") == "records_handshake":
                used_fields.extend(
                    [
                        "tls.server_version",
                        "tls.client_cipher_count",
                        "tls.client_extension_count",
                        "tls.server_extension_count",
                        "tls.alpn",
                    ]
                )
            return _deep_prediction_evidence(
                prediction,
                agent_name=self.name,
                agent_version=(
                    "4.0" if self.backend == "deep_tls_v4" else "3.0"
                ),
                feature_group=FeatureGroup.TLS,
                used_fields=used_fields,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        score = -1.0
        reasons: list[str] = []
        version = str(tls.get("version", "")).lower()
        if version in {"ssl3", "tls1.0", "tlsv1", "tls1.1"}:
            score += 1.0
            reasons.append("legacy TLS version")
        if tls.get("certificate_valid") is False:
            score += 1.25
            reasons.append("invalid or self-signed certificate")
        if tls.get("handshake_complete") is False:
            score += 0.55
            reasons.append("incomplete handshake")
        if (
            not self.contract_native_fields
            and tls.get("known_bad_fingerprint") is True
        ):
            score += 2.0
            reasons.append("fingerprint matched configured threat intelligence")
        if version in {"tls1.3", "tlsv1.3", "quic"} and tls.get(
            "certificate_valid"
        ) is True:
            score -= 0.45
            reasons.append("modern protocol with valid certificate")

        risk = _sigmoid(score)
        consumed = [
            key
            for key in (
                "version",
                "certificate_valid",
                "handshake_complete",
            )
            if key in tls and tls[key] not in (None, "", [], {})
        ]
        uncertainty = (
            0.2
            if (
                len(consumed) >= 3
                if self.contract_native_fields
                else len(tls) >= 3
            )
            else 0.4
        )
        benign, malicious, confidence = _finalize_support(risk, uncertainty)
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.TLS,
            benign_support=benign,
            malicious_support=malicious,
            confidence=confidence,
            uncertainty=uncertainty,
            evidence=reasons or ["no strong protocol anomaly"],
            used_fields=sorted(
                f"tls.{key}"
                for key in (consumed if self.contract_native_fields else tls)
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


FAMILIES = [
    "Cridex",
    "Geodo",
    "Htbot",
    "Miuref",
    "Neris",
    "Nsis-ay",
    "Shifu",
    "Tinba",
    "Virut",
    "Zeus",
]


class FamilyAttributionAgent(DetectorAgent):
    """A deliberately weak MVP heuristic; replace with a calibrated model later."""

    name = "FamilyAttributionAgent"

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        started = time.perf_counter()
        stats = detector_input.stats
        seq = detector_input.sequence
        if not stats or len(seq.packet_lengths) < 4:
            return AgentEvidence(
                agent_name=self.name,
                feature_group=FeatureGroup.STATS,
                benign_support=0,
                malicious_support=0,
                confidence=0,
                uncertainty=1,
                abstained=True,
                contributes_to_verdict=False,
                evidence=["insufficient evidence for family attribution"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        signature = (
            int(stats.get("packet_count", 0))
            + int(stats.get("mean_packet_length", 0) // 32)
            + int(stats.get("duration", 0) // 5)
        )
        primary = signature % len(FAMILIES)
        secondary = (primary + 3) % len(FAMILIES)
        scores = {family: 0.0 for family in FAMILIES}
        scores[FAMILIES[primary]] = 0.46
        scores[FAMILIES[secondary]] = 0.24
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.STATS,
            benign_support=0,
            malicious_support=0,
            confidence=0.46,
            uncertainty=0.54,
            contributes_to_verdict=False,
            evidence=[
                "MVP heuristic family attribution; below production-grade confidence"
            ],
            used_fields=[
                "stats.packet_count",
                "stats.mean_packet_length",
                "stats.duration",
            ],
            family_scores=scores,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class PayloadRepresentationAgent(DetectorAgent):
    name = "PayloadRepresentationAgent"

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.PAYLOAD,
            benign_support=0,
            malicious_support=0,
            confidence=0,
            uncertainty=1,
            abstained=True,
            evidence=["payload representation is disabled in MVP"],
        )


class IntentAgent(DetectorAgent):
    name = "IntentAgent"

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.CONTEXT,
            benign_support=0,
            malicious_support=0,
            confidence=0,
            uncertainty=1,
            abstained=True,
            contributes_to_verdict=False,
            evidence=["USTC-TFC2016 has no reliable intent labels"],
            intent="unknown",
        )
