from mad_etd.base import CaseState, Coordinator, DetectorAgent
from mad_etd.engine import DetectionEngine, build_default_engine
from mad_etd.field_audit import FieldAuditAgent
from mad_etd.fusion import FusionAgent
from mad_etd.guard import PseudoFeatureGuardAgent
from mad_etd.reporter import ReporterAgent
from mad_etd.schemas import (
    AgentEvidence,
    CoordinatorAction,
    CoordinatorDecision,
    DetectorInput,
    FeatureGroup,
    FieldRole,
)


def test_forbidden_roles_have_zero_visible_input_fields(suspicious_flow):
    report, audit = build_default_engine().analyze(suspicious_flow)
    forbidden = {
        decision.path
        for event in audit.events
        if event.event_type == "FIELD_POLICY_APPLIED"
        for decision in []
    }
    field_event = next(
        event for event in audit.events if event.event_type == "FIELD_POLICY_APPLIED"
    )
    forbidden = {
        item["path"]
        for item in field_event.output_summary["decisions"]
        if item["role"] in {"BLOCKED", "LABEL_ONLY", "PROVENANCE"}
    }
    agent_visible = {
        path
        for event in audit.events
        if event.event_type == "AGENT_EVIDENCE"
        for path in event.input_summary["visible_fields"]
    }

    assert forbidden.isdisjoint(agent_visible)
    assert report.blocked_fields


def test_every_agent_audit_snapshot_reports_zero_blocked_intersection(
    suspicious_flow,
):
    _, audit = build_default_engine().analyze(suspicious_flow)
    agent_events = [
        event for event in audit.events if event.event_type == "AGENT_EVIDENCE"
    ]

    assert agent_events
    assert all(
        event.input_summary["blocked_field_intersection"] == []
        for event in agent_events
    )


def test_detector_input_schema_structurally_excludes_label_and_provenance():
    fields = set(DetectorInput.model_fields)

    assert "labels" not in fields
    assert "provenance" not in fields
    assert "trace_id" not in fields
    assert "sample_id" not in fields


class _SpyDetector(DetectorAgent):
    name = "SpyDetector"

    def __init__(self):
        self.seen: DetectorInput | None = None

    def analyze(self, detector_input: DetectorInput) -> AgentEvidence:
        self.seen = detector_input
        return AgentEvidence(
            agent_name=self.name,
            feature_group=FeatureGroup.STATS,
            benign_support=0.6,
            malicious_support=0.2,
            confidence=0.6,
            uncertainty=0.2,
        )


class _MutatingCoordinator(Coordinator):
    def decide(self, state: CaseState) -> CoordinatorDecision:
        if state.called_agents:
            return CoordinatorDecision(
                action=CoordinatorAction.STOP_AND_FUSE,
                remaining_budget=state.remaining_budget,
            )
        state.safe_flow.context["src_ip"] = "coordinator-injected"
        state.safe_flow.labels["binary"] = "malicious"
        state.field_audit.blocked_fields.clear()
        return CoordinatorDecision(
            action=CoordinatorAction.DISPATCH,
            agents=["SpyDetector"],
            remaining_budget=state.remaining_budget - 1,
        )


def test_coordinator_cannot_mutate_prebuilt_detector_input(benign_flow):
    spy = _SpyDetector()
    engine = DetectionEngine(
        coordinator=_MutatingCoordinator(),
        detectors=[spy],
        field_auditor=FieldAuditAgent(),
        feature_guard=PseudoFeatureGuardAgent(),
        fusion_agent=FusionAgent(),
        reporter=ReporterAgent(),
    )

    engine.analyze(benign_flow)

    assert spy.seen is not None
    assert "src_ip" not in spy.seen.context
    assert not hasattr(spy.seen, "labels")


def test_no_field_audit_ablation_keeps_hard_label_boundary(benign_flow):
    _, audit = build_default_engine(enable_field_audit=False).analyze(benign_flow)
    field_event = next(
        event for event in audit.events if event.event_type == "FIELD_POLICY_BYPASSED"
    )

    assert "context.src_ip" in field_event.output_summary["detector_visible_fields"]
    assert all(
        not path.startswith(("labels.", "provenance."))
        for path in field_event.output_summary["detector_visible_fields"]
    )


def test_field_audit_classifies_all_three_forbidden_roles(suspicious_flow):
    result = FieldAuditAgent().audit(suspicious_flow)
    roles = {decision.path: decision.role for decision in result.decisions}

    assert roles["trace_id"] == FieldRole.BLOCKED
    assert roles["labels.binary"] == FieldRole.LABEL_ONLY
    assert roles["provenance.source_file"] == FieldRole.PROVENANCE
