import pytest
from pathlib import Path
import importlib.util

from mad_etd.field_audit import FieldAuditAgent
from mad_etd.io import load_flow_records


def pytest_collection_modifyitems(items):
    """Expose unavailable private-input checks as skips, never as passed tests."""
    base = Path(__file__).resolve().parents[1]
    private_inputs = {
        "test_performance_manifest_freezes_disjoint_groups_and_strong_gate": [
            base / "data/runs/mad_etd_hybrid_multiagent_evidence_v1/per_mode_predictions.npz"],
        "test_v20_runtime_hash_and_safety_invariants_are_unchanged": [
            base / "data/runs/mad_etd_thesis_final_experiment_closure_v20/source_artifact_manifest.json",
            base / "data/runs/mad_etd_thesis_final_experiment_closure_v20/artifact_manifest.json",
            base / "data/runs/mad_etd_thesis_final_experiment_closure_v20/ustc_probability_average_bootstrap.json"],
    }
    for item in items:
        missing = [p.relative_to(base).as_posix() for p in private_inputs.get(item.name, []) if not p.exists()]
        if missing:
            item.add_marker(pytest.mark.skip(reason="Private inputs not bundled: " + ", ".join(missing)))
        if item.name in {"test_missing_gating_weights_zero", "test_edl_finite_and_backpropagates",
                         "test_gating_masks_and_renormalizes", "test_network_parameter_counts"}:
            if importlib.util.find_spec("torch") is None:
                item.add_marker(pytest.mark.skip(reason="Optional PyTorch dependency not installed; see the v2 extra"))


@pytest.fixture
def sample_flows():
    return load_flow_records("examples/sample_flows.jsonl")


@pytest.fixture
def benign_flow(sample_flows):
    return sample_flows[0].model_copy(deep=True)


@pytest.fixture
def suspicious_flow(sample_flows):
    return sample_flows[1].model_copy(deep=True)


@pytest.fixture
def benign_detector_input(benign_flow):
    auditor = FieldAuditAgent()
    result = auditor.audit(benign_flow)
    return auditor.make_detector_input(benign_flow, result)
