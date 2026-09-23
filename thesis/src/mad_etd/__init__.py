"""Audit-robust multi-agent encrypted traffic detection."""

from .engine import DetectionEngine, build_default_engine
from .memory import JsonlCaseMemory
from .schemas import (
    DetectionReport,
    EvidenceUtilityEstimate,
    FlowRecord,
    FutureFeatureFlags,
    TLSRecordSequence,
    Verdict,
)

__all__ = [
    "DetectionEngine",
    "DetectionReport",
    "EvidenceUtilityEstimate",
    "FlowRecord",
    "FutureFeatureFlags",
    "JsonlCaseMemory",
    "TLSRecordSequence",
    "Verdict",
    "build_default_engine",
]

__version__ = "0.1.0"
