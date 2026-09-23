from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .schemas import (
    DatasetAuditReport,
    DatasetFieldProfile,
    DetectorInput,
    FieldAuditResult,
    FieldDecision,
    FieldRole,
    FlowRecord,
)
from .field_contract import contract_entry


BLOCKED_TOKENS = {
    "id",
    "record_id",
    "cesnet_id",
    "label",
    "target",
    "class",
    "ground_truth",
    "split",
    "fold",
    "is_malicious",
    "family",
    "application",
    "app",
    "os",
}
PROVENANCE_TOKENS = {
    "source_file",
    "capture_file",
    "pcap",
    "dataset",
    "provenance",
    "scenario",
}
CONTEXT_TOKENS = {
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "src_port",
    "dst_port",
    "source_port",
    "destination_port",
    "timestamp",
    "time",
    "session_id",
    "flow_id",
    "stream_id",
    "endpoint",
    "service",
    "sni",
}

def _token_match(path: str, candidates: set[str]) -> bool:
    normalized = path.lower().replace("-", "_")
    leaf = normalized.rsplit(".", 1)[-1]
    return leaf in candidates or any(
        normalized.endswith(f".{candidate}") for candidate in candidates
    )


def _flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(item, path)
    elif isinstance(value, list):
        yield prefix, value
    else:
        yield prefix, value


@dataclass(slots=True)
class AuditPolicy:
    mode: Literal["strict", "legacy"] = "strict"
    contract_version: Literal["2.8", "2.9"] = "2.8"
    allow_tls_sni: bool = False
    block_high_cardinality_context: bool = True


class FieldAuditAgent:
    name = "FieldAuditAgent"
    version = "1.0"

    def __init__(self, policy: AuditPolicy | None = None) -> None:
        self.policy = policy or AuditPolicy()

    def audit(self, flow: FlowRecord) -> FieldAuditResult:
        decisions: list[FieldDecision] = []
        ignored_empty: list[str] = []
        for path, value in _flatten(flow.model_dump()):
            if value in (None, "", [], {}):
                ignored_empty.append(path)
                continue
            role, risk, reasons = self.classify_field(path)
            decisions.append(
                FieldDecision(path=path, role=role, risk_score=risk, reasons=reasons)
            )

        blocked = sorted(
            decision.path
            for decision in decisions
            if decision.role in {FieldRole.BLOCKED, FieldRole.LABEL_ONLY}
        )
        context_only = sorted(
            decision.path
            for decision in decisions
            if decision.role in {FieldRole.CONTEXT_ONLY, FieldRole.PROVENANCE}
        )
        allowed = sorted(
            decision.path
            for decision in decisions
            if decision.role == FieldRole.DETECTION_ALLOWED
        )
        unknown = sorted(
            decision.path
            for decision in decisions
            if decision.role == FieldRole.UNKNOWN
        )
        risks = [decision.risk_score for decision in decisions]
        leakage_risk = max(risks, default=0)
        warnings = [
            f"{decision.path}: {', '.join(decision.reasons)}"
            for decision in decisions
            if decision.risk_score >= 0.7
        ]
        return FieldAuditResult(
            decisions=decisions,
            allowed_fields=allowed,
            blocked_fields=blocked,
            context_only_fields=context_only,
            unknown_fields=unknown,
            ignored_empty_fields=sorted(ignored_empty),
            leakage_risk=leakage_risk,
            warnings=warnings,
        )

    def classify_field(self, path: str) -> tuple[FieldRole, float, list[str]]:
        lower = path.lower()
        if lower.startswith("labels.") or _token_match(path, BLOCKED_TOKENS):
            return FieldRole.LABEL_ONLY, 1.0, ["label or label-proxy field"]
        if lower.startswith("provenance.") or _token_match(path, PROVENANCE_TOKENS):
            return FieldRole.PROVENANCE, 0.9, ["capture provenance can identify class"]
        if _token_match(path, CONTEXT_TOKENS):
            if path.lower().endswith(".sni") and self.policy.allow_tls_sni:
                return (
                    FieldRole.DETECTION_ALLOWED,
                    0.45,
                    ["SNI explicitly allowed but remains environment-sensitive"],
                )
            return (
                FieldRole.CONTEXT_ONLY,
                0.75,
                ["identifier, endpoint, service, or time context"],
            )
        if lower in {"trace_id", "sample_id", "schema_version"}:
            return FieldRole.BLOCKED, 0.95, ["runtime identifier, not behavior"]
        if self.policy.mode == "strict":
            entry = contract_entry(lower, self.policy.contract_version)
            if entry is not None:
                role = FieldRole(entry["role"])
                risk = (
                    0.1
                    if role == FieldRole.DETECTION_ALLOWED
                    else 0.75
                    if role == FieldRole.CONTEXT_ONLY
                    else 0.95
                )
                return role, risk, [
                    (
                        f"field-contract-v{self.policy.contract_version}: "
                        f"{entry['reason']}"
                    )
                ]
            return FieldRole.UNKNOWN, 0.85, [
                "populated field is absent from "
                f"field-contract-v{self.policy.contract_version}"
            ]
        return FieldRole.DETECTION_ALLOWED, 0.1, []

    def make_safe_flow(
        self, flow: FlowRecord, result: FieldAuditResult
    ) -> FlowRecord:
        safe = flow.model_copy(deep=True)
        safe.labels = {}
        safe.provenance = {}
        safe.context = {
            key: value
            for key, value in safe.context.items()
            if f"context.{key}" in result.allowed_fields
        }
        safe.tls = {
            key: value
            for key, value in safe.tls.items()
            if f"tls.{key}" in result.allowed_fields
        }
        safe.stats = {
            key: value
            for key, value in safe.stats.items()
            if f"stats.{key}" in result.allowed_fields
        }
        if "payload_tokens" not in result.allowed_fields:
            safe.payload_tokens = None
        return safe

    def make_detector_input(
        self,
        flow: FlowRecord,
        result: FieldAuditResult,
        *,
        enforce: bool = True,
    ) -> DetectorInput:
        if not enforce:
            return DetectorInput(
                stats=flow.stats,
                sequence=flow.sequence,
                tls=flow.tls,
                payload_tokens=flow.payload_tokens,
                context=flow.context,
            )
        safe = self.make_safe_flow(flow, result)
        return DetectorInput(
            stats=safe.stats,
            sequence=safe.sequence,
            tls=safe.tls,
            payload_tokens=safe.payload_tokens,
            context=safe.context,
        )

    def profile_dataset(
        self, rows: list[dict[str, Any]], label_field: str
    ) -> DatasetAuditReport:
        if not rows:
            return DatasetAuditReport(
                row_count=0,
                label_field=label_field,
                fields=[],
                duplicate_row_ratio=0,
                high_risk_fields=[],
            )

        flattened_rows = [dict(_flatten(row)) for row in rows]
        fields = sorted({key for row in flattened_rows for key in row})
        labels = [str(row.get(label_field, "")) for row in flattened_rows]
        profiles: list[DatasetFieldProfile] = []

        for field_name in fields:
            raw_values = [row.get(field_name) for row in flattened_rows]
            values = [self._stable_value(value) for value in raw_values]
            unique_ratio = len(set(values)) / len(values)
            role, base_risk, reasons = self.classify_field(field_name)
            purity = None
            enough_rows_for_statistics = len(rows) >= 10
            if (
                enough_rows_for_statistics
                and field_name != label_field
                and any(labels)
            ):
                purity = self._label_purity(values, labels)
                if purity >= 0.98 and len(set(values)) <= max(50, len(values) // 4):
                    base_risk = max(base_risk, 0.95)
                    reasons.append("field nearly determines the label")
                    role = FieldRole.BLOCKED
            scalar_field = all(
                value is None or isinstance(value, (str, int, float, bool))
                for value in raw_values
            )
            if (
                enough_rows_for_statistics
                and scalar_field
                and unique_ratio >= 0.98
                and self.policy.block_high_cardinality_context
            ):
                base_risk = max(base_risk, 0.8)
                reasons.append("near-unique identifier risk")
                if role == FieldRole.DETECTION_ALLOWED:
                    role = FieldRole.CONTEXT_ONLY
            profiles.append(
                DatasetFieldProfile(
                    field=field_name,
                    role=role,
                    unique_ratio=unique_ratio,
                    label_purity=purity,
                    risk_score=base_risk,
                    reasons=reasons,
                )
            )

        hashes = [
            hashlib.sha256(
                json.dumps(row, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            for row in rows
        ]
        duplicate_ratio = 1 - len(set(hashes)) / len(hashes)
        return DatasetAuditReport(
            row_count=len(rows),
            label_field=label_field,
            fields=profiles,
            duplicate_row_ratio=duplicate_ratio,
            high_risk_fields=[
                profile.field for profile in profiles if profile.risk_score >= 0.7
            ],
        )

    @staticmethod
    def _label_purity(values: list[str], labels: list[str]) -> float:
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for value, label in zip(values, labels, strict=True):
            grouped[value][label] += 1
        correct = sum(counter.most_common(1)[0][1] for counter in grouped.values())
        return correct / len(values)

    @staticmethod
    def _stable_value(value: Any) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
