from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .audit import AuditLogger
from .schemas import (
    KnowledgeExplanation,
    KnowledgeQueryContext,
    KnowledgeReference,
    KnowledgeSupportResult,
)


KNOWLEDGE_SCHEMA_VERSION = "1.0"
DEFAULT_ALLOWED_DOCUMENTS = (
    "agent_capabilities.md",
    "detection_playbook.md",
    "malicious_intent_patterns.md",
    "mitre_attack_mapping.md",
    "tls_anomaly_notes.md",
    "ood_and_unknown_policy.md",
    "response_actions.md",
)
DISABLED_DOCUMENTS = ("malware_family_notes.md",)
SUPPORTED_TOPICS = (
    "c2_beaconing",
    "possible_exfiltration",
    "tls_anomaly",
    "suspicious_unknown_policy",
    "ood_reliability_policy",
    "response_action_rationale",
    "unlabeled_evaluation_policy",
)
TOPIC_QUERIES = {
    "c2_beaconing": (
        "periodic inter arrival time stable direction pattern beaconing "
        "command and control C2 encrypted traffic"
    ),
    "possible_exfiltration": (
        "high outbound upload ratio unusual burst data exfiltration encrypted traffic"
    ),
    "tls_anomaly": (
        "TLS QUIC handshake certificate cipher fingerprint anomaly interpretation"
    ),
    "suspicious_unknown_policy": (
        "difference between suspicious and unknown insufficient conflicting evidence"
    ),
    "ood_reliability_policy": (
        "out of distribution severe distribution shift reliability discount abstain unknown"
    ),
    "response_action_rationale": (
        "increase monitoring analyst review retain preserve isolate blocking "
        "endpoint telemetry forensic response action rationale"
    ),
    "unlabeled_evaluation_policy": (
        "unlabeled external dataset cannot calculate accuracy F1 PR AUC"
    ),
}
TOPIC_DOCUMENT_HINTS = {
    "c2_beaconing": {"malicious_intent_patterns", "detection_playbook"},
    "possible_exfiltration": {"malicious_intent_patterns", "detection_playbook"},
    "tls_anomaly": {"tls_anomaly_notes"},
    "suspicious_unknown_policy": {"ood_and_unknown_policy"},
    "ood_reliability_policy": {"ood_and_unknown_policy"},
    "response_action_rationale": {"response_actions"},
    "unlabeled_evaluation_policy": {"ood_and_unknown_policy"},
}
TOPIC_STATEMENTS = {
    "c2_beaconing": (
        "Periodic inter-arrival timing and stable direction patterns can be "
        "consistent with beacon-like command-and-control behavior, but they do "
        "not establish malicious intent by themselves."
    ),
    "possible_exfiltration": (
        "A high outbound ratio combined with unusual bursts can be consistent "
        "with data staging or exfiltration, but benign uploads and backups are "
        "plausible alternatives."
    ),
    "tls_anomaly": (
        "TLS metadata anomalies are contextual indicators whose meaning depends "
        "on handshake completeness, deployment policy, and feature reliability."
    ),
    "suspicious_unknown_policy": (
        "Suspicious means risk evidence exists but is not decisive; unknown "
        "means the available evidence is too incomplete, unreliable, shifted, "
        "or conflicting to support a directional conclusion."
    ),
    "ood_reliability_policy": (
        "Severe distribution shift reduces trust in detector outputs and can "
        "justify abstention or an unknown result without changing the measured "
        "OOD score."
    ),
    "unlabeled_evaluation_policy": (
        "Without ground-truth labels, classification metrics such as accuracy, "
        "F1, and PR-AUC are undefined and must be skipped rather than inferred."
    ),
}


def fusion_sha256(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "section"


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    document_id: str
    source_path: str
    heading: str
    chunk_id: str
    content: str
    content_sha256: str


class KnowledgeRetrievalService:
    """Deterministic, local-only knowledge retrieval for report explanation."""

    name = "KnowledgeRetrievalService"
    version = "1.0"

    def __init__(
        self,
        knowledge_base_dir: str | Path,
        *,
        top_k: int = 4,
        min_score: float = 0.05,
        allowed_documents: tuple[str, ...] = DEFAULT_ALLOWED_DOCUMENTS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score must be between zero and one")
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.top_k = top_k
        self.min_score = min_score
        self.allowed_documents = tuple(allowed_documents)
        if set(self.allowed_documents) & set(DISABLED_DOCUMENTS):
            raise ValueError("malware family notes are disabled in RAG MVP")
        if any(Path(name).name != name for name in self.allowed_documents):
            raise ValueError("knowledge documents must be direct child file names")

    def retrieve(
        self,
        context: KnowledgeQueryContext,
        audit: AuditLogger | None = None,
    ) -> KnowledgeSupportResult:
        started = time.perf_counter()
        retrieval_id = _sha256_text(
            f"{context.trace_id}:{context.fusion_sha256}:"
            f"{','.join(context.query_topics)}"
        )[:20]
        self._log(
            audit,
            "RAG_QUERY_BUILT",
            input_summary={
                "context_schema": context.schema_version,
                "fusion_sha256": context.fusion_sha256,
                "query_topics": context.query_topics,
                "field_paths_only": [item.path for item in context.field_audit],
            },
            output_summary={"retrieval_id": retrieval_id},
            reason="Built a value-free, post-fusion explanatory query.",
        )
        try:
            self._validate_context(context)
            self._log(
                audit,
                "RAG_POLICY_CHECKED",
                input_summary={
                    "allowed_documents": list(self.allowed_documents),
                    "disabled_documents": list(DISABLED_DOCUMENTS),
                },
                output_summary={
                    "status": "accepted",
                    "blocked_field_value_count": 0,
                    "family_knowledge_enabled": False,
                    "network_enabled": False,
                },
                reason="RAG input and local knowledge allowlist passed policy checks.",
            )
            chunks, version, base_hash = self._load_chunks()
            references = self._search(context, chunks)
            result = self._build_result(
                context,
                references,
                retrieval_id=retrieval_id,
                knowledge_base_version=version,
                knowledge_base_sha256=base_hash,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            self._log(
                audit,
                "RAG_RETRIEVAL_COMPLETED",
                input_summary={
                    "knowledge_base_sha256": base_hash,
                    "retriever_config": result.retriever_config,
                },
                output_summary={
                    "status": result.status,
                    "references": [
                        item.model_dump(mode="json") for item in result.references
                    ],
                },
                reason="Completed deterministic local TF-IDF retrieval.",
                duration_ms=result.latency_ms,
            )
            self._validate_result(result)
            self._log(
                audit,
                "RAG_OUTPUT_VALIDATED",
                input_summary={"retrieval_id": retrieval_id},
                output_summary={
                    "schema_version": result.schema_version,
                    "explanatory_only": result.explanatory_only,
                    "reference_count": len(result.references),
                    "forbidden_output_fields": [],
                    "fusion_sha256": context.fusion_sha256,
                },
                reason="RAG output contains cited explanation only.",
            )
            return result
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            self._log(
                audit,
                "RAG_RETRIEVAL_FAILED",
                input_summary={"retrieval_id": retrieval_id},
                output_summary={
                    "error_type": type(exc).__name__,
                    "status": "failed",
                },
                reason=str(exc),
                duration_ms=latency_ms,
            )
            return KnowledgeSupportResult(
                status="failed",
                retrieval_id=retrieval_id,
                retriever_config=self._retriever_config(),
                query_topics=context.query_topics,
                limitations=[
                    "Knowledge enhancement failed; the detection result is unchanged.",
                    f"Failure type: {type(exc).__name__}",
                ],
                latency_ms=latency_ms,
            )

    def _validate_context(self, context: KnowledgeQueryContext) -> None:
        if fusion_sha256(context.fusion_snapshot) != context.fusion_sha256:
            raise ValueError("fusion snapshot checksum mismatch")
        unsupported = sorted(set(context.query_topics) - set(SUPPORTED_TOPICS))
        if unsupported:
            raise ValueError(f"unsupported RAG query topics: {unsupported}")
        if any(item.path == "" for item in context.field_audit):
            raise ValueError("field audit summaries require field paths")

    def _load_chunks(self) -> tuple[list[KnowledgeChunk], str, str]:
        root = self.knowledge_base_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"knowledge base directory not found: {root}")
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = str(manifest.get("knowledge_base_version", ""))
        if not version:
            raise ValueError("knowledge base manifest has no version")

        chunks: list[KnowledgeChunk] = []
        hash_parts: list[str] = [
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        ]
        for name in self.allowed_documents:
            path = (root / name).resolve()
            if path.parent != root:
                raise ValueError("knowledge path escapes the configured root")
            text = path.read_text(encoding="utf-8")
            hash_parts.append(f"{name}\n{text}")
            chunks.extend(self._chunk_markdown(name, text))
        if not chunks:
            raise ValueError("knowledge base contains no retrievable chunks")
        return chunks, version, _sha256_text("\n".join(hash_parts))

    @staticmethod
    def _chunk_markdown(name: str, text: str) -> list[KnowledgeChunk]:
        document_id = Path(name).stem
        chunks: list[KnowledgeChunk] = []
        heading = document_id.replace("_", " ")
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if not content:
                return
            index = len(chunks) + 1
            chunk_id = f"{document_id}-{_slug(heading)}-{index:03d}"
            chunks.append(
                KnowledgeChunk(
                    document_id=document_id,
                    source_path=name,
                    heading=heading,
                    chunk_id=chunk_id,
                    content=content,
                    content_sha256=_sha256_text(content),
                )
            )

        for line in text.splitlines():
            match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
            if match:
                flush()
                body = []
                heading = match.group(2).strip()
            elif not line.startswith("# "):
                body.append(line)
        flush()
        return chunks

    def _search(
        self,
        context: KnowledgeQueryContext,
        chunks: list[KnowledgeChunk],
    ) -> list[KnowledgeReference]:
        if not context.query_topics:
            return []
        evidence_text = " ".join(
            evidence
            for agent in context.agents
            for evidence in agent.evidence[:3]
        )
        corpus = [chunk.content for chunk in chunks]
        topic_queries = [
            TOPIC_QUERIES[topic]
            for topic in context.query_topics
        ]
        combined_query = f"{' '.join(topic_queries)} {evidence_text}"
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        matrix = vectorizer.fit_transform(
            [*corpus, *topic_queries, combined_query]
        )
        corpus_matrix = matrix[: len(corpus)]
        selected_indices: list[int] = []
        selected_scores: dict[int, float] = {}
        for offset, topic in enumerate(context.query_topics):
            scores = cosine_similarity(
                matrix[len(corpus) + offset],
                corpus_matrix,
            ).ravel()
            hints = TOPIC_DOCUMENT_HINTS.get(topic, set())
            candidates = [
                index
                for index, chunk in enumerate(chunks)
                if not hints or chunk.document_id in hints
            ]
            ranked = sorted(
                candidates,
                key=lambda index: (
                    -float(scores[index]),
                    chunks[index].source_path,
                    chunks[index].chunk_id,
                ),
            )
            if not ranked:
                continue
            index = ranked[0]
            score = float(np.clip(scores[index], 0, 1))
            if score < self.min_score:
                continue
            if index not in selected_scores:
                selected_indices.append(index)
            selected_scores[index] = max(selected_scores.get(index, 0), score)
            if len(selected_indices) >= self.top_k:
                break

        if len(selected_indices) < self.top_k:
            combined_scores = cosine_similarity(
                matrix[-1],
                corpus_matrix,
            ).ravel()
            ranked = sorted(
                range(len(chunks)),
                key=lambda index: (
                    -float(combined_scores[index]),
                    chunks[index].source_path,
                    chunks[index].chunk_id,
                ),
            )
            for index in ranked:
                score = float(np.clip(combined_scores[index], 0, 1))
                if score < self.min_score:
                    continue
                if index not in selected_scores:
                    selected_indices.append(index)
                    selected_scores[index] = score
                if len(selected_indices) >= self.top_k:
                    break

        selected = [
            (chunks[index], selected_scores[index])
            for index in selected_indices
        ]
        return [
            KnowledgeReference(
                reference_id=f"KB-{position:03d}",
                document_id=chunk.document_id,
                source_path=chunk.source_path,
                heading=chunk.heading,
                chunk_id=chunk.chunk_id,
                content_sha256=chunk.content_sha256,
                retrieval_score=score,
                excerpt=chunk.content[:500].strip(),
            )
            for position, (chunk, score) in enumerate(selected, start=1)
        ]

    def _build_result(
        self,
        context: KnowledgeQueryContext,
        references: list[KnowledgeReference],
        *,
        retrieval_id: str,
        knowledge_base_version: str,
        knowledge_base_sha256: str,
        latency_ms: float,
    ) -> KnowledgeSupportResult:
        reference_ids = [item.reference_id for item in references]
        by_document: dict[str, list[str]] = {}
        for item in references:
            by_document.setdefault(item.document_id, []).append(item.reference_id)

        def refs_for(topic: str) -> list[str]:
            hinted = TOPIC_DOCUMENT_HINTS.get(topic, set())
            matched = [
                reference_id
                for document in sorted(hinted)
                for reference_id in by_document.get(document, [])
            ]
            return matched or reference_ids[:1]

        support: list[KnowledgeExplanation] = []
        intent: list[KnowledgeExplanation] = []
        policy: list[KnowledgeExplanation] = []
        if references:
            for topic in context.query_topics:
                statement = TOPIC_STATEMENTS.get(topic)
                topic_refs = refs_for(topic)
                if not statement or not topic_refs:
                    continue
                explanation = KnowledgeExplanation(
                    statement=statement,
                    epistemic_status=(
                        "possible_interpretation"
                        if topic in {"c2_beaconing", "possible_exfiltration"}
                        else "policy_explanation"
                        if topic
                        in {
                            "suspicious_unknown_policy",
                            "ood_reliability_policy",
                            "unlabeled_evaluation_policy",
                        }
                        else "background"
                    ),
                    reference_ids=topic_refs,
                )
                if topic in {"c2_beaconing", "possible_exfiltration"}:
                    intent.append(explanation)
                elif explanation.epistemic_status == "policy_explanation":
                    policy.append(explanation)
                else:
                    support.append(explanation)

        response_refs = refs_for("response_action_rationale") if references else []
        response = [
            KnowledgeExplanation(
                statement=(
                    f"The existing response action '{action}' is unchanged; "
                    "the cited playbook provides explanatory rationale only."
                ),
                epistemic_status="background",
                reference_ids=response_refs,
            )
            for action in context.recommended_actions
            if response_refs
        ]
        return KnowledgeSupportResult(
            status="success" if references else "no_match",
            retrieval_id=retrieval_id,
            knowledge_base_version=knowledge_base_version,
            knowledge_base_sha256=knowledge_base_sha256,
            retriever_config=self._retriever_config(),
            query_topics=context.query_topics,
            knowledge_support=support,
            possible_intent_explanations=intent,
            response_rationales=response,
            policy_explanations=policy,
            references=references,
            limitations=[
                "All retrieved material is explanatory_only and is not detection evidence.",
                "Knowledge retrieval cannot modify FusionResult, OOD state, field policy, family attribution, or response actions.",
            ],
            latency_ms=latency_ms,
        )

    def _validate_result(self, result: KnowledgeSupportResult) -> None:
        known = {item.reference_id for item in result.references}
        explanations = [
            *result.knowledge_support,
            *result.possible_intent_explanations,
            *result.response_rationales,
            *result.policy_explanations,
        ]
        for explanation in explanations:
            if not set(explanation.reference_ids) <= known:
                raise ValueError("RAG explanation contains an invalid reference")
        forbidden = {
            "verdict",
            "confidence",
            "uncertainty",
            "severity",
            "coverage",
            "family",
            "family_confidence",
            "recommended_actions",
            "distribution_shift_level",
        }
        if forbidden & set(result.model_dump()):
            raise ValueError("RAG result exposes forbidden decision fields")

    def _retriever_config(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "min_score": self.min_score,
            "vectorizer": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "allowed_documents": list(self.allowed_documents),
            "disabled_documents": list(DISABLED_DOCUMENTS),
            "network_enabled": False,
            "generation_model": None,
        }

    def _log(
        self,
        audit: AuditLogger | None,
        event_type: str,
        *,
        input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None,
        reason: str = "",
        duration_ms: float = 0,
    ) -> None:
        if audit is not None:
            audit.log(
                self.name,
                event_type,
                input_summary=input_summary,
                output_summary=output_summary,
                reason=reason,
                duration_ms=duration_ms,
            )
