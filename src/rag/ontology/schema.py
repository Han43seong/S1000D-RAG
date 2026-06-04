"""Ontology-first RAG v2 data contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    DESCRIBE = "describe"
    LIST_COMPONENTS = "list_components"
    PROCEDURE = "procedure"
    DMC_LOOKUP = "dmc_lookup"
    DOCUMENT_SUMMARY = "document_summary"
    FAULT = "fault"
    VISUAL = "visual"
    FOLLOW_UP = "follow_up"
    UNKNOWN = "unknown"


class SupportLevel(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    RELATED = "related"
    NONE = "none"


class DetailLevel(StrEnum):
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"


class Audience(StrEnum):
    GENERAL = "general"
    TECHNICIAN = "technician"
    EXPERT = "expert"


class AnswerMode(StrEnum):
    EXPLANATION = "explanation"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    LOOKUP = "lookup"


@dataclass(frozen=True)
class OntologyNode:
    dmc: str
    title: str
    dm_type: str
    sns_code: str | None = None
    target: str | None = None
    action: str | None = None
    applicability: str | None = None
    source_file: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    normalized: str
    intent: Intent
    target: str | None = None
    action: str | None = None
    dm_type: str | None = None
    detail_level: DetailLevel = DetailLevel.NORMAL
    audience: Audience = Audience.GENERAL
    answer_mode: AnswerMode = AnswerMode.EXPLANATION
    requested_sections: tuple[str, ...] = ()
    referenced_dmcs: tuple[str, ...] = ()
    confidence: float = 0.0
    matched_aliases: tuple[str, ...] = ()
    follow_up: bool = False


@dataclass(frozen=True)
class CandidateEvidence:
    node: OntologyNode
    support: SupportLevel
    reason: str
    score: float = 0.0


@dataclass(frozen=True)
class ResolutionResult:
    parsed: ParsedQuery
    support: SupportLevel
    candidates: tuple[CandidateEvidence, ...]
    reason: str


@dataclass(frozen=True)
class EvidencePlan:
    resolution: ResolutionResult
    dmcs: tuple[str, ...]
    retrieval_query: str
    max_chunks: int = 6


@dataclass(frozen=True)
class QualityGateResult:
    ok: bool
    reasons: tuple[str, ...] = ()
