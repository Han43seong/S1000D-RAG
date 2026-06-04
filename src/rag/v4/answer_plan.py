"""Structured answer planning for v4 Graph RAG.

The answer plan is the contract between symbolic graph/evidence resolution and
LLM natural-language synthesis.  The LLM may verbalize claims, but it should not
invent claims outside this plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from src.rag.ontology import DetailLevel, Intent, ParsedQuery

if TYPE_CHECKING:
    from .graph_builder import GraphContext


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    evidence_dmcs: tuple[str, ...]
    section: str = "근거 기반 설명"


@dataclass(frozen=True)
class AnswerPlan:
    query: str
    intent: Intent
    detail_level: DetailLevel
    audience: str
    claims: tuple[AnswerClaim, ...]
    required_citations: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    sections: tuple[str, ...]


def build_answer_plan(parsed: ParsedQuery, documents: list[Document], graph_context: "GraphContext | None" = None) -> AnswerPlan:
    citations = tuple(dict.fromkeys(str(doc.metadata.get("dmc", "")) for doc in documents if doc.metadata.get("dmc")))
    if graph_context is not None:
        related = graph_context.related_dmcs_for_target(parsed.target)
        citations = tuple(dict.fromkeys((*citations, *related)))
    claims = tuple(_claim_from_document(doc) for doc in documents if doc.page_content.strip() and doc.metadata.get("dmc"))
    if not claims and citations:
        claims = (AnswerClaim(text="관련 S1000D 문서가 확인되었습니다.", evidence_dmcs=citations),)

    return AnswerPlan(
        query=parsed.original,
        intent=parsed.intent,
        detail_level=parsed.detail_level,
        audience=parsed.audience.value if hasattr(parsed.audience, "value") else str(parsed.audience),
        claims=claims,
        required_citations=citations,
        forbidden_claims=_forbidden_claims(parsed.intent),
        sections=_sections_for(parsed),
    )


def _claim_from_document(doc: Document) -> AnswerClaim:
    dmc = str(doc.metadata.get("dmc"))
    text = " ".join(doc.page_content.split())
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    return AnswerClaim(text=text, evidence_dmcs=(dmc,))


def _forbidden_claims(intent: Intent) -> tuple[str, ...]:
    base = ("unsupported facts", "uncited DMCs")
    if intent == Intent.PROCEDURE:
        return base + ("unretrieved procedure steps", "fabricated tools or supplies")
    return base + ("unsupported procedure steps",)


def _sections_for(parsed: ParsedQuery) -> tuple[str, ...]:
    if parsed.intent == Intent.PROCEDURE:
        return ("지원 여부", "절차 근거", "주의사항")
    if parsed.detail_level == DetailLevel.DETAILED:
        return ("구성 관계", "작동 흐름", "정비상 의미", "근거 문서")
    return ("요약", "근거 문서")
