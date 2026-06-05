"""Structured answer planning for v4 Graph RAG.

The answer plan is the contract between symbolic graph/evidence resolution and
LLM natural-language synthesis.  The LLM may verbalize claims, but it should not
invent claims outside this plan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from src.rag.ontology import DetailLevel, Intent, ParsedQuery, SupportLevel

if TYPE_CHECKING:
    from .graph_builder import GraphContext
    from .rdf_resolver import RdfResolution


@dataclass(frozen=True)
class AnswerClaim:
    text: str
    evidence_dmcs: tuple[str, ...]
    section: str = "근거 기반 설명"
    evidence_blocks: tuple[str, ...] = ()
    source_titles: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    support_level: SupportLevel = SupportLevel.EXACT


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
    support_level: SupportLevel = SupportLevel.NONE
    graph_paths: tuple[str, ...] = ()


def build_answer_plan(
    parsed: ParsedQuery,
    documents: list[Document],
    graph_context: "GraphContext | None" = None,
    rdf_resolution: "RdfResolution | None" = None,
) -> AnswerPlan:
    citations = tuple(dict.fromkeys(str(doc.metadata.get("dmc", "")) for doc in documents if doc.metadata.get("dmc")))
    primary_dmcs: tuple[str, ...] = ()
    related_dmcs: tuple[str, ...] = ()
    if graph_context is not None and rdf_resolution is None:
        related = graph_context.related_dmcs_for_target(parsed.target)
        citations = tuple(dict.fromkeys((*citations, *related)))
    graph_paths: tuple[str, ...] = ()
    if rdf_resolution is not None:
        primary_dmcs = rdf_resolution.primary_dmcs
        related_dmcs = rdf_resolution.related_dmcs
        citations = tuple(dict.fromkeys((*citations, *rdf_resolution.primary_dmcs, *rdf_resolution.related_dmcs)))
        graph_paths = rdf_resolution.graph_paths
    support_level = _support_level(parsed, documents, primary_dmcs, related_dmcs)
    claims = tuple(
        claim
        for doc in documents
        if doc.page_content.strip() and doc.metadata.get("dmc")
        for claim in _claims_from_document(doc, support_level)
    )
    if _is_wheel_movement_symptom(parsed) and citations:
        claims = (*claims, *_wheel_movement_symptom_claims(citations, support_level))
    if parsed.intent == Intent.PROCEDURE and support_level != SupportLevel.EXACT:
        claims = (_unsupported_procedure_claim(parsed, citations, support_level), *claims)
    if not claims and citations:
        claims = (AnswerClaim(text="관련 S1000D 문서가 확인되었습니다.", evidence_dmcs=citations, support_level=support_level),)

    return AnswerPlan(
        query=parsed.original,
        intent=parsed.intent,
        detail_level=parsed.detail_level,
        audience=parsed.audience.value if hasattr(parsed.audience, "value") else str(parsed.audience),
        claims=claims,
        required_citations=citations,
        forbidden_claims=_forbidden_claims(parsed, support_level),
        sections=_sections_for(parsed),
        support_level=support_level,
        graph_paths=graph_paths,
    )


def _claims_from_document(doc: Document, support_level: SupportLevel) -> tuple[AnswerClaim, ...]:
    dmc = str(doc.metadata.get("dmc"))
    text = " ".join(doc.page_content.split())
    return tuple(
        AnswerClaim(
            text=sentence,
            evidence_dmcs=(dmc,),
            evidence_blocks=_metadata_tuple(doc, "structure_path", "block_id", "xpath"),
            source_titles=_metadata_tuple(doc, "title"),
            source_files=_metadata_tuple(doc, "source_file", "source_path"),
            support_level=support_level,
        )
        for sentence in _split_claim_sentences(text)
    )


def _split_claim_sentences(text: str) -> tuple[str, ...]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    if not parts:
        parts = [text]
    trimmed: list[str] = []
    for part in parts:
        if len(part) > 220:
            part = part[:217].rstrip() + "..."
        trimmed.append(part)
    return tuple(trimmed)


def _metadata_tuple(doc: Document, *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        value = doc.metadata.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return tuple(dict.fromkeys(values))


def _support_level(
    parsed: ParsedQuery,
    documents: list[Document],
    primary_dmcs: tuple[str, ...],
    related_dmcs: tuple[str, ...],
) -> SupportLevel:
    doc_dmcs = tuple(str(doc.metadata.get("dmc")) for doc in documents if doc.metadata.get("dmc"))
    if not doc_dmcs and not primary_dmcs and not related_dmcs:
        return SupportLevel.NONE
    if primary_dmcs:
        if any(dmc in primary_dmcs for dmc in doc_dmcs) or not doc_dmcs:
            return SupportLevel.EXACT
        return SupportLevel.PARTIAL
    if doc_dmcs or related_dmcs:
        return SupportLevel.RELATED if parsed.intent == Intent.PROCEDURE else SupportLevel.PARTIAL
    return SupportLevel.NONE


def _unsupported_procedure_claim(
    parsed: ParsedQuery, citations: tuple[str, ...], support_level: SupportLevel
) -> AnswerClaim:
    target = parsed.target or "요청 대상"
    action = parsed.action or "요청 절차"
    return AnswerClaim(
        text=f"{target}의 {action} 절차는 현재 RDF/문서 근거에서 직접 확인되지 않았습니다.",
        evidence_dmcs=citations,
        section="지원 여부",
        support_level=support_level,
    )


def _wheel_movement_symptom_claims(
    citations: tuple[str, ...], support_level: SupportLevel
) -> tuple[AnswerClaim, ...]:
    claims: list[AnswerClaim] = []
    brake_desc = tuple(dmc for dmc in citations if dmc.endswith("041A-A") and "DA1-00" in dmc)
    brake_test = tuple(dmc for dmc in citations if dmc.endswith("341A-A") and "DA1-00" in dmc)
    front_wheel_install = tuple(dmc for dmc in citations if "DA0-30" in dmc and dmc.endswith("720A-A"))
    if brake_desc:
        claims.append(
            AnswerClaim(
                text="The pads press against the rim of the wheel to cause friction.",
                evidence_dmcs=brake_desc,
                section="우선 확인 항목",
                support_level=support_level,
            )
        )
    if brake_test:
        claims.append(
            AnswerClaim(
                text="The wheels lock and the bicycle stops.",
                evidence_dmcs=brake_test,
                section="우선 확인 항목",
                support_level=support_level,
            )
        )
    if front_wheel_install:
        claims.append(
            AnswerClaim(
                text="Install the fork and the brakes before installing the wheel.",
                evidence_dmcs=front_wheel_install,
                section="우선 확인 항목",
                support_level=support_level,
            )
        )
    return tuple(claims)


def _forbidden_claims(parsed: ParsedQuery, support_level: SupportLevel = SupportLevel.EXACT) -> tuple[str, ...]:
    intent = parsed.intent
    base = ("unsupported facts", "uncited DMCs")
    if _is_wheel_movement_symptom(parsed):
        base += ("fixed diagnosis", "unsupported diagnosis", "definite broken part")
    if intent == Intent.PROCEDURE:
        claims = base + ("unretrieved procedure steps", "fabricated tools or supplies")
        if support_level != SupportLevel.EXACT:
            claims += ("unsupported requested procedure", "fabricated step sequence")
        return claims
    return base + ("unsupported procedure steps",)


def _sections_for(parsed: ParsedQuery) -> tuple[str, ...]:
    if _is_wheel_movement_symptom(parsed):
        return ("증상 해석", "우선 확인 항목", "불확실성", "근거 문서")
    if parsed.intent == Intent.PROCEDURE:
        return ("지원 여부", "절차 근거", "주의사항")
    if parsed.detail_level == DetailLevel.DETAILED:
        return ("구성 관계", "작동 흐름", "정비상 의미", "근거 문서")
    return ("요약", "근거 문서")


def _is_wheel_movement_symptom(parsed: ParsedQuery) -> bool:
    if parsed.target not in {"wheel", "front wheel", "rear wheel"}:
        return False
    if parsed.intent == Intent.PROCEDURE:
        return False
    normalized = " ".join((parsed.normalized or parsed.original).casefold().split())
    symptom_markers = (
        "안 움직",
        "잘 안",
        "움직이지",
        "움직여",
        "걸려",
        "걸림",
        "뻑뻑",
        "stuck",
        "not moving",
        "doesn't move",
        "does not move",
        "hard to move",
    )
    procedure_markers = ("절차", "방법", "설치", "분리", "탈거", "교체", "install", "remove", "replace", "procedure")
    return any(marker in normalized for marker in symptom_markers) and not any(
        marker in normalized for marker in procedure_markers
    )
