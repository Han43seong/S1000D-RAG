"""Ontology-first RAG v2 pipeline.

The v2 path resolves DMCs and support level before retrieval/composition.  The
local LLM is deliberately optional; deterministic composition handles the demo's
technical-document answers and the quality gate rejects malformed artifacts.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langsmith import traceable

from src.types.rag import RagOptions, RagResult, SessionMeta

from .evidence_trail import collect_reference_materials
from .ontology import (
    check_answer_quality,
    compose_answer,
    load_ontology_manifest,
    parse_query,
    plan_evidence,
    resolve_ontology,
    retrieve_evidence,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM
    from langchain_core.vectorstores import VectorStore
    from sentence_transformers import CrossEncoder


@traceable(run_type="chain", name="rag_v2_pipeline_async")
async def run_rag_query(
    query: str,
    vectorstore: "VectorStore",
    llm: "BaseLLM",
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: "CrossEncoder" | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    return run_rag_query_sync(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        session_meta=session_meta,
        options=options,
        cross_encoder=cross_encoder,
        conversation_history=conversation_history,
    )


@traceable(run_type="chain", name="rag_v2_pipeline")
def run_rag_query_sync(
    query: str,
    vectorstore: "VectorStore" | None = None,
    llm: "BaseLLM" | None = None,
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: "CrossEncoder" | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    parsed = _parse_query_stage(query)
    if parsed.follow_up and conversation_history:
        previous_dmc = _latest_history_dmc(conversation_history)
        if previous_dmc:
            parsed = _parse_query_stage(f"{previous_dmc} 문서 내용 요약")
    nodes = load_ontology_manifest()
    resolution = _resolve_stage(parsed, nodes)
    plan = _plan_stage(resolution, options)
    documents, evidences = _retrieve_stage(plan, vectorstore)
    answer = _compose_stage(resolution, documents)
    gate = _quality_gate_stage(answer)
    if not gate.ok:
        answer = "답변 품질 검사에서 생성 오류가 감지되어 원문 답변을 제공하지 않습니다. 관련 문서를 다시 확인해 주세요."
    return RagResult(answer=answer, evidences=evidences, reference_materials=collect_reference_materials(evidences))


@traceable(run_type="chain", name="rag_v2_parse_query")
def _parse_query_stage(query: str):
    return parse_query(query)


@traceable(run_type="chain", name="rag_v2_resolve_ontology")
def _resolve_stage(parsed, nodes):
    return resolve_ontology(parsed, nodes)


@traceable(run_type="chain", name="rag_v2_plan_evidence")
def _plan_stage(resolution, options: RagOptions | None):
    max_chunks = (options.top_k if options else 6) or 6
    return plan_evidence(resolution, max_chunks=max_chunks)


@traceable(run_type="chain", name="rag_v2_retrieve_evidence")
def _retrieve_stage(plan, vectorstore):
    return retrieve_evidence(plan, vectorstore)


@traceable(run_type="chain", name="rag_v2_compose_answer")
def _compose_stage(resolution, documents):
    return compose_answer(resolution, documents)


@traceable(run_type="chain", name="rag_v2_quality_gate")
def _quality_gate_stage(answer: str):
    return check_answer_quality(answer)


_DMC_RE = re.compile(r"\b[A-Z0-9]+-[A-Z0-9-]+\b")


def _latest_history_dmc(conversation_history: list[tuple[str, str]]) -> str | None:
    for _user, assistant in reversed(conversation_history):
        match = _DMC_RE.search(assistant or "")
        if match:
            return match.group(0)
    return None
