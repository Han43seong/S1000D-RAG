"""Ontology-guided Graph RAG v4 pipeline.

This is the first v4 implementation slice.  It preserves the v3 ontology
resolution/retrieval baseline, then adds a structured AnswerPlan and LLM
verbalizer contract.  Later slices should replace manifest-only resolution with
a richer graph resolver and stronger quality gates.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langsmith import traceable

from src.types.rag import RagOptions, RagResult, SessionMeta

from .evidence_trail import collect_reference_materials
from .ontology import check_answer_quality, load_ontology_manifest, parse_query, plan_evidence, resolve_ontology, retrieve_evidence
from .v4 import RdfResolution, build_answer_plan, build_graph_context, build_rdf_ontology_store, verbalize_answer_plan

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM
    from langchain_core.vectorstores import VectorStore
    from sentence_transformers import CrossEncoder


@traceable(run_type="chain", name="rag_v4_pipeline_async")
async def run_rag_query(
    query: str,
    vectorstore: "VectorStore",
    llm: "BaseLLM",
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: "CrossEncoder | None" = None,
    conversation_history: list[tuple[str, str]] | None = None,
    config: dict | None = None,
) -> RagResult:
    return run_rag_query_sync(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        session_meta=session_meta,
        options=options,
        cross_encoder=cross_encoder,
        conversation_history=conversation_history,
        config=config,
    )


@traceable(run_type="chain", name="rag_v4_pipeline")
def run_rag_query_sync(
    query: str,
    vectorstore: "VectorStore | None" = None,
    llm: "BaseLLM | None" = None,
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: "CrossEncoder | None" = None,
    conversation_history: list[tuple[str, str]] | None = None,
    config: dict | None = None,
) -> RagResult:
    parsed = parse_query(query)
    nodes = load_ontology_manifest()
    graph = build_graph_context(nodes)
    rdf_store = build_rdf_ontology_store(nodes, sparql_endpoint=os.getenv("S1000D_SPARQL_ENDPOINT"))
    rdf_resolution = rdf_store.resolve_query(parsed)
    resolution = resolve_ontology(parsed, nodes)
    max_chunks = (options.top_k if options else 6) or 6
    plan = plan_evidence(resolution, max_chunks=max_chunks)
    documents, evidences = retrieve_evidence(plan, vectorstore)
    answer_plan = build_answer_plan(parsed, documents, graph_context=graph, rdf_resolution=rdf_resolution)
    answer = verbalize_answer_plan(answer_plan, llm=llm)
    gate = check_answer_quality(answer)
    if not gate.ok:
        answer = "답변 품질 검사에서 생성 오류가 감지되어 원문 답변을 제공하지 않습니다. 관련 문서를 다시 확인해 주세요."
    return RagResult(answer=answer, evidences=evidences, reference_materials=collect_reference_materials(evidences))
