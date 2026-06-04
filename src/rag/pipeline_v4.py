"""Ontology-guided Graph RAG v4 pipeline.

This is the first v4 implementation slice.  It preserves the v3 ontology
resolution/retrieval baseline, then adds a structured AnswerPlan and LLM
verbalizer contract.  Later slices should replace manifest-only resolution with
a richer graph resolver and stronger quality gates.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.tracing import traceable

from src.types.rag import RagOptions, RagResult, SessionMeta, V4ResponseMetadata

from .evidence_trail import collect_reference_materials
from .ontology import (
    CandidateEvidence,
    OntologyNode,
    ParsedQuery,
    ResolutionResult,
    SupportLevel,
    check_answer_quality,
    load_ontology_manifest,
    parse_query,
    plan_evidence,
    resolve_ontology,
    retrieve_evidence,
)
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
    rdf_store = build_rdf_ontology_store(
        nodes,
        sparql_endpoint=os.getenv("S1000D_SPARQL_ENDPOINT"),
        backend=os.getenv("S1000D_RDF_BACKEND"),
    )
    rdf_resolution = rdf_store.resolve_query(parsed)
    resolution = _prefer_rdf_resolution(resolve_ontology(parsed, nodes), rdf_resolution, nodes, parsed)
    max_chunks = (options.top_k if options else 6) or 6
    plan = plan_evidence(resolution, max_chunks=max_chunks)
    documents, evidences = retrieve_evidence(plan, vectorstore)
    answer_plan = build_answer_plan(parsed, documents, graph_context=graph, rdf_resolution=rdf_resolution)
    uses_deterministic_fallback = answer_plan.support_level != SupportLevel.EXACT and parsed.intent.value == "procedure"
    synthesis_llm = None if uses_deterministic_fallback else llm
    answer = verbalize_answer_plan(answer_plan, llm=synthesis_llm)
    gate = check_answer_quality(answer)
    if not gate.ok:
        answer = "답변 품질 검사에서 생성 오류가 감지되어 원문 답변을 제공하지 않습니다. 관련 문서를 다시 확인해 주세요."
    return RagResult(
        answer=answer,
        evidences=evidences,
        reference_materials=collect_reference_materials(evidences),
        v4_metadata=V4ResponseMetadata(
            support_level=answer_plan.support_level.value,
            runtime_mode="deterministic_fallback" if uses_deterministic_fallback else "llm_synthesis",
            required_citations=list(answer_plan.required_citations),
            forbidden_claims=list(answer_plan.forbidden_claims),
            ontology_trace={
                "intent": parsed.intent.value,
                "target": parsed.target,
                "action": parsed.action,
                "rdf_primary_dmcs": list(rdf_resolution.primary_dmcs),
                "rdf_related_dmcs": list(rdf_resolution.related_dmcs),
                "graph_paths": list(answer_plan.graph_paths),
            },
        ),
    )


def _prefer_rdf_resolution(
    manifest_resolution: ResolutionResult,
    rdf_resolution: RdfResolution,
    nodes: list[OntologyNode],
    parsed: ParsedQuery,
) -> ResolutionResult:
    """Make RDF/SPARQL-selected DMCs the primary v4 evidence candidates.

    The legacy manifest resolver is still used as a safe fallback when the RDF
    layer selects nothing, but once RDF returns primary/related DMCs the evidence
    planner should search those DMCs in graph order instead of stale manifest
    candidates.
    """
    selected_dmcs = rdf_resolution.all_dmcs
    if not selected_dmcs:
        return manifest_resolution

    by_dmc = {node.dmc: node for node in nodes}
    candidates: list[CandidateEvidence] = []
    for index, dmc in enumerate(selected_dmcs):
        node = by_dmc.get(dmc) or _placeholder_node_for_rdf_dmc(dmc)
        support = SupportLevel.EXACT if dmc in rdf_resolution.primary_dmcs else SupportLevel.RELATED
        candidates.append(
            CandidateEvidence(
                node=node,
                support=support,
                reason="rdf_primary_dmc" if support == SupportLevel.EXACT else "rdf_related_dmc",
                score=max(0.1, 1.0 - index * 0.08),
            )
        )

    overall_support = SupportLevel.EXACT if rdf_resolution.primary_dmcs else SupportLevel.RELATED
    return ResolutionResult(
        parsed=getattr(manifest_resolution, "parsed", parsed),
        support=overall_support,
        candidates=tuple(candidates),
        reason="rdf_graph_primary" if rdf_resolution.primary_dmcs else "rdf_graph_related",
    )


def _placeholder_node_for_rdf_dmc(dmc: str) -> OntologyNode:
    return OntologyNode(dmc=dmc, title=dmc, dm_type="descriptive", metadata={"source": "rdf_resolution"})
