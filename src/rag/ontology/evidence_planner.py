"""Evidence planning and optional vectorstore retrieval for ontology RAG v2."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from src.types.rag import Evidence

from .schema import EvidencePlan, ResolutionResult


def plan_evidence(resolution: ResolutionResult, *, max_chunks: int = 6) -> EvidencePlan:
    dmcs = tuple(dict.fromkeys(c.node.dmc for c in resolution.candidates))
    parsed = resolution.parsed
    retrieval_query = " ".join(part for part in (parsed.target, parsed.action, parsed.original) if part)
    return EvidencePlan(resolution=resolution, dmcs=dmcs, retrieval_query=retrieval_query, max_chunks=max_chunks)


def retrieve_evidence(plan: EvidencePlan, vectorstore: Any | None = None) -> tuple[list[Document], list[Evidence]]:
    documents: list[Document] = []
    if vectorstore is not None and plan.dmcs:
        for dmc in plan.dmcs[: plan.max_chunks]:
            documents.extend(_similarity_by_dmc(vectorstore, plan.retrieval_query, dmc))
    if not documents:
        for candidate in plan.resolution.candidates[: plan.max_chunks]:
            node = candidate.node
            documents.append(Document(page_content=node.title, metadata={"dmc": node.dmc, "title": node.title, "dm_type": node.dm_type, "source": "ontology_manifest"}))
    evidences = [_evidence_from_doc(doc, i + 1) for i, doc in enumerate(documents[: plan.max_chunks])]
    return documents[: plan.max_chunks], evidences


def _similarity_by_dmc(vectorstore: Any, query: str, dmc: str) -> list[Document]:
    try:
        return [doc for doc, _score in vectorstore.similarity_search_with_score(query, k=2, filter={"dmc": dmc})]
    except TypeError:
        try:
            return vectorstore.similarity_search(query, k=2, filter={"dmc": dmc})
        except Exception:
            return []
    except Exception:
        return []


def _evidence_from_doc(doc: Document, rank: int) -> Evidence:
    meta = dict(doc.metadata or {})
    return Evidence(
        dmc=str(meta.get("dmc", "")),
        chunk_id=str(meta.get("chunk_id") or meta.get("id") or rank),
        score=float(meta.get("score", 1.0 / rank)),
        dm_type=meta.get("dm_type"),
        title=meta.get("title"),
        rank=rank,
        text=doc.page_content,
    )
