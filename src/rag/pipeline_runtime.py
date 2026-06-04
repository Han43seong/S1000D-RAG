"""Runtime router for S1000D-RAG pipeline generations.

Default runtime remains the current ontology-first deterministic baseline (v3),
while v4 can be enabled explicitly for ontology-guided Graph RAG development.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.types.rag import RagOptions, RagResult, SessionMeta

from . import pipeline_v2 as pipeline_v3
from . import pipeline_v4

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM
    from langchain_core.vectorstores import VectorStore
    from sentence_transformers import CrossEncoder

_RUNTIME_ENV = "S1000D_RAG_PIPELINE"


def get_pipeline_version() -> str:
    requested = os.getenv(_RUNTIME_ENV, "v3").strip().lower()
    if requested in {"v4", "graph", "graph-rag"}:
        return "v4"
    return "v3"


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
    runner = pipeline_v4.run_rag_query_sync if get_pipeline_version() == "v4" else pipeline_v3.run_rag_query_sync
    return runner(
        query=query,
        vectorstore=vectorstore,
        llm=llm,
        session_meta=session_meta,
        options=options,
        cross_encoder=cross_encoder,
        conversation_history=conversation_history,
    )


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
