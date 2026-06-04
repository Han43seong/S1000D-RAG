"""RAG package exports with lazy optional-dependency loading."""

from __future__ import annotations

_EXPORT_MODULES = {
    "run_rag_query": "src.rag.pipeline_v2",
    "run_rag_query_sync": "src.rag.pipeline_v2",
    "build_prompt": "src.rag.prompt",
    "enhance_query": "src.rag.query_enhancer",
    "expand_query": "src.rag.query_enhancer",
    "extract_sns_code": "src.rag.query_enhancer",
    "MetaFilter": "src.rag.retriever",
    "retrieve": "src.rag.retriever",
    "retrieve_two_stage": "src.rag.retriever",
    "rerank": "src.rag.reranker",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module 'src.rag' has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
