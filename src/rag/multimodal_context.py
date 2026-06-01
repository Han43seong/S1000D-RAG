"""Offline bridge from visual caption JSON files to multimodal RAG context.

This module intentionally depends only on pure project helpers. It does not read
image bytes, build indexes, import Chroma/LangChain, or load embedding/VLM models.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Any, Iterable

from src.vlm.documents import load_caption_documents

DEFAULT_VISUAL_SCORE = 1.0


def find_caption_json_files(captions_path: str | Path) -> list[Path]:
    """Return caption JSON files from a directory or glob-like path.

    Directory inputs are searched recursively for ``*.json``. Glob inputs are
    resolved with Python's stdlib ``glob`` and filtered to files only.
    """

    path = Path(captions_path)
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())

    pattern = str(captions_path)
    if _looks_like_glob(pattern):
        return sorted(Path(candidate) for candidate in glob(pattern, recursive=True) if Path(candidate).is_file())

    return [path] if path.is_file() and path.suffix.lower() == ".json" else []


def load_caption_candidates(captions_path: str | Path, *, limit: int | None = None, score: float = DEFAULT_VISUAL_SCORE) -> list[dict[str, Any]]:
    """Load caption JSON as document-like visual retrieval candidates.

    Returned dictionaries contain ``page_content``, ``metadata``, and ``score``.
    Metadata is tagged for the visual lane while preserving fields produced by
    ``src.vlm.documents.load_caption_documents``.
    """

    paths = find_caption_json_files(captions_path)
    if limit is not None:
        paths = paths[:limit]

    candidates: list[dict[str, Any]] = []
    for doc in load_caption_documents(paths):
        candidate = dict(doc)
        metadata = dict(candidate.get("metadata") or {})
        metadata.setdefault("modality", "image")
        metadata.setdefault("content_role", "visual_caption")
        metadata["source_lane"] = "visual"
        candidate["metadata"] = metadata
        candidate.setdefault("page_content", "")
        candidate["score"] = score
        candidates.append(candidate)
    return candidates


def build_multimodal_context(
    *,
    query: str,
    text_candidates: Iterable[dict[str, Any]] = (),
    caption_candidates: Iterable[dict[str, Any]] = (),
    limit: int | None = None,
) -> tuple[Any, list[dict[str, Any]], str]:
    """Route, fuse, and format text plus visual-caption candidates offline."""

    from src.rag.query_router import format_fused_context, fuse_ranked_candidates, route_query

    route = route_query(query)
    fused = fuse_ranked_candidates([*text_candidates, *caption_candidates], route, top_k=limit)
    return route, fused, format_fused_context(fused)


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[]")
