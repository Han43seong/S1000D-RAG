"""Pure conversion of visual caption records into document-like dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.vlm.types import VisualCaptionRecord


def caption_to_document(caption: VisualCaptionRecord | Mapping[str, Any], *, caption_path: str | Path | None = None) -> dict[str, Any]:
    """Convert one caption record into a dependency-free document dict."""

    record = caption if isinstance(caption, VisualCaptionRecord) else VisualCaptionRecord.from_mapping(caption)
    content_parts = [record.summary]
    if record.ocr_text:
        content_parts.append(f"OCR text: {record.ocr_text}")
    if record.components:
        content_parts.append("Components: " + ", ".join(record.components))
    if record.safety_notes:
        content_parts.append("Safety notes: " + "; ".join(record.safety_notes))
    if record.keywords:
        content_parts.append("Keywords: " + ", ".join(record.keywords))

    metadata: dict[str, Any] = {
        "modality": "image",
        "asset_key": record.asset_key,
        "asset_path": record.asset_path or "",
        "dmc": record.dmc or "",
        "structure_path": record.structure_path or "",
        "caption_path": str(caption_path or ""),
        "content_role": "visual_caption",
        "caption_status": record.status,
        "model_profile": record.model_profile,
        "backend": record.backend,
        "prompt_profile": record.prompt_profile,
        "ref_id": record.ref_id or "",
        "kind": record.kind or "",
        "title": record.title or "",
    }
    return {
        "page_content": "\n".join(part for part in content_parts if part),
        "metadata": metadata,
    }


def captions_to_documents(captions: Iterable[VisualCaptionRecord | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert caption records to document dicts.

    If a mapping contains ``caption_path`` it is propagated into metadata.
    """

    docs: list[dict[str, Any]] = []
    for caption in captions:
        caption_path = caption.get("caption_path") if isinstance(caption, Mapping) else None
        docs.append(caption_to_document(caption, caption_path=caption_path))
    return docs


def load_caption_documents(caption_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load JSON caption files and convert them into document dicts."""

    docs: list[dict[str, Any]] = []
    for path_like in caption_paths:
        path = Path(path_like)
        data = json.loads(path.read_text(encoding="utf-8"))
        docs.append(caption_to_document(data, caption_path=path))
    return docs
