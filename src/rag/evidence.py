"""Dependency-light API/UI evidence serialization helpers.

The helpers in this module adapt fused records produced by
``src.rag.query_router.fuse_ranked_candidates`` into plain dictionaries that can
be returned by APIs or rendered by UI cards.  They intentionally avoid imports
from retrievers, vector stores, LangChain, embedding models, or VLM backends.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_IMAGE_RECORD_MODALITIES = {"image_caption", "image", "figure", "diagram", "visual", "photo", "picture"}
_TEXT_RECORD_MODALITIES = {"text", "chunk", "document"}


def evidence_from_fused_record(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Convert one fused retrieval record into an API/UI-friendly evidence dict.

    Text evidence keeps the familiar ``dmc``, chunk identifier, and score fields.
    Visual-caption evidence additionally exposes asset/caption metadata and card
    labels without opening files, loading models, or mutating indexes.
    """

    metadata = _metadata(record)
    content_role = _content_role(record, metadata)
    modality = _evidence_modality(record, metadata, content_role)
    is_visual = modality == "image" or content_role == "visual_caption"

    evidence: dict[str, Any] = {
        "dmc": _string_or_empty(_first_value(record, metadata, "dmc")),
        "score": _score(record),
        "final_score": _optional_float(_first_value(record, metadata, "final_score")),
        "rank": _optional_int(_first_value(record, metadata, "rank")),
        "modality": modality,
        "content_role": content_role,
        "display_label": "",
        "source_label": "",
    }

    for key in ("chunk_id", "chunk_index", "id"):
        value = _first_value(record, metadata, key)
        if value is not None:
            evidence[key] = str(value)

    for key in ("dm_type", "security", "applicability"):
        value = _first_value(record, metadata, key)
        if value is not None:
            evidence[key] = value

    if is_visual:
        for key in ("asset_key", "asset_path", "caption_path", "title", "kind", "ref_id"):
            value = _first_value(record, metadata, key)
            if value is not None:
                evidence[key] = str(value)

    page_content = _page_content(record)
    if page_content:
        evidence["text"] = page_content

    evidence["display_label"] = _display_label(evidence, is_visual=is_visual)
    evidence["source_label"] = _source_label(evidence, is_visual=is_visual)
    return _drop_none_values(evidence)


def evidences_from_fused_records(records: Sequence[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
    """Convert fused records to evidence dictionaries while preserving order."""

    return [evidence_from_fused_record(record) for record in records]


def _metadata(record: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        metadata = record.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}
    metadata = getattr(record, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _page_content(record: Mapping[str, Any] | Any) -> str:
    value = _candidate_value(record, "page_content")
    if value is None:
        value = _candidate_value(record, "content")
    if value is None:
        value = _candidate_value(record, "text")
    return "" if value is None else str(value)


def _candidate_value(record: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _first_value(record: Mapping[str, Any] | Any, metadata: Mapping[str, Any], key: str) -> Any:
    value = _candidate_value(record, key)
    if value is not None:
        return value
    return metadata.get(key)


def _content_role(record: Mapping[str, Any] | Any, metadata: Mapping[str, Any]) -> str:
    role = str(_first_value(record, metadata, "content_role") or "").strip().lower()
    if role:
        return role
    modality = str(_first_value(record, metadata, "modality") or "").strip().lower()
    if modality in _IMAGE_RECORD_MODALITIES:
        return "visual_caption"
    if modality in _TEXT_RECORD_MODALITIES:
        return "text"
    return ""


def _evidence_modality(record: Mapping[str, Any] | Any, metadata: Mapping[str, Any], content_role: str) -> str:
    modality = str(_first_value(record, metadata, "modality") or "").strip().lower()
    metadata_modality = str(metadata.get("modality") or "").strip().lower()
    if content_role == "visual_caption" or modality in _IMAGE_RECORD_MODALITIES or metadata_modality in _IMAGE_RECORD_MODALITIES:
        return "image"
    if modality in _TEXT_RECORD_MODALITIES or metadata_modality in _TEXT_RECORD_MODALITIES:
        return "text"
    return modality or metadata_modality or "unknown"


def _score(record: Mapping[str, Any] | Any) -> float:
    for key in ("score", "base_score", "final_score"):
        value = _candidate_value(record, key)
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    metadata = _metadata(record)
    for key in ("score", "base_score", "final_score"):
        parsed = _optional_float(metadata.get(key))
        if parsed is not None:
            return parsed
    return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def _display_label(evidence: Mapping[str, Any], *, is_visual: bool) -> str:
    if is_visual:
        title = evidence.get("title")
        if title:
            return str(title)
        kind = evidence.get("kind")
        ref_id = evidence.get("ref_id")
        if kind and ref_id:
            return f"{kind} {ref_id}"
        return str(evidence.get("asset_key") or evidence.get("asset_path") or evidence.get("caption_path") or "Visual caption")

    chunk = evidence.get("chunk_id") or evidence.get("chunk_index") or evidence.get("id")
    dmc = evidence.get("dmc")
    if dmc and chunk:
        return f"{dmc} · chunk {chunk}"
    return str(dmc or chunk or "Text evidence")


def _source_label(evidence: Mapping[str, Any], *, is_visual: bool) -> str:
    dmc = evidence.get("dmc") or "unknown DMC"
    if is_visual:
        asset = evidence.get("asset_key") or evidence.get("ref_id") or evidence.get("asset_path") or evidence.get("caption_path")
        return f"{dmc} · {asset}" if asset else str(dmc)

    chunk = evidence.get("chunk_id") or evidence.get("chunk_index") or evidence.get("id")
    return f"{dmc} · chunk {chunk}" if chunk else str(dmc)


def _drop_none_values(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items() if value is not None}
