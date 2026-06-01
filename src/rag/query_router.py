"""Dependency-light multimodal query routing and fusion helpers.

This module intentionally stays pure-Python so query intent and fusion behavior
can be tested without Chroma, LangChain, embedding models, or VLM backends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

KOREAN_VISUAL_TERMS: tuple[str, ...] = (
    "그림",
    "도면",
    "이미지",
    "표",
    "사진",
    "위치",
    "모양",
    "라벨",
    "캡션",
    "회로",
    "배선",
)

ENGLISH_VISUAL_TERMS: tuple[str, ...] = (
    "figure",
    "diagram",
    "image",
    "table",
    "shown",
    "label",
    "photo",
    "picture",
    "drawing",
    "schematic",
    "circuit",
    "wiring",
)

KOREAN_TEXT_TERMS: tuple[str, ...] = (
    "절차",
    "방법",
    "교체",
    "정비",
    "탈거",
    "장착",
    "점검",
    "검사",
    "수리",
    "설치",
    "분해",
    "조립",
)

ENGLISH_TEXT_TERMS: tuple[str, ...] = (
    "remove",
    "removal",
    "install",
    "installation",
    "replace",
    "replacement",
    "procedure",
    "inspection",
    "inspect",
    "maintenance",
    "service",
    "repair",
    "check",
    "test",
)

_TEXT_MODALITIES = {"text", "chunk", "document"}
_IMAGE_MODALITIES = {"image", "figure", "diagram", "visual", "photo", "picture"}


@dataclass(frozen=True)
class QueryRoute:
    """Routing decision for a user query.

    Text retrieval is always enabled; visual intent only increases the weight of
    image-caption candidates rather than replacing the text lane.
    """

    query: str
    visual_intent: bool
    text_intent: bool
    visual_weight: float
    text_weight: float
    matched_terms: tuple[str, ...]
    reason: str


def route_query(query: str) -> QueryRoute:
    """Detect text/visual intent and return lane weights for fusion."""

    visual_matches = _matched_terms(query, KOREAN_VISUAL_TERMS, ENGLISH_VISUAL_TERMS)
    text_matches = _matched_terms(query, KOREAN_TEXT_TERMS, ENGLISH_TEXT_TERMS)

    visual_intent = bool(visual_matches)
    # Every query remains text-capable. Explicit procedural matches make it
    # text-first, but even visual-only questions retain text retrieval.
    text_intent = True

    if visual_intent and text_matches:
        visual_weight = 1.15
        text_weight = 1.25
        reason = "mixed visual and procedural/text terms; keep procedural text first with visual-caption support"
    elif visual_intent:
        visual_weight = 1.35
        text_weight = 1.0
        reason = "visual terms detected; boost visual-caption lane while retaining text retrieval"
    elif text_matches:
        visual_weight = 0.85
        text_weight = 1.25
        reason = "procedural/text terms detected; keep text lane preferred"
    else:
        visual_weight = 1.0
        text_weight = 1.0
        reason = "no explicit modality terms detected; balanced text-capable route"

    return QueryRoute(
        query=query,
        visual_intent=visual_intent,
        text_intent=text_intent,
        visual_weight=visual_weight,
        text_weight=text_weight,
        matched_terms=tuple(dict.fromkeys((*visual_matches, *text_matches))),
        reason=reason,
    )


def candidate_modality(candidate: Mapping[str, Any] | Any) -> str:
    """Return normalized candidate modality: ``image_caption``, ``text``, or ``unknown``."""

    metadata = _metadata(candidate)
    modality = str(metadata.get("modality", "")).strip().lower()
    content_role = str(metadata.get("content_role", "")).strip().lower()

    if content_role == "visual_caption" or modality in _IMAGE_MODALITIES:
        return "image_caption"
    if modality in _TEXT_MODALITIES or content_role in {"text", "chunk", "procedure", "description"}:
        return "text"
    return "unknown"


def score_candidate_for_route(candidate: Mapping[str, Any] | Any, route: QueryRoute) -> dict[str, Any]:
    """Score one retrieval candidate for a route with explanatory fields."""

    metadata = dict(_metadata(candidate))
    modality = candidate_modality(candidate)
    base_score = _base_score(candidate)

    if modality == "image_caption":
        modality_boost = route.visual_weight
    elif modality == "text":
        modality_boost = route.text_weight
    else:
        modality_boost = 1.0

    final_score = base_score * modality_boost
    return {
        "page_content": _page_content(candidate),
        "metadata": metadata,
        "modality": modality,
        "base_score": base_score,
        "modality_boost": modality_boost,
        "final_score": final_score,
        "route_reason": route.reason,
        "matched_terms": list(route.matched_terms),
        "dedupe_key": candidate_dedupe_key(candidate),
    }


def fuse_ranked_candidates(
    candidates: Sequence[Mapping[str, Any] | Any],
    route: QueryRoute,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Score, deduplicate, and rank candidates for prompt context."""

    best_by_key: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        scored = score_candidate_for_route(candidate, route)
        scored["source_index"] = index
        key = scored["dedupe_key"]
        current = best_by_key.get(key)
        if current is None or scored["final_score"] > current["final_score"]:
            best_by_key[key] = scored

    ranked = sorted(best_by_key.values(), key=lambda item: (-item["final_score"], item["source_index"]))
    if top_k is not None:
        ranked = ranked[:top_k]

    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def format_fused_context(records: Sequence[Mapping[str, Any]]) -> str:
    """Format fused records for prompts using metadata-only headers.

    The function never opens paths from metadata; path strings are only echoed in
    headers to help downstream prompts cite visual-caption evidence.
    """

    blocks: list[str] = []
    for record in records:
        metadata = _metadata(record)
        content = _page_content(record).strip()
        modality = str(record.get("modality") or candidate_modality(record)) if isinstance(record, Mapping) else candidate_modality(record)
        if modality == "image_caption":
            path = _meta(metadata, "asset_path") or _meta(metadata, "caption_path")
            header = (
                f"[IMAGE_CAPTION DMC={_meta(metadata, 'dmc')} "
                f"ASSET={_meta(metadata, 'asset_key')} PATH={path}]"
            )
        else:
            chunk = _meta(metadata, "chunk_index") or _meta(metadata, "chunk_id") or _meta(metadata, "id")
            header = f"[TEXT DMC={_meta(metadata, 'dmc')} CHUNK={chunk}]"
        blocks.append(f"{header}\n{content}" if content else header)
    return "\n\n".join(blocks)


def candidate_dedupe_key(candidate: Mapping[str, Any] | Any) -> str:
    """Build stable dedupe key from image asset/caption or text chunk metadata."""

    metadata = _metadata(candidate)
    modality = candidate_modality(candidate)
    if modality == "image_caption":
        asset_key = _meta(metadata, "asset_key")
        caption_path = _meta(metadata, "caption_path")
        asset_path = _meta(metadata, "asset_path")
        value = asset_key or caption_path or asset_path or _page_content(candidate)
        return f"image:{value}"

    dmc = _meta(metadata, "dmc")
    chunk = _meta(metadata, "chunk_index") or _meta(metadata, "chunk_id") or _meta(metadata, "id")
    if dmc or chunk:
        return f"text:{dmc}:{chunk}"
    return f"unknown:{_page_content(candidate)}"


def _matched_terms(query: str, korean_terms: Sequence[str], english_terms: Sequence[str]) -> list[str]:
    lowered = query.lower()
    matches = [term for term in korean_terms if term in query]
    for term in english_terms:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])", lowered):
            matches.append(term)
    return matches


def _metadata(candidate: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        metadata = candidate.get("metadata", {})
        return metadata if isinstance(metadata, Mapping) else {}
    metadata = getattr(candidate, "metadata", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _page_content(candidate: Mapping[str, Any] | Any) -> str:
    if isinstance(candidate, Mapping):
        return str(candidate.get("page_content") or candidate.get("content") or candidate.get("text") or "")
    return str(getattr(candidate, "page_content", getattr(candidate, "content", "")) or "")


def _base_score(candidate: Mapping[str, Any] | Any) -> float:
    value = _candidate_value(candidate, "score")
    if value is not None:
        return _float_or_default(value, 0.0)

    distance = _candidate_value(candidate, "distance")
    if distance is not None:
        distance_value = max(_float_or_default(distance, 0.0), 0.0)
        return 1.0 / (1.0 + distance_value)

    return 1.0


def _candidate_value(candidate: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(candidate, Mapping):
        if key in candidate:
            return candidate[key]
        metadata = candidate.get("metadata", {})
        if isinstance(metadata, Mapping):
            return metadata.get(key)
        return None
    return getattr(candidate, key, None)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key, "")
    return "" if value is None else str(value)
