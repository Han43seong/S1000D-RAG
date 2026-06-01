"""Prompt construction for deterministic S1000D technical-manual captioning."""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.vlm.types import CAPTION_JSON_FIELDS, manifest_asset_key


PROMPT_PROFILE = "s1000d-technical-manual-v1"


def build_technical_manual_caption_prompt(asset: Mapping[str, Any] | Any) -> str:
    """Build a deterministic VLM prompt for one visual manifest entry.

    The prompt asks future VLM backends for a strict JSON object and includes
    only manifest metadata, so tests and mock mode do not inspect image bytes.
    """

    asset_dict = _asset_to_mapping(asset)
    context = {
        "asset_key": manifest_asset_key(asset_dict),
        "kind": asset_dict.get("kind"),
        "title": asset_dict.get("title"),
        "dmc": asset_dict.get("dmc"),
        "ref_id": asset_dict.get("ref_id"),
        "info_entity_ident": asset_dict.get("info_entity_ident"),
        "structure_path": asset_dict.get("structure_path"),
        "status": asset_dict.get("status"),
        "metadata_only": asset_dict.get("metadata_only"),
    }
    field_list = ", ".join(f'"{field}"' for field in CAPTION_JSON_FIELDS)
    return (
        "You are captioning a visual asset from an S1000D technical manual.\n"
        "Use concise maintenance-manual language. Identify equipment, labels, "
        "procedural cues, warnings/cautions, and searchable keywords.\n"
        "Return only valid JSON with exactly these fields: "
        f"{field_list}.\n"
        "Field requirements:\n"
        "- summary: one concise technical sentence.\n"
        "- ocr_text: visible text transcribed from the image, or empty string.\n"
        "- components: array of visible parts, controls, labels, or diagram elements.\n"
        "- safety_notes: array of warnings, cautions, hazards, or empty array.\n"
        "- keywords: array of short retrieval terms.\n"
        "Do not invent unreadable label text; use null-free JSON values.\n"
        "Manifest metadata:\n"
        f"{json.dumps(context, ensure_ascii=False, sort_keys=True)}"
    )


def _asset_to_mapping(asset: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(asset, Mapping):
        return dict(asset)
    if hasattr(asset, "to_manifest_dict"):
        try:
            return dict(asset.to_manifest_dict(asset.asset_path.parent if asset.asset_path else "."))
        except Exception:
            pass
    if hasattr(asset, "__dict__"):
        return dict(asset.__dict__)
    raise TypeError("asset must be a manifest mapping or manifest-like object")
