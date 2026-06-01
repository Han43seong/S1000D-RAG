"""Captioner abstractions and offline mock implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from src.vlm.prompts import PROMPT_PROFILE, build_technical_manual_caption_prompt
from src.vlm.types import VisualCaptionRecord, manifest_asset_key, resolve_manifest_asset_path


class CaptionerUnavailableError(RuntimeError):
    """Raised when a real VLM backend is requested but unavailable."""


@runtime_checkable
class VisualCaptioner(Protocol):
    """Dependency-light captioner protocol."""

    model_profile: str
    backend: str

    def caption_asset(self, asset: Mapping[str, Any], *, data_dir: str | Path | None = None) -> VisualCaptionRecord:
        """Caption a manifest entry without mutating the manifest."""


class MockVisualCaptioner:
    """Deterministic offline captioner using manifest metadata only."""

    model_profile = "mock-vlm-captioner"
    backend = "mock"

    def caption_asset(self, asset: Mapping[str, Any], *, data_dir: str | Path | None = None) -> VisualCaptionRecord:
        key = manifest_asset_key(asset)
        title = _clean(asset.get("title"))
        kind = _clean(asset.get("kind")) or "visual asset"
        dmc = _clean(asset.get("dmc"))
        ref_id = _clean(asset.get("ref_id"))
        info_entity = _clean(asset.get("info_entity_ident"))
        status = _clean(asset.get("status")) or "unknown"
        metadata_only = bool(asset.get("metadata_only")) or status == "metadata_only"
        resolved_path = resolve_manifest_asset_path(asset, data_dir)

        noun = title or info_entity or ref_id or key
        summary = f"Mock caption for {kind} '{noun}'"
        if dmc:
            summary += f" in data module {dmc}"
        summary += "."
        if metadata_only:
            summary += " Caption is based on manifest metadata only."

        components = [item for item in [title, kind, ref_id, info_entity] if item]
        keywords = _dedupe([kind, title, dmc, ref_id, info_entity, status])
        safety_notes = []
        lower_blob = " ".join(str(asset.get(name, "")) for name in ("title", "structure_path", "ref_id", "info_entity_ident")).lower()
        if any(token in lower_blob for token in ("warning", "caution", "danger", "hazard", "safety")):
            safety_notes.append("Manifest metadata suggests possible safety-related visual content.")

        return VisualCaptionRecord(
            asset_key=key,
            asset_path=str(resolved_path) if resolved_path is not None else _optional_str(asset.get("asset_path")),
            status="mock_captioned_metadata_only" if metadata_only else "mock_captioned",
            summary=summary,
            ocr_text="",
            components=_dedupe(components),
            safety_notes=safety_notes,
            keywords=keywords,
            model_profile=self.model_profile,
            backend=self.backend,
            prompt_profile=PROMPT_PROFILE,
            dmc=_optional_str(asset.get("dmc")),
            structure_path=_optional_str(asset.get("structure_path")),
            ref_id=_optional_str(asset.get("ref_id")),
            title=_optional_str(asset.get("title")),
            kind=_optional_str(asset.get("kind")),
            info_entity_ident=_optional_str(asset.get("info_entity_ident")),
            source_path=_optional_str(asset.get("source_path")),
            metadata={
                "manifest_status": status,
                "metadata_only": metadata_only,
                "prompt": build_technical_manual_caption_prompt(asset),
            },
        )


class DisabledVisualCaptioner:
    """Placeholder for future real VLM backends that fails with guidance."""

    def __init__(self, *, backend: str = "disabled", model_profile: str = "unconfigured") -> None:
        self.backend = backend
        self.model_profile = model_profile

    def caption_asset(self, asset: Mapping[str, Any], *, data_dir: str | Path | None = None) -> VisualCaptionRecord:
        raise CaptionerUnavailableError(
            "Real VLM captioning is not available in this scaffold. Re-run with --mock for offline "
            "placeholder captions, or configure a future VLM backend/model path before using non-mock mode. "
            "This path intentionally avoids importing or downloading VLM runtimes."
        )


def create_captioner(*, mock: bool, backend: str | None = None, model_profile: str | None = None) -> VisualCaptioner:
    """Return a captioner implementation without heavy imports."""

    if mock:
        return MockVisualCaptioner()
    return DisabledVisualCaptioner(backend=backend or "disabled", model_profile=model_profile or "unconfigured")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_str(value: Any) -> str | None:
    text = _clean(value)
    return text or None


def _dedupe(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result
