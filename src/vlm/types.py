"""Structured, dependency-light types for visual caption records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CAPTION_JSON_FIELDS = ("summary", "ocr_text", "components", "safety_notes", "keywords")


@dataclass(frozen=True)
class VisualCaptionRecord:
    """Normalized caption output for one visual asset manifest entry.

    The fields intentionally mirror a future VLM JSON response while retaining
    deterministic asset/model metadata needed for indexing and auditability.
    """

    asset_key: str
    asset_path: str | None
    status: str
    summary: str
    ocr_text: str = ""
    components: list[str] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    model_profile: str = "mock-vlm-captioner"
    backend: str = "mock"
    prompt_profile: str = "s1000d-technical-manual-v1"
    dmc: str | None = None
    structure_path: str | None = None
    ref_id: str | None = None
    title: str | None = None
    kind: str | None = None
    info_entity_ident: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VisualCaptionRecord":
        """Build a record from JSON/dict data, tolerating absent optional fields."""

        return cls(
            asset_key=str(value.get("asset_key") or value.get("key") or "unknown-asset"),
            asset_path=_optional_str(value.get("asset_path")),
            status=str(value.get("status") or "captioned"),
            summary=str(value.get("summary") or ""),
            ocr_text=str(value.get("ocr_text") or ""),
            components=_string_list(value.get("components")),
            safety_notes=_string_list(value.get("safety_notes")),
            keywords=_string_list(value.get("keywords")),
            model_profile=str(value.get("model_profile") or "unknown"),
            backend=str(value.get("backend") or "unknown"),
            prompt_profile=str(value.get("prompt_profile") or "s1000d-technical-manual-v1"),
            dmc=_optional_str(value.get("dmc")),
            structure_path=_optional_str(value.get("structure_path")),
            ref_id=_optional_str(value.get("ref_id")),
            title=_optional_str(value.get("title")),
            kind=_optional_str(value.get("kind")),
            info_entity_ident=_optional_str(value.get("info_entity_ident")),
            source_path=_optional_str(value.get("source_path")),
            metadata=dict(value.get("metadata") or {}),
            created_at=str(value.get("created_at") or datetime.now(timezone.utc).isoformat()),
        )


def manifest_asset_key(asset: Mapping[str, Any]) -> str:
    """Return the stable key used to name caption files for a manifest entry."""

    return str(asset.get("key") or asset.get("asset_key") or asset.get("ref_id") or asset.get("info_entity_ident") or "unknown-asset")


def safe_caption_filename(asset_key: str) -> str:
    """Return a filesystem-safe JSON filename for an asset key."""

    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in asset_key).strip("._")
    return f"{safe or 'unknown-asset'}.json"


def resolve_manifest_asset_path(asset: Mapping[str, Any], data_dir: str | Path | None = None) -> Path | None:
    """Resolve an asset path from a manifest entry without reading the file."""

    raw = asset.get("asset_path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute() or data_dir is None:
        return path
    return Path(data_dir) / path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]
