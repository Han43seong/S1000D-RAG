"""Dependency-light visual asset manifest extraction for local S1000D CSDB data.

This module intentionally does not parse image bytes, load models, or touch Chroma.
It only scans DMC XML files for visual references and resolves likely local asset
paths from S1000D ``infoEntityIdent`` values.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config import PROJECT_ROOT
from src.csdb.adapter import DmFilter
from src.csdb.local_adapter import LocalCsdbAdapter
from src.parser.visual_refs import extract_visual_refs_from_xml
from src.types.visual import VisualArtifactKind, VisualArtifactRef

COMMON_ASSET_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".gif",
    ".svg",
    ".cgm",
    ".pdf",
)


@dataclass(frozen=True)
class VisualAssetRecord:
    """Manifest-ready resolution record for one visual reference."""

    ref: VisualArtifactRef
    status: str
    asset_path: Path | None = None
    metadata_only: bool = False

    def to_manifest_dict(self, data_dir: Path) -> dict[str, Any]:
        source_path = self.ref.source_path
        return {
            "key": self.ref.stable_key,
            "kind": self.ref.kind.value,
            "dmc": self.ref.dmc,
            "ref_id": self.ref.ref_id,
            "title": self.ref.title,
            "info_entity_ident": self.ref.info_entity_ident,
            "structure_path": self.ref.structure_path,
            "source_path": _relative_or_str(source_path, data_dir) if source_path else None,
            "asset_path": _relative_or_str(self.asset_path, data_dir) if self.asset_path else None,
            "status": self.status,
            "metadata_only": self.metadata_only,
            "metadata": self.ref.metadata,
        }


@dataclass(frozen=True)
class VisualAssetExtractionResult:
    """Collected visual refs and resolved asset records for a CSDB directory."""

    data_dir: Path
    dmcs: list[str]
    refs: list[VisualArtifactRef]
    assets: list[VisualAssetRecord]
    parse_errors: list[str]

    @property
    def dm_count(self) -> int:
        return len(self.dmcs)

    @property
    def visual_ref_count(self) -> int:
        return len(self.refs)

    @property
    def found_asset_count(self) -> int:
        return sum(1 for asset in self.assets if asset.status == "found")

    @property
    def missing_asset_count(self) -> int:
        return sum(1 for asset in self.assets if asset.status == "missing")

    @property
    def table_ref_count(self) -> int:
        return sum(1 for ref in self.refs if ref.kind == VisualArtifactKind.TABLE)


def _relative_or_str(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _current_git_commit(repo_root: Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _build_asset_index(data_dir: Path, extensions: Iterable[str] = COMMON_ASSET_EXTENSIONS) -> dict[str, Path]:
    """Index candidate asset files by case-folded stem and filename."""

    wanted_suffixes = {ext.casefold() for ext in extensions}
    index: dict[str, Path] = {}
    for path in sorted(data_dir.rglob("*"), key=lambda p: str(p).casefold()):
        if not path.is_file():
            continue
        if path.suffix.casefold() not in wanted_suffixes:
            continue
        # Prefer the first deterministic path found for duplicate case variants.
        index.setdefault(path.stem.casefold(), path)
        index.setdefault(path.name.casefold(), path)
    return index


def resolve_asset_path(ref: VisualArtifactRef, data_dir: str | Path, asset_index: dict[str, Path] | None = None) -> Path | None:
    """Resolve a likely local asset path for a visual ref without reading image bytes."""

    if not ref.info_entity_ident:
        return None

    root = Path(data_dir)
    index = asset_index if asset_index is not None else _build_asset_index(root)
    ident = ref.info_entity_ident.strip()
    if not ident:
        return None

    candidates = [ident.casefold()]
    for ext in COMMON_ASSET_EXTENSIONS:
        candidates.append(f"{ident}{ext}".casefold())

    for key in candidates:
        match = index.get(key)
        if match is not None:
            return match
    return None


def _record_for_ref(ref: VisualArtifactRef, data_dir: Path, asset_index: dict[str, Path]) -> VisualAssetRecord:
    if ref.kind == VisualArtifactKind.TABLE:
        return VisualAssetRecord(ref=ref, status="metadata_only", metadata_only=True)

    asset_path = resolve_asset_path(ref, data_dir, asset_index)
    if asset_path is not None:
        return VisualAssetRecord(ref=ref, status="found", asset_path=asset_path)
    return VisualAssetRecord(ref=ref, status="missing")


def extract_visual_assets(
    data_dir: str | Path,
    *,
    model_ident_code: str | None = None,
    limit: int | None = None,
) -> VisualAssetExtractionResult:
    """Scan local DMC XML files and resolve visual refs to manifest records."""

    root = Path(data_dir)
    adapter = LocalCsdbAdapter(root)
    dm_filter = DmFilter(model_ident_code=model_ident_code) if model_ident_code else None
    dmcs = asyncio.run(adapter.list_data_modules(dm_filter))
    if limit is not None:
        dmcs = dmcs[:limit]

    asset_index = _build_asset_index(root)
    refs: list[VisualArtifactRef] = []
    assets: list[VisualAssetRecord] = []
    parse_errors: list[str] = []

    for dmc in dmcs:
        try:
            xml = asyncio.run(adapter.get_data_module_xml(dmc))
            source_path = adapter.get_data_module_path(dmc)
            dm_refs = extract_visual_refs_from_xml(xml, dmc=dmc, source_path=source_path)
        except Exception as exc:  # keep extraction best-effort across a CSDB set
            parse_errors.append(f"{dmc}: {exc}")
            continue
        refs.extend(dm_refs)
        assets.extend(_record_for_ref(ref, root, asset_index) for ref in dm_refs)

    return VisualAssetExtractionResult(
        data_dir=root,
        dmcs=dmcs,
        refs=refs,
        assets=assets,
        parse_errors=parse_errors,
    )


def build_visual_asset_manifest(
    result: VisualAssetExtractionResult,
    *,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable visual asset manifest."""

    data_dir = result.data_dir
    refs = [asset.to_manifest_dict(data_dir) for asset in result.assets]
    missing = [ref for ref in refs if ref["status"] == "missing"]
    return {
        "data_dir": str(data_dir),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "dm_count": result.dm_count,
        "visual_ref_count": result.visual_ref_count,
        "found_asset_count": result.found_asset_count,
        "missing_asset_count": result.missing_asset_count,
        "table_ref_count": result.table_ref_count,
        "parse_error_count": len(result.parse_errors),
        "git_commit": git_commit if git_commit is not None else _current_git_commit(),
        "sample_dmcs": result.dmcs[:10],
        "sample_missing_refs": missing[:10],
        "sample_errors": result.parse_errors[:10],
        "refs": refs,
        # Alias retained for downstream VLM captioning code that expects assets.
        "assets": refs,
    }


def write_visual_asset_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write a visual asset manifest JSON file to a caller-selected path."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def extract_and_write_visual_asset_manifest(
    data_dir: str | Path,
    output_path: str | Path,
    *,
    model_ident_code: str | None = None,
    limit: int | None = None,
) -> tuple[VisualAssetExtractionResult, Path, dict[str, Any]]:
    """Convenience wrapper used by the ingest CLI."""

    result = extract_visual_assets(data_dir, model_ident_code=model_ident_code, limit=limit)
    manifest = build_visual_asset_manifest(result)
    path = write_visual_asset_manifest(manifest, output_path)
    return result, path, manifest
