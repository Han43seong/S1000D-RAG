"""Media extraction helpers for S1000D visual asset manifests."""

from .asset_extractor import (
    VisualAssetExtractionResult,
    VisualAssetRecord,
    build_visual_asset_manifest,
    extract_and_write_visual_asset_manifest,
    extract_visual_assets,
    resolve_asset_path,
    write_visual_asset_manifest,
)

__all__ = [
    "VisualAssetExtractionResult",
    "VisualAssetRecord",
    "build_visual_asset_manifest",
    "extract_and_write_visual_asset_manifest",
    "extract_visual_assets",
    "resolve_asset_path",
    "write_visual_asset_manifest",
]
