#!/usr/bin/env python3
"""Generate dependency-light visual caption JSON files from an assets manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHROMA_PERSIST_DIR  # noqa: E402
from src.vlm.captioner import CaptionerUnavailableError, create_captioner  # noqa: E402
from src.vlm.types import safe_caption_filename  # noqa: E402


def _default_manifest(chroma_dir: str | Path) -> Path:
    return Path(chroma_dir) / "assets_manifest.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible(asset: dict[str, Any]) -> bool:
    return str(asset.get("status") or "").casefold() in {"found", "metadata_only"} or bool(asset.get("metadata_only"))


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest) if args.manifest else _default_manifest(args.chroma_dir)
    manifest = _load_manifest(manifest_path)
    data_dir = manifest.get("data_dir")
    assets = list(manifest.get("assets") or manifest.get("refs") or [])
    if args.limit is not None:
        assets = assets[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captioner = create_captioner(mock=args.mock)

    counts = {"seen": 0, "written": 0, "skipped_missing": 0, "skipped_existing": 0, "errors": 0}
    for raw_asset in assets:
        counts["seen"] += 1
        asset = dict(raw_asset)
        status = str(asset.get("status") or "").casefold()
        if not _eligible(asset):
            if status == "missing":
                counts["skipped_missing"] += 1
            else:
                counts["skipped_missing"] += 1
            continue

        key = str(asset.get("key") or asset.get("asset_key") or asset.get("ref_id") or asset.get("info_entity_ident") or f"asset-{counts['seen']}")
        target = output_dir / safe_caption_filename(key)
        if target.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            continue

        try:
            caption = captioner.caption_asset(asset, data_dir=data_dir)
        except CaptionerUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # keep CLI robust for malformed individual records
            counts["errors"] += 1
            print(f"warning: failed to caption {key}: {exc}", file=sys.stderr)
            continue

        target.write_text(json.dumps(caption.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        counts["written"] += 1

    print(
        "Caption assets summary: "
        f"manifest={manifest_path} output_dir={output_dir} mock={args.mock} "
        f"seen={counts['seen']} written={counts['written']} "
        f"skipped_missing={counts['skipped_missing']} skipped_existing={counts['skipped_existing']} "
        f"errors={counts['errors']}"
    )
    return 0 if counts["errors"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption S1000D visual assets from an assets manifest")
    parser.add_argument("--manifest", default=None, help="Assets manifest path (default: <chroma-dir>/assets_manifest.json)")
    parser.add_argument("--chroma-dir", default=CHROMA_PERSIST_DIR, help="Chroma directory used to derive the default manifest path")
    parser.add_argument("--output-dir", default="artifacts/visual_captions", help="Directory for per-asset caption JSON files")
    parser.add_argument("--limit", type=int, default=None, help="Maximum manifest entries to inspect")
    parser.add_argument("--mock", action="store_true", help="Generate deterministic offline placeholder captions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption JSON files")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
