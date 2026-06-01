#!/usr/bin/env python3
"""Preview offline multimodal RAG context from caption JSON files only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.multimodal_context import build_multimodal_context, load_caption_candidates  # noqa: E402


def run(args: argparse.Namespace) -> int:
    captions_dir = Path(args.captions_dir)
    caption_candidates = load_caption_candidates(captions_dir, limit=args.limit)
    route, fused, context = build_multimodal_context(
        query=args.query,
        caption_candidates=caption_candidates,
        limit=args.limit,
    )

    print("Route summary:")
    print(f"  query={route.query}")
    print(f"  visual_intent={route.visual_intent} text_intent={route.text_intent}")
    print(f"  visual_weight={route.visual_weight} text_weight={route.text_weight}")
    print(f"  matched_terms={', '.join(route.matched_terms) if route.matched_terms else '(none)'}")
    print(f"  reason={route.reason}")
    print(f"Loaded visual caption candidates: {len(caption_candidates)}")
    print(f"Fused context records: {len(fused)}")
    print("\nFormatted context:")
    print(context if context else "(empty)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview offline multimodal context from caption JSON files")
    parser.add_argument("--captions-dir", required=True, help="Directory containing caption JSON files; searched recursively")
    parser.add_argument("--query", required=True, help="Query to route and use for multimodal fusion")
    parser.add_argument("--limit", type=int, default=5, help="Maximum caption files/context records to include")
    return parser


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
