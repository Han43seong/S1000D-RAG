#!/usr/bin/env python3
"""Build a lightweight graph manifest from the configured Chroma collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, S1000D_GRAPH_MANIFEST_PATH
from src.rag.graph_retrieval import build_graph_from_chunk_metadata, save_graph_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S1000D graph manifest from Chroma metadata")
    parser.add_argument("--chroma-dir", default=CHROMA_PERSIST_DIR)
    parser.add_argument("--collection", default=CHROMA_COLLECTION_NAME)
    parser.add_argument("--output", default=S1000D_GRAPH_MANIFEST_PATH)
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)
    data = collection.get(include=["metadatas"])
    metadata_rows = [dict(meta) for meta in (data.get("metadatas") or [])]
    manifest = build_graph_from_chunk_metadata(metadata_rows)
    save_graph_manifest(manifest, args.output)
    summary = {
        "chroma_dir": args.chroma_dir,
        "collection": args.collection,
        "output": str(Path(args.output).resolve()),
        "procedure_count": len(manifest.procedures),
        "unique_dmc_count": len({proc.dmc for proc in manifest.procedures}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
