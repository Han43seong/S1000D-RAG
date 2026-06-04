#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.rag.ontology import build_ontology_manifest, save_ontology_manifest
from src.rag.ontology.manifest_builder import DEFAULT_ONTOLOGY_MANIFEST_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ontology-first RAG v2 manifest")
    parser.add_argument("--output", default=str(DEFAULT_ONTOLOGY_MANIFEST_PATH))
    args = parser.parse_args()
    nodes = build_ontology_manifest()
    path = save_ontology_manifest(nodes, args.output)
    print(f"wrote {len(nodes)} ontology nodes to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
