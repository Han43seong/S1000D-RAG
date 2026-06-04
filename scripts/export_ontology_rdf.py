"""Export the current S1000D ontology manifest as RDF/Turtle.

Usage:
    python scripts/export_ontology_rdf.py
    python scripts/export_ontology_rdf.py --output data/ontology/s1000d.ttl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ontology import load_ontology_manifest
from src.rag.v4.rdf_exporter import export_ontology_turtle

DEFAULT_OUTPUT = Path("data/ontology/s1000d.ttl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export S1000D ontology manifest to RDF/Turtle")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Target .ttl path")
    args = parser.parse_args()

    nodes = load_ontology_manifest()
    turtle = export_ontology_turtle(nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(turtle, encoding="utf-8")
    print(f"wrote {args.output} ({len(nodes)} ontology nodes, {len(turtle.splitlines())} turtle lines)")


if __name__ == "__main__":
    main()
