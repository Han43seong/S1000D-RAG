"""Export the current S1000D ontology manifest as RDF/Turtle or JSON-LD.

Usage:
    python scripts/export_ontology_rdf.py
    python scripts/export_ontology_rdf.py --output data/ontology/s1000d.ttl
    python scripts/export_ontology_rdf.py --format jsonld --output data/ontology/s1000d.jsonld
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ontology import load_ontology_manifest
from src.rag.v4.rdf_exporter import export_ontology_jsonld, export_ontology_turtle

DEFAULT_OUTPUT = Path("data/ontology/s1000d.ttl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export S1000D ontology manifest to RDF/Turtle or JSON-LD")
    parser.add_argument("--format", choices=("turtle", "jsonld"), default="turtle", help="Export format")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Target output path")
    args = parser.parse_args()

    nodes = load_ontology_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonld":
        payload = export_ontology_jsonld(nodes)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output} ({len(nodes)} ontology nodes, {len(payload['@graph'])} jsonld graph items)")
        return

    turtle = export_ontology_turtle(nodes)
    args.output.write_text(turtle, encoding="utf-8")
    print(f"wrote {args.output} ({len(nodes)} ontology nodes, {len(turtle.splitlines())} turtle lines)")


if __name__ == "__main__":
    main()
