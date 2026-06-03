#!/usr/bin/env python3
"""Build ontology-style S1000D node/edge exports.

Outputs:
- s1000d_ontology_graph.json: property-graph node/edge snapshot
- s1000d_ontology.jsonld: JSON-LD export
- s1000d_ontology.ttl: Turtle-like RDF triples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.graph_retrieval import load_graph_manifest
from src.rag.ontology import (
    OntologyGraph,
    build_ontology_from_manifest,
    build_ontology_from_xml_dir,
    write_ontology_exports,
)

DEFAULT_XML_DIR = PROJECT_ROOT / "docs" / "S1000D Issue 6" / "Bike Data Set for Release number 6 R2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ontology"


def merge_graphs(primary: OntologyGraph, secondary: OntologyGraph) -> OntologyGraph:
    merged = OntologyGraph()
    for graph in (primary, secondary):
        for node in graph.nodes.values():
            merged.add_node(node.id, node.type, node.label, **node.properties)
        for edge in graph.edges:
            merged.add_edge(edge.source, edge.predicate, edge.target, **edge.properties)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S1000D ontology exports")
    parser.add_argument("--xml-dir", default=str(DEFAULT_XML_DIR))
    parser.add_argument("--graph-manifest", default=None, help="Optional graph manifest JSON path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    manifest = load_graph_manifest(args.graph_manifest) if args.graph_manifest else load_graph_manifest()
    manifest_graph = build_ontology_from_manifest(manifest) if manifest else OntologyGraph()
    xml_graph = build_ontology_from_xml_dir(args.xml_dir)
    graph = merge_graphs(manifest_graph, xml_graph)
    outputs = write_ontology_exports(graph, args.output_dir)
    summary = {
        "xml_dir": str(Path(args.xml_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "node_count": graph.node_count(),
        "edge_count": len(graph.edges),
        "node_type_counts": graph.node_type_counts(),
        "edge_predicate_counts": graph.edge_predicate_counts(),
        "data_modules": graph.node_count("DataModule"),
        "components": graph.node_count("Component"),
        "procedures": graph.node_count("Procedure"),
        "faults": graph.node_count("Fault"),
        "references": graph.node_count("Reference"),
        "figures": graph.node_count("Figure"),
        "graphic_assets": graph.node_count("GraphicAsset"),
        "hotspots": graph.node_count("Hotspot"),
        "warnings": graph.node_count("Warning"),
        "cautions": graph.node_count("Caution"),
        "outputs": outputs,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
