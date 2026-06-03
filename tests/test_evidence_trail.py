from __future__ import annotations

import json

from src.rag.evidence_trail import collect_reference_materials
from src.types.rag import Evidence, ReferenceMaterials


def test_collect_reference_materials_returns_empty_when_graph_missing(tmp_path):
    materials = collect_reference_materials(
        [Evidence(dmc="DMC-001", chunk_id="c1", score=0.9)],
        graph_path=tmp_path / "missing.json",
    )

    assert materials == ReferenceMaterials()


def test_collect_reference_materials_uses_ontology_edges_and_stable_categories(tmp_path):
    graph = {
        "nodes": [
            {"id": "dm:DMC-A", "type": "DataModule", "label": "Main DM", "properties": {"dmc": "DMC-A"}},
            {"id": "procedure:main", "type": "Procedure", "label": "Main procedure", "properties": {"target": "brake", "action": "clean"}},
            {"id": "warning:DMC-A:1", "type": "Warning", "label": "warning title", "properties": {"dmc": "DMC-A", "text": "Wear goggles"}},
            {"id": "caution:DMC-A:1", "type": "Caution", "label": "caution title", "properties": {"dmc": "DMC-A", "text": "Do not spray hubs"}},
            {"id": "figure:DMC-A:fig-1", "type": "Figure", "label": "Brake figure", "properties": {"dmc": "DMC-A", "figure_id": "fig-1"}},
            {"id": "asset:ICN-1", "type": "GraphicAsset", "label": "ICN-1", "properties": {"icn": "ICN-1"}},
            {"id": "hotspot:DMC-A:hot-1", "type": "Hotspot", "label": "hot-1", "properties": {"hotspot_id": "hot-1"}},
            {"id": "reference:DMC-A:DMC-B", "type": "Reference", "label": "DMC-B", "properties": {"dmc": "DMC-B", "source_dmc": "DMC-A"}},
            {"id": "dm:DMC-B", "type": "DataModule", "label": "Referenced DM", "properties": {"dmc": "DMC-B"}},
        ],
        "edges": [
            {"source": "dm:DMC-A", "predicate": "HAS_WARNING", "target": "warning:DMC-A:1", "properties": {}},
            {"source": "dm:DMC-A", "predicate": "HAS_CAUTION", "target": "caution:DMC-A:1", "properties": {}},
            {"source": "dm:DMC-A", "predicate": "HAS_FIGURE", "target": "figure:DMC-A:fig-1", "properties": {}},
            {"source": "figure:DMC-A:fig-1", "predicate": "USES_ASSET", "target": "asset:ICN-1", "properties": {}},
            {"source": "figure:DMC-A:fig-1", "predicate": "HAS_HOTSPOT", "target": "hotspot:DMC-A:hot-1", "properties": {}},
            {"source": "dm:DMC-A", "predicate": "REFERENCES", "target": "reference:DMC-A:DMC-B", "properties": {}},
            {"source": "reference:DMC-A:DMC-B", "predicate": "REFERENCES", "target": "dm:DMC-B", "properties": {}},
            {"source": "procedure:main", "predicate": "GROUNDED_IN", "target": "dm:DMC-A", "properties": {}},
        ],
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    materials = collect_reference_materials(
        [Evidence(dmc="DMC-A", chunk_id="c1", score=0.9)],
        graph_path=graph_path,
    )

    assert [item.id for item in materials.data_modules] == ["dm:DMC-A"]
    assert [item.id for item in materials.procedures] == ["procedure:main"]
    assert [item.text for item in materials.warnings] == ["Wear goggles"]
    assert [item.text for item in materials.cautions] == ["Do not spray hubs"]
    assert [item.title for item in materials.figures] == ["Brake figure"]
    assert [item.id for item in materials.graphic_assets] == ["asset:ICN-1"]
    assert [item.id for item in materials.hotspots] == ["hotspot:DMC-A:hot-1"]
    assert [item.dmc for item in materials.references] == ["DMC-B"]


def test_collect_reference_materials_matches_shortened_ontology_dmc_variants():
    materials = collect_reference_materials(
        [Evidence(dmc="BRAKE-AAA-DA1-00-00-00AA-041A-A", chunk_id="c1", score=0.9)]
    )

    assert any(item.title == "Cantilever brake with straddle cable" for item in materials.figures)
    assert any(item.id.startswith("asset:ICN-") for item in materials.graphic_assets)
