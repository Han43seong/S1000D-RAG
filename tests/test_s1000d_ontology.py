"""S1000D ontology graph export tests."""

from __future__ import annotations

from pathlib import Path

from src.rag.graph_retrieval import FaultNode, GraphManifest, ProcedureNode
from src.rag.ontology import (
    build_ontology_from_manifest,
    build_ontology_from_xml_dir,
    ontology_to_jsonld,
    ontology_to_turtle,
)


def test_ontology_from_manifest_has_component_action_dm_edges():
    manifest = GraphManifest(
        procedures=[
            ProcedureNode(
                dmc="S1000DBIKE-AAA-DA4-10-00-00AA-241A-A",
                title="Chain - Oil",
                target="chain",
                action="oil",
                sns_code="DA4",
            )
        ],
        faults=[
            FaultNode(
                dmc="S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A",
                title="Lights - Fault",
                target="lights",
                sns_code="D00",
            )
        ],
    )

    graph = build_ontology_from_manifest(manifest)

    assert graph.node_count("DataModule") == 2
    assert graph.node_count("Component") == 2
    assert graph.has_edge("component:chain", "HAS_PROCEDURE", "procedure:chain:oil")
    assert graph.has_edge("procedure:chain:oil", "GROUNDED_IN", "dm:S1000DBIKE-AAA-DA4-10-00-00AA-241A-A")
    assert graph.has_edge("component:lights", "HAS_FAULT_DOC", "dm:S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A")


def test_ontology_exports_jsonld_and_turtle():
    manifest = GraphManifest(
        procedures=[ProcedureNode(dmc="DMC-CHAIN-OIL", title="Chain - Oil", target="chain", action="oil")]
    )
    graph = build_ontology_from_manifest(manifest)

    jsonld = ontology_to_jsonld(graph)
    turtle = ontology_to_turtle(graph)

    assert jsonld["@context"]["s1000d"] == "https://example.org/s1000d#"
    assert any(node["@id"] == "dm:DMC-CHAIN-OIL" for node in jsonld["@graph"])
    assert "s1000d:HAS_PROCEDURE" in turtle
    assert "dm:DMC-CHAIN-OIL" in turtle


def test_ontology_from_xml_extracts_dmref_and_figure_relations(tmp_path: Path):
    xml = tmp_path / "DMC-SAMPLE.XML"
    xml.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<dmodule>
  <identAndStatusSection>
    <dmAddress>
      <dmIdent><dmCode modelIdentCode='S1000D' systemDiffCode='AAA' systemCode='DA4' subSystemCode='10' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='241' infoCodeVariant='A' itemLocationCode='A'/></dmIdent>
      <dmAddressItems><dmTitle><techName>Chain</techName><infoName>Oil</infoName></dmTitle></dmAddressItems>
    </dmAddress>
  </identAndStatusSection>
  <content>
    <procedure>
      <mainProcedure>
        <proceduralStep>
          <para>Refer to another module.</para>
          <dmRef><dmRefIdent><dmCode modelIdentCode='S1000D' systemDiffCode='AAA' systemCode='DA1' subSystemCode='00' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='341' infoCodeVariant='A' itemLocationCode='A'/></dmRefIdent></dmRef>
          <figure id='fig-0001'><title>Chain lubrication</title><graphic infoEntityIdent='ICN-SAMPLE-0001'/><hotspot id='hot-1'/></figure>
        </proceduralStep>
      </mainProcedure>
    </procedure>
  </content>
</dmodule>
""",
        encoding="utf-8",
    )

    graph = build_ontology_from_xml_dir(tmp_path)

    source_dm = "dm:S1000D-AAA-DA4-10-00-00AA-241A-A"
    target_dm = "dm:S1000D-AAA-DA1-00-00-00AA-341A-A"
    assert graph.has_edge(source_dm, "REFERENCES", target_dm)
    assert graph.has_edge(source_dm, "HAS_FIGURE", "figure:S1000D-AAA-DA4-10-00-00AA-241A-A:fig-0001")
    assert graph.has_edge("figure:S1000D-AAA-DA4-10-00-00AA-241A-A:fig-0001", "USES_ASSET", "asset:ICN-SAMPLE-0001")
    assert graph.has_edge("figure:S1000D-AAA-DA4-10-00-00AA-241A-A:fig-0001", "HAS_HOTSPOT", "hotspot:S1000D-AAA-DA4-10-00-00AA-241A-A:hot-1")


def test_xml_dmref_only_uses_dmref_context_and_deduplicates_edges(tmp_path: Path):
    xml = tmp_path / "DMC-REFS.XML"
    xml.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<dmodule>
  <identAndStatusSection><dmAddress><dmIdent><dmCode modelIdentCode='SRC' systemDiffCode='AAA' systemCode='DA4' subSystemCode='10' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='241' infoCodeVariant='A' itemLocationCode='A'/></dmIdent></dmAddress></identAndStatusSection>
  <content>
    <randomContainer><dmCode modelIdentCode='NOTREF' systemDiffCode='AAA' systemCode='D00' subSystemCode='00' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='000' infoCodeVariant='A' itemLocationCode='A'/></randomContainer>
    <para><dmRef><dmRefIdent><dmCode modelIdentCode='TGT' systemDiffCode='AAA' systemCode='DA1' subSystemCode='00' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='341' infoCodeVariant='A' itemLocationCode='A'/></dmRefIdent></dmRef></para>
    <para><dmRef><dmRefIdent><dmCode modelIdentCode='TGT' systemDiffCode='AAA' systemCode='DA1' subSystemCode='00' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='341' infoCodeVariant='A' itemLocationCode='A'/></dmRefIdent></dmRef></para>
  </content>
</dmodule>
""",
        encoding="utf-8",
    )

    graph = build_ontology_from_xml_dir(tmp_path)

    source_dm = "dm:SRC-AAA-DA4-10-00-00AA-241A-A"
    target_dm = "dm:TGT-AAA-DA1-00-00-00AA-341A-A"
    non_ref_dm = "dm:NOTREF-AAA-D00-00-00-00AA-000A-A"
    assert target_dm in graph.nodes
    assert non_ref_dm not in graph.nodes
    reference_id = next(node_id for node_id, node in graph.nodes.items() if node.type == "Reference")
    assert graph.node_count("Reference") == 1
    assert graph.has_edge(source_dm, "REFERENCES", reference_id)
    assert graph.has_edge(reference_id, "REFERENCES", target_dm)
    assert sum(1 for edge in graph.edges if edge.source == source_dm and edge.predicate == "REFERENCES" and edge.target == target_dm) == 1


def test_xml_extracts_warning_and_caution_nodes(tmp_path: Path):
    xml = tmp_path / "DMC-SAFETY.XML"
    xml.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<dmodule>
  <identAndStatusSection><dmAddress><dmIdent><dmCode modelIdentCode='SAFE' systemDiffCode='AAA' systemCode='D00' subSystemCode='00' subSubSystemCode='00' assyCode='00' disassyCode='00' disassyCodeVariant='AA' infoCode='720' infoCodeVariant='A' itemLocationCode='A'/></dmIdent></dmAddress></identAndStatusSection>
  <content><procedure><mainProcedure><proceduralStep>
    <warning warningType='hazard'><warningAndCautionPara>Keep hands clear.</warningAndCautionPara></warning>
    <caution><warningAndCautionPara>Do not over-tighten.</warningAndCautionPara></caution>
  </proceduralStep></mainProcedure></procedure></content>
</dmodule>
""",
        encoding="utf-8",
    )

    graph = build_ontology_from_xml_dir(tmp_path)

    source_dm = "dm:SAFE-AAA-D00-00-00-00AA-720A-A"
    assert graph.node_count("Warning") == 1
    assert graph.node_count("Caution") == 1
    warning_id = next(node_id for node_id, node in graph.nodes.items() if node.type == "Warning")
    caution_id = next(node_id for node_id, node in graph.nodes.items() if node.type == "Caution")
    assert graph.nodes[warning_id].label == "Keep hands clear."
    assert graph.nodes[warning_id].properties["warning_type"] == "hazard"
    assert graph.has_edge(source_dm, "HAS_WARNING", warning_id)
    assert graph.has_edge(source_dm, "HAS_CAUTION", caution_id)


def test_property_graph_export_is_stable_and_includes_semantic_counts(tmp_path: Path):
    manifest = GraphManifest(
        procedures=[ProcedureNode(dmc="DMC-CHAIN-OIL", title="Chain - Oil", target="chain", action="oil")]
    )
    graph = build_ontology_from_manifest(manifest)
    first = graph.to_dict()
    second = graph.to_dict()

    assert first == second
    assert [node["id"] for node in first["nodes"]] == sorted(node["id"] for node in first["nodes"])
    assert [edge["predicate"] for edge in first["edges"]] == sorted(edge["predicate"] for edge in first["edges"])

    jsonld = ontology_to_jsonld(graph)
    turtle = ontology_to_turtle(graph)
    assert any(item.get("@type") == "s1000d:Procedure" for item in jsonld["@graph"])
    assert "s1000d:GROUNDED_IN" in turtle
