from src.rag.ontology import OntologyNode
from src.rag.v4.graph_builder import build_graph_context
from src.rag.v4.graph_schema import NodeType, RelationType


def test_v4_builds_graph_context_from_ontology_manifest_nodes():
    nodes = [
        OntologyNode(
            dmc="BRAKE-AAA-DA1-00-00-00AA-041A-A",
            title="Brake system - Description",
            dm_type="descriptive",
            target="brake system",
            aliases=("브레이크",),
        ),
        OntologyNode(
            dmc="BRAKE-AAA-DA1-20-00-00AA-520A-A",
            title="Brake pad - Clean",
            dm_type="procedural",
            target="brake pad",
            action="clean",
        ),
    ]

    graph = build_graph_context(nodes)

    assert graph.find_node("dmc:BRAKE-AAA-DA1-00-00-00AA-041A-A").node_type == NodeType.DATA_MODULE
    assert graph.find_node("system:brake-system").node_type == NodeType.SYSTEM
    assert graph.find_node("procedure:brake-pad:clean").node_type == NodeType.PROCEDURE
    assert any(edge.relation == RelationType.HAS_DESCRIPTION for edge in graph.edges)
    assert any(edge.relation == RelationType.HAS_PROCEDURE for edge in graph.edges)


def test_v4_graph_context_expands_related_documents_by_target_family():
    nodes = [
        OntologyNode("BRAKE-DESC", "Brake system description", "descriptive", target="brake system"),
        OntologyNode("BRAKE-PAD-CLEAN", "Brake pad clean", "procedural", target="brake pad", action="clean"),
        OntologyNode("WHEEL-INSTALL", "Wheel install", "procedural", target="wheel", action="install"),
    ]
    graph = build_graph_context(nodes)

    related = graph.related_dmcs_for_target("brake system")

    assert "BRAKE-DESC" in related
    assert "BRAKE-PAD-CLEAN" in related
    assert "WHEEL-INSTALL" not in related
