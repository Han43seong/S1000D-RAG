from src.rag.ontology import Intent, OntologyNode, ParsedQuery
from src.rag.v4.rdf_exporter import export_ontology_jsonld, export_ontology_turtle, ontology_nodes_to_triples
from src.rag.v4.rdf_resolver import RdfOntologyStore


def _sample_nodes() -> list[OntologyNode]:
    return [
        OntologyNode(
            dmc="BRAKE-DESC",
            title="Brake system - Description",
            dm_type="descriptive",
            sns_code="DA1",
            target="brake system",
            aliases=("브레이크", "brake"),
            metadata={"components": ["brake lever", "brake cable", "brake pad"]},
        ),
        OntologyNode(
            dmc="BRAKE-PAD-CLEAN",
            title="Brake pad - Clean",
            dm_type="procedural",
            sns_code="DA1",
            target="brake pad",
            action="clean",
        ),
        OntologyNode(
            dmc="WHEEL-INSTALL",
            title="Wheel - Install",
            dm_type="procedural",
            sns_code="DA2",
            target="wheel",
            action="install",
        ),
    ]


def test_ontology_nodes_export_to_rdf_triples_and_turtle():
    triples = ontology_nodes_to_triples(_sample_nodes())
    turtle = export_ontology_turtle(_sample_nodes())

    assert ("<https://example.org/s1000d/dm/BRAKE-DESC>", "rdf:type", "s1000d:DescriptiveDataModule") in triples
    assert ("<https://example.org/s1000d/dm/BRAKE-PAD-CLEAN>", "s1000d:hasTarget", "<https://example.org/s1000d/entity/brake-pad>") in triples
    assert ("<https://example.org/s1000d/dm/BRAKE-PAD-CLEAN>", "s1000d:hasAction", "<https://example.org/s1000d/action/clean>") in triples
    assert ("<https://example.org/s1000d/entity/brake-system>", "s1000d:hasComponent", "<https://example.org/s1000d/entity/brake-pad>") in triples
    assert "@prefix s1000d:" in turtle
    assert "s1000d:ProceduralDataModule" in turtle
    assert "Brake pad - Clean" in turtle


def test_ontology_nodes_export_to_jsonld_graph():
    jsonld = export_ontology_jsonld(_sample_nodes())

    assert jsonld["@context"]["s1000d"] == "https://example.org/s1000d/"
    graph = jsonld["@graph"]
    brake_desc = next(item for item in graph if item.get("s1000d:dmc") == "BRAKE-DESC")

    assert brake_desc["@id"] == "https://example.org/s1000d/dm/BRAKE-DESC"
    assert brake_desc["@type"] == "s1000d:DescriptiveDataModule"
    assert brake_desc["s1000d:describes"]["@id"] == "https://example.org/s1000d/entity/brake-system"
    assert {component["@id"] for component in brake_desc["s1000d:hasComponent"]} >= {
        "https://example.org/s1000d/entity/brake-lever",
        "https://example.org/s1000d/entity/brake-pad",
    }

    brake_clean = next(item for item in graph if item.get("s1000d:dmc") == "BRAKE-PAD-CLEAN")
    assert brake_clean["s1000d:hasAction"]["@id"] == "https://example.org/s1000d/action/clean"


def test_rdf_store_resolves_description_and_procedure_with_sparql_like_queries():
    store = RdfOntologyStore.from_nodes(_sample_nodes())

    desc = store.find_descriptive_dmcs(target="brake system")
    proc = store.find_procedure_dmcs(target="brake pad", action="clean")

    assert desc == ("BRAKE-DESC",)
    assert proc == ("BRAKE-PAD-CLEAN",)


def test_rdf_store_expands_related_dmc_family_without_wheel_noise():
    store = RdfOntologyStore.from_nodes(_sample_nodes())

    related = store.related_dmcs_for_target("brake system")

    assert "BRAKE-DESC" in related
    assert "BRAKE-PAD-CLEAN" in related
    assert "WHEEL-INSTALL" not in related


def test_rdf_store_resolves_parsed_query_to_primary_and_related_dmcs():
    store = RdfOntologyStore.from_nodes(_sample_nodes())
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )

    result = store.resolve_query(parsed)

    assert result.primary_dmcs == ("BRAKE-PAD-CLEAN",)
    assert "BRAKE-DESC" in result.related_dmcs
    assert "WHEEL-INSTALL" not in result.related_dmcs
