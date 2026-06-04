from src.rag.ontology import Intent, OntologyNode, ParsedQuery
from src.rag.v4.rdf_resolver import FallingBackOntologyStore, SparqlEndpointOntologyStore, build_rdf_ontology_store


def test_sparql_endpoint_resolves_procedure_dmcs_from_json_bindings():
    captured_queries: list[str] = []

    def fake_query(query: str) -> dict:
        captured_queries.append(query)
        assert "s1000d:hasTarget" in query
        assert "s1000d:hasAction" in query
        assert "brake-pad" in query
        assert "clean" in query
        return {
            "head": {"vars": ["dmc"]},
            "results": {
                "bindings": [
                    {"dmc": {"type": "literal", "value": "BRAKE-PAD-CLEAN"}},
                    {"dmc": {"type": "literal", "value": "BRAKE-PAD-CLEAN"}},
                ]
            },
        }

    store = SparqlEndpointOntologyStore(endpoint="http://graphdb.example/repositories/s1000d", query_fn=fake_query)

    dmcs = store.find_procedure_dmcs("brake pad", "clean")

    assert dmcs == ("BRAKE-PAD-CLEAN",)
    assert captured_queries


def test_sparql_endpoint_resolves_description_dmcs_from_json_bindings():
    def fake_query(query: str) -> dict:
        assert "s1000d:describes" in query
        assert "brake-system" in query
        return {
            "results": {
                "bindings": [
                    {"dmc": {"type": "literal", "value": "BRAKE-DESC"}},
                ]
            }
        }

    store = SparqlEndpointOntologyStore(endpoint="http://graphdb.example/repositories/s1000d", query_fn=fake_query)

    assert store.find_descriptive_dmcs("brake system") == ("BRAKE-DESC",)


def test_rdf_store_factory_uses_sparql_endpoint_when_configured():
    nodes = [
        OntologyNode(
            dmc="BRAKE-DESC",
            title="Brake description",
            dm_type="descriptive",
            target="brake system",
        )
    ]

    store = build_rdf_ontology_store(nodes, sparql_endpoint="http://graphdb.example/repositories/s1000d")

    assert isinstance(store, FallingBackOntologyStore)
    assert isinstance(store.primary, SparqlEndpointOntologyStore)


def test_sparql_endpoint_resolve_query_preserves_related_dmcs_from_primary_resolution():
    def fake_query(query: str) -> dict:
        if "s1000d:hasAction" in query:
            return {"results": {"bindings": [{"dmc": {"value": "BRAKE-PAD-CLEAN"}}]}}
        return {"results": {"bindings": [{"dmc": {"value": "BRAKE-DESC"}}]}}

    store = SparqlEndpointOntologyStore(endpoint="http://graphdb.example/repositories/s1000d", query_fn=fake_query)
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )

    resolution = store.resolve_query(parsed)

    assert resolution.primary_dmcs == ("BRAKE-PAD-CLEAN",)
    assert resolution.related_dmcs == ("BRAKE-DESC",)


def test_sparql_endpoint_failure_falls_back_to_local_rdf_store():
    nodes = [
        OntologyNode(
            dmc="BRAKE-PAD-CLEAN",
            title="Brake pad cleaning",
            dm_type="procedural",
            target="brake pad",
            action="clean",
        ),
        OntologyNode(
            dmc="BRAKE-DESC",
            title="Brake system description",
            dm_type="descriptive",
            target="brake system",
        ),
    ]

    def broken_query(_query: str) -> dict:
        raise TimeoutError("GraphDB endpoint unavailable")

    primary = SparqlEndpointOntologyStore(
        endpoint="http://graphdb.example/repositories/s1000d",
        query_fn=broken_query,
    )
    fallback = build_rdf_ontology_store(nodes)
    store = FallingBackOntologyStore(primary=primary, fallback=fallback)
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )

    resolution = store.resolve_query(parsed)

    assert resolution.primary_dmcs == ("BRAKE-PAD-CLEAN",)
    assert resolution.related_dmcs == ("BRAKE-DESC",)
    assert any("fallback" in path.casefold() for path in resolution.graph_paths)
