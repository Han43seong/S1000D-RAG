import importlib


def test_v4_pipeline_passes_sparql_endpoint_env_to_rdf_store_factory(monkeypatch):
    pipeline = importlib.reload(importlib.import_module("src.rag.pipeline_v4"))
    captured: dict[str, str | None] = {}

    class FakeStore:
        def resolve_query(self, parsed):
            return pipeline.RdfResolution(primary_dmcs=("BRAKE-DESC",), related_dmcs=())

    def fake_factory(nodes, sparql_endpoint=None, backend=None):
        captured["endpoint"] = sparql_endpoint
        captured["backend"] = backend
        return FakeStore()

    monkeypatch.setenv("S1000D_SPARQL_ENDPOINT", "http://graphdb.example/repositories/s1000d")
    monkeypatch.setenv("S1000D_RDF_BACKEND", "rdflib")
    monkeypatch.setattr(pipeline, "build_rdf_ontology_store", fake_factory)

    result = pipeline.run_rag_query_sync("브레이크 작동원리 설명해줘")

    assert captured["endpoint"] == "http://graphdb.example/repositories/s1000d"
    assert captured["backend"] == "rdflib"
    assert "BRAKE-DESC" in result.answer
