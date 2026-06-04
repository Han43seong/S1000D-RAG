import importlib

from langchain_core.documents import Document

from src.rag.ontology import CandidateEvidence, Intent, OntologyNode, ParsedQuery, ResolutionResult, SupportLevel


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


def test_v4_pipeline_does_not_call_llm_for_related_only_procedure(monkeypatch):
    pipeline = importlib.reload(importlib.import_module("src.rag.pipeline_v4"))
    parsed = ParsedQuery(
        original="브레이크 케이블 제거 후 재설치 절차 알려줘",
        normalized="브레이크 케이블 제거 후 재설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake cable",
        action="remove and install",
    )

    class FakeStore:
        def resolve_query(self, _parsed):
            return pipeline.RdfResolution(primary_dmcs=(), related_dmcs=("BRAKE-DESC",))

    class UnsafeLLM:
        invoked = False

        def invoke(self, _prompt):
            self.invoked = True
            return "Step 1: fabricate an unsupported cable removal sequence."

    unsafe_llm = UnsafeLLM()
    monkeypatch.setattr(pipeline, "parse_query", lambda _query: parsed)
    monkeypatch.setattr(pipeline, "load_ontology_manifest", lambda: [])
    monkeypatch.setattr(pipeline, "build_rdf_ontology_store", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(pipeline, "resolve_ontology", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "plan_evidence", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda *_args, **_kwargs: ([Document(page_content="Brake cable routing is described.", metadata={"dmc": "BRAKE-DESC"})], []),
    )

    result = pipeline.run_rag_query_sync(parsed.original, llm=unsafe_llm)

    assert unsafe_llm.invoked is False
    assert "직접 확인되지 않았습니다" in result.answer
    assert "Step 1" not in result.answer


def test_v4_pipeline_returns_answer_plan_metadata_for_related_only_procedure(monkeypatch):
    pipeline = importlib.reload(importlib.import_module("src.rag.pipeline_v4"))
    parsed = ParsedQuery(
        original="브레이크 케이블 제거 후 재설치 절차 알려줘",
        normalized="브레이크 케이블 제거 후 재설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake cable",
        action="remove and install",
    )

    class FakeStore:
        def resolve_query(self, _parsed):
            return pipeline.RdfResolution(primary_dmcs=(), related_dmcs=("BRAKE-DESC",))

    monkeypatch.setattr(pipeline, "parse_query", lambda _query: parsed)
    monkeypatch.setattr(pipeline, "load_ontology_manifest", lambda: [])
    monkeypatch.setattr(pipeline, "build_rdf_ontology_store", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(pipeline, "resolve_ontology", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "plan_evidence", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda *_args, **_kwargs: ([Document(page_content="Brake cable routing is described.", metadata={"dmc": "BRAKE-DESC"})], []),
    )

    result = pipeline.run_rag_query_sync(parsed.original, llm=object())

    assert result.v4_metadata.support_level == "related"
    assert result.v4_metadata.runtime_mode == "deterministic_fallback"
    assert result.v4_metadata.required_citations == ["BRAKE-DESC"]
    assert "unsupported requested procedure" in result.v4_metadata.forbidden_claims
    assert result.v4_metadata.ontology_trace["rdf_related_dmcs"] == ["BRAKE-DESC"]


def test_v4_pipeline_uses_rdf_primary_dmcs_for_evidence_planning(monkeypatch):
    pipeline = importlib.reload(importlib.import_module("src.rag.pipeline_v4"))
    parsed = ParsedQuery(
        original="브레이크 작동원리 설명해줘",
        normalized="브레이크 작동원리 설명해줘",
        intent=Intent.DESCRIBE,
        target="brake system",
    )
    rdf_node = OntologyNode(dmc="RDF-BRAKE-DESC", title="RDF brake system description", dm_type="descriptive", target="brake system")
    stale_node = OntologyNode(dmc="STALE-BRAKE-DESC", title="stale manifest resolver choice", dm_type="descriptive", target="brake system")
    stale_resolution = ResolutionResult(
        parsed=parsed,
        support=SupportLevel.EXACT,
        candidates=(CandidateEvidence(node=stale_node, support=SupportLevel.EXACT, reason="stale_manifest_choice"),),
        reason="stale_manifest_choice",
    )
    captured: dict[str, tuple[str, ...]] = {}

    class FakeStore:
        def resolve_query(self, _parsed):
            return pipeline.RdfResolution(primary_dmcs=("RDF-BRAKE-DESC",), related_dmcs=())

    def fake_retrieve(plan, _vectorstore=None):
        captured["dmcs"] = plan.dmcs
        return [Document(page_content="RDF-selected brake evidence", metadata={"dmc": plan.dmcs[0], "title": "RDF brake system description"})], []

    monkeypatch.setattr(pipeline, "parse_query", lambda _query: parsed)
    monkeypatch.setattr(pipeline, "load_ontology_manifest", lambda: [rdf_node, stale_node])
    monkeypatch.setattr(pipeline, "build_rdf_ontology_store", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(pipeline, "resolve_ontology", lambda *_args, **_kwargs: stale_resolution)
    monkeypatch.setattr(pipeline, "retrieve_evidence", fake_retrieve)

    result = pipeline.run_rag_query_sync(parsed.original, llm=None)

    assert captured["dmcs"] == ("RDF-BRAKE-DESC",)
    assert result.v4_metadata.ontology_trace["rdf_primary_dmcs"] == ["RDF-BRAKE-DESC"]
    assert "RDF-BRAKE-DESC" in result.answer
