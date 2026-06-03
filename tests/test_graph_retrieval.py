"""Graph-first retrieval tests for S1000D structured metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.rag.graph_retrieval import (
    GraphManifest,
    FaultNode,
    ProcedureNode,
    build_graph_from_chunk_metadata,
    resolve_graph_candidates,
)
from src.rag.pipeline import run_rag_query_sync
from src.types.rag import RagOptions, RerankOptions


def test_graph_builder_extracts_procedures_from_s1000d_titles():
    manifest = build_graph_from_chunk_metadata([
        {
            "dmc": "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A",
            "title": "Chain - Oil",
            "dm_type": "procedural",
            "sns_code": "DA4",
            "chunk_id": "chain-oil-001",
        },
        {
            "dmc": "S1000DBIKE-AAA-DA2-20-00-00AA-520A-A",
            "title": "Handlebar - Remove procedures",
            "dm_type": "procedural",
            "sns_code": "DA2",
            "chunk_id": "handlebar-remove-001",
        },
    ])

    assert manifest.find_procedures(target="chain", action="oil")[0].dmc == "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"
    assert manifest.find_procedures(target="handlebar", action="remove")[0].dmc == "S1000DBIKE-AAA-DA2-20-00-00AA-520A-A"


def test_graph_builder_normalizes_action_modifiers_and_fault_documents():
    manifest = build_graph_from_chunk_metadata([
        {
            "dmc": "S1000DBIKE-AAA-DA1-10-00-00AA-251A-A",
            "title": "Brake pads - Clean with rubbing alcohol",
            "dm_type": "procedural",
            "sns_code": "DA1",
        },
        {
            "dmc": "S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A",
            "title": "Lights - Fault isolation",
            "dm_type": "fault",
            "sns_code": "D00",
        },
    ])

    assert manifest.find_procedures(target="brake pad", action="clean")[0].dmc == "S1000DBIKE-AAA-DA1-10-00-00AA-251A-A"
    assert manifest.find_faults(target="lights")[0].dmc == "S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A"


def test_graph_resolver_uses_full_corpus_relationships_not_hardcoded_sns():
    manifest = GraphManifest(
        procedures=[
            ProcedureNode(dmc="S1000DBIKE-AAA-DA4-10-00-00AA-241A-A", title="Chain - Oil", target="chain", action="oil"),
            ProcedureNode(dmc="S1000DBIKE-AAA-DA2-10-00-00AA-520A-A", title="Stem - Remove procedures", target="stem", action="remove"),
            ProcedureNode(dmc="S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A", title="Lights - Manual test", target="lights", action="test"),
        ],
        faults=[
            FaultNode(dmc="S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A", title="Lights - Fault isolation", target="lights"),
        ],
    )

    assert resolve_graph_candidates("체인에 오일을 바르는 절차를 알려줘", manifest).dmcs == ["S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"]
    assert resolve_graph_candidates("스템을 분리하는 절차를 알려줘", manifest).dmcs == ["S1000DBIKE-AAA-DA2-10-00-00AA-520A-A"]
    assert resolve_graph_candidates("자전거 조명 시스템이 정상 동작하는지 어떻게 확인해?", manifest).dmcs == ["S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A"]
    assert resolve_graph_candidates("조명이 켜지지 않을 때 어떤 fault 문서를 봐야 해?", manifest).dmcs == ["S1000DLIGHTING-AAA-D00-00-00-00AA-413A-A"]


class RecordingVectorStore:
    def __init__(self):
        self.filters = []

    def similarity_search_with_relevance_scores(self, query, **kwargs):
        self.filters.append(kwargs.get("filter"))
        filt = kwargs.get("filter") or {}
        dmc = filt.get("dmc") if isinstance(filt, dict) else None
        if dmc == "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A":
            return [(
                Document(
                    page_content="Apply oil to the chain pivots.",
                    metadata={
                        "dmc": dmc,
                        "chunk_id": "chain-oil-001",
                        "dm_type": "procedural",
                        "security": "01",
                        "title": "Chain - Oil",
                    },
                ),
                0.91,
            )]
        return []


def test_rag_pipeline_prefers_graph_dmc_candidates_before_global_vector_search():
    manifest = GraphManifest(procedures=[
        ProcedureNode(dmc="S1000DBIKE-AAA-DA4-10-00-00AA-241A-A", title="Chain - Oil", target="chain", action="oil"),
    ])
    vectorstore = RecordingVectorStore()
    llm = MagicMock()
    llm.invoke.return_value = "체인에 오일을 바릅니다.\n근거: S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"

    with patch("src.rag.pipeline.load_graph_manifest", return_value=manifest):
        result = run_rag_query_sync(
            query="체인에 오일을 바르는 절차를 알려줘",
            vectorstore=vectorstore,
            llm=llm,
            options=RagOptions(rerank=RerankOptions(enabled=False, top_k=3), relevance_threshold=0.0),
        )

    assert vectorstore.filters[0] == {"dmc": "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"}
    assert result.evidences[0].dmc == "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"


def test_rag_pipeline_falls_back_when_llm_returns_empty_answer():
    manifest = GraphManifest(procedures=[
        ProcedureNode(dmc="S1000DBIKE-AAA-DA4-10-00-00AA-241A-A", title="Chain - Oil", target="chain", action="oil"),
    ])
    vectorstore = RecordingVectorStore()
    llm = MagicMock()
    llm.invoke.return_value = ""

    with patch("src.rag.pipeline.load_graph_manifest", return_value=manifest):
        result = run_rag_query_sync(
            query="체인에 오일을 바르는 절차를 알려줘.",
            vectorstore=vectorstore,
            llm=llm,
            options=RagOptions(rerank=RerankOptions(enabled=False, top_k=3), relevance_threshold=0.0),
        )

    assert result.answer
    assert "S1000DBIKE-AAA-DA4-10-00-00AA-241A-A" in result.answer
    assert "Chain - Oil" in result.answer or "체인" in result.answer


def test_guarded_missing_procedure_answer_does_not_use_evidence_label():
    from src.rag.pipeline import _guard_procedure_question
    from src.types.rag import Evidence

    doc = Document(
        page_content="Find the brake pads. Apply rubbing alcohol.",
        metadata={
            "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
            "chunk_id": "brake-pad-clean-001",
            "dm_type": "procedural",
            "title": "Brake pads - Clean with rubbing alcohol",
        },
    )
    evidences = [Evidence(dmc="BRAKE-AAA-DA1-10-00-00AA-251A-A", chunk_id="brake-pad-clean-001", score=0.9)]

    guarded = _guard_procedure_question(
        "브레이크 패드를 교체하거나 장착할 때 주의할 점은 뭐야?",
        [(doc, 0.9)],
        evidences,
    )

    assert guarded is not None
    assert "근거:" not in guarded.answer
    assert "BRAKE-AAA-DA1-10-00-00AA-251A-A" in guarded.answer
