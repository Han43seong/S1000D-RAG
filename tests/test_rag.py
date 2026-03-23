"""Phase 5 테스트: RAG Pipeline (retriever, reranker, pipeline).

모델 로딩 없이 mock으로 로직 검증.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_core.documents import Document

from src.rag.retriever import MetaFilter, _build_where_filter, retrieve_two_stage
from src.rag.reranker import rerank
from src.rag.pipeline import (
    _build_context,
    _build_meta_filter,
    run_rag_query_sync,
)
from src.rag.prompt import build_prompt
from src.types.rag import (
    Evidence,
    RagOptions,
    RagResult,
    RerankOptions,
    SessionMeta,
)


# ═══════════════════════════════════════════════════════════════════════
# Retriever 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestBuildWhereFilter:
    def test_none_filter(self):
        assert _build_where_filter(None) is None

    def test_empty_filter(self):
        assert _build_where_filter(MetaFilter()) is None

    def test_single_field(self):
        f = _build_where_filter(MetaFilter(security="01"))
        assert f == {"security": "01"}

    def test_multiple_fields(self):
        f = _build_where_filter(MetaFilter(security="01", dm_type="procedural"))
        assert f == {"$and": [{"security": "01"}, {"dm_type": "procedural"}]}

    def test_dmc_filter(self):
        f = _build_where_filter(MetaFilter(dmc="BRAKE-AAA-DA1"))
        assert f == {"dmc": "BRAKE-AAA-DA1"}

    def test_sns_code_filter(self):
        f = _build_where_filter(MetaFilter(sns_code="DA1"))
        assert f == {"sns_code": "DA1"}

    def test_sns_code_with_security(self):
        f = _build_where_filter(MetaFilter(security="01", sns_code="DA1"))
        assert f == {"$and": [{"security": "01"}, {"sns_code": "DA1"}]}


class TestBuildMetaFilter:
    def test_none_session(self):
        assert _build_meta_filter(None) is None

    def test_with_security_clearance(self):
        meta = SessionMeta(security_clearance="01")
        result = _build_meta_filter(meta)
        assert result is not None
        assert result.security == "01"

    def test_without_clearance(self):
        meta = SessionMeta(user_id="user1")
        result = _build_meta_filter(meta)
        assert result is not None
        assert result.security is None


# ═══════════════════════════════════════════════════════════════════════
# Reranker 테스트
# ═══════════════════════════════════════════════════════════════════════


def _make_doc_pairs(n: int) -> list[tuple[Document, float]]:
    """테스트용 (Document, score) 리스트 생성."""
    return [
        (
            Document(
                page_content=f"Content for chunk {i}",
                metadata={
                    "dmc": f"DMC-{i:03d}",
                    "chunk_id": f"chunk-{i:03d}",
                    "dm_type": "procedural",
                    "security": "01",
                    "applicability": "All",
                },
            ),
            0.9 - i * 0.1,
        )
        for i in range(n)
    ]


class TestReranker:
    def test_disabled_passthrough(self):
        """리랭커 비활성화 시 top_k까지만 잘라서 반환."""
        pairs = _make_doc_pairs(5)
        options = RerankOptions(enabled=False, top_k=3)
        result = rerank("query", pairs, options)

        assert len(result) == 3
        # 원본 순서 유지
        assert result[0][0].metadata["chunk_id"] == "chunk-000"

    def test_empty_input(self):
        options = RerankOptions(enabled=True, top_k=3)
        result = rerank("query", [], options)
        assert result == []

    def test_with_mock_cross_encoder(self):
        """mock CrossEncoder로 리랭킹 로직 검증."""
        pairs = _make_doc_pairs(4)
        options = RerankOptions(enabled=True, top_k=2)

        # CrossEncoder mock: 역순으로 점수 부여
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.1, 0.5, 0.9, 0.3]

        result = rerank("query", pairs, options, cross_encoder=mock_encoder)

        # rerank score 기준 top_k=2: index 2 (0.9), index 1 (0.5)
        assert len(result) == 2
        assert result[0][0].metadata["chunk_id"] == "chunk-002"
        assert result[1][0].metadata["chunk_id"] == "chunk-001"
        assert result[0][1] == 0.9
        assert result[1][1] == 0.5


# ═══════════════════════════════════════════════════════════════════════
# Pipeline 컨텍스트 빌더 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestBuildContext:
    def test_basic_context(self):
        pairs = _make_doc_pairs(3)
        context, evidences = _build_context(pairs, max_chars=10000)

        assert len(evidences) == 3
        assert "DMC-000" in context
        assert "Content for chunk 0" in context

        assert evidences[0].dmc == "DMC-000"
        assert evidences[0].chunk_id == "chunk-000"
        assert evidences[0].score == 0.9

    def test_max_chars_limit(self):
        """max_chars 제한으로 일부 문서만 포함."""
        pairs = _make_doc_pairs(5)
        # 각 chunk text ~20자 + header ~40자 = ~60자/chunk
        context, evidences = _build_context(pairs, max_chars=150)

        assert len(evidences) < 5
        assert len(evidences) >= 1

    def test_empty_input(self):
        context, evidences = _build_context([], max_chars=10000)
        assert context == ""
        assert evidences == []


# ═══════════════════════════════════════════════════════════════════════
# Pipeline 통합 테스트 (mock LLM + mock vectorstore)
# ═══════════════════════════════════════════════════════════════════════


class TestRunRagQuerySync:
    def test_basic_pipeline(self):
        """mock으로 전체 파이프라인 동작 검증."""
        # mock vectorstore
        mock_vs = MagicMock()
        pairs = _make_doc_pairs(3)
        mock_vs.similarity_search_with_relevance_scores.return_value = pairs

        # mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The brake system consists of pads and levers."

        result = run_rag_query_sync(
            query="How does the brake system work?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(
                top_k=3,
                rerank=RerankOptions(enabled=False),
            ),
        )

        assert isinstance(result, RagResult)
        assert "brake" in result.answer.lower()
        assert len(result.evidences) == 3
        assert result.evidences[0].dmc == "DMC-000"

        # LLM이 호출되었는지 확인
        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "Context" in prompt_arg
        assert "Question" in prompt_arg

    def test_no_results(self):
        """검색 결과가 없을 때."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = []

        mock_llm = MagicMock()

        result = run_rag_query_sync(
            query="nonexistent topic",
            vectorstore=mock_vs,
            llm=mock_llm,
        )

        assert "찾을 수 없습니다" in result.answer
        assert result.evidences == []
        # LLM은 호출되지 않아야 함
        mock_llm.invoke.assert_not_called()

    def test_with_session_meta(self):
        """SessionMeta가 필터에 반영."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(2)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"

        session = SessionMeta(security_clearance="01", user_id="test")

        result = run_rag_query_sync(
            query="test query",
            vectorstore=mock_vs,
            llm=mock_llm,
            session_meta=session,
            options=RagOptions(rerank=RerankOptions(enabled=False)),
        )

        # security 필터가 적용되었는지 확인
        call_kwargs = mock_vs.similarity_search_with_relevance_scores.call_args
        assert call_kwargs[1].get("filter") == {"security": "01"}

    def test_with_reranker(self):
        """리랭커 활성화 시 동작."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(4)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Reranked answer"

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.1, 0.9, 0.5, 0.3]

        result = run_rag_query_sync(
            query="test",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(
                top_k=4,
                rerank=RerankOptions(enabled=True, top_k=2),
            ),
            cross_encoder=mock_encoder,
        )

        # 리랭킹으로 top_k=2만 evidence에 포함
        assert len(result.evidences) == 2
        # 최상위 evidence는 rerank score 0.9인 chunk-001
        assert result.evidences[0].dmc == "DMC-001"

    def test_llm_output_with_content_attr(self):
        """LLM이 .content 속성을 가진 객체를 반환할 때."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(1)

        mock_response = MagicMock()
        mock_response.content = "  Answer with content attr  "

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        result = run_rag_query_sync(
            query="test",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False)),
        )

        assert result.answer == "Answer with content attr"

    def test_with_conversation_history(self):
        """대화 이력이 프롬프트에 포함."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(2)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Follow-up answer"

        history = [("이전 질문", "이전 답변")]

        result = run_rag_query_sync(
            query="후속 질문",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(
                rerank=RerankOptions(enabled=False),
                expand_query=False,
            ),
            conversation_history=history,
        )

        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "이전 대화" in prompt_arg
        assert "이전 질문" in prompt_arg
        assert "이전 답변" in prompt_arg

    def test_query_expansion_in_pipeline(self):
        """쿼리 확장이 벡터 검색에 반영."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(2)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"

        result = run_rag_query_sync(
            query="브레이크 점검",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(
                rerank=RerankOptions(enabled=False),
                expand_query=True,
                sns_filter=False,  # SNS 2단계 비활성화
            ),
        )

        # 벡터 검색에 확장 쿼리가 사용되었는지 확인
        search_call = mock_vs.similarity_search_with_relevance_scores.call_args
        search_query = search_call[0][0]
        assert "brake" in search_query
        assert "check" in search_query or "inspection" in search_query

    def test_expand_query_disabled(self):
        """expand_query=False이면 원본 쿼리 사용."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(2)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Answer"

        run_rag_query_sync(
            query="브레이크 점검",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(
                rerank=RerankOptions(enabled=False),
                expand_query=False,
            ),
        )

        search_call = mock_vs.similarity_search_with_relevance_scores.call_args
        search_query = search_call[0][0]
        assert search_query == "브레이크 점검"


# ═══════════════════════════════════════════════════════════════════════
# 프롬프트 빌더 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestBuildPrompt:
    def test_basic_prompt(self):
        """기본 프롬프트 생성."""
        prompt = build_prompt("브레이크 교체 방법", "Some context here")
        assert "S1000D" in prompt
        assert "브레이크 교체 방법" in prompt
        assert "Some context here" in prompt
        assert "한↔영" in prompt

    def test_with_history(self):
        """대화 이력 포함."""
        history = [("이전 질문", "이전 답변")]
        prompt = build_prompt("후속 질문", "context", conversation_history=history)
        assert "이전 대화" in prompt
        assert "이전 질문" in prompt
        assert "이전 답변" in prompt

    def test_without_history(self):
        """대화 이력 없으면 이력 섹션 미포함."""
        prompt = build_prompt("질문", "context", conversation_history=None)
        assert "이전 대화" not in prompt

    def test_bilingual_guide(self):
        """한↔영 용어 가이드 포함."""
        prompt = build_prompt("질문", "context")
        assert "Remove" in prompt
        assert "Install" in prompt


# ═══════════════════════════════════════════════════════════════════════
# 2단계 검색 테스트
# ═══════════════════════════════════════════════════════════════════════


def _make_doc_pairs_with_sns(
    n: int, sns_code: str = "DA1"
) -> list[tuple[Document, float]]:
    """sns_code 메타데이터가 포함된 테스트용 (Document, score) 리스트."""
    return [
        (
            Document(
                page_content=f"Content for {sns_code} chunk {i}",
                metadata={
                    "dmc": f"DMC-{sns_code}-{i:03d}",
                    "chunk_id": f"chunk-{sns_code}-{i:03d}",
                    "dm_type": "procedural",
                    "security": "01",
                    "applicability": "All",
                    "sns_code": sns_code,
                },
            ),
            0.9 - i * 0.1,
        )
        for i in range(n)
    ]


class TestRetrieveTwoStage:
    def test_no_sns_fallback(self):
        """sns_code=None이면 기존 retrieve와 동일."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(3)

        results = retrieve_two_stage(mock_vs, "query", top_k=3, sns_code=None)
        assert len(results) == 3
        mock_vs.similarity_search_with_relevance_scores.assert_called_once()

    def test_sns_sufficient(self):
        """SNS 필터 결과가 top_k 이상이면 글로벌 검색 안함."""
        mock_vs = MagicMock()
        sns_docs = _make_doc_pairs_with_sns(5, "DA1")
        mock_vs.similarity_search_with_relevance_scores.return_value = sns_docs

        results = retrieve_two_stage(mock_vs, "query", top_k=3, sns_code="DA1")
        assert len(results) == 3
        # SNS 검색 1회만 호출
        assert mock_vs.similarity_search_with_relevance_scores.call_count == 1

    def test_sns_insufficient_supplements_global(self):
        """SNS 결과 부족 시 글로벌 검색으로 보충."""
        mock_vs = MagicMock()
        sns_docs = _make_doc_pairs_with_sns(1, "DA1")
        global_docs = _make_doc_pairs(5)

        # 1차 SNS 호출 → 2차 글로벌 호출
        mock_vs.similarity_search_with_relevance_scores.side_effect = [
            sns_docs,
            global_docs,
        ]

        results = retrieve_two_stage(mock_vs, "query", top_k=3, sns_code="DA1")
        assert len(results) == 3
        # SNS 1개 + 글로벌 보충 2개
        assert results[0][0].metadata["sns_code"] == "DA1"
        assert mock_vs.similarity_search_with_relevance_scores.call_count == 2
