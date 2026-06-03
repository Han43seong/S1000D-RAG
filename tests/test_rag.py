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
    _build_context_with_optional_visual,
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

    def test_evidence_includes_text_snippet_for_api_display(self):
        pairs = _make_doc_pairs(1)
        context, evidences = _build_context(pairs, max_chars=10000)

        assert "Content for chunk 0" in context
        assert evidences[0].text == "Content for chunk 0"

    def test_visual_query_fuses_caption_candidates_into_context_and_evidence(self, tmp_path):
        chroma_dir = tmp_path / "chroma_db"
        captions_dir = chroma_dir / "visual_captions"
        captions_dir.mkdir(parents=True)
        (captions_dir / "brake-caption.json").write_text(
            """{
  "asset_key": "DMC-BRAKE:figure:fig-1",
  "asset_path": "ICN-BRAKE.CGM",
  "dmc": "DMC-BRAKE",
  "kind": "figure",
  "ref_id": "fig-1",
  "summary": "Mock caption for brake circuit diagram labels.",
  "title": "Brake circuit diagram",
  "components": ["brake lever", "straddle cable"],
  "keywords": ["figure", "diagram", "label"],
  "ocr_text": "",
  "safety_notes": [],
  "status": "mock_captioned",
  "model_profile": "mock-vlm-captioner"
}
""",
            encoding="utf-8",
        )
        docs = [
            (
                Document(
                    page_content="Text description of the brake system.",
                    metadata={"dmc": "DMC-TEXT", "chunk_id": "chunk-1", "dm_type": "descriptive"},
                ),
                0.8,
            )
        ]
        opts = RagOptions(rerank=RerankOptions(enabled=False, top_k=3))

        with patch("src.rag.pipeline.CHROMA_PERSIST_DIR", str(chroma_dir)):
            context, evidences = _build_context_with_optional_visual(
                "브레이크 회로 도면의 라벨 위치를 보여줘",
                docs,
                opts,
            )

        assert "[IMAGE_CAPTION DMC=DMC-BRAKE" in context
        assert any(ev.modality == "image" and ev.content_role == "visual_caption" for ev in evidences)
        visual = next(ev for ev in evidences if ev.modality == "image")
        assert visual.asset_key == "DMC-BRAKE:figure:fig-1"
        assert visual.title == "Brake circuit diagram"


# ═══════════════════════════════════════════════════════════════════════
# Pipeline 통합 테스트 (mock LLM + mock vectorstore)
# ═══════════════════════════════════════════════════════════════════════


class TestRunRagQuerySync:
    def test_physical_brake_pad_location_query_does_not_route_to_visual_caption(self, tmp_path):
        """물리적 부품 위치 질의의 '위치'는 도면/라벨 캡션 답변으로 라우팅하지 않는다."""
        chroma_dir = tmp_path / "chroma_db"
        captions_dir = chroma_dir / "visual_captions"
        captions_dir.mkdir(parents=True)
        (captions_dir / "brake-caption.json").write_text(
            """{
  "asset_key": "DMC-BRAKE:figure:fig-1",
  "asset_path": "ICN-BRAKE.CGM",
  "dmc": "DMC-BRAKE",
  "kind": "figure",
  "ref_id": "fig-1",
  "summary": "Mock caption for figure 'Cantilever brake with straddle cable'.",
  "title": "Cantilever brake with straddle cable",
  "components": ["Cantilever brake with straddle cable"],
  "keywords": ["figure", "diagram", "label"],
  "ocr_text": "",
  "safety_notes": [],
  "status": "mock_captioned",
  "model_profile": "mock-vlm-captioner"
}
""",
            encoding="utf-8",
        )
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Brake pads are installed at the front and rear wheel rims.",
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                        "chunk_id": "brake-pad-location",
                        "dm_type": "descriptive",
                        "title": "Brake system",
                    },
                ),
                0.9,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "앞/뒤 바퀴 브레이크 패드는 각 바퀴의 림 양쪽에 위치해 림을 눌러 감속합니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )

        with patch("src.rag.pipeline.CHROMA_PERSIST_DIR", str(chroma_dir)):
            result = run_rag_query_sync(
                query="앞/뒤 바퀴 브레이크 패드 위치를 알려줘",
                vectorstore=mock_vs,
                llm=mock_llm,
                options=RagOptions(top_k=3, rerank=RerankOptions(enabled=False, top_k=2), expand_query=False),
            )

        assert "림" in result.answer
        assert "Cantilever brake" not in result.answer
        assert not any(ev.content_role == "visual_caption" for ev in result.evidences)
        mock_llm.invoke.assert_called_once()

    def test_visual_query_returns_visual_evidence_without_context_leak(self, tmp_path):
        chroma_dir = tmp_path / "chroma_db"
        captions_dir = chroma_dir / "visual_captions"
        captions_dir.mkdir(parents=True)
        (captions_dir / "brake-caption.json").write_text(
            """{
  "asset_key": "DMC-BRAKE:figure:fig-1",
  "asset_path": "ICN-BRAKE.CGM",
  "dmc": "DMC-BRAKE",
  "kind": "figure",
  "ref_id": "fig-1",
  "summary": "Mock caption for brake circuit diagram labels.",
  "title": "Brake circuit diagram",
  "components": ["brake lever", "straddle cable"],
  "keywords": ["figure", "diagram", "label"],
  "ocr_text": "",
  "safety_notes": [],
  "status": "mock_captioned",
  "model_profile": "mock-vlm-captioner"
}
""",
            encoding="utf-8",
        )
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Brake text chunk.",
                    metadata={"dmc": "DMC-TEXT", "chunk_id": "chunk-1", "dm_type": "descriptive"},
                ),
                0.8,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Context / 참고 문서:\n[IMAGE_CAPTION DMC=DMC-BRAKE]"

        with patch("src.rag.pipeline.CHROMA_PERSIST_DIR", str(chroma_dir)):
            result = run_rag_query_sync(
                query="브레이크 회로 도면의 라벨 위치를 보여줘",
                vectorstore=mock_vs,
                llm=mock_llm,
                options=RagOptions(top_k=3, rerank=RerankOptions(enabled=False, top_k=2)),
            )

        assert not result.answer.startswith("Context / 참고 문서")
        assert "도면" in result.answer
        assert any(ev.modality == "image" and ev.content_role == "visual_caption" for ev in result.evidences)

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

    def test_brake_system_components_query_uses_deterministic_grounded_answer(self):
        """브레이크 주요 구성품 질의는 041A 근거가 있으면 DMC 포함 답변을 직접 반환한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "The brake system has these primary components: the brake lever, "
                "the brake cable, the brake arm, the brake clamp (callipers), and the brake pads."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Brake system",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "브레이크 시스템의 주요 구성품은 브레이크 리버, 브레이크 케이블, "
            "브레이크 팔, 브레이크 클램프(또는 콜리퍼), 브레이크 패드입니다."
        )

        result = run_rag_query_sync(
            query="브레이크 시스템의 주요 구성품은 무엇입니까?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 시스템의 주요 구성품은 브레이크 레버, 브레이크 케이블, 브레이크 암, "
            "브레이크 클램프(콜리퍼), 브레이크 패드입니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        mock_llm.invoke.assert_not_called()

    def test_brake_related_description_document_query_uses_deterministic_grounded_answer(self):
        """QA 188: 브레이크 관련 설명 문서 내용 질의는 LLM 빈 출력에 의존하지 않고 답한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "The brake system has these primary components: the brake lever, "
                "the brake cable, the brake arm, the brake clamp (callipers), and the brake pads. "
                "There are four brake pads on the bicycle. Two are found on the front wheel and two on the rear wheel."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Brake system - Description of how it is made",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""

        result = run_rag_query_sync(
            query="브레이크 관련 설명 문서에는 어떤 내용이 있나요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 관련 설명 문서는 브레이크 시스템의 구성과 패드 배치를 설명합니다. "
            "주요 구성품으로 브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(콜리퍼), "
            "브레이크 패드를 제시하고, 패드는 앞바퀴와 뒷바퀴에 각각 두 개씩 있음을 설명합니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        mock_llm.invoke.assert_not_called()

    def test_brake_cable_detail_query_uses_korean_grounded_answer(self):
        """QA 212: 브레이크 케이블 상세 설명 질의는 영어 원문 답변을 노출하지 않는다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "The brake system has these primary components: the brake lever, "
                "the brake cable, the brake arm, the brake clamp (callipers), and the brake pads. "
                "The adjuster lock nut holds the brake cable. This lock nut adjusts the tension of the cable."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Brake system - Description of how it is made",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "Brake lever\nThe adjuster lock nut holds the brake cable. "
            "This lock nut adjusts the tension of the cable."
        )

        result = run_rag_query_sync(
            query="브레이크 케이블을 조금 더 자세히 설명해줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 케이블은 브레이크 시스템의 주요 구성품 중 하나이며, "
            "브레이크 레버와 브레이크 암/패드 쪽 동작을 연결하는 역할을 합니다. "
            "문서에서는 조정 잠금 너트가 브레이크 케이블을 고정하고 케이블 장력을 조정한다고 설명합니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        mock_llm.invoke.assert_not_called()

    def test_brake_lever_operation_query_uses_deterministic_grounded_answer(self):
        """QA 114: 브레이크 레버 작동 설명은 LLM 빈 출력에 의존하지 않고 근거 기반으로 답한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "The brake system has these primary components: the brake lever, "
                "the brake cable, the brake arm, the brake clamp (callipers), and the brake pads. "
                "The pads press against the rim of the bicycle wheel to decrease the speed of the bicycle."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Brake system - Description of how it is made",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = ""

        result = run_rag_query_sync(
            query="브레이크 레버를 작동하면 어떤 일이 일어나나요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 레버를 작동하면 브레이크 케이블을 통해 브레이크 암과 패드가 움직이고, "
            "브레이크 패드가 바퀴 림을 눌러 마찰력을 만들어 자전거 속도를 줄입니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        mock_llm.invoke.assert_not_called()

    def test_brake_lever_operation_guard_does_not_intercept_procedure_queries(self):
        """브레이크 레버 절차 질의는 설명용 deterministic guard가 가로채지 않는다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "The brake system has these primary components: the brake lever, "
                "the brake cable, the brake arm, the brake clamp, and the brake pads."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Brake system - Description of how it is made",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "절차성 질문은 일반 경로에서 처리합니다.\n참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"

        result = run_rag_query_sync(
            query="브레이크 레버 작동 절차를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer != (
            "브레이크 레버를 작동하면 브레이크 케이블을 통해 브레이크 암과 패드가 움직이고, "
            "브레이크 패드가 바퀴 림을 눌러 마찰력을 만들어 자전거 속도를 줄입니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        assert "브레이크 레버 절차" in result.answer
        assert "찾을 수 없습니다" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_bicycle_major_components_query_uses_korean_grounded_answer(self):
        """자전거 주요 구성품 질의는 영어 괄호명을 노출하지 않고 한국어로 답한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "Item | Refer to | Definition\n"
                "Frame | A bicycle frame gives support to all parts.\n"
                "Wheel | The wheels let the bicycle move.\n"
                "Seat and Seat Post | Supports the rider.\n"
                "Handle Bar | Used to steer the bicycle.\n"
                "Brakes | Used to slow or stop the bicycle.\n"
                "Shifters | Used to change gears.\n"
                "Crank | Holds the pedals.\n"
                "Pedals | The rider pushes the pedals.\n"
                "Chain | Transfers power."
            ),
            metadata={
                "dmc": "S1000DBIKE-AAA-D00-00-00-00AA-041A-A",
                "dm_type": "descriptive",
                "title": "Physical description of a bicycle",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.99)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "제공된 문서 기준으로는 **답변:**\n"
            "자전거의 주요 구성품은 다음과 같습니다:\n"
            "- 프레임 (Frame)\n- 바퀴 (Wheel)\n- 좌석 및 좌석대 (Seat and Seat Post)\n"
            "- 핸들바 (Handle Bar)\n- 브레이크 (Brakes)\n- 시프터 (Shifters)\n"
            "- 크랭크 (Crank)\n- 페달 (Pedals)\n- 체인 (Chain)\n"
            "참고 문서: DBIKE-AAA-D00-00-00-00AA-041A-A"
        )

        result = run_rag_query_sync(
            query="자전거의 주요 구성품은 무엇인가요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "제공된 문서 기준으로 자전거의 주요 구성품은 프레임, 바퀴, 좌석 및 좌석대, "
            "핸들바, 브레이크, 시프터, 크랭크, 페달, 체인입니다.\n"
            "참고 문서: S1000DBIKE-AAA-D00-00-00-00AA-041A-A"
        )
        for english_term in ("Frame", "Wheel", "Seat", "Handle Bar", "Brakes", "Shifters", "Crank", "Pedals", "Chain"):
            assert english_term not in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_manual_test_query_uses_deterministic_grounded_answer(self):
        """브레이크 수동 테스트 질의는 341A 근거가 있으면 LLM 오판 없이 답한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content="Apply the brakes. Make sure the wheel locks and the bicycle stops.",
            metadata={
                "dmc": "BRAKE-AAA-DA1-00-00-00AA-341A-A",
                "dm_type": "procedural",
                "title": "Brake system - Manual test",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        mock_llm = MagicMock()

        result = run_rag_query_sync(
            query="브레이크 수동 테스트 절차를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "자전거를 세운 뒤" in result.answer
        assert "바퀴가 잠기고" in result.answer
        assert "BRAKE-AAA-DA1-00-00-00AA-341A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_cleaning_dmc_lookup_answers_from_evidence(self):
        """DMC 조회 질의는 LLM 생성 대신 evidence DMC를 직접 반환한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content="Clean the brake pads.",
            metadata={
                "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
                "dm_type": "procedural",
                "title": "Brake pads - Clean",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        mock_llm = MagicMock()

        result = run_rag_query_sync(
            query="브레이크 패드 청소 문서의 DMC를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == "브레이크 패드 청소 문서의 DMC는 BRAKE-AAA-DA1-10-00-00AA-251A-A입니다."
        mock_llm.invoke.assert_not_called()

    def test_brake_pad_cleaning_query_uses_rubbing_alcohol_not_oil(self):
        """UI E2E 회귀: 브레이크 패드 청소 절차는 오일이 아니라 rubbing alcohol 근거로 답한다."""
        mock_vs = MagicMock()
        doc = Document(
            page_content=(
                "Do a visual inspection of the brakes as given in the pre-ride checks. "
                "Clean the brake pads. Find each of the brake pads. "
                "Apply a thin layer of the rubbing alcohol on each of the brake pads using a clean cloth. "
                "Rub the surface until you have applied it to the complete surface of the pad."
            ),
            metadata={
                "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
                "dm_type": "procedural",
                "title": "Brake pads - Clean with rubbing alcohol",
            },
        )
        mock_vs.similarity_search_with_relevance_scores.return_value = [(doc, 0.9)]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "브레이크 패드를 청소하려면 시각 검사 후 각 브레이크 패드를 찾고, "
            "얇은 층의 오일을 패드 전체 표면에 적용합니다.\n"
            "참고 문서: BRAKE-AAA-DA1-10-00-00AA-251A-A"
        )

        result = run_rag_query_sync(
            query="브레이크 패드 청소 절차를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "오일" not in result.answer
        assert "알코올" in result.answer
        assert "깨끗한 천" in result.answer
        assert "BRAKE-AAA-DA1-10-00-00AA-251A-A" in result.answer
        mock_llm.invoke.assert_not_called()

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

    def test_llm_output_strips_prompt_placeholders_and_context_copy(self):
        """8B 모델이 출력 형식 예시/컨텍스트를 복사해도 최종 답변만 남긴다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(1)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = """
<한국어 답변>
참고 문서: <DMC 목록 또는 없음>

브레이크 패드는 바퀴 림에 마찰을 만들어 속도를 줄입니다.
참고 문서: DMC-000

---

[DMC: DMC-000 | Type: descriptive]
The brake system raw context should not leak.
Alright, let's reason about the answer.
"""

        result = run_rag_query_sync(
            query="브레이크 패드는 어떤 역할을 하나요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == "브레이크 패드는 바퀴 림에 마찰을 만들어 속도를 줄입니다.\n참고 문서: DMC-000"

    def test_llm_output_keeps_first_repeated_answer_block_from_langsmith_trace(self):
        """LangSmith에서 관측된 답변 블록 반복은 첫 유효 블록만 남긴다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(1)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = """
<한국어 답변>
참고 문서: <DMC 목록 또는 없음>

답변: 브레이크 시스템은 브레이크 레버를 통해 브레이크 케이블을 당겨, 브레이크 패드가 바퀴의 허리 부분에 마찰력을 발생시켜 자전거 속도를 감소시킵니다.
참고 문서: DMC-000
답변: 브레이크 시스템은 브레이크 레버를 통해 브레이크 케이블을 당겨, 브레이케 패드가 바퀴의 허리 부분에 마찰력을 발생시켜 자전거 속도를 감소시킵니다.
참고 문서: DMC-000
답변: 브레이크 시스템은 브레이크 레버를 통해 브레이크 케이블을 당겨, 브레이크 패드가 바퀴의 허리 부분에 마찰력을 발생시켜 자전거 속도를 감소시킵니다.
참고 문서: DMC-000
"""

        result = run_rag_query_sync(
            query="브레이크 시스템 원리에 대해 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 시스템은 브레이크 레버를 통해 브레이크 케이블을 당겨, "
            "브레이크 패드가 바퀴 림에 마찰력을 발생시켜 자전거 속도를 감소시킵니다.\n"
            "참고 문서: DMC-000"
        )
        assert "브레이케" not in result.answer
        assert result.answer.count("브레이크 시스템은") == 1

    def test_llm_output_normalizes_repeated_evidence_lines(self):
        """DMC/근거 반복과 영어 인용은 짧은 근거 줄로 정리한다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = _make_doc_pairs(1)
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = """
브레이크 케이블 장력은 조정 잠금 너트를 사용하여 조정합니다.
참고 문서: DMC DMC-000에서 "Adjuster lock nut holds the brake cable."라고 명시되어 있습니다.
브레이크 케이블 장력은 조정 잠금 너트를 사용하여 조정합니다.
참고 문서: DMC DMC-000에서 "Adjuster lock nut holds the brake cable."라고 명시되어 있습니다.
DMC: DMC-000 | Type: descriptive
DMC: DMC-000
"""

        result = run_rag_query_sync(
            query="브레이크 케이블 장력은 무엇으로 조정하나요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == "브레이크 케이블 장력은 조정 잠금 너트를 사용하여 조정합니다.\n참고 문서: DMC-000"

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

    def test_brake_cable_adjustment_location_query_uses_korean_grounded_answer(self):
        """QA 193: 브레이크 케이블 장력 조정 위치 질의는 영어 figure/context leak 없이 답한다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "The brake system has these primary components: the brake lever, "
                        "the brake cable, the brake arm, the brake clamp and the brake pads. "
                        "The adjuster lock nut holds the brake cable."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                        "chunk_id": "brake-cable-adjustment",
                        "dm_type": "descriptive",
                        "title": "Brake system",
                    },
                ),
                0.4221,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "DMC 문서에서 브레이크 케이블 장력 조정에 대한 설명은 "
            "[Figure: Typical components of a mountain bicycle lever] 에서 확인할 수 있습니다.\n\n"
            "DMC 문서에서 브레이크 케이블 장\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )

        result = run_rag_query_sync(
            query="브레이크 케이블 장력 조정 설명은 어디에 나오나요?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert result.answer == (
            "브레이크 케이블 장력 조정 관련 설명은 브레이크 시스템 설명 문서에 나옵니다. "
            "해당 문서는 브레이크 케이블을 브레이크 시스템 구성품으로 설명합니다.\n"
            "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"
        )
        assert "[Figure:" not in result.answer
        assert "Typical components" not in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_arm_description_query_uses_deterministic_guard(self):
        """QA 110: 브레이크 암 설명은 LLM decode 오류 없이 근거 기반으로 응답한다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "The brake system has these primary components: the brake lever, "
                        "the brake cable, the brake arm, the brake clamp and the brake pads."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                        "chunk_id": "brake-arm-components",
                        "dm_type": "descriptive",
                        "title": "Brake system",
                    },
                ),
                0.9,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("llama_decode returned 1")

        result = run_rag_query_sync(
            query="브레이크 암에 대해 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 암" in result.answer
        assert "브레이크 케이블" in result.answer
        assert "브레이크 패드" in result.answer
        assert "BRAKE-AAA-DA1-00-00-00AA-041A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_procedure_question_without_matching_procedure_does_not_call_llm(self):
        """절차 질문은 작업/대상이 맞는 절차 문서가 없으면 생성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Put the bicycle in a vertical position. Hold the handle bars and apply the brakes.",
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-341A-A",
                        "chunk_id": "manual-test",
                        "dm_type": "procedural",
                        "title": "Brake system - manual test",
                    },
                ),
                0.8,
            ),
            (
                Document(
                    page_content="The brake cable transfers force from the brake lever to the brake pads.",
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                        "chunk_id": "brake-cable-desc",
                        "dm_type": "descriptive",
                        "title": "Brake cable",
                    },
                ),
                0.7,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "환각 절차"

        result = run_rag_query_sync(
            query="브레이크 케이블 교체 방법은?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 케이블 교체 절차" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "BRAKE-AAA-DA1-00-00-00AA-041A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_arm_installation_query_is_rejected_without_specific_procedure(self):
        """QA 056: 브레이크 암 장착은 브레이크/포크 절차 후보만으로 생성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "It is necessary to install the fork before installing the brakes\n"
                        "Hold the front of the bicycle.\n"
                        "Install the front brakes on the fork.\n"
                        "Put the frame on the floor."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA1-20-00-00AA-720A-A",
                        "chunk_id": "fork-install-mentions-brakes",
                        "dm_type": "procedural",
                        "title": "Front fork - Install procedures",
                    },
                ),
                0.4399,
            ),
            (
                Document(
                    page_content=(
                        "Put the bicycle in a vertical position.\n"
                        "Hold the handle bars and push the bicycle forwards.\n"
                        "Apply the brakes.\n"
                        "Make sure that the wheels lock and the bicycle stops."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-341A-A",
                        "chunk_id": "brake-manual-test",
                        "dm_type": "procedural",
                        "title": "Brake system - Manual test",
                    },
                ),
                0.394,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "브레이크 암 장착 환각 답변"

        result = run_rag_query_sync(
            query="브레이크 암 장착 방법은?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 암" in result.answer
        assert "장착" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "S1000DBIKE-AAA-DA1-20-00-00AA-720A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_cable_change_order_query_is_rejected_without_procedure(self):
        """QA 043: '바꾸는 순서' 표현도 비지원 절차로 보고 생성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "The brake system has these primary components: the brake lever, "
                        "the brake cable, the brake arm, the brake clamp and the brake pads."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A",
                        "chunk_id": "brake-cable-components",
                        "dm_type": "descriptive",
                        "title": "Brake system",
                    },
                ),
                0.4251,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "브레이크 케이블 교체 절차 환각 답변"

        result = run_rag_query_sync(
            query="브레이크 케이블을 새 부품으로 바꾸는 순서를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 케이블 교체 절차" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "BRAKE-AAA-DA1-00-00-00AA-041A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_rear_wheel_body_mentions_do_not_bypass_unsupported_procedure_guard(self):
        """QA 루프의 비지원 휠 절차는 후보 절차가 검색되어도 생성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "Remove the rear wheel. Make sure that there is no air in the tube. "
                        "Loosen the cap on the valve stem."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-D00-00-00-00AA-663A-A",
                        "chunk_id": "standard-repair-rear-wheel-prereq",
                        "dm_type": "procedural",
                        "title": "Bicycle - Standard repair procedures",
                    },
                ),
                0.93,
            ),
            (
                Document(
                    page_content=(
                        "Hold the rear of the bicycle. Push the wheel forwards and down "
                        "to disengage the chain from the sprocket."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-20-00-00AA-520A-A",
                        "chunk_id": "rear-wheel-body-only",
                        "dm_type": "procedural",
                        "title": "Rear wheel - Remove procedures",
                    },
                ),
                0.38,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "뒷바퀴 탈거 방법 환각 답변"

        result = run_rag_query_sync(
            query="뒷바퀴 탈거 방법은?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "뒷바퀴" in result.answer
        assert "탈거" in result.answer
        assert "찾을 수 없습니다" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_english_rear_wheel_removal_procedure_query_is_rejected(self):
        """QA 049: 'removal procedure' 영어 표현도 비지원 휠 절차로 보고 생성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Prepare the rear wheel for the removal of the tire",
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-20-00-00AA-412A-A",
                        "chunk_id": "rear-wheel-fault",
                        "dm_type": "fault",
                    },
                ),
                0.6933,
            ),
            (
                Document(
                    page_content="Lift the wheel away from the frame.\nPut the frame on the floor.",
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-30-00-00AA-520A-A",
                        "chunk_id": "wheel-body-only-1",
                        "dm_type": "procedural",
                    },
                ),
                0.4409,
            ),
            (
                Document(
                    page_content=(
                        "Hold the rear of the bicycle.\n"
                        "Push the wheel forwards and down to disengage the chain from the sprocket.\n"
                        "Turn the wheel to the side and lift it away from the frame.\n"
                        "Put the frame on the floor."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-20-00-00AA-520A-A",
                        "chunk_id": "rear-wheel-body-only-2",
                        "dm_type": "procedural",
                    },
                ),
                0.3628,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "뒤 바퀴 탈거 절차 환각 답변"

        result = run_rag_query_sync(
            query="rear wheel removal procedure를 한국어로 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "뒷바퀴" in result.answer
        assert "탈거" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "S1000DBIKE-AAA-DA0-20-00-00AA-412A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_tire_replacement_query_does_not_synthesize_remove_and_install_steps(self):
        """QA 039: 타이어 교체는 검색 후보가 있어도 지원 절차로 합성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "[Prerequisite] The tire is removed.\n"
                        "Remove the old inner-tube.\nInstall the new .\n"
                        "[Close requirement] Replace the tire."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-10-10-00AA-921A-A",
                        "chunk_id": "tire-remove-inner-tube",
                        "dm_type": "procedural",
                        "title": "Tire - Remove procedures",
                    },
                ),
                0.6452,
            ),
            (
                Document(
                    page_content=(
                        "Deflate the tire.\n"
                        "Use the from the and remove the old tire from the wheel.\n"
                        "Use the from the and attach the new to the wheel. Refer to\n"
                        "Inflate the tire (refer to ).\nInstall the wheel."
                    ),
                    metadata={
                        "dmc": "S1000DBIKE-AAA-DA0-10-20-00AA-921A-A",
                        "chunk_id": "tire-install-degraded-refs",
                        "dm_type": "procedural",
                        "title": "Tire - Install procedures",
                    },
                ),
                0.4724,
            ),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "타이어 교체 절차 환각 답변"

        result = run_rag_query_sync(
            query="타이어 교체 절차를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "타이어" in result.answer
        assert "교체" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "S1000DBIKE-AAA-DA0-10-10-00AA-921A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_pad_cleaning_vs_manual_test_does_not_invent_missing_side(self):
        """QA 196: 비교 질의에서 한쪽 근거만 있으면 다른 절차를 합성하지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "Do a visual inspection of the brakes as given in the pre-ride checks.\n"
                        "Clean the brake pads.\n"
                        "Apply a thin layer on each of the brake pads.\n"
                        "Rub the surface until you have applied it to the complete surface of the pad."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
                        "chunk_id": "brake-pad-cleaning",
                        "dm_type": "procedural",
                        "title": "Brake pads - Clean",
                    },
                ),
                0.507,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = (
            "DMC 관련 내용을 참고하여 답변을 작성하세요.\n"
            "브레이크 패드 청소는 문서 근거가 있고, 수동 테스트는 브레이크를 작동해 반응 속도와 힘을 점검합니다.\n"
            "참고 문서: DBIKE-AAA-DA1-10-00-00AA-251A-A"
        )

        result = run_rag_query_sync(
            query="브레이크 패드 청소와 수동 테스트를 비교해줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 패드 청소" in result.answer
        assert "수동 테스트" in result.answer
        assert "찾을 수 없습니다" in result.answer
        assert "반응 속도" not in result.answer
        assert "DMC 관련 내용을 참고" not in result.answer
        assert "BRAKE-AAA-DA1-10-00-00AA-251A-A" in result.answer
        mock_llm.invoke.assert_not_called()

    def test_brake_pad_cleaning_and_manual_test_combined_query_returns_both_sides(self):
        """복합 질의에서는 청소 guard가 수동 테스트 근거를 가로막지 않는다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content=(
                        "Clean the brake pads. Apply a thin layer of rubbing alcohol on each of the brake pads "
                        "using a clean cloth. Rub the surface until complete."
                    ),
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
                        "chunk_id": "brake-pad-cleaning-alcohol",
                        "dm_type": "procedural",
                        "title": "Brake pads - Clean with rubbing alcohol",
                    },
                ),
                0.9,
            ),
            (
                Document(
                    page_content="Apply the brakes. Make sure the wheel locks and the bicycle stops.",
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-00-00-00AA-341A-A",
                        "chunk_id": "brake-manual-test",
                        "dm_type": "procedural",
                        "title": "Brake system - Manual test",
                    },
                ),
                0.88,
            ),
        ]
        mock_llm = MagicMock()

        result = run_rag_query_sync(
            query="브레이크 패드 청소 및 수동 테스트 절차를 알려줘",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "브레이크 패드 청소" in result.answer
        assert "브레이크 수동 테스트" in result.answer
        assert "BRAKE-AAA-DA1-10-00-00AA-251A-A" in result.answer
        assert "BRAKE-AAA-DA1-00-00-00AA-341A-A" in result.answer
        assert {ev.dmc for ev in result.evidences} == {
            "BRAKE-AAA-DA1-10-00-00AA-251A-A",
            "BRAKE-AAA-DA1-00-00-00AA-341A-A",
        }
        mock_llm.invoke.assert_not_called()

    def test_matching_procedure_question_still_calls_llm(self):
        """작업/대상이 맞는 절차 문서는 정상 생성으로 넘긴다."""
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [
            (
                Document(
                    page_content="Clean the brake pads with a clean cloth. Inspect the brake pads after cleaning.",
                    metadata={
                        "dmc": "BRAKE-AAA-DA1-10-00-00AA-251A-A",
                        "chunk_id": "brake-pad-clean",
                        "dm_type": "procedural",
                        "title": "Brake pads - clean",
                    },
                ),
                0.9,
            )
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "브레이크 패드 청소 절차 답변"

        result = run_rag_query_sync(
            query="브레이크 패드 청소 방법은?",
            vectorstore=mock_vs,
            llm=mock_llm,
            options=RagOptions(rerank=RerankOptions(enabled=False), expand_query=False),
        )

        assert "청소 절차" in result.answer
        mock_llm.invoke.assert_called_once()


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

    def test_prompt_forbids_context_copy_and_requires_fixed_korean_format(self):
        prompt = build_prompt("브레이크 케이블 설명", "Brake cable context")

        assert "Context 원문을 그대로 복사하지 마세요" in prompt
        assert "반드시 한국어" in prompt
        assert "답변:" in prompt
        assert "참고 문서:" in prompt
        assert "/no_think" in prompt


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
