"""E2E 통합 테스트: XML 파싱 → 청킹 → Document → ChromaDB → RAG 질의.

실제 모델 로딩 없이 mock 임베딩/LLM으로 전체 파이프라인 검증.
CI 환경에서도 실행 가능.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langchain_core.documents import Document

from src.chunker.chunker import ChunkingOptions, chunk_dm
from src.chunker.indexer import chunks_to_documents
from src.csdb.local_adapter import LocalCsdbAdapter
from src.parser.dm_parser import parse_dm_xml
from src.rag.pipeline import run_rag_query_sync
from src.rag.retriever import MetaFilter, retrieve
from src.types.rag import RagOptions, RerankOptions

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "docs" / "S1000D Issue 6 Bike Sample Data Set" / "Bike Data Set for Release number 6 R2"


def _skip_if_no_samples():
    if not SAMPLE_DIR.exists():
        pytest.skip("Sample data directory not found")


class FakeEmbeddings:
    """테스트용 고정 차원 임베딩. 실제 모델 로딩 없이 ChromaDB 동작 검증."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), 256), 2)]
        while len(vec) < 128:
            vec.append(0.5)
        return vec[:128]


def _ingest_samples(max_dms: int | None = None):
    """샘플 DM을 파싱+청킹하여 Document 리스트 반환."""
    adapter = LocalCsdbAdapter(SAMPLE_DIR)
    dmcs = asyncio.run(adapter.list_data_modules())
    if max_dms:
        dmcs = dmcs[:max_dms]

    all_chunks = []
    for dmc in dmcs:
        xml_str = asyncio.run(adapter.get_data_module_xml(dmc))
        try:
            dm = parse_dm_xml(xml_str)
            all_chunks.extend(chunk_dm(dm))
        except Exception:
            pass
    return chunks_to_documents(all_chunks)


def _build_vectorstore(documents, persist_dir: str):
    """ChromaDB 인메모리 vectorstore 생성."""
    from langchain_chroma import Chroma

    return Chroma.from_documents(
        documents=documents,
        embedding=FakeEmbeddings(),
        persist_directory=persist_dir,
        collection_name="test",
    )


# ═══════════════════════════════════════════════════════════════════════
# 인제스천 파이프라인 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestE2EIngestPipeline:

    def test_full_ingest_pipeline(self):
        """전체 인제스천 파이프라인 (Document 변환까지) 검증."""
        _skip_if_no_samples()

        adapter = LocalCsdbAdapter(SAMPLE_DIR)
        dmcs = asyncio.run(adapter.list_data_modules())
        assert len(dmcs) > 0

        all_chunks = []
        parsed_count = 0

        for dmc in dmcs:
            xml_str = asyncio.run(adapter.get_data_module_xml(dmc))
            try:
                dm_json = parse_dm_xml(xml_str)
                chunks = chunk_dm(dm_json, ChunkingOptions(block_count=3, overlap=1))
                all_chunks.extend(chunks)
                parsed_count += 1
            except Exception:
                pass

        assert parsed_count > 0
        assert len(all_chunks) > 0

        documents = chunks_to_documents(all_chunks)
        assert len(documents) == len(all_chunks)

        for doc in documents:
            assert "dmc" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert doc.page_content

    def test_ingest_to_chroma(self, tmp_path):
        """ChromaDB에 실제로 인덱싱 후 검색."""
        _skip_if_no_samples()

        documents = _ingest_samples(max_dms=5)
        assert len(documents) > 0

        vectorstore = _build_vectorstore(documents, str(tmp_path / "chroma"))

        results = vectorstore.similarity_search_with_relevance_scores("brake system", k=3)
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert "dmc" in doc.metadata

        # 명시적 해제 (Windows 파일 잠금 방지)
        del vectorstore


# ═══════════════════════════════════════════════════════════════════════
# RAG 질의 파이프라인 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestE2EQueryPipeline:

    def test_full_rag_pipeline(self, tmp_path):
        """Parse → Chunk → Index → Query 전체 검증."""
        _skip_if_no_samples()

        documents = _ingest_samples(max_dms=5)
        vectorstore = _build_vectorstore(documents, str(tmp_path / "chroma"))

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "The brake system uses cantilever brakes with pads."

        result = run_rag_query_sync(
            query="How does the brake system work?",
            vectorstore=vectorstore,
            llm=mock_llm,
            options=RagOptions(top_k=3, rerank=RerankOptions(enabled=False)),
        )

        assert result.answer
        assert "brake" in result.answer.lower()
        assert len(result.evidences) > 0

        prompt = mock_llm.invoke.call_args[0][0]
        assert "Context" in prompt
        assert "DMC:" in prompt

        del vectorstore

    def test_meta_filter_query(self, tmp_path):
        """메타 필터(security) 적용된 검색."""
        _skip_if_no_samples()

        documents = _ingest_samples(max_dms=5)
        vectorstore = _build_vectorstore(documents, str(tmp_path / "chroma"))

        results = retrieve(
            vectorstore=vectorstore,
            query="brake",
            top_k=5,
            meta_filter=MetaFilter(security="01"),
        )

        assert len(results) > 0
        for doc, score in results:
            assert doc.metadata["security"] == "01"

        del vectorstore


# ═══════════════════════════════════════════════════════════════════════
# DM 커버리지 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestE2EDmCoverage:

    def test_all_sample_dms(self):
        """모든 샘플 DM을 파싱하여 성공률 확인."""
        _skip_if_no_samples()

        adapter = LocalCsdbAdapter(SAMPLE_DIR)
        dmcs = asyncio.run(adapter.list_data_modules())

        success_list = []
        error_list = []
        type_counts: dict[str, int] = {}

        for dmc in dmcs:
            xml_str = asyncio.run(adapter.get_data_module_xml(dmc))
            try:
                dm = parse_dm_xml(xml_str)
                success_list.append(dmc)
                t = dm.dm_type.value
                type_counts[t] = type_counts.get(t, 0) + 1
                assert dm.dmc
                assert dm.title
                assert dm.language
                assert dm.security
            except Exception as e:
                error_list.append((dmc, str(e)))

        total = len(dmcs)
        success = len(success_list)
        print(f"\n=== DM Coverage ===")
        print(f"  Total: {total}, Success: {success}, Error: {total - success}")
        print(f"  Type distribution: {type_counts}")

        if error_list:
            print(f"  Errors:")
            for dmc, err in error_list:
                print(f"    - {dmc}: {err}")

        assert success / total >= 0.8, f"파싱 성공률이 80% 미만: {success}/{total}"
