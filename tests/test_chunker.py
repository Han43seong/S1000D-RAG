"""Phase 4 테스트: Chunker & Indexer.

실제 Bike 샘플을 파싱한 결과를 청킹하여 검증.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import S1000D_DATA_DIR
from src.chunker.chunker import ChunkingOptions, chunk_dm, _sliding_window_chunk
from src.chunker.indexer import chunks_to_documents
from src.parser.dm_parser import parse_dm_xml
from src.types.dm import ContentBlock, ContentBlockRole, DmType, S1000DDmJson

SAMPLE_DIR = S1000D_DATA_DIR
DESC_DM_FILE = SAMPLE_DIR / "DMC-BRAKE-AAA-DA1-00-00-00AA-041A-A_004-00_EN-US.XML"
PROC_SIMPLE_FILE = SAMPLE_DIR / "DMC-BRAKE-AAA-DA1-00-00-00AA-341A-A_004-00_EN-US.XML"
PROC_COMPLEX_FILE = SAMPLE_DIR / "DMC-S1000DBIKE-AAA-D00-00-00-00AA-258A-A_011-00_EN-US.XML"


def _skip_if_no_file(path: Path):
    if not path.exists():
        pytest.skip(f"Sample file not found: {path.name}")


# ═══════════════════════════════════════════════════════════════════════
# 슬라이딩 윈도우 단위 테스트
# ═══════════════════════════════════════════════════════════════════════


def _make_blocks(n: int, text_len: int = 100) -> list[ContentBlock]:
    """테스트용 ContentBlock 리스트 생성."""
    return [
        ContentBlock(
            id=f"para-{i+1}",
            role=ContentBlockRole.PARA,
            text="x" * text_len,
            structure_path=f"test/para[{i+1}]",
        )
        for i in range(n)
    ]


class TestSlidingWindow:
    def test_exact_fit(self):
        """블록 수가 block_count의 배수일 때."""
        blocks = _make_blocks(6, text_len=50)
        result = _sliding_window_chunk(blocks, block_count=3, max_chars=9999, overlap=0)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3

    def test_with_overlap(self):
        """overlap=1 동작 확인."""
        blocks = _make_blocks(5, text_len=50)
        result = _sliding_window_chunk(blocks, block_count=3, max_chars=9999, overlap=1)
        # window1: [0,1,2], step=2 → window2: [2,3,4]
        assert len(result) == 2
        # 겹침 블록 확인
        assert result[0][-1].id == result[1][0].id

    def test_max_chars_shrinks_window(self):
        """max_chars 제한으로 윈도우가 축소."""
        blocks = _make_blocks(6, text_len=400)
        # block_count=3이면 1200자 → max_chars=900이면 2개로 축소
        result = _sliding_window_chunk(blocks, block_count=3, max_chars=900, overlap=0)
        for window in result:
            total = sum(len(b.text) for b in window)
            # 단일 블록은 허용하므로 2개 이하
            assert len(window) <= 3
            if len(window) > 1:
                assert total <= 900

    def test_single_large_block(self):
        """단일 블록이 max_chars 초과해도 포함."""
        blocks = _make_blocks(1, text_len=2000)
        result = _sliding_window_chunk(blocks, block_count=3, max_chars=500, overlap=0)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_empty_input(self):
        result = _sliding_window_chunk([], block_count=3, max_chars=1000, overlap=0)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# chunk_dm 통합 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestChunkDm:
    def test_chunk_simple_procedural(self):
        """Simple procedural DM (4 steps) 청킹."""
        _skip_if_no_file(PROC_SIMPLE_FILE)
        dm = parse_dm_xml(PROC_SIMPLE_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)

        assert len(chunks) >= 1
        for c in chunks:
            assert c.dmc == dm.dmc
            assert c.dm_type == DmType.PROCEDURAL
            assert c.chunk_id.startswith(dm.dmc)
            assert c.text  # 비어있지 않음
            assert c.security == "01"

    def test_chunk_ids_unique(self):
        """청크 ID가 모두 고유."""
        _skip_if_no_file(PROC_SIMPLE_FILE)
        dm = parse_dm_xml(PROC_SIMPLE_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_safety_separation(self):
        """warning/caution이 독립 청크로 분리."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        dm = parse_dm_xml(PROC_COMPLEX_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm, ChunkingOptions(separate_safety=True))

        # 안전 관련 블록이 독립 청크로 존재해야 함
        safety_chunks = [
            c for c in chunks
            if "warning" in c.metadata.get("role_distribution", {})
            or "caution" in c.metadata.get("role_distribution", {})
        ]
        assert len(safety_chunks) >= 1

        # 각 safety 청크는 블록 1개
        for sc in safety_chunks:
            assert sc.metadata["block_count"] == 1

    def test_no_safety_separation(self):
        """separate_safety=False이면 분리하지 않음."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        dm = parse_dm_xml(PROC_COMPLEX_FILE.read_text(encoding="utf-8"))

        chunks_sep = chunk_dm(dm, ChunkingOptions(separate_safety=True))
        chunks_no_sep = chunk_dm(dm, ChunkingOptions(separate_safety=False))

        # 분리하지 않으면 청크 수가 더 적음 (안전 블록이 일반 윈도우에 합쳐짐)
        assert len(chunks_no_sep) <= len(chunks_sep)

    def test_structure_path_range(self):
        """structure_path_range가 정상 생성."""
        _skip_if_no_file(DESC_DM_FILE)
        dm = parse_dm_xml(DESC_DM_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm, ChunkingOptions(block_count=3, overlap=0))

        for c in chunks:
            assert c.structure_path_range  # 비어있지 않음

    def test_custom_options(self):
        """커스텀 청킹 옵션 적용."""
        _skip_if_no_file(DESC_DM_FILE)
        dm = parse_dm_xml(DESC_DM_FILE.read_text(encoding="utf-8"))

        # 블록 2개씩, overlap 없이
        chunks_2 = chunk_dm(dm, ChunkingOptions(block_count=2, overlap=0, separate_safety=False))
        # 블록 10개씩
        chunks_10 = chunk_dm(dm, ChunkingOptions(block_count=10, overlap=0, separate_safety=False))

        assert len(chunks_2) >= len(chunks_10)

    def test_applicability_dict_to_str(self):
        """dict applicability가 문자열로 변환."""
        _skip_if_no_file(DESC_DM_FILE)
        dm = parse_dm_xml(DESC_DM_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)

        for c in chunks:
            assert isinstance(c.applicability, str)


# ═══════════════════════════════════════════════════════════════════════
# Indexer (Document 변환) 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestChunksToDocuments:
    def test_basic_conversion(self):
        """S1000DChunk → Document 변환."""
        _skip_if_no_file(PROC_SIMPLE_FILE)
        dm = parse_dm_xml(PROC_SIMPLE_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)
        docs = chunks_to_documents(chunks)

        assert len(docs) == len(chunks)
        for doc, chunk in zip(docs, chunks):
            assert doc.page_content == chunk.text
            assert doc.metadata["dmc"] == chunk.dmc
            assert doc.metadata["chunk_id"] == chunk.chunk_id
            assert doc.metadata["dm_type"] == chunk.dm_type.value
            assert doc.metadata["security"] == chunk.security

    def test_metadata_fields(self):
        """Document metadata에 필수 필드 포함."""
        _skip_if_no_file(PROC_SIMPLE_FILE)
        dm = parse_dm_xml(PROC_SIMPLE_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)
        docs = chunks_to_documents(chunks)

        required_keys = {"dmc", "chunk_id", "dm_type", "security", "applicability",
                         "structure_path_range", "title", "issue", "language"}
        for doc in docs:
            assert required_keys.issubset(doc.metadata.keys())

    def test_no_dict_values_in_metadata(self):
        """ChromaDB 호환: metadata 값에 dict/list가 없어야 함."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        dm = parse_dm_xml(PROC_COMPLEX_FILE.read_text(encoding="utf-8"))
        chunks = chunk_dm(dm)
        docs = chunks_to_documents(chunks)

        for doc in docs:
            for k, v in doc.metadata.items():
                assert not isinstance(v, (dict, list)), f"metadata[{k}] is {type(v)}"
