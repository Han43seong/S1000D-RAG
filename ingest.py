"""S1000D DM XML 인제스천 CLI.

사용법:
    python ingest.py [XML_DIR] [--chroma-dir DIR] [--collection NAME]

기본값:
    XML_DIR = docs/S1000D Issue 6 Bike Sample Data Set/Bike Data Set for Release number 6 R2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.chunker.chunker import ChunkingOptions, chunk_dm
from src.chunker.indexer import chunks_to_documents, build_chroma_index
from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, PROJECT_ROOT
from src.csdb.adapter import DmFilter
from src.csdb.local_adapter import LocalCsdbAdapter
from src.parser.dm_parser import parse_dm_xml
from src.rag.models import get_embeddings
from src.types.chunk import S1000DChunk

DEFAULT_XML_DIR = PROJECT_ROOT / "docs" / "S1000D Issue 6 Bike Sample Data Set" / "Bike Data Set for Release number 6 R2"


def ingest(
    xml_dir: Path,
    chroma_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    chunk_options: ChunkingOptions | None = None,
    model_ident_code: str | None = None,
) -> int:
    """XML 디렉터리를 스캔하여 인제스천 실행.

    Returns:
        인덱싱된 총 청크 수.
    """
    adapter = LocalCsdbAdapter(xml_dir)
    opts = chunk_options or ChunkingOptions()

    # 1. DM 목록 스캔
    import asyncio
    dm_filter = DmFilter(model_ident_code=model_ident_code) if model_ident_code else None
    dmcs = asyncio.run(adapter.list_data_modules(dm_filter))
    print(f"[1/4] {len(dmcs)}개 DM 파일 발견")

    if not dmcs:
        print("인덱싱할 DM이 없습니다.")
        return 0

    # 2. 파싱 + 청킹
    all_chunks: list[S1000DChunk] = []
    parse_errors: list[str] = []

    for dmc in dmcs:
        try:
            xml_str = asyncio.run(adapter.get_data_module_xml(dmc))
            dm_json = parse_dm_xml(xml_str)
            chunks = chunk_dm(dm_json, opts)
            all_chunks.extend(chunks)
            print(f"  ✓ {dmc} → {len(dm_json.content_blocks)} blocks → {len(chunks)} chunks")
        except Exception as e:
            parse_errors.append(f"{dmc}: {e}")
            print(f"  ✗ {dmc}: {e}")

    print(f"[2/4] 파싱 완료: {len(all_chunks)} chunks ({len(parse_errors)} errors)")

    if not all_chunks:
        print("인덱싱할 청크가 없습니다.")
        return 0

    # 3. Document 변환
    documents = chunks_to_documents(all_chunks)
    print(f"[3/4] {len(documents)}개 Document 변환 완료")

    # 4. 임베딩 + ChromaDB 인덱싱
    print("[4/4] 임베딩 모델 로딩 + ChromaDB 인덱싱...")
    t0 = time.time()
    embedding_fn = get_embeddings()
    vectorstore = build_chroma_index(
        documents=documents,
        embedding_fn=embedding_fn,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )
    elapsed = time.time() - t0
    print(f"  완료! ({elapsed:.1f}s)")

    print(f"\n=== 인제스천 결과 ===")
    print(f"  DM 파일: {len(dmcs)}개")
    print(f"  파싱 성공: {len(dmcs) - len(parse_errors)}개")
    print(f"  총 청크: {len(all_chunks)}개")
    print(f"  ChromaDB: {chroma_dir} / {collection_name}")

    if parse_errors:
        print(f"\n  파싱 실패:")
        for err in parse_errors:
            print(f"    - {err}")

    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S1000D DM XML 인제스천 CLI"
    )
    parser.add_argument(
        "xml_dir",
        nargs="?",
        default=str(DEFAULT_XML_DIR),
        help="DM XML 파일이 있는 디렉터리 경로",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_PERSIST_DIR,
        help="ChromaDB 영속화 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--block-count",
        type=int,
        default=5,
        help="청크당 블록 수 (기본: 5)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="청크 최대 문자 수 (기본: 1500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="청크 오버랩 블록 수 (기본: 1)",
    )
    parser.add_argument(
        "--model-ident",
        type=str,
        default=None,
        help="모델식별코드 필터 (예: S1000DBIKE). 지정 시 해당 코드의 DM만 인제스천",
    )

    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    if not xml_dir.is_dir():
        print(f"Error: 디렉터리를 찾을 수 없습니다: {xml_dir}")
        sys.exit(1)

    chunk_opts = ChunkingOptions(
        block_count=args.block_count,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )

    total = ingest(
        xml_dir=xml_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_options=chunk_opts,
        model_ident_code=args.model_ident,
    )

    sys.exit(0 if total > 0 else 1)


if __name__ == "__main__":
    main()
