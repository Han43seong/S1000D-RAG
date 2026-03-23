"""S1000D RAG 질의 CLI.

사용법:
    python query.py "브레이크 시스템은 어떻게 동작하나요?"
    python query.py --interactive
"""

from __future__ import annotations

import argparse
import sys

from src.chunker.indexer import load_chroma_index
from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.rag.models import get_embeddings, get_llm
from src.rag.pipeline import run_rag_query_sync
from src.types.rag import RagOptions, RerankOptions, SessionMeta


def query_once(
    question: str,
    chroma_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    use_reranker: bool = False,
) -> None:
    """단일 질의 실행 및 결과 출력."""
    # 모델 로드
    print("모델 로딩 중...")
    embedding_fn = get_embeddings()
    llm = get_llm()

    # 벡터스토어 로드
    vectorstore = load_chroma_index(
        embedding_fn=embedding_fn,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )

    # RAG 옵션
    options = RagOptions(
        rerank=RerankOptions(enabled=use_reranker),
    )

    # 실행
    print(f"\nQ: {question}\n")
    result = run_rag_query_sync(
        query=question,
        vectorstore=vectorstore,
        llm=llm,
        options=options,
    )

    # 결과 출력
    print(f"A: {result.answer}\n")

    if result.evidences:
        print("--- Evidence ---")
        for i, ev in enumerate(result.evidences, 1):
            print(f"  [{i}] DMC: {ev.dmc}")
            print(f"      Score: {ev.score:.4f} | Type: {ev.dm_type} | Security: {ev.security}")
        print()


def interactive_mode(
    chroma_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    use_reranker: bool = False,
) -> None:
    """대화형 질의 모드."""
    print("모델 로딩 중...")
    embedding_fn = get_embeddings()
    llm = get_llm()

    vectorstore = load_chroma_index(
        embedding_fn=embedding_fn,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )

    options = RagOptions(
        rerank=RerankOptions(enabled=use_reranker),
    )

    print("\n=== S1000D RAG Interactive Mode ===")
    print("질문을 입력하세요. 종료: 'quit' 또는 'exit'\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("종료합니다.")
            break

        result = run_rag_query_sync(
            query=question,
            vectorstore=vectorstore,
            llm=llm,
            options=options,
        )

        print(f"\nA: {result.answer}\n")
        if result.evidences:
            for i, ev in enumerate(result.evidences, 1):
                print(f"  [{i}] {ev.dmc} (score: {ev.score:.4f})")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="S1000D RAG 질의 CLI")
    parser.add_argument(
        "question",
        nargs="?",
        help="질의 문자열 (없으면 대화형 모드)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_PERSIST_DIR,
        help="ChromaDB 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="리랭커 활성화",
    )

    args = parser.parse_args()

    if args.interactive or args.question is None:
        interactive_mode(
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            use_reranker=args.reranker,
        )
    else:
        query_once(
            question=args.question,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            use_reranker=args.reranker,
        )


if __name__ == "__main__":
    main()
