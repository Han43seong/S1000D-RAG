# S1000D-RAG - S1000D DM XML 기반 로컬 LLM RAG 파이프라인

## 프로젝트 개요
S1000D Data Module XML을 파싱하여 구조적 청킹 → 벡터 인덱싱 → LangChain RAG 파이프라인을 구축하는 프로젝트.
향후 LangGraph로 승격 가능한 구조로 설계.

## 기술 스택
- **언어**: Python 3.11
- **LLM**: `Qwen3-14B` (GGUF Q4_K_M, LlamaCpp)
- **Embedding**: `dragonkue/BGE-m3-ko` (HuggingFace)
- **Reranker**: `bge-reranker-v2-m3-ko` (CrossEncoder, 옵션)
- **NLI**: `mDeBERTa-v3-base-nli` (답변 검증용, 향후)
- **XML 파싱**: lxml
- **타입**: Pydantic v2
- **벡터DB**: 추상화 (초기 구현: ChromaDB)
- **파이프라인**: LangChain → LangGraph 승격 구조
- **Python**: 3.11.9 (pyenv)

## 프로젝트 구조
```
D:\S1000D-RAG\
├── src/
│   ├── __init__.py
│   ├── config.py              # 환경변수, 설정 상수
│   ├── types/
│   │   ├── dm.py              # S1000DDmJson, ContentBlock, DmType
│   │   ├── chunk.py           # S1000DChunk
│   │   └── rag.py             # RagResult, Evidence, SessionMeta
│   ├── csdb/
│   │   ├── adapter.py         # CsdbAdapter ABC (데이터 소스 추상화)
│   │   └── local_adapter.py   # 로컬 파일시스템 구현
│   ├── parser/
│   │   ├── dm_parser.py       # lxml 기반 XML → DM JSON 변환
│   │   ├── normalizer.py      # 텍스트/메타 정규화, DM 타입 판별
│   │   └── llm_helpers.py     # (선택) LLM 보조 유틸
│   ├── chunker/
│   │   ├── chunker.py         # content_blocks 기반 청킹 엔진
│   │   └── indexer.py         # 벡터 인덱스 저장 (LangChain Document)
│   ├── rag/
│   │   ├── retriever.py       # 벡터 검색 + 메타 필터 + SNS 2단계 검색
│   │   ├── reranker.py        # CrossEncoder 리랭커 hook (on/off)
│   │   ├── pipeline.py        # run_rag_query() 메인 함수
│   │   ├── query_enhancer.py  # 한→영 쿼리 확장 + SNS 코드 추출
│   │   ├── prompt.py          # 프롬프트 템플릿 (대화 이력 지원)
│   │   └── models.py          # LLM/임베딩/리랭커 싱글턴 관리
│   └── graph/
│       ├── state.py           # LangGraph 상태 정의
│       ├── nodes.py           # 노드 래퍼
│       └── workflow.py        # 그래프 정의
├── tests/
├── test-data/bike-sample/     # S1000D Bike 샘플 DM XML
├── models/                    # 로컬 모델 파일 (gitignore)
├── ingest.py                  # 인제스천 CLI
├── .env                       # 환경변수 (gitignore)
├── pyproject.toml
└── docker-compose.yml         # (향후) pgvector 등
```

## S1000D 핵심 개념
- **Data Module (DM)**: 최소 단위 문서 (절차/설명/부품/고장 등)
- **DMC (Data Module Code)**: DM 고유 ID
- **PM (Publication Module)**: DM 묶음 → 교범 구조
- **CSDB**: DM/PM 등을 저장하는 공용 저장소

## 파이프라인 레이어
```
[CSDB Adapter] → [DM Parser] → [Chunker] → [Indexer] → [VectorStore]
                                                              ↓
[Models (LLM/Embed/Rerank)] → [RAG Pipeline (run_rag_query)] ←┘
                                         ↓
                                 [LangGraph Nodes] (향후)
```

## 구현 진행 상황
- [x] Phase 0: 프로젝트 초기화 (디렉터리, pyproject.toml, 모델)
- [x] Phase 1: Pydantic 타입 시스템 (DM JSON, Chunk, RAG 결과)
- [x] Phase 2: CSDB Adapter (인터페이스 + 로컬 어댑터)
- [x] Phase 3: DM Parser / Normalizer (lxml 기반 XML → DM JSON)
- [x] Phase 4: Chunker & Indexer
- [x] Phase 5: RAG Pipeline (run_rag_query)
- [ ] Phase 6: LangGraph 승격 구조 (보류 - LangChain 구조 우선)
- [x] Phase 7: 테스트 & 검증 (86 tests, 80 DM 파싱 100%)
- [x] Phase 8: RAG 업그레이드 (쿼리 확장, SNS 2단계 검색, 대화 이력, Qwen3-14B)

## 참고
- 기존 프로젝트: `D:\RAG-proj` (Python PDF 기반, 레퍼런스용)
- 계획 문서: Obsidian `01-Projects/Plans/2026-03-13-s1000d-rag-pipeline-rebuild.md`
- 가상환경: `D:\S1000D-RAG\.venv`
- Python 경로: `C:\Users\hskim\.pyenv\pyenv-win\versions\3.11.9`
