# S1000D-RAG

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)

S1000D DM XML 기술 매뉴얼을 대상으로 한 **폐쇄망 로컬 LLM RAG 파이프라인**. 한국어 질의에 S1000D 구조(DMC, 절차, 경고, 적용성)를 보존하며 출처 근거 기반 답변을 생성한다.

---

## 목적

S1000D 기술 문서는 DMC, 절차 단계, warning/caution, 도해, 적용성 같은 구조 정보를 담고 있어 일반 벡터 RAG로는 "절차인지 설명인지", "지원되는 작업인지 아닌지"를 구분하기 어렵다. 이 프로젝트는 온톨로지가 **무엇을 말할 수 있는지를 통제**하고, RAG가 **출처 증거를 검색**하며, LLM이 **증거를 한국어로 합성**하는 파이프라인을 구현한다. 인터넷 연결 없이 완전히 온프레미스로 동작한다.

답변을 그럴듯하게 생성하는 것보다, 어떤 문서를 근거로 답했는지와 근거가 부족한 질문을 구분할 수 있는지를 중점적으로 확인했다.

---

## 원리 / 동작 방식

```
S1000D DM XML
  → dm_parser.py  (lxml, descriptive / procedure / generic 타입별 파싱)
  → chunker.py    (ContentBlock 슬라이딩 윈도우, warning/caution 독립 청크)
  → ChromaDB      (BAAI/bge-m3 임베딩, chroma_db_full/)
                                            ↑ ingest.py

질의
  → parse_query          (intent / target / action 추출)
  → resolve_ontology     (RDF/OWL 온톨로지 매니페스트 → DMC 매핑)
  → plan_evidence        (primary/related/warning/figure 플랜)
  → retrieve_evidence    (Graph-first → Vector fallback → bge-reranker-v2-m3 리랭킹)
  → build_answer_plan    (AnswerPlan: claims + evidence + 금지 주장)
  → verbalize_answer_plan (로컬 LLM GGUF → 한국어 설명)
  → 품질 검증            (DMC 그라운딩 · 지원 수준 · 안전 경고 보존; v4는 support_level + grounding fallback)
  → RagResult(answer, evidences, reference_materials)
                                            ↑ pipeline_v4.py
```

파이프라인은 v1(순수 벡터 RAG) → v2(부분 온톨로지 힌트) → v3(결정론적 온톨로지 우선) → **v4(RDF/OWL Graph RAG + LLM 합성)** 순으로 진화했다. `src/rag/pipeline.py`(v3 baseline)와 `src/rag/pipeline_v4.py`(v4 target) 모두 유지된다.

---

## 주요 기능

| 모듈 | 기능 |
|---|---|
| `ingest.py` | DM XML 디렉터리 스캔 → 파싱 → ChromaDB 인덱싱 CLI |
| `query.py` | 단일·대화형 RAG 질의 CLI |
| `app_web.py` | FastAPI 백엔드 (WinneAI), Material Design 3 정적 프론트엔드 |
| `app.py` | Streamlit 질의 UI |
| `src/parser/dm_parser.py` | S1000D XML → `ContentBlock` 변환 (절차·설명·경고·테이블·도해 참조) |
| `src/chunker/chunker.py` | 슬라이딩 윈도우 청킹, warning/caution 별도 분리 |
| `src/rag/pipeline_v4.py` | Ontology-guided Graph RAG v4 (RDF/SPARQL, AnswerPlan, 품질 게이트) |
| `src/vlm/` | VLM(Qwen3-VL) 기반 도해 캡셔닝, 멀티모달 컨텍스트 브리지 |
| `scripts/export_ontology_rdf.py` | 온톨로지 매니페스트 → Turtle / JSON-LD 내보내기 |

---

## 설치 & 사용법

```bash
# 1. 의존성 설치
poetry install
# 또는
uv sync

# 2. 환경 설정 (.env)
cp .env.example .env
# .env에서 S1000D_TEXT_MODEL_PATH, S1000D_DATA_DIR 등 설정

# 3. XML 인덱싱
python ingest.py                  # S1000D_DATA_DIR 하위 DM XML 전체 인덱싱
python ingest.py --data-dir /path/to/xmls --limit 50

# 4. CLI 질의
python query.py "브레이크 패드 교체 절차 알려줘"
python query.py                   # 대화형 모드

# 5. 웹 서버
uvicorn app_web:app --host 0.0.0.0 --port 8000 --reload
# http://localhost:8000

# 6. Streamlit UI
streamlit run app.py
```

---

## 요구사항 / 의존성

- **Python** ≥ 3.11, < 3.13
- **로컬 GGUF 모델** — `models/` 디렉토리에 배치, `.env`의 `S1000D_TEXT_MODEL_PATH` 지정
  - 기본 프로파일: `qwen36_27b_iq4` (16 GB VRAM), 경량: `qwen3_8b_q5`
- **VLM** (선택) — `S1000D_VLM_MODEL_PATH` + `S1000D_VLM_MMPROJ_PATH`
- 주요 패키지: `llama-cpp-python`, `sentence-transformers` (bge-m3), `chromadb`, `langchain`, `lxml`, `rdflib`, `fastapi`

---

## 주요 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-03-13 | 프로젝트 초기 설정 — S1000D RAG 파이프라인 스캐폴딩 |
| 2026-03-23 | 핵심 RAG 파이프라인 구현 (Phase 3–8) + CLI 도구(`ingest`, `query`, `export_index`) + 테스트 스위트 + FastAPI · Streamlit UI |
| 2026-06-01 | 멀티모달 지원 추가 — VLM 캡셔닝, 비주얼 에셋 매니페스트, 웹 채팅 잡 진행 상태 |
| 2026-06-03 | S1000D 온톨로지 RDF 내보내기, 그래프 기반 검색 QA 루프, 온톨로지 인식 QA 강화 |
| 2026-06-04 | v4 RDF/OWL 온톨로지 레이어 및 Graph RAG 런타임 구현 (`pipeline_v4.py`, rdflib 백엔드) |
| 2026-06-05 | v4 한국어 증상 답변 그라운딩 완성 및 검증 (Korean composer, grounded verbalizer) |
