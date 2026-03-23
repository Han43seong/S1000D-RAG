# S1000D RAG Pipeline — 아키텍처 문서

## 목차

1. [개요](#1-개요)
2. [S1000D 도메인 이해](#2-s1000d-도메인-이해)
3. [타입 시스템 (Pydantic 모델)](#3-타입-시스템)
4. [파이프라인 전체 흐름](#4-파이프라인-전체-흐름)
5. [레이어별 상세 설계](#5-레이어별-상세-설계)
   - 5.1 CSDB Adapter
   - 5.2 DM Parser
   - 5.3 Normalizer
   - 5.4 Chunker
   - 5.5 Indexer
   - 5.6 Retriever
   - 5.7 Reranker
   - 5.8 RAG Pipeline (Chain)
   - 5.9 Models (싱글턴)
6. [왜 이렇게 설계했는가 — 설계 근거](#6-설계-근거)
7. [모델 선정 근거](#7-모델-선정-근거)
8. [설정값 해설](#8-설정값-해설)
9. [데이터 흐름 다이어그램](#9-데이터-흐름-다이어그램)
10. [향후 확장 포인트](#10-향후-확장-포인트)

---

## 1. 개요

이 프로젝트는 **S1000D 규격의 Data Module(DM) XML**을 파싱하여 구조적으로 청킹하고,
벡터 인덱싱을 거쳐 **로컬 LLM 기반 RAG(Retrieval-Augmented Generation) 질의응답**을 수행하는 파이프라인이다.

핵심 특징:
- S1000D XML 구조를 **도메인 지식 기반**으로 파싱 (단순 텍스트 추출이 아님)
- DM 타입별 파싱 전략 분리 (절차서 / 설명서 / 고장진단 / 부품목록)
- 안전 블록(WARNING/CAUTION) 분리 청킹
- 100% 로컬 실행 (외부 API 호출 없음)
- LangGraph 승격 가능한 함수형 인터페이스

---

## 2. S1000D 도메인 이해

이 파이프라인은 S1000D의 핵심 개념을 코드에 직접 반영한다.

### Data Module (DM)

S1000D의 **최소 문서 단위**. 하나의 DM은 하나의 주제를 다룬다.
예: "브레이크 패드 교체 절차", "조명 시스템 개요", "프레임 부품 목록".

각 DM은 **DMC(Data Module Code)** 라는 고유 식별자를 가진다:

```
DMC 구조:
  modelIdentCode - systemDiffCode - systemCode - subSystemCode
  - subSubSystemCode - assyCode - disassyCode + Variant
  - infoCode + Variant - itemLocationCode

예: S1000DBIKE-AAA-D00-00-00-00AA-041A-A
```

DMC의 `infoCode` 3자리가 DM의 성격을 결정한다:

| infoCode 범위 | DM 타입 | 설명 |
|---------------|---------|------|
| 001, 002, 010, 040~043 | DESCRIPTIVE | 시스템/부품 설명 |
| 100, 121, 130, 200, 300, 500~900 | PROCEDURAL | 정비/수리 절차 |
| 270~272 | FAULT | 고장 진단/보고 |
| 0A3 | IPD | 부품 목록 (Illustrated Parts Data) |
| 051, 052 | CREW | 운용/승무원 정보 |

### CSDB (Common Source DataBase)

DM, PM(Publication Module) 등을 저장하는 **공용 저장소** 개념.
이 프로젝트에서는 `CsdbAdapter` ABC로 추상화하여, 로컬 파일시스템(`LocalCsdbAdapter`)을
현재 구현체로 사용한다. 향후 원격 DB나 API 기반 어댑터로 교체 가능.

### DM의 XML 내부 구조

S1000D DM XML은 크게 두 부분으로 나뉜다:

```xml
<dmodule>
  <identAndStatusSection>    ← 메타데이터 (DMC, 보안, 적용성, 제목 등)
    <dmAddress>
      <dmIdent>
        <dmCode modelIdentCode="S1000DBIKE" systemCode="D00" infoCode="041A" .../>
        <issueInfo issueNumber="004" inWork="00"/>
        <language languageIsoCode="en" countryIsoCode="US"/>
      </dmIdent>
      <dmTitle><techName>Bicycle</techName><infoName>Description</infoName></dmTitle>
    </dmAddress>
    <dmStatus>
      <security securityClassification="01"/>
      <applic>...</applic>
    </dmStatus>
  </identAndStatusSection>

  <content>                  ← 실제 내용
    <description>            ← DM 타입에 따라 다름
      <levelledPara>...</levelledPara>
    </description>
    <!-- 또는 -->
    <procedure>
      <mainProcedure>
        <proceduralStep>...</proceduralStep>
      </mainProcedure>
    </procedure>
  </content>
</dmodule>
```

파서는 이 구조를 **도메인 지식 기반으로** 해석한다:
- `<content>` 하위의 첫 번째 자식 태그로 DM 타입을 판별
- DM 타입별로 다른 파싱 전략을 적용
- 각 텍스트 요소에 **의미적 역할(Role)** 을 부여

---

## 3. 타입 시스템

모든 데이터 구조는 Pydantic v2 모델로 정의되어 있어 유효성 검증과 직렬화가 자동으로 이루어진다.

### ContentBlock — XML에서 추출한 의미 단위

```python
class ContentBlockRole(str, Enum):
    TITLE    = "title"       # 제목/소제목
    STEP     = "step"        # 절차 단계
    NOTE     = "note"        # 참고사항
    WARNING  = "warning"     # 경고 (안전)
    CAUTION  = "caution"     # 주의 (안전)
    PARA     = "para"        # 일반 단락
    TABLE    = "table"       # 표
    FIGURE_REF = "figure_ref" # 그림 참조

class ContentBlock(BaseModel):
    id: str                          # "para-3", "step-5"
    role: ContentBlockRole           # 의미적 역할
    text: str                        # 정규화된 텍스트
    structure_path: Optional[str]    # XML 구조 경로
```

`structure_path`는 원본 XML에서의 위치를 추적한다:
```
"description/levelledPara#LP1/para[2]"
"procedure/mainProcedure/step[3]/para[1]"
"procedure/preliminaryRqmts/warning"
```

### S1000DDmJson — 파싱된 DM 전체

```python
class S1000DDmJson(BaseModel):
    dmc: str                         # DMC 식별자
    dm_type: DmType                  # PROCEDURAL, DESCRIPTIVE, ...
    issue: str                       # 발행 번호 (예: "004-00")
    language: str                    # "en-US"
    security: str                    # 보안 등급
    applicability: str | dict        # 적용성 (기종/형상)
    title: str                       # DM 제목
    meta: dict                       # issue_date, responsible_company, skill_level
    content_blocks: list[ContentBlock]  # 파싱된 콘텐츠 블록들
```

### S1000DChunk — 벡터 인덱싱 단위

```python
class S1000DChunk(BaseModel):
    dmc: str                         # 원본 DMC (역추적용)
    chunk_id: str                    # "DMC-XXX__chunk-001"
    dm_type: DmType                  # 필터링용
    security: str                    # 접근 제어용
    applicability: str               # 적용성
    structure_path_range: str        # 포함된 블록의 경로 범위
    text: str                        # 결합된 텍스트 (벡터화 대상)
    metadata: dict                   # title, issue, language, block_count, role_distribution
```

### RAG I/O 타입

```python
class Evidence(BaseModel):       # 검색된 근거 문서
    dmc: str
    chunk_id: str
    score: float
    dm_type: Optional[DmType]
    security: Optional[str]
    applicability: Optional[str]

class RagResult(BaseModel):      # 최종 응답
    answer: str
    evidences: list[Evidence]

class RagOptions(BaseModel):     # 파이프라인 설정
    top_k: int = 10                  # 벡터 검색 후보 수
    rerank: RerankOptions            # 리랭커 설정
    max_context_chars: int = 10000   # LLM 컨텍스트 최대 길이

class SessionMeta(BaseModel):    # 사용자 세션 (접근 제어)
    security_clearance: Optional[str]  # 보안 등급 → 메타 필터링
```

---

## 4. 파이프라인 전체 흐름

### 인제스천 (XML → 벡터DB)

```
 XML 파일 디렉터리
       │
       ▼
 ┌─────────────────┐
 │  CSDB Adapter    │  DM XML 파일 목록 조회 + 읽기
 │  (LocalCsdb)     │
 └────────┬────────┘
          │ XML 문자열
          ▼
 ┌─────────────────┐
 │  DM Parser       │  XML → S1000DDmJson
 │  + Normalizer    │  (DM 타입 판별 + 구조적 파싱)
 └────────┬────────┘
          │ S1000DDmJson (content_blocks)
          ▼
 ┌─────────────────┐
 │  Chunker         │  content_blocks → S1000DChunk 리스트
 │                  │  (안전블록 분리 + 슬라이딩 윈도우)
 └────────┬────────┘
          │ S1000DChunk[]
          ▼
 ┌─────────────────┐
 │  Indexer         │  Chunk → LangChain Document → ChromaDB
 │  + Embeddings    │  (BGE-m3-ko 임베딩 + 메타데이터 저장)
 └────────┬────────┘
          │
          ▼
     ChromaDB (디스크 영속화)
```

### 질의 (Query → Answer)

```
 사용자 질문
       │
       ▼
 ┌─────────────────┐
 │  Retriever       │  벡터 유사도 검색 (cosine similarity)
 │  + MetaFilter    │  보안/타입 필터링 적용
 └────────┬────────┘
          │ (Document, score)[] — top_k개 후보
          ▼
 ┌─────────────────┐
 │  Reranker        │  CrossEncoder로 의미적 재정렬
 │  (optional)      │  (query, doc) 쌍별 관련도 재측정
 └────────┬────────┘
          │ (Document, score)[] — rerank_top_k개
          ▼
 ┌─────────────────┐
 │  Context Builder │  문서 → "[DMC: X | Type: Y]\n내용" 포맷
 │  + Evidence 생성 │  max_context_chars까지 누적
 └────────┬────────┘
          │ context 문자열 + Evidence[]
          ▼
 ┌─────────────────┐
 │  LLM (LlamaCpp) │  프롬프트: Context + Question → Answer
 │  Konan-LLM-OND  │
 └────────┬────────┘
          │
          ▼
   RagResult(answer, evidences)
```

---

## 5. 레이어별 상세 설계

### 5.1 CSDB Adapter (`src/csdb/`)

**역할**: 데이터 소스를 추상화하여 파서와 분리.

```python
# adapter.py — 추상 인터페이스
class CsdbAdapter(ABC):
    async def list_data_modules(filters: DmFilter | None) -> list[str]
    async def get_data_module_xml(dmc: str) -> str

# local_adapter.py — 로컬 파일시스템 구현
class LocalCsdbAdapter(CsdbAdapter):
    def __init__(self, root_dir: Path)
    # DMC-*.xml 파일을 glob으로 탐색
    # dmc 이름으로 파일 경로를 역산하여 읽기
```

**설계 이유**: S1000D 실무에서 CSDB는 DB, 파일서버, API 등 다양한 형태.
인터페이스를 분리해두면 파서/청커 코드 변경 없이 데이터 소스만 교체 가능.

---

### 5.2 DM Parser (`src/parser/dm_parser.py`)

**역할**: S1000D XML → `S1000DDmJson` 변환. **가장 S1000D 도메인 지식이 집중된 모듈.**

#### DM 타입별 파싱 전략

파서는 `<content>` 하위의 자식 요소를 보고 DM 타입을 판별한 뒤,
**각 타입에 최적화된 파싱 로직**을 적용한다:

**PROCEDURAL (절차서)**
```
<procedure>
  ├── <commonInfo>           → PARA 블록들
  ├── <preliminaryRqmts>     → WARNING/CAUTION 블록들
  ├── <mainProcedure>
  │   ├── <proceduralStep>   → STEP 블록 (재귀적 번호 매김: 1, 1.1, 1.1.2)
  │   │   ├── <para>         → STEP 텍스트
  │   │   ├── <note>         → NOTE 블록
  │   │   ├── <warning>      → WARNING 블록
  │   │   └── <proceduralStep>  → 하위 단계 (재귀)
  │   └── ...
  └── <closeRqmts>           → NOTE 블록들
```

- 재귀적으로 `proceduralStep`을 순회하며 단계 번호를 자동 생성
- 각 단계의 `para` 텍스트에 단계 번호를 접두사로 부여
- structure_path 예: `"procedure/mainProcedure/step[3]/step[1]"`

**DESCRIPTIVE (설명서)**
```
<description>
  └── <levelledPara>          → 계층적 문단
      ├── <title>             → TITLE 블록
      ├── <para>              → PARA 블록
      ├── <table>             → TABLE 블록
      ├── <figure>            → FIGURE_REF 블록
      ├── <note>              → NOTE 블록
      └── <levelledPara>      → 재귀 하위 문단
```

- `levelledPara`를 재귀적으로 순회
- 깊이(depth)를 추적하여 structure_path에 반영
- `@id` 속성이 있으면 경로에 포함 (예: `levelledPara#LP1`)

**FAULT / IPD / CREW / PROCESS (기타)**
- 구조가 다양하므로 generic 파싱으로 처리
- 모든 `para`, `simplePara`, `notePara` 요소를 PARA 블록으로 추출

#### 특수 요소 파싱

- **Table**: 제목 + 파이프(`|`) 구분 행으로 텍스트 변환
- **Figure**: `"[Figure: {title} (ICN: {icn})]"` 텍스트 참조로 변환
- **Warning/Caution**: `"⚠ WARNING: ..."` / `"⚠ CAUTION: ..."` 접두사

---

### 5.3 Normalizer (`src/parser/normalizer.py`)

**역할**: 텍스트 정규화, DMC 문자열 조립, DM 타입 판별.

#### 핵심 함수들

| 함수 | 역할 |
|------|------|
| `detect_dm_type(content_el, info_code)` | 자식 태그 → infoCode 매핑 → 기본값 순으로 타입 결정 |
| `build_dmc_string(dm_code_el)` | 11개 속성을 조합하여 DMC 문자열 생성 |
| `extract_text_content(el)` | 요소 트리에서 텍스트만 재귀 추출 (인라인 마크업 처리) |
| `clean_text(raw)` | 다중 공백 정리, 앞뒤 공백 제거 |
| `BlockIdGenerator` | 역할별 순차 ID 생성기 (`para-1`, `step-3`, `warning-2`) |

#### DM 타입 판별 우선순위

```
1순위: <content> 하위 자식 태그명 직접 확인
       <procedure>      → PROCEDURAL
       <description>    → DESCRIPTIVE
       <faultReporting> → FAULT
       ...

2순위: infoCode 매핑 테이블 (40여 개 코드 매핑)
       "041A" → "041" → DESCRIPTIVE
       "200A" → "200" → PROCEDURAL

3순위: 기본값 → DESCRIPTIVE
```

---

### 5.4 Chunker (`src/chunker/chunker.py`)

**역할**: `S1000DDmJson`의 `content_blocks`를 RAG에 최적화된 `S1000DChunk` 단위로 분할.

#### 청킹 전략

```python
class ChunkingOptions:
    block_count: int = 5      # 청크당 목표 블록 수
    max_chars: int = 1500     # 청크 최대 문자 수
    overlap: int = 1          # 인접 청크 간 중첩 블록 수
    separate_safety: bool = True  # WARNING/CAUTION 분리
```

#### 알고리즘

**1단계: 안전 블록 분리** (`separate_safety=True`)

WARNING과 CAUTION 블록은 별도의 1-블록 청크로 분리한다.

이유:
- 안전 정보는 다른 내용과 섞이면 검색 정밀도가 떨어짐
- "브레이크 정비 시 주의사항" 질의에 정확히 매칭되어야 함
- S1000D에서 WARNING/CAUTION은 독립적 의미 단위

**2단계: 슬라이딩 윈도우** (나머지 블록)

```
블록: [A, B, C, D, E, F, G, H]
block_count=5, overlap=1

윈도우 1: [A, B, C, D, E]     (0~4)
윈도우 2: [E, F, G, H]        (4~7, overlap=1이므로 E 중첩)
```

- 윈도우 크기는 `block_count`가 목표이나, `max_chars`를 초과하면 축소
- 단일 블록이 `max_chars`보다 크면 그대로 유지 (강제 분할 안 함)
- 포인터는 `len(window) - overlap`만큼 전진

**3단계: 청크 생성**

각 윈도우의 블록들을 결합하여 `S1000DChunk` 생성:
- `text`: 블록 텍스트들을 `\n\n`으로 결합
- `chunk_id`: `"{dmc}__chunk-{001}"` 형식
- `structure_path_range`: 첫 블록 경로 ~ 마지막 블록 경로
- `metadata.role_distribution`: `{"para": 3, "step": 2}` (어떤 종류의 내용인지)

#### 예시

```
입력 DM: 8개 블록 [para, para, warning, para, para, caution, para, para]
설정: separate_safety=True, block_count=5, overlap=1

출력:
  Chunk 1: [warning]                    ← 안전 블록 분리
  Chunk 2: [caution]                    ← 안전 블록 분리
  Chunk 3: [para, para, para, para, para]  ← 슬라이딩 윈도우
  Chunk 4: [para, para]                 ← 나머지
```

---

### 5.5 Indexer (`src/chunker/indexer.py`)

**역할**: `S1000DChunk` → LangChain `Document` 변환 + ChromaDB 저장/로드.

#### Document 메타데이터 매핑

```python
Document(
    page_content = chunk.text,          # 벡터화 대상
    metadata = {
        "dmc": "S1000DBIKE-AAA-D00-...",   # 필터링 + 추적
        "chunk_id": "DMC-XXX__chunk-001",  # 고유 식별
        "dm_type": "procedural",           # 필터링
        "security": "01",                  # 접근 제어
        "applicability": "All",            # 적용성 필터
        "structure_path_range": "...",     # 원본 위치 추적
        "title": "Brake - Removal",        # 표시용
        "issue": "004-00",                 # 버전 추적
        "language": "en-US",               # 언어 필터
        "block_count": 5,                  # 청크 크기 정보
        "role_distribution": "{...}",      # JSON 문자열
    }
)
```

ChromaDB는 dict/list 값을 지원하지 않으므로, `role_distribution` 같은 복합 값은
JSON 문자열로 변환하여 저장한다.

#### ChromaDB 인터페이스

```python
# 인덱스 생성 (인제스천 시)
build_chroma_index(documents, embedding_fn) → VectorStore

# 인덱스 로드 (질의 시)
load_chroma_index(embedding_fn) → VectorStore
```

LangChain의 `Chroma` 래퍼를 사용하므로, 향후 FAISS, Pinecone 등으로 교체 시
이 인터페이스만 변경하면 된다.

---

### 5.6 Retriever (`src/rag/retriever.py`)

**역할**: 벡터 유사도 검색 + 메타데이터 기반 필터링.

```python
def retrieve(vectorstore, query, top_k, meta_filter) -> list[tuple[Document, float]]
```

#### 메타데이터 필터링

`SessionMeta.security_clearance`가 설정되면, 해당 보안 등급의 문서만 검색:

```python
class MetaFilter:
    security: Optional[str]    # 보안 등급 필터
    dm_type: Optional[str]     # DM 타입 필터
    dmc: Optional[str]         # 특정 DM 필터

# ChromaDB where 절 변환:
# 단일 조건: {"security": "01"}
# 복합 조건: {"$and": [{"security": "01"}, {"dm_type": "procedural"}]}
```

#### 검색 방식

LangChain의 `similarity_search_with_relevance_scores()` 사용:
- 임베딩 모델(BGE-m3-ko)로 query를 벡터화
- ChromaDB에서 cosine similarity 기반 top_k개 후보 반환
- (Document, score) 쌍의 리스트 반환

---

### 5.7 Reranker (`src/rag/reranker.py`)

**역할**: 벡터 검색 결과를 CrossEncoder로 의미적으로 재정렬.

```python
def rerank(query, doc_score_pairs, options, cross_encoder) -> list[tuple[Document, float]]
```

#### 왜 Reranker가 필요한가

벡터 검색(Bi-Encoder)은 빠르지만, query와 document를 **독립적으로** 인코딩한다.
CrossEncoder는 (query, document) 쌍을 **함께** 인코딩하여 더 정밀한 관련도를 측정한다.

```
벡터 검색 결과:
  1. score=0.87  "브레이크 오일 교체 절차"
  2. score=0.85  "브레이크 시스템 개요"
  3. score=0.82  "프레임 구조 설명" (← 관련 없지만 점수 높음)

Reranker 결과:
  1. score=0.92  "브레이크 시스템 개요"      (← 질문에 더 적합)
  2. score=0.88  "브레이크 오일 교체 절차"
  3. score=0.45  "프레임 구조 설명"           (← 낮은 점수로 밀림)
```

#### 동작 방식

1. `options.enabled`가 `False`면 원본 순서 그대로 top_k만 잘라서 반환 (패스스루)
2. `True`면 CrossEncoder로 (query, doc.page_content) 쌍 예측
3. 예측 점수로 내림차순 정렬 → top_k개 반환

---

### 5.8 RAG Pipeline — Chain 구성 (`src/rag/pipeline.py`)

**역할**: Retriever → Reranker → Context Builder → LLM을 연결하는 메인 함수.

#### Chain이 LangChain LCEL이 아닌 이유

이 파이프라인은 LangChain의 LCEL(LangChain Expression Language) 체인을 사용하지 않고,
**순수 함수 호출**로 구성했다.

이유:
1. **LangGraph 승격 대비**: 각 단계가 독립 함수이면 그대로 LangGraph 노드로 감쌀 수 있다
2. **제어 흐름 명확성**: 조건부 로직(빈 결과 처리, 리랭커 on/off)이 명시적
3. **디버깅 용이**: 각 단계의 입출력을 직접 확인 가능
4. **테스트 용이**: 함수별 단위 테스트 가능

#### 프롬프트 설계

```
당신은 S1000D 기술 교범 기반 질의응답 어시스턴트입니다.
아래 참고 문서(Context)를 기반으로 질문에 정확하게 답변하세요.
참고 문서에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.

## Context
[DMC: S1000DBIKE-...-041A-A | Type: descriptive]
브레이크 시스템은 유압식으로 동작하며...

---

[DMC: S1000DBIKE-...-200A-A | Type: procedural]
1. 브레이크 레버를 당겨 작동 상태를 확인한다...

## Question
브레이크 시스템은 어떻게 작동하나요?

## Answer
```

핵심 설계:
- Context에 **DMC와 DM 타입**을 헤더로 포함 → LLM이 출처를 구분 가능
- 문서 간 `---` 구분선으로 경계 명확화
- `max_context_chars`로 LLM 컨텍스트 윈도우 초과 방지
- 문서가 없으면 LLM 호출 자체를 건너뛰고 즉시 응답

#### Context 구성 로직 (`_build_context`)

```python
for doc, score in ranked_docs:
    if total_chars + len(chunk_text) > max_context_chars:
        break  # 최대 길이 도달 시 중단

    header = f"[DMC: {dmc} | Type: {dm_type}]"
    context_parts.append(f"{header}\n{chunk_text}")

    evidences.append(Evidence(dmc=..., score=..., ...))

context = "\n\n---\n\n".join(context_parts)
```

---

### 5.9 Models 싱글턴 (`src/rag/models.py`)

**역할**: 무거운 모델(LLM, 임베딩, 리랭커)의 lazy 싱글턴 관리.

```python
_llm_instance = None         # 전역 캐시

def get_llm() -> BaseLLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LlamaCpp(model_path=..., n_ctx=8192, ...)
    return _llm_instance
```

| 모델 | 로딩 시간 (CPU) | 메모리 |
|------|----------------|--------|
| LLM (GGUF Q8_0) | ~30초 | ~8GB |
| Embedding (BGE-m3-ko) | ~10초 | ~1GB |
| Reranker (CrossEncoder) | ~5초 | ~500MB |

Streamlit UI에서는 `@st.cache_resource`로 이 싱글턴을 다시 감싸서
페이지 새로고침 시에도 모델이 재로딩되지 않도록 한다.

---

## 6. 설계 근거

### Q: 왜 일반 텍스트 추출이 아닌 구조적 파싱인가?

S1000D XML은 풍부한 구조 정보를 담고 있다. 단순 텍스트 추출 시 잃어버리는 것들:

| 정보 | 단순 추출 | 구조적 파싱 |
|------|----------|------------|
| DM 타입 | 없음 | `PROCEDURAL`, `DESCRIPTIVE` 등 |
| 단계 번호 | 없음 | `1.1.2` 재귀 추출 |
| 안전 블록 | 본문에 섞임 | 별도 청크로 분리 |
| 메타데이터 | 없음 | 보안, 적용성, DMC |
| 구조 경로 | 없음 | 원본 위치 추적 가능 |
| 표/그림 | 태그 제거됨 | 텍스트 표현으로 변환 |

### Q: 왜 WARNING/CAUTION을 별도 청크로 분리하는가?

1. **검색 정밀도**: "브레이크 정비 주의사항" 질의에 정확히 매칭
2. **안전 우선**: 안전 정보가 긴 절차 텍스트에 묻히면 검색에서 누락될 수 있음
3. **S1000D 도메인 특성**: 안전 블록은 독립적 의미 단위로 취급됨

### Q: 왜 LangChain LCEL 체인 대신 순수 함수 조합인가?

1. **LangGraph 승격 용이**: 각 함수를 노드로 감싸기만 하면 됨
2. **조건부 로직**: 빈 결과 조기 반환, 리랭커 on/off 등이 LCEL보다 명시적
3. **디버깅**: 각 단계 입출력을 직접 로깅/검사 가능
4. **의존성 주입**: LLM, VectorStore, CrossEncoder를 인자로 받아 테스트 시 Mock 주입 가능

### Q: 왜 ChromaDB인가?

| 벡터DB | 장점 | 단점 |
|--------|------|------|
| **ChromaDB** | 설치 간단, 파일 기반 영속화, LangChain 통합 | 대규모 시 성능 |
| FAISS | 빠른 검색 | 메타 필터링 제한 |
| Pinecone | 관리형, 대규모 | 외부 서비스 |
| pgvector | SQL 통합 | 설치 복잡 |

이 프로젝트는 로컬 실행이 목표이고, DM 수가 수백~수천 개 수준이므로
ChromaDB가 가장 적합. `Indexer`가 LangChain VectorStore 인터페이스를 사용하므로
향후 교체 가능.

### Q: 왜 Overlap 청킹인가?

인접 청크 간 1개 블록을 중첩하면:
- 블록 경계에 걸치는 정보를 양쪽 청크에서 모두 검색 가능
- 전체 벡터 수 증가는 미미 (블록당 ~100자 기준 10~15% 증가)

---

## 7. 모델 선정 근거

### LLM: Konan-LLM-OND (GGUF Q8_0)

- **한국어 특화**: 한국어 기술 문서 응답에 최적화
- **8bit 양자화**: 모델 크기 대비 품질 저하 최소 (Q8_0은 거의 무손실)
- **LlamaCpp**: CPU 추론 가능, GPU 없는 환경에서도 동작
- **로컬 실행**: 보안 문서 처리 시 외부 전송 불가 상황 대응

### Embedding: dragonkue/BGE-m3-ko

- **다국어**: 한국어 + 영어 문서 동시 처리 (S1000D는 영어 기반)
- **M3 아키텍처**: Dense + Sparse + Multi-vector 통합
- **한국어 Fine-tuned**: 기본 BGE-m3보다 한국어 성능 향상
- **L2 정규화**: cosine similarity 직접 사용 가능

### Reranker: bge-reranker-v2-m3-ko

- **CrossEncoder**: (query, doc) 쌍을 함께 인코딩하여 정밀 비교
- **BGE 계열 일관성**: 임베딩 모델과 동일 계열로 호환성 좋음
- **한국어 지원**: 다국어 리랭킹

---

## 8. 설정값 해설

```python
# src/config.py

# LLM 파라미터
LLM_N_CTX = 8192              # 컨텍스트 윈도우 (입력+출력 토큰 합)
LLM_MAX_TOKENS = 1024         # 최대 응답 길이
LLM_TEMPERATURE = 0.1         # 낮을수록 결정적 (기술 문서이므로 낮게)
LLM_TOP_P = 0.9               # Nucleus sampling (상위 90% 확률 토큰에서 선택)
LLM_REPEAT_PENALTY = 1.15     # 반복 방지 (기술 문서에서 반복 표현 억제)

# 청킹 파라미터
CHUNK_BLOCK_COUNT = 5          # 청크당 5개 블록 → 평균 ~500자
CHUNK_MAX_SIZE = 1500          # 임베딩 모델 최적 입력 길이 범위
CHUNK_OVERLAP = 1              # 경계 정보 보존, 벡터 수 증가 최소화

# 검색 파라미터
VECTOR_CANDIDATE_K = 10        # 초기 후보 수 (리랭커 입력)
RERANK_TOP_K = 3               # 리랭킹 후 최종 문서 수
RELEVANCE_THRESHOLD = 0.1      # 최소 관련도 (향후 필터링용)
MAX_CONTEXT_CHARS = 10000      # LLM 컨텍스트 최대 길이 (n_ctx 대비 여유)
```

---

## 9. 데이터 흐름 다이어그램

### 인제스천 예시 (DM 1개 기준)

```
DMC-BRAKE-AAA-DA1-00-00-00AA-041A-A.XML
│
├─ parse_dm_xml() ─────────────────────────────────────────┐
│   identAndStatusSection:                                  │
│     dmc = "BRAKE-AAA-DA1-00-00-00AA-041A-A"              │
│     dm_type = DESCRIPTIVE (infoCode "041")                │
│     title = "Brake system - Description"                  │
│     security = "01"                                       │
│   content → <description>:                                │
│     levelledPara → 12개 ContentBlock 생성                 │
│       [TITLE, PARA, PARA, TABLE, NOTE,                    │
│        TITLE, PARA, PARA, WARNING, PARA, PARA, FIGURE_REF]│
│                                                           │
├─ chunk_dm() ─────────────────────────────────────────────┤
│   안전 블록 분리:                                         │
│     Chunk 1: [WARNING]                                    │
│   슬라이딩 윈도우 (block_count=5, overlap=1):             │
│     Chunk 2: [TITLE, PARA, PARA, TABLE, NOTE]            │
│     Chunk 3: [NOTE, TITLE, PARA, PARA, PARA]             │
│     Chunk 4: [PARA, FIGURE_REF]                          │
│                                                           │
├─ chunks_to_documents() ──────────────────────────────────┤
│   4개 LangChain Document 생성                             │
│   각각 page_content + metadata 포함                       │
│                                                           │
└─ build_chroma_index() ───────────────────────────────────┘
    BGE-m3-ko로 4개 텍스트 벡터화 → ChromaDB에 저장
```

### 질의 예시

```
질문: "브레이크 시스템의 구조는?"
│
├─ retrieve() ──────────────────────────────────────────────┐
│   BGE-m3-ko로 질문 벡터화                                 │
│   ChromaDB cosine 검색 → top 10 후보                      │
│     (0.91) BRAKE-...-041A chunk-002  "브레이크 시스템은..." │
│     (0.87) BRAKE-...-341A chunk-001  "브레이크 점검..."    │
│     (0.85) S1000DBIKE-...-041A chunk-003  "자전거 개요..." │
│     ... 7개 더                                            │
│                                                           │
├─ rerank() (enabled=True) ────────────────────────────────┤
│   CrossEncoder로 (질문, 문서) 쌍별 점수 재측정             │
│   top 3 선택:                                             │
│     (0.94) BRAKE-...-041A chunk-002                       │
│     (0.89) BRAKE-...-341A chunk-001                       │
│     (0.72) BRAKE-...-041A chunk-003                       │
│                                                           │
├─ _build_context() ───────────────────────────────────────┤
│   [DMC: BRAKE-...-041A | Type: descriptive]              │
│   브레이크 시스템은 유압식으로 동작하며...                  │
│   ---                                                     │
│   [DMC: BRAKE-...-341A | Type: procedural]               │
│   브레이크 점검 절차: 1. 레버를 당겨...                    │
│                                                           │
└─ LLM.invoke(prompt) ─────────────────────────────────────┘
    → RagResult(answer="브레이크 시스템은...", evidences=[...])
```

---

## 10. 향후 확장 포인트

| 영역 | 현재 | 향후 |
|------|------|------|
| **파이프라인** | 순수 함수 체인 | LangGraph 노드 래핑 |
| **데이터 소스** | 로컬 파일 | 원격 CSDB API 어댑터 |
| **벡터DB** | ChromaDB (파일) | pgvector (Docker) |
| **답변 검증** | 프롬프트 지시만 | mDeBERTa NLI 기반 검증 |
| **질의 확장** | 없음 | Query Rewriting (LLM 기반) |
| **멀티홉** | 단일 검색 | 다단계 검색 (LangGraph) |
| **적용성 필터** | 문자열 매칭 | 구조화된 기종/형상 필터 |
| **청킹** | 블록 카운트 기반 | 의미 경계 기반 (Semantic) |
