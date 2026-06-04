# Ontology-Guided Graph RAG v4 Implementation Plan

> **For Hermes:** Use subagent-driven-development or a producer/reviewer loop to implement this plan task-by-task. This is a v4 target architecture, not a patch to the v1/v2/v3 pipelines.

**Goal:** Build the final S1000D-RAG architecture as an ontology-guided Graph RAG + LLM synthesis chatbot that uses S1000D structure to control retrieval, answer planning, LLM generation, and quality verification.

**Architecture:** Replace the current deterministic ontology-first v3 runtime with a graph-centered pipeline. The ontology/knowledge graph parses the question, resolves targets/actions/document relationships, plans evidence, retrieves exact source material, builds a structured answer plan, asks the LLM to verbalize only that grounded plan, and gates the final answer before UI delivery.

**Tech Stack:** Python, FastAPI, Chroma/vector retrieval, optional BM25/keyword retrieval, S1000D XML metadata, ontology graph models, local GGUF LLM via llama.cpp, BGE embeddings/reranker, LangSmith traces, pytest regression gates.

---

## 1. Target architecture

```text
User Question
  ↓
Query Understanding
  - intent
  - target/component/system
  - action
  - detail_level
  - audience
  - answer_mode
  - follow-up references
  ↓
Ontology / Knowledge Graph Resolver
  - exact DMC candidates
  - partial/related DMCs
  - component/system relationships
  - procedure/description/fault/figure/reference links
  - support level
  ↓
Evidence Planner
  - primary evidence
  - related evidence
  - warnings/cautions
  - figures/captions
  - tools/supplies/internalRef expansions
  - retrieval strategy
  ↓
Retrieval Layer
  - DMC direct lookup
  - graph-expanded metadata retrieval
  - vector search
  - optional keyword/BM25 search
  - reranking
  ↓
Structured Answer Plan
  - claim list
  - evidence mapping per claim
  - required citations
  - forbidden claims
  - UI sections
  ↓
LLM Verbalizer
  - Korean answer synthesis
  - detail-level adaptation
  - no unsupported facts
  - cite DMCs
  ↓
Quality Gate
  - grounding check
  - support-level check
  - warning/caution check
  - no fabricated steps/tools
  - answer/reference/UI separation
  ↓
Final Response
  - answer bubble
  - evidences
  - reference materials
  - warnings/limitations
  - related documents
```

---

## 2. Why v4 is needed

The current v3 runtime is fast and stable because it avoids most LLM generation. That is useful for a demo baseline, but it limits the product.

### v3 strengths to preserve

- Ontology-first control flow.
- Explicit support levels.
- Deterministic behavior for exact DMC/procedure cases.
- Fast response time.
- Reduced hallucination.
- Good UI separation between answer, evidence, and reference materials.

### v3 limitations to fix

- Answer composer is still partly hardcoded.
- LLM reasoning is mostly unused.
- Detail level is not deeply modeled.
- Multi-document synthesis is limited.
- Related documents are not fully graph-planned.
- Component/procedure/figure/warning/reference relationships are not rich enough.
- The system can explain known cases, but cannot yet behave like a flexible technical assistant.

### v4 design principle

```text
Do not let the LLM decide what is true.
Let the ontology and retrieved evidence decide what is true.
Use the LLM to explain that truth clearly.
```

---

## 3. Core data model extensions

### 3.1 Query model

Extend `ParsedQuery` beyond intent/target/action.

Recommended fields:

```python
@dataclass(frozen=True)
class ParsedQuery:
    original: str
    normalized: str
    intent: Intent
    target: str | None = None
    action: str | None = None
    dm_type: str | None = None
    detail_level: DetailLevel = DetailLevel.NORMAL
    audience: Audience = Audience.GENERAL
    answer_mode: AnswerMode = AnswerMode.EXPLANATION
    requested_sections: tuple[str, ...] = ()
    follow_up: bool = False
    referenced_dmcs: tuple[str, ...] = ()
    confidence: float = 0.0
    matched_aliases: tuple[str, ...] = ()
```

Add enums:

```python
class DetailLevel(StrEnum):
    BRIEF = "brief"
    NORMAL = "normal"
    DETAILED = "detailed"

class Audience(StrEnum):
    GENERAL = "general"
    TECHNICIAN = "technician"
    EXPERT = "expert"

class AnswerMode(StrEnum):
    EXPLANATION = "explanation"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    LOOKUP = "lookup"
```

### 3.2 Graph node and edge model

Recommended node types:

```text
DataModule
System
Component
Procedure
Action
Fault
Warning
Caution
Note
Figure
Graphic
Tool
Supply
Reference
Applicability
SecurityClass
```

Recommended edge types:

```text
HAS_COMPONENT
HAS_PROCEDURE
HAS_DESCRIPTION
HAS_FAULT
HAS_WARNING
HAS_CAUTION
HAS_FIGURE
USES_TOOL
USES_SUPPLY
REFERENCES
APPLIES_TO
PART_OF
OPERATES_BY
PRODUCES_EFFECT
RELATED_TO
```

A graph edge should preserve source DMC and XML location when possible.

```python
@dataclass(frozen=True)
class OntologyEdge:
    source_id: str
    relation: RelationType
    target_id: str
    source_dmc: str | None = None
    structure_path: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.3 Evidence plan model

```python
@dataclass(frozen=True)
class EvidencePlan:
    primary_dmcs: tuple[str, ...]
    related_dmcs: tuple[str, ...] = ()
    required_blocks: tuple[str, ...] = ()
    include_warnings: bool = True
    include_figures: bool = False
    include_tools: bool = True
    retrieval_modes: tuple[RetrievalMode, ...] = (
        RetrievalMode.DMC_DIRECT,
        RetrievalMode.GRAPH_EXPANDED,
        RetrievalMode.VECTOR,
    )
    support: SupportLevel = SupportLevel.NONE
    reason: str = ""
```

### 3.4 Structured answer plan

The LLM should receive a plan, not raw unstructured chunks only.

```python
@dataclass(frozen=True)
class AnswerClaim:
    text: str
    evidence_dmcs: tuple[str, ...]
    evidence_block_ids: tuple[str, ...] = ()
    required: bool = True

@dataclass(frozen=True)
class AnswerPlan:
    query: ParsedQuery
    support: SupportLevel
    claims: tuple[AnswerClaim, ...]
    related_documents: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    output_sections: tuple[str, ...] = ()
```

---

## 4. Runtime modes

v4 should not send every question to the LLM. It should route intelligently.

| Query type | Runtime path | Reason |
| --- | --- | --- |
| Exact DMC lookup | deterministic | exact lookup is faster and safer |
| Simple component list | deterministic or hybrid | graph facts are enough |
| Unsupported procedure | deterministic | must not let LLM invent steps |
| Exact procedure summary | hybrid | deterministic step extraction + optional LLM explanation |
| Operation principle | LLM synthesis | requires explanatory wording and relationship traversal |
| Multi-document comparison | LLM synthesis | needs controlled synthesis |
| Troubleshooting | LLM synthesis + strict gate | high value but high risk |
| Follow-up question | graph/session memory + LLM or deterministic | depends on previous evidence |

---

## 5. Implementation phases

### Phase 0: Freeze v3 as the baseline

Objective: Keep the current deterministic ontology-first runtime as a known-good fallback.

Tasks:

1. Mark `pipeline_v2.py` as the v3 baseline in docs/comments.
2. Ensure v3 regression tests remain green.
3. Do not keep expanding deterministic answers for every new behavior.
4. Add a feature flag for future v4 runtime selection.

Suggested flag:

```text
S1000D_RAG_RUNTIME=v3_deterministic | v4_graph_llm
```

### Phase 1: Build the v4 graph schema

Objective: Convert the current ontology manifest into a richer graph.

Files likely involved:

```text
src/rag/ontology/schema.py
src/rag/ontology/manifest_builder.py
src/rag/ontology/graph_schema.py
src/rag/ontology/graph_builder.py
tests/test_ontology_graph_builder.py
```

Acceptance criteria:

- DataModule nodes are created for each parsed DMC.
- Component/System/Procedure nodes are extracted from titles and metadata.
- Edges connect DataModules to target/action/component relationships.
- Brake system graph contains brake lever/cable/arm/pad relationships.
- Wheel/tire/chain/lighting/handlebar relationships are represented without hand-written SNS false positives.

### Phase 2: Add detail-level and answer-mode parsing

Objective: Parse user intent beyond target/action.

Required examples:

| Query | Expected detail_level | Expected answer_mode |
| --- | --- | --- |
| `브레이크 작동원리 간단히 알려줘` | brief | explanation |
| `브레이크 작동원리를 자세히 설명해줘` | detailed | explanation |
| `정비사가 이해할 수 있게 설명해줘` | detailed | explanation, audience=technician |
| `브레이크 패드 청소 절차 알려줘` | normal | procedure |
| `앞서 말한 문서 내용은?` | normal | summary, follow_up=true |

### Phase 3: Implement graph-aware evidence planning

Objective: Use graph traversal to decide which documents and blocks to retrieve.

Example for brake operation principle:

```text
Query: 브레이크 작동원리를 자세히 설명해줘

Plan:
- primary descriptive DMC: BRAKE-AAA-DA1-00-00-00AA-041A-A
- required relationships:
  - brake lever -> brake cable -> brake arm -> brake pad -> wheel rim
- related procedural DMCs:
  - brake manual test
  - brake pad cleaning
- include warnings: false unless present in primary evidence
- include figures: true if user asks for 그림/도식/위치
```

### Phase 4: Create structured answer planner

Objective: Convert resolved graph/evidence into claims before LLM generation.

Example claim plan:

```text
Claim 1: Brake system transmits force from lever through cable.
Evidence: BRAKE-AAA-DA1-00-00-00AA-041A-A

Claim 2: Cable pulls brake levers together.
Evidence: BRAKE-AAA-DA1-00-00-00AA-041A-A

Claim 3: Brake pads press against the outer wheel rim.
Evidence: BRAKE-AAA-DA1-00-00-00AA-041A-A

Claim 4: Friction reduces wheel speed.
Evidence: BRAKE-AAA-DA1-00-00-00AA-041A-A
```

The LLM prompt should receive claims and evidence snippets with strict instructions:

```text
Use only the provided claims and evidence.
Do not add tools, steps, warnings, or measurements not present in evidence.
If evidence is partial, explicitly say it is partial.
Cite DMCs exactly as provided.
Return Korean text.
```

### Phase 5: Add LLM verbalizer

Objective: Use the local LLM for answer synthesis only after the answer plan is constructed.

Files likely involved:

```text
src/rag/ontology/llm_verbalizer.py
src/rag/ontology/prompts.py
tests/test_ontology_llm_verbalizer.py
```

Required behavior:

- Detailed explanation queries use LLM synthesis.
- Simple lookup queries can bypass LLM.
- LLM output must preserve required claims and DMCs.
- LLM output must not include forbidden claims.
- If LLM output fails the gate, fall back to deterministic claim rendering.

### Phase 6: Upgrade quality gate

Objective: Verify both the answer and its relationship to evidence.

Checks:

```text
- Required DMCs present.
- Required claims represented.
- Unsupported procedure not invented.
- Warning/caution blocks not omitted when required.
- Tools/supplies not fabricated.
- No prompt/template artifacts.
- No repeated DMC blocks.
- Answer bubble does not dump raw reference metadata.
- Reference materials include source evidence.
```

### Phase 7: UI/API integration

Objective: Expose v4 behavior cleanly.

API response should distinguish:

```text
answer: user-facing explanation
support_level: exact/partial/related/none
runtime_mode: deterministic | graph_llm | fallback
ontology_trace: parse/resolve/plan summary
related_documents: list
reference_materials: existing UI panel data
warnings: limitations/safety notes
```

### Phase 8: Evaluation and portfolio evidence

Objective: Prove that v4 is better than v1/v2/v3.

Evaluation sets:

1. Exact DMC lookup.
2. Procedure exact support.
3. Procedure unsupported/partial support.
4. Descriptive operation principles.
5. Component relationship questions.
6. Multi-document relation questions.
7. Follow-up questions.
8. Figure/reference questions.
9. Safety/warning questions.
10. Negative hallucination traps.

Metrics:

```text
- DMC accuracy
- support-level accuracy
- required-claim coverage
- unsupported-claim rate
- warning/caution preservation
- answer Korean quality
- latency by runtime mode
- fallback rate
```

---

## 6. Initial v4 golden questions

```text
브레이크 작동원리를 자세히 설명해줘
브레이크 시스템의 구성품과 각각의 역할을 설명해줘
브레이크 패드 청소 절차와 정비상 주의점을 알려줘
브레이크 제동력이 약하면 어떤 문서들을 봐야 해?
브레이크 케이블 제거 후 다시 설치하는 절차가 있나?
바퀴 교체 방법 알려줘
앞바퀴 설치 절차와 관련된 문서 알려줘
조명 시스템 점검 방법 알려줘
체인에 오일 바르는 방법과 필요한 준비물을 알려줘
앞에서 알려준 문서의 핵심 내용을 정비사 관점으로 요약해줘
```

Each question should verify:

```text
- parsed intent/target/action/detail_level
- ontology candidates
- evidence plan
- answer claims
- final answer
- citations/reference materials
```

---

## 7. Repository migration recommendation

Recommended file organization:

```text
src/rag/
  pipeline_v3.py              # current deterministic ontology-first baseline
  pipeline_v4.py              # new graph + LLM runtime
  ontology/
    schema.py
    graph_schema.py
    graph_builder.py
    query_parser.py
    resolver.py
    evidence_planner.py
    retriever.py
    answer_planner.py
    llm_verbalizer.py
    quality_gate.py
    trace_models.py

tests/
  test_pipeline_v3_regressions.py
  test_pipeline_v4_*.py
  test_ontology_graph_*.py
```

Do not continue growing legacy `src/rag/pipeline.py` guards.

Do not keep calling the current deterministic runtime “final ontology chatbot.” It is the v3 baseline.

---

## 8. Acceptance criteria for v4 MVP

v4 MVP is acceptable when:

1. Runtime can be selected with a feature flag.
2. v3 remains available as fallback.
3. At least 20 golden questions produce traceable parse/resolve/plan/answer/gate artifacts.
4. Detailed explanation queries use LLM synthesis from an answer plan.
5. Unsupported procedure queries do not hallucinate steps.
6. Every answer has DMC-grounded evidence or explicitly says support is missing/partial.
7. Quality gate can reject malformed LLM output and fall back safely.
8. UI separates answer, evidence, references, and related documents.
9. Latency is measured separately for deterministic vs graph-LLM modes.
10. Documentation explains v1→v4 evolution for portfolio review.

---

## 9. Implementation warning

A full v4 rewrite should avoid two traps:

1. **Do not rebuild v2/v3 as more rules.**
   - More deterministic cases will make the demo look better but will not create an LLM chatbot.

2. **Do not let the LLM bypass the ontology.**
   - If the LLM receives raw chunks without a structured evidence plan, the system regresses toward v1.

The correct middle ground is:

```text
Symbolic ontology for control.
Retrieval for source evidence.
LLM for explanation.
Quality gate for trust.
```
