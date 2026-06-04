# Ontology-First RAG v2 Rewrite Plan

> **For Hermes:** Use subagent-driven-development or an OMX/Codex producer-reviewer loop to implement this plan task-by-task. This is a full rewrite of the RAG decision path, not a compatibility patch for v1.

**Goal:** Replace the guard-heavy v1 RAG pipeline with an ontology-first S1000D RAG pipeline that reasons over DMC metadata, document type, target, action, and evidence support level before composing answers.

**Architecture:** Build a corpus-derived ontology manifest from the existing full Chroma metadata and S1000D XML structure. Parse user questions into ontology intents, resolve exact/partial/related candidate DMCs, retrieve chunks inside the selected evidence plan, then compose deterministic or tightly constrained answers with an explicit quality gate. The v1 pipeline may be removed after v2 tests and LangSmith traces pass.

**Tech Stack:** Python, FastAPI, LangChain/Chroma, LangSmith tracing, pytest, S1000D XML metadata, local GGUF LLM only as optional rewrite/composition support.

---

## Design principles

1. Ontology first, vector second.
   - Use DMC/title/dm_type/target/action metadata before semantic vector search.

2. Corpus-derived taxonomy beats hand-written SNS guesses.
   - Never hard-code `체인 -> DA2` or similar unless derived and tested against the manifest.

3. Support level is first-class.
   - Every answer path must know whether evidence is exact, partial, related, or absent.

4. Deterministic composition where possible.
   - DMC lookup, procedures, component lists, partial-support messages, and not-found messages should not depend on the local LLM.

5. LLM is a constrained helper, not the source of truth.
   - Use it only to rewrite verified evidence into Korean when deterministic templates are insufficient.

6. LangSmith trace every decision boundary.
   - Debugging must show parse, resolve, evidence plan, support verification, composition mode, and quality gate result.

7. Remove v1 guard assumptions.
   - Do not preserve brake-focused QA restrictions such as blocking wheel/tire procedures.

---

## Target file layout

Create:

```text
src/rag/ontology/
  __init__.py
  schema.py
  manifest_builder.py
  query_parser.py
  resolver.py
  evidence_planner.py
  answer_composer.py
  quality_gate.py

src/rag/pipeline_v2.py
scripts/build_ontology_manifest.py
scripts/run_ontology_regression.py

tests/test_ontology_query_parser.py
tests/test_ontology_resolver.py
tests/test_ontology_answer_composer.py
tests/test_pipeline_v2_regressions.py
```

Modify after v2 passes:

```text
app_web.py
query.py
src/rag/pipeline.py or delete/replace with pipeline_v2 facade
src/rag/query_enhancer.py or retire from runtime path
```

---

## Core schema

### `src/rag/ontology/schema.py`

Define dataclasses or Pydantic models:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Intent(StrEnum):
    DESCRIBE = "describe"
    LIST_COMPONENTS = "list_components"
    PROCEDURE = "procedure"
    DMC_LOOKUP = "dmc_lookup"
    DOCUMENT_SUMMARY = "document_summary"
    FAULT = "fault"
    VISUAL = "visual"
    FOLLOW_UP = "follow_up"
    UNKNOWN = "unknown"


class SupportLevel(StrEnum):
    EXACT = "exact"
    PARTIAL = "partial"
    RELATED = "related"
    NONE = "none"


@dataclass(frozen=True)
class OntologyNode:
    dmc: str
    title: str
    dm_type: str
    sns_code: str | None = None
    target: str | None = None
    action: str | None = None
    applicability: str | None = None
    source_file: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedQuery:
    original: str
    normalized: str
    intent: Intent
    target: str | None = None
    action: str | None = None
    dm_type: str | None = None
    confidence: float = 0.0
    matched_aliases: tuple[str, ...] = ()
    follow_up: bool = False


@dataclass(frozen=True)
class CandidateEvidence:
    node: OntologyNode
    support: SupportLevel
    reason: str
    score: float = 0.0


@dataclass(frozen=True)
class ResolutionResult:
    parsed: ParsedQuery
    support: SupportLevel
    candidates: tuple[CandidateEvidence, ...]
    reason: str


@dataclass(frozen=True)
class QualityGateResult:
    ok: bool
    reasons: tuple[str, ...] = ()
```

---

## Regression questions

These are the initial acceptance suite. They should be encoded in tests and in `scripts/run_ontology_regression.py`.

```python
REGRESSION_CASES = [
    {
        "id": "brake-system-description",
        "question": "브레이크 시스템에 대해 알려줘",
        "intent": "describe",
        "target": "brake system",
        "support": "exact",
        "expected_dmcs": ["BRAKE-AAA-DA1-00-00-00AA-041A-A"],
        "forbidden": ["브레이크 버", "레이크 이블", "레이크 드", "곽", "도를 입니다"],
    },
    {
        "id": "brake-components",
        "question": "브레이크 시스템 주요 구성품 알려줘",
        "intent": "list_components",
        "target": "brake system",
        "support": "exact",
        "must_contain": ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"],
    },
    {
        "id": "brake-pad-cleaning",
        "question": "브레이크 패드 청소 절차 알려줘",
        "intent": "procedure",
        "target": "brake pad",
        "action": "clean",
        "support": "exact",
        "expected_dmcs": ["BRAKE-AAA-DA1-10-00-00AA-251A-A"],
    },
    {
        "id": "brake-cable-remove-install",
        "question": "브레이크 케이블 제거 후 다시 설치하는 방법은?",
        "intent": "procedure",
        "target": "brake cable",
        "support": "related",
        "must_contain": ["정확한", "절차", "찾지 못했습니다"],
    },
    {
        "id": "front-wheel-install",
        "question": "앞바퀴 설치 절차 알려줘",
        "intent": "procedure",
        "target": "front wheel",
        "action": "install",
        "support": "exact",
        "expected_dmcs": ["S1000DBIKE-AAA-DA0-30-00-00AA-720A-A"],
    },
    {
        "id": "wheel-replacement-partial",
        "question": "바퀴 교체 방법 알려줘",
        "intent": "procedure",
        "target": "wheel",
        "action": "replace",
        "support": "partial",
        "must_contain": ["바퀴 자체", "단일 절차", "타이어", "휠"],
    },
    {
        "id": "chain-oil",
        "question": "체인에 오일 바르는 방법 알려줘",
        "intent": "procedure",
        "target": "chain",
        "action": "oil",
        "support": "exact",
        "expected_dmcs": ["S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"],
    },
    {
        "id": "handlebar-remove",
        "question": "핸들바 탈거 방법 알려줘",
        "intent": "procedure",
        "target": "handlebar",
        "action": "remove",
        "support": "exact",
        "expected_dmcs": ["S1000DBIKE-AAA-DA2-20-00-00AA-520A-A"],
    },
    {
        "id": "lighting-test",
        "question": "조명 시스템 점검 방법 알려줘",
        "intent": "procedure",
        "target": "lights",
        "action": "test",
        "support": "exact",
        "expected_dmcs": ["S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A"],
        "forbidden_routing": ["stem", "handlebar", "DA4"],
    },
]
```

---

## Implementation tasks

### Task 1: Add v1 retrospective documentation

**Objective:** Preserve the failure analysis for portfolio and future design decisions.

**Files:**
- Create: `docs/retrospectives/rag-v1-failure-analysis.md`

**Verification:**

Run:

```bash
test -f docs/retrospectives/rag-v1-failure-analysis.md
```

Expected: exit code 0.

---

### Task 2: Add ontology rewrite plan

**Objective:** Save this implementation plan in the repository.

**Files:**
- Create: `docs/plans/ontology-first-rag-v2-rewrite.md`

**Verification:**

Run:

```bash
test -f docs/plans/ontology-first-rag-v2-rewrite.md
```

Expected: exit code 0.

---

### Task 3: Create ontology schema

**Objective:** Define shared typed objects for v2 parsing, resolution, support, and quality gating.

**Files:**
- Create: `src/rag/ontology/__init__.py`
- Create: `src/rag/ontology/schema.py`
- Test: `tests/test_ontology_schema.py`

**Test cases:**

- `SupportLevel.EXACT.value == "exact"`
- `ParsedQuery(...).intent` stores `Intent.PROCEDURE`
- `ResolutionResult` can hold candidates and support level

**Command:**

```bash
pytest tests/test_ontology_schema.py -q
```

---

### Task 4: Build ontology manifest from Chroma metadata

**Objective:** Generate a richer ontology manifest from the full Chroma collection metadata.

**Files:**
- Create: `src/rag/ontology/manifest_builder.py`
- Create: `scripts/build_ontology_manifest.py`
- Test: `tests/test_ontology_manifest_builder.py`

**Required behavior:**

- Read `chroma_db_full/s1000d_chunks_full` metadata.
- Deduplicate by DMC/title/dm_type/target/action.
- Include procedure nodes, descriptive nodes, fault nodes, and visual metadata if available.
- Preserve `sns_code`, `source_file`, `applicability`, and title.
- Add generated aliases from title and known Korean synonym map.

**Verification:**

```bash
python scripts/build_ontology_manifest.py \
  --chroma-dir chroma_db_full \
  --collection s1000d_chunks_full \
  --output chroma_db_full/ontology_manifest.json

python - <<'PY'
import json
from pathlib import Path
p=Path('chroma_db_full/ontology_manifest.json')
data=json.loads(p.read_text())
print(len(data['nodes']))
assert any(n['target']=='front wheel' and n['action']=='install' for n in data['nodes'])
assert any(n['target']=='chain' and n['action']=='oil' for n in data['nodes'])
assert any(n['target']=='handlebar' and n['action']=='remove' for n in data['nodes'])
PY
```

---

### Task 5: Implement token/phrase-safe query parser

**Objective:** Replace substring SNS routing with ontology query parsing.

**Files:**
- Create: `src/rag/ontology/query_parser.py`
- Test: `tests/test_ontology_query_parser.py`

**Required mappings:**

Targets:

- `브레이크 시스템`, `브레이크` -> `brake system`
- `브레이크 패드` -> `brake pad`
- `브레이크 케이블` -> `brake cable`
- `앞바퀴`, `전륜` -> `front wheel`
- `뒷바퀴`, `후륜` -> `rear wheel`
- `바퀴`, `휠` -> `wheel`
- `타이어` -> `tire`
- `체인` -> `chain`
- `핸들바`, `핸들` -> `handlebar`
- `스템` -> `stem`
- `조명 시스템`, `조명`, `라이트` -> `lights`

Actions:

- `설치`, `장착`, `조립` -> `install`
- `탈거`, `제거`, `분리` -> `remove`
- `교체`, `바꾸` -> `replace`
- `청소`, `세척` -> `clean`
- `오일`, `윤활`, `기름` -> `oil`
- `점검`, `시험`, `테스트`, `동작 확인` -> `test`
- `공기압`, `압력` -> `check_pressure`

Important negative test:

```python
def test_system_does_not_match_stem():
    parsed = parse_query("조명 시스템 점검 방법 알려줘")
    assert parsed.target == "lights"
    assert "stem" not in parsed.matched_aliases
```

---

### Task 6: Implement ontology resolver with support levels

**Objective:** Resolve parsed queries to exact/partial/related/none evidence candidates.

**Files:**
- Create: `src/rag/ontology/resolver.py`
- Test: `tests/test_ontology_resolver.py`

**Rules:**

- Exact:
  - target and action match a procedural node.
  - target matches a descriptive node for describe/list-components.
- Partial:
  - target is broader than available target, or action maps to a combined remove/install document.
  - `wheel + replace` can match tire remove/install and wheel remove/install as partial.
- Related:
  - same subsystem or same target but wrong action.
- None:
  - no useful ontology candidate.

**Acceptance checks:**

- `앞바퀴 설치 절차 알려줘` -> exact, front wheel install DMC.
- `바퀴 교체 방법 알려줘` -> partial, tire/wheel candidates.
- `브레이크 케이블 제거 후 다시 설치하는 방법은?` -> related, not exact.
- `조명 시스템 점검 방법 알려줘` -> exact, lights manual test DMC.

---

### Task 7: Implement candidate chunk retrieval by ontology plan

**Objective:** Retrieve text chunks from Chroma by resolved candidate DMCs rather than only global vector search.

**Files:**
- Create: `src/rag/ontology/evidence_planner.py`
- Test: `tests/test_ontology_evidence_planner.py`

**Rules:**

- If exact/partial candidates have DMCs, retrieve chunks filtered by DMC.
- Use vector similarity only inside selected DMCs where possible.
- If resolver support is none, fall back to global vector search but mark support as uncertain/related until verified.
- Preserve evidence metadata for answer composition.

---

### Task 8: Implement deterministic answer composer

**Objective:** Generate stable Korean answers for common technical-document cases without relying on free-form LLM generation.

**Files:**
- Create: `src/rag/ontology/answer_composer.py`
- Test: `tests/test_ontology_answer_composer.py`

**Modes:**

- `compose_exact_procedure`
- `compose_exact_description`
- `compose_component_list`
- `compose_dmc_lookup`
- `compose_partial_support`
- `compose_related_only`
- `compose_none`
- `compose_follow_up_summary`

**Required behavior:**

- Always cite DMCs.
- Do not expose raw English if a deterministic Korean phrase is available.
- For procedures, return numbered Korean steps when source step text is clear.
- For incomplete source text caused by unresolved `internalRef`, phrase conservatively and cite the source.

---

### Task 9: Implement quality gate

**Objective:** Reject malformed generated answers before they reach users.

**Files:**
- Create: `src/rag/ontology/quality_gate.py`
- Test: `tests/test_ontology_quality_gate.py`

**Reject patterns:**

- `브레이크 버`
- `레이크 이블`
- `레이크 드`
- `곽`
- `도를 입니다`
- `DM0000000000`
- repeated `답변:` blocks
- repeated `DMC:` blocks beyond a small threshold
- `<한국어 답변>`
- `<think>` / `</think>`
- context header leakage such as `[DMC: ... | Type: ...]`

**Behavior:**

- If deterministic answer fails quality gate, treat as a bug.
- If LLM rewrite fails quality gate, fall back to deterministic evidence summary.

---

### Task 10: Implement `pipeline_v2.py`

**Objective:** Wire parser, resolver, evidence planner, composer, and quality gate into a new pipeline.

**Files:**
- Create: `src/rag/pipeline_v2.py`
- Test: `tests/test_pipeline_v2_regressions.py`

**Traceable stages:**

Use LangSmith `@traceable` names:

- `rag_pipeline_v2`
- `parse_query`
- `resolve_ontology`
- `plan_evidence`
- `retrieve_candidate_chunks`
- `verify_support`
- `compose_answer`
- `quality_gate`

**Important:** Do not call v1 `_guard_*` functions from v2.

---

### Task 11: Add regression runner

**Objective:** Run the v2 acceptance suite against the running server or in-process pipeline.

**Files:**
- Create: `scripts/run_ontology_regression.py`

**Output shape:**

```json
{
  "passed": 9,
  "failed": 0,
  "cases": [
    {
      "id": "front-wheel-install",
      "support": "exact",
      "dmcs": ["S1000DBIKE-AAA-DA0-30-00-00AA-720A-A"],
      "quality_ok": true
    }
  ]
}
```

---

### Task 12: Switch app to v2

**Objective:** Replace the app runtime path with v2 after tests pass.

**Files:**
- Modify: `app_web.py`
- Modify: `query.py`
- Possibly modify: `src/rag/pipeline.py` to become a thin v2 facade or delete after imports are updated.

**Verification:**

```bash
pytest tests/test_pipeline_v2_regressions.py -q
pytest tests/ -q
python scripts/run_ontology_regression.py
```

Then start server:

```bash
eval "$(python scripts/local_model_env.py)"
uvicorn app_web:app --host 127.0.0.1 --port 8000
```

Smoke API:

```bash
python - <<'PY'
import json, urllib.request, uuid
for q in [
    '브레이크 시스템에 대해 알려줘',
    '앞바퀴 설치 절차 알려줘',
    '바퀴 교체 방법 알려줘',
    '조명 시스템 점검 방법 알려줘',
]:
    payload=json.dumps({'session_id': str(uuid.uuid4()), 'question': q}).encode()
    req=urllib.request.Request('http://127.0.0.1:8000/api/chat', data=payload, headers={'Content-Type':'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=120) as r:
        data=json.load(r)
    print('\nQ:', q)
    print(data.get('answer'))
    print('DMCs:', [e.get('dmc') for e in data.get('evidences', [])[:3]])
PY
```

---

## Definition of done

- v1 failure analysis is committed to docs.
- Ontology v2 plan is committed to docs.
- v2 pipeline does not call v1 query guards.
- Regression cases pass.
- `시스템` no longer routes to `stem`.
- wheel/front-wheel/tire queries no longer false-negative due to brake-focused policy.
- malformed Korean artifacts are blocked by quality gate.
- LangSmith traces show ontology parse/resolve/support/compose stages.
- Browser/API demo answers are understandable and cite DMCs.

---

## Optional future work

- Resolve S1000D `internalRef` values to supply/tool names during parsing so procedure text does not lose words.
- Add visual ontology edges for figures, ICNs, hotspots, and captions.
- Add a small Korean technical terminology table per target/action.
- Add an evaluator that compares answer support level against ontology resolution.
- Generate a portfolio architecture diagram comparing v1 and v2.
