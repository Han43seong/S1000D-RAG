# RAG v1 Failure Analysis: Why the Guard-Heavy Pipeline Failed

> Portfolio note: this document records the failure modes discovered while testing the first S1000D-RAG chatbot pipeline. The goal is to preserve the engineering lessons, not to hide the failed path.

## Summary

The first production-like RAG pipeline reached a useful demo state, but it became unstable as the project moved from a narrow brake-focused QA loop to a full S1000D Bike corpus demo.

The main failure was not a single bad prompt or a missing vector index. The live app was already using the full Chroma index, but multiple pipeline layers had diverging assumptions:

- query expansion used hand-written Korean substring rules;
- SNS routing used stale taxonomy assumptions;
- graph-first retrieval used a narrower ontology than the corpus actually contained;
- procedure guards encoded old brake-focused QA constraints;
- LLM generation was expected to translate and summarize technical English reliably, but the local 8B GGUF model often produced malformed Korean;
- post-processing tried to clean artifacts after generation instead of preventing bad answers from being selected.

The result was a system that could pass selected smoke tests but failed under normal exploratory use.

## Evidence from LangSmith and local inspection

### Live index was not the primary problem

The live FastAPI server was verified to use the full index:

- Chroma persist dir: `/home/hskim/projects/S1000D-RAG/chroma_db_full`
- collection: `s1000d_chunks_full`
- chunk count: 171
- data modules parsed: 106
- parse errors: 0

The smaller legacy smoke index still existed:

- `chroma_db` / `s1000d_chunks`
- 5 data modules
- 7 chunks

However, `/api/status` reported 171 chunks and the running process environment pointed at `chroma_db_full`, so the main answer-quality failures were not caused by accidentally using the small smoke DB.

### Bad answer despite correct retrieval

User question:

```text
브레이크 시스템에 대해 알려줘
```

Observed bad answer:

```text
브레이크 시스템은 주로 브레이크 버, 레이크 이블, 레이크 , 레이크 클램프(또는 리퍼), 그리고 레이크 드로 구성됩니다. ...
```

LangSmith showed that retrieval was correct:

- `BRAKE-AAA-DA1-00-00-00AA-041A-A`
- `S1000DBIKE-AAA-DA1-00-00-00AA-041A-A`

The retrieved text clearly contained:

- brake lever
- brake cable
- brake arm
- brake clamp / callipers
- brake pads
- pads pressing against the wheel rim to reduce speed

Root cause classification:

- retrieval: pass
- reranking/context: pass
- LLM generation: fail
- post-processing/quality gate: fail

The raw LlamaCpp output included repeated answer blocks, repeated DMC lines, prompt template fragments, and malformed Korean terminology. The pipeline sometimes cleaned this enough to look acceptable, but not reliably.

### False negative: front wheel install

User question:

```text
앞바퀴 설치 절차 알려줘
```

The system previously tended to answer that the procedure could not be found.

But the full corpus contains:

- DMC: `S1000DBIKE-AAA-DA0-30-00-00AA-720A-A`
- title: `Front wheel - Install procedures`
- source XML includes:
  - `Hold the front of the bicycle.`
  - `Install the wheel ...`

Root cause classification:

- corpus: contains answer
- graph resolver: failed to extract `front wheel` target from Korean `앞바퀴`
- procedure guard: treated wheel/tire procedures as unsupported because the old QA loop was brake-focused
- final answer policy: incorrectly collapsed partial/exact evidence into a no-answer response

### Partial support mishandled: wheel replacement

User question:

```text
바퀴 교체 방법 알려줘
```

The corpus does not appear to contain a clean "wheel replacement" procedure as a single exact document, but it does contain related procedures:

- `Tire - Remove and install a new item`
- `Front wheel - Remove procedures`
- `Front wheel - Install procedures`
- `Rear wheel - Remove procedures`

A good answer should distinguish exact support from partial support:

```text
바퀴 자체를 교체하는 단일 절차는 찾지 못했습니다. 다만 관련 문서에는 타이어 교체 절차와 앞바퀴 탈거/장착 절차가 있습니다...
```

v1 often answered as if nothing useful existed.

Root cause classification:

- support-level model missing
- no distinction between exact, partial, related, and none
- procedure guard overrode useful evidence

### Follow-up failure: "알려준 문서 내용은 뭔데?"

User question:

```text
알려준 문서 내용은 뭔데?
```

Expected behavior:

- use the previous turn's evidence DMC(s);
- summarize the previously referenced document.

Observed behavior:

- the system performed a fresh global search;
- it retrieved an unrelated process/tutorial chunk;
- the local LLM emitted a `DM000000...` repetition artifact.

Root cause classification:

- missing conversational evidence memory
- follow-up/anaphora not modeled
- quality gate did not reject obvious generation artifacts

## Structural causes

### 1. Stale and incorrect SNS mapping

`src/rag/query_enhancer.py` used hand-written SNS keyword rules. Some of these contradicted the actual full Bike corpus metadata.

Code assumptions included:

- `체인` -> `DA2`
- `핸들/스템` -> `DA4`
- `조명` -> `D05`

Actual full-corpus graph manifest showed:

- `chain` -> `DA4`
- `handlebar`, `stem`, `headset`, `spacer` -> `DA2`
- `lighting`, `lights` -> `D00` in the sample lighting DMCs
- `front wheel`, `rear wheel`, `tire` -> `DA0`

This meant the router sometimes filtered the vector search into the wrong subsystem.

### 2. Substring matching caused Korean false positives

The SNS extractor used `kw in query_lower` matching. This caused false positives such as:

- `시스템` containing `스템`
- `조명 시스템 점검 방법` routed to the handlebar/stem bucket

Token-aware phrase matching or ontology-derived aliases are required.

### 3. Graph manifest existed but was underused

The full index already generated `s1000d_graph_manifest.json` with useful procedure/fault metadata. Example targets/actions included:

- `front wheel` / `install`
- `front wheel` / `remove`
- `chain` / `oil`
- `handlebar` / `remove`
- `stem` / `install`
- `lights` / `test`

However, `graph_retrieval.py` only recognized a small set of Korean/English target aliases. It did not robustly map `앞바퀴`, `뒷바퀴`, `바퀴`, `조명 시스템`, and other common Korean queries to graph targets.

### 4. Guard-heavy pipeline encoded old QA constraints

`src/rag/pipeline.py` contained many `_guard_*` functions that were introduced to pass targeted QA loops or to prevent hallucinations for known cases.

This created two problems:

1. The pipeline became brittle and hard to reason about.
2. Old product assumptions remained active after the scope changed.

The most damaging example was `_is_unsupported_wheel_procedure()`, which intentionally blocked wheel/tire procedures because an earlier QA loop was brake-focused. In the later full Bike demo, this policy became wrong.

### 5. No support-level abstraction

v1 did not consistently model whether retrieved evidence was:

- exact support;
- partial support;
- related context only;
- unsupported.

Without this abstraction, the system oscillated between hallucinating procedures and saying "not found" even when related evidence was useful.

### 6. LLM output was treated as primary truth

The local Qwen3-8B GGUF model was asked to:

- read English S1000D context;
- translate to Korean;
- summarize accurately;
- obey strict output format;
- avoid repetition;
- cite DMCs.

In practice, the model often emitted:

- malformed Korean technical terms;
- repeated `답변:` blocks;
- repeated `DMC:` blocks;
- prompt placeholders;
- context-copy artifacts;
- long repeated zero-like DMC strings.

The post-processor cleaned some artifacts, but it was not a semantic quality gate. Bad generated answers could still pass through.

### 7. XML text extraction dropped internal references

Some parsed procedure text lost internal reference labels, producing incomplete sentences such as:

```text
Apply a thin layer of the on each of the brake pads using a .
```

This happened when S1000D `internalRef` nodes referenced supplies/tools that were not resolved into readable text. This reduced evidence quality and made deterministic answer composition harder.

## Engineering lessons

1. A full vector index does not guarantee good RAG behavior.
2. Corpus-derived ontology should be the source of truth for routing, not hand-written subsystem guesses.
3. Hard filters are dangerous when taxonomy confidence is low; prefer graph candidates and soft boosts.
4. QA-specific guards should not live in the core product pipeline unless feature-flagged.
5. "Not found" should be based on explicit support-level reasoning, not only retrieval score thresholds.
6. Local LLM output needs a semantic quality gate, not only string cleanup.
7. Follow-up questions require evidence memory, not just chat text history.
8. S1000D structure is valuable: DMC, dm_type, title, applicability, procedures, figures, warnings, references, and SNS codes should be first-class retrieval signals.

## Decision

The v1 guard-heavy pipeline should be replaced with an ontology-first RAG v2 pipeline.

The rewrite should not preserve v1 behavior for compatibility. The goal is to preserve v1 lessons in documentation and tests while replacing the architecture.

## Success criteria for v2

v2 should pass these representative cases:

| Query | Expected support | Expected behavior |
| --- | --- | --- |
| `브레이크 시스템에 대해 알려줘` | exact descriptive | clean Korean summary grounded in brake system DM |
| `브레이크 시스템 주요 구성품 알려줘` | exact descriptive/list | deterministic component list |
| `브레이크 패드 청소 절차 알려줘` | exact procedural | step summary from brake pad cleaning DM |
| `브레이크 케이블 제거 후 다시 설치하는 방법은?` | unsupported/related | no exact procedure; show related brake docs |
| `앞바퀴 설치 절차 알려줘` | exact procedural | front wheel install DM, no false negative |
| `바퀴 교체 방법 알려줘` | partial/related | explain wheel replacement is not exact; cite tire/wheel procedures |
| `체인에 오일 바르는 방법 알려줘` | exact procedural | chain oil DM under DA4 |
| `핸들바 탈거 방법 알려줘` | exact procedural | handlebar remove DM under DA2 |
| `조명 시스템 점검 방법 알려줘` | exact procedural | lighting/lights manual test DM; no `시스템` -> `스템` false route |
| `알려준 문서 내용은 뭔데?` | follow-up | summarize previous evidence DMC |

## Files most implicated in v1 failure

- `src/rag/query_enhancer.py`
  - stale SNS mapping
  - substring false positives
- `src/rag/graph_retrieval.py`
  - useful start, but incomplete alias/target/action model
- `src/rag/pipeline.py`
  - too many hard-coded guards
  - old brake-focused policy leaked into full Bike demo
  - weak not-found semantics
  - LLM artifact handling after the fact
- `src/rag/prompt.py`
  - asked too much of the local LLM
- XML parser/chunker modules
  - internalRef expansion needs improvement for tool/supply names

## Recommended next architecture

Move to:

```text
query
-> ontology query parser
-> ontology resolver
-> evidence planner
-> candidate chunk retrieval
-> support verifier
-> deterministic answer composer or constrained LLM rewrite
-> quality gate
-> final answer
```

LangSmith traces should expose each stage:

- `parse_query`
- `resolve_ontology`
- `plan_evidence`
- `retrieve_candidate_chunks`
- `verify_support`
- `compose_answer`
- `quality_gate`
- `rag_pipeline_v2`
