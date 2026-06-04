# S1000D-RAG Evolution: v1 to v4

> Portfolio history note: this document intentionally records the engineering detours, failed assumptions, and architectural pivots behind S1000D-RAG. The goal is not to present the current system as if it was obvious from the start, but to show why the final target became an ontology-guided Graph RAG + LLM chatbot.

## Executive summary

S1000D-RAG evolved through four architectural stages:

```text
v1: Vector DB RAG
    Basic vector retrieval over S1000D chunks, then local LLM generation.

v2: Vector RAG with partial ontology concepts
    Vector retrieval remained central, but graph/SNS/DMC metadata and hand-written guards were added to patch failures.

v3: Deterministic ontology-first RAG
    Query parsing, ontology resolution, evidence planning, deterministic answer composition, and quality gates became the main runtime path. LLM generation became optional or mostly unused.

v4: Target architecture — ontology-guided Graph RAG + LLM synthesis
    A full ontology/knowledge graph controls retrieval and evidence planning. RAG retrieves source-grounded evidence. The LLM synthesizes user-adapted answers from structured evidence. Quality gates verify grounding, safety, and UI separation.
```

The current runtime is best described as v3: it is ontology-first and fast, but it is not yet the final ontology-based LLM chatbot. v4 is the final product direction.

---

## Version taxonomy

| Version | Short name | Main idea | Why it existed | Main limitation | Status |
| --- | --- | --- | --- | --- | --- |
| v1 | Vector DB RAG | Search chunks by embeddings, pass context to local LLM | Fastest way to prove S1000D XML can become a local RAG chatbot | Retrieval ambiguity, hallucination, weak DMC control, bad Korean generation | Superseded |
| v2 | Vector RAG + partial ontology | Add DMC/SNS/graph hints, guards, reranking, QA patches | Patch failures found in v1 without a full rewrite | Became guard-heavy and brittle; ontology was not the source of truth | Superseded |
| v3 | Deterministic ontology-first RAG | Parse intent/target/action, resolve ontology support, compose deterministic Korean answers | Stabilize demo, remove LLM output artifacts, make support levels explicit | Very fast and reliable, but not a full LLM reasoning chatbot | Current transition state |
| v4 | Ontology-guided Graph RAG + LLM chatbot | Ontology graph plans retrieval; RAG supplies evidence; LLM synthesizes; quality gate verifies | Final target for a maintainable, explainable technical-document assistant | Requires graph schema, LLM synthesis controls, richer evaluation | Target |

---

## v1 — Vector DB RAG

### What v1 tried to do

v1 followed the common RAG pattern:

```text
S1000D XML
-> parse into chunks
-> embed chunks
-> store in Chroma
-> retrieve top-k chunks by semantic similarity
-> pass context to local LLM
-> post-process answer
```

This was a reasonable first architecture because the earliest goal was to prove that S1000D Data Modules could be parsed, indexed, and queried locally.

### Why this approach was chosen

v1 optimized for speed of implementation:

- Chroma + embeddings provided immediate semantic search.
- Local GGUF LLMs enabled closed-network operation.
- XML parsing and chunking could be tested independently.
- The first demo questions were narrow enough that vector retrieval looked sufficient.

### What v1 revealed

The vector index was not the main problem. In several failures, the correct DMC appeared in retrieved evidence, but the answer was still wrong.

Representative failure modes:

- The local 8B GGUF LLM produced malformed Korean technical terms.
- The model repeated prompt fragments, answer blocks, and DMC lines.
- Similar chunks from related S1000D modules confused the answer.
- DMC lookup and procedure requests were not distinguished strongly enough.
- Follow-up questions lost the previous evidence context.
- There was no explicit support level: exact, partial, related, or unsupported.

### Why v1 was not enough

S1000D is not just unstructured text. It has strong structure:

- DMC
- infoCode / document type
- SNS-like system code
- title
- procedure vs description vs fault
- figure/caption/reference links
- warning/caution/note blocks
- applicability and security metadata

A vector-only pipeline underused these signals.

---

## v2 — Vector RAG with partial ontology concepts

### What v2 added

v2 tried to improve v1 by adding ontology-like hints without fully replacing the vector-centric architecture:

- graph manifest metadata;
- DMC-aware retrieval;
- SNS routing assumptions;
- Korean alias rules;
- reranking;
- answer guards;
- targeted QA regression cases.

This stage was useful because it exposed what the final ontology layer needed to know.

### Why v2 was developed this way

v2 was a pragmatic patching phase. The project had working ingestion, UI, local models, and a demo path, so the fastest response to observed failures was to add targeted fixes around the existing pipeline.

Examples:

- If a brake answer hallucinated, add a brake guard.
- If a wheel query routed incorrectly, add an alias.
- If a DMC lookup failed, add a direct lookup path.
- If the LLM generated malformed Korean, add post-processing.

This approach is common in early RAG systems because every bug seems locally fixable.

### What v2 taught

v2 showed that partial ontology concepts help, but they must be first-class. If ontology metadata is only used as a patch around vector retrieval, the system still behaves like a vector RAG with exceptions.

Key lessons:

1. Corpus-derived taxonomy should beat hand-written SNS guesses.
2. Korean substring matching is unsafe; token/phrase-safe aliases are required.
3. Support levels must be explicit.
4. QA-specific guards must not become product policy.
5. Graph relationships should plan retrieval, not merely decorate results.

### Why v2 was not enough

The architecture became hard to reason about:

```text
vector retrieval
+ reranking
+ graph hints
+ Korean substring rules
+ DMC exceptions
+ procedure guards
+ post-processing
```

Each new guard reduced one failure but increased hidden coupling. The pipeline could pass a targeted QA set while still failing exploratory user questions.

---

## v3 — Deterministic ontology-first RAG

### What v3 is

v3 changes the order of operations:

```text
query
-> parse_query
-> load_ontology_manifest
-> resolve_ontology
-> plan_evidence
-> retrieve_evidence
-> deterministic compose_answer
-> quality_gate
-> final answer
```

The current runtime path is closest to this v3 architecture.

### Why v3 was built

v3 was built to stabilize the product after v1/v2 failures.

The key decision was to stop treating the local LLM as the source of truth. Instead, the system first resolves the query against structured S1000D metadata and then composes safe Korean answers deterministically for known technical-document cases.

This solved important problems:

- no malformed Korean from local LLM generation;
- faster responses because token generation is mostly avoided;
- explicit support-level reasoning;
- cleaner DMC citation;
- deterministic demo behavior;
- less hallucination risk in closed-network conditions.

### Current benefits of v3

v3 is valuable and should not be dismissed as a failure. It is the bridge that made the final architecture visible.

Strengths:

- Very fast after model/index warm-up.
- Predictable output.
- Good for DMC lookup, exact procedure support, not-found messaging, and simple descriptive answers.
- Easier to test than free-form LLM generation.
- Provides the skeleton of the final ontology-first flow.

### Current limitations of v3

v3 is not yet an ontology-based LLM chatbot.

Its answer composer is still too deterministic:

```text
if intent == DESCRIBE and target == "brake system":
    return fixed Korean explanation
```

This is reliable, but it limits the product:

- The chatbot does not truly synthesize across multiple documents.
- Answer depth is not adapted to user intent or expertise.
- “Why/how/compare/troubleshoot” questions cannot fully exploit LLM reasoning.
- Related documents, figures, warnings, tools, and references are not deeply woven into the answer.
- The ontology is mostly metadata-driven, not a rich domain graph of entities and relations.

### Why v3 is not the final target

v3 proves that ontology-first control is necessary. But because LLM reasoning is mostly absent, it cannot show the full value of an ontology-guided technical assistant.

The final assistant should not be a set of deterministic templates. It should be a grounded LLM system where the ontology controls what the model is allowed to synthesize.

---

## v4 — Target: ontology-guided Graph RAG + LLM chatbot

### Final target definition

v4 is the intended final architecture:

```text
User question
-> Ontology query parser
-> Knowledge graph / ontology resolver
-> Evidence planner
-> Graph-expanded retrieval + DMC direct lookup + vector/BM25 retrieval
-> Structured answer plan
-> LLM synthesis / verbalizer
-> Quality gate
-> UI answer + evidence + reference materials
```

The core principle is:

```text
The ontology controls retrieval and reasoning boundaries.
RAG supplies source evidence.
The LLM explains and synthesizes, but does not invent unsupported facts.
The quality gate verifies grounding and safety before the user sees the answer.
```

### What “full ontology” means for this project

A full ontology should represent more than DMC metadata. It should model technical relationships.

Example:

```text
Brake system
  hasComponent -> brake lever
  hasComponent -> brake cable
  hasComponent -> brake arm
  hasComponent -> brake pad
  operatesBy -> cable tension
  producesEffect -> friction on wheel rim
  hasDescriptiveDM -> BRAKE-AAA-DA1-00-00-00AA-041A-A
  hasProcedure -> brake pad cleaning
  hasProcedure -> brake manual test
  hasRelatedFigure -> figure/caption nodes
  hasWarningOrCaution -> safety blocks
```

For S1000D, useful first-class node/edge types include:

- DataModule
- Component
- System
- Procedure
- Action
- Tool / supply / support equipment
- Warning / caution / note
- Figure / caption / graphic
- Fault / symptom / corrective action
- Reference / internalRef
- Applicability / security classification

### What role RAG still plays

v4 does not remove RAG. It makes RAG ontology-guided.

Ontology decides:

- which target/action/document type is relevant;
- whether support is exact, partial, related, or absent;
- which DMCs and relations should be traversed;
- which evidence must be retrieved;
- what the LLM may or may not claim.

RAG retrieves:

- source paragraphs;
- procedural steps;
- warnings/cautions;
- figures/captions;
- support equipment references;
- related DMC content.

LLM synthesizes:

- operation principles;
- step explanations;
- multi-document summaries;
- user-level explanations;
- related-document guidance.

Quality gate verifies:

- DMC grounding;
- unsupported procedure prevention;
- warning/caution preservation;
- no fabricated tools or steps;
- answer/reference/UI separation;
- answer detail level matches the query.

---

## Why v4 is portfolio-worthy

The portfolio value is not only “a chatbot over manuals.” The stronger story is the engineering evolution:

1. Basic vector RAG was easy to build but could not reliably handle technical-document structure.
2. Patching vector RAG with guards created brittleness.
3. Deterministic ontology-first routing stabilized the system and reduced hallucination.
4. The final design combines symbolic structure and neural language generation.

This demonstrates practical judgment:

- recognizing when vector search is insufficient;
- preserving failed approaches as lessons;
- using domain metadata as a control plane;
- separating evidence planning from answer wording;
- designing for closed-network, on-prem technical support;
- balancing deterministic reliability with LLM expressiveness.

---

## Decision

The final target is v4, not v3.

v3 remains useful as the safe baseline and transition architecture. However, future implementation should be planned as a v4 rewrite or major refactor:

```text
v3 deterministic ontology-first RAG
-> v4 ontology-guided Graph RAG + LLM synthesis + quality gate
```

The v4 implementation may replace large parts of the current runtime. Backward compatibility with v1/v2 guard behavior is not a requirement. The important artifacts to preserve are:

- ingestion and S1000D parsing lessons;
- regression questions;
- ontology manifest generation experience;
- support-level semantics;
- UI separation of answer/evidence/reference materials;
- LangSmith trace stages and quality evidence.

---

## Recommended repository narrative

Use these documents as the project history spine:

```text
docs/architecture.md
  Original S1000D local RAG architecture and ingestion model.

docs/retrospectives/rag-v1-failure-analysis.md
  Detailed failure analysis of vector/guard-heavy pipelines.

docs/plans/ontology-first-rag-v2-rewrite.md
  Earlier plan that led to the current deterministic ontology-first transition.

docs/retrospectives/rag-evolution-v1-to-v4.md
  This versioned history and portfolio narrative.

docs/plans/ontology-guided-graph-rag-v4.md
  Final target architecture and implementation roadmap.
```
