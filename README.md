# S1000D-RAG

S1000D technical-manual chatbot project for closed-network maintenance support.

This repository is a portfolio project that explores how to build a reliable AI assistant over S1000D Data Module XML manuals. The project started as a basic vector RAG chatbot, exposed the limits of generic RAG on structured maintenance manuals, and is now being redesigned toward an RDF/OWL-based ontology-guided Graph RAG + LLM reasoning architecture.

---

## 1. Project in one minute

**Problem:**

Technical maintenance manuals such as S1000D are not ordinary text documents. They contain structured Data Module Codes(DMC), document types, procedures, warnings, figures, applicability, references, and system/component relationships. A basic vector search chatbot can retrieve similar text, but it often fails to understand whether a question is asking for a procedure, a component explanation, a DMC lookup, a partial match, or an unsupported maintenance action.

**Goal:**

Build an on-prem / closed-network AI maintenance assistant that can answer Korean user questions over S1000D manuals while preserving source grounding, DMC evidence, procedure safety, and technical-document structure.

**Current direction:**

```text
Vector-only RAG
→ ontology-aware RAG
→ deterministic ontology-first RAG
→ RDF/OWL-based ontology-guided Graph RAG + LLM synthesis + quality gate
```

**Final target:**

An RDF/OWL-based ontology-guided Graph RAG chatbot where:

- RDF/OWL knowledge graph is the canonical ontology representation;
- SPARQL/GraphDB-compatible retrieval controls entities, DMCs, document relationships, and support level;
- Vector DB is used as a constrained chunk-level evidence retriever inside graph-selected S1000D data modules;
- RAG retrieves exact source evidence from S1000D XML/chunks;
- LLM synthesizes user-friendly Korean explanations from structured evidence;
- quality gates reject unsupported procedures, hallucinated tools/steps, malformed citations, and unsafe answers.

---

## 2. Why this project matters

Generic RAG often looks good in demos but breaks on technical maintenance manuals because retrieval similarity is not the same as operational correctness.

This project focuses on questions such as:

```text
브레이크 작동원리를 자세히 설명해줘
브레이크 패드 청소 절차 알려줘
브레이크 케이블 제거 후 다시 설치하는 절차가 있나?
바퀴 교체 방법 알려줘
앞에서 알려준 문서 내용은 뭔데?
```

A useful assistant must distinguish:

- exact procedure vs related procedure;
- description document vs maintenance procedure;
- component relationship vs step-by-step instruction;
- direct DMC lookup vs conversational follow-up;
- supported answer vs unsupported action;
- answer text vs evidence/reference materials.

That is why the project evolved beyond simple vector search.

---

## 3. Architecture evolution

The most important part of this repository is not only the current code, but the recorded engineering evolution.

| Version | Architecture | What it tried to solve | Why it was not final |
| --- | --- | --- | --- |
| **v1** | Basic Vector DB RAG | Parse S1000D XML, index chunks in Chroma, retrieve by embeddings, answer with local LLM | Correct chunks could be retrieved but the local LLM produced malformed Korean, repeated DMCs, and unsupported claims. There was no strong support-level model. |
| **v2** | Vector RAG + partial ontology concepts | Add DMC/SNS/graph hints, reranking, Korean aliases, and guard rules | Became patch-heavy. Ontology metadata existed but did not control the pipeline as the source of truth. |
| **v3** | Deterministic ontology-first RAG | Parse query intent/target/action, resolve ontology metadata, plan evidence, compose deterministic Korean answers | Fast and stable, but LLM reasoning is mostly unused. It is an excellent baseline, not the final chatbot architecture. |
| **v4** | RDF/OWL-based ontology-guided Graph RAG + LLM synthesis | Full target: RDF/OWL canonical ontology, SPARQL/GraphDB-compatible document selection, source-grounded LLM synthesis, and quality gates | Current roadmap / active implementation target. |

Detailed history:

- `docs/retrospectives/rag-evolution-v1-to-v4.md`
- `docs/retrospectives/rag-v1-failure-analysis.md`
- `docs/plans/ontology-guided-graph-rag-v4.md`
- `docs/plans/ontology-first-rag-v2-rewrite.md`

---

## 4. Current runtime state

The current transition runtime is closest to **v3 deterministic ontology-first RAG**.

High-level flow:

```text
User question
  ↓
parse_query
  ↓
load_ontology_manifest
  ↓
resolve_ontology
  ↓
plan_evidence
  ↓
retrieve_evidence
  ↓
compose_answer
  ↓
quality_gate
  ↓
answer + evidences + reference materials
```

This runtime is intentionally fast and stable because the answer composer can produce known technical-document answers without relying on local LLM token generation for every request.

Current strengths:

- fast responses after model/index warm-up;
- lower hallucination risk;
- explicit DMC grounding;
- exact/partial/unsupported answer patterns;
- stable demo behavior;
- UI separation of answer, evidence, and reference materials.

Current limitations:

- not yet a full ontology-based LLM reasoning chatbot;
- answer composer still contains deterministic domain-specific responses;
- multi-document synthesis is limited;
- detail level and user expertise are not fully modeled;
- graph relationships such as component/procedure/figure/warning/tool/reference links need to become first-class.

---

## 5. Final target: v4 ontology-guided Graph RAG

The final architecture should not be “LLM over chunks.” It should be a controlled neuro-symbolic pipeline:

```text
User Question
  ↓
Ontology Query Parser
  - intent
  - target/component/system
  - action
  - detail_level
  - audience
  - follow-up references
  ↓
Knowledge Graph Resolver
  - exact DMCs
  - partial/related DMCs
  - component/system/procedure relationships
  - support level
  ↓
Evidence Planner
  - primary source documents
  - related documents
  - warnings/cautions
  - figures/captions
  - tools/supplies/internal references
  ↓
Retrieval Layer
  - DMC direct lookup
  - graph-expanded metadata retrieval
  - vector search
  - keyword/BM25 search where useful
  - reranking
  ↓
Structured Answer Plan
  - claims
  - evidence per claim
  - required citations
  - forbidden unsupported claims
  ↓
LLM Verbalizer
  - grounded Korean explanation
  - detail-level adaptation
  - no unsupported facts
  ↓
Quality Gate
  - DMC grounding
  - support-level correctness
  - warning/caution preservation
  - no fabricated procedures/tools
  - clean answer/reference/UI separation
```

Core principle:

```text
The ontology controls what can be said.
RAG retrieves the source evidence.
The LLM explains the evidence.
The quality gate decides whether the answer is safe to show.
```

Current RDF/OWL transition utilities:

```bash
# Export the ontology manifest as Turtle/JSON-LD for GraphDB/Fuseki-compatible loading.
python scripts/export_ontology_rdf.py --output data/ontology/s1000d.ttl
python scripts/export_ontology_rdf.py --format jsonld --output data/ontology/s1000d.jsonld

# Validate the ontology manifest against local SHACL-like shape rules.
python scripts/validate_ontology_shapes.py

# Run v4 locally with the in-memory RDF resolver.
S1000D_RAG_PIPELINE=v4 python query.py "브레이크 작동원리를 자세히 설명해줘"

# Optional future backend: route v4 RDF resolution to a SPARQL endpoint.
S1000D_RAG_PIPELINE=v4 \
S1000D_SPARQL_ENDPOINT=http://localhost:7200/repositories/s1000d \
python query.py "브레이크 패드 청소 절차 알려줘"
```

---

## 6. Representative demo questions

Useful interview/demo questions:

```text
브레이크 작동원리를 자세히 설명해줘
브레이크 시스템의 주요 구성품 알려줘
브레이크 패드 청소 절차 알려줘
브레이크 케이블 제거 후 다시 설치하는 절차가 있나?
바퀴 교체 방법 알려줘
앞바퀴 설치 절차 알려줘
체인에 오일 바르는 방법 알려줘
조명 시스템 점검 방법 알려줘
앞에서 알려준 문서 내용은 뭔데?
```

These questions are intentionally chosen because they test more than semantic similarity:

- operation principle explanation;
- component listing;
- exact procedure retrieval;
- unsupported or related procedure handling;
- partial support reasoning;
- subsystem routing;
- conversational evidence memory;
- DMC citation quality.

---

## 7. Repository guide

Key documentation:

```text
docs/architecture.md
  Original S1000D local RAG architecture and XML/chunking design.

docs/retrospectives/rag-v1-failure-analysis.md
  Detailed analysis of vector/guard-heavy RAG failures.

docs/retrospectives/rag-evolution-v1-to-v4.md
  Portfolio narrative explaining v1 → v2 → v3 → v4.

docs/plans/ontology-first-rag-v2-rewrite.md
  Earlier rewrite plan that led to the current deterministic ontology-first baseline.

docs/plans/ontology-guided-graph-rag-v4.md
  Final target architecture and implementation roadmap.

docs/local_model_stack.md
  Local GGUF/embedding/reranker/VLM model stack and environment setup notes.

docs/demo_golden_questions.md
  Demo and regression question set.
```

Key runtime/code areas:

```text
app_web.py
  FastAPI web app entrypoint.

query.py
  CLI query entrypoint.

src/rag/pipeline_v2.py
  Current ontology-first runtime path, best understood as v3 baseline.

src/rag/ontology/
  Query parsing, ontology schema, resolver, evidence planning, answer composer, quality gate.

src/rag/pipeline.py
  Legacy guard-heavy pipeline. Should not be expanded for future product behavior.
```

---

## 8. Local model / closed-network emphasis

This project is designed with closed-network/on-prem deployment in mind.

Local stack includes:

- GGUF text LLM through llama.cpp / llama-cpp-python;
- local BGE embedding model;
- local reranker;
- Chroma vector database;
- optional local VLM assets for future figure/graphic understanding;
- no dependency on public cloud inference for core operation.

This matters for defense, MRO, manufacturing, and technical support scenarios where maintenance manuals, procedures, and equipment data cannot be sent to external APIs.

---

## 9. Engineering lessons highlighted by this project

1. A full vector index does not guarantee a correct technical assistant.
2. S1000D structure is a retrieval control signal, not just metadata.
3. “Not found” must be based on support-level reasoning, not only vector score thresholds.
4. Guard rules can stabilize demos but become technical debt if they replace domain modeling.
5. Local LLMs need structured evidence plans and quality gates.
6. Ontology and graph relationships are the right control plane for technical-document RAG.
7. The final system should combine symbolic control with neural language generation.

---

## 10. Current status

Current project status:

```text
Current stable direction: v3 deterministic ontology-first baseline
Final implementation target: v4 ontology-guided Graph RAG + LLM synthesis
Portfolio narrative: documented
Next engineering milestone: implement v4 graph schema, answer planner, LLM verbalizer, and stronger quality gate
```

The project should be evaluated as an evolving engineering system: the earlier approaches are preserved as documented lessons, and the current roadmap explicitly targets a more rigorous ontology-guided LLM chatbot architecture.
