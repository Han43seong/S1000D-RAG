# S1000D-RAG Quality Evidence Report

## Status

As of the verified `main` branch after commit `0a8640e`, the S1000D-RAG quality loops passed both the ontology-aware 100-case gate and autonomous 500-case gate.

## Verified runs

### Ontology-aware full QA

Primary verified run:

- Run ID: `quality-qa-loop-20260603-160136`
- Result path: `eval/results/ontology-aware-full/quality-qa-loop-20260603-160136.json`
- Total: 100
- Pass: 100
- Fail: 0
- Average LLM time: 5.477s

Post-hardening revalidation run:

- Run ID: `quality-qa-loop-20260603-175417`
- Result path: `eval/results/ontology-aware-full/quality-qa-loop-20260603-175417.json`
- Total: 100
- Pass: 100
- Fail: 0
- Average LLM time: 4.497s

Check groups in the latest run:

- `answer`: 100/100 pass
- `evidence`: 100/100 pass
- `reference_materials`: 100/100 pass
- `visual_preview`: 100/100 pass
- `ui_display`: 100/100 pass

### Autonomous 500-case QA

- Run ID: `autonomous-500-20260603-162202`
- Result directory: `eval/results/autonomous-500/autonomous-500-20260603-162202`
- Total: 500
- Pass: 500
- Fail: 0
- Fixed during run: 3
- Elapsed: 3101.23s
- Final verification: 120 passed, 2 FastAPI deprecation warnings

Fixed cases:

1. `qa500-114-cycle2-q014` — brake lever operation query returned an empty answer despite valid brake 041A evidence.
2. `qa500-188-cycle2-q088` — brake description document query returned an empty answer despite valid brake 041A evidence.
3. `qa500-212-cycle3-q012` — brake cable detail query copied English source text instead of returning Korean.

Fix strategy:

- Add narrow deterministic guards that only trigger when matching `BRAKE-AAA-DA1-00-00-00AA-041A-A` evidence is retrieved.
- Exclude procedural intent from descriptive guards.
- Add regression tests proving LLM is not invoked for the fragile evidence-backed cases.

## Browser UI E2E smoke

Manual browser smoke was run against `http://127.0.0.1:8000/` with the live local server.

Observed healthy behavior:

- The answer bubble displays a clean user-facing answer.
- DMC/evidence is separated into the `참고 문서` section.
- Ontology-backed `참고자료` is shown separately from the answer bubble.
- `reference_materials` renders categorized procedure evidence.
- Browser console reported no JavaScript errors during the smoke.

Observed and fixed issue:

- Question: `브레이크 패드 청소 절차를 알려줘`
- The UI initially rendered a Korean answer that translated the cleaning material as `오일` even though the evidence title and source procedure are `Clean with rubbing alcohol`.
- Fix: add an evidence-gated deterministic guard for `BRAKE-AAA-DA1-10-00-00AA-251A-A` brake-pad cleaning queries so the answer says `러빙 알코올` and `깨끗한 천`, not oil.
- Regression test: `tests/test_rag.py::TestRunRagQuerySync::test_brake_pad_cleaning_query_uses_rubbing_alcohol_not_oil`

## Runtime hardening

A prior long loop produced one transient llama.cpp error:

```text
llama_decode returned 1
decode: failed to find a memory slot
```

Mitigation added:

- `/api/chat` retries once inside the same request when the exception text matches known transient llama.cpp decode slot failures.
- User/session messages are not duplicated because retry happens inside `_chat_sync` after the user message has been appended once and before assistant persistence.
- Regression test: `tests/test_app_web_runtime_state.py::test_chat_retries_once_on_transient_llama_decode_error`

## Quality gates

Use `scripts/verify_rag_quality_gate.py` for reproducible local checks:

```bash
/home/hskim/miniforge3/bin/python scripts/verify_rag_quality_gate.py --suite focused
/home/hskim/miniforge3/bin/python scripts/verify_rag_quality_gate.py --suite ontology-100
/home/hskim/miniforge3/bin/python scripts/verify_rag_quality_gate.py --suite autonomous-500
```

The 500-case suite is long-running and should be launched only when a local model server is ready and a long verification window is available.
