# S1000D-RAG Demo Script and Golden Questions

This script is intended for browser demos after the 100-case and 500-case QA loops have passed.

## Demo goals

Show that the app can:

1. Answer supported S1000D maintenance questions in Korean.
2. Keep the answer bubble clean while moving DMC/evidence into reference panels.
3. Say that information is not available when the procedure is not present.
4. Distinguish descriptive data modules from procedural data modules.
5. Render ontology-backed reference materials and graphic/figure preview metadata.

## Recommended flow

### 1. Clean answer + procedure evidence

Question:

```text
브레이크 패드 청소 절차를 알려줘
```

Expected demo points:

- Answer mentions visual inspection, locating brake pads, rubbing alcohol, and clean cloth.
- Answer must not say `오일`.
- Evidence should include `BRAKE-AAA-DA1-10-00-00AA-251A-A` or `S1000DBIKE-AAA-DA1-10-00-00AA-251A-A`.
- `참고자료` should show related procedure material.

### 2. Manual test retrieval

Question:

```text
브레이크 시험 방법은?
```

Expected demo points:

- Answer explains standing the bicycle, holding the handlebars, applying brakes, and checking wheel lock/stop.
- Evidence should include `BRAKE-AAA-DA1-00-00-00AA-341A-A` or S1000D bike equivalent.
- Demonstrates graph-first mapping from generic `브레이크 시험` to brake-system manual test.

### 3. Descriptive component explanation

Question:

```text
브레이크 관련 설명 문서에는 어떤 내용이 있나요?
```

Expected demo points:

- Answer describes brake system components: lever, cable, arm, clamp/calliper, pads.
- Evidence should include `BRAKE-AAA-DA1-00-00-00AA-041A-A`.
- Demonstrates deterministic guard for fragile small-model empty answers.

### 4. Cable detail in Korean

Question:

```text
브레이크 케이블을 조금 더 자세히 설명해줘
```

Expected demo points:

- Answer stays in Korean.
- Answer does not copy English source text.
- Evidence should include `BRAKE-AAA-DA1-00-00-00AA-041A-A`.

### 5. No-answer guardrail

Question:

```text
브레이크 케이블 교체 절차가 없다면 관련 설명만 알려줘
```

Expected demo points:

- Answer should not invent a cable replacement procedure.
- It should say the replacement procedure is not confirmed and only provide related descriptive information.

### 6. DMC lookup

Question:

```text
브레이크 패드 청소 문서의 DMC를 알려줘
```

Expected demo points:

- Answer should directly provide the DMC.
- This is allowed to include DMC in the answer because the user explicitly asked for it.

### 7. Unsupported procedure

Question:

```text
앞바퀴 설치 절차가 있으면 알려줘
```

Expected demo points:

- Answer should not invent a front-wheel installation procedure if the exact procedure is not in the provided corpus.
- It may list related candidate documents only when grounded.

## Browser QA checklist

For each demo question:

- [ ] Answer bubble is clean and user-facing.
- [ ] No `근거:` / `참고 문서:` metadata line leaks unless the user explicitly asked for DMC.
- [ ] Evidence cards are visible under `참고 문서`.
- [ ] Ontology-backed `참고자료` renders separately.
- [ ] Figures/graphic assets show preview status or fallback when available.
- [ ] Browser console has no JavaScript errors.
- [ ] The answer does not contradict the retrieved evidence.

## Known long-run note

A one-time llama.cpp decode slot error was observed in an earlier long loop. `/api/chat` now retries once for known transient decode slot failures. If it recurs frequently, collect the server log and run directory before changing model/runtime settings.
