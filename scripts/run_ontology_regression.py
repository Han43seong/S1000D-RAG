#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from src.rag.pipeline_v2 import run_rag_query_sync
from src.rag.ontology import check_answer_quality, parse_query, resolve_ontology, load_ontology_manifest

REGRESSION_CASES = [
    {"question": "브레이크 시스템에 대해 알려줘", "support": "exact", "expected_dmcs": ["BRAKE-AAA-DA1-00-00-00AA-041A-A"], "forbidden": ["브레이크 버", "레이크 이블", "레이크 드", "곽", "도를 입니다"]},
    {"question": "브레이크 시스템 주요 구성품 알려줘", "support": "exact", "must_contain": ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"]},
    {"question": "브레이크 패드 청소 절차 알려줘", "support": "exact", "expected_dmcs": ["BRAKE-AAA-DA1-10-00-00AA-251A-A"]},
    {"question": "브레이크 케이블 제거 후 다시 설치하는 방법은?", "support": "related", "must_contain": ["정확한", "절차", "찾지 못했습니다"]},
    {"question": "앞바퀴 설치 절차 알려줘", "support": "exact", "expected_dmcs": ["S1000DBIKE-AAA-DA0-30-00-00AA-720A-A"]},
    {"question": "바퀴 교체 방법 알려줘", "support": "partial", "must_contain": ["바퀴 자체", "단일 절차", "타이어", "휠"]},
    {"question": "체인에 오일 바르는 방법 알려줘", "support": "exact", "expected_dmcs": ["S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"]},
    {"question": "핸들바 탈거 방법 알려줘", "support": "exact", "expected_dmcs": ["S1000DBIKE-AAA-DA2-20-00-00AA-520A-A"]},
    {"question": "조명 시스템 점검 방법 알려줘", "support": "exact", "expected_dmcs": ["S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A"], "forbidden": ["DA2", "DA4"]},
]


def main() -> int:
    nodes = load_ontology_manifest()
    failures: list[str] = []
    for case in REGRESSION_CASES:
        question = case["question"]
        resolution = resolve_ontology(parse_query(question), nodes)
        result = run_rag_query_sync(question)
        got_dmcs = [e.dmc for e in result.evidences]
        if resolution.support.value != case["support"]:
            failures.append(f"{question}: support {resolution.support.value} != {case['support']}")
        for dmc in case.get("expected_dmcs", []):
            if dmc not in got_dmcs and dmc not in result.answer:
                failures.append(f"{question}: missing DMC {dmc}")
        for text in case.get("must_contain", []):
            if text not in result.answer:
                failures.append(f"{question}: missing text {text}")
        for text in case.get("forbidden", []):
            if text in result.answer:
                failures.append(f"{question}: forbidden text present {text}")
        gate = check_answer_quality(result.answer)
        if not gate.ok:
            failures.append(f"{question}: quality gate failed {gate.reasons}")
        print(f"PASS {question} -> {resolution.support.value}: {result.answer.splitlines()[0]}")
    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
