from src.rag.pipeline_v2 import run_rag_query_sync
from src.rag.ontology import check_answer_quality

CASES = [
    ("브레이크 시스템에 대해 알려줘", ["BRAKE-AAA-DA1-00-00-00AA-041A-A"], ["브레이크 시스템"]),
    ("브레이크 시스템 주요 구성품 알려줘", [], ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"]),
    ("브레이크 패드 청소 절차 알려줘", ["BRAKE-AAA-DA1-10-00-00AA-251A-A"], []),
    ("브레이크 케이블 제거 후 다시 설치하는 방법은?", [], ["정확한", "절차", "찾지 못했습니다"]),
    ("앞바퀴 설치 절차 알려줘", ["S1000DBIKE-AAA-DA0-30-00-00AA-720A-A"], []),
    ("바퀴 교체 방법 알려줘", [], ["바퀴 자체", "단일 절차", "타이어", "휠"]),
    ("체인에 오일 바르는 방법 알려줘", ["S1000DBIKE-AAA-DA4-10-00-00AA-241A-A"], []),
    ("핸들바 탈거 방법 알려줘", ["S1000DBIKE-AAA-DA2-20-00-00AA-520A-A"], []),
    ("조명 시스템 점검 방법 알려줘", ["S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A"], []),
]


def test_pipeline_v2_initial_regressions():
    for question, dmcs, must_contain in CASES:
        result = run_rag_query_sync(question)
        evidence_dmcs = [e.dmc for e in result.evidences]
        for dmc in dmcs:
            assert dmc in evidence_dmcs or dmc in result.answer
        for text in must_contain:
            assert text in result.answer
        assert check_answer_quality(result.answer).ok


def test_lighting_regression_not_stem_or_chain_route():
    result = run_rag_query_sync("조명 시스템 점검 방법 알려줘")
    assert "S1000DLIGHTING-AAA-D00-00-00-00AA-341A-A" in result.answer or any("LIGHTING" in e.dmc for e in result.evidences)
    assert not any("DA2" in e.dmc or "DA4" in e.dmc for e in result.evidences)


def test_follow_up_summarizes_previous_evidence_dmc():
    history = [("앞바퀴 설치 절차 알려줘", "근거 DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A")]
    result = run_rag_query_sync("알려준 문서 내용은 뭔데?", conversation_history=history)
    assert "S1000DBIKE-AAA-DA0-30-00-00AA-720A-A" in result.answer
    assert "문서" in result.answer
