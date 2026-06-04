from src.rag.ontology import compose_answer, load_ontology_manifest, parse_query, resolve_ontology


def answer(q):
    return compose_answer(resolve_ontology(parse_query(q), load_ontology_manifest()))


def test_brake_components_are_clean_korean_terms():
    text = answer("브레이크 시스템 주요 구성품 알려줘")
    for term in ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"]:
        assert term in text
    assert "레이크 이블" not in text


def test_brake_principle_detail_answer_is_richer_than_one_line_summary():
    text = answer("브레이크 작동원리를 더 자세히 설명해줘")
    for term in ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드", "휠 림", "마찰"]:
        assert term in text
    assert "조정 잠금 너트" in text
    assert "근거 DMC: BRAKE-AAA-DA1-00-00-00AA-041A-A" in text
    assert len(text) > 220


def test_partial_wheel_answer_explains_decomposition():
    text = answer("바퀴 교체 방법 알려줘")
    for term in ["바퀴 자체", "단일 절차", "타이어", "휠"]:
        assert term in text


def test_related_brake_cable_answer_states_exact_not_found():
    text = answer("브레이크 케이블 제거 후 다시 설치하는 방법은?")
    assert "정확한" in text and "찾지 못했습니다" in text
