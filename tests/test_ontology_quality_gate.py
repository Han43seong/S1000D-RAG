from src.rag.ontology import check_answer_quality


def test_rejects_known_bad_artifacts():
    for bad in ["브레이크 버", "레이크 이블", "레이크 드", "곽", "도를 입니다", "DM0000000000", "<한국어 답변>", "<think>", "</think>"]:
        assert not check_answer_quality(f"문장 {bad}").ok


def test_rejects_repeated_blocks_and_context_header():
    assert not check_answer_quality("답변: a\n답변: b").ok
    assert not check_answer_quality("DMC: A\nDMC: B").ok
    assert not check_answer_quality("[DMC: ABC | Type: procedural]").ok


def test_accepts_clean_answer():
    assert check_answer_quality("브레이크 시스템 설명입니다.\n근거 DMC: BRAKE-AAA-DA1-00-00-00AA-041A-A").ok
