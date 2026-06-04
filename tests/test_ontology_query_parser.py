from src.rag.ontology import parse_query
from src.rag.ontology.schema import Intent


def test_lighting_system_does_not_match_stem():
    parsed = parse_query("조명 시스템 점검 방법 알려줘")
    assert parsed.intent == Intent.PROCEDURE
    assert parsed.target == "lights"
    assert parsed.action == "test"
    assert "스템" not in parsed.matched_aliases


def test_front_wheel_install_parse():
    parsed = parse_query("앞바퀴 설치 절차 알려줘")
    assert parsed.target == "front wheel"
    assert parsed.action == "install"


def test_component_list_parse():
    parsed = parse_query("브레이크 시스템 주요 구성품 알려줘")
    assert parsed.intent == Intent.LIST_COMPONENTS
    assert parsed.target == "brake system"
