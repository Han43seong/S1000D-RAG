from src.rag.ontology import load_ontology_manifest, parse_query, resolve_ontology
from src.rag.ontology.schema import SupportLevel


def _resolve(q):
    return resolve_ontology(parse_query(q), load_ontology_manifest())


def test_exact_front_wheel_install():
    result = _resolve("앞바퀴 설치 절차 알려줘")
    assert result.support == SupportLevel.EXACT
    assert result.candidates[0].node.dmc == "S1000DBIKE-AAA-DA0-30-00-00AA-720A-A"


def test_wheel_replacement_partial():
    result = _resolve("바퀴 교체 방법 알려줘")
    assert result.support == SupportLevel.PARTIAL
    assert any(c.node.target == "tire" for c in result.candidates)


def test_brake_cable_related_not_none():
    result = _resolve("브레이크 케이블 제거 후 다시 설치하는 방법은?")
    assert result.support == SupportLevel.RELATED
