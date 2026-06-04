from langchain_core.documents import Document

from src.rag.ontology import DetailLevel, Intent, ParsedQuery
from src.rag.v4.answer_plan import AnswerClaim, AnswerPlan, build_answer_plan
from src.rag.v4.graph_schema import GraphEdge, GraphNode, NodeType, RelationType
from src.rag.v4.verbalizer import verbalize_answer_plan


def test_v4_graph_schema_models_s1000d_relationships():
    brake = GraphNode(id="system:brake", node_type=NodeType.SYSTEM, label="Brake system")
    pad = GraphNode(id="component:brake-pad", node_type=NodeType.COMPONENT, label="Brake pad")
    edge = GraphEdge(
        source_id=brake.id,
        relation=RelationType.HAS_COMPONENT,
        target_id=pad.id,
        source_dmc="BRAKE-AAA-DA1-00-00-00AA-041A-A",
    )

    assert edge.relation == RelationType.HAS_COMPONENT
    assert edge.source_dmc == "BRAKE-AAA-DA1-00-00-00AA-041A-A"


def test_v4_answer_plan_maps_claims_to_evidence_and_forbidden_claims():
    parsed = ParsedQuery(
        original="브레이크 작동원리를 정비사가 이해할 수 있게 자세히 설명해줘",
        normalized="브레이크 작동원리를 정비사가 이해할 수 있게 자세히 설명해줘",
        intent=Intent.DESCRIBE,
        target="brake system",
        detail_level=DetailLevel.DETAILED,
    )
    docs = [
        Document(
            page_content="Brake cable transmits force from the brake lever to the brake arms. Brake pads press the wheel rim.",
            metadata={"dmc": "BRAKE-AAA-DA1-00-00-00AA-041A-A", "title": "Brake system description"},
        )
    ]

    plan = build_answer_plan(parsed, docs)

    assert isinstance(plan, AnswerPlan)
    assert plan.detail_level == DetailLevel.DETAILED
    assert plan.claims
    assert all(claim.evidence_dmcs for claim in plan.claims)
    assert "unsupported procedure steps" in plan.forbidden_claims


def test_v4_verbalizer_uses_llm_for_synthesis_but_keeps_grounding_contract():
    plan = AnswerPlan(
        query="브레이크 작동원리 자세히",
        intent=Intent.DESCRIBE,
        detail_level=DetailLevel.DETAILED,
        audience="technician",
        claims=(
            AnswerClaim(
                text="브레이크 레버의 힘은 케이블을 통해 브레이크 암으로 전달된다.",
                evidence_dmcs=("BRAKE-AAA-DA1-00-00-00AA-041A-A",),
            ),
        ),
        required_citations=("BRAKE-AAA-DA1-00-00-00AA-041A-A",),
        forbidden_claims=("unsupported procedure steps",),
        sections=("작동 흐름", "정비상 의미"),
    )

    class FakeLLM:
        def invoke(self, prompt):
            assert "unsupported procedure steps" in prompt
            assert "BRAKE-AAA-DA1-00-00-00AA-041A-A" in prompt
            return "브레이크 레버의 힘은 케이블을 통해 브레이크 암으로 전달됩니다.\n근거 DMC: BRAKE-AAA-DA1-00-00-00AA-041A-A"

    answer = verbalize_answer_plan(plan, llm=FakeLLM())

    assert "근거 DMC: BRAKE-AAA-DA1-00-00-00AA-041A-A" in answer
