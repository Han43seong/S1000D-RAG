from typing import Any, cast

from langchain_core.documents import Document

from src.rag.ontology import DetailLevel, Intent, ParsedQuery, SupportLevel
from src.rag.v4.answer_plan import AnswerClaim, AnswerPlan, build_answer_plan
from src.rag.v4.graph_schema import GraphEdge, GraphNode, NodeType, RelationType
from src.rag.v4.rdf_resolver import RdfResolution
from src.rag.v4.verbalizer import build_verbalizer_prompt, verbalize_answer_plan


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


def test_v4_answer_plan_splits_claims_and_preserves_evidence_metadata():
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
            metadata={
                "dmc": "BRAKE-DESC",
                "title": "Brake system description",
                "structure_path": "description/para[2]",
                "source_file": "DMC-BRAKE-DESC.XML",
            },
        )
    ]
    rdf_resolution = RdfResolution(primary_dmcs=("BRAKE-DESC",), related_dmcs=())

    plan = build_answer_plan(parsed, docs, rdf_resolution=rdf_resolution)

    assert plan.support_level == SupportLevel.EXACT
    assert [claim.text for claim in plan.claims] == [
        "Brake cable transmits force from the brake lever to the brake arms.",
        "Brake pads press the wheel rim.",
    ]
    assert all(claim.support_level == SupportLevel.EXACT for claim in plan.claims)
    assert plan.claims[0].evidence_blocks == ("description/para[2]",)
    assert plan.claims[0].source_titles == ("Brake system description",)
    assert plan.claims[0].source_files == ("DMC-BRAKE-DESC.XML",)


def test_v4_answer_plan_marks_related_only_procedure_as_unsupported_for_step_synthesis():
    parsed = ParsedQuery(
        original="브레이크 케이블 제거 후 재설치 절차 알려줘",
        normalized="브레이크 케이블 제거 후 재설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake cable",
        action="remove and install",
    )
    docs = [Document(page_content="Brake cable routing is described.", metadata={"dmc": "BRAKE-DESC"})]
    rdf_resolution = RdfResolution(primary_dmcs=(), related_dmcs=("BRAKE-DESC",))

    plan = build_answer_plan(parsed, docs, rdf_resolution=rdf_resolution)

    assert plan.support_level == SupportLevel.RELATED
    assert "unsupported requested procedure" in plan.forbidden_claims
    assert "fabricated step sequence" in plan.forbidden_claims
    assert any("직접 확인되지 않았습니다" in claim.text for claim in plan.claims)
    fallback = verbalize_answer_plan(plan)
    assert fallback.count("Brake cable routing is described.") == 1
    assert "[관련 근거]" in fallback


def test_v4_answer_plan_includes_rdf_primary_and_related_citations():
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )
    docs = [Document(page_content="Clean brake pad with approved material.", metadata={"dmc": "BRAKE-PAD-CLEAN"})]
    rdf_resolution = RdfResolution(primary_dmcs=("BRAKE-PAD-CLEAN",), related_dmcs=("BRAKE-DESC",))

    plan = build_answer_plan(parsed, docs, rdf_resolution=rdf_resolution)

    assert plan.required_citations == ("BRAKE-PAD-CLEAN", "BRAKE-DESC")


def test_v4_answer_plan_does_not_expand_citations_with_legacy_graph_when_rdf_resolution_exists():
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )
    docs = [Document(page_content="Clean brake pad with approved material.", metadata={"dmc": "BRAKE-PAD-CLEAN"})]
    rdf_resolution = RdfResolution(primary_dmcs=("BRAKE-PAD-CLEAN",), related_dmcs=("BRAKE-DESC",))

    class LegacyGraphContext:
        def related_dmcs_for_target(self, _target):
            return ("LEGACY-BROAD-DMC", "BRAKE-PAD-CLEAN")

    plan = build_answer_plan(parsed, docs, graph_context=cast(Any, LegacyGraphContext()), rdf_resolution=rdf_resolution)

    assert plan.required_citations == ("BRAKE-PAD-CLEAN", "BRAKE-DESC")
    assert "LEGACY-BROAD-DMC" not in plan.required_citations


def test_v4_answer_plan_includes_rdf_graph_paths_for_explainability():
    parsed = ParsedQuery(
        original="브레이크 패드 청소 절차 알려줘",
        normalized="브레이크 패드 청소 절차 알려줘",
        intent=Intent.PROCEDURE,
        target="brake pad",
        action="clean",
    )
    docs = [Document(page_content="Clean brake pad with approved material.", metadata={"dmc": "BRAKE-PAD-CLEAN"})]
    rdf_resolution = RdfResolution(
        primary_dmcs=("BRAKE-PAD-CLEAN",),
        related_dmcs=("BRAKE-DESC",),
        graph_paths=(
            "BRAKE-PAD-CLEAN -[s1000d:hasTarget]-> brake pad",
            "BRAKE-DESC -[s1000d:describes]-> brake system",
        ),
    )

    plan = build_answer_plan(parsed, docs, rdf_resolution=rdf_resolution)
    prompt = build_verbalizer_prompt(plan)
    fallback = verbalize_answer_plan(plan)

    assert plan.graph_paths == rdf_resolution.graph_paths
    assert "RDF graph paths" in prompt
    assert "BRAKE-PAD-CLEAN -[s1000d:hasTarget]-> brake pad" in prompt
    assert "AnswerPlan을 재출력하지 마세요" in prompt
    assert "최종 답변만 한국어로 작성하세요" in prompt
    assert "[온톨로지 선택 근거]" in fallback
    assert "BRAKE-DESC -[s1000d:describes]-> brake system" in fallback


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


def test_v4_verbalizer_strips_prompt_contract_leakage_from_llm_output():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(
                text="앞바퀴를 설치하기 전에 포크와 브레이크를 먼저 설치해야 합니다.",
                evidence_dmcs=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
                source_titles=("Front wheel - Install procedures",),
            ),
        ),
        required_citations=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    class LeakyLLM:
        def invoke(self, _prompt):
            return (
                "AnswerPlan:[근거 기반 설명] It is necessary to install the fork and the brakes before installing the wheel. "
                "[DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A; support: exact; titles: Front wheel - Install procedures]"
                "Answer:AnswerPlan에 따라, 앞바퀴 설치 절차는 다음과 같습니다: 포크와 브레이크를 먼저 설치하세요."
            )

    answer = verbalize_answer_plan(plan, llm=LeakyLLM())

    assert answer.startswith("앞바퀴 설치 절차는 다음과 같습니다")
    assert "AnswerPlan" not in answer
    assert "Answer:" not in answer
    assert "required citations" not in answer
    assert "forbidden claims" not in answer
    assert "titles:" not in answer
    assert "근거 DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A" in answer


def test_v4_verbalizer_strips_claim_labels_and_inline_metadata_from_llm_output():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(
                text="앞바퀴를 설치하기 전에 포크와 브레이크를 먼저 설치해야 합니다.",
                evidence_dmcs=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
            ),
        ),
        required_citations=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    class MetadataEchoLLM:
        def invoke(self, _prompt):
            return (
                "[근거 기반 설명] It is necessary to install the fork and the brakes before installing the wheel. "
                "[DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A; support: exact; titles: Front wheel - Install procedures] "
                "[근거 기반 설명] Put the bike on the floor."
            )

    answer = verbalize_answer_plan(plan, llm=MetadataEchoLLM())

    assert "[근거 기반 설명]" not in answer
    assert "[DMC:" not in answer
    assert "support:" not in answer
    assert "titles:" not in answer
    assert "포크와 브레이크" in answer
    assert "It is necessary" not in answer
    assert "근거 DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A" in answer


def test_v4_verbalizer_strips_thinking_tail_from_llm_output():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(AnswerClaim(text="Install the fork before installing the wheel.", evidence_dmcs=("DMC-FRONT-WHEEL",)),),
        required_citations=("DMC-FRONT-WHEEL",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    class ThinkingTailLLM:
        def invoke(self, _prompt):
            return "1. Install the fork before installing the wheel.\n]\nOkay, let me think through the AnswerPlan."

    answer = verbalize_answer_plan(plan, llm=ThinkingTailLLM())

    assert "]" not in answer
    assert "Okay" not in answer
    assert "let me think" not in answer
    assert "AnswerPlan" not in answer
    assert "포크" in answer
    assert "Install the fork" not in answer
    assert "근거 DMC: DMC-FRONT-WHEEL" in answer


def test_v4_verbalizer_rewrites_english_evidence_to_clean_korean_user_answer():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(
                text="Install the fork and the brakes before installing the wheel. Hold the front of the bicycle.",
                evidence_dmcs=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
            ),
            AnswerClaim(
                text="Install the wheel and be careful to not damage the chainring.",
                evidence_dmcs=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
            ),
            AnswerClaim(
                text="Put the bike on the floor.",
                evidence_dmcs=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
            ),
        ),
        required_citations=("S1000DBIKE-AAA-DA0-30-00-00AA-720A-A",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    class EnglishLLM:
        def invoke(self, _prompt):
            return (
                "1. Install the fork and the brakes before installing the wheel. Hold the front of the bicycle.\n"
                "2. Install the wheel and be careful to not damage the chainring.\n"
                "3. Put the bike on the floor."
            )

    answer = verbalize_answer_plan(plan, llm=EnglishLLM())

    assert answer.startswith("앞바퀴 설치 절차는 다음과 같습니다")
    assert "포크와 브레이크" in answer
    assert "체인링" in answer
    assert "자전거를 바닥에" in answer
    assert "합니다고" not in answer
    assert "Install the fork" not in answer
    assert "Put the bike on the floor" not in answer
    assert "AnswerPlan" not in answer
    assert "[DMC:" not in answer
    assert "근거 DMC: S1000DBIKE-AAA-DA0-30-00-00AA-720A-A" in answer
