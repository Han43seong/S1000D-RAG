from src.rag.ontology import Intent, DetailLevel
from src.rag.v4.answer_plan import AnswerClaim, AnswerPlan
from src.rag.v4.korean_composer import compose_korean_draft


def test_korean_composer_builds_descriptive_answer_without_internal_metadata():
    plan = AnswerPlan(
        query="브레이크 시스템 원리에 대해 설명해줘",
        intent=Intent.DESCRIBE,
        detail_level=DetailLevel.DETAILED,
        audience="technician",
        claims=(
            AnswerClaim(
                text="The brake system has these primary components: the brake lever the brake cable the brake arm the brake clamp also known as callipers the brake pads.",
                evidence_dmcs=("BRAKE-DESC",),
            ),
            AnswerClaim(
                text="This presses the brake pads against the outer rim of the wheel, which decreases the speed of the bicycle.",
                evidence_dmcs=("BRAKE-DESC",),
            ),
        ),
        required_citations=("BRAKE-DESC",),
        forbidden_claims=("unsupported procedure steps",),
        sections=("구성 관계", "작동 흐름"),
    )

    draft = compose_korean_draft(plan)

    assert draft.startswith("브레이크 시스템에 대해")
    assert "브레이크 레버" in draft
    assert "속도가 줄어듭니다" in draft
    assert "BRAKE-DESC" not in draft
    assert "The brake system" not in draft
    assert "[DMC:" not in draft


def test_korean_composer_builds_procedure_answer_with_numbered_units():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(text="Install the fork before installing the wheel. Hold the front of the bicycle.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Put the bike on the floor.", evidence_dmcs=("DMC-WHEEL",)),
        ),
        required_citations=("DMC-WHEEL",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    draft = compose_korean_draft(plan)

    assert draft.splitlines()[0] == "앞바퀴 설치 절차는 다음과 같습니다."
    assert "1. 바퀴를 설치하기 전에 포크가 먼저 장착되어 있는지 확인합니다." in draft
    assert "2. 자전거 앞부분을 안정적으로 잡습니다." in draft
    assert "3. 자전거를 바닥에 내려놓습니다." in draft
    assert "근거 DMC" not in draft


def test_korean_composer_preserves_more_procedure_units_than_descriptive_default():
    plan = AnswerPlan(
        query="앞바퀴 설치 절차 알려줘",
        intent=Intent.PROCEDURE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(text="Install the fork and the brakes before installing the wheel. Hold the front of the bicycle.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Install the wheel and be careful to not damage the chainring.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Close the light circuit breaker located on the handlebar. Put the bike on the floor.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Open the light circuit breaker located on the handlebar. Disengage the fork from the chainring.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Pushing the wheel forwards and down. Use specific oil if the fork do not desengage easily.", evidence_dmcs=("DMC-WHEEL",)),
            AnswerClaim(text="Lift the wheel away from the frame. If not available, use any oil compliant with requirements.", evidence_dmcs=("DMC-WHEEL",)),
        ),
        required_citations=("DMC-WHEEL",),
        forbidden_claims=("fabricated step sequence",),
        sections=("절차",),
    )

    draft = compose_korean_draft(plan)

    assert "9. 바퀴를 앞으로 밀고 아래로 내려 작업합니다." in draft
    assert "10. 포크가 쉽게 분리되지 않으면 지정된 오일을 사용합니다." in draft
    assert "11. 바퀴를 프레임에서 들어 올려 분리합니다." in draft
    assert "12. 지정 오일이 없으면 요구사항을 만족하는 오일을 사용합니다." in draft


def test_korean_composer_answers_symptom_query_as_possible_checks_not_fixed_diagnosis():
    plan = AnswerPlan(
        query="바퀴가 잘 안 움직여",
        intent=Intent.DESCRIBE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(text="The pads press against the rim of the wheel to cause friction when the you operate the brake levers.", evidence_dmcs=("BRAKE-DESC",)),
            AnswerClaim(text="Install the fork and the brakes before installing the wheel.", evidence_dmcs=("WHEEL-INSTALL",)),
        ),
        required_citations=("BRAKE-DESC", "WHEEL-INSTALL"),
        forbidden_claims=("unsupported diagnosis",),
        sections=("증상 해석", "우선 확인 항목"),
    )

    draft = compose_korean_draft(plan)

    assert draft.startswith("바퀴가 잘 안 움직이는 증상은 문서 근거상 다음 항목과 관련해 우선 확인할 수 있습니다.")
    assert "브레이크 패드가 바퀴의 바깥쪽 림을 누릅니다." in draft
    assert "바퀴를 설치하기 전에 포크와 브레이크가 먼저 장착되어 있는지 확인합니다." in draft
    assert "문서 근거만으로 특정 고장 원인을 확정할 수는 없습니다." in draft
    assert "고장났" not in draft
    assert "교체" not in draft
