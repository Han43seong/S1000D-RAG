import json

from src.rag.ontology import DetailLevel, Intent
from src.rag.v4.answer_plan import AnswerClaim, AnswerPlan
from src.rag.v4.grounded_generator import build_evidence_packet, generate_grounded_answer
from src.rag.v4.verbalizer import verbalize_answer_plan


class RecordingLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return self.response


def _wheel_symptom_plan() -> AnswerPlan:
    return AnswerPlan(
        query="바퀴가 잘 안 움직여",
        intent=Intent.DESCRIBE,
        detail_level=DetailLevel.NORMAL,
        audience="technician",
        claims=(
            AnswerClaim(
                text="The pads press against the rim of the wheel to cause friction when the you operate the brake levers.",
                evidence_dmcs=("BRAKE-DESC",),
                source_titles=("Brake system - Description of how it is made",),
            ),
            AnswerClaim(
                text="Install the fork and the brakes before installing the wheel.",
                evidence_dmcs=("WHEEL-INSTALL",),
                source_titles=("Front wheel - Install",),
            ),
            AnswerClaim(
                text="Make sure that the wheels lock and the bicycle stops.",
                evidence_dmcs=("BRAKE-TEST",),
                source_titles=("Brake system - Manual test",),
            ),
        ),
        required_citations=("BRAKE-DESC", "WHEEL-INSTALL", "BRAKE-TEST"),
        forbidden_claims=("unsupported diagnosis", "uncited DMCs"),
        sections=("증상 해석", "우선 확인 항목", "근거 문서"),
        graph_paths=("wheel -> affected_by -> brake pads", "wheel -> installed_by -> front wheel procedure"),
    )


def test_build_evidence_packet_contains_question_claims_relations_and_citations():
    packet = build_evidence_packet(_wheel_symptom_plan())

    assert packet.question == "바퀴가 잘 안 움직여"
    assert packet.intent == "describe"
    assert packet.claims[0]["text"].startswith("The pads press")
    assert packet.claims[0]["citations"] == ["BRAKE-DESC"]
    assert packet.relations == ["wheel -> affected_by -> brake pads", "wheel -> installed_by -> front wheel procedure"]
    assert packet.required_citations == ["BRAKE-DESC", "WHEEL-INSTALL", "BRAKE-TEST"]
    assert "unsupported diagnosis" in packet.forbidden_claims


def test_grounded_generator_uses_evidence_packet_not_korean_polish_draft():
    response = json.dumps(
        {
            "answer": "바퀴가 잘 안 움직이는 증상은 문서 근거상 브레이크 패드의 림 접촉과 앞바퀴 장착 상태를 우선 확인할 수 있습니다. 문서 근거만으로 특정 고장 원인을 확정할 수는 없습니다.",
            "check_items": ["브레이크 해제 상태에서 패드가 림에 닿는지 확인합니다.", "앞바퀴가 포크에 올바르게 장착되어 있는지 확인합니다."],
            "uncertainty": "문서 근거만으로 특정 고장 원인을 확정할 수 없습니다.",
            "used_citations": ["BRAKE-DESC", "WHEEL-INSTALL"],
        },
        ensure_ascii=False,
    )
    llm = RecordingLLM(response)

    answer = generate_grounded_answer(_wheel_symptom_plan(), llm)

    prompt = llm.prompts[0]
    assert "EvidencePacket" in prompt
    assert "사용자 질문: 바퀴가 잘 안 움직여" in prompt
    assert "wheel -> affected_by -> brake pads" in prompt
    assert "한국어 초안:" not in prompt
    assert "문장만 다듬으세요" not in prompt
    assert "브레이크 패드의 림 접촉" in answer
    assert "1. 브레이크 해제 상태에서 패드가 림에 닿는지 확인합니다." in answer
    assert "특정 고장 원인을 확정할 수 없습니다" in answer
    assert "근거 DMC: BRAKE-DESC, WHEEL-INSTALL" in answer


def test_verbalizer_falls_back_to_composer_when_grounded_llm_overreaches():
    bad = json.dumps(
        {
            "answer": "베어링이 고장났으니 즉시 교체하세요. [DMC: FAKE]",
            "check_items": ["새 베어링으로 교체합니다."],
            "uncertainty": "",
            "used_citations": ["FAKE"],
        },
        ensure_ascii=False,
    )
    llm = RecordingLLM(bad)

    answer = verbalize_answer_plan(_wheel_symptom_plan(), llm=llm)

    assert "베어링" not in answer
    assert "교체하세요" not in answer
    assert "[DMC:" not in answer
    assert "근거 DMC: BRAKE-DESC, WHEEL-INSTALL, BRAKE-TEST" in answer
