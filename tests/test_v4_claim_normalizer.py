from src.rag.v4.claim_normalizer import normalize_claim_text, normalize_claims


def test_claim_normalizer_converts_brake_descriptive_evidence_to_korean_units():
    text = (
        "The brake system has these primary components: the brake lever (refer to ) "
        "the brake cable the brake arm the brake clamp (also known as callipers) the brake pads (refer to ). "
        "A cable that goes from the brake levers on the handlebars pulls the two levers on the brakes together. "
        "This presses the brake pads against the outer rim of the wheel, which decreases the speed of the bicycle."
    )

    units = normalize_claim_text(text)

    assert "브레이크 시스템은 브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(캘리퍼), 브레이크 패드로 구성됩니다." in units
    assert "핸들바의 브레이크 레버에서 이어진 케이블이 브레이크 쪽 두 레버를 함께 당깁니다." in units
    assert "브레이크 패드가 바퀴의 바깥쪽 림을 누릅니다." in units
    assert "이 마찰로 자전거 속도가 줄어듭니다." in units
    assert all("The brake system" not in unit for unit in units)
    assert all("refer to" not in unit for unit in units)


def test_claim_normalizer_deduplicates_overlapping_claims_preserving_order():
    claims = [
        "Install the fork before installing the wheel. Hold the front of the bicycle.",
        "Install the fork before installing the wheel.",
        "Put the bike on the floor.",
    ]

    units = normalize_claims(claims)

    assert units == [
        "바퀴를 설치하기 전에 포크가 먼저 장착되어 있는지 확인합니다.",
        "자전거 앞부분을 안정적으로 잡습니다.",
        "자전거를 바닥에 내려놓습니다.",
    ]


def test_claim_normalizer_keeps_existing_korean_claim_and_drops_unknown_english():
    units = normalize_claims([
        "브레이크 케이블 절차는 직접 확인되지 않았습니다.",
        "Brake cable routing is described.",
    ])

    assert units == ["브레이크 케이블 절차는 직접 확인되지 않았습니다."]


def test_claim_normalizer_removes_subsumed_brake_component_summary():
    units = normalize_claim_text(
        "The brake system has these primary components: the brake lever the brake cable "
        "the brake arm the brake clamp also known as callipers the brake pads."
    )

    assert units == [
        "브레이크 시스템은 브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(캘리퍼), 브레이크 패드로 구성됩니다.",
        "브레이크 클램프는 캘리퍼라고도 부릅니다.",
    ]

