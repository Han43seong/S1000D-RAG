from scripts.run_quality_qa_loop import QaCase, build_cases, classify, classify_detailed


def test_classify_detailed_passes_with_reference_materials_and_visual_preview():
    case = QaCase(
        id="ontology-pass",
        question="브레이크 패드 청소 절차와 그림 근거를 알려줘",
        expected="supported",
        required_reference_categories=("data_modules",),
        require_reference_materials_when_evidence=True,
        require_visual_preview_status=True,
        require_clean_display_answer=True,
        expected_dmc_substrings=("DMC-BRAKE-PAD",),
    )
    response = {
        "answer": "브레이크 패드 청소 절차는 제공 문서 DMC-BRAKE-PAD-CLEAN 기준으로 수행합니다.",
        "evidences": [{"dmc": "DMC-BRAKE-PAD-CLEAN", "text": "clean the brake pad"}],
        "reference_materials": {
            "data_modules": [{"dmc": "DMC-BRAKE-PAD-CLEAN", "title": "Brake pad cleaning"}],
            "graphic_assets": [
                {
                    "asset_id": "fig-1",
                    "preview_status": "available",
                    "preview_available": True,
                    "preview_url": "/api/assets/fig-1/preview.png",
                    "original_url": "/api/assets/fig-1.cgm",
                    "asset_format": "cgm",
                }
            ],
        },
    }

    result = classify_detailed(case, response)

    assert result["status"] == "pass"
    assert result["issues"] == []
    assert result["checks"]["reference_materials"]["status"] == "pass"
    assert result["checks"]["visual_preview"]["status"] == "pass"


def test_classify_detailed_accepts_clean_answer_when_dmc_is_in_evidence_and_reference_materials():
    case = QaCase(
        id="clean-answer-grounded-elsewhere",
        question="브레이크 설명을 알려줘",
        expected="supported",
        require_reference_materials_when_evidence=True,
        required_reference_categories=("data_modules",),
        require_clean_display_answer=True,
    )
    response = {
        "answer": "브레이크 시스템은 자전거 속도를 줄이는 주요 안전 장치입니다.",
        "evidences": [{"dmc": "DMC-BRAKE-DESC", "text": "brake system"}],
        "reference_materials": {"data_modules": [{"dmc": "DMC-BRAKE-DESC", "title": "Brake system"}]},
    }

    result = classify_detailed(case, response)

    assert result["status"] == "pass"
    assert "missing_dmc_in_answer" not in result["issues"]


def test_classify_detailed_accepts_broad_grounded_answer_with_structured_references():
    case = QaCase(
        id="grounded-broad",
        question="타이어 공기압 기준은 얼마인가요?",
        expected="broad",
    )
    response = {
        "answer": "타이어 공기압 기준은 2000 hPa에서 2700 hPa 사이입니다.",
        "evidences": [{"dmc": "DMC-TIRE-PRESSURE", "text": "Tire pressure should between 2000 hPa to 2700 hPa."}],
        "reference_materials": {"data_modules": [{"dmc": "DMC-TIRE-PRESSURE", "title": "Tire - Check pressure"}]},
    }

    result = classify_detailed(case, response)

    assert result["status"] == "pass"
    assert "scope_not_limited" not in result["issues"]


def test_build_cases_requires_dmc_substrings_only_when_question_requests_dmc_or_grounding():
    cases = {case.id: case for case in build_cases()}

    assert cases["q076"].expected_dmc_substrings
    assert cases["q081"].expected_dmc_substrings
    assert cases["q099"].expected_dmc_substrings == ()


def test_classify_detailed_flags_missing_reference_materials_when_evidence_requires_it():
    case = QaCase(
        id="missing-refs",
        question="브레이크 케이블 설명 문서의 DMC는?",
        expected="supported",
        require_reference_materials_when_evidence=True,
    )
    response = {
        "answer": "제공 문서 DMC-BRAKE-CABLE-DESC 기준으로 확인됩니다.",
        "evidences": [{"dmc": "DMC-BRAKE-CABLE-DESC", "text": "brake cable"}],
    }

    result = classify_detailed(case, response)

    assert result["status"] == "fail"
    assert "missing_reference_materials" in result["issues"]
    assert "missing_reference_materials" in result["checks"]["reference_materials"]["issues"]


def test_classify_detailed_flags_missing_required_reference_category():
    case = QaCase(
        id="missing-category",
        question="브레이크 시스템 설명과 근거를 알려줘",
        expected="supported",
        required_reference_categories=("data_modules",),
    )
    response = {
        "answer": "제공 문서 DMC-BRAKE-SYSTEM-DESC 기준입니다.",
        "evidences": [{"dmc": "DMC-BRAKE-SYSTEM-DESC", "text": "brake system"}],
        "reference_materials": {"procedures": [{"dmc": "DMC-BRAKE-PROC"}]},
    }

    result = classify_detailed(case, response)

    assert "missing_reference_category:data_modules" in result["issues"]


def test_classify_detailed_flags_missing_graphic_assets_when_visual_preview_required():
    case = QaCase(
        id="missing-visual-assets",
        question="브레이크 그림을 보여줘",
        expected="supported",
        require_visual_preview_status=True,
    )
    response = {
        "answer": "제공 문서 DMC-BRAKE-FIG 기준 그림이 있습니다.",
        "evidences": [{"dmc": "DMC-BRAKE-FIG", "text": "figure"}],
        "reference_materials": {"data_modules": [{"dmc": "DMC-BRAKE-FIG"}]},
    }

    result = classify_detailed(case, response)

    assert result["status"] == "fail"
    assert "missing_graphic_assets" in result["issues"]


def test_classify_detailed_flags_missing_and_invalid_visual_preview_status():
    case = QaCase(
        id="bad-visuals",
        question="브레이크 그림을 보여줘",
        expected="supported",
        require_visual_preview_status=True,
    )
    missing_status = {
        "answer": "제공 문서 DMC-BRAKE-FIG 기준 그림이 있습니다.",
        "evidences": [{"dmc": "DMC-BRAKE-FIG", "text": "figure"}],
        "reference_materials": {"graphic_assets": [{"asset_id": "fig-1"}]},
    }
    invalid_status = {
        "answer": "제공 문서 DMC-BRAKE-FIG 기준 그림이 있습니다.",
        "evidences": [{"dmc": "DMC-BRAKE-FIG", "text": "figure"}],
        "reference_materials": {"graphic_assets": [{"asset_id": "fig-1", "preview_status": "broken"}]},
    }

    assert "missing_visual_preview_status" in classify_detailed(case, missing_status)["issues"]
    assert "invalid_visual_preview_status" in classify_detailed(case, invalid_status)["issues"]


def test_classify_detailed_flags_trailing_ui_metadata_not_mid_sentence_reference_phrase():
    case = QaCase(
        id="ui-clean",
        question="브레이크 설명을 알려줘",
        expected="supported",
        require_clean_display_answer=True,
    )
    base = {"evidences": [{"dmc": "DMC-BRAKE-DESC", "text": "brake"}]}

    leaked = dict(base, answer="브레이크 설명입니다.\n참고 문서: DMC-BRAKE-DESC")
    leaked_fullwidth_colon = dict(base, answer="브레이크 설명입니다.\n근거： DMC-BRAKE-DESC")
    ordinary = dict(base, answer="참고 문서라는 표현은 일반 문장 중간에서 설명할 수 있습니다. DMC-BRAKE-DESC")

    assert "ui_metadata_leak" in classify_detailed(case, leaked)["issues"]
    assert "ui_metadata_leak" in classify_detailed(case, leaked_fullwidth_colon)["issues"]
    assert "ui_metadata_leak" not in classify_detailed(case, ordinary)["issues"]


def test_classify_remains_backward_compatible_tuple():
    case = QaCase(id="compat", question="브레이크 설명", expected="supported")
    status, issues = classify(case, {"answer": "", "evidences": []})

    assert status == "fail"
    assert isinstance(issues, list)
    assert "empty_answer" in issues


def test_build_cases_still_returns_100_and_some_ontology_expectations():
    cases = build_cases()

    assert len(cases) == 100
    assert any(c.require_reference_materials_when_evidence for c in cases)
    assert any(c.require_visual_preview_status for c in cases)
    assert any(c.required_reference_categories for c in cases)
