"""Phase 3 테스트: DM Parser + Normalizer.

실제 Bike 샘플 DM XML 파일을 사용한 통합 테스트.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import S1000D_DATA_DIR
from src.parser.dm_parser import parse_dm_xml
from src.parser.normalizer import (
    BlockIdGenerator,
    build_dmc_string,
    clean_text,
    detect_dm_type,
    extract_info_code,
    extract_text_content,
)
from src.types.dm import ContentBlockRole, DmType

# ── 테스트 데이터 경로 ──
SAMPLE_DIR = S1000D_DATA_DIR

# Descriptive DM
DESC_DM_FILE = SAMPLE_DIR / "DMC-BRAKE-AAA-DA1-00-00-00AA-041A-A_004-00_EN-US.XML"
# Procedural DM (simple)
PROC_SIMPLE_FILE = SAMPLE_DIR / "DMC-BRAKE-AAA-DA1-00-00-00AA-341A-A_004-00_EN-US.XML"
# Procedural DM (complex - nested steps, warnings, notes)
PROC_COMPLEX_FILE = SAMPLE_DIR / "DMC-S1000DBIKE-AAA-D00-00-00-00AA-258A-A_011-00_EN-US.XML"
# Descriptive DM with table
DESC_TABLE_FILE = SAMPLE_DIR / "DMC-S1000DBIKE-AAA-D00-00-00-00AA-041A-A_012-00_EN-US.XML"


# ═══════════════════════════════════════════════════════════════════════
# Normalizer 단위 테스트
# ═══════════════════════════════════════════════════════════════════════


class TestCleanText:
    def test_multi_space(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_newlines(self):
        assert clean_text("line1\n  line2\n") == "line1 line2"

    def test_empty(self):
        assert clean_text("") == ""


class TestBlockIdGenerator:
    def test_sequential_ids(self):
        gen = BlockIdGenerator()
        assert gen.next_id(ContentBlockRole.PARA) == "para-1"
        assert gen.next_id(ContentBlockRole.PARA) == "para-2"
        assert gen.next_id(ContentBlockRole.STEP) == "step-1"
        assert gen.next_id(ContentBlockRole.PARA) == "para-3"


class TestBuildDmcString:
    def test_from_xml(self):
        from lxml import etree
        xml = '<dmCode modelIdentCode="BRAKE" systemDiffCode="AAA" systemCode="DA1" subSystemCode="0" subSubSystemCode="0" assyCode="00" disassyCode="00" disassyCodeVariant="AA" infoCode="041" infoCodeVariant="A" itemLocationCode="A"/>'
        el = etree.fromstring(xml)
        dmc = build_dmc_string(el)
        assert dmc == "BRAKE-AAA-DA1-00-00-00AA-041A-A"

    def test_extract_info_code(self):
        from lxml import etree
        xml = '<dmCode infoCode="341"/>'
        el = etree.fromstring(xml)
        assert extract_info_code(el) == "341"


class TestDetectDmType:
    def test_procedure_by_content(self):
        from lxml import etree
        content = etree.fromstring("<content><procedure/></content>")
        assert detect_dm_type(content, "999") == DmType.PROCEDURAL

    def test_description_by_content(self):
        from lxml import etree
        content = etree.fromstring("<content><description/></content>")
        assert detect_dm_type(content, "999") == DmType.DESCRIPTIVE

    def test_by_info_code(self):
        assert detect_dm_type(None, "041") == DmType.DESCRIPTIVE
        assert detect_dm_type(None, "341") == DmType.PROCEDURAL

    def test_fallback(self):
        from lxml import etree
        content = etree.fromstring("<content><unknownTag/></content>")
        assert detect_dm_type(content, "999") == DmType.DESCRIPTIVE


# ═══════════════════════════════════════════════════════════════════════
# DM Parser 통합 테스트 (실제 XML 파일)
# ═══════════════════════════════════════════════════════════════════════


def _skip_if_no_file(path: Path):
    if not path.exists():
        pytest.skip(f"Sample file not found: {path.name}")


class TestParseDescriptiveDM:
    """Descriptive DM 파싱 테스트 - BRAKE-041A (Brake system description)."""

    def test_parse_brake_description(self):
        _skip_if_no_file(DESC_DM_FILE)
        xml = DESC_DM_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        # 메타데이터 검증
        assert result.dmc == "BRAKE-AAA-DA1-00-00-00AA-041A-A"
        assert result.dm_type == DmType.DESCRIPTIVE
        assert result.issue == "004-00"
        assert result.language == "en-US"
        assert result.security == "01"
        assert result.title == "Brake system - Description of how it is made"

        # content_blocks가 비어있지 않아야 함
        assert len(result.content_blocks) > 0

        # 첫 번째 블록은 title이어야 함
        titles = [b for b in result.content_blocks if b.role == ContentBlockRole.TITLE]
        assert len(titles) >= 1
        assert "Brake system" in titles[0].text

        # para 블록 존재
        paras = [b for b in result.content_blocks if b.role == ContentBlockRole.PARA]
        assert len(paras) >= 2

        # structure_path 포함
        assert all(b.structure_path for b in result.content_blocks)

    def test_nested_levelled_para(self):
        """중첩 levelledPara에서 하위 title과 para 추출."""
        _skip_if_no_file(DESC_DM_FILE)
        xml = DESC_DM_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        # 하위 levelledPara 타이틀들
        titles = [b for b in result.content_blocks if b.role == ContentBlockRole.TITLE]
        title_texts = [t.text for t in titles]
        assert "Cantilever brake" in title_texts
        assert "Brake pads" in title_texts
        assert "Brake lever" in title_texts


class TestParseSimpleProcedural:
    """Simple Procedural DM 테스트 - BRAKE-341A (Manual test)."""

    def test_parse_manual_test(self):
        _skip_if_no_file(PROC_SIMPLE_FILE)
        xml = PROC_SIMPLE_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        assert result.dmc == "BRAKE-AAA-DA1-00-00-00AA-341A-A"
        assert result.dm_type == DmType.PROCEDURAL
        assert result.title == "Brake system - Manual test"

        # 4개의 step 블록
        steps = [b for b in result.content_blocks if b.role == ContentBlockRole.STEP]
        assert len(steps) == 4

        # 순서 검증
        assert "vertical position" in steps[0].text
        assert "push" in steps[1].text.lower()
        assert "brakes" in steps[2].text.lower()
        assert "stops" in steps[3].text

        # structure_path 검증
        assert steps[0].structure_path == "procedure/mainProcedure/step[1]"
        assert steps[3].structure_path == "procedure/mainProcedure/step[4]"


class TestParseComplexProcedural:
    """Complex Procedural DM 테스트 - S1000DBIKE-258A (Cleaning procedure)."""

    def test_parse_cleaning_procedure(self):
        _skip_if_no_file(PROC_COMPLEX_FILE)
        xml = PROC_COMPLEX_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        assert result.dmc == "S1000DBIKE-AAA-D00-00-00-00AA-258A-A"
        assert result.dm_type == DmType.PROCEDURAL
        assert "clean" in result.title.lower()

    def test_warnings_and_cautions(self):
        """preliminaryRqmts에서 warning/caution 추출."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        xml = PROC_COMPLEX_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        warnings = [b for b in result.content_blocks if b.role == ContentBlockRole.WARNING]
        cautions = [b for b in result.content_blocks if b.role == ContentBlockRole.CAUTION]

        assert len(warnings) >= 1
        assert len(cautions) >= 2

    def test_nested_steps(self):
        """중첩 proceduralStep 파싱."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        xml = PROC_COMPLEX_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        steps = [b for b in result.content_blocks if b.role == ContentBlockRole.STEP]
        # 중첩 스텝이 있으므로 4개 이상
        assert len(steps) > 4

        # 중첩 스텝의 structure_path 검증
        nested_paths = [s.structure_path for s in steps if "." in s.structure_path.split("step[")[-1]]
        assert len(nested_paths) > 0  # 중첩 스텝 존재

    def test_notes_in_steps(self):
        """스텝 내 note 블록."""
        _skip_if_no_file(PROC_COMPLEX_FILE)
        xml = PROC_COMPLEX_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        notes = [b for b in result.content_blocks if b.role == ContentBlockRole.NOTE]
        # prerequisite + close requirement + inline notes
        assert len(notes) >= 1


class TestParseDescriptiveWithTable:
    """테이블 포함 Descriptive DM 테스트 - S1000DBIKE-041A."""

    def test_table_extraction(self):
        _skip_if_no_file(DESC_TABLE_FILE)
        xml = DESC_TABLE_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        tables = [b for b in result.content_blocks if b.role == ContentBlockRole.TABLE]
        assert len(tables) >= 1
        assert "Bicycle parts" in tables[0].text

    def test_figure_ref(self):
        _skip_if_no_file(DESC_TABLE_FILE)
        xml = DESC_TABLE_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)

        figs = [b for b in result.content_blocks if b.role == ContentBlockRole.FIGURE_REF]
        assert len(figs) >= 1
        assert "Complete bicycle" in figs[0].text


class TestApplicability:
    def test_display_text_applicability(self):
        _skip_if_no_file(DESC_TABLE_FILE)
        xml = DESC_TABLE_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)
        assert "Mountain bicycle" in str(result.applicability)

    def test_assert_applicability(self):
        _skip_if_no_file(DESC_DM_FILE)
        xml = DESC_DM_FILE.read_text(encoding="utf-8")
        result = parse_dm_xml(xml)
        # BRAKE DM은 assert 기반 applicability
        assert isinstance(result.applicability, dict)
        assert "SerialNo" in result.applicability or "model" in result.applicability
