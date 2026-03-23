"""쿼리 확장 + SNS 코드 추출 테스트."""

from __future__ import annotations

from src.rag.query_enhancer import (
    EnhancedQuery,
    enhance_query,
    expand_query,
    extract_sns_code,
)


class TestExpandQuery:
    def test_korean_expansion(self):
        """한국어 키워드가 영어로 확장."""
        result = expand_query("브레이크 교체 절차")
        assert "brake" in result
        assert "remove" in result
        assert "install" in result
        assert "브레이크 교체 절차" in result  # 원본 유지

    def test_no_expansion(self):
        """매칭 키워드 없으면 원본 반환."""
        result = expand_query("How does the brake system work?")
        assert result == "How does the brake system work?"

    def test_multiple_keywords(self):
        """복수 키워드 동시 확장."""
        result = expand_query("타이어 점검 방법")
        assert "tire" in result
        assert "check" in result or "inspection" in result

    def test_single_keyword(self):
        """단일 키워드 확장."""
        result = expand_query("윤활 주기")
        assert "lubricate" in result


class TestExtractSnsCode:
    def test_brake_keyword(self):
        """브레이크 키워드 → DA1."""
        assert extract_sns_code("브레이크 교체 방법") == "DA1"

    def test_tire_keyword(self):
        """타이어 키워드 → DA0."""
        assert extract_sns_code("타이어 공기압 점검") == "DA0"

    def test_chain_keyword(self):
        """체인 키워드 → DA2."""
        assert extract_sns_code("체인 장력 조정") == "DA2"

    def test_pedal_keyword(self):
        """페달 키워드 → DA3."""
        assert extract_sns_code("페달 교체 절차") == "DA3"

    def test_handlebar_keyword(self):
        """핸들 키워드 → DA4."""
        assert extract_sns_code("핸들 조정 방법") == "DA4"

    def test_saddle_keyword(self):
        """안장 키워드 → DA5."""
        assert extract_sns_code("안장 높이 조정") == "DA5"

    def test_no_match(self):
        """매칭 없으면 None."""
        assert extract_sns_code("일반적인 질문입니다") is None

    def test_english_keyword(self):
        """영어 키워드도 매칭."""
        assert extract_sns_code("brake pad replacement") == "DA1"

    def test_multiple_matches_picks_best(self):
        """복수 시스템 키워드 시 매칭 수가 많은 것 선택."""
        # "브레이크 패드 레버" → DA1에 3개 매칭
        assert extract_sns_code("브레이크 패드 레버") == "DA1"


class TestEnhanceQuery:
    def test_combined(self):
        """expand + sns 추출 조합."""
        result = enhance_query("브레이크 교체 절차")
        assert isinstance(result, EnhancedQuery)
        assert result.original == "브레이크 교체 절차"
        assert "brake" in result.expanded
        assert result.sns_code == "DA1"

    def test_no_match(self):
        """매칭 없는 경우."""
        result = enhance_query("general question")
        assert result.original == "general question"
        assert result.expanded == "general question"
        assert result.sns_code is None
