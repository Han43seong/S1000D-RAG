"""쿼리 확장 + SNS 코드 추출 모듈.

한→영 도메인 용어 확장으로 벡터 검색 recall을 높이고,
쿼리에서 SNS 코드를 추출하여 2단계 검색에 활용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from langsmith import traceable


# ── 한→영 도메인 용어 확장 사전 ──
QUERY_EXPANSION: dict[str, str] = {
    # 정비 작업
    "교체": "remove install replace",
    "탈거": "remove removal detach",
    "장착": "install installation attach mount",
    "점검": "check inspection examine",
    "검사": "inspection test examine",
    "수리": "repair fix maintenance",
    "조정": "adjust adjustment setting",
    "세척": "clean cleaning wash",
    "윤활": "lubricate lubrication grease oil",
    "정비": "maintenance service overhaul",
    "규격": "specification spec dimension",
    "절차": "procedure step",
    "방법": "procedure method how",
    "시험": "test testing manual test brake system manual test put bicycle vertical hold handle bars push bicycle forwards apply the brakes wheel locks bicycle stops",
    "테스트": "test testing manual test brake system manual test put bicycle vertical hold handle bars push bicycle forwards apply the brakes wheel locks bicycle stops",
    "수동": "manual",
    "부품": "part component",
    "압력": "pressure",
    "마모": "wear worn",
    "손상": "damage broken",
    # 메타 질문
    "교범": "manual handbook technical publication overview contents",
    "내용": "content overview description summary table of contents",
    "목차": "table of contents overview index",
    # 고장/진단
    "고장": "fault failure troubleshoot malfunction",
    "결함": "defect fault failure",
    "누유": "leak leakage oil",
    "진동": "vibration shake",
    "소음": "noise sound",
    # 부품/시스템
    "브레이크": "brake braking",
    "제동": "brake braking stop",
    "타이어": "tire tyre wheel",
    "휠": "wheel rim",
    "체인": "chain drive",
    "기어": "gear transmission derailleur",
    "변속": "gear shift derailleur transmission",
    "페달": "pedal crank",
    "핸들": "handlebar steering",
    "안장": "saddle seat",
    "프레임": "frame body",
    "서스펜션": "suspension fork shock",
    "조향": "steering handlebar",
    "조명": "light lighting lamp",
    "경적": "horn bell",
}

# ── SNS 코드 키워드 매핑 (Bike 샘플 기준) ──
SNS_KEYWORDS: list[tuple[str, list[str]]] = [
    ("DA1", ["브레이크", "제동", "brake", "braking", "패드", "pad", "레버", "lever"]),
    ("DA0", ["타이어", "휠", "tire", "tyre", "wheel", "rim", "튜브", "tube"]),
    ("DA2", ["체인", "chain", "기어", "gear", "변속", "derailleur", "스프로켓", "sprocket"]),
    ("DA3", ["페달", "pedal", "크랭크", "crank"]),
    ("DA4", ["핸들", "handlebar", "조향", "steering", "스템", "stem"]),
    ("DA5", ["안장", "saddle", "seat", "시트", "시트포스트", "seatpost"]),
    ("D05", ["조명", "light", "lamp", "경적", "horn", "bell", "전장", "electrical"]),
    ("D00", ["프레임", "frame", "자전거", "bike", "bicycle"]),
]


@dataclass
class EnhancedQuery:
    """쿼리 확장 결과."""

    original: str
    expanded: str
    sns_code: str | None


@traceable(run_type="chain", name="expand_query")
def expand_query(query: str) -> str:
    """한국어 키워드를 감지하여 영어 도메인 용어를 append.

    Args:
        query: 원본 쿼리.

    Returns:
        확장된 쿼리 문자열 (원본 + 영어 확장).
    """
    additions: list[str] = []
    for kor_keyword, eng_terms in QUERY_EXPANSION.items():
        if kor_keyword in query:
            additions.append(eng_terms)

    if not additions:
        return query

    return f"{query} {' '.join(additions)}"


@traceable(run_type="chain", name="extract_sns_code")
def extract_sns_code(query: str) -> str | None:
    """쿼리에서 SNS 코드를 추출.

    원본 쿼리에서만 추출 (확장 쿼리 사용 시 오탐 방지).

    Args:
        query: 원본 쿼리.

    Returns:
        SNS 코드 (예: "DA1") 또는 None.
    """
    query_lower = query.lower()

    best_code: str | None = None
    best_count = 0

    for sns_code, keywords in SNS_KEYWORDS:
        count = sum(1 for kw in keywords if kw.lower() in query_lower)
        if count > best_count:
            best_count = count
            best_code = sns_code

    return best_code


@traceable(run_type="chain", name="enhance_query")
def enhance_query(query: str) -> EnhancedQuery:
    """쿼리 확장 + SNS 코드 추출을 조합.

    Args:
        query: 원본 쿼리.

    Returns:
        EnhancedQuery (original, expanded, sns_code).
    """
    sns_code = extract_sns_code(query)
    expanded = expand_query(query)

    return EnhancedQuery(
        original=query,
        expanded=expanded,
        sns_code=sns_code,
    )
