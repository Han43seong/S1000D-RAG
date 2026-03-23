"""S1000D DM XML → S1000DDmJson 변환 파서.

lxml 기반 규칙 기반(deterministic) 파싱.
Descriptive / Procedural DM을 우선 지원하며,
IPD / Fault 등은 content_blocks에 para 역할로 폴백 처리.
"""

from __future__ import annotations

from lxml import etree

from src.types.dm import ContentBlock, ContentBlockRole, DmType, S1000DDmJson
from .normalizer import (
    BlockIdGenerator,
    build_dmc_string,
    clean_text,
    detect_dm_type,
    extract_info_code,
    extract_text_content,
)


def parse_dm_xml(xml_str: str) -> S1000DDmJson:
    """S1000D DM XML 문자열을 파싱하여 S1000DDmJson 반환.

    Raises:
        ValueError: 필수 요소(dmCode, content 등)가 없는 경우.
    """
    root = etree.fromstring(xml_str.encode("utf-8"))

    # ── identAndStatusSection 추출 ──
    ident = _extract_ident_status(root)

    # ── content 추출 ──
    content_el = root.find(".//content")
    if content_el is None:
        raise ValueError("Missing <content> element in DM XML")

    dm_type = detect_dm_type(content_el, ident["info_code"])

    # ── content_blocks 생성 ──
    id_gen = BlockIdGenerator()
    blocks: list[ContentBlock] = []

    if dm_type == DmType.PROCEDURAL:
        procedure_el = content_el.find("procedure")
        if procedure_el is not None:
            blocks = _parse_procedure(procedure_el, id_gen)
    elif dm_type == DmType.DESCRIPTIVE:
        desc_el = content_el.find("description")
        if desc_el is not None:
            blocks = _parse_description(desc_el, id_gen)
    else:
        # Fault, IPD, Crew, Process 등 → 일반 텍스트 추출 폴백
        blocks = _parse_generic(content_el, id_gen)

    return S1000DDmJson(
        dmc=ident["dmc"],
        dm_type=dm_type,
        issue=ident["issue"],
        language=ident["language"],
        security=ident["security"],
        applicability=ident["applicability"],
        title=ident["title"],
        meta=ident["meta"],
        content_blocks=blocks,
    )


# ═══════════════════════════════════════════════════════════════════════
# identAndStatusSection 추출
# ═══════════════════════════════════════════════════════════════════════


def _extract_ident_status(root: etree._Element) -> dict:
    """identAndStatusSection에서 메타데이터 추출."""
    dm_code_el = root.find(".//dmIdent/dmCode")
    if dm_code_el is None:
        raise ValueError("Missing <dmCode> element in DM XML")

    dmc = build_dmc_string(dm_code_el)
    info_code = extract_info_code(dm_code_el)

    # issue
    issue_el = root.find(".//dmIdent/issueInfo")
    issue = ""
    if issue_el is not None:
        issue_num = issue_el.get("issueNumber", "")
        in_work = issue_el.get("inWork", "")
        issue = f"{issue_num}-{in_work}"

    # language
    lang_el = root.find(".//dmIdent/language")
    language = ""
    if lang_el is not None:
        lang_code = lang_el.get("languageIsoCode", "")
        country_code = lang_el.get("countryIsoCode", "")
        language = f"{lang_code}-{country_code}" if country_code else lang_code

    # security
    sec_el = root.find(".//dmStatus/security")
    security = ""
    if sec_el is not None:
        security = sec_el.get("securityClassification", "")

    # applicability
    applicability = _extract_applicability(root)

    # title
    title = _extract_title(root)

    # meta (추가 메타데이터)
    meta: dict = {}
    issue_date_el = root.find(".//dmAddressItems/issueDate")
    if issue_date_el is not None:
        y = issue_date_el.get("year", "")
        m = issue_date_el.get("month", "")
        d = issue_date_el.get("day", "")
        meta["issue_date"] = f"{y}-{m}-{d}"

    rpc_el = root.find(".//responsiblePartnerCompany/enterpriseName")
    if rpc_el is not None and rpc_el.text:
        meta["responsible_company"] = clean_text(rpc_el.text)

    skill_el = root.find(".//dmStatus/skillLevel")
    if skill_el is not None:
        meta["skill_level"] = skill_el.get("skillLevelCode", "")

    return {
        "dmc": dmc,
        "info_code": info_code,
        "issue": issue,
        "language": language,
        "security": security,
        "applicability": applicability,
        "title": title,
        "meta": meta,
    }


def _extract_title(root: etree._Element) -> str:
    """dmTitle에서 techName + infoName 조합."""
    title_el = root.find(".//dmAddressItems/dmTitle")
    if title_el is None:
        return ""
    tech = title_el.findtext("techName", default="").strip()
    info = title_el.findtext("infoName", default="").strip()
    if tech and info:
        return f"{tech} - {info}"
    return tech or info


def _extract_applicability(root: etree._Element) -> str | dict[str, str]:
    """dmStatus/applic에서 적용성 정보 추출.

    displayText가 있으면 문자열, 없으면 assert 속성들을 dict로 반환.
    """
    applic_el = root.find(".//dmStatus/applic")
    if applic_el is None:
        return "All"

    # displayText 우선
    display = applic_el.find("displayText/simplePara")
    if display is not None and display.text:
        return clean_text(display.text)

    # assert 속성들 수집
    asserts = applic_el.findall(".//assert")
    if asserts:
        result: dict[str, str] = {}
        for a in asserts:
            ident = a.get("applicPropertyIdent", "")
            values = a.get("applicPropertyValues", "")
            if ident:
                result[ident] = values
        return result

    return "All"


# ═══════════════════════════════════════════════════════════════════════
# Descriptive DM 파싱 (description/levelledPara)
# ═══════════════════════════════════════════════════════════════════════


def _parse_description(desc_el: etree._Element, id_gen: BlockIdGenerator) -> list[ContentBlock]:
    """description 요소 하위의 levelledPara들을 재귀 순회."""
    blocks: list[ContentBlock] = []
    for levelled_para in desc_el.findall("levelledPara"):
        _walk_levelled_para(levelled_para, blocks, id_gen, path_prefix="description")
    return blocks


def _walk_levelled_para(
    lp_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
    depth: int = 1,
) -> None:
    """levelledPara 재귀 순회하여 ContentBlock 생성."""
    lp_id = lp_el.get("id", "")
    lp_path = f"{path_prefix}/levelledPara[{depth}]"
    if lp_id:
        lp_path = f"{path_prefix}/levelledPara#{lp_id}"

    # title
    title_el = lp_el.find("title")
    if title_el is not None:
        title_text = extract_text_content(title_el)
        if title_text:
            blocks.append(ContentBlock(
                id=id_gen.next_id(ContentBlockRole.TITLE),
                role=ContentBlockRole.TITLE,
                text=title_text,
                structure_path=f"{lp_path}/title",
            ))

    # 직접 자식 순회
    para_idx = 0
    child_lp_idx = 0
    for child in lp_el:
        if child.tag == "para":
            para_idx += 1
            text = extract_text_content(child)
            if text:
                blocks.append(ContentBlock(
                    id=id_gen.next_id(ContentBlockRole.PARA),
                    role=ContentBlockRole.PARA,
                    text=text,
                    structure_path=f"{lp_path}/para[{para_idx}]",
                ))
        elif child.tag == "levelledPara":
            child_lp_idx += 1
            _walk_levelled_para(child, blocks, id_gen, lp_path, child_lp_idx)
        elif child.tag == "table":
            _parse_table(child, blocks, id_gen, lp_path)
        elif child.tag == "figure":
            _parse_figure_ref(child, blocks, id_gen, lp_path)
        elif child.tag == "note":
            _parse_note(child, blocks, id_gen, lp_path)
        elif child.tag == "warning":
            _parse_warning_caution(child, blocks, id_gen, lp_path, ContentBlockRole.WARNING)
        elif child.tag == "caution":
            _parse_warning_caution(child, blocks, id_gen, lp_path, ContentBlockRole.CAUTION)
        elif child.tag == "foldout":
            # foldout 내부의 figure 처리
            for fig in child.findall("figure"):
                _parse_figure_ref(fig, blocks, id_gen, lp_path)


# ═══════════════════════════════════════════════════════════════════════
# Procedural DM 파싱 (procedure/mainProcedure/proceduralStep)
# ═══════════════════════════════════════════════════════════════════════


def _parse_procedure(proc_el: etree._Element, id_gen: BlockIdGenerator) -> list[ContentBlock]:
    """procedure 요소 파싱: commonInfo + preliminaryRqmts + mainProcedure + closeRqmts."""
    blocks: list[ContentBlock] = []

    # commonInfo
    common_info = proc_el.find("commonInfo/commonInfoDescrPara")
    if common_info is not None:
        for para in common_info.findall("para"):
            text = extract_text_content(para)
            if text:
                blocks.append(ContentBlock(
                    id=id_gen.next_id(ContentBlockRole.PARA),
                    role=ContentBlockRole.PARA,
                    text=text,
                    structure_path="procedure/commonInfo",
                ))

    # preliminaryRqmts → warning / caution 추출
    prelim = proc_el.find("preliminaryRqmts")
    if prelim is not None:
        _parse_prelim_rqmts(prelim, blocks, id_gen)

    # mainProcedure
    main_proc = proc_el.find("mainProcedure")
    if main_proc is not None:
        step_idx = 0
        for step in main_proc.findall("proceduralStep"):
            step_idx += 1
            _walk_procedural_step(
                step, blocks, id_gen,
                path_prefix="procedure/mainProcedure",
                step_num=str(step_idx),
            )

    # closeRqmts
    close = proc_el.find("closeRqmts")
    if close is not None:
        for req_cond in close.findall(".//reqCond"):
            text = extract_text_content(req_cond)
            if text:
                blocks.append(ContentBlock(
                    id=id_gen.next_id(ContentBlockRole.NOTE),
                    role=ContentBlockRole.NOTE,
                    text=f"[Close requirement] {text}",
                    structure_path="procedure/closeRqmts",
                ))

    return blocks


def _parse_prelim_rqmts(
    prelim_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
) -> None:
    """preliminaryRqmts에서 warning, caution, 조건 등 추출."""
    path = "procedure/preliminaryRqmts"

    # reqCondGroup → 사전 조건
    for req_cond in prelim_el.findall(".//reqCondGroup//reqCond"):
        text = extract_text_content(req_cond)
        if text:
            blocks.append(ContentBlock(
                id=id_gen.next_id(ContentBlockRole.NOTE),
                role=ContentBlockRole.NOTE,
                text=f"[Prerequisite] {text}",
                structure_path=f"{path}/reqCondGroup",
            ))

    # reqSafety → warning / caution
    for warning in prelim_el.findall(".//reqSafety//warning"):
        _parse_warning_caution(warning, blocks, id_gen, path, ContentBlockRole.WARNING)
    for caution in prelim_el.findall(".//reqSafety//caution"):
        _parse_warning_caution(caution, blocks, id_gen, path, ContentBlockRole.CAUTION)


def _walk_procedural_step(
    step_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
    step_num: str,
) -> None:
    """proceduralStep 재귀 순회 (중첩 스텝 지원)."""
    step_path = f"{path_prefix}/step[{step_num}]"

    # 스텝 본문 텍스트 수집
    step_texts: list[str] = []
    sub_step_idx = 0

    for child in step_el:
        if child.tag == "para":
            text = extract_text_content(child)
            if text:
                step_texts.append(text)
        elif child.tag == "note":
            _parse_note(child, blocks, id_gen, step_path)
        elif child.tag == "warning":
            _parse_warning_caution(child, blocks, id_gen, step_path, ContentBlockRole.WARNING)
        elif child.tag == "caution":
            _parse_warning_caution(child, blocks, id_gen, step_path, ContentBlockRole.CAUTION)
        elif child.tag == "proceduralStep":
            # 먼저 현재 스텝 텍스트가 있으면 블록으로 저장
            if step_texts:
                blocks.append(ContentBlock(
                    id=id_gen.next_id(ContentBlockRole.STEP),
                    role=ContentBlockRole.STEP,
                    text=" ".join(step_texts),
                    structure_path=step_path,
                ))
                step_texts.clear()
            sub_step_idx += 1
            _walk_procedural_step(
                child, blocks, id_gen,
                path_prefix=step_path,
                step_num=f"{step_num}.{sub_step_idx}",
            )
        elif child.tag == "figure":
            _parse_figure_ref(child, blocks, id_gen, step_path)
        elif child.tag == "table":
            _parse_table(child, blocks, id_gen, step_path)

    # 남은 텍스트를 step 블록으로 저장
    if step_texts:
        blocks.append(ContentBlock(
            id=id_gen.next_id(ContentBlockRole.STEP),
            role=ContentBlockRole.STEP,
            text=" ".join(step_texts),
            structure_path=step_path,
        ))


# ═══════════════════════════════════════════════════════════════════════
# 공통 요소 파서
# ═══════════════════════════════════════════════════════════════════════


def _parse_note(
    note_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
) -> None:
    """note 요소 파싱."""
    for np in note_el.findall("notePara"):
        text = extract_text_content(np)
        if text:
            blocks.append(ContentBlock(
                id=id_gen.next_id(ContentBlockRole.NOTE),
                role=ContentBlockRole.NOTE,
                text=text,
                structure_path=f"{path_prefix}/note",
            ))


def _parse_warning_caution(
    el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
    role: ContentBlockRole,
) -> None:
    """warning 또는 caution 요소 파싱."""
    for wcp in el.findall("warningAndCautionPara"):
        text = extract_text_content(wcp)
        if text:
            blocks.append(ContentBlock(
                id=id_gen.next_id(role),
                role=role,
                text=text,
                structure_path=f"{path_prefix}/{role.value}",
            ))


def _parse_table(
    table_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
) -> None:
    """테이블을 텍스트로 변환. 제목 + 행별 셀 텍스트."""
    table_id = table_el.get("id", "")
    table_path = f"{path_prefix}/table#{table_id}" if table_id else f"{path_prefix}/table"

    title_el = table_el.find("title")
    title_text = extract_text_content(title_el) if title_el is not None else ""

    rows_text: list[str] = []
    for row in table_el.findall(".//row"):
        cells: list[str] = []
        for entry in row.findall("entry"):
            cell_text = extract_text_content(entry)
            if cell_text:
                cells.append(cell_text)
        if cells:
            rows_text.append(" | ".join(cells))

    table_content = "\n".join(rows_text)
    if title_text:
        table_content = f"[Table: {title_text}]\n{table_content}"

    if table_content.strip():
        blocks.append(ContentBlock(
            id=id_gen.next_id(ContentBlockRole.TABLE),
            role=ContentBlockRole.TABLE,
            text=table_content,
            structure_path=table_path,
        ))


def _parse_figure_ref(
    fig_el: etree._Element,
    blocks: list[ContentBlock],
    id_gen: BlockIdGenerator,
    path_prefix: str,
) -> None:
    """figure 요소에서 참조 정보 추출."""
    fig_id = fig_el.get("id", "")
    title_el = fig_el.find("title")
    title_text = extract_text_content(title_el) if title_el is not None else ""

    graphic_el = fig_el.find("graphic")
    entity_ident = ""
    if graphic_el is not None:
        entity_ident = graphic_el.get("infoEntityIdent", "")

    text_parts = []
    if title_text:
        text_parts.append(f"[Figure: {title_text}]")
    if entity_ident:
        text_parts.append(f"(ICN: {entity_ident})")

    if text_parts:
        blocks.append(ContentBlock(
            id=id_gen.next_id(ContentBlockRole.FIGURE_REF),
            role=ContentBlockRole.FIGURE_REF,
            text=" ".join(text_parts),
            structure_path=f"{path_prefix}/figure#{fig_id}" if fig_id else f"{path_prefix}/figure",
        ))


# ═══════════════════════════════════════════════════════════════════════
# 제네릭 폴백 파서 (IPD, Fault 등)
# ═══════════════════════════════════════════════════════════════════════


def _parse_generic(content_el: etree._Element, id_gen: BlockIdGenerator) -> list[ContentBlock]:
    """알려지지 않은 DM 타입에 대한 폴백: 모든 텍스트를 para 블록으로 수집."""
    blocks: list[ContentBlock] = []
    for el in content_el.iter():
        if el.tag in ("para", "simplePara", "notePara", "warningAndCautionPara"):
            text = extract_text_content(el)
            if text:
                blocks.append(ContentBlock(
                    id=id_gen.next_id(ContentBlockRole.PARA),
                    role=ContentBlockRole.PARA,
                    text=text,
                    structure_path=f"content/{el.tag}",
                ))
    return blocks
