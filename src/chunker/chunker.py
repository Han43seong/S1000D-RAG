"""S1000D content_blocks 기반 청킹 엔진.

DM JSON(S1000DDmJson)의 content_blocks를 슬라이딩 윈도우 방식으로
묶어 S1000DChunk 리스트를 생성한다.

청킹 전략:
1. block_count: N개 블록씩 묶기 (기본)
2. max_chars: 최대 문자 수 제한으로 자동 분할
3. overlap: 이전 청크와 겹치는 블록 수
4. role 기반 분리: warning/caution은 독립 청크 또는 인접 블록에 병합
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import CHUNK_BLOCK_COUNT, CHUNK_MAX_SIZE, CHUNK_OVERLAP
from src.types.chunk import S1000DChunk
from src.types.dm import ContentBlock, ContentBlockRole, S1000DDmJson


@dataclass
class ChunkingOptions:
    """청킹 전략 설정."""

    block_count: int = CHUNK_BLOCK_COUNT
    max_chars: int = CHUNK_MAX_SIZE
    overlap: int = CHUNK_OVERLAP
    separate_safety: bool = True  # warning/caution을 독립 청크로 분리


_SAFETY_ROLES = {ContentBlockRole.WARNING, ContentBlockRole.CAUTION}


def chunk_dm(dm: S1000DDmJson, options: ChunkingOptions | None = None) -> list[S1000DChunk]:
    """S1000DDmJson → S1000DChunk 리스트 변환.

    1. safety 블록 분리 (옵션)
    2. 나머지 블록을 슬라이딩 윈도우로 묶기
    3. max_chars 초과 시 윈도우를 줄여서 분할
    """
    opts = options or ChunkingOptions()
    blocks = dm.content_blocks
    if not blocks:
        return []

    # applicability를 문자열로 변환
    applic_str = _applic_to_str(dm.applicability)

    chunks: list[S1000DChunk] = []
    chunk_counter = 0

    if opts.separate_safety:
        safety_blocks, normal_blocks = _split_safety(blocks)
    else:
        safety_blocks = []
        normal_blocks = list(blocks)

    # safety 블록들을 독립 청크로 생성
    for sb in safety_blocks:
        chunk_counter += 1
        chunks.append(_make_chunk(
            dm=dm,
            blocks=[sb],
            chunk_num=chunk_counter,
            applic_str=applic_str,
        ))

    # 일반 블록 슬라이딩 윈도우 청킹
    if normal_blocks:
        window_chunks = _sliding_window_chunk(
            normal_blocks, opts.block_count, opts.max_chars, opts.overlap
        )
        for window in window_chunks:
            chunk_counter += 1
            chunks.append(_make_chunk(
                dm=dm,
                blocks=window,
                chunk_num=chunk_counter,
                applic_str=applic_str,
            ))

    return chunks


def _split_safety(
    blocks: list[ContentBlock],
) -> tuple[list[ContentBlock], list[ContentBlock]]:
    """warning/caution 블록을 분리."""
    safety: list[ContentBlock] = []
    normal: list[ContentBlock] = []
    for b in blocks:
        if b.role in _SAFETY_ROLES:
            safety.append(b)
        else:
            normal.append(b)
    return safety, normal


def _sliding_window_chunk(
    blocks: list[ContentBlock],
    block_count: int,
    max_chars: int,
    overlap: int,
) -> list[list[ContentBlock]]:
    """슬라이딩 윈도우로 블록 그룹 생성.

    block_count개씩 묶되, max_chars 초과 시 윈도우를 줄인다.
    overlap만큼 이전 청크와 겹친다.
    """
    result: list[list[ContentBlock]] = []
    i = 0
    n = len(blocks)

    while i < n:
        # block_count개까지 가져오되 max_chars 이내로
        end = min(i + block_count, n)
        window = blocks[i:end]

        # max_chars 초과 시 윈도우 축소
        while len(window) > 1 and _total_chars(window) > max_chars:
            window = window[:-1]
            end -= 1

        # 단일 블록이 max_chars 초과해도 그대로 포함 (분할하지 않음)
        result.append(window)

        # 끝까지 도달했으면 종료
        if end >= n:
            break

        # 다음 시작점: 현재 끝 - overlap
        step = max(len(window) - overlap, 1)
        i += step

    return result


def _total_chars(blocks: list[ContentBlock]) -> int:
    """블록 리스트의 총 문자 수."""
    return sum(len(b.text) for b in blocks)


def _make_chunk(
    dm: S1000DDmJson,
    blocks: list[ContentBlock],
    chunk_num: int,
    applic_str: str,
) -> S1000DChunk:
    """ContentBlock 리스트에서 S1000DChunk 생성."""
    # structure_path_range 계산
    paths = [b.structure_path for b in blocks if b.structure_path]
    if len(paths) == 0:
        path_range = ""
    elif len(paths) == 1:
        path_range = paths[0]
    else:
        path_range = f"{paths[0]} ~ {paths[-1]}"

    # 텍스트 합산
    text = "\n".join(b.text for b in blocks)

    # 블록 role 분포를 메타데이터에 포함
    role_counts: dict[str, int] = {}
    for b in blocks:
        role_counts[b.role.value] = role_counts.get(b.role.value, 0) + 1

    return S1000DChunk(
        dmc=dm.dmc,
        chunk_id=f"{dm.dmc}__chunk-{chunk_num:03d}",
        dm_type=dm.dm_type,
        security=dm.security,
        applicability=applic_str,
        structure_path_range=path_range,
        text=text,
        metadata={
            "title": dm.title,
            "issue": dm.issue,
            "language": dm.language,
            "block_count": len(blocks),
            "block_ids": [b.id for b in blocks],
            "role_distribution": role_counts,
            "source_file": dm.meta.get("source_file", ""),
            "source_path": dm.meta.get("source_path", ""),
        },
    )


def _applic_to_str(applicability: str | dict[str, str]) -> str:
    """applicability를 문자열로 변환."""
    if isinstance(applicability, str):
        return applicability
    # dict → "key=value, ..." 형태
    return ", ".join(f"{k}={v}" for k, v in applicability.items())
