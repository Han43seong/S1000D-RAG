"""ChromaDB 인덱스를 Android 앱용 바이너리/JSON으로 내보내기.

사용법:
    python export_index.py [--output-dir DIR]

출력:
    export/embeddings.bin   - float32 임베딩 벡터 (N x D)
    export/metadata.json    - 각 chunk의 메타데이터 + 텍스트
    export/manifest.json    - 인덱스 메타 정보 (차원, chunk 수 등)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_PATH
from src.rag.models import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "export"


def _build_summary_text(
    metadatas: list[dict],
    documents: list[str],
) -> str:
    """메타데이터를 분석하여 교범 전체 요약 텍스트를 생성한다."""
    from collections import defaultdict

    # SNS 코드별 DM 정보 수집
    sns_info: dict[str, dict] = defaultdict(lambda: {"titles": set(), "dm_types": set(), "count": 0})
    for meta in metadatas:
        dmc = meta.get("dmc", "")
        if not dmc:
            continue
        # SNS 코드 추출: S1000DBIKE-AAA-Dxx-... → Dxx 부분
        parts = dmc.split("-")
        if len(parts) >= 3:
            sns = parts[2]  # D00, DA0, DA1, ...
        else:
            sns = "기타"
        sns_info[sns]["titles"].add(meta.get("title", ""))
        sns_info[sns]["dm_types"].add(meta.get("dm_type", ""))
        sns_info[sns]["count"] += 1

    # SNS 코드 → 한글 서브시스템 이름 매핑
    SNS_NAMES = {
        "D00": "자전거 전체 (Bicycle)",
        "DA0": "바퀴/타이어 (Wheel/Tire)",
        "DA1": "브레이크 (Brake)",
        "DA2": "핸들/조향 (Handlebar/Steering)",
        "DA3": "프레임 (Frame)",
        "DA4": "체인/구동계 (Chain/Drivetrain)",
        "DA5": "기어/변속 (Gear)",
        "DA6": "조명 (Lighting)",
    }

    # 정보 코드 유형 → 한글 매핑
    DM_TYPE_NAMES = {
        "descriptive": "구조/설명",
        "procedural": "절차/정비",
        "fault": "고장탐구",
    }

    lines = [
        "이 기술 교범은 자전거(S1000DBIKE)에 대한 S1000D 기술 문서입니다.",
        "",
        "서브시스템 구성:",
    ]

    for sns in sorted(sns_info.keys()):
        info = sns_info[sns]
        name = SNS_NAMES.get(sns, sns)
        types_kr = [DM_TYPE_NAMES.get(t, t) for t in sorted(info["dm_types"]) if t]
        type_str = ", ".join(types_kr) if types_kr else ""
        titles = sorted(t for t in info["titles"] if t)
        title_str = ", ".join(titles[:5])  # 최대 5개 타이틀
        line = f"- {sns}: {name}"
        if type_str:
            line += f" - {type_str}"
        if title_str:
            line += f" ({title_str})"
        lines.append(line)

    n_dmcs = len(set(m.get("dmc", "") for m in metadatas if m.get("dmc", "")))
    lines.append("")
    lines.append(f"총 {len(metadatas)}개 청크, {n_dmcs}개 DM 문서 포함.")

    return "\n".join(lines)


def export_index(
    chroma_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """ChromaDB 컬렉션에서 임베딩 + 메타데이터를 내보낸다."""

    # ChromaDB 클라이언트
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)

    # 모든 데이터 가져오기
    results = collection.get(
        include=["embeddings", "metadatas", "documents"],
    )

    ids = list(results["ids"])
    embeddings = [list(v) for v in results["embeddings"]]
    metadatas = list(results["metadatas"])
    documents = list(results["documents"])

    if embeddings is None or len(embeddings) == 0:
        print("ERROR: 컬렉션에 데이터가 없습니다.", file=sys.stderr)
        sys.exit(1)

    n_dims = len(embeddings[0])

    print(f"컬렉션: {collection_name}")
    print(f"  원본 chunks: {len(embeddings)}")
    print(f"  차원: {n_dims}")

    # ── 요약 청크 자동 생성 ──
    summary_text = _build_summary_text(metadatas, documents)
    print(f"\n요약 청크 생성 중 (임베딩 계산)...")
    emb_model = get_embeddings()
    summary_vec = emb_model.embed_documents([summary_text])[0]

    # 기존 데이터에 요약 청크 추가
    ids.append("SUMMARY_0")
    embeddings.append(summary_vec)
    metadatas.append({
        "dmc": "SUMMARY",
        "chunk_id": "SUMMARY_0",
        "dm_type": "summary",
        "security": "",
        "applicability": "",
        "structure_path_range": "",
        "title": "교범 전체 요약",
    })
    documents.append(summary_text)
    print(f"  요약 청크 추가 완료 ({len(summary_text)} chars)")

    n_chunks = len(embeddings)

    print(f"  총 chunks (요약 포함): {n_chunks}")

    # 출력 디렉터리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- embeddings.bin ---
    # 헤더: [n_chunks (int32), n_dims (int32)] + float32 배열
    embeddings_path = output_dir / "embeddings.bin"
    with open(embeddings_path, "wb") as f:
        f.write(struct.pack("<ii", n_chunks, n_dims))
        for vec in embeddings:
            f.write(struct.pack(f"<{n_dims}f", *vec))

    emb_size = embeddings_path.stat().st_size
    print(f"  embeddings.bin: {emb_size:,} bytes ({emb_size / 1024:.1f} KB)")

    # --- metadata.json ---
    metadata_list = []
    for i in range(n_chunks):
        entry = {
            "id": ids[i],
            "text": documents[i],
            "dmc": metadatas[i].get("dmc", ""),
            "chunk_id": metadatas[i].get("chunk_id", ""),
            "dm_type": metadatas[i].get("dm_type", ""),
            "security": metadatas[i].get("security", ""),
            "applicability": metadatas[i].get("applicability", ""),
            "structure_path_range": metadatas[i].get("structure_path_range", ""),
            "title": metadatas[i].get("title", ""),
        }
        metadata_list.append(entry)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta_size = metadata_path.stat().st_size
    print(f"  metadata.json: {meta_size:,} bytes ({meta_size / 1024:.1f} KB)")

    # --- manifest.json ---
    unique_dmcs = set(m.get("dmc", "") for m in metadatas)
    manifest = {
        "version": 1,
        "n_chunks": n_chunks,
        "n_dims": n_dims,
        "n_dmcs": len(unique_dmcs),
        "collection_name": collection_name,
        "embedding_model": "BGE-m3-ko",
        "files": {
            "embeddings": "embeddings.bin",
            "metadata": "metadata.json",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  manifest.json: {manifest_path.stat().st_size} bytes")
    print(f"\n내보내기 완료: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB → Android 인덱스 내보내기")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 디렉터리 (기본: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=CHROMA_PERSIST_DIR,
        help=f"ChromaDB 디렉터리 (기본: {CHROMA_PERSIST_DIR})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=CHROMA_COLLECTION_NAME,
        help=f"컬렉션 이름 (기본: {CHROMA_COLLECTION_NAME})",
    )
    args = parser.parse_args()

    export_index(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
