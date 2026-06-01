"""S1000D DM XML 인제스천 CLI.

사용법:
    python ingest.py [XML_DIR] [--data-dir DIR] [--chroma-dir DIR] [--collection NAME]

기본값:
    XML_DIR = src.config.S1000D_DATA_DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, PROJECT_ROOT, S1000D_DATA_DIR
from src.csdb.adapter import DmFilter
from src.csdb.local_adapter import LocalCsdbAdapter
from src.parser.dm_parser import parse_dm_xml
from src.types.chunk import S1000DChunk


def _load_chunker_symbols() -> tuple[Any, Any]:
    """Load chunker.py without importing src.chunker.__init__ (which needs LangChain)."""
    chunker_path = PROJECT_ROOT / "src" / "chunker" / "chunker.py"
    spec = importlib.util.spec_from_file_location("s1000d_ingest_chunker", chunker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load chunker module from {chunker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ChunkingOptions, module.chunk_dm


ChunkingOptions, chunk_dm = _load_chunker_symbols()

DEFAULT_XML_DIR = S1000D_DATA_DIR


@dataclass(frozen=True)
class IngestScanResult:
    data_dir: Path
    dm_file_count: int
    parse_success_count: int
    parse_error_count: int
    chunk_count: int
    parse_errors: list[str]


def _list_dmcs(adapter: LocalCsdbAdapter, model_ident_code: str | None, limit: int | None) -> list[str]:
    import asyncio

    dm_filter = DmFilter(model_ident_code=model_ident_code) if model_ident_code else None
    dmcs = asyncio.run(adapter.list_data_modules(dm_filter))
    if limit is not None:
        dmcs = dmcs[:limit]
    return dmcs


def _parse_dmcs(
    adapter: LocalCsdbAdapter,
    dmcs: list[str],
    chunk_options: Any,
    *,
    collect_chunks: bool,
) -> tuple[list[S1000DChunk], list[str]]:
    import asyncio

    all_chunks: list[S1000DChunk] = []
    parse_errors: list[str] = []

    for dmc in dmcs:
        try:
            xml_str = asyncio.run(adapter.get_data_module_xml(dmc))
            dm_json = parse_dm_xml(xml_str)
            if hasattr(adapter, "get_data_module_path"):
                source_path = adapter.get_data_module_path(dmc)
                dm_json.meta["source_file"] = source_path.name
                dm_json.meta["source_path"] = str(source_path)
            chunks = chunk_dm(dm_json, chunk_options)
            if collect_chunks:
                all_chunks.extend(chunks)
            print(f"  ✓ {dmc} → {len(dm_json.content_blocks)} blocks → {len(chunks)} chunks")
        except Exception as e:
            parse_errors.append(f"{dmc}: {e}")
            print(f"  ✗ {dmc}: {e}")

    return all_chunks, parse_errors


def scan_data_modules(
    xml_dir: Path,
    chunk_options: Any | None = None,
    model_ident_code: str | None = None,
    limit: int | None = None,
    *,
    collect_chunks: bool = False,
) -> IngestScanResult:
    """Scan and parse DMs without loading embeddings or touching ChromaDB."""
    adapter = LocalCsdbAdapter(xml_dir)
    opts = chunk_options or ChunkingOptions()
    dmcs = _list_dmcs(adapter, model_ident_code, limit)
    print(f"[dry-run] Data dir: {xml_dir}")
    print(f"[dry-run] DM file count: {len(dmcs)}")

    all_chunks, parse_errors = _parse_dmcs(adapter, dmcs, opts, collect_chunks=collect_chunks)
    success_count = len(dmcs) - len(parse_errors)
    print(f"[dry-run] Parse success count: {success_count}")
    print(f"[dry-run] Parse error count: {len(parse_errors)}")

    return IngestScanResult(
        data_dir=xml_dir,
        dm_file_count=len(dmcs),
        parse_success_count=success_count,
        parse_error_count=len(parse_errors),
        chunk_count=len(all_chunks),
        parse_errors=parse_errors,
    )


def _current_git_commit(repo_root: Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def build_index_manifest(
    *,
    data_dir: str | Path,
    collection_name: str,
    dm_count: int,
    parse_success_count: int,
    parse_error_count: int,
    chunk_count: int,
    dmcs: list[str],
    parse_errors: list[str],
    created_at: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build dependency-light index manifest data for unit tests and ingest."""
    from src.runtime.model_registry import get_model_runtime_config

    cfg = get_model_runtime_config()
    return {
        "data_dir": str(data_dir),
        "collection_name": collection_name,
        "dm_count": dm_count,
        "parse_success_count": parse_success_count,
        "parse_error_count": parse_error_count,
        "chunk_count": chunk_count,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "model_backend": cfg.backend,
        "text_model_profile": cfg.text_profile.name,
        "vlm_model_profile": cfg.vlm_profile.name,
        "embedding_model": cfg.embedding.model,
        "reranker_model": cfg.reranker.model,
        "git_commit": git_commit if git_commit is not None else _current_git_commit(),
        "sample_dmcs": dmcs[:10],
        "sample_errors": parse_errors[:10],
    }


def write_index_manifest(manifest: dict[str, Any], chroma_dir: str | Path) -> Path:
    """Write manifest.json next to the selected Chroma persist directory."""
    target_dir = Path(chroma_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _reset_chroma_dir(chroma_dir: str | Path) -> None:
    """Safely remove only the selected Chroma persist directory."""
    target = Path(chroma_dir).expanduser().resolve()
    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
        (PROJECT_ROOT / "docs").resolve(),
    }
    if target in forbidden or target == target.parent:
        raise ValueError(f"Refusing to reset unsafe Chroma directory: {target}")
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"Chroma path is not a directory: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def ingest(
    xml_dir: Path,
    chroma_dir: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    chunk_options: Any | None = None,
    model_ident_code: str | None = None,
    limit: int | None = None,
    reset_index: bool = False,
) -> int:
    """XML 디렉터리를 스캔하여 인제스천 실행.

    Returns:
        인덱싱된 총 청크 수.
    """
    adapter = LocalCsdbAdapter(xml_dir)
    opts = chunk_options or ChunkingOptions()

    # 1. DM 목록 스캔
    dmcs = _list_dmcs(adapter, model_ident_code, limit)
    print(f"[1/4] {len(dmcs)}개 DM 파일 발견")

    if not dmcs:
        print("인덱싱할 DM이 없습니다.")
        return 0

    # 2. 파싱 + 청킹
    all_chunks, parse_errors = _parse_dmcs(adapter, dmcs, opts, collect_chunks=True)

    print(f"[2/4] 파싱 완료: {len(all_chunks)} chunks ({len(parse_errors)} errors)")

    if not all_chunks:
        print("인덱싱할 청크가 없습니다.")
        return 0

    # 3. Document 변환
    from src.chunker.indexer import chunks_to_documents, build_chroma_index

    documents = chunks_to_documents(all_chunks)
    print(f"[3/4] {len(documents)}개 Document 변환 완료")

    # 4. 임베딩 + ChromaDB 인덱싱
    if reset_index:
        _reset_chroma_dir(chroma_dir)
        print(f"[4/4] 기존 ChromaDB 디렉터리 초기화 완료: {chroma_dir}")
    print("[4/4] 임베딩 모델 로딩 + ChromaDB 인덱싱...")
    t0 = time.time()
    from src.rag.models import get_embeddings

    embedding_fn = get_embeddings()
    build_chroma_index(
        documents=documents,
        embedding_fn=embedding_fn,
        persist_directory=chroma_dir,
        collection_name=collection_name,
    )
    manifest = build_index_manifest(
        data_dir=xml_dir,
        collection_name=collection_name,
        dm_count=len(dmcs),
        parse_success_count=len(dmcs) - len(parse_errors),
        parse_error_count=len(parse_errors),
        chunk_count=len(all_chunks),
        dmcs=dmcs,
        parse_errors=parse_errors,
    )
    manifest_path = write_index_manifest(manifest, chroma_dir)
    elapsed = time.time() - t0
    print(f"  완료! ({elapsed:.1f}s)")

    print(f"\n=== 인제스천 결과 ===")
    print(f"  DM 파일: {len(dmcs)}개")
    print(f"  파싱 성공: {len(dmcs) - len(parse_errors)}개")
    print(f"  총 청크: {len(all_chunks)}개")
    print(f"  ChromaDB: {chroma_dir} / {collection_name}")
    print(f"  Manifest: {manifest_path}")

    if parse_errors:
        print(f"\n  파싱 실패:")
        for err in parse_errors:
            print(f"    - {err}")

    return len(all_chunks)


def _resolve_xml_dir(positional_xml_dir: str | None, data_dir: str | None) -> Path:
    # Explicit --data-dir wins over the backward-compatible positional XML_DIR.
    if data_dir:
        return Path(data_dir)
    if positional_xml_dir:
        return Path(positional_xml_dir)
    return DEFAULT_XML_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S1000D DM XML 인제스천 CLI"
    )
    parser.add_argument(
        "xml_dir",
        nargs="?",
        default=None,
        help="DM XML 파일이 있는 디렉터리 경로 (하위 호환용 positional 인자)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="DM XML 파일이 있는 디렉터리 경로 (--data-dir가 positional XML_DIR보다 우선)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩/ChromaDB 없이 DM 스캔과 XML 파싱까지만 수행",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 DM 개수 제한 (빠른 스모크 테스트용)",
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="인덱싱 전에 선택된 ChromaDB 영속화 디렉터리만 안전하게 삭제",
    )
    parser.add_argument(
        "--chroma-dir",
        default=CHROMA_PERSIST_DIR,
        help="ChromaDB 영속화 디렉터리",
    )
    parser.add_argument(
        "--collection",
        default=CHROMA_COLLECTION_NAME,
        help="ChromaDB 컬렉션 이름",
    )
    parser.add_argument(
        "--block-count",
        type=int,
        default=5,
        help="청크당 블록 수 (기본: 5)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1500,
        help="청크 최대 문자 수 (기본: 1500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="청크 오버랩 블록 수 (기본: 1)",
    )
    parser.add_argument(
        "--model-ident",
        type=str,
        default=None,
        help="모델식별코드 필터 (예: S1000DBIKE). 지정 시 해당 코드의 DM만 인제스천",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        print("Error: --limit은 0 이상이어야 합니다.")
        sys.exit(1)

    xml_dir = _resolve_xml_dir(args.xml_dir, args.data_dir)
    if not xml_dir.is_dir():
        print(f"Error: 디렉터리를 찾을 수 없습니다: {xml_dir}")
        sys.exit(1)

    chunk_opts = ChunkingOptions(
        block_count=args.block_count,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )

    if args.dry_run:
        result = scan_data_modules(
            xml_dir=xml_dir,
            chunk_options=chunk_opts,
            model_ident_code=args.model_ident,
            limit=args.limit,
        )
        sys.exit(0 if result.dm_file_count > 0 and result.parse_success_count > 0 else 1)

    total = ingest(
        xml_dir=xml_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        chunk_options=chunk_opts,
        model_ident_code=args.model_ident,
        limit=args.limit,
        reset_index=args.reset_index,
    )

    sys.exit(0 if total > 0 else 1)


if __name__ == "__main__":
    main()
