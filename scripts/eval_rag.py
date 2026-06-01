#!/usr/bin/env python3
"""Lightweight S1000D text RAG evaluation scaffold.

Default modes intentionally avoid Chroma/model imports:
  python scripts/eval_rag.py --list-questions
  python scripts/eval_rag.py --check-config

Retrieval is opt-in with --retrieve and degrades clearly when optional
LangChain/Chroma dependencies or an index are unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions" / "s1000d_bike.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_questions(path: str | Path = DEFAULT_QUESTIONS) -> dict[str, Any]:
    question_path = Path(path)
    with question_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Question file must contain a non-empty 'questions' list: {question_path}")
    allowed_modalities = {"text", "image", "multimodal"}
    for item in questions:
        if not isinstance(item, dict) or not item.get("id") or not item.get("question"):
            raise ValueError("Each question must contain at least 'id' and 'question'.")
        modality = item.get("modality") or ("text" if data.get("modality") == "mixed" else data.get("modality", "text"))
        if modality not in allowed_modalities:
            raise ValueError(f"Unsupported question modality {modality!r}; expected one of {sorted(allowed_modalities)}.")
    return data


def list_questions(path: str | Path = DEFAULT_QUESTIONS) -> int:
    data = load_questions(path)
    print(f"Dataset: {data.get('dataset', 'unknown')}")
    print(f"Modality: {data.get('modality', 'text')}")
    for idx, item in enumerate(data["questions"], start=1):
        hints = ", ".join(item.get("expected_dmc_substrings", [])) or "no DMC hint"
        modality = item.get("modality") or ("text" if data.get("modality") == "mixed" else data.get("modality", "text"))
        vlm_flag = " requires_vlm" if item.get("requires_vlm") else ""
        print(f"{idx}. {item['id']} ({modality}{vlm_flag}): {item['question']} [{hints}]")
    return 0


def check_config(path: str | Path = DEFAULT_QUESTIONS, chroma_dir: str | Path | None = None) -> int:
    data = load_questions(path)
    from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR, S1000D_DATA_DIR
    from src.runtime.model_registry import get_model_runtime_config

    cfg = get_model_runtime_config()
    selected_chroma_dir = Path(chroma_dir or CHROMA_PERSIST_DIR)
    manifest_path = selected_chroma_dir / "manifest.json"
    modality_counts: dict[str, int] = {}
    for item in data["questions"]:
        modality = item.get("modality") or ("text" if data.get("modality") == "mixed" else data.get("modality", "text"))
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    print(f"questions: {len(data['questions'])} ({Path(path)})")
    print(f"modalities: {modality_counts}")
    print(f"data_dir: {S1000D_DATA_DIR} exists={S1000D_DATA_DIR.exists()}")
    print(f"chroma_dir: {selected_chroma_dir} exists={selected_chroma_dir.exists()}")
    print(f"collection_name: {CHROMA_COLLECTION_NAME}")
    print(f"manifest: {manifest_path} exists={manifest_path.exists()}")
    print(f"text_model_profile: {cfg.text_profile.name}")
    print(f"vlm_model_profile: {cfg.vlm_profile.name}")
    print(f"vlm_model_configured: {bool(cfg.vlm_model_path and cfg.vlm_mmproj_path)}")
    print(f"embedding_model: {cfg.embedding.model}")
    print(f"reranker_model: {cfg.reranker.model}")
    return 0


def run_retrieval(path: str | Path = DEFAULT_QUESTIONS, chroma_dir: str | Path | None = None, k: int = 3) -> int:
    data = load_questions(path)
    try:
        from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
        from src.chunker.indexer import load_chroma_index
        from src.rag.models import get_embeddings
    except ImportError as exc:
        print(f"Retrieval unavailable: optional dependency missing: {exc}", file=sys.stderr)
        return 2

    selected_chroma_dir = Path(chroma_dir or CHROMA_PERSIST_DIR)
    if not selected_chroma_dir.exists():
        print(f"Retrieval unavailable: Chroma directory not found: {selected_chroma_dir}", file=sys.stderr)
        return 2

    embedding_fn = get_embeddings()
    vectorstore = load_chroma_index(
        embedding_fn=embedding_fn,
        persist_directory=str(selected_chroma_dir),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    for item in data["questions"]:
        docs = vectorstore.similarity_search(item["question"], k=k)
        print(f"\n## {item['id']}")
        print(item["question"])
        for rank, doc in enumerate(docs, start=1):
            metadata = getattr(doc, "metadata", {})
            print(f"{rank}. {metadata.get('dmc', '')} {metadata.get('chunk_id', '')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight S1000D RAG evaluation helper")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="Question JSON file")
    parser.add_argument("--chroma-dir", default=None, help="Optional Chroma persist directory")
    parser.add_argument("--list-questions", action="store_true", help="List configured questions without loading models")
    parser.add_argument("--check-config", action="store_true", help="Validate eval/config paths without loading models")
    parser.add_argument("--retrieve", action="store_true", help="Run optional retrieval against an existing Chroma index")
    parser.add_argument("-k", type=int, default=3, help="Retrieval result count")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_questions:
        return list_questions(args.questions)
    if args.check_config:
        return check_config(args.questions, args.chroma_dir)
    if args.retrieve:
        return run_retrieval(args.questions, args.chroma_dir, args.k)
    return list_questions(args.questions)


if __name__ == "__main__":
    raise SystemExit(main())
