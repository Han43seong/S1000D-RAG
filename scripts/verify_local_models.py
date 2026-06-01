#!/usr/bin/env python3
"""Verify the local first-pass offline model stack without loading models.

The checks are intentionally filesystem-only: no ML runtimes are imported and no
model weights are opened beyond normal stat/walk metadata inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class ExpectedArtifact:
    category: str
    path: str
    kind: str = "file"
    required: bool = True
    description: str = ""


EXPECTED_ARTIFACTS: tuple[ExpectedArtifact, ...] = (
    ExpectedArtifact(
        category="text_llm",
        path="models/llm/qwen36-27b/Qwen3.6-27B-IQ4_NL.gguf",
        description="Qwen3.6 27B IQ4_NL text LLM GGUF",
    ),
    ExpectedArtifact(
        category="vlm",
        path="models/vlm/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        description="Qwen3-VL 8B Q4_K_M model GGUF",
    ),
    ExpectedArtifact(
        category="vlm",
        path="models/vlm/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
        description="Qwen3-VL multimodal projector GGUF",
    ),
    ExpectedArtifact(
        category="embedding",
        path="models/embedding/bge-m3",
        kind="directory",
        description="BGE-M3 embedding model directory",
    ),
    ExpectedArtifact(category="embedding", path="models/embedding/bge-m3/pytorch_model.bin"),
    ExpectedArtifact(category="embedding", path="models/embedding/bge-m3/config.json"),
    ExpectedArtifact(category="embedding", path="models/embedding/bge-m3/tokenizer.json"),
    ExpectedArtifact(category="embedding", path="models/embedding/bge-m3/tokenizer_config.json"),
    ExpectedArtifact(category="embedding", path="models/embedding/bge-m3/sentencepiece.bpe.model"),
    ExpectedArtifact(
        category="reranker",
        path="models/reranker/bge-reranker-v2-m3",
        kind="directory",
        description="BGE reranker v2 m3 model directory",
    ),
    ExpectedArtifact(category="reranker", path="models/reranker/bge-reranker-v2-m3/model.safetensors"),
    ExpectedArtifact(category="reranker", path="models/reranker/bge-reranker-v2-m3/config.json"),
    ExpectedArtifact(category="reranker", path="models/reranker/bge-reranker-v2-m3/tokenizer.json"),
    ExpectedArtifact(category="reranker", path="models/reranker/bge-reranker-v2-m3/tokenizer_config.json"),
    ExpectedArtifact(category="reranker", path="models/reranker/bge-reranker-v2-m3/sentencepiece.bpe.model"),
)

HEAVY_MODULE_PREFIXES = (
    "llama_cpp",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "sentence_transformers",
    "chromadb",
    "torch",
    "transformers",
)


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
    return 0


def _size_gib(size_bytes: int) -> float:
    return round(size_bytes / BYTES_PER_GIB, 6)


def inspect_artifact(root: Path, expected: ExpectedArtifact) -> dict[str, Any]:
    local_path = Path(expected.path)
    absolute_path = root / local_path
    exists = absolute_path.is_dir() if expected.kind == "directory" else absolute_path.is_file()
    size_bytes = _size_bytes(absolute_path) if exists else 0
    status = "present" if exists else "missing"
    return {
        "category": expected.category,
        "path": expected.path,
        "local_path": expected.path,
        "kind": expected.kind,
        "required": expected.required,
        "description": expected.description,
        "status": status,
        "exists": exists,
        "size_bytes": size_bytes,
        "size_gib": _size_gib(size_bytes),
    }


def build_manifest(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    artifacts = [inspect_artifact(root, expected) for expected in EXPECTED_ARTIFACTS]
    missing_required = [item for item in artifacts if item["required"] and item["status"] != "present"]
    total_size_bytes = sum(item["size_bytes"] for item in artifacts if item["kind"] == "file")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "verifier": "scripts/verify_local_models.py",
        "dependency_light": True,
        "no_model_weight_loads": True,
        "no_benchmark_metrics_claimed": True,
        "summary": {
            "required_count": sum(1 for item in artifacts if item["required"]),
            "present_count": sum(1 for item in artifacts if item["required"] and item["status"] == "present"),
            "missing_required_count": len(missing_required),
            "total_required_file_size_bytes": total_size_bytes,
            "total_required_file_size_gib": _size_gib(total_size_bytes),
            "status": "complete" if not missing_required else "incomplete",
        },
        "artifacts": artifacts,
        "missing_required": [item["path"] for item in missing_required],
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_text(manifest: dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "S1000D-RAG local model verification",
        f"Repo root: {manifest['repo_root']}",
        f"Status: {summary['status']}",
        f"Required artifacts: {summary['present_count']}/{summary['required_count']} present",
        f"Total required file size: {summary['total_required_file_size_bytes']} bytes ({summary['total_required_file_size_gib']} GiB)",
        "",
        "Artifacts:",
    ]
    for item in manifest["artifacts"]:
        lines.append(
            f"- [{item['status']}] {item['category']} {item['local_path']} "
            f"({item['size_bytes']} bytes, {item['size_gib']} GiB)"
        )
    if manifest["missing_required"]:
        lines.extend(["", "Missing required artifacts:"])
        lines.extend(f"- {path}" for path in manifest["missing_required"])
    return "\n".join(lines)


def assert_no_heavy_imports(modules: Iterable[str] = HEAVY_MODULE_PREFIXES) -> None:
    loaded = sorted(name for name in sys.modules for prefix in modules if name == prefix or name.startswith(prefix + "."))
    if loaded:
        raise RuntimeError(f"Verifier loaded heavy ML modules unexpectedly: {', '.join(loaded)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root to inspect (default: script repo root).")
    parser.add_argument("--output", type=Path, help="Optional JSON manifest/report output path.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Stdout format.")
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even when required artifacts are missing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.root)
    assert_no_heavy_imports()

    if args.output:
        write_manifest(manifest, args.output)

    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(render_text(manifest))

    missing_count = manifest["summary"]["missing_required_count"]
    return 0 if args.no_fail or missing_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
