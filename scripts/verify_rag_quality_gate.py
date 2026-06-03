#!/usr/bin/env python3
"""Reproducible S1000D-RAG quality gates.

Examples:
    python scripts/verify_rag_quality_gate.py --suite static
    python scripts/verify_rag_quality_gate.py --suite focused
    python scripts/verify_rag_quality_gate.py --suite ontology-100
    python scripts/verify_rag_quality_gate.py --suite autonomous-500
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("S1000D_PYTHON", sys.executable))

COMMON_ENV = {
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}

FOCUSED_TESTS = [
    "tests/test_rag.py",
    "tests/test_quality_qa_loop.py",
    "tests/test_graph_retrieval.py",
    "tests/test_query_enhancer.py",
    "tests/test_eval_rag.py",
    "tests/test_app_web_runtime_state.py",
    "tests/test_static_app_ui.py",
]

PY_COMPILE_TARGETS = [
    "app_web.py",
    "src/rag/pipeline.py",
    "src/rag/graph_retrieval.py",
    "scripts/run_quality_qa_loop.py",
    "scripts/run_autonomous_500_qa_loop.py",
    "scripts/verify_rag_quality_gate.py",
]


def _run(cmd: list[str], *, timeout: int | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    env = os.environ.copy()
    env.update(COMMON_ENV)
    subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout, check=True)


def run_static() -> None:
    _run(["git", "diff", "--check"], timeout=60)
    _run(["git", "diff", "--cached", "--check"], timeout=60)
    _run([str(PYTHON), "-m", "py_compile", *PY_COMPILE_TARGETS], timeout=120)


def run_focused() -> None:
    run_static()
    _run([str(PYTHON), "-m", "pytest", *FOCUSED_TESTS, "-q"], timeout=300)


def run_ontology_100(out_dir: str) -> None:
    _run(
        [
            str(PYTHON),
            "scripts/run_quality_qa_loop.py",
            "--count",
            "100",
            "--timeout",
            "240",
            "--out-dir",
            out_dir,
        ],
        timeout=60 * 60,
    )


def run_autonomous_500() -> None:
    _run(
        [
            str(PYTHON),
            "scripts/run_autonomous_500_qa_loop.py",
            "--total",
            "500",
            "--timeout",
            "240",
            "--progress-every",
            "25",
            "--max-fix-attempts",
            "3",
        ],
        timeout=4 * 60 * 60,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("static", "focused", "ontology-100", "autonomous-500"),
        default="focused",
        help="Quality gate suite to run.",
    )
    parser.add_argument(
        "--ontology-out-dir",
        default="eval/results/ontology-aware-full",
        help="Output directory for --suite ontology-100.",
    )
    args = parser.parse_args(argv)

    if args.suite == "static":
        run_static()
    elif args.suite == "focused":
        run_focused()
    elif args.suite == "ontology-100":
        run_ontology_100(args.ontology_out_dir)
    elif args.suite == "autonomous-500":
        run_autonomous_500()
    else:  # pragma: no cover - argparse prevents this.
        raise AssertionError(args.suite)

    print(f"\nPASS: {args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
