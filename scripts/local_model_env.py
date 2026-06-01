#!/usr/bin/env python3
"""Emit environment variables for the first-pass local model stack.

This helper is intentionally dependency-light: it performs path construction and
optional filesystem checks only. It does not import ML runtimes, load model
weights, run inference, download files, or mutate indexes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import site
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import verify_local_models

ENV_PATHS = {
    "S1000D_TEXT_MODEL_PATH": "models/llm/qwen36-27b/Qwen3.6-27B-IQ4_NL.gguf",
    "S1000D_VLM_MODEL_PATH": "models/vlm/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "S1000D_VLM_MMPROJ_PATH": "models/vlm/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf",
    "S1000D_EMBEDDING_MODEL": "models/embedding/bge-m3",
    "S1000D_RERANKER_MODEL": "models/reranker/bge-reranker-v2-m3",
}

ENV_LITERALS = {
    "S1000D_TEXT_MODEL_PROFILE": "qwen36_27b_iq4",
    "S1000D_VLM_MODEL_PROFILE": "qwen3_vl_8b_q4",
    "S1000D_MODEL_BACKEND": "llama_cpp_python",
}

CUDA_PYTHON_LIB_RELATIVE_PATHS = (
    "nvidia/cuda_runtime/lib",
    "nvidia/cublas/lib",
    "nvidia/curand/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/cuda_nvrtc/lib",
    "nvidia/cudnn/lib",
)

# Keep a stable, readable order for shell output and JSON env objects.
ENV_ORDER = (
    "S1000D_TEXT_MODEL_PROFILE",
    "S1000D_VLM_MODEL_PROFILE",
    "S1000D_TEXT_MODEL_PATH",
    "S1000D_VLM_MODEL_PATH",
    "S1000D_VLM_MMPROJ_PATH",
    "S1000D_EMBEDDING_MODEL",
    "S1000D_RERANKER_MODEL",
    "S1000D_MODEL_BACKEND",
    "LD_LIBRARY_PATH",
)


def _resolve_under_root(root: Path, local_path: str) -> str:
    return str((root / local_path).resolve())


def _cuda_python_lib_paths() -> list[str]:
    """Return existing CUDA shared-library dirs installed by NVIDIA Python wheels."""

    paths: list[str] = []
    for site_dir in site.getsitepackages():
        base = Path(site_dir)
        for relative_path in CUDA_PYTHON_LIB_RELATIVE_PATHS:
            path = base / relative_path
            if path.is_dir():
                paths.append(str(path.resolve()))
    return paths


def build_env(root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Return env assignments for the downloaded first-pass local stack."""

    resolved_root = root.resolve()
    env = dict(ENV_LITERALS)
    env.update({name: _resolve_under_root(resolved_root, path) for name, path in ENV_PATHS.items()})
    cuda_lib_paths = _cuda_python_lib_paths()
    if cuda_lib_paths:
        env["LD_LIBRARY_PATH"] = ":".join(cuda_lib_paths)
    return {name: env[name] for name in ENV_ORDER if name in env}


def build_payload(root: Path = PROJECT_ROOT, check: bool = False) -> dict[str, Any]:
    """Build a machine-readable payload, optionally including verifier status."""

    resolved_root = root.resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "repo_root": str(resolved_root),
        "dependency_light": True,
        "no_model_weight_loads": True,
        "env": build_env(resolved_root),
    }
    if check:
        manifest = verify_local_models.build_manifest(resolved_root)
        payload["check"] = {
            "status": manifest["summary"]["status"],
            "required_count": manifest["summary"]["required_count"],
            "present_count": manifest["summary"]["present_count"],
            "missing_required_count": manifest["summary"]["missing_required_count"],
            "missing_required": manifest["missing_required"],
        }
    return payload


def render_shell(env: dict[str, str]) -> str:
    lines: list[str] = []
    for name, value in env.items():
        if name == "LD_LIBRARY_PATH":
            lines.append(f"export LD_LIBRARY_PATH={shlex.quote(value)}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}")
        else:
            lines.append(f"export {name}={shlex.quote(value)}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root for local model paths.")
    parser.add_argument("--format", choices=("shell", "json"), default="shell", help="Output format.")
    parser.add_argument("--check", action="store_true", help="Fail if required local artifacts are missing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args.root, check=args.check)
    verify_local_models.assert_no_heavy_imports()

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_shell(payload["env"]))

    if args.check and payload["check"]["missing_required_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
