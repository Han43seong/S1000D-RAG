from __future__ import annotations

import json
import os
import subprocess
import sys

from scripts import local_model_env, verify_local_models


def _write_fake_stack(root):
    for expected in verify_local_models.EXPECTED_ARTIFACTS:
        path = root / expected.path
        if expected.kind == "directory":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake\n")


def test_build_env_uses_resolved_first_pass_paths(tmp_path):
    env = local_model_env.build_env(tmp_path)

    assert env["S1000D_TEXT_MODEL_PROFILE"] == "qwen3_8b_q5"
    assert env["S1000D_VLM_MODEL_PROFILE"] == "qwen3_vl_8b_q4"
    assert env["S1000D_TEXT_MODEL_PATH"] == str(
        (tmp_path / "models/llm/qwen3-8b/Qwen3-8B-Q5_K_M.gguf").resolve()
    )
    assert env["S1000D_VLM_MODEL_PATH"] == str(
        (tmp_path / "models/vlm/qwen3-vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf").resolve()
    )
    assert env["S1000D_VLM_MMPROJ_PATH"] == str(
        (tmp_path / "models/vlm/qwen3-vl-8b/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf").resolve()
    )
    assert env["S1000D_EMBEDDING_MODEL"] == str((tmp_path / "models/embedding/bge-m3").resolve())
    assert env["S1000D_RERANKER_MODEL"] == str((tmp_path / "models/reranker/bge-reranker-v2-m3").resolve())
    assert env["S1000D_CHROMA_PERSIST_DIR"] == str((tmp_path / "chroma_db_full").resolve())
    assert env["S1000D_CHROMA_COLLECTION_NAME"] == "s1000d_chunks_full"
    assert env["S1000D_MODEL_BACKEND"] == "llama_cpp_python"
    assert env["S1000D_LLM_N_CTX"] == "2048"
    assert env["S1000D_LLM_MAX_TOKENS"] == "1024"


def test_default_shell_output_is_eval_safe(tmp_path):
    completed = subprocess.run(
        [sys.executable, "scripts/local_model_env.py", "--root", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "export S1000D_TEXT_MODEL_PROFILE=qwen3_8b_q5" in completed.stdout
    assert "export S1000D_MODEL_BACKEND=llama_cpp_python" in completed.stdout
    assert "export S1000D_CHROMA_COLLECTION_NAME=s1000d_chunks_full" in completed.stdout
    assert str((tmp_path / "chroma_db_full").resolve()) in completed.stdout
    assert "export S1000D_LLM_N_CTX=2048" in completed.stdout
    assert "export S1000D_LLM_MAX_TOKENS=1024" in completed.stdout
    assert str((tmp_path / "models/embedding/bge-m3").resolve()) in completed.stdout

    eval_check = subprocess.run(
        [
            "bash",
            "-c",
            'eval "$($PYTHON scripts/local_model_env.py --root "$ROOT")"; printf "%s" "$S1000D_TEXT_MODEL_PATH"',
        ],
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHON": sys.executable, "ROOT": str(tmp_path)},
    )
    assert eval_check.stdout == str(
        (tmp_path / "models/llm/qwen3-8b/Qwen3-8B-Q5_K_M.gguf").resolve()
    )


def test_json_check_reports_complete_fake_stack(tmp_path):
    _write_fake_stack(tmp_path)

    completed = subprocess.run(
        [sys.executable, "scripts/local_model_env.py", "--root", str(tmp_path), "--format", "json", "--check"],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["dependency_light"] is True
    assert payload["no_model_weight_loads"] is True
    assert payload["check"]["status"] == "complete"
    assert payload["check"]["missing_required_count"] == 0
    assert payload["env"]["S1000D_RERANKER_MODEL"] == str(
        (tmp_path / "models/reranker/bge-reranker-v2-m3").resolve()
    )


def test_check_fails_when_required_artifacts_are_missing(tmp_path):
    completed = subprocess.run(
        [sys.executable, "scripts/local_model_env.py", "--root", str(tmp_path), "--format", "json", "--check"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["check"]["status"] == "incomplete"
    assert payload["check"]["missing_required_count"] == len(verify_local_models.EXPECTED_ARTIFACTS)


def test_import_does_not_import_heavy_ml_modules(monkeypatch):
    for module_name in ["scripts.local_model_env", *verify_local_models.HEAVY_MODULE_PREFIXES]:
        sys.modules.pop(module_name, None)

    import scripts.local_model_env as env_helper

    env_helper.build_env
    for module_name in verify_local_models.HEAVY_MODULE_PREFIXES:
        assert module_name not in sys.modules
