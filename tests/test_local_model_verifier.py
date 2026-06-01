from __future__ import annotations

import json
import subprocess
import sys

from scripts import verify_local_models


HEAVY_MODULES = [
    "llama_cpp",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "sentence_transformers",
    "chromadb",
    "torch",
    "transformers",
]


def _write_fake_stack(root):
    for expected in verify_local_models.EXPECTED_ARTIFACTS:
        path = root / expected.path
        if expected.kind == "directory":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fake {expected.path}\n".encode("utf-8"))


def test_build_manifest_reports_complete_fake_stack(tmp_path):
    _write_fake_stack(tmp_path)

    manifest = verify_local_models.build_manifest(tmp_path)

    assert manifest["summary"]["status"] == "complete"
    assert manifest["summary"]["missing_required_count"] == 0
    assert manifest["dependency_light"] is True
    assert manifest["no_model_weight_loads"] is True
    assert manifest["no_benchmark_metrics_claimed"] is True
    paths = {item["path"] for item in manifest["artifacts"]}
    assert "models/llm/qwen36-27b/Qwen3.6-27B-IQ4_NL.gguf" in paths
    assert "models/embedding/bge-m3/pytorch_model.bin" in paths
    assert "models/reranker/bge-reranker-v2-m3/model.safetensors" in paths
    assert all(item["status"] == "present" for item in manifest["artifacts"])


def test_missing_required_artifacts_make_cli_fail_unless_no_fail(tmp_path):
    output_path = tmp_path / "manifest.json"

    failing = subprocess.run(
        [sys.executable, "scripts/verify_local_models.py", "--root", str(tmp_path), "--output", str(output_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failing.returncode == 1
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["status"] == "incomplete"
    assert manifest["summary"]["missing_required_count"] == len(verify_local_models.EXPECTED_ARTIFACTS)

    no_fail = subprocess.run(
        [sys.executable, "scripts/verify_local_models.py", "--root", str(tmp_path), "--no-fail"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_fail.returncode == 0
    assert "Status: incomplete" in no_fail.stdout


def test_cli_json_output_and_optional_manifest(tmp_path):
    _write_fake_stack(tmp_path)
    output_path = tmp_path / "reports" / "local-models.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_local_models.py",
            "--root",
            str(tmp_path),
            "--format",
            "json",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    stdout_manifest = json.loads(completed.stdout)
    file_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_manifest["summary"]["status"] == "complete"
    assert file_manifest["summary"]["status"] == "complete"
    assert stdout_manifest["summary"]["required_count"] == len(verify_local_models.EXPECTED_ARTIFACTS)


def test_import_does_not_import_heavy_ml_modules(monkeypatch):
    for module_name in ["scripts.verify_local_models", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import scripts.verify_local_models as verifier

    verifier.build_manifest
    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
