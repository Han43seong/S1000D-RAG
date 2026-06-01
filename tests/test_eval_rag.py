from __future__ import annotations

import json
import subprocess
import sys

from scripts.eval_rag import load_questions


def test_load_questions_validates_minimal_json(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": [{"id": "q1", "question": "What?"}]}), encoding="utf-8")

    data = load_questions(path)

    assert data["questions"][0]["id"] == "q1"


def test_load_questions_accepts_multimodal_entries(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "modality": "mixed",
                "questions": [{"id": "q1", "question": "What is shown?", "modality": "image"}],
            }
        ),
        encoding="utf-8",
    )

    data = load_questions(path)

    assert data["questions"][0]["modality"] == "image"


def test_eval_rag_list_questions_mode_does_not_require_models():
    result = subprocess.run(
        [sys.executable, "scripts/eval_rag.py", "--list-questions"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dataset:" in result.stdout
    assert "bike_overview_description" in result.stdout
    assert "bike_figure_identification_readiness (image requires_vlm)" in result.stdout


def test_eval_rag_check_config_mode_does_not_require_models():
    result = subprocess.run(
        [sys.executable, "scripts/eval_rag.py", "--check-config"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "questions:" in result.stdout
    assert "embedding_model:" in result.stdout
    assert "modalities:" in result.stdout
    assert "vlm_model_profile:" in result.stdout
