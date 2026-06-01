from __future__ import annotations

import subprocess
import sys

from src.config import S1000D_DATA_DIR


def test_dry_run_limit_does_not_load_models():
    result = subprocess.run(
        [sys.executable, "ingest.py", "--dry-run", "--limit", "1"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[dry-run] Data dir: {S1000D_DATA_DIR}" in result.stdout
    assert "[dry-run] DM file count: 1" in result.stdout
    assert "[dry-run] Parse success count: 1" in result.stdout
    assert "임베딩 모델 로딩" not in result.stdout
