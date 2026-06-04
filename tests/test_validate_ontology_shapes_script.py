import subprocess
import sys


def test_validate_ontology_shapes_script_reports_current_manifest_is_valid():
    result = subprocess.run(
        [sys.executable, "scripts/validate_ontology_shapes.py"],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ontology shape validation passed" in result.stdout
    assert "issues=0" in result.stdout
