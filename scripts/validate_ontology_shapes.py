"""Validate the current S1000D ontology manifest against v4 shape rules.

Usage:
    python scripts/validate_ontology_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ontology import load_ontology_manifest
from src.rag.v4.shape_validator import ShapeSeverity, validate_ontology_nodes


def main() -> int:
    nodes = load_ontology_manifest()
    issues = validate_ontology_nodes(nodes)
    if not issues:
        print(f"ontology shape validation passed: nodes={len(nodes)} issues=0")
        return 0

    error_count = sum(1 for issue in issues if issue.severity == ShapeSeverity.ERROR)
    warning_count = len(issues) - error_count
    print(f"ontology shape validation failed: nodes={len(nodes)} issues={len(issues)} errors={error_count} warnings={warning_count}")
    for issue in issues:
        dmc = issue.dmc or "<unknown-dmc>"
        print(f"{issue.severity.value}\t{issue.code}\t{dmc}\t{issue.message}")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
