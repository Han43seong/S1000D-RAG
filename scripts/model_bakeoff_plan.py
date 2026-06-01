#!/usr/bin/env python3
"""Offline model/profile bake-off plan generator for S1000D-RAG.

This scaffold is intentionally dependency-light. It may read the local
src.runtime.model_registry metadata, but it must not download models, load model
files, or import heavyweight ML/runtime packages.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = PROJECT_ROOT / "eval" / "model_bakeoff_profiles.json"
PENDING = "pending_measurement"

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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_static_profiles(path: str | Path = DEFAULT_PROFILES) -> dict[str, Any]:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = [
        "hardware_profiles",
        "text_llm_candidates",
        "vlm_candidates",
        "embedding_candidates",
        "reranker_candidates",
        "evaluation_dimensions",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Bake-off profile file missing required keys: {', '.join(missing)}")
    return data


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def read_model_registry() -> dict[str, Any]:
    """Read dependency-free model registry metadata if present."""
    try:
        from src.runtime import model_registry as registry
    except ImportError:
        return {"available": False, "reason": "src.runtime.model_registry unavailable"}

    return {
        "available": True,
        "default_text_profile": getattr(registry, "DEFAULT_TEXT_PROFILE", None),
        "default_vlm_profile": getattr(registry, "DEFAULT_VLM_PROFILE", None),
        "default_embedding_model": getattr(registry, "DEFAULT_EMBEDDING_MODEL", None),
        "default_reranker_model": getattr(registry, "DEFAULT_RERANKER_MODEL", None),
        "text_profiles": {
            name: _dataclass_to_dict(profile)
            for name, profile in getattr(registry, "TEXT_MODEL_PROFILES", {}).items()
        },
        "vlm_profiles": {
            name: _dataclass_to_dict(profile)
            for name, profile in getattr(registry, "VLM_MODEL_PROFILES", {}).items()
        },
    }


def load_measurements(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    measurement_path = Path(path)
    with measurement_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Measurement input must be a JSON object keyed by candidate/profile id.")
    return data


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("profile") or candidate.get("model") or candidate.get("display_name"))


def _metric_block(candidate_id: str, dimensions: list[str], measurements: dict[str, Any]) -> dict[str, Any]:
    candidate_measurements = measurements.get(candidate_id, {})
    if not isinstance(candidate_measurements, dict):
        candidate_measurements = {}
    return {dimension: candidate_measurements.get(dimension, PENDING) for dimension in dimensions}


def _registry_profile(registry: dict[str, Any], kind: str, profile_name: str) -> dict[str, Any]:
    key = "text_profiles" if kind == "text" else "vlm_profiles"
    profiles = registry.get(key, {}) if registry.get("available") else {}
    profile = profiles.get(profile_name, {})
    return profile if isinstance(profile, dict) else {}


def _enrich_candidates(
    candidates: list[dict[str, Any]],
    *,
    kind: str,
    dimensions: list[str],
    measurements: dict[str, Any],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        candidate_id = _candidate_id(item)
        if kind in {"text", "vlm"}:
            item["registry_profile"] = _registry_profile(registry, kind, str(item.get("profile", "")))
        item["metrics"] = _metric_block(candidate_id, dimensions, measurements)
        enriched.append(item)
    return enriched


def build_plan(
    *,
    hardware_profile: str,
    profile_file: str | Path = DEFAULT_PROFILES,
    measurements_file: str | Path | None = None,
) -> dict[str, Any]:
    static = load_static_profiles(profile_file)
    hardware_profiles = static["hardware_profiles"]
    if hardware_profile not in hardware_profiles:
        raise ValueError(
            f"Unknown hardware profile '{hardware_profile}'. Available: {', '.join(sorted(hardware_profiles))}"
        )

    measurements = load_measurements(measurements_file)
    dimensions = list(static["evaluation_dimensions"])
    registry = read_model_registry()

    plan = {
        "schema_version": static.get("schema_version", 1),
        "status": "planned_not_executed",
        "measurement_default": PENDING,
        "no_downloads_or_model_loads": True,
        "hardware_profile": hardware_profile,
        "hardware_constraints": hardware_profiles[hardware_profile],
        "model_registry": registry,
        "evaluation_dimensions": dimensions,
        "candidates": {
            "text_llms": _enrich_candidates(
                static["text_llm_candidates"],
                kind="text",
                dimensions=dimensions,
                measurements=measurements,
                registry=registry,
            ),
            "vlms": _enrich_candidates(
                static["vlm_candidates"],
                kind="vlm",
                dimensions=dimensions,
                measurements=measurements,
                registry=registry,
            ),
            "embeddings": _enrich_candidates(
                static["embedding_candidates"],
                kind="embedding",
                dimensions=dimensions,
                measurements=measurements,
                registry=registry,
            ),
            "rerankers": _enrich_candidates(
                static["reranker_candidates"],
                kind="reranker",
                dimensions=dimensions,
                measurements=measurements,
                registry=registry,
            ),
        },
        "later_commands_not_executed_now": [
            "python scripts/eval_rag.py --check-config",
            "python scripts/eval_rag.py --list-questions",
            "python scripts/eval_rag.py --retrieve --chroma-dir <existing_chroma_dir> -k 3",
            "python scripts/model_bakeoff_plan.py --format markdown --hardware-profile rtx4080super_16gb --measurements <results.json>",
        ],
    }
    return plan


def _markdown_candidate_list(title: str, candidates: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    for candidate in candidates:
        lines.append(f"### {candidate.get('display_name', _candidate_id(candidate))}")
        lines.append(f"- Profile: `{candidate.get('profile', _candidate_id(candidate))}`")
        if candidate.get("model"):
            lines.append(f"- Model: `{candidate['model']}`")
        if candidate.get("planned_quantization"):
            lines.append(f"- Planned quantization: `{candidate['planned_quantization']}`")
        if candidate.get("role"):
            lines.append(f"- Role: {candidate['role']}")
        registry_profile = candidate.get("registry_profile") or {}
        if registry_profile.get("repo_id"):
            lines.append(f"- Registry repo: `{registry_profile['repo_id']}`")
        if registry_profile.get("recommended_first_quant"):
            lines.append(f"- Registry recommended quant: `{registry_profile['recommended_first_quant']}`")
        constraints = candidate.get("qualitative_constraints") or []
        if constraints:
            lines.append("- Qualitative constraints:")
            for constraint in constraints:
                lines.append(f"  - {constraint}")
        lines.append("- Metrics:")
        for metric, value in candidate.get("metrics", {}).items():
            lines.append(f"  - {metric}: `{value}`")
        lines.append("")
    return lines


def render_markdown(plan: dict[str, Any]) -> str:
    hw = plan["hardware_constraints"]
    candidates = plan["candidates"]
    lines = [
        "# S1000D-RAG Offline Model Bake-off Plan",
        "",
        f"Status: `{plan['status']}`",
        "",
        "> This scaffold does not download models, load models, or claim benchmark results. ",
        f"> Unprovided metrics are marked `{PENDING}`.",
        "",
        "## Hardware Constraints",
        "",
        f"- Hardware profile: `{plan['hardware_profile']}`",
        f"- Display name: {hw.get('display_name')}",
        f"- GPU: {hw.get('gpu')}",
        f"- VRAM: {hw.get('vram_gb')} GB",
        f"- Mode: `{hw.get('mode')}`",
        "- Constraints:",
    ]
    for constraint in hw.get("constraints", []):
        lines.append(f"  - {constraint}")
    lines.append("")

    lines.extend(_markdown_candidate_list("Text LLM Candidates", candidates["text_llms"]))
    lines.extend(_markdown_candidate_list("VLM Candidates", candidates["vlms"]))
    lines.extend(_markdown_candidate_list("Embedding/Reranker Candidates", candidates["embeddings"] + candidates["rerankers"]))

    lines.extend([
        "## Evaluation Dimensions",
        "",
    ])
    for dimension in plan["evaluation_dimensions"]:
        lines.append(f"- {dimension}")
    lines.append("")

    lines.extend([
        "## Commands To Run Later (Not Executed Now)",
        "",
    ])
    for command in plan["later_commands_not_executed_now"]:
        lines.append(f"- `{command}`")
    lines.append("")

    all_candidates = candidates["text_llms"] + candidates["vlms"] + candidates["embeddings"] + candidates["rerankers"]
    lines.extend([
        "## Decision Table",
        "",
        "| Candidate | Role | answer_correctness | citation_grounding | visual_grounding | latency | VRAM fit | offline_usability | Decision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for candidate in all_candidates:
        metrics = candidate.get("metrics", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate.get("display_name", _candidate_id(candidate))),
                    str(candidate.get("role", "candidate")),
                    f"`{metrics.get('answer_correctness', PENDING)}`",
                    f"`{metrics.get('citation_grounding', PENDING)}`",
                    f"`{metrics.get('visual_grounding', PENDING)}`",
                    f"`{metrics.get('latency', PENDING)}`",
                    f"`{metrics.get('vram_fit', PENDING)}`",
                    f"`{metrics.get('offline_usability', PENDING)}`",
                    f"`{PENDING}`",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an offline model bake-off plan/report skeleton.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="Output format")
    parser.add_argument("--output", default=None, help="Optional output file. Defaults to stdout.")
    parser.add_argument("--hardware-profile", default="rtx4080super_16gb", help="Hardware profile key")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES), help="Bake-off profile config JSON")
    parser.add_argument("--measurements", default=None, help="Optional JSON measurements to replace pending markers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(
        hardware_profile=args.hardware_profile,
        profile_file=args.profiles,
        measurements_file=args.measurements,
    )
    output = render_markdown(plan) if args.format == "markdown" else render_json(plan)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
