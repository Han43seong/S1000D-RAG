"""Dependency-light VLM interface scaffolding for S1000D-RAG.

This module defines the multimodal request/response contract without importing
or loading heavy VLM runtimes. A disabled client is returned until a future task
wires a concrete backend such as Qwen3-VL GGUF + mmproj.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class VlmImageInput:
    """Image payload placeholder for future VLM calls.

    Exactly one of path or bytes_data is expected by real backends. The current
    disabled backend never reads either form.
    """

    path: str | Path | None = None
    bytes_data: bytes | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualEvidenceRequest:
    """Prompt plus visual evidence metadata for a VLM backend."""

    prompt: str
    images: tuple[VlmImageInput, ...] = ()
    image_paths: tuple[str | Path, ...] = ()
    context_metadata: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None


@dataclass(frozen=True)
class VisualEvidenceResponse:
    """Normalized VLM response returned by concrete backends."""

    text: str
    model_profile: str
    latency_sec: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VlmClient(Protocol):
    """Protocol all VLM backends must satisfy."""

    model_profile: str

    def generate(self, request: VisualEvidenceRequest) -> VisualEvidenceResponse:
        """Generate text from visual evidence."""


class DisabledVlmClient:
    """Dependency-free placeholder that fails clearly on inference."""

    def __init__(
        self,
        *,
        model_profile: str,
        backend: str,
        model_path: str | None,
        mmproj_path: str | None,
    ) -> None:
        self.model_profile = model_profile
        self.backend = backend
        self.model_path = model_path
        self.mmproj_path = mmproj_path

    def generate(self, request: VisualEvidenceRequest) -> VisualEvidenceResponse:
        missing = []
        if not self.model_path:
            missing.append("S1000D_VLM_MODEL_PATH")
        if not self.mmproj_path:
            missing.append("S1000D_VLM_MMPROJ_PATH")
        missing_text = f" Missing configuration: {', '.join(missing)}." if missing else ""
        raise RuntimeError(
            "VLM inference is not enabled in this build. "
            f"Configured profile={self.model_profile!r}, backend={self.backend!r}."
            f"{missing_text} Future VLM support should provide a concrete backend without coupling RAG callers "
            "to runtime-specific imports."
        )


def get_vlm_client() -> VlmClient:
    """Return the configured VLM client, reading runtime config lazily.

    Importing this module is dependency-free; the registry is imported only when
    a caller asks for a client. Until a concrete VLM runtime is implemented, this
    returns a disabled client that raises a clear RuntimeError on generate().
    """

    from src.runtime.model_registry import get_model_runtime_config

    cfg = get_model_runtime_config()
    return DisabledVlmClient(
        model_profile=cfg.vlm_profile.name,
        backend=cfg.backend,
        model_path=cfg.vlm_model_path,
        mmproj_path=cfg.vlm_mmproj_path,
    )
