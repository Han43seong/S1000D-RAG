"""Dependency-light visual artifact types for multimodal S1000D indexing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class VisualArtifactKind(str, Enum):
    """S1000D visual evidence categories that may be linked to chunks later."""

    FIGURE = "figure"
    TABLE = "table"
    ILLUSTRATION = "illustration"
    GRAPHIC = "graphic"


@dataclass(frozen=True)
class VisualArtifactRef:
    """Metadata-only reference to a visual artifact found in a data module."""

    kind: VisualArtifactKind
    ref_id: str | None = None
    title: str | None = None
    info_entity_ident: str | None = None
    dmc: str | None = None
    structure_path: str | None = None
    source_path: Path | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stable_key(self) -> str:
        """Return a deterministic key suitable for future index metadata."""

        parts = [self.dmc or "unknown-dmc", self.kind.value, self.ref_id or self.info_entity_ident or "unidentified"]
        return ":".join(parts)
