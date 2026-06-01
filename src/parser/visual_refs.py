"""Small XML helpers for extracting S1000D visual references.

The helpers are intentionally dependency-light and metadata-only. They do not
resolve, parse, or download image assets; they only surface figure/table/graphic
references that can later be connected to DMC chunk metadata or a VLM runtime.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from src.types.visual import VisualArtifactKind, VisualArtifactRef


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children_named(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in list(element) if _local_name(child.tag) == name)


def _first_child_text(element: ET.Element, name: str) -> str | None:
    for child in _children_named(element, name):
        text = " ".join(part.strip() for part in child.itertext() if part and part.strip())
        return text or None
    return None


def _element_path(element: ET.Element, index: int) -> str:
    ref_id = element.get("id")
    suffix = f"#{ref_id}" if ref_id else f"[{index}]"
    return f"content//{_local_name(element.tag)}{suffix}"


def extract_visual_refs_from_xml(
    xml: str,
    *,
    dmc: str | None = None,
    source_path: str | Path | None = None,
) -> list[VisualArtifactRef]:
    """Extract figure/table visual references from a tiny S1000D XML string.

    Args:
        xml: Raw DM XML content.
        dmc: Optional DMC to attach to each reference.
        source_path: Optional source XML path for traceability.

    Raises:
        ValueError: When the XML is not well-formed.
    """

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML while extracting visual references: {exc}") from exc

    refs: list[VisualArtifactRef] = []
    source = Path(source_path) if source_path is not None else None
    counters = {"figure": 0, "table": 0}

    for element in root.iter():
        name = _local_name(element.tag)
        if name not in counters:
            continue
        counters[name] += 1
        title = _first_child_text(element, "title")
        if name == "figure":
            graphics = list(_children_named(element, "graphic"))
            if graphics:
                for graphic_idx, graphic in enumerate(graphics, start=1):
                    refs.append(
                        VisualArtifactRef(
                            kind=VisualArtifactKind.FIGURE,
                            ref_id=element.get("id"),
                            title=title,
                            info_entity_ident=graphic.get("infoEntityIdent"),
                            dmc=dmc,
                            structure_path=f"{_element_path(element, counters[name])}/graphic[{graphic_idx}]",
                            source_path=source,
                            metadata={"graphic_index": graphic_idx},
                        )
                    )
            else:
                refs.append(
                    VisualArtifactRef(
                        kind=VisualArtifactKind.FIGURE,
                        ref_id=element.get("id"),
                        title=title,
                        dmc=dmc,
                        structure_path=_element_path(element, counters[name]),
                        source_path=source,
                    )
                )
        elif name == "table":
            refs.append(
                VisualArtifactRef(
                    kind=VisualArtifactKind.TABLE,
                    ref_id=element.get("id"),
                    title=title,
                    dmc=dmc,
                    structure_path=_element_path(element, counters[name]),
                    source_path=source,
                )
            )

    return refs
