from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ContentBlockRole(str, Enum):
    TITLE = "title"
    STEP = "step"
    NOTE = "note"
    WARNING = "warning"
    CAUTION = "caution"
    PARA = "para"
    TABLE = "table"
    FIGURE_REF = "figure-ref"


class DmType(str, Enum):
    PROCEDURAL = "procedural"
    DESCRIPTIVE = "descriptive"
    IPD = "ipd"
    FAULT = "fault"
    CREW = "crew"
    PROCESS = "process"


class ContentBlock(BaseModel):
    id: str
    role: ContentBlockRole
    text: str
    structure_path: Optional[str] = None


class S1000DDmJson(BaseModel):
    dmc: str
    dm_type: DmType
    issue: str
    language: str
    security: str
    applicability: str | dict[str, str]
    title: str
    meta: dict = {}
    content_blocks: list[ContentBlock]
