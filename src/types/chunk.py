from __future__ import annotations

from pydantic import BaseModel

from .dm import DmType


class S1000DChunk(BaseModel):
    dmc: str
    chunk_id: str
    dm_type: DmType
    security: str
    applicability: str
    structure_path_range: str
    text: str
    metadata: dict = {}
