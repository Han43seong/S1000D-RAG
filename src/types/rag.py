from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .dm import DmType


class Evidence(BaseModel):
    dmc: str
    chunk_id: str
    score: float
    dm_type: Optional[DmType] = None
    security: Optional[str] = None
    applicability: Optional[str] = None
    # Optional API/UI bridge fields.  Keeping these nullable preserves existing
    # text-only callers while allowing visual-caption evidence to be serialized
    # without a separate model/index dependency.
    chunk_index: Optional[str] = None
    id: Optional[str] = None
    final_score: Optional[float] = None
    rank: Optional[int] = None
    modality: Optional[str] = None
    content_role: Optional[str] = None
    asset_key: Optional[str] = None
    asset_path: Optional[str] = None
    caption_path: Optional[str] = None
    title: Optional[str] = None
    kind: Optional[str] = None
    ref_id: Optional[str] = None
    display_label: Optional[str] = None
    source_label: Optional[str] = None
    text: Optional[str] = None


class ReferenceMaterialItem(BaseModel):
    id: str
    label: str | None = None
    title: str | None = None
    dmc: str | None = None
    type: str | None = None
    category: str | None = None
    relation: str | None = None
    source_dmc: str | None = None
    target_dmc: str | None = None
    text: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    preview_url: str | None = None
    original_url: str | None = None
    asset_format: str | None = None
    preview_available: bool = False
    preview_status: str | None = None
    # Internal deterministic ordering fields are serialized for traceability;
    # UI callers can ignore them.
    source_rank: int = 0
    sort_key: str = ""


class ReferenceMaterials(BaseModel):
    data_modules: list[ReferenceMaterialItem] = Field(default_factory=list)
    procedures: list[ReferenceMaterialItem] = Field(default_factory=list)
    faults: list[ReferenceMaterialItem] = Field(default_factory=list)
    references: list[ReferenceMaterialItem] = Field(default_factory=list)
    warnings: list[ReferenceMaterialItem] = Field(default_factory=list)
    cautions: list[ReferenceMaterialItem] = Field(default_factory=list)
    figures: list[ReferenceMaterialItem] = Field(default_factory=list)
    graphic_assets: list[ReferenceMaterialItem] = Field(default_factory=list)
    hotspots: list[ReferenceMaterialItem] = Field(default_factory=list)


class V4ResponseMetadata(BaseModel):
    support_level: str = "none"
    runtime_mode: str = "unknown"
    required_citations: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    ontology_trace: dict[str, object] = Field(default_factory=dict)


class RagResult(BaseModel):
    answer: str
    evidences: list[Evidence]
    reference_materials: ReferenceMaterials = Field(default_factory=ReferenceMaterials)
    v4_metadata: V4ResponseMetadata | None = None


class SessionMeta(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    locale: Optional[str] = None
    security_clearance: Optional[str] = None


class RerankOptions(BaseModel):
    enabled: bool = True
    top_k: int = 3
    model_path: str = "models/bge-reranker-v2-m3-ko/"


class RagOptions(BaseModel):
    top_k: int = 10
    relevance_threshold: float = 0.3
    rerank: RerankOptions = RerankOptions()
    max_context_chars: int = 10000
    rewrite_query: bool = False
    expand_query: bool = True
    sns_filter: bool = True
