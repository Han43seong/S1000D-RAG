from __future__ import annotations

import json

from src.chunker.indexer import build_chunk_metadata, chunks_to_documents
from src.types.chunk import S1000DChunk
from src.types.dm import DmType


def _chunk() -> S1000DChunk:
    return S1000DChunk(
        dmc="DMC-S1000DBIKE-AAA-DA1-00-00-00AA-341A-A",
        chunk_id="DMC-S1000DBIKE-AAA-DA1-00-00-00AA-341A-A__chunk-001",
        dm_type=DmType.PROCEDURAL,
        security="01",
        applicability="All",
        structure_path_range="procedure/mainProcedure/proceduralStep[1]",
        text="Remove the brake assembly.",
        metadata={
            "title": "Brake - Remove procedure",
            "issue": "010-00",
            "language": "en-US",
            "block_ids": ["step-1", "warning-1"],
            "role_distribution": {"step": 1, "warning": 1},
            "source_file": "DMC-S1000DBIKE-AAA-DA1-00-00-00AA-341A-A_010-00_EN-US.XML",
            "source_path": "/tmp/DMC-S1000DBIKE-AAA-DA1-00-00-00AA-341A-A_010-00_EN-US.XML",
        },
    )


def test_build_chunk_metadata_normalizes_s1000d_evidence_fields():
    metadata = build_chunk_metadata(_chunk())

    assert metadata["dmc"] == "DMC-S1000DBIKE-AAA-DA1-00-00-00AA-341A-A"
    assert metadata["chunk_id"].endswith("chunk-001")
    assert metadata["dm_type"] == "procedural"
    assert metadata["security"] == "01"
    assert metadata["applicability"] == "All"
    assert metadata["sns_code"] == "DA1"
    assert metadata["issue"] == "010-00"
    assert metadata["language"] == "en-US"
    assert metadata["title"] == "Brake - Remove procedure"
    assert metadata["structure_path_range"].startswith("procedure")
    assert json.loads(metadata["block_ids"]) == ["step-1", "warning-1"]
    assert json.loads(metadata["role_distribution"]) == {"step": 1, "warning": 1}
    assert metadata["source_file"].startswith("DMC-")
    assert metadata["source_path"].endswith(".XML")
    assert metadata["modality"] == "text"


def test_chunks_to_documents_uses_primitive_metadata_without_heavy_requirements():
    docs = chunks_to_documents([_chunk()])

    assert len(docs) == 1
    assert docs[0].page_content == "Remove the brake assembly."
    for value in docs[0].metadata.values():
        assert isinstance(value, (str, int, float, bool))
