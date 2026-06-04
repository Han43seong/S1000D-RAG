import importlib

from src.types.rag import RagResult


def test_runtime_router_defaults_to_v3_pipeline(monkeypatch):
    monkeypatch.delenv("S1000D_RAG_PIPELINE", raising=False)
    router = importlib.reload(importlib.import_module("src.rag.pipeline_runtime"))

    assert router.get_pipeline_version() == "v3"


def test_runtime_router_preserves_traceable_config_keyword(monkeypatch):
    monkeypatch.delenv("S1000D_RAG_PIPELINE", raising=False)
    router = importlib.reload(importlib.import_module("src.rag.pipeline_runtime"))

    def fake_v3(**kwargs):
        assert kwargs["query"] == "브레이크 설명"
        return RagResult(answer="v3-compatible", evidences=[])

    monkeypatch.setattr(router.pipeline_v3, "run_rag_query_sync", fake_v3)

    result = router.run_rag_query_sync("브레이크 설명", config={"metadata": {"test": True}})

    assert result.answer == "v3-compatible"


def test_runtime_router_selects_v4_pipeline(monkeypatch):
    monkeypatch.setenv("S1000D_RAG_PIPELINE", "v4")
    router = importlib.reload(importlib.import_module("src.rag.pipeline_runtime"))

    assert router.get_pipeline_version() == "v4"


def test_runtime_router_dispatches_to_v4_when_enabled(monkeypatch):
    monkeypatch.setenv("S1000D_RAG_PIPELINE", "v4")
    router = importlib.reload(importlib.import_module("src.rag.pipeline_runtime"))

    def fake_v4(**kwargs):
        return RagResult(answer=f"v4:{kwargs['query']}", evidences=[])

    monkeypatch.setattr(router.pipeline_v4, "run_rag_query_sync", fake_v4)

    result = router.run_rag_query_sync("브레이크 작동원리를 자세히 설명해줘")

    assert result.answer == "v4:브레이크 작동원리를 자세히 설명해줘"
