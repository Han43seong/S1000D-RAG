import importlib


def test_project_traceable_is_noop_by_default_even_if_langsmith_env_was_true(monkeypatch):
    monkeypatch.delenv("S1000D_LANGSMITH_TRACING", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    tracing = importlib.reload(importlib.import_module("src.tracing"))

    @tracing.traceable(run_type="chain", name="unit_noop")
    def sample(value):
        return value + 1

    assert sample(2) == 3
    assert tracing.tracing_enabled() is False
    assert tracing.os.environ["LANGSMITH_TRACING"] == "false"
    assert tracing.os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_project_traceable_can_be_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("S1000D_LANGSMITH_TRACING", "true")

    tracing = importlib.reload(importlib.import_module("src.tracing"))

    assert tracing.tracing_enabled() is True
