import pytest

from src.config import get_settings
from src.tools import _kb_index


def pytest_addoption(parser):
    parser.addoption("--run-models", action="store_true", help="Run actual local-model integration/Golden Set checks")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-models"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Requires --run-models and existing local artifacts"))


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "tickets.db").as_posix())
    monkeypatch.setenv("API_KEY", "test-only-key")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_MODEL", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    _kb_index.cache_clear()
    yield
    get_settings.cache_clear()
    _kb_index.cache_clear()
