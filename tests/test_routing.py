from types import SimpleNamespace

import pytest

from src import routers
from src.config import get_settings


@pytest.mark.parametrize("text,route", [
    ("Production database corruption suspected", "escalate"),
    ("According to the docs, what port is used?", "qa"),
    ("According to the deployment guide, which port must be exposed?", "qa"),
    ("My PostgreSQL pool is at 98% and requests time out.", "tools"),
    ("My API returns 503 after deployment.", "tools"),
    ("According to the docs, production down with data loss", "escalate"),
])
def test_required_routes(text, route, monkeypatch):
    monkeypatch.setattr(routers, "classifier_route", lambda _: pytest.fail("Hard rule must win"))
    assert routers.baseline_router(text)["route"] == route


def test_regression_corrupted_is_escalation(monkeypatch):
    """Brief used 'corruption', but its scenario and saved G02 failure use 'corrupted'."""
    monkeypatch.setattr(routers, "classifier_route", lambda _: pytest.fail("Must bypass model"))
    assert routers.baseline_router("Production database may be corrupted after a failed migration.")["route"] == "escalate"


def test_regression_absent_documentation_does_not_retrieve():
    """First integrated G09 run incorrectly returned an unrelated WebSphere KB answer."""
    result = routers.baseline_router("The app is failing but no logs, documentation, or system information are provided.")
    assert result["route"] == "support"


def test_regression_supplied_diagnostics_are_synthesized():
    """First integrated G10 run escalated a supplied report as an unknown service."""
    result = routers.baseline_router("A diagnostic tool returned these results: memory usage 91%, disk health OK.")
    assert result["route"] == "support"


@pytest.mark.parametrize("intent,route", list(routers.INTENT_ROUTES.items()))
def test_actual_labels(intent, route, monkeypatch):
    monkeypatch.setattr(routers, "classifier_route", lambda _: {"intent": intent, "confidence": .95})
    assert routers.baseline_router("Please help me")["route"] == route


def test_low_confidence_uses_llm(monkeypatch):
    monkeypatch.setattr(routers, "classifier_route", lambda _: {"intent": "technical", "confidence": .4})
    monkeypatch.setattr(routers, "llm_router", lambda _: {"route": "qa", "confidence": .9, "source": "groq"})
    decision = routers.baseline_router("Ambiguous question")
    assert decision["route"] == "qa"
    assert decision["classifier_confidence"] == .4
    assert decision["intent"] == "technical"


@pytest.mark.parametrize("payload,expected", [
    ('{"route":"qa","confidence":0.9}', "qa"),
    ('{"route":"shell","confidence":0.9}', "support"),
    ('{"route":"tools","confidence":4}', "support"),
    ('{"route":"qa","confidence":"0.9"}', "support"),
    ('```json\n{"route":"qa"}\n```', "support"),
    ('not JSON', "support"),
    ('{"route":"qa","confidence":0.9,"command":"delete"}', "support"),
])
def test_groq_schema(monkeypatch, payload, expected):
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-router")
    get_settings.cache_clear()

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    monkeypatch.setattr(routers, "OpenAI", Client)
    assert routers.llm_router("Ambiguous")["route"] == expected


def test_groq_missing_credentials():
    assert routers.llm_router("Ambiguous")["fallback_reason"] == "groq_not_configured"


def test_groq_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GROQ_MODEL", "test")
    get_settings.cache_clear()
    def broken(**kwargs): raise TimeoutError()
    monkeypatch.setattr(routers, "OpenAI", broken)
    assert routers.llm_router("Ambiguous")["fallback_reason"] == "TimeoutError"
