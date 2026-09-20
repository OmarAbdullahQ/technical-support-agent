import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import get_settings
from src.observability import trace_config

workflow = importlib.import_module("src.graph")
HEADERS = {"Authorization": "Bearer test-only-key"}


def request(client, text, **kwargs):
    return client.post("/v1/chat/completions", headers=HEADERS,
                       json={"messages": [{"role": "user", "content": text}], **kwargs})


def test_health_models_auth():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        models = client.get("/v1/models", headers=HEADERS).json()
        assert [m["id"] for m in models["data"]] == ["tuwaiq-tech-support-agent"]


def test_real_escalation_graph_and_api():
    with TestClient(app) as client:
        result = request(client, "Production database may be corrupted after a failed migration.")
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["system_metadata"]["route"] == "escalate"
        assert "ticket #" in payload["choices"][0]["message"]["content"]
        assert payload["object"] == "chat.completion"
        assert "usage" in payload and len(payload["system_metadata"]["trace_id"]) == 32


@pytest.mark.parametrize("text,expected", [("According to the docs, port?", "qa"),
                                           ("API health check", "tools")])
def test_graph_specialist_calls(monkeypatch, text, expected):
    monkeypatch.setattr(workflow, "run_qa_model", lambda **kwargs: "8000")
    monkeypatch.setattr(workflow, "run_support_model", lambda **kwargs: "Tool synthesis")
    monkeypatch.setattr(workflow.knowledge_base_search, "func", lambda query: {"passages": []})
    with TestClient(app) as client:
        result = request(client, text).json()
    assert result["system_metadata"]["route"] == expected


def test_unknown_service_reaches_human():
    result = workflow.graph.invoke({"user_message": "Check health of the unknown-service"})
    assert result["escalate"]
    assert "human support" in result["answer"]


def test_buffered_stream():
    with TestClient(app) as client:
        response = request(client, "Production down", stream=True)
    data = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert data[-1] == "[DONE]"
    assert json.loads(data[0])["object"] == "chat.completion.chunk"
    assert json.loads(data[-2])["choices"][0]["finish_reason"] == "stop"


def test_invalid_requests_and_fail_closed(monkeypatch):
    with TestClient(app) as client:
        assert request(client, "hello", model="other").status_code == 404
        assert request(client, " ").status_code == 400
        assert client.post("/v1/chat/completions", headers=HEADERS, json={"messages": []}).status_code == 422
        monkeypatch.setenv("API_KEY", "")
        get_settings.cache_clear()
        assert client.get("/v1/models", headers=HEADERS).status_code == 503


def test_model_failure_returns_503(monkeypatch):
    monkeypatch.setattr(workflow, "baseline_router", lambda _: {"route": "support"})
    def broken(**kwargs): raise OSError("missing weights at private path")
    monkeypatch.setattr(workflow, "run_support_model", broken)
    with TestClient(app) as client:
        result = request(client, "help")
    assert result.status_code == 503
    assert "private path" not in result.text


def test_langfuse_disabled():
    assert trace_config("request", "a" * 32)["callbacks"] == []
