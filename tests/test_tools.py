import pytest

from src import tools
from src.config import get_settings


def test_all_13_contracts():
    assert len(tools.ALL_TOOLS) == 13
    for tool in tools.ALL_TOOLS:
        assert tool.args_schema.model_json_schema()["properties"]


def test_health_tool():
    result = tools.system_health_check.invoke({"service": "api"})
    assert result["status"] == "healthy" and result["mock"]
    assert tools.system_health_check.invoke({"service": "unregistered"})["status"] == "unknown"


def test_ticket_lifecycle_and_sql():
    ticket = tools.ticket_create.invoke({"title": "503 issue", "description": "deployment"})
    found = tools.ticket_search.invoke({"query": "503"})["tickets"]
    assert found[0]["id"] == ticket["ticket_id"]
    assert tools.ticket_search.invoke({"query": "' OR 1=1 --"})["tickets"] == []
    assert tools.sql_query.invoke({"query": "SELECT count(*) AS count FROM tickets"})["rows"] == [{"count": 1}]


@pytest.mark.parametrize("query", ["DELETE FROM tickets", "SELECT 1; DROP TABLE tickets", "SELECT pg_sleep(10)",
                                  "WITH x AS (DELETE FROM tickets RETURNING *) SELECT * FROM x"])
def test_sql_rejects_non_allowlisted_queries(query):
    assert not tools.sql_query.invoke({"query": query})["ok"]


def test_escalation_persists():
    result = tools.escalate_to_human.invoke({"reason": "corruption", "evidence": "production"})
    assert result["escalated"] and not result["notification_sent"]
    assert tools.ticket_search.invoke({"query": "ESCALATION"})["tickets"]


def test_diagnostic_runbook():
    args = {"issue_type": "technical", "service": "database", "symptom": "connections"}
    result = tools.diagnostic_runbook.invoke(args)
    assert result["status"] == "diagnostics_complete"
    assert [s["step"] for s in result["steps"]] == ["health_check", "prior_tickets"]
    assert tools.diagnostic_runbook.invoke({**args, "service": "unknown"})["status"] == "needs_human"


def test_log_analyzer():
    result = tools.log_analyzer.invoke({"log_text": "INFO fine\nERROR failed\nWARNING slow\nfatal crash"})
    assert result["error_count"] == 2
    assert result["warnings"] == ["WARNING slow"]


@pytest.mark.parametrize("expression,ok", [("(91 - 22) / 3", True), ("1/0", False),
    ("__import__('os')", False), ("9**999999", False), ("1e300", False), ("True", False)])
def test_calculator(expression, ok):
    assert tools.calculator.invoke({"expression": expression})["ok"] is ok


def test_todo7_kb_retrieval(monkeypatch, tmp_path):
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "ports.md").write_text("The deployment API port is 8000.")
    (docs / "vpn.md").write_text("VPN authentication requires MFA.")
    monkeypatch.setenv("KB_DOCS_PATH", str(docs))
    monkeypatch.setenv("KB_DATASET_PATH", str(tmp_path / "absent"))
    get_settings.cache_clear()
    result = tools.knowledge_base_search.invoke({"query": "deployment port"})
    assert result["passages"][0]["source_id"] == "docs/ports.md"
    assert result["passages"][0]["score"] > 0
    assert tools.knowledge_base_search.invoke({"query": "zyxwq"})["passages"] == []


def test_mock_contracts_are_honest():
    assert tools.web_search.invoke({"query": "anything"})["mock"]
    assert tools.file_search.invoke({"query": "secret", "path": "../../"})["matches"] == []
    assert tools.package_lookup.invoke({"package_name": "torch"})["compatibility"] is None
