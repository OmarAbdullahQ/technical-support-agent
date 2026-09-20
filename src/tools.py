"""The brief's 13 structured tool contracts, with local and PostgreSQL backends."""
import ast
import hashlib
import math
import operator
import re
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from src.config import ROOT, get_settings


@contextmanager
def _database():
    url = get_settings().database_url
    postgres = url.startswith(("postgresql://", "postgres://"))
    if postgres:
        import psycopg
        from psycopg.rows import dict_row
        con = psycopg.connect(url, row_factory=dict_row, connect_timeout=5)
        identity = "SERIAL PRIMARY KEY"
    elif url.startswith("sqlite:///"):
        path = Path(url.removeprefix("sqlite:///"))
        path = path if path.is_absolute() else ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, timeout=10)
        con.row_factory = sqlite3.Row
        identity = "INTEGER PRIMARY KEY AUTOINCREMENT"
    else:
        raise ValueError("DATABASE_URL must use sqlite:/// or postgresql://")
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS tickets (id {identity}, title TEXT NOT NULL, "
                    "description TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL, resolution TEXT DEFAULT '')")
        con.commit()
        yield con, "%s" if postgres else "?"
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


@tool
def ticket_search(query: str) -> dict:
    """Search up to five prior tickets by title using a parameterized query."""
    with _database() as (con, bind):
        rows = con.execute("SELECT id, title, status, resolution FROM tickets WHERE title LIKE "
                           + bind + " ORDER BY id DESC LIMIT 5", (f"%{query}%",)).fetchall()
    return {"tickets": [dict(row) for row in rows]}


@tool
def ticket_create(title: str, description: str, priority: Literal["low", "medium", "high"] = "medium") -> dict:
    """Persist a new support ticket; no external notification is sent."""
    if not title.strip() or len(title) > 500 or len(description) > 20000:
        return {"ok": False, "error": "Invalid ticket title or description length"}
    with _database() as (con, bind):
        row = con.execute("INSERT INTO tickets(title, description, priority, status) VALUES ("
                          + ", ".join([bind] * 3) + ", 'open') RETURNING id",
                          (title, description, priority)).fetchone()
    return {"ok": True, "ticket_id": row["id"], "status": "open"}


@tool
def system_health_check(service: str) -> dict:
    """Return deterministic LAB MOCK service health, never a live measurement."""
    registry = {"api": {"status": "healthy", "latency_ms": 42},
                "database": {"status": "degraded", "connections_pct": 91},
                "gpu-worker": {"status": "healthy", "gpu_utilization": 74}}
    return {"service": service, "mock": True, **registry.get(service, {"status": "unknown"})}


@tool
def log_analyzer(log_text: str) -> dict:
    """Extract ERROR/FATAL/EXCEPTION and WARN lines from supplied log text."""
    lines = log_text[:100000].splitlines()
    errors = [line for line in lines if re.search(r"\b(ERROR|FATAL|EXCEPTION)\b", line, re.I)]
    warnings = [line for line in lines if re.search(r"\bWARN(?:ING)?\b", line, re.I)]
    return {"errors": errors[:20], "warnings": warnings[:20], "error_count": len(errors),
            "truncated": len(log_text) > 100000}


@tool
def escalate_to_human(reason: str, evidence: str) -> dict:
    """Record a high-priority escalation ticket and evidence for human review."""
    ticket = ticket_create.invoke({"title": f"ESCALATION: {reason}"[:500],
                                   "description": evidence[:20000], "priority": "high"})
    return {"escalated": ticket.get("ok", False), "reason": reason, "ticket": ticket,
            "notification_sent": False}


@tool
def diagnostic_runbook(issue_type: str, service: str, symptom: str) -> dict:
    """Run the brief's health then prior-ticket diagnostic sequence."""
    health = system_health_check.invoke({"service": service})
    prior = ticket_search.invoke({"query": symptom})
    return {"status": "needs_human" if health["status"] == "unknown" else "diagnostics_complete",
            "issue_type": issue_type, "mock_health": True,
            "steps": [{"step": "health_check", "result": health}, {"step": "prior_tickets", "result": prior}]}


@tool
def calculator(expression: str) -> dict:
    """Evaluate bounded arithmetic without eval, names, function calls or powers."""
    binary = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in binary:
            result = binary[type(node.op)](calculate(node.left), calculate(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("Unsupported arithmetic")
        if not math.isfinite(result) or abs(result) > 1e15:
            raise ValueError("Result exceeds limit")
        return result

    try:
        if len(expression) > 200:
            raise ValueError("Expression exceeds limit")
        return {"ok": True, "result": calculate(ast.parse(expression.strip(), mode="eval").body)}
    except (ValueError, SyntaxError, ArithmeticError, RecursionError) as exc:
        return {"ok": False, "error": str(exc)}


@lru_cache(maxsize=1)
def _kb_index():
    """Read existing contexts without rebuilding or changing a training dataset."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    settings = get_settings()
    passages = []
    seen = set()
    if settings.kb_docs_path.exists():
        for path in sorted(settings.kb_docs_path.glob("*.md")):
            passages.append({"source_id": f"docs/{path.name}", "text": path.read_text(encoding="utf-8")})
    if settings.kb_dataset_path.exists():
        from datasets import load_from_disk, DatasetDict
        dataset = load_from_disk(str(settings.kb_dataset_path))
        rows = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
        for row in rows:
            for context in row.get("contexts", []):
                text = context["text"]
                digest = hashlib.sha256(text.encode()).hexdigest()[:12]
                if digest not in seen:
                    seen.add(digest)
                    passages.append({"source_id": f"{context.get('filename', 'TechQA')}#{digest}", "text": text})
    if not passages:
        return [], None, None
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(p["text"] for p in passages)
    return passages, vectorizer, matrix


@tool
def knowledge_base_search(query: str) -> dict:
    """Retrieve three scored source passages using cached local TF-IDF (TODO 7)."""
    passages, vectorizer, matrix = _kb_index()
    if vectorizer is None:
        return {"passages": []}
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    top = scores.argsort()[::-1][:3]
    return {"passages": [{**passages[i], "score": float(scores[i])} for i in top
                         if scores[i] >= get_settings().kb_min_score]}


@tool
def documentation_search(query: str) -> dict:
    """Search the same indexed local technical documentation as the KB."""
    return knowledge_base_search.invoke({"query": query})


@tool
def package_lookup(package_name: str, version: str = "") -> dict:
    """Return a deterministic mock; no external compatibility claim is made."""
    return {"mock": True, "package": package_name, "version": version,
            "status": "unverified", "compatibility": None}


@tool
def sql_query(query: str) -> dict:
    """Run only an allowlisted ticket SELECT; arbitrary SQL is never executed."""
    allowed = {"select id, title, status, resolution from tickets limit 20",
               "select count(*) as count from tickets"}
    normalized = " ".join(query.strip().lower().split())
    if normalized not in allowed:
        return {"ok": False, "error": "Only allowlisted read-only ticket queries are supported"}
    with _database() as (con, _):
        rows = con.execute(normalized).fetchall()
    return {"ok": True, "rows": [dict(row) for row in rows]}


@tool
def file_search(query: str, path: str = "data/uploads") -> dict:
    """Deterministic mock for approved uploads; does not read arbitrary paths."""
    return {"mock": True, "query": query, "matches": [], "status": "upload_backend_not_configured"}


@tool
def web_search(query: str) -> dict:
    """Deterministic external-search mock with no invented search results."""
    return {"mock": True, "query": query, "results": [], "status": "provider_not_configured"}


ALL_TOOLS = [knowledge_base_search, ticket_search, ticket_create, system_health_check,
             log_analyzer, documentation_search, package_lookup, sql_query, calculator,
             file_search, web_search, escalate_to_human, diagnostic_runbook]
