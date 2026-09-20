"""The brief's route -> QA/tools/support/escalation LangGraph."""
import json
import re
from typing import Literal, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from src.routers import baseline_router, missing_evidence, supplied_tool_results
from src.specialists import UNVERIFIED, run_qa_model, run_support_model
from src.tools import (diagnostic_runbook, escalate_to_human, knowledge_base_search,
                       log_analyzer, ticket_create, ticket_search)


class SupportState(TypedDict, total=False):
    user_message: str
    route: str
    intent: str
    confidence: float
    classifier_confidence: float
    source: str
    fallback_reason: str
    router_latency_seconds: float
    context: list[dict]
    tool_results: list[dict]
    answer: str
    escalate: bool
    request_id: str
    trace_id: str


def route_node(state: SupportState):
    return baseline_router(state["user_message"])


def qa_node(state: SupportState):
    message = state["user_message"]
    # A strict supplied-information request must never retrieve unrelated material.
    if re.search(r"(?:only.{0,35}(?:information|context)|answer only from)", message, re.I | re.S):
        return {"context": [], "answer": UNVERIFIED}
    kb = knowledge_base_search.invoke({"query": message})
    answer = RunnableLambda(lambda data: run_qa_model(**data), name="model_b_qa").invoke(
        {"question": message, "contexts": kb["passages"]})
    return {"context": kb["passages"], "answer": answer}


def infer_service(message: str) -> str:
    lower = message.lower()
    if any(word in lower for word in ("postgres", "database", "sql", "pool")):
        return "database"
    if any(word in lower for word in ("cuda", "gpu")):
        return "gpu-worker"
    if any(word in lower for word in ("api", "deployment", "endpoint")):
        return "api"
    return "unknown"


def tool_node(state: SupportState):
    message = state["user_message"]
    if re.search(r"\b(?:create|open|file)\s+(?:a\s+)?(?:support\s+)?ticket\b", message, re.I):
        result = ticket_create.invoke({"title": message[:200], "description": message})
    elif re.search(r"\b(?:find|search|look up)\b.{0,30}\btickets?\b", message, re.I):
        result = ticket_search.invoke({"query": message})
    elif re.search(r"\b(log|logs|ERROR|WARN|FATAL|EXCEPTION)\b", message, re.I):
        result = log_analyzer.invoke({"log_text": message})
    else:
        result = diagnostic_runbook.invoke({"issue_type": state.get("intent", "unknown"),
                   "service": infer_service(message), "symptom": message})
    return {"tool_results": [result], "escalate": result.get("status") == "needs_human"}


def support_node(state: SupportState):
    if missing_evidence(state["user_message"]) and not state.get("tool_results"):
        return {"answer": "I cannot determine the root cause from the information provided. "
                          "Please share the exact error, relevant logs, recent changes, and affected service."}
    evidence = state.get("tool_results", [])
    if supplied_tool_results(state["user_message"]):
        evidence = evidence + [{"source": "user_supplied_unverified", "text": state["user_message"]}]
    answer = RunnableLambda(lambda data: run_support_model(**data), name="model_c_support").invoke({
        "user_message": state["user_message"], "context": state.get("context", []),
        "tool_results": evidence})
    # Preserve the actual tool evidence even when the small model omits/misstates it.
    if evidence:
        answer += "\n\nEvidence (registry health is a lab mock; user-supplied results are unverified):\n" + json.dumps(evidence, ensure_ascii=False)
    return {"answer": answer}


def escalation_node(state: SupportState):
    result = escalate_to_human.invoke({"reason": "Policy or unresolved technical incident",
                                       "evidence": json.dumps(dict(state), ensure_ascii=False)})
    return {"escalate": True, "tool_results": state.get("tool_results", []) + [result],
            "answer": f"Escalated to human support: ticket #{result['ticket']['ticket_id']} recorded for review. "
                      "Preserve logs and evidence; avoid destructive changes. No external notification has been sent."}


def choose_after_router(state: SupportState) -> Literal["qa", "tools", "support", "escalate"]:
    route = state.get("route", "support")
    return route if route in {"qa", "tools", "support", "escalate"} else "support"


def choose_after_tools(state: SupportState) -> Literal["support", "escalate"]:
    return "escalate" if state.get("escalate") else "support"


builder = StateGraph(SupportState)
for name, node in (("route", route_node), ("qa", qa_node), ("tools", tool_node),
                   ("support", support_node), ("escalate", escalation_node)):
    builder.add_node(name, node)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", choose_after_router)
builder.add_edge("qa", END)
builder.add_conditional_edges("tools", choose_after_tools)
builder.add_edge("support", END)
builder.add_edge("escalate", END)
graph = builder.compile()
