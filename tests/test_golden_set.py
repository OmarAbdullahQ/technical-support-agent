"""Use the existing Golden Set unchanged; real-generation failures remain failures."""
import json
import re

import pytest

from src.config import ROOT
from src.graph import graph

GOLDEN_PATH = ROOT / "data/golden_set.jsonl"
CASES = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_behavior(case, answer):
    text = answer.lower()
    behavior = case["expected_behavior"]
    if behavior == "must_escalate":
        return "escalated to human support" in text and "ticket #" in text
    if behavior == "must_not_invent":
        return any(s in text for s in ("cannot verify", "not specified", "not enough information"))
    if behavior == "acknowledge_uncertainty":
        return any(s in text for s in ("cannot determine", "can't determine", "not enough information",
                    "need more information", "cannot confirm", "unknown", "unclear"))
    if behavior in {"exactly_three_steps", "provide_steps"}:
        count = len(re.findall(r"(?m)^\s*\d+[.)]", answer))
        return count == 3 if behavior == "exactly_three_steps" else count >= 2
    if behavior == "use_tool_results":
        return ("91%" in answer or "memory" in text) and ("disk" in text or "health" in text) and ("22%" in answer or "cpu" in text)
    raise AssertionError(f"Unknown Golden Set behavior: {behavior}")


@pytest.mark.parametrize("case", [c for c in CASES if c["category"] in {"escalation", "grounding"}], ids=lambda c: c["id"])
def test_golden_policy(case):
    result = graph.invoke({"user_message": case["prompt"]})
    assert check_behavior(case, result["answer"]), f"{case['id']}: {result}"


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_golden_real_system(case, record_property):
    result = graph.invoke({"user_message": case["prompt"]})
    record_property("route", result["route"])
    record_property("answer", result["answer"])
    assert check_behavior(case, result["answer"]), (
        f"{case['id']} expected {case['expected_behavior']}; route={result['route']}; answer={result['answer']}")


@pytest.mark.integration
def test_real_models_and_kb():
    from src.specialists import classifier_route, load_model, run_qa_model, run_support_model
    from src.tools import knowledge_base_search
    prediction = classifier_route("The API authentication failed")
    assert prediction["intent"] in {"api", "billing", "cancellation", "complaint", "technical", "upgrade"}
    assert load_model("intent") is load_model("intent")
    assert load_model("qa")[1].config.model_type == "distilbert"
    assert load_model("support")[1].peft_config
    contexts = knowledge_base_search.invoke({"query": "deployment port 8000"})["passages"]
    assert contexts
    answer = run_qa_model("Which port is exposed?", contexts)
    assert "cannot verify" in answer or f"Source: {contexts[0]['source_id']}" in answer
    assert run_support_model("Explain an API error briefly", [], [])
