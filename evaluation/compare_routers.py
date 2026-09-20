import json
import statistics
import time
from pathlib import Path

from sklearn.metrics import f1_score

from src.config import get_settings
from src.routers import baseline_router, llm_router

CASES = [
    ("According to the deployment guide, which port must be exposed?", "qa"),
    ("What does the documentation say about API authentication?", "qa"),
    ("My PostgreSQL pool is at 98% and requests time out.", "tools"),
    ("The API returns 503 after deployment; health says database degraded.", "tools"),
    ("What is QLoRA?", "support"),
    ("Can someone help me understand why a Docker container may exit after startup?", "support"),
    ("Production database may be corrupted after a failed migration.", "escalate"),
    ("We suspect a security breach in production.", "escalate"),
]

ROUTES = ["qa", "tools", "support", "escalate"]


def router_a(text):
    return baseline_router(text, use_llm=False)


def router_b(text):
    return llm_router(text)


def hybrid_router(text):
    return baseline_router(text, use_llm=True)


def evaluate(name, router):
    rows = []
    expected = []
    predicted = []
    times = []

    for prompt, target in CASES:
        start = time.perf_counter()
        result = router(prompt)
        elapsed = time.perf_counter() - start

        route = result["route"]
        expected.append(target)
        predicted.append(route)
        times.append(elapsed)

        rows.append({
            "prompt": prompt,
            "expected": target,
            "predicted": route,
            "correct": route == target,
            "source": result.get("source", ""),
            "confidence": result.get("confidence"),
            "fallback_reason": result.get("fallback_reason"),
            "seconds": round(elapsed, 4),
        })

    output = {
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
        "macro_f1": round(
            f1_score(expected, predicted, labels=ROUTES, average="macro", zero_division=0),
            4,
        ),
        "mean_seconds": round(statistics.mean(times), 4),
        "llm_usage_rate": round(
            sum(r["source"] == "groq" for r in rows) / len(rows), 4
        ),
        "fallback_error_rate": round(
            sum(r["source"] == "fallback" for r in rows) / len(rows), 4
        ),
        "wrong_routes": [
            {
                "prompt": r["prompt"],
                "expected": r["expected"],
                "predicted": r["predicted"],
                "source": r["source"],
            }
            for r in rows
            if not r["correct"]
        ],
        "cases": rows,
    }

    if name == "llm":
        output["json_valid_rate"] = round(
            sum(r["source"] == "groq" for r in rows) / len(rows), 4
        )

    return output


def main():
    settings = get_settings()

    # Warm-up classifier
    baseline_router("warm up request", use_llm=False)

    routers = {
        "rules_classifier": router_a,
        "llm": router_b,
        "hybrid": hybrid_router,
    }

    report = {"case_count": len(CASES), "routers": {}}

    for name, router in routers.items():
        if name == "llm" and not (settings.groq_api_key and settings.groq_model):
            report["routers"][name] = {"status": "not_run"}
            continue

        result = evaluate(name, router)
        report["routers"][name] = result

        print(f"\n{name}")
        print("accuracy:", result["accuracy"])
        print("macro_f1:", result["macro_f1"])
        print("mean_seconds:", result["mean_seconds"])
        print("llm_usage_rate:", result["llm_usage_rate"])
        print("fallback_error_rate:", result["fallback_error_rate"])

        if "json_valid_rate" in result:
            print("json_valid_rate:", result["json_valid_rate"])

        print("wrong_routes:", len(result["wrong_routes"]))

        for wrong in result["wrong_routes"]:
            print(
                "-",
                wrong["expected"],
                "->",
                wrong["predicted"],
                "|",
                wrong["prompt"],
            )

    path = Path("reports/router_comparison.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved:", path)


if __name__ == "__main__":
    main()