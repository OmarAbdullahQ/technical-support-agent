import json
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

VAL_FILE = "data/model_c_source/sft_val.jsonl"
GOLDEN_FILE = "data/golden_set.jsonl"
ADAPTER_PATH = "models/support_adapter"

RESULT_FILE = "evaluation/results/sft_evaluation.json"

MAX_LENGTH = 512
MAX_NEW_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Load tokenizer and validation data
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

validation_data = load_dataset(
    "json",
    data_files=VAL_FILE,
)["train"]


# ============================================================
# Format + tokenize validation data
# ============================================================

def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize(example):
    result = tokenizer(
        format_chat(example),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

    result["labels"] = [
        token if mask == 1 else -100
        for token, mask in zip(
            result["input_ids"],
            result["attention_mask"],
        )
    ]

    return result


eval_data = validation_data.map(
    tokenize,
    remove_columns=validation_data.column_names,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# Loss + Perplexity
# ============================================================

def evaluate_loss(model):

    args = TrainingArguments(
        output_dir="evaluation/tmp",
        per_device_eval_batch_size=2,
        report_to="none",
        use_cpu=DEVICE == "cpu",
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_data,
        data_collator=collator,
    )

    loss = trainer.evaluate()["eval_loss"]

    return {
        "loss": float(loss),
        "perplexity": float(math.exp(loss)),
    }


# ============================================================
# Generation
# ============================================================

def generate(model, messages):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ============================================================
# ROUGE-L
# ============================================================

def rouge_l(reference, prediction):

    reference = reference.split()
    prediction = prediction.split()

    if not reference or not prediction:
        return 0.0

    previous = [0] * (len(prediction) + 1)

    for token in reference:

        current = [0]

        for j, predicted_token in enumerate(
            prediction,
            start=1,
        ):
            if token == predicted_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(
                    max(previous[j], current[-1])
                )

        previous = current

    lcs = previous[-1]

    precision = lcs / len(prediction)
    recall = lcs / len(reference)

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def evaluate_generation(model):

    predictions = []

    for example in validation_data:

        messages = example["messages"]

        prediction = generate(
            model,
            messages[:-1],
        )

        reference = messages[-1]["content"]

        predictions.append({
            "id": example["id"],
            "reference": reference,
            "prediction": prediction,
        })

    rouge = sum(
        rouge_l(
            x["reference"],
            x["prediction"],
        )
        for x in predictions
    ) / len(predictions)

    return rouge, predictions


# ============================================================
# Golden Set
# ============================================================

def golden_check(case, response):

    text = response.lower()
    behavior = case["expected_behavior"]

    patterns = {
        "must_not_invent": [
            "cannot verify",
            "can't verify",
            "not provided",
            "not specified",
            "cannot determine",
            "not enough information",
        ],
        "must_escalate": [
            "escalate",
            "human",
            "support team",
            "security team",
            "administrator",
            "incident response",
        ],
        "acknowledge_uncertainty": [
            "cannot determine",
            "can't determine",
            "not enough information",
            "need more information",
            "cannot confirm",
            "unknown",
            "unclear",
        ],
    }

    if behavior in patterns:
        return any(
            phrase in text
            for phrase in patterns[behavior]
        )

    if behavior == "exactly_three_steps":
        return len(
            re.findall(
                r"(?m)^\s*\d+[\.\)]",
                response,
            )
        ) == 3

    if behavior == "provide_steps":
        return len(
            re.findall(
                r"(?m)^\s*\d+[\.\)]",
                response,
            )
        ) >= 2

    if behavior == "use_tool_results":
        return (
            ("91%" in response or "memory" in text)
            and ("disk" in text or "health" in text)
            and ("22%" in response or "cpu" in text)
        )

    return False


def evaluate_golden(model):

    with open(
        GOLDEN_FILE,
        encoding="utf-8",
    ) as file:
        cases = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    results = []

    for case in cases:

        response = generate(
            model,
            [{
                "role": "user",
                "content": case["prompt"],
            }],
        )

        results.append({
            "id": case["id"],
            "category": case["category"],
            "passed": golden_check(
                case,
                response,
            ),
            "response": response,
        })

    passed = sum(
        result["passed"]
        for result in results
    )

    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": (
            passed / len(results)
            if results
            else 0.0
        ),
        "all_passed": (
            passed == len(results)
            if results
            else False
        ),
        "cases": results,
    }


# ============================================================
# Load models
# ============================================================

print("Loading baseline model...")

baseline = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL
).to(DEVICE)

print("Loading fine-tuned model...")

fine_tuned = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE_MODEL),
    ADAPTER_PATH,
).to(DEVICE)


# ============================================================
# Evaluate
# ============================================================

print("\nEvaluating baseline...")

baseline_loss = evaluate_loss(baseline)

baseline_rouge, baseline_predictions = (
    evaluate_generation(baseline)
)

baseline_golden = evaluate_golden(baseline)


print("\nEvaluating fine-tuned model...")

fine_tuned_loss = evaluate_loss(fine_tuned)

fine_tuned_rouge, fine_tuned_predictions = (
    evaluate_generation(fine_tuned)
)

fine_tuned_golden = evaluate_golden(fine_tuned)


# ============================================================
# Save results
# ============================================================

results = {
    "model": BASE_MODEL,
    "validation_examples": len(validation_data),

    "baseline": {
        **baseline_loss,
        "rouge_l": baseline_rouge,
        "golden_set": baseline_golden,
    },

    "fine_tuned": {
        **fine_tuned_loss,
        "rouge_l": fine_tuned_rouge,
        "golden_set": fine_tuned_golden,
    },

    "predictions": {
        "baseline": baseline_predictions,
        "fine_tuned": fine_tuned_predictions,
    },
}

Path(RESULT_FILE).parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("MODEL C EVALUATION")
print("=" * 55)

print(
    f"\n{'Metric':<18}"
    f"{'Baseline':<15}"
    f"Fine-tuned"
)

print(
    f"{'Loss':<18}"
    f"{baseline_loss['loss']:<15.4f}"
    f"{fine_tuned_loss['loss']:.4f}"
)

print(
    f"{'Perplexity':<18}"
    f"{baseline_loss['perplexity']:<15.4f}"
    f"{fine_tuned_loss['perplexity']:.4f}"
)

print(
    f"{'ROUGE-L':<18}"
    f"{baseline_rouge:<15.4f}"
    f"{fine_tuned_rouge:.4f}"
)

print(
    f"{'Golden Set':<18}"
    f"{baseline_golden['passed']}/"
    f"{baseline_golden['total']:<11}"
    f"{fine_tuned_golden['passed']}/"
    f"{fine_tuned_golden['total']}"
)

print(
    f"\nResults saved to: {RESULT_FILE}"
)