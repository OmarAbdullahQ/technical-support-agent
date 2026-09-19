import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)


# ============================================================
# Configuration
# ============================================================

MODEL_B = "models/qa_model"
DATASET_PATH = "data/model_b_dataset"

MAX_LENGTH = 384
DOC_STRIDE = 96
MAX_QUESTION_LENGTH = 128

RANDOM_STATE = 42

# Maximum predicted answer length in tokens
MAX_ANSWER_LENGTH = 128

# Number of long-context examples to show for manual inspection
MANUAL_CASES = 10

RESULTS_DIR = Path("evaluation/results/qa_model")


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("Loading Model B evaluation dataset...")
print("=" * 60)

dataset = load_from_disk(DATASET_PATH)

print(dataset)


# ============================================================
# Convert TechQA to SQuAD-style examples
# ============================================================

def build_qa_examples(dataset_split):
    """
    Convert TechQA examples into:

        id
        question
        context
        answer_text
        answer_start

    Only answerable examples where the answer appears in a
    provided context are kept.
    """
    qa_examples = []

    for example in dataset_split:

        if example["is_impossible"]:
            continue

        question = example["question"]
        answer = example["answer"].strip()

        matched_context = None
        answer_start = None

        for context_item in example["contexts"]:

            context = context_item["text"]

            # Exact match
            start = context.find(answer)

            if start != -1:
                matched_context = context
                answer_start = start
                break

            # Case-insensitive fallback
            start_lower = context.lower().find(answer.lower())

            if start_lower != -1:
                matched_context = context
                answer_start = start_lower
                break

        if matched_context is None:
            continue

        qa_examples.append(
            {
                "id": example["id"],
                "question": question,
                "context": matched_context,
                "answer_text": answer,
                "answer_start": answer_start,
            }
        )

    return qa_examples


qa_examples = build_qa_examples(dataset["train"])

print()
print(f"Valid QA examples: {len(qa_examples)}")


# ============================================================
# Recreate the same held-out validation split
# ============================================================

train_examples, val_examples = train_test_split(
    qa_examples,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

val_dataset = Dataset.from_list(val_examples)

print()
print("=" * 60)
print("Held-out validation set")
print("=" * 60)

print(f"Validation examples: {len(val_dataset)}")


# ============================================================
# Load model and tokenizer
# ============================================================

print()
print("=" * 60)
print("Loading trained Model B...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_B)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_B
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Device: {device}")


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalize text for SQuAD-style EM and token-level F1.

    - lowercase
    - remove punctuation
    - normalize whitespace
    """

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = " ".join(text.split())

    return text


# ============================================================
# Exact Match
# ============================================================

def exact_match_score(prediction, ground_truth):

    return int(
        normalize_text(prediction)
        == normalize_text(ground_truth)
    )


# ============================================================
# Token-level F1
# ============================================================

def token_f1_score(prediction, ground_truth):

    prediction_tokens = normalize_text(
        prediction
    ).split()

    ground_truth_tokens = normalize_text(
        ground_truth
    ).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(
        prediction_tokens
    ) & Counter(
        ground_truth_tokens
    )

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = (
        num_same / len(prediction_tokens)
    )

    recall = (
        num_same / len(ground_truth_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# Import Counter
# ============================================================

from collections import Counter


# ============================================================
# Prepare evaluation features
# ============================================================

def prepare_evaluation_features(example):

    question = example["question"].strip()

    # Limit long TechQA questions
    question_tokens = tokenizer(
        question,
        truncation=True,
        max_length=MAX_QUESTION_LENGTH,
        add_special_tokens=False,
    )

    processed_question = tokenizer.decode(
        question_tokens["input_ids"],
        skip_special_tokens=True,
    )

    tokenized = tokenizer(
        processed_question,
        example["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    return tokenized


# ============================================================
# Find best answer from one feature
# ============================================================

def get_best_span(
    start_logits,
    end_logits,
    offsets,
    sequence_ids,
):
    """
    Find the highest-scoring valid answer span inside
    the context portion of one tokenized feature.
    """

    context_indices = [
        i
        for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    if not context_indices:
        return None

    best_score = -float("inf")
    best_start = None
    best_end = None

    # Keep more start/end candidates to improve span selection.
    top_k = 35

    start_candidates = sorted(
        context_indices,
        key=lambda i: start_logits[i],
        reverse=True,
    )[:top_k]

    end_candidates = sorted(
        context_indices,
        key=lambda i: end_logits[i],
        reverse=True,
    )[:top_k]

    for start_index in start_candidates:

        for end_index in end_candidates:

            if end_index < start_index:
                continue

            if (
                end_index - start_index + 1
                > MAX_ANSWER_LENGTH
            ):
                continue

            start_offset = offsets[start_index][0]
            end_offset = offsets[end_index][1]

            if end_offset <= start_offset:
                continue

            score = (
                start_logits[start_index]
                + end_logits[end_index]
            )

            if score > best_score:

                best_score = score
                best_start = start_index
                best_end = end_index

    if best_start is None:
        return None

    return {
        "score": float(best_score),
        "start_token": best_start,
        "end_token": best_end,
        "start_char": offsets[best_start][0],
        "end_char": offsets[best_end][1],
    }


# ============================================================
# Evaluate one example
# ============================================================

def evaluate_example(example):

    features = prepare_evaluation_features(example)

    best_prediction = None

    num_features = len(
        features["input_ids"]
    )

    for feature_index in range(num_features):

        input_ids = torch.tensor(
            [features["input_ids"][feature_index]],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.tensor(
            [features["attention_mask"][feature_index]],
            dtype=torch.long,
            device=device,
        )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = (
            outputs.start_logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        end_logits = (
            outputs.end_logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        offsets = features[
            "offset_mapping"
        ][feature_index]

        # Use the sequence IDs from the exact feature that
        # was already tokenized above.
        sequence_ids = features.sequence_ids(
            feature_index
        )

        candidate = get_best_span(
            start_logits,
            end_logits,
            offsets,
            sequence_ids,
        )

        if candidate is None:
            continue

        if (
            best_prediction is None
            or candidate["score"]
            > best_prediction["score"]
        ):
            best_prediction = candidate

    # --------------------------------------------------------
    # Extract predicted text
    # --------------------------------------------------------

    if best_prediction is None:

        prediction = ""

    else:

        prediction = example["context"][
            best_prediction["start_char"]:
            best_prediction["end_char"]
        ]

    return {
        "id": example["id"],
        "question": example["question"],
        "context": example["context"],
        "ground_truth": example["answer_text"],
        "prediction": prediction,
        "num_features": num_features,
        "best_score": (
            best_prediction["score"]
            if best_prediction
            else None
        ),
    }


# ============================================================
# Run evaluation
# ============================================================

print()
print("=" * 60)
print("Running held-out QA evaluation...")
print("=" * 60)

results = []

for index, example in enumerate(val_dataset):

    result = evaluate_example(example)

    results.append(result)

    print(
        f"[{index + 1:03d}/{len(val_dataset)}] "
        f"{result['id']}"
    )


# ============================================================
# Calculate metrics
# ============================================================

em_scores = []
f1_scores = []

for result in results:

    em = exact_match_score(
        result["prediction"],
        result["ground_truth"],
    )

    f1 = token_f1_score(
        result["prediction"],
        result["ground_truth"],
    )

    result["exact_match"] = em
    result["token_f1"] = f1

    em_scores.append(em)
    f1_scores.append(f1)


exact_match = float(
    np.mean(em_scores)
)

token_f1 = float(
    np.mean(f1_scores)
)


# ============================================================
# Acceptance gate
# ============================================================

EM_THRESHOLD = 0.30
F1_THRESHOLD = 0.30

em_pass = exact_match >= EM_THRESHOLD
f1_pass = token_f1 >= F1_THRESHOLD

overall_pass = em_pass and f1_pass


# ============================================================
# Print metrics
# ============================================================

print()
print("=" * 60)
print("MODEL B ACCEPTANCE GATE")
print("=" * 60)

print()

print(f"Exact Match:    {exact_match:.4f}")
print(f"Required:       >= {EM_THRESHOLD:.2f}")
print(f"Status:         {'PASS' if em_pass else 'FAIL'}")

print()

print(f"Token-level F1: {token_f1:.4f}")
print(f"Required:       >= {F1_THRESHOLD:.2f}")
print(f"Status:         {'PASS' if f1_pass else 'FAIL'}")

print()

print(
    f"Overall Gate:   "
    f"{'PASS' if overall_pass else 'FAIL'}"
)


# ============================================================
# Long-context boundary cases
# ============================================================

print()
print("=" * 60)
print("Manual inspection: long-context cases")
print("=" * 60)

long_context_results = [
    result
    for result in results
    if result["num_features"] > 1
]

print(
    f"Examples requiring multiple features: "
    f"{len(long_context_results)}"
)

# Sort by number of features so the most challenging
# examples appear first.
long_context_results.sort(
    key=lambda x: x["num_features"],
    reverse=True,
)

manual_cases = long_context_results[
    :MANUAL_CASES
]

for index, result in enumerate(manual_cases, start=1):

    print()
    print("-" * 60)
    print(f"CASE {index}")
    print("-" * 60)

    print(f"ID: {result['id']}")
    print(f"Number of features: {result['num_features']}")

    print()
    print("Question:")
    print(result["question"][:1000])

    print()
    print("Ground truth:")
    print(result["ground_truth"])

    print()
    print("Prediction:")
    print(result["prediction"])

    print()
    print(f"EM: {result['exact_match']}")
    print(f"F1: {result['token_f1']:.4f}")


# ============================================================
# Save results
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

metrics = {
    "model": MODEL_B,
    "dataset": DATASET_PATH,
    "validation_examples": len(val_dataset),
    "exact_match": exact_match,
    "token_f1": token_f1,
    "thresholds": {
        "exact_match": EM_THRESHOLD,
        "token_f1": F1_THRESHOLD,
    },
    "gate": {
        "exact_match_pass": em_pass,
        "token_f1_pass": f1_pass,
        "overall_pass": overall_pass,
    },
}


with open(
    RESULTS_DIR / "metrics.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


with open(
    RESULTS_DIR / "predictions.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 60)
print("Evaluation files saved")
print("=" * 60)

print(
    f"Metrics: "
    f"{RESULTS_DIR / 'metrics.json'}"
)

print(
    f"Predictions: "
    f"{RESULTS_DIR / 'predictions.json'}"
)

print()
print("Model B evaluation completed.")