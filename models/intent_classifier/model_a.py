from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from collections import Counter


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = "cngchis/Support-Ticket-Router-12K-Cleaned"
MODEL_A = "distilbert-base-uncased"
OUTPUT_DIR = "models/intent_classifier"

TRAIN_SIZE = 3000
VALIDATION_SIZE = 300
TEST_SIZE = 300

RANDOM_SEED = 42


# ============================================================
# 1. Load Dataset
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

ds = load_dataset(DATASET_NAME)

print("\nDataset:")
print(ds)


# ============================================================
# 2. Inspect Labels
# ============================================================

dataset = ds["train"]

labels = sorted(set(dataset["label"]))

print("\nLabels found in dataset:")
for i, label in enumerate(labels):
    print(f"{i}: {label}")

print(f"\nNumber of labels: {len(labels)}")


# ============================================================
# 3. Check TODO Requirement
#    At least 12 examples per intent
# ============================================================

label_counts = Counter(dataset["label"])

print("\nExamples per intent:")
for label in labels:
    print(f"{label}: {label_counts[label]}")


minimum_examples = min(label_counts.values())

if minimum_examples < 12:
    raise ValueError(
        "Dataset does not satisfy the requirement of at least "
        "12 examples per intent."
    )

print(
    f"\nRequirement satisfied: every intent has at least "
    f"{minimum_examples} examples."
)


# ============================================================
# 4. Create Label Mapping
# ============================================================

label2id = {
    label: i
    for i, label in enumerate(labels)
}

id2label = {
    i: label
    for label, i in label2id.items()
}

print("\nLabel mapping:")
for label, label_id in label2id.items():
    print(f"{label} -> {label_id}")


# ============================================================
# 5. Convert String Labels to Integer IDs
# ============================================================

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example


dataset = dataset.map(encode_labels)


# ============================================================
# 6. Stratified Train / Validation / Test Split
# ============================================================

total_required = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE

if len(dataset) < total_required:
    raise ValueError(
        f"Dataset contains only {len(dataset)} examples, "
        f"but {total_required} are required."
    )

all_indices = np.arange(len(dataset))

labels_for_split = np.array(dataset["label"])


# First split:
# 3000 training
# remaining 600 temporary data
train_indices, temp_indices = train_test_split(
    all_indices,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels_for_split,
)


# Second split:
# 300 validation
# 300 test
temp_labels = labels_for_split[temp_indices]

validation_indices, test_indices = train_test_split(
    temp_indices,
    train_size=VALIDATION_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)


train_a = dataset.select(train_indices.tolist())
val_a = dataset.select(validation_indices.tolist())
test_a = dataset.select(test_indices.tolist())


print("\n" + "=" * 60)
print("Stratified Split")
print("=" * 60)

print(f"Training samples:   {len(train_a)}")
print(f"Validation samples: {len(val_a)}")
print(f"Test samples:       {len(test_a)}")


# ============================================================
# 7. Verify Stratification
# ============================================================

def print_distribution(name, split):
    counts = Counter(split["label"])

    print(f"\n{name} distribution:")

    for label_id in range(len(labels)):
        label_name = id2label[label_id]
        count = counts[label_id]
        percentage = (count / len(split)) * 100

        print(
            f"{label_name:15} "
            f"{count:4} "
            f"({percentage:5.2f}%)"
        )


print_distribution("Train", train_a)
print_distribution("Validation", val_a)
print_distribution("Test", test_a)


# ============================================================
# 8. Tokenizer
# ============================================================

print("\n" + "=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer_a = AutoTokenizer.from_pretrained(MODEL_A)


def tokenize_a(batch):
    return tokenizer_a(
        batch["text"],
        truncation=True,
        max_length=128,
    )


train_a = train_a.map(tokenize_a, batched=True)
val_a = val_a.map(tokenize_a, batched=True)
test_a = test_a.map(tokenize_a, batched=True)


# ============================================================
# 9. Dynamic Padding
# ============================================================

data_collator_a = DataCollatorWithPadding(
    tokenizer=tokenizer_a
)


# ============================================================
# 10. Model
# ============================================================

print("\n" + "=" * 60)
print("Loading model...")
print("=" * 60)

model_a = AutoModelForSequenceClassification.from_pretrained(
    MODEL_A,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)


# ============================================================
# 11. Evaluation Metrics
# ============================================================

def compute_cls_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels_true = eval_pred.label_ids

    # Some models may return predictions as a tuple.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_true,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(
            labels_true,
            preds,
        ),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


# ============================================================
# 12. Training Arguments
# ============================================================

args_a = TrainingArguments(
    output_dir=OUTPUT_DIR,

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=3,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",
    greater_is_better=True,

    report_to="none",
)


# ============================================================
# 13. Trainer
# ============================================================

trainer_a = Trainer(
    model=model_a,
    args=args_a,

    train_dataset=train_a,
    eval_dataset=val_a,

    processing_class=tokenizer_a,

    data_collator=data_collator_a,

    compute_metrics=compute_cls_metrics,
)


# ============================================================
# 14. Train
# ============================================================

print("\n" + "=" * 60)
print("Starting Model A training...")
print("=" * 60)

trainer_a.train()


# ============================================================
# 15. Evaluate on Test Set
# ============================================================

print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

test_results = trainer_a.evaluate(
    test_a
)

print("\nTest Results:")

for key, value in test_results.items():
    print(f"{key}: {value}")


# ============================================================
# 16. Save Model + Tokenizer
# ============================================================

print("\n" + "=" * 60)
print("Saving Model A...")
print("=" * 60)

trainer_a.save_model(OUTPUT_DIR)
tokenizer_a.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nTraining completed successfully.")