from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path("models/intent_classifier")
DATASET_NAME = "cngchis/Support-Ticket-Router-12K-Cleaned"

TEST_SIZE = 300
RANDOM_STATE = 42

OUTPUT_DIR = Path("evaluation/results/intent_classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load model and tokenizer
# --------------------------------------------------

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)

labels = sorted(set(dataset["train"]["label"]))

label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("\nLabels:")
for label_id, label in id2label.items():
    print(f"{label_id}: {label}")


# --------------------------------------------------
# Convert labels to IDs
# --------------------------------------------------

def encode_labels(example):
    example["label"] = label2id[example["label"]]
    return example


dataset = dataset.map(encode_labels)


# --------------------------------------------------
# Create the same stratified test split used in training
# --------------------------------------------------

texts = np.array(dataset["train"]["text"])
train_labels = np.array(dataset["train"]["label"])

_, test_indices = train_test_split(
    np.arange(len(texts)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=train_labels,
)

test_dataset = dataset["train"].select(test_indices)

print(f"\nEvaluation samples: {len(test_dataset)}")


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )


test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
)


# Keep only the columns needed by the model
test_dataset = test_dataset.remove_columns(
    [
        column
        for column in test_dataset.column_names
        if column not in ["input_ids", "attention_mask", "label"]
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
)

print("\nRunning predictions...")

predictions = trainer.predict(test_dataset)

predicted_ids = np.argmax(
    predictions.predictions,
    axis=-1,
)

true_ids = np.array(test_dataset["label"])


# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

accuracy = accuracy_score(
    true_ids,
    predicted_ids,
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        average="macro",
        zero_division=0,
    )
)


print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {precision_macro:.4f}")
print(f"Macro Recall:      {recall_macro:.4f}")
print(f"Macro F1:          {f1_macro:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

report = classification_report(
    true_ids,
    predicted_ids,
    labels=list(range(len(labels))),
    target_names=labels,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    true_ids,
    predicted_ids,
    labels=list(range(len(labels))),
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted\n")

# Header
header = f"{'Actual':<15}"

for label in labels:
    header += f"{label:>15}"

print(header)

# Matrix rows
for i, row in enumerate(cm):
    row_text = f"{labels[i]:<15}"

    for value in row:
        row_text += f"{value:>15}"

    print(row_text)


# --------------------------------------------------
# Recall requirement check
# --------------------------------------------------

class_precision, class_recall, class_f1, class_support = (
    precision_recall_fscore_support(
        true_ids,
        predicted_ids,
        labels=list(range(len(labels))),
        zero_division=0,
    )
)

print("\n" + "=" * 60)
print("RECALL REQUIREMENT CHECK")
print("=" * 60)

recall_requirement = 0.60

all_pass = True

for i, label in enumerate(labels):

    recall = class_recall[i]

    status = "PASS" if recall >= recall_requirement else "FAIL"

    print(
        f"{label:<15} "
        f"Recall: {recall:.4f} "
        f"-> {status}"
    )

    if recall < recall_requirement:
        all_pass = False


print("\nMinimum recall requirement:", recall_requirement)

if all_pass:
    print("Result: PASS - No class has recall below 0.60.")
else:
    print("Result: FAIL - At least one class has recall below 0.60.")


# --------------------------------------------------
# Save results
# --------------------------------------------------

report_path = OUTPUT_DIR / "classification_report.txt"
matrix_path = OUTPUT_DIR / "confusion_matrix.npy"
matrix_text_path = OUTPUT_DIR / "confusion_matrix.txt"


# Save classification report
with open(report_path, "w", encoding="utf-8") as file:
    file.write("Model A - Intent Classifier Evaluation\n")
    file.write("=" * 60 + "\n\n")

    file.write("Overall Metrics\n")
    file.write("-" * 60 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision_macro:.4f}\n")
    file.write(f"Macro Recall: {recall_macro:.4f}\n")
    file.write(f"Macro F1: {f1_macro:.4f}\n\n")

    file.write("Classification Report\n")
    file.write("-" * 60 + "\n")
    file.write(report)

    file.write("\nRecall Requirement Check\n")
    file.write("-" * 60 + "\n")

    for i, label in enumerate(labels):
        status = (
            "PASS"
            if class_recall[i] >= recall_requirement
            else "FAIL"
        )

        file.write(
            f"{label}: "
            f"recall={class_recall[i]:.4f} "
            f"{status}\n"
        )

    file.write(
        f"\nOverall requirement: "
        f"{'PASS' if all_pass else 'FAIL'}\n"
    )


# Save raw NumPy matrix
np.save(matrix_path, cm)


# Save human-readable confusion matrix
with open(matrix_text_path, "w", encoding="utf-8") as file:
    file.write("Model A - Confusion Matrix\n")
    file.write("=" * 60 + "\n\n")

    file.write("Rows = Actual\n")
    file.write("Columns = Predicted\n\n")

    header = f"{'Actual':<15}"

    for label in labels:
        header += f"{label:>15}"

    file.write(header + "\n")

    for i, row in enumerate(cm):
        row_text = f"{labels[i]:<15}"

        for value in row:
            row_text += f"{value:>15}"

        file.write(row_text + "\n")


print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)

print(f"Classification report: {report_path}")
print(f"Confusion matrix (.npy): {matrix_path}")
print(f"Confusion matrix (.txt): {matrix_text_path}")