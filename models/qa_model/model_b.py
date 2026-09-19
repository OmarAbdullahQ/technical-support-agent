from datasets import load_from_disk, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)


# ============================================================
# Configuration
# ============================================================

MODEL_B = "distilbert-base-uncased"

DATASET_PATH = "data/model_b_dataset"
OUTPUT_DIR = "models/qa_model"

MAX_LENGTH = 384
DOC_STRIDE = 96

# TechQA questions can contain long problem descriptions.
# Limit the question portion so enough space remains for context.
MAX_QUESTION_LENGTH = 128

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

print("=" * 60)
print("Loading Model B dataset...")
print("=" * 60)

dataset = load_from_disk(DATASET_PATH)

print(dataset)


# ============================================================
# Convert TechQA dataset to SQuAD-style format
# ============================================================

def build_qa_examples(dataset_split):
    """
    Convert the downloaded TechQA format:

        question
        answer
        is_impossible
        contexts[]

    into:

        question
        context
        answer_text
        answer_start

    Only examples where the answer appears inside one of the
    provided contexts are kept.
    """

    qa_examples = []

    impossible_count = 0
    unmatched_count = 0

    for example in dataset_split:

        if example["is_impossible"]:
            impossible_count += 1
            continue

        question = example["question"]
        answer = example["answer"].strip()

        matched_context = None
        answer_start = None

        for context_item in example["contexts"]:

            context = context_item["text"]

            # Exact matching
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
            unmatched_count += 1
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

    print()
    print("Dataset conversion results:")
    print(f"  Valid QA examples: {len(qa_examples)}")
    print(f"  Impossible examples skipped: {impossible_count}")
    print(f"  Unmatched answers skipped: {unmatched_count}")

    return qa_examples


qa_examples = build_qa_examples(dataset["train"])


# ============================================================
# Minimum dataset requirement
# ============================================================

MIN_QA_PAIRS = 30

if len(qa_examples) < MIN_QA_PAIRS:
    raise ValueError(
        f"Only {len(qa_examples)} valid QA pairs were found. "
        f"At least {MIN_QA_PAIRS} are required."
    )

print()
print(
    f"Minimum requirement check: PASS "
    f"({len(qa_examples)} QA pairs)"
)


# ============================================================
# Create train / validation split
# ============================================================

train_examples, val_examples = train_test_split(
    qa_examples,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train_dataset = Dataset.from_list(train_examples)
val_dataset = Dataset.from_list(val_examples)

print()
print("=" * 60)
print("Train / Validation split")
print("=" * 60)

print(f"Train examples:      {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")


# ============================================================
# Load tokenizer and model
# ============================================================

print()
print("=" * 60)
print("Loading DistilBERT QA model...")
print("=" * 60)

tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B)

model_b = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_B
)


# ============================================================
# Prepare QA features
# ============================================================

def prepare_qa_features(examples):

    # --------------------------------------------------------
    # Limit question length
    # --------------------------------------------------------
    #
    # Some TechQA "questions" contain long problem descriptions.
    # With truncation="only_second", an extremely long question
    # can leave too little room for the context and make the
    # requested DOC_STRIDE invalid.
    #
    # We therefore truncate only the question portion.
    # The context and answer span remain unchanged.
    # --------------------------------------------------------

    processed_questions = []

    for question in examples["question"]:

        question_tokens = tokenizer_b(
            question.strip(),
            truncation=True,
            max_length=MAX_QUESTION_LENGTH,
            add_special_tokens=False,
        )

        truncated_question = tokenizer_b.decode(
            question_tokens["input_ids"],
            skip_special_tokens=True,
        )

        processed_questions.append(truncated_question)

    # --------------------------------------------------------
    # Tokenize question + context
    # --------------------------------------------------------

    tokenized = tokenizer_b(
        processed_questions,
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop(
        "overflow_to_sample_mapping"
    )

    offsets = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    # --------------------------------------------------------
    # Find answer token positions
    # --------------------------------------------------------

    valid_feature_indices = []

    for feature_index, feature_offsets in enumerate(offsets):

        input_ids = tokenized["input_ids"][feature_index]

        cls_index = input_ids.index(
            tokenizer_b.cls_token_id
        )

        sequence_ids = tokenized.sequence_ids(
            feature_index
        )

        sample_index = sample_mapping[feature_index]

        answer_start = examples["answer_start"][sample_index]

        answer_text = examples["answer_text"][sample_index]

        answer_end = answer_start + len(answer_text)

        # ----------------------------------------------------
        # Find context token boundaries
        # ----------------------------------------------------

        context_start = 0

        while (
            context_start < len(sequence_ids)
            and sequence_ids[context_start] != 1
        ):
            context_start += 1

        context_end = len(sequence_ids) - 1

        while (
            context_end >= 0
            and sequence_ids[context_end] != 1
        ):
            context_end -= 1

        # Safety check
        if (
            context_start >= len(sequence_ids)
            or context_end < 0
        ):
            continue

        # ----------------------------------------------------
        # Check whether answer is inside this feature
        # ----------------------------------------------------

        if (
            feature_offsets[context_start][0] > answer_start
            or feature_offsets[context_end][1] < answer_end
        ):
            # This feature does not contain the answer.
            # Since every retained example is answerable,
            # there is no reason to train this feature as CLS.
            continue

        # ----------------------------------------------------
        # Find answer start token
        # ----------------------------------------------------

        token_start = context_start

        while (
            token_start <= context_end
            and feature_offsets[token_start][1] <= answer_start
        ):
            token_start += 1

        # ----------------------------------------------------
        # Find answer end token
        # ----------------------------------------------------

        token_end = context_end

        while (
            token_end >= context_start
            and feature_offsets[token_end][0] >= answer_end
        ):
            token_end -= 1

        # Safety check
        if (
            token_start > context_end
            or token_end < context_start
        ):
            continue

        start_positions.append(token_start)
        end_positions.append(token_end)
        valid_feature_indices.append(feature_index)

    # --------------------------------------------------------
    # Keep only features that contain the answer
    # --------------------------------------------------------

    tokenized = {
        key: [
            values[index]
            for index in valid_feature_indices
        ]
        for key, values in tokenized.items()
    }

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


# ============================================================
# Tokenize datasets
# ============================================================

print()
print("=" * 60)
print("Tokenizing training data...")
print("=" * 60)

qa_train_features = train_dataset.map(
    prepare_qa_features,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print()
print("Tokenizing validation data...")

qa_val_features = val_dataset.map(
    prepare_qa_features,
    batched=True,
    remove_columns=val_dataset.column_names,
)

print()
print(f"Train features:      {len(qa_train_features)}")
print(f"Validation features: {len(qa_val_features)}")


# ============================================================
# Training arguments
# ============================================================

args_b = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    # More training for the small QA dataset
    num_train_epochs=4,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    weight_decay=0.01,

    report_to="none",
)


# ============================================================
# Trainer
# ============================================================

trainer_b = Trainer(
    model=model_b,
    args=args_b,
    train_dataset=qa_train_features,
    eval_dataset=qa_val_features,
)


# ============================================================
# Train
# ============================================================

print()
print("=" * 60)
print("Starting Model B training...")
print("=" * 60)

trainer_b.train()


# ============================================================
# Evaluate
# ============================================================

print()
print("=" * 60)
print("Evaluating Model B...")
print("=" * 60)

evaluation_results = trainer_b.evaluate()

print()
print("Evaluation results:")

for key, value in evaluation_results.items():
    print(f"{key}: {value}")


# ============================================================
# Save model and tokenizer
# ============================================================

print()
print("=" * 60)
print("Saving Model B...")
print("=" * 60)

trainer_b.save_model(OUTPUT_DIR)

tokenizer_b.save_pretrained(OUTPUT_DIR)

print()
print(f"Model saved to: {OUTPUT_DIR}")
print("Model B training completed successfully.")