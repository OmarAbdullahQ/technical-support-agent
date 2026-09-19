from pathlib import Path
import json
import random

from datasets import load_dataset


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/model_c_source")
OUTPUT_DIR = Path("data")

RANDOM_SEED = 42

SAMPLES_PER_SOURCE = 50

SOURCES = [
    "gridco_synthetic",
    "ubuntu_dialogue_qa",
]


# ============================================================
# Load local parquet files
# ============================================================

# Search recursively because the downloaded repository
# stores the parquet files under a nested data/ directory.
parquet_files = sorted(DATA_DIR.rglob("train-*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No parquet files found inside: {DATA_DIR}"
    )

print("Parquet files found:")

for path in parquet_files:
    print(f"  - {path}")


# ============================================================
# Load dataset
# ============================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "parquet",
    data_files=[str(path) for path in parquet_files],
)["train"]

print(f"Total records loaded: {len(dataset)}")


# ============================================================
# Check available sources
# ============================================================

sources = set(dataset["source"])

print("\nAvailable sources:")

for source in sorted(sources):
    count = sum(
        1
        for value in dataset["source"]
        if value == source
    )

    print(f"  {source}: {count}")


# ============================================================
# Select examples
# ============================================================

selected = []

for source in SOURCES:

    source_dataset = dataset.filter(
        lambda example: example["source"] == source
    )

    print(
        f"\n{source}: "
        f"{len(source_dataset)} available"
    )

    if len(source_dataset) < SAMPLES_PER_SOURCE:
        raise ValueError(
            f"Source '{source}' contains only "
            f"{len(source_dataset)} records, but "
            f"{SAMPLES_PER_SOURCE} are required."
        )

    source_dataset = source_dataset.shuffle(
        seed=RANDOM_SEED
    ).select(range(SAMPLES_PER_SOURCE))

    selected.extend(source_dataset)


print(f"\nSelected records: {len(selected)}")


# ============================================================
# Convert conversations -> messages
# ============================================================

prepared = []

for example in selected:

    messages = []

    for turn in example["conversations"]:

        messages.append(
            {
                "role": turn["role"],
                "content": turn["content"],
            }
        )

    prepared.append(
        {
            "id": example["id"],
            "source": example["source"],
            "use_case": example["use_case"],
            "messages": messages,
        }
    )


# ============================================================
# Split 80/20 per source
# ============================================================

train_data = []
validation_data = []

for source in SOURCES:

    source_examples = [
        example
        for example in prepared
        if example["source"] == source
    ]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(source_examples)

    split_index = int(len(source_examples) * 0.8)

    train_data.extend(
        source_examples[:split_index]
    )

    validation_data.extend(
        source_examples[split_index:]
    )


# Shuffle final datasets

rng = random.Random(RANDOM_SEED)

rng.shuffle(train_data)
rng.shuffle(validation_data)


# ============================================================
# Save JSONL
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_path = OUTPUT_DIR / "sft_train.jsonl"
validation_path = OUTPUT_DIR / "sft_val.jsonl"


def save_jsonl(path, records):

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


save_jsonl(
    train_path,
    train_data
)

save_jsonl(
    validation_path,
    validation_data
)


# ============================================================
# Summary
# ============================================================

print("\nModel C data preparation complete.")

print(
    f"\nTraining examples:   {len(train_data)}"
)

print(
    f"Validation examples: {len(validation_data)}"
)


print("\nTraining distribution:")

for source in SOURCES:

    count = sum(
        1
        for example in train_data
        if example["source"] == source
    )

    print(
        f"  {source}: {count}"
    )


print("\nValidation distribution:")

for source in SOURCES:

    count = sum(
        1
        for example in validation_data
        if example["source"] == source
    )

    print(
        f"  {source}: {count}"
    )


print("\nSaved files:")

print(f"  {train_path}")
print(f"  {validation_path}")