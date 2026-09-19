from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "nvidia/TechQA-RAG-Eval"
OUTPUT_DIR = Path("data/model_b_dataset")


def main():
    print("Loading Model B dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset:")
    print(dataset)

    print("\nSplits:")
    for split in dataset:
        print(f"{split}: {len(dataset[split])} samples")

    print("\nFeatures:")
    print(dataset["train"].features)

    print("\nFirst example:")
    print(dataset["train"][0])

    print("\nSaving dataset locally...")

    dataset.save_to_disk(OUTPUT_DIR)

    print(f"\nDataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()