from datasets import load_dataset


DATASET_NAME = "cngchis/Support-Ticket-Router-12K-Cleaned"


def main():
    print("Loading Model A dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset:")
    print(dataset)

    print("\nFeatures:")
    print(dataset["train"].features)

    print("\nNumber of samples:")
    for split in dataset:
        print(f"{split}: {len(dataset[split])}")

    print("\nFirst training example:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()