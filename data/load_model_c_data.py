from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "nodevs/gridco-helpdesk-finetune"

OUTPUT_DIR = Path("data/model_c_source")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print(f"Downloading dataset: {REPO_ID}...")

path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=OUTPUT_DIR,
)

print(f"\nDataset downloaded successfully.")
print(f"Location: {path}")