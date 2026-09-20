"""Central environment configuration; paths are relative to the repository."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    groq_api_key: str = Field(default="", repr=False)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = Field(default="", repr=False)
    langfuse_host: str = "https://cloud.langfuse.com"
    database_url: str = Field(default="sqlite:///data/mock_support.db", repr=False)
    api_key: str = Field(default="", repr=False)
    model_path: Path = ROOT / "models"
    support_base_model: str = ""
    local_files_only: bool = True
    model_device: str = "cpu"
    classifier_threshold: float = Field(default=0.80, ge=0, le=1)
    qa_min_score: float = Field(default=0.05, ge=0, le=1)
    kb_min_score: float = Field(default=0.08, ge=0, le=1)
    max_new_tokens: int = Field(default=192, ge=1, le=512)
    groq_timeout: float = Field(default=15, gt=0, le=120)
    kb_dataset_path: Path = ROOT / "data/model_b_dataset"
    kb_docs_path: Path = ROOT / "docs/kb"
    golden_set_path: Path = ROOT / "data/golden_set.jsonl"
    public_model: str = "tuwaiq-tech-support-agent"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    values = {name: os.environ[name.upper()] for name in Settings.model_fields
              if name.upper() in os.environ and name != "public_model"}
    for name in ("model_path", "kb_dataset_path", "kb_docs_path", "golden_set_path"):
        if name in values:
            path = Path(values[name])
            values[name] = path if path.is_absolute() else ROOT / path
    return Settings(**values)
