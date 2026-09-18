"""Typed settings loaded from the environment. Every model and path reference reads
from here, anchored to the repo root so a notebook works from its own directory."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

load_dotenv(REPO_ROOT / ".env")

# MLflow 3.x raises on the plain filesystem tracking backend unless this is set. The
# workshop stays on a local file store: no account, no separate database to manage.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    judge_model: str
    judge_tpm: int
    max_concurrency: int
    mlflow_tracking_uri: str
    random_seed: int


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            f"Copy .env.example to .env and fill it in, then re-run."
        )
    return value


def _tracking_uri() -> str:
    """Resolve a relative `file:` URI against the repo root, so runs from a notebook
    land in the same store as runs from the command line."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    if uri.startswith("file:"):
        path = Path(uri[len("file:") :])
        if not path.is_absolute():
            return (REPO_ROOT / path).as_uri()
    return uri


def load_settings() -> Settings:
    return Settings(
        groq_api_key=_require("GROQ_API_KEY"),
        judge_model=os.environ.get("JUDGE_MODEL", "groq:/openai/gpt-oss-20b"),
        judge_tpm=int(os.environ.get("JUDGE_TPM", "8000")),
        max_concurrency=int(os.environ.get("MAX_CONCURRENCY", "3")),
        mlflow_tracking_uri=_tracking_uri(),
        random_seed=int(os.environ.get("RANDOM_SEED", "17")),
    )


settings = load_settings()
