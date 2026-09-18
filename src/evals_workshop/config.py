"""Typed settings loaded from the environment. Every model and path reference reads
from here, anchored to the repo root so a notebook works from its own directory."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

load_dotenv(REPO_ROOT / ".env")


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
    """Anchor a relative tracking store to the repo root, so a notebook running from
    notebooks/ logs to the same place as the command line and the MLflow UI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    for scheme, sep in (("sqlite:///", "sqlite:///"), ("file:", "file:")):
        if uri.startswith(scheme):
            path = Path(uri[len(sep) :])
            if path.is_absolute():
                return uri
            resolved = REPO_ROOT / path
            return (
                f"sqlite:///{resolved.as_posix()}"
                if scheme.startswith("sqlite")
                else resolved.as_uri()
            )
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
