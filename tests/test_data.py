"""Tests against the built parquet. Skipped on a fresh clone before `make data`."""

import re
from pathlib import Path

import pandas as pd
import pytest

from evals_workshop import data

SPLITS_PATH = Path("data/splits.parquet")

pytestmark = pytest.mark.skipif(
    not SPLITS_PATH.exists(), reason="data/splits.parquet not built; run `make data`"
)


@pytest.fixture
def df():
    return pd.read_parquet(SPLITS_PATH)


def test_split_sizes(df):
    counts = df["source_split"].value_counts().to_dict()
    assert counts == {"explore": 12, "dev": 16, "calib": 24, "prod": 24}


def test_dev_calib_balanced(df):
    for name in ("dev", "calib"):
        sub = df[df["source_split"] == name]
        assert sub["label"].sum() == len(sub) // 2


def test_all_ids_disjoint(df):
    grouped = df.groupby("source_split")["id"]
    seen = set()
    for _, ids in grouped:
        ids = set(ids)
        assert not (seen & ids)
        seen |= ids


def test_dev_documents_disjoint_from_calib_and_prod():
    data.check_integrity()  # raises AssertionError on any violation


def test_load_split_works_from_any_directory(tmp_path, monkeypatch):
    """Jupyter runs a notebook from notebooks/, so paths must not depend on the CWD."""
    monkeypatch.chdir(tmp_path)
    assert len(data.load_split("explore")) == 12


def test_token_budget_respected(df):
    assert (df["n_tokens"] <= 400).all()


def test_explore_prevalence_near_natural(df):
    # Against the retained pool's rate from data/report.md, not the combined splits,
    # whose rate is inflated by the balanced dev and calib sets.
    natural_rate = float(
        re.search(
            r"Retained pool hallucination rate: ([\d.]+)", Path("data/report.md").read_text()
        ).group(1)
    )
    explore_rate = df.loc[df["source_split"] == "explore", "label"].mean()
    assert abs(explore_rate - natural_rate) <= 0.10


@pytest.fixture
def locked_prod(tmp_path, monkeypatch):
    """Point the unlock sentinel at a temp dir, so the suite can't unlock the real one."""
    monkeypatch.setattr(data, "_UNLOCK_SENTINEL", tmp_path / ".prod_unlocked")


def test_load_split_prod_raises_before_unlock(locked_prod):
    with pytest.raises(RuntimeError):
        data.load_split("prod")


def test_load_split_prod_hides_labels_before_unlock(locked_prod):
    prod = data.load_split("prod", with_labels=False)
    assert "label" not in prod.columns
    assert "spans" not in prod.columns


def test_load_split_prod_reveals_labels_after_unlock(locked_prod):
    data.unlock_prod_labels()
    prod = data.load_split("prod")
    assert "label" in prod.columns
