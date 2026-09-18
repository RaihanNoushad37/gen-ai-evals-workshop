"""Load cached splits, enforce the prod label lock, and assert split integrity.

No network access here — `prepare_data.py` builds data/splits.parquet once, offline
from that point on.
"""

import hashlib

import pandas as pd

from .config import DATA_DIR

_SPLITS_PATH = DATA_DIR / "splits.parquet"
_UNLOCK_SENTINEL = DATA_DIR / ".prod_unlocked"
_LABEL_COLUMNS = ["label", "spans"]


def _prod_unlocked() -> bool:
    return _UNLOCK_SENTINEL.exists()


def unlock_prod_labels() -> None:
    _UNLOCK_SENTINEL.parent.mkdir(exist_ok=True)
    _UNLOCK_SENTINEL.touch(exist_ok=True)


def load_split(name: str, with_labels: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(_SPLITS_PATH)
    split = df[df["source_split"] == name].reset_index(drop=True)
    if name == "prod" and with_labels and not _prod_unlocked():
        raise RuntimeError(
            "prod labels are locked. Call unlock_prod_labels() when the notebook "
            "reaches the reveal step."
        )
    if name == "prod" and not with_labels:
        split = split.drop(columns=_LABEL_COLUMNS)
    return split


def _doc_id(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def check_integrity() -> None:
    df = pd.read_parquet(_SPLITS_PATH)
    splits = {name: set(sub["id"]) for name, sub in df.groupby("source_split")}

    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = splits[a] & splits[b]
            assert not overlap, f"id overlap between {a} and {b}: {overlap}"

    df["_doc_id"] = df["context"].map(_doc_id)
    dev_docs = set(df.loc[df["source_split"] == "dev", "_doc_id"])
    for name in ("calib", "prod"):
        other_docs = set(df.loc[df["source_split"] == name, "_doc_id"])
        overlap = dev_docs & other_docs
        assert not overlap, f"dev shares source documents with {name}: {overlap}"
