"""Build the four workshop splits from RAGTruth. Deterministic, seeded, idempotent.

Run once by the instructor: `make data`. See docs/ragtruth-schema.md for the source
schema (probed once; do not call load_dataset again).
"""

import hashlib
import json
from collections.abc import Iterator

import pandas as pd
import tiktoken
from datasets import load_dataset

from evals_workshop.config import DATA_DIR, settings

# Sized so the whole workshop runs twice inside a 200,000 token/day budget, leaving
# room to re-run after a mistake. Small enough that the intervals are wide — which is
# the subject of §6.4, not a flaw to hide. See README "API budget".
TOKEN_BUDGET = 400
SPLIT_SIZES = {"explore": 12, "dev": 16, "calib": 24, "prod": 24}


def _label_and_spans(raw_labels: str) -> tuple[int, str]:
    spans = json.loads(raw_labels) if raw_labels else []
    return (1 if spans else 0), json.dumps(spans)


def _doc_id(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def load_qa_pool() -> tuple[pd.DataFrame, str]:
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    df = ds.to_pandas()
    df = df[df["task_type"] == "QA"].copy()

    enc = tiktoken.get_encoding("cl100k_base")
    labels, spans = zip(*df["hallucination_labels"].map(_label_and_spans), strict=True)
    df["label"] = labels
    df["spans"] = spans
    df["response"] = df["output"]
    df["generator_model"] = df["model"]
    df["n_tokens"] = (df["context"] + " " + df["response"]).map(lambda t: len(enc.encode(t)))
    df["source_doc_id"] = df["context"].map(_doc_id)
    return df[
        [
            "id",
            "query",
            "context",
            "response",
            "generator_model",
            "label",
            "spans",
            "n_tokens",
            "source_doc_id",
        ]
    ], ds._fingerprint


def _sample_balanced(pool: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    half = n // 2
    pos = pool[pool["label"] == 1].sample(n=half, random_state=rng)
    neg = pool[pool["label"] == 0].sample(n=n - half, random_state=rng)
    return pd.concat([pos, neg])


def build_splits(pool: pd.DataFrame, seed: int) -> Iterator[tuple[str, pd.DataFrame]]:
    doc_ids = pool["source_doc_id"].drop_duplicates().sample(frac=1, random_state=seed).tolist()

    dev_docs: list[str] = []
    dev_pos = dev_neg = 0
    target_half = SPLIT_SIZES["dev"] // 2
    for doc in doc_ids:
        if dev_pos >= target_half and dev_neg >= target_half:
            break
        rows = pool[pool["source_doc_id"] == doc]
        dev_docs.append(doc)
        dev_pos += int((rows["label"] == 1).sum())
        dev_neg += int((rows["label"] == 0).sum())

    dev_pool = pool[pool["source_doc_id"].isin(dev_docs)]
    rest_pool = pool[~pool["source_doc_id"].isin(dev_docs)]

    dev = _sample_balanced(dev_pool, SPLIT_SIZES["dev"], seed)

    calib = _sample_balanced(rest_pool, SPLIT_SIZES["calib"], seed + 1)
    rest_pool = rest_pool.drop(calib.index)

    prod = rest_pool.sample(n=SPLIT_SIZES["prod"], random_state=seed + 2)
    rest_pool = rest_pool.drop(prod.index)

    explore = rest_pool.sample(n=SPLIT_SIZES["explore"], random_state=seed + 3)

    for name, split in [("explore", explore), ("dev", dev), ("calib", calib), ("prod", prod)]:
        split = split.copy()
        split["source_split"] = name
        yield name, split


def write_report(
    pool_all: pd.DataFrame,
    retained: pd.DataFrame,
    discarded: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    seed: int,
    revision: str,
) -> None:
    lines = ["# Data preparation report", ""]
    lines.append(f"Source: `wandb/RAGTruth-processed`, fingerprint `{revision}`. Seed: {seed}.")
    lines.append("")
    lines.append("## Filtering")
    lines.append(f"- QA task rows: {len(pool_all)}")
    lines.append(
        f"- Retained (n_tokens <= {TOKEN_BUDGET}): {len(retained)} "
        f"({len(retained) / len(pool_all):.1%})"
    )
    lines.append(
        f"- Discarded (n_tokens > {TOKEN_BUDGET}): {len(discarded)} "
        f"({len(discarded) / len(pool_all):.1%})"
    )
    lines.append("")
    lines.append("## Hallucination rate: retained vs discarded (transportability caveat)")
    r_rate = retained["label"].mean()
    d_rate = discarded["label"].mean() if len(discarded) else float("nan")
    lines.append(f"- Retained pool hallucination rate: {r_rate:.4f}")
    lines.append(f"- Discarded pool hallucination rate: {d_rate:.4f}")
    lines.append(
        "- If these differ meaningfully, sensitivity/specificity measured on the "
        "(short, retained) calibration set may not transport to longer contexts."
    )
    lines.append("")
    lines.append("## Splits")
    lines.append("| split | n | prevalence | median n_tokens |")
    lines.append("|---|---|---|---|")
    for name, split in splits.items():
        lines.append(
            f"| {name} | {len(split)} | {split['label'].mean():.3f} | "
            f"{split['n_tokens'].median():.0f} |"
        )
    (DATA_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    seed = settings.random_seed

    pool_all, revision = load_qa_pool()
    retained = pool_all[pool_all["n_tokens"] <= TOKEN_BUDGET].copy()
    discarded = pool_all[pool_all["n_tokens"] > TOKEN_BUDGET].copy()

    splits = dict(build_splits(retained, seed))
    combined = pd.concat(splits.values(), ignore_index=True).drop(columns=["source_doc_id"])
    combined.to_parquet(DATA_DIR / "splits.parquet", index=False)

    write_report(pool_all, retained, discarded, splits, seed, revision)
    print(f"Wrote {len(combined)} rows to data/splits.parquet")
    print("Wrote data/report.md")


if __name__ == "__main__":
    main()
