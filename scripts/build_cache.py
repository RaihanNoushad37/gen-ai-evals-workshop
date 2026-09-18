"""Harvest data/.cache/*.json into data/cached_results.parquet for offline fallback.

Run by the instructor after a live session, once the caches hold real judge output.
Ships in the repo so USE_CACHED_RESULTS=1 works for students with no key left. See
README "API budget".
"""

import json

import pandas as pd

from evals_workshop.config import DATA_DIR

CACHE_DIR = DATA_DIR / ".cache"
OUTPUT_PATH = DATA_DIR / "cached_results.parquet"


def harvest() -> pd.DataFrame:
    rows = {}
    for path in sorted(CACHE_DIR.glob("*.json")):
        cache = json.loads(path.read_text())
        for key, result in cache.items():
            if result.get("error") is None:
                rows[key] = result["text"]
    return pd.DataFrame(
        {"cache_key": list(rows.keys()), "text": list(rows.values()), "error": None}
    )


def main() -> None:
    if not CACHE_DIR.exists() or not any(CACHE_DIR.glob("*.json")):
        raise SystemExit(f"No cache files under {CACHE_DIR}. Run the notebooks first.")

    df = harvest()
    if df.empty:
        raise SystemExit(f"{CACHE_DIR} has no successful (error-free) entries to harvest.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
