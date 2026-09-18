"""Token-aware pacing, backoff, and an on-disk cache around raw model calls.

Students read this rather than write it. The binding constraint on a free tier is
tokens per minute, not requests per minute, so a concurrency limit alone is not
enough — see workshop-spec.md §10.
"""

import asyncio
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

import litellm
import pandas as pd
import tiktoken

from .config import DATA_DIR

_ENC = tiktoken.get_encoding("cl100k_base")
_MAX_RETRIES = 8
_MAX_BACKOFF_SECONDS = 60
_GIVE_UP_AFTER_SECONDS = 300  # a wait longer than this won't clear inside a session
_OFFLINE_CACHE_PATH = DATA_DIR / "cached_results.parquet"
_EXPECTED_COMPLETION_TOKENS = 150  # reserved up front, corrected once usage is known


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


def _load_cache(cache_path: Path | None) -> dict:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def _save_cache(cache_path: Path | None, cache: dict) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))


def _load_offline_cache() -> dict:
    if not _OFFLINE_CACHE_PATH.exists():
        return {}
    df = pd.read_parquet(_OFFLINE_CACHE_PATH)
    return {
        row["cache_key"]: {"text": row["text"], "error": row["error"]}
        for row in df.to_dict("records")
    }


class _TokenWindow:
    """Rolling one-minute token budget: paced rather than exceeded.

    `acquire` reserves an estimate before the call so admission never oversubscribes
    the window; `correct` replaces it with the usage the provider actually reports.
    """

    def __init__(self, tpm: int):
        self._tpm = tpm
        self._events: list[list[float | int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, n_tokens: int) -> list[float | int]:
        # Sleep outside the lock, so other workers can still check the budget.
        while True:
            async with self._lock:
                now = time.monotonic()
                self._events = [e for e in self._events if now - e[0] < 60]
                used = sum(e[1] for e in self._events)
                if used + n_tokens <= self._tpm:
                    event = [now, n_tokens]
                    self._events.append(event)
                    return event
                oldest = self._events[0][0]
                wait = max(60 - (now - oldest), 0.5)
            await asyncio.sleep(wait)

    async def correct(self, event: list[float | int], actual_tokens: int) -> None:
        async with self._lock:
            if event in self._events:
                event[1] = actual_tokens


_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


class DailyBudgetExhausted(RuntimeError):
    """The provider's per-day token allowance is spent. Waiting will not help today."""


def _retry_after_seconds(exc: Exception, attempt: int, cap: bool = True) -> float:
    """Seconds to wait before retrying. `cap=False` returns the provider's raw
    estimate, which is how the caller tells a momentary limit from an exhausted one."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            wait = float(retry_after)
            return min(wait, _MAX_BACKOFF_SECONDS) if cap else wait
        except ValueError:
            pass
    # Groq sends no Retry-After via litellm; the wait is in the error text, as
    # "try again in 5.37s" (per-minute) or "7m22.8s" (per-day).
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes = int(match.group(1)) if match.group(1) else 0
        wait = minutes * 60 + float(match.group(2))
    else:
        wait = 2**attempt
    return min(wait, _MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5) if cap else wait


async def _call_one(
    model: str, prompt: str, semaphore: asyncio.Semaphore, window: "_TokenWindow", event: list
) -> dict:
    litellm_model = model.replace(":/", "/")
    async with semaphore:
        for attempt in range(_MAX_RETRIES):
            try:
                # No max_tokens: a reasoning model spends its allowance thinking and
                # would return an empty completion.
                response = await litellm.acompletion(
                    model=litellm_model, messages=[{"role": "user", "content": prompt}]
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    await window.correct(event, usage.total_tokens)
                return {"text": response.choices[0].message.content, "error": None}
            except litellm.exceptions.RateLimitError as exc:
                # The daily window rolls continuously, so a per-day limit often clears
                # in seconds. Give up only when the provider's own estimate says the
                # wait is longer than we're willing to sit in a workshop.
                wait = _retry_after_seconds(exc, attempt, cap=False)
                if wait > _GIVE_UP_AFTER_SECONDS:
                    raise DailyBudgetExhausted(
                        f"{model}: token allowance exhausted for about {wait / 60:.0f} "
                        "more minutes. Re-run with USE_CACHED_RESULTS=1 to continue "
                        "from shipped results, or switch JUDGE_MODEL — the daily "
                        "allowance is per model."
                    ) from exc
                await asyncio.sleep(min(wait, _MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5))
            except (
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.APIConnectionError,
            ):
                await asyncio.sleep(min(2**attempt, _MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5))
        return {"text": None, "error": f"exceeded {_MAX_RETRIES} retries"}


async def _run_batch(prompts: list[str], model: str, tpm: int, max_concurrency: int) -> list[dict]:
    window = _TokenWindow(tpm)
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[dict | None] = [None] * len(prompts)
    completed = 0

    async def worker(i: int, prompt: str) -> None:
        nonlocal completed
        event = await window.acquire(len(_ENC.encode(prompt)) + _EXPECTED_COMPLETION_TOKENS)
        results[i] = await _call_one(model, prompt, semaphore, window, event)
        completed += 1
        if completed % 25 == 0:
            print(f"\r{completed}/{len(prompts)} judged", end="", flush=True)

    await asyncio.gather(*(worker(i, p) for i, p in enumerate(prompts)))
    print()
    return results


def judge_batch(
    prompts: list[str],
    model: str,
    tpm: int,
    max_concurrency: int,
    cache_path: Path | None = None,
) -> list[dict]:
    """Run `prompts` through `model`, paced to `tpm`, cached on disk by (model, prompt).

    Pass `cache_path=None` to force fresh calls, as the noise-floor measurement does.
    Set USE_CACHED_RESULTS=1 to serve verdicts from data/cached_results.parquet and
    make no network calls at all.
    """
    cache = _load_cache(cache_path)
    offline = os.environ.get("USE_CACHED_RESULTS") == "1"
    if offline:
        cache = {**_load_offline_cache(), **cache}

    keys = [_cache_key(model, p) for p in prompts]
    to_run = [(i, p) for i, (p, k) in enumerate(zip(prompts, keys, strict=True)) if k not in cache]

    if to_run and offline:
        raise RuntimeError(
            f"{len(to_run)} prompts have no cached result and USE_CACHED_RESULTS=1 "
            "blocks network calls."
        )

    fresh_by_index: dict[int, dict] = {}
    if to_run:
        indices, run_prompts = zip(*to_run, strict=True)
        fresh = asyncio.run(_run_batch(list(run_prompts), model, tpm, max_concurrency))
        fresh_by_index = dict(zip(indices, fresh, strict=True))
        # Cache successes only, so a transient failure is retried next run.
        for i, result in fresh_by_index.items():
            if result.get("error") is None:
                cache[keys[i]] = result
        _save_cache(cache_path, cache)

    return [cache[keys[i]] if keys[i] in cache else fresh_by_index[i] for i in range(len(prompts))]
