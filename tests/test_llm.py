"""llm.py contract tests. All model calls are mocked; nothing here touches the network."""

import asyncio

import litellm
import pandas as pd
import pytest

from evals_workshop import llm


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, content, total_tokens=None):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(total_tokens) if total_tokens is not None else None


def test_judge_batch_caches_successful_calls(tmp_path, monkeypatch):
    call_count = 0

    async def fake_acompletion(model, messages):
        nonlocal call_count
        call_count += 1
        return _FakeResponse("Reason: ok.\nVerdict: no")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    cache_path = tmp_path / "cache.json"

    llm.judge_batch(
        ["prompt a", "prompt b"], "groq:/dummy", tpm=8000, max_concurrency=2, cache_path=cache_path
    )
    assert call_count == 2

    llm.judge_batch(
        ["prompt a", "prompt b"], "groq:/dummy", tpm=8000, max_concurrency=2, cache_path=cache_path
    )
    assert call_count == 2  # second call served entirely from cache


def test_judge_batch_does_not_permanently_cache_failures(tmp_path, monkeypatch):
    call_count = 0

    async def always_fails(model, messages):
        nonlocal call_count
        call_count += 1
        raise litellm.exceptions.APIConnectionError(
            message="boom", llm_provider="groq", model="dummy"
        )

    monkeypatch.setattr(litellm, "acompletion", always_fails)
    monkeypatch.setattr(llm, "_MAX_RETRIES", 1)
    cache_path = tmp_path / "cache.json"

    result1 = llm.judge_batch(
        ["prompt a"], "groq:/dummy", tpm=8000, max_concurrency=1, cache_path=cache_path
    )
    assert result1[0]["error"] is not None
    calls_after_first = call_count

    result2 = llm.judge_batch(
        ["prompt a"], "groq:/dummy", tpm=8000, max_concurrency=1, cache_path=cache_path
    )
    assert result2[0]["error"] is not None
    assert call_count > calls_after_first  # retried, not served from a cached failure


def test_retry_after_parsed_from_groq_message():
    exc = litellm.exceptions.RateLimitError(
        message="Rate limit reached. Please try again in 5.37s.",
        llm_provider="groq",
        model="dummy",
    )
    assert llm._retry_after_seconds(exc, attempt=0) == pytest.approx(5.37, abs=0.5)


def test_retry_after_parses_minutes_and_caps_the_wait():
    exc = litellm.exceptions.RateLimitError(
        message="Rate limit reached. Please try again in 7m22.8s.",
        llm_provider="groq",
        model="dummy",
    )
    # 442.8s parsed, then capped so a cell can't hang on a single hint
    assert llm._retry_after_seconds(exc, attempt=0) <= llm._MAX_BACKOFF_SECONDS + 0.5


def test_daily_limit_raises_when_the_wait_is_long(monkeypatch):
    """A long wait won't clear inside a session, so stop rather than burn retries."""
    calls = 0

    async def daily_limit(model, messages):
        nonlocal calls
        calls += 1
        raise litellm.exceptions.RateLimitError(
            message="tokens per day (TPD): Limit 200000. Please try again in 7m22.8s.",
            llm_provider="groq",
            model="dummy",
        )

    monkeypatch.setattr(litellm, "acompletion", daily_limit)

    async def run():
        window = llm._TokenWindow(tpm=10_000)
        event = await window.acquire(100)
        await llm._call_one("groq:/dummy", "p", asyncio.Semaphore(1), window, event)

    with pytest.raises(llm.DailyBudgetExhausted):
        asyncio.run(run())
    assert calls == 1


def test_daily_limit_retries_when_the_wait_is_short(monkeypatch):
    """The daily window rolls continuously, so a few seconds' wait is recoverable."""
    calls = 0

    async def briefly_limited(model, messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise litellm.exceptions.RateLimitError(
                message="tokens per day (TPD): Limit 200000. Please try again in 0.01s.",
                llm_provider="groq",
                model="dummy",
            )
        return _FakeResponse("Reason: ok.\nVerdict: no")

    monkeypatch.setattr(litellm, "acompletion", briefly_limited)

    async def run():
        window = llm._TokenWindow(tpm=10_000)
        event = await window.acquire(100)
        return await llm._call_one("groq:/dummy", "p", asyncio.Semaphore(1), window, event)

    result = asyncio.run(run())
    assert result["error"] is None
    assert calls == 2


def test_call_one_corrects_window_with_actual_usage(monkeypatch):
    """The up-front reservation is a guess; reported usage replaces it."""

    async def fake_acompletion(model, messages):
        return _FakeResponse("Reason: ok.\nVerdict: no", total_tokens=999)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def run():
        window = llm._TokenWindow(tpm=10_000)
        event = await window.acquire(150)  # a low guess
        semaphore = asyncio.Semaphore(1)
        await llm._call_one("groq:/dummy", "prompt", semaphore, window, event)
        return event

    event = asyncio.run(run())
    assert event[1] == 999


def test_token_window_releases_lock_while_waiting(monkeypatch):
    """Holding the lock while sleeping would serialize every other worker."""
    window = llm._TokenWindow(tpm=10)
    lock_states = []

    class _StopAfterFirstSleep(Exception):
        pass

    async def fake_sleep(seconds):
        lock_states.append(window._lock.locked())
        raise _StopAfterFirstSleep

    async def run():
        await window.acquire(10)  # fills the window; no sleep needed
        monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
        await window.acquire(5)  # must wait; triggers fake_sleep

    with pytest.raises(_StopAfterFirstSleep):
        asyncio.run(run())
    assert lock_states == [False]


def test_judge_batch_serves_offline_cache_without_calling_model(tmp_path, monkeypatch):
    call_count = 0

    async def fake_acompletion(model, messages):
        nonlocal call_count
        call_count += 1
        return _FakeResponse("Reason: ok.\nVerdict: no")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    key = llm._cache_key("groq:/dummy", "prompt a")
    offline_path = tmp_path / "cached_results.parquet"
    offline_df = pd.DataFrame(
        {"cache_key": [key], "text": ["Reason: ok.\nVerdict: no"], "error": [None]}
    )
    offline_df.to_parquet(offline_path)
    monkeypatch.setattr(llm, "_OFFLINE_CACHE_PATH", offline_path)
    monkeypatch.setenv("USE_CACHED_RESULTS", "1")

    result = llm.judge_batch(["prompt a"], "groq:/dummy", tpm=8000, max_concurrency=1)

    assert result[0]["text"] == "Reason: ok.\nVerdict: no"
    assert call_count == 0


def test_judge_batch_offline_raises_on_missing_prompt(tmp_path, monkeypatch):
    call_count = 0

    async def fake_acompletion(model, messages):
        nonlocal call_count
        call_count += 1
        return _FakeResponse("Reason: ok.\nVerdict: no")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    offline_path = tmp_path / "cached_results.parquet"
    pd.DataFrame({"cache_key": [], "text": [], "error": []}).to_parquet(offline_path)
    monkeypatch.setattr(llm, "_OFFLINE_CACHE_PATH", offline_path)
    monkeypatch.setenv("USE_CACHED_RESULTS", "1")

    with pytest.raises(RuntimeError):
        llm.judge_batch(["prompt a"], "groq:/dummy", tpm=8000, max_concurrency=1)

    assert call_count == 0


def test_retry_after_falls_back_to_exponential_backoff():
    exc = litellm.exceptions.RateLimitError(
        message="no timing info here", llm_provider="groq", model="dummy"
    )
    seconds = llm._retry_after_seconds(exc, attempt=2)
    assert 4 <= seconds <= 5
