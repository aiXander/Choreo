"""Transport retry hardening: jittered backoff, Retry-After, AIMD, phase stats.

Why this exists: the first whole-community wintercircus run raises in-flight
LLM calls from 48 to 128. The old retry (1 s, 2 s, 4 s, unjittered, no
Retry-After) made a 429 burst retry in lockstep and give up after ~7 s, and a
give-up degrades SILENTLY downstream (embed-only pair ranks, generic intro
cards). These pin the new policy — and that the deadline-bounded live query
re-rank keeps the old one exactly.

Offline and fast: the transport is stubbed, backoff sleeps are recorded and
skipped, the jitter RNG is stubbed where a test needs an exact value.
"""

import asyncio
import json
import random
import types

import httpx
import openai
import pytest

from choreo import llm as llm_mod
from choreo.llm import (
    ADAPTIVE_INCREASE_EVERY,
    BATCH_MAX_RETRIES,
    LEGACY_MAX_RETRIES,
    RETRY_DELAY_CAP_S,
    AdaptiveConcurrencyLimiter,
    JSONExtractionError,
    LLMWrapper,
    compute_retry_delay,
    parse_retry_after,
    run_coro_blocking,
)

_REAL_SLEEP = asyncio.sleep
_TICKER = 3600.0  # progress_interval used in these tests; never a backoff value


# ── fixtures / helpers ───────────────────────────────────────────────────────


class _StubClient:
    async def close(self):
        return None


class _EdgeRng:
    """`uniform(lo, hi)` returns one edge, so a delay is exact, not sampled."""

    def __init__(self, edge="hi"):
        self.edge = edge
        self.calls = []

    def uniform(self, lo, hi):
        self.calls.append((lo, hi))
        return hi if self.edge == "hi" else lo


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_async_openrouter_client", lambda: _StubClient())

    async def _noop():
        return None

    # It would cancel pytest's own task tree.
    monkeypatch.setattr(llm_mod, "cleanup_background_tasks", _noop)


@pytest.fixture
def sleeps(monkeypatch):
    """Record every backoff sleep and skip it (the ticker's long sleep stays real
    — it is cancelled when the batch ends)."""
    recorded = []

    async def _fake_sleep(delay, *args, **kwargs):
        if delay == _TICKER:
            return await _REAL_SLEEP(delay)
        recorded.append(delay)
        return await _REAL_SLEEP(0)

    monkeypatch.setattr(llm_mod.asyncio, "sleep", _fake_sleep)
    return recorded


def _status_error(status, headers=None):
    """A real openai SDK status error, carrying real httpx response headers."""
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    cls = {
        400: openai.BadRequestError,
        429: openai.RateLimitError,
        503: openai.InternalServerError,
    }[status]
    return cls(f"Error code: {status}", response=response, body=None)


def _ok(payload=None):
    message = types.SimpleNamespace(content=json.dumps(payload or {"ok": True}))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)


def _transport(monkeypatch, script, latency=0.0):
    """Stub `async_chat_completion`. `script(prompt_text, attempt_no)` returns a
    response or raises. Tracks live concurrency."""
    state = {"in_flight": 0, "peak": 0, "attempts": {}, "total": 0}

    async def _fake(client, messages, model, **kwargs):
        prompt = messages[0]["content"].split("\n\n")[0]
        n = state["attempts"].get(prompt, 0)
        state["attempts"][prompt] = n + 1
        state["total"] += 1
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await _REAL_SLEEP(latency)
            return script(prompt, n)
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(llm_mod, "async_chat_completion", _fake)
    return state


def _wrapper(monkeypatch, cache_dir=None, **kwargs):
    wrapper = LLMWrapper(cache_dir=cache_dir, **kwargs)
    monkeypatch.setattr(wrapper, "_record_usage", lambda *a, **k: None)
    return wrapper


def _batch(wrapper, prompts, **kwargs):
    kwargs.setdefault("progress_interval", _TICKER)
    return run_coro_blocking(wrapper.batch_json_complete(
        prompts=prompts, model="fake/llm", **kwargs,
    ))


# ── compute_retry_delay ──────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["rate_limit", "transient"])
def test_full_jitter_stays_within_the_exponential_window_and_the_cap(kind):
    rng = random.Random(1234)
    for attempt in range(12):
        ceiling = min(RETRY_DELAY_CAP_S, 1.0 * 2 ** attempt)
        for _ in range(50):
            d = compute_retry_delay(attempt, kind, base=1.0, rng=rng)
            assert 0.0 <= d <= ceiling


def test_jitter_spreads_a_burst_instead_of_lockstep():
    """The thundering-herd fix itself: 128 calls failing together must not all
    come back at the same instant."""
    rng = random.Random(7)
    delays = {round(compute_retry_delay(3, "rate_limit", base=1.0, rng=rng), 3)
              for _ in range(128)}
    assert len(delays) > 100


def test_window_is_capped():
    d = compute_retry_delay(20, "rate_limit", base=1.0, rng=_EdgeRng("hi"))
    assert d == RETRY_DELAY_CAP_S


def test_malformed_sample_gets_a_short_pause_that_does_not_grow():
    """A bad sample isn't a rate limit: re-sampling is the fix, so attempt 5
    waits no longer than attempt 0."""
    for attempt in (0, 5, 20):
        assert compute_retry_delay(attempt, "sample", base=1.0, rng=_EdgeRng("hi")) == 1.0


def test_retry_after_is_a_floor_plus_jitter_and_clamped():
    assert compute_retry_delay(0, "rate_limit", base=1.0, retry_after=5.0,
                               rng=_EdgeRng("lo")) == 5.0
    # attempt 2 → spread 4 s on top of the server's 5 s
    assert compute_retry_delay(2, "rate_limit", base=1.0, retry_after=5.0,
                               rng=_EdgeRng("hi")) == 9.0
    # a server asking for an hour gets the cap
    assert compute_retry_delay(0, "rate_limit", base=1.0, retry_after=3600.0,
                               rng=_EdgeRng("lo")) == RETRY_DELAY_CAP_S


# ── parse_retry_after ────────────────────────────────────────────────────────


def test_retry_after_seconds_and_ms_from_a_real_sdk_error():
    assert parse_retry_after(_status_error(429, {"retry-after": "7"})) == 7.0
    assert parse_retry_after(_status_error(429, {"retry-after": "2.5"})) == 2.5
    assert parse_retry_after(_status_error(429, {"retry-after-ms": "1500"})) == 1.5
    # the more precise header wins when both are present
    assert parse_retry_after(_status_error(
        429, {"retry-after-ms": "250", "retry-after": "9"})) == 0.25


def test_retry_after_http_date():
    err = _status_error(503, {"retry-after": "Wed, 21 Oct 2026 07:28:10 GMT"})
    now = 1792567680.0  # 2026-10-21 07:28:00 UTC
    assert parse_retry_after(err, now=now) == pytest.approx(10.0)
    # a date in the past means "now", not a negative sleep
    assert parse_retry_after(err, now=now + 100) == 0.0


@pytest.mark.parametrize("headers", [
    {},
    {"retry-after": "soon"},
    {"retry-after": "-5"},
    {"retry-after": "nan"},
    {"retry-after": "inf"},
    {"retry-after": "9" * 500},             # huge
    {"retry-after-ms": "lots"},             # garbage ms, no fallback present
    {"retry-after": "Mon, 99 Foo 2026 99:99:99"},
])
def test_garbage_retry_after_is_no_advice(headers):
    assert parse_retry_after(_status_error(429, headers)) is None


def test_garbage_ms_falls_back_to_seconds():
    assert parse_retry_after(_status_error(
        429, {"retry-after-ms": "lots", "retry-after": "3"})) == 3.0


def test_errors_without_usable_headers_are_no_advice():
    class _Exploding:
        def get(self, name):
            raise RuntimeError("boom")

    assert parse_retry_after(ValueError("plain")) is None
    assert parse_retry_after(types.SimpleNamespace(response=None)) is None
    assert parse_retry_after(types.SimpleNamespace(
        response=types.SimpleNamespace(headers=_Exploding()))) is None
    # a plain dict with odd casing still works
    assert parse_retry_after(types.SimpleNamespace(
        response=types.SimpleNamespace(headers={"Retry-After": "4"}))) == 4.0


# ── the retry loop, end to end through batch_json_complete ───────────────────


def test_429_honors_retry_after_then_succeeds(monkeypatch, sleeps):
    monkeypatch.setattr(llm_mod, "_RETRY_RNG", _EdgeRng("lo"))
    _transport(monkeypatch, lambda p, n: (_ for _ in ()).throw(
        _status_error(429, {"retry-after": "2"})) if n == 0 else _ok())
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, ["a"], progress_label="score")

    assert results == [{"ok": True}]
    assert sleeps == [2.0]
    stats = wrapper.get_phase_stats()[0]
    assert (stats["dispatched"], stats["retries"], stats["rate_limited"], stats["failed"]) == (1, 1, 1, 0)


def test_non_retryable_4xx_fails_fast(monkeypatch, sleeps):
    state = _transport(monkeypatch, lambda p, n: (_ for _ in ()).throw(_status_error(400)))
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, ["a"])

    assert isinstance(results[0], openai.BadRequestError)
    assert state["total"] == 1 and sleeps == []
    assert wrapper.get_stats()["failed"] == 1
    assert wrapper.get_stats()["retries"] == 0


def _always_429(p, n):
    raise _status_error(429)


def test_unbounded_phase_defaults_to_the_batch_retry_budget(monkeypatch, sleeps):
    state = _transport(monkeypatch, _always_429)
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, ["a"])

    assert isinstance(results[0], openai.RateLimitError)
    assert state["total"] == BATCH_MAX_RETRIES + 1
    assert all(0.0 <= d <= RETRY_DELAY_CAP_S for d in sleeps)
    stats = wrapper.get_stats()
    assert stats["rate_limited"] == BATCH_MAX_RETRIES + 1  # the final 429 counts too
    assert stats["retries"] == BATCH_MAX_RETRIES


def test_wrapper_max_retries_sets_the_unbounded_default(monkeypatch, sleeps):
    state = _transport(monkeypatch, _always_429)
    _batch(_wrapper(monkeypatch, max_retries=1), ["a"])
    assert state["total"] == 2


@pytest.mark.parametrize("deadline_s", [None, 60.0])
def test_explicit_max_retries_is_exact_on_both_paths(monkeypatch, sleeps, deadline_s):
    state = _transport(monkeypatch, _always_429)
    _batch(_wrapper(monkeypatch), ["a"], max_retries=2, deadline_s=deadline_s)
    assert state["total"] == 3


def test_json_failures_retry_with_short_pauses_and_never_touch_the_limiter(monkeypatch, sleeps):
    """A malformed sample whose text says '429 rate limit' is still a sample
    failure — it must not be counted as (or cut concurrency like) a 429."""
    monkeypatch.setattr(llm_mod, "_RETRY_RNG", _EdgeRng("hi"))

    def _script(p, n):
        if n < 3:
            message = types.SimpleNamespace(content="429 rate limit, no json here")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)
        return _ok()

    _transport(monkeypatch, _script)
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, ["a"], max_concurrent=16)

    assert results == [{"ok": True}]
    assert sleeps == [1.0, 1.0, 1.0]  # ≤ base, not 1/2/4
    stats = wrapper.get_phase_stats()[0]
    assert stats["rate_limited"] == 0 and stats["retries"] == 3
    assert stats["min_concurrency"] == 16


def test_json_extraction_error_stays_retryable():
    wrapper = LLMWrapper(cache_dir=None)
    assert wrapper._retry_kind(JSONExtractionError("bad 429")) == "sample"
    assert wrapper._retry_kind(_status_error(429)) == "rate_limit"
    assert wrapper._retry_kind(_status_error(503)) == "transient"
    assert wrapper._retry_kind(_status_error(400)) is None


# ── the deadline-bounded live query path keeps the legacy policy ─────────────


def test_deadline_path_keeps_legacy_retries_and_unjittered_sleeps(monkeypatch, sleeps):
    """`find_matches` re-rank passes no max_retries and a deadline. Before this
    change it got 3 retries at 1 s, 2 s, 4 s — and still must, even when the
    server sends Retry-After (the deadline is that path's retry budget)."""
    state = _transport(monkeypatch, lambda p, n: (_ for _ in ()).throw(
        _status_error(429, {"retry-after": "20"})))
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, ["a"], deadline_s=60.0, progress_label="query_rerank")

    assert isinstance(results[0], openai.RateLimitError)
    assert state["total"] == LEGACY_MAX_RETRIES + 1
    assert sleeps == [1.0, 2.0, 4.0]


def test_deadline_path_concurrency_is_fixed_not_adaptive(monkeypatch, sleeps):
    _transport(monkeypatch, lambda p, n: (_ for _ in ()).throw(_status_error(429))
               if n == 0 else _ok(), latency=0.001)
    wrapper = _wrapper(monkeypatch)

    _batch(wrapper, [str(i) for i in range(12)], max_concurrent=10, deadline_s=60.0)

    assert wrapper.get_phase_stats()[0]["min_concurrency"] == 10  # never cut


# ── AdaptiveConcurrencyLimiter ───────────────────────────────────────────────


def test_limiter_halves_on_rate_limit_down_to_the_floor():
    lim = AdaptiveConcurrencyLimiter(128, floor=8)
    seen = []
    while lim.on_rate_limit():
        seen.append(lim.limit)
    assert seen == [64, 32, 16, 8]
    assert lim.limit == lim.min_limit == 8


def test_limiter_floor_never_exceeds_a_small_ceiling():
    lim = AdaptiveConcurrencyLimiter(4, floor=8)
    assert lim.floor == 4
    assert not lim.on_rate_limit() and lim.limit == 4


def test_one_burst_cuts_once():
    """128 calls dispatched under the same limit all 429 together: one cut."""
    lim = AdaptiveConcurrencyLimiter(128)
    burst_epoch = lim.epoch
    cuts = sum(lim.on_rate_limit(burst_epoch) for _ in range(128))
    assert cuts == 1 and lim.limit == 64
    # a request sent AFTER the cut can cut again
    assert lim.on_rate_limit(lim.epoch) and lim.limit == 32


def test_limiter_recovers_additively_and_never_exceeds_the_ceiling():
    lim = AdaptiveConcurrencyLimiter(16, floor=2)
    lim.on_rate_limit()
    assert lim.limit == 8
    for _ in range(ADAPTIVE_INCREASE_EVERY - 1):
        lim.on_success()
    assert lim.limit == 8
    lim.on_success()
    assert lim.limit == 9
    for _ in range(ADAPTIVE_INCREASE_EVERY * 100):
        lim.on_success()
    assert lim.limit == 16
    assert lim.min_limit == 8


def test_a_rate_limit_resets_the_success_streak():
    lim = AdaptiveConcurrencyLimiter(16, floor=1)
    lim.on_rate_limit()  # 8
    for _ in range(ADAPTIVE_INCREASE_EVERY - 1):
        lim.on_success()
    lim.on_rate_limit(epoch=-1)  # stale: no cut, but the streak is broken
    lim.on_success()
    assert lim.limit == 8


def test_shrinking_withholds_permits_until_in_flight_drops_below_the_limit():
    async def _run():
        lim = AdaptiveConcurrencyLimiter(4, floor=1)
        for _ in range(4):
            await lim.acquire()
        lim.on_rate_limit()  # 4 → 2 with 4 in flight
        waiter = asyncio.create_task(lim.acquire())
        await _REAL_SLEEP(0)
        lim.release()  # 3 in flight
        await _REAL_SLEEP(0)
        assert not waiter.done()
        lim.release()  # 2
        await _REAL_SLEEP(0)
        assert not waiter.done()
        lim.release()  # 1 → the waiter gets one
        await _REAL_SLEEP(0)
        assert waiter.done() and lim.in_flight == 2
    asyncio.run(_run())


def test_no_permit_leaks_when_a_queued_waiter_is_cancelled():
    async def _run():
        lim = AdaptiveConcurrencyLimiter(1, adaptive=False)
        await lim.acquire()
        waiter = asyncio.create_task(lim.acquire())
        await _REAL_SLEEP(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        lim.release()
        assert lim.in_flight == 0 and not lim._waiters
    asyncio.run(_run())


def test_no_permit_leaks_when_a_waiter_is_cancelled_after_being_handed_one():
    """The race a naive design leaks on: release() hands B the permit, then B is
    cancelled before it ever runs."""
    async def _run():
        lim = AdaptiveConcurrencyLimiter(1, adaptive=False)
        await lim.acquire()
        waiter = asyncio.create_task(lim.acquire())
        await _REAL_SLEEP(0)
        lim.release()          # handed to the waiter (in_flight stays 1)
        assert lim.in_flight == 1
        waiter.cancel()        # ...which is cancelled before resuming
        await asyncio.gather(waiter, return_exceptions=True)
        assert lim.in_flight == 0 and not lim._waiters
    asyncio.run(_run())


def test_retries_jump_the_queue():
    async def _run():
        lim = AdaptiveConcurrencyLimiter(1, adaptive=False)
        await lim.acquire()
        order = []

        async def _take(tag, priority):
            await lim.acquire(priority)
            order.append(tag)
            lim.release()

        fresh = asyncio.create_task(_take("fresh", False))
        await _REAL_SLEEP(0)
        retry = asyncio.create_task(_take("retry", True))
        await _REAL_SLEEP(0)
        lim.release()
        await asyncio.gather(fresh, retry)
        assert order == ["retry", "fresh"]
    asyncio.run(_run())


def test_over_release_is_an_error():
    with pytest.raises(ValueError):
        AdaptiveConcurrencyLimiter(2).release()


# ── AIMD inside batch_json_complete ──────────────────────────────────────────


def test_a_429_burst_shrinks_the_limit_once_and_every_prompt_still_lands(monkeypatch, sleeps):
    """Ceiling 32: the first 32 requests all fly at once and all 429 → ONE cut
    to 16 (not five cuts to the floor); their retries and the remaining prompts
    succeed, and real concurrency never exceeds the ceiling."""
    burst = 32
    calls = {"n": 0}

    def _script(p, n):
        calls["n"] += 1
        if calls["n"] <= burst:
            raise _status_error(429)
        return _ok({"p": p})

    state = _transport(monkeypatch, _script, latency=0.002)
    wrapper = _wrapper(monkeypatch)
    prompts = [f"q{i}" for i in range(120)]

    results = _batch(wrapper, prompts, max_concurrent=burst, progress_label="score")

    assert [r["p"] for r in results] == prompts
    assert state["peak"] <= burst
    stats = wrapper.get_phase_stats()[0]
    assert stats["min_concurrency"] == 16
    assert stats["rate_limited"] == burst and stats["retries"] == burst
    assert stats["failed"] == 0 and stats["dispatched"] == 120


def test_after_a_cut_real_concurrency_respects_the_lower_limit(monkeypatch, sleeps):
    """Once in-flight work drains below the cut limit, no new request starts
    above it (+1 per ADAPTIVE_INCREASE_EVERY successes of regrowth)."""
    state = {"in_flight": 0, "cut": False, "settled": False, "peak_after": 0, "ok": 0}

    async def _fake(client, messages, model, **kwargs):
        prompt = messages[0]["content"].split("\n\n")[0]
        if state["cut"] and state["in_flight"] < 8:
            state["settled"] = True
        state["in_flight"] += 1
        if state["settled"]:
            state["peak_after"] = max(state["peak_after"], state["in_flight"])
        try:
            await _REAL_SLEEP(0.002)
            if prompt == "q0" and not state["cut"]:
                state["cut"] = True
                raise _status_error(429)
            state["ok"] += 1
            return _ok()
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(llm_mod, "async_chat_completion", _fake)
    wrapper = _wrapper(monkeypatch)

    _batch(wrapper, [f"q{i}" for i in range(60)], max_concurrent=16)

    assert wrapper.get_phase_stats()[0]["min_concurrency"] == 8
    assert state["settled"]
    assert state["peak_after"] <= 8 + state["ok"] // ADAPTIVE_INCREASE_EVERY
    assert state["peak_after"] < 16


def test_deadline_cancellation_leaves_no_permit_behind(monkeypatch, sleeps):
    """Stragglers cancelled at the deadline — mid-request AND still queued —
    must all give their permits back."""
    made = []

    class _Recording(AdaptiveConcurrencyLimiter):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(llm_mod, "AdaptiveConcurrencyLimiter", _Recording)

    async def _hang(client, messages, model, **kwargs):
        await _REAL_SLEEP(5.0)
        return _ok()

    monkeypatch.setattr(llm_mod, "async_chat_completion", _hang)
    wrapper = _wrapper(monkeypatch)

    results = _batch(wrapper, [str(i) for i in range(6)], max_concurrent=2, deadline_s=0.1)

    assert results == [None] * 6
    (limiter,) = made
    assert limiter.in_flight == 0
    assert not [f for f in limiter._waiters if not f.done()]
    assert wrapper.get_phase_stats()[0]["cancelled"] == 6


# ── stats ────────────────────────────────────────────────────────────────────


def test_phase_stats_accumulate_and_print_one_summary_line(monkeypatch, sleeps, tmp_path, capsys):
    def _script(p, n):
        if p == "bad":
            raise _status_error(400)
        if p == "flaky" and n == 0:
            raise _status_error(503)
        return _ok({"p": p})

    _transport(monkeypatch, _script)
    wrapper = _wrapper(monkeypatch, cache_dir=str(tmp_path))
    prompts = ["a", "b", "flaky", "bad"]
    keys = ["ka", "kb", "kf", "kbad"]

    _batch(wrapper, prompts, cache_keys=keys, progress_label="score", max_concurrent=8)
    _batch(wrapper, prompts, cache_keys=keys, progress_label="introduce", max_concurrent=8)

    first, second = wrapper.get_phase_stats()
    assert first["label"] == "score"
    assert (first["prompts"], first["cache_hits"], first["dispatched"],
            first["retries"], first["rate_limited"], first["failed"]) == (4, 0, 4, 1, 0, 1)
    # second phase: three answers replay from cache, only the failure re-calls
    assert (second["cache_hits"], second["dispatched"], second["failed"]) == (3, 1, 1)

    totals = wrapper.get_stats()
    assert totals["phases"] == 2
    assert totals["prompts"] == 8 and totals["cache_hits"] == 3
    assert totals["dispatched"] == 5 and totals["failed"] == 2
    assert totals["retries"] == 1 and totals["min_concurrency"] == 8
    assert totals["total_calls"] == 3  # successful API calls, as before

    out = capsys.readouterr().out
    assert out.count("[score] LLM phase:") == 1
    assert out.count("[introduce] LLM phase:") == 1


def test_fully_cached_phase_still_records_stats(monkeypatch, sleeps, tmp_path):
    _transport(monkeypatch, lambda p, n: _ok())
    wrapper = _wrapper(monkeypatch, cache_dir=str(tmp_path))
    _batch(wrapper, ["a"], cache_keys=["k"])
    _batch(wrapper, ["a"], cache_keys=["k"])
    assert wrapper.get_phase_stats()[1]["cache_hits"] == 1
    assert wrapper.get_phase_stats()[1]["dispatched"] == 0


def test_get_stats_before_any_phase():
    stats = LLMWrapper(cache_dir=None).get_stats()
    assert stats["total_calls"] == 0 and stats["phases"] == 0
    assert stats["min_concurrency"] is None
