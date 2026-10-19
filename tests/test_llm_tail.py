"""Tail latency of a batch phase: hedging, the tail cutoff, the attempt cap.

Why this exists: on 2026-10-19 a 1,580-member community rehearsal sat at
intro 3118/3119 for ~7 minutes on ONE request — the SDK's 120 s "timeout" is
httpx's per-read timeout and never fired — so a 2-minute phase took ten.
These pin the three fixes: a wall-clock cap per HTTP attempt, a second
(hedged) request for the last ~1% of a phase, and a cutoff after which an
unanswered slot comes back `None` instead of holding the run.

Offline: `async_chat_completion` is stubbed with sleeps; the real retry path
(`_async_json_complete_with_retry`) runs.
"""

import asyncio
import json
import types

import httpx
import openai
import pytest

from choreo import llm as llm_mod
from choreo.llm import LLMWrapper, run_coro_blocking

_REAL_SLEEP = asyncio.sleep
_TICKER = 3600.0


class _StubClient:
    async def close(self):
        return None


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(llm_mod, "make_async_openrouter_client", lambda: _StubClient())

    async def _noop():
        return None

    monkeypatch.setattr(llm_mod, "cleanup_background_tasks", _noop)


def _ok(payload):
    message = types.SimpleNamespace(content=json.dumps(payload))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)


def _bad_request():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError("Error code: 400", response=response, body=None)


def _transport(monkeypatch, behave):
    """Stub the request. `behave(prompt, attempt_no)` returns (delay_s, outcome):
    outcome None → a valid answer, an Exception → raised after the delay.
    `attempt_no` counts every request for that prompt — retries AND hedges."""
    state = {"attempts": {}, "total": 0, "in_flight": 0}

    async def _fake(client, messages, model, **kwargs):
        prompt = messages[0]["content"].split("\n\n")[0]
        n = state["attempts"].get(prompt, 0)
        state["attempts"][prompt] = n + 1
        state["total"] += 1
        state["in_flight"] += 1
        try:
            delay, outcome = behave(prompt, n)
            await _REAL_SLEEP(delay)
            if isinstance(outcome, Exception):
                raise outcome
            return _ok({"p": prompt, "attempt": n})
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(llm_mod, "async_chat_completion", _fake)
    return state


def _wrapper(monkeypatch):
    wrapper = LLMWrapper(cache_dir=None, max_concurrent_llm_calls=128)
    monkeypatch.setattr(wrapper, "_record_usage", lambda *a, **k: None)
    return wrapper


def _timed_batch(wrapper, prompts, **kwargs):
    kwargs.setdefault("progress_interval", _TICKER)
    kwargs.setdefault("retry_delay_base", 0.01)

    async def _run():
        loop = asyncio.get_running_loop()
        start = loop.time()
        out = await wrapper.batch_json_complete(prompts=prompts, model="fake/llm", **kwargs)
        return out, loop.time() - start

    return run_coro_blocking(_run())


PROMPTS = [f"p{i}" for i in range(200)]


def test_one_stuck_request_is_hedged_instead_of_holding_the_phase(monkeypatch):
    """The incident shape: 199 answers land, one request hangs. Its hedge
    answers, the hung original is cancelled, and the phase ends in seconds."""
    state = _transport(monkeypatch, lambda p, n: (20.0, None) if (p, n) == ("p7", 0) else (0.0, None))
    wrapper = _wrapper(monkeypatch)
    wrapper.tail_grace_s = 10.0

    results, elapsed = _timed_batch(wrapper, PROMPTS)

    assert elapsed < 5.0
    assert results[7] == {"p": "p7", "attempt": 1}   # the hedge's answer
    assert all(r["p"] == f"p{i}" for i, r in enumerate(results))
    phase = wrapper.get_phase_stats()[0]
    assert phase["hedged"] >= 1
    assert phase["cancelled"] == 0 and phase["failed"] == 0
    assert state["in_flight"] == 0                    # the loser did not keep running


def test_tail_cutoff_returns_none_for_a_straggler_that_never_answers(monkeypatch):
    """Original and hedge both hang: after the grace the slot is `None` (no
    answer, the caller's fallback) — not a phase stuck until the class timeout."""
    _transport(monkeypatch, lambda p, n: (20.0, None) if p == "p7" else (0.0, None))
    wrapper = _wrapper(monkeypatch)
    wrapper.tail_grace_s = 0.3

    results, elapsed = _timed_batch(wrapper, PROMPTS)

    assert elapsed < 5.0
    assert results[7] is None
    assert sum(r is None for r in results) == 1
    assert wrapper.get_phase_stats()[0]["cancelled"] == 1


def test_a_failed_original_waits_for_its_hedge(monkeypatch):
    """An Exception settles a slot only when no twin is left: the original
    fails fast, the hedge answers a moment later, and the answer wins."""
    def behave(p, n):
        if p != "p7":
            return 0.0, None
        return (0.3, _bad_request()) if n == 0 else (0.6, None)

    _transport(monkeypatch, behave)
    wrapper = _wrapper(monkeypatch)

    results, _ = _timed_batch(wrapper, PROMPTS)

    assert results[7] == {"p": "p7", "attempt": 1}
    assert wrapper.get_phase_stats()[0]["failed"] == 0


def test_a_small_phase_never_hedges(monkeypatch):
    """1% of 50 prompts rounds to zero: one request per prompt, as before."""
    state = _transport(monkeypatch, lambda p, n: (0.3, None) if p == "p7" else (0.0, None))
    wrapper = _wrapper(monkeypatch)

    results, _ = _timed_batch(wrapper, PROMPTS[:50])

    assert state["total"] == 50
    assert results[7]["p"] == "p7"
    assert wrapper.get_phase_stats()[0]["hedged"] == 0


def test_a_deadline_bounded_phase_never_hedges(monkeypatch):
    """The live re-rank keeps its own budget: no second requests there."""
    state = _transport(monkeypatch, lambda p, n: (0.5, None) if p == "p7" else (0.0, None))
    wrapper = _wrapper(monkeypatch)

    results, _ = _timed_batch(wrapper, PROMPTS, deadline_s=5.0)

    assert state["total"] == 200
    assert results[7]["p"] == "p7"


def test_attempt_wall_clock_cap_retries_a_hung_request(monkeypatch):
    """A request that never returns is cut at `attempt_timeout_s` and retried
    — the SDK timeout alone did not bound it."""
    state = _transport(monkeypatch, lambda p, n: (5.0, None) if n == 0 else (0.0, None))
    wrapper = _wrapper(monkeypatch)
    wrapper.attempt_timeout_s = 0.2

    results, elapsed = _timed_batch(wrapper, ["solo"])

    assert elapsed < 2.0
    assert results[0] == {"p": "solo", "attempt": 1}
    assert state["attempts"]["solo"] == 2
    assert wrapper.get_phase_stats()[0]["retries"] == 1
