"""`batch_json_complete(deadline_s=...)` — the wave's wall-clock budget.

Offline: the transport is stubbed with sleeps, so these pin the dispatch
semantics (what lands, what gets cancelled, what the caller sees) without an
API key or real latency.
"""

import asyncio

import numpy as np
import pytest

from choreo import embed as embed_mod
from choreo import llm as llm_mod
from choreo.llm import LLMWrapper, run_coro_blocking
from choreo.query import run_query_match
from conftest import keyword_embed


class _StubClient:
    async def close(self):
        return None


@pytest.fixture(autouse=True)
def _offline_transport(monkeypatch):
    """No real client; `cleanup_background_tasks` would cancel the pytest task
    tree, so stub it out too."""
    monkeypatch.setattr(llm_mod, "make_async_openrouter_client", lambda: _StubClient())

    async def _noop():
        return None

    monkeypatch.setattr(llm_mod, "cleanup_background_tasks", _noop)


def _wrapper_with_latencies(monkeypatch, latencies):
    """An LLMWrapper whose i-th prompt takes `latencies[i]` seconds."""
    wrapper = LLMWrapper(cache_dir=None)

    async def _fake_call(prompt, model, **kwargs):
        delay = latencies[int(prompt)]
        await asyncio.sleep(delay)
        return {"prompt": prompt, "delay": delay}

    monkeypatch.setattr(wrapper, "_async_json_complete_with_retry", _fake_call)
    return wrapper


def test_deadline_keeps_fast_calls_and_cancels_stragglers(monkeypatch):
    """The point of the knob: one slow call must not hold the wave. Fast slots
    carry their result; the straggler's slot comes back None (NOT an
    Exception) so the caller can treat it as "no answer", not "failure"."""
    latencies = [0.01, 0.01, 5.0]
    wrapper = _wrapper_with_latencies(monkeypatch, latencies)

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["0", "1", "2"], model="fake/llm", deadline_s=0.5,
    ))

    assert results[0]["prompt"] == "0"
    assert results[1]["prompt"] == "1"
    assert results[2] is None


def test_deadline_is_wall_clock_not_per_call(monkeypatch):
    """The budget bounds the whole wave, so a batch of uniformly slow calls
    returns at the deadline rather than at the model's tail."""
    wrapper = _wrapper_with_latencies(monkeypatch, [5.0] * 4)

    async def _timed():
        loop = asyncio.get_running_loop()
        start = loop.time()
        out = await wrapper.batch_json_complete(
            prompts=["0", "1", "2", "3"], model="fake/llm", deadline_s=0.3,
        )
        return out, loop.time() - start

    results, elapsed = run_coro_blocking(_timed())

    assert all(r is None for r in results)
    assert elapsed < 2.0        # bounded by the deadline, not the 5s calls


def test_no_deadline_waits_for_every_call(monkeypatch):
    """Default (None) must keep plain-gather semantics — nothing is dropped."""
    wrapper = _wrapper_with_latencies(monkeypatch, [0.01, 0.3, 0.01])

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["0", "1", "2"], model="fake/llm",
    ))

    assert [r["prompt"] for r in results] == ["0", "1", "2"]


def test_results_map_back_to_their_own_slots(monkeypatch):
    """Out-of-order completion under a deadline must not scramble the mapping
    between prompt and result — the caller zips these against its chunks."""
    wrapper = _wrapper_with_latencies(monkeypatch, [0.25, 0.01, 0.15, 5.0])

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["0", "1", "2", "3"], model="fake/llm", deadline_s=1.0,
    ))

    assert [r["prompt"] if r else None for r in results] == ["0", "1", "2", None]


# ── composition: the deadline reaching all the way through run_query_match ────


def test_deadline_stragglers_drop_out_of_the_shortlist(monkeypatch, fake_llm, fake_embed_fn):
    """The whole point of the two changes together, on the real query path with
    the real LLMWrapper: a slow chunk is cancelled at the deadline AND its
    candidates drop out — rather than riding raw embed_norm into the shortlist,
    which is how an unscored candidate used to outrank a scored one."""
    from conftest import build_pool

    monkeypatch.setattr(
        embed_mod, "get_embeddings",
        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]),
    )
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
        "hugo": {"skills": "AGENTS platform", "needs": "DESIGN"},
    }, fake_llm, fake_embed_fn)
    pool_sections = {s.id: s.sections for s in sections}

    wrapper = LLMWrapper(cache_dir=None)
    seen = {"chunks": 0}

    async def _fake_call(prompt, model, **kwargs):
        # chunk width 1 (n_profiles_to_score_together=2) => one call per
        # candidate; stall every chunk after the first past the deadline.
        seen["chunks"] += 1
        if seen["chunks"] > 1:
            await asyncio.sleep(5.0)
        import re
        keys = re.findall(r'"([^"]+)": "0\.\.1"', prompt)
        return {k: 0.9 for k in keys}

    monkeypatch.setattr(wrapper, "_async_json_complete_with_retry", _fake_call)

    config = {
        "models": {"embedding": "fake/embedding-model", "embedding_dimensions": None,
                   "extraction_llm": "fake/llm", "pair_llm": "fake/llm",
                   "reasoning_effort": "low"},
        "instruction_prompt": {"goal": "test goal"},
        "budgets": {"n_profiles_to_score_together": 2},
        "recipe": {"instruction": "score it", "section_weights": {"skills": 1.0},
                   "cross_section_weights": {}},
        "blending": {"embed_weight": 0.35, "llm_weight": 0.65},
        "concurrency": {"max_concurrent_llm_calls": 16},
        "query": {"rerank_pool_multiplier": 4, "rerank_max_retries": 0,
                  "rerank_deadline_s": 0.5},
    }

    result = run_query_match(
        query={"skills": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        top_k=1, generate_intros=False, llm_wrapper=wrapper,
    )

    # Exactly one chunk beat the deadline; the rest were cancelled and their
    # candidates are gone from the shortlist rather than embed-ranked into it.
    assert len(result.shortlist) == 1
    assert result.shortlist[0]["llm_score"] is not None
    assert any("Dropped" in n and "unscored" in n for n in result.notes)
