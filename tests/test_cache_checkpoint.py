"""`LLMWrapper(checkpoint=...)` — the host hook that makes the response cache
durable between phases (a Modal adapter passes `volume.commit`).

Offline: the transport is stubbed, so these pin WHEN the hook fires (after a
batch that wrote new entries, never after an all-cache-hit batch), that a
failing hook never fails the run, and that it runs off the event loop.
"""

import json
import threading
import types

from choreo import llm as llm_mod
from choreo.llm import LLMWrapper, run_coro_blocking


class _StubClient:
    async def close(self):
        return None


def _stub_transport(monkeypatch):
    """Every call answers a fixed JSON object; returns the list of sent prompts."""
    monkeypatch.setattr(llm_mod, "make_async_openrouter_client", lambda: _StubClient())

    async def _noop():
        return None

    # cleanup_background_tasks would cancel the pytest task tree.
    monkeypatch.setattr(llm_mod, "cleanup_background_tasks", _noop)
    sent = []

    async def _fake_completion(client, messages, model, **kwargs):
        sent.append(messages)
        message = types.SimpleNamespace(content=json.dumps({"score": 7}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(llm_mod, "async_chat_completion", _fake_completion)
    return sent


def _wrapper(tmp_path, monkeypatch, checkpoint):
    wrapper = LLMWrapper(cache_dir=str(tmp_path), max_retries=0, checkpoint=checkpoint)
    monkeypatch.setattr(wrapper, "_record_usage", lambda *a, **k: None)
    return wrapper


def _phase(wrapper, keys):
    return run_coro_blocking(wrapper.batch_json_complete(
        prompts=[f"prompt {k}" for k in keys], model="fake/llm", cache_keys=keys,
    ))


def test_checkpoint_fires_once_after_a_phase_that_wrote_entries(tmp_path, monkeypatch):
    _stub_transport(monkeypatch)
    calls = []
    wrapper = _wrapper(tmp_path, monkeypatch, lambda: calls.append(wrapper.cache_writes))

    _phase(wrapper, ["a", "b", "c"])

    # Once per phase (not per entry), and only after every entry was on disk.
    assert calls == [3]


def test_no_checkpoint_after_an_all_cache_hit_phase(tmp_path, monkeypatch):
    sent = _stub_transport(monkeypatch)
    calls = []
    wrapper = _wrapper(tmp_path, monkeypatch, lambda: calls.append(1))

    _phase(wrapper, ["a", "b"])
    _phase(wrapper, ["a", "b"])  # the retried run: everything served from cache

    assert len(sent) == 2
    assert calls == [1]


def test_partially_cached_phase_still_checkpoints_its_new_entries(tmp_path, monkeypatch):
    _stub_transport(monkeypatch)
    calls = []
    wrapper = _wrapper(tmp_path, monkeypatch, lambda: calls.append(wrapper.cache_writes))

    _phase(wrapper, ["a"])
    _phase(wrapper, ["a", "b"])

    assert calls == [1, 2]


def test_a_raising_checkpoint_never_fails_the_run(tmp_path, monkeypatch):
    _stub_transport(monkeypatch)

    def _boom():
        raise RuntimeError("volume commit failed")

    wrapper = _wrapper(tmp_path, monkeypatch, _boom)

    results = _phase(wrapper, ["a", "b"])

    assert results == [{"score": 7}, {"score": 7}]
    assert (wrapper.cache_dir / "a.json").exists()


def test_checkpoint_runs_off_the_event_loop_thread(tmp_path, monkeypatch):
    _stub_transport(monkeypatch)
    seen = {}

    async def _completion_thread(*args, **kwargs):
        seen["loop"] = threading.get_ident()
        message = types.SimpleNamespace(content=json.dumps({"score": 7}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(llm_mod, "async_chat_completion", _completion_thread)
    wrapper = _wrapper(tmp_path, monkeypatch, lambda: seen.setdefault("hook", threading.get_ident()))

    _phase(wrapper, ["a"])

    assert seen["hook"] != seen["loop"]


def test_no_cache_dir_means_no_checkpoint(monkeypatch):
    """Nothing is written without a cache dir, so there is nothing to persist."""
    _stub_transport(monkeypatch)
    calls = []
    wrapper = LLMWrapper(cache_dir=None, max_retries=0, checkpoint=lambda: calls.append(1))
    monkeypatch.setattr(wrapper, "_record_usage", lambda *a, **k: None)

    run_coro_blocking(wrapper.batch_json_complete(prompts=["p"], model="fake/llm"))

    assert calls == []
