"""Every Choreo LLM call routes through exactly ONE mechanism.

A bare slug (standalone Choreo, no host OpenRouter account) carries the
`data_collection: deny` provider floor, as it always has. A slug carrying an
`@preset/<name>` suffix carries NO provider block at all: the preset already
names a complete routing policy on the account, and a request-level `provider`
object replaces it wholesale — every request would keep returning 200 while the
provider order, quantization floor and retention posture silently vanished.
"""

import pytest

from choreo.llm import _build_chat_params, _build_extra_body


def test_a_bare_slug_keeps_the_data_collection_floor() -> None:
    assert _build_extra_body(None, "provider/model") == {
        "provider": {"data_collection": "deny"},
    }


def test_the_floor_survives_reasoning_configuration() -> None:
    params = _build_chat_params(
        [{"role": "user", "content": "member profile"}],
        "provider/model",
        "low",
    )
    assert params["extra_body"] == {
        "provider": {"data_collection": "deny"},
        "reasoning": {"effort": "low"},
    }


def test_a_preset_suffixed_slug_carries_no_provider_block() -> None:
    params = _build_chat_params(
        [{"role": "user", "content": "member profile"}],
        "provider/model@preset/zwerm",
        "low",
    )
    assert params["extra_body"] == {"reasoning": {"effort": "low"}}
    assert "provider" not in params["extra_body"]


def test_a_fallback_chain_becomes_a_prioritized_models_array() -> None:
    chain = ["a/primary@preset/p", "b/fallback@preset/p"]
    params = _build_chat_params(
        [{"role": "user", "content": "hi"}],
        chain[0],
        None,
        chain,
    )
    assert params["model"] == chain[0]
    assert params["extra_body"]["models"] == chain


def test_a_one_entry_chain_sends_no_models_array() -> None:
    # A `models` array of one buys nothing and costs a field on every request.
    body = _build_extra_body(None, "a/primary", ["a/primary"])
    assert "models" not in body


def test_a_missing_model_raises_rather_than_substituting_one() -> None:
    """A phase whose model resolved to None used to fall through to a packaged
    Gemini slug — silently, on a call carrying member profile material, at a
    different price and a different provider policy."""
    with pytest.raises(ValueError, match="no model for this phase"):
        _build_chat_params([{"role": "user", "content": "hi"}], "")


def test_embed_calls_carry_denied_data_collection_and_ordered_routing(monkeypatch) -> None:
    """The embed path had NO provider block until 2026-08-22, which is half of
    why the gemini-embedding-2 outage was possible: the slug's only
    batch-capable endpoint was a data-collecting provider the account excludes,
    so every batched embed 404'd while the chat path (which always sent the
    policy) looked healthy. `data_collection: deny` is now the floor here too,
    and the routing preference is ORDERED WITH FALLBACKS — never a hard pin."""
    import numpy as np

    from choreo import embed as embed_mod

    captured = {}

    class _Response:
        data = [type("Item", (), {"embedding": [0.1, 0.2]})()]
        usage = None
        model_extra = {}

    class _Embeddings:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Response()

    class _Client:
        embeddings = _Embeddings()

    monkeypatch.setattr(embed_mod, "get_openrouter_client", lambda: _Client())
    monkeypatch.setattr(embed_mod, "get_cost_tracker", lambda: None)

    out = embed_mod.get_embeddings(["hello"], "provider/model")
    assert isinstance(out, np.ndarray)

    provider = captured["extra_body"]["provider"]
    assert provider["data_collection"] == "deny"
    assert provider["order"] == ["nebius", "deepinfra", "siliconflow"]
    assert provider["allow_fallbacks"] is True
    assert "only" not in provider, "a hard provider pin recreates the outage shape"


def test_embed_provider_override_cannot_re_allow_data_collection(monkeypatch) -> None:
    from choreo import embed as embed_mod

    captured = {}

    class _Response:
        data = [type("Item", (), {"embedding": [0.1]})()]
        usage = None
        model_extra = {}

    class _Client:
        embeddings = type("E", (), {"create": lambda self, **kw: (captured.update(kw), _Response())[1]})()

    monkeypatch.setattr(embed_mod, "get_openrouter_client", lambda: _Client())
    monkeypatch.setattr(embed_mod, "get_cost_tracker", lambda: None)

    embed_mod.get_embeddings(
        ["hello"], "provider/model",
        provider={"order": ["deepinfra"], "data_collection": "allow"},
    )
    provider = captured["extra_body"]["provider"]
    assert provider["order"] == ["deepinfra"]
    assert provider["data_collection"] == "deny"
