"""Every Choreo LLM call carries the OpenRouter no-data-collection policy."""

from choreo.llm import _build_chat_params, _build_extra_body


def test_provider_data_collection_is_denied_without_reasoning() -> None:
    assert _build_extra_body(None) == {
        "provider": {"data_collection": "deny"},
    }


def test_provider_block_survives_reasoning_configuration() -> None:
    params = _build_chat_params(
        [{"role": "user", "content": "member profile"}],
        "provider/model",
        "low",
    )
    assert params["extra_body"] == {
        "provider": {"data_collection": "deny"},
        "reasoning": {"effort": "low"},
    }


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
