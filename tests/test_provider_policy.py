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
