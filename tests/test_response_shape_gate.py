"""`required_key_sets` — the response-SHAPE gate over JSON completions.

Regression suite for the 2026-09-02 wintercircus match run, where two published
match cards carried placeholder prose. `response_format={"type":"json_object"}`
guarantees syntax, not schema: one call came back as `{"": "{\\"intro_for_a\\": …}"}`
(the real answer, stringified inside an empty-key wrapper) and one as
`{": ": ", "}` (degenerate). Both parsed, both were cached, neither was retried,
and `dict.get(key, <default>)` turned them into cheerful filler.

Offline throughout — the transport is stubbed, no API key needed.
"""

import json
import types

import pytest

from choreo import llm as llm_mod
from choreo.introduction import (
    INTRODUCTION_KEY_SETS,
    generate_introductions_for_matches,
)
from choreo.llm import (
    JSONExtractionError,
    LLMWrapper,
    run_coro_blocking,
    satisfies_key_sets,
    unwrap_json_envelope,
)
from choreo.schemas import Edge

# The two shapes actually observed in the incident, kept verbatim.
STRINGIFIED_ENVELOPE = {
    "": json.dumps(
        {
            "intro_for_a": "What Lorin can offer Dorien.",
            "intro_for_b": "What Dorien can offer Lorin.",
            "starter_topics": "- topic one\n- topic two",
        }
    )
}
DEGENERATE = {": ": ", "}
GOOD = {
    "intro_for_a": "A about B",
    "intro_for_b": "B about A",
    "starter_topics": "- topic one",
}


# ---------------------------------------------------------------------------
# satisfies_key_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [GOOD, {"intro": "x", "starter_topics": "y"}])
def test_either_alternative_key_set_passes(value):
    """Directional AND legacy formats are both legitimate — a custom
    `introduction_prompt_text` may still ask for the single-`intro` shape, and
    failing it would burn a full retry wave on every pair."""
    assert satisfies_key_sets(value, INTRODUCTION_KEY_SETS)


@pytest.mark.parametrize(
    "value",
    [
        STRINGIFIED_ENVELOPE,
        DEGENERATE,
        {},
        {"intro_for_a": "A", "intro_for_b": "B"},          # no starter_topics
        {"intro_for_a": "A", "intro_for_b": "", "starter_topics": "t"},  # blank half
        {"intro_for_a": "A", "intro_for_b": None, "starter_topics": "t"},
        {"intro_for_a": "A", "intro_for_b": "  ", "starter_topics": "t"},
        "a bare string",
        ["a", "list"],
        None,
    ],
)
def test_incomplete_shapes_fail(value):
    assert not satisfies_key_sets(value, INTRODUCTION_KEY_SETS)


def test_no_key_sets_means_no_requirement():
    """The opt-out that keeps every undeclared phase (scoring, extraction,
    hyde, the query re-rank) behaving exactly as before."""
    for value in (DEGENERATE, {}, None, "x", 3):
        assert satisfies_key_sets(value, None)
        assert satisfies_key_sets(value, ())


def test_zero_and_false_count_as_filled():
    """Numeric/boolean answers are real answers — a score of 0 is not absent."""
    assert satisfies_key_sets({"score": 0}, (("score",),))
    assert satisfies_key_sets({"ok": False}, (("ok",),))


# ---------------------------------------------------------------------------
# unwrap_json_envelope
# ---------------------------------------------------------------------------

def test_unwrap_recovers_the_stringified_answer():
    """The whole point: the correct intro was one JSON envelope away."""
    out = unwrap_json_envelope(STRINGIFIED_ENVELOPE, INTRODUCTION_KEY_SETS)
    assert out["intro_for_a"] == "What Lorin can offer Dorien."
    assert out["starter_topics"] == "- topic one\n- topic two"


def test_unwrap_is_a_noop_on_a_well_formed_response():
    assert unwrap_json_envelope(GOOD, INTRODUCTION_KEY_SETS) is GOOD


def test_unwrap_returns_the_original_when_nothing_inside_satisfies():
    """Conservative by construction — it may only ever return an object that
    passes the gate, never a half-salvaged guess."""
    assert unwrap_json_envelope(DEGENERATE, INTRODUCTION_KEY_SETS) is DEGENERATE
    nested_junk = {"wrapper": json.dumps({"intro_for_a": "only half"})}
    assert unwrap_json_envelope(nested_junk, INTRODUCTION_KEY_SETS) is nested_junk


def test_unwrap_never_promotes_a_scalar_string_value():
    """`{"score": "0.8"}` must stay itself: the value parses as JSON, but as a
    scalar, and promoting it would silently replace an object with a float."""
    value = {"score": "0.8"}
    assert unwrap_json_envelope(value, (("score",),)) is value


def test_unwrap_is_inert_without_key_sets():
    assert unwrap_json_envelope(STRINGIFIED_ENVELOPE, None) is STRINGIFIED_ENVELOPE


def test_unwrap_survives_a_self_referential_payload():
    """Depth/node budget: a pathological response must terminate, not hang."""
    deep = {"a": json.dumps({"b": json.dumps({"c": json.dumps({"d": "x"})})})}
    assert unwrap_json_envelope(deep, INTRODUCTION_KEY_SETS) is deep


# ---------------------------------------------------------------------------
# The completion path: reject, retry, and never cache a rejected shape
# ---------------------------------------------------------------------------

class _StubClient:
    async def close(self):
        return None


def _stub_transport(monkeypatch, payloads):
    """Stub the OpenRouter call so it answers with `payloads` in order (the last
    one repeats). Returns the list of prompts actually sent."""
    monkeypatch.setattr(llm_mod, "make_async_openrouter_client", lambda: _StubClient())

    async def _noop():
        return None

    monkeypatch.setattr(llm_mod, "cleanup_background_tasks", _noop)

    sent = []

    async def _fake_completion(client, messages, model, **kwargs):
        idx = min(len(sent), len(payloads) - 1)
        sent.append(messages)
        text = payloads[idx]
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=None
        )

    monkeypatch.setattr(llm_mod, "async_chat_completion", _fake_completion)
    return sent


def _wrapper(tmp_path, monkeypatch):
    wrapper = LLMWrapper(cache_dir=str(tmp_path), max_retries=2)
    monkeypatch.setattr(wrapper, "_record_usage", lambda *a, **k: None)
    return wrapper


def test_bad_shape_is_retried_then_raises_and_is_not_cached(tmp_path, monkeypatch):
    """The two behaviors the incident lacked: a re-sample, and a clean cache.
    Caching a rejected response would pin one bad generation under a
    content-addressed key and replay it on every later run."""
    sent = _stub_transport(monkeypatch, [json.dumps(DEGENERATE)])
    wrapper = _wrapper(tmp_path, monkeypatch)

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["p"], model="fake/llm", cache_keys=["intro_pair"],
        required_key_sets=INTRODUCTION_KEY_SETS,
        max_retries=2, retry_delay_base=0.0,
    ))

    assert isinstance(results[0], JSONExtractionError)
    assert len(sent) == 3  # 1 attempt + 2 retries
    assert list(wrapper.cache_dir.glob("*.json")) == []


def test_a_stringified_answer_is_salvaged_and_cached_unwrapped(tmp_path, monkeypatch):
    sent = _stub_transport(monkeypatch, [json.dumps(STRINGIFIED_ENVELOPE)])
    wrapper = _wrapper(tmp_path, monkeypatch)

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["p"], model="fake/llm", cache_keys=["intro_pair"],
        required_key_sets=INTRODUCTION_KEY_SETS,
    ))

    assert len(sent) == 1  # salvaged in place — no re-spend
    assert results[0]["intro_for_a"] == "What Lorin can offer Dorien."
    cached = json.loads((wrapper.cache_dir / "intro_pair.json").read_text())
    assert cached["intro_for_a"] == "What Lorin can offer Dorien."


def test_undeclared_phases_still_accept_any_object(tmp_path, monkeypatch):
    """No `required_key_sets` ⇒ byte-for-byte the previous behavior, including
    caching. Scoring/extraction/hyde must not change."""
    _stub_transport(monkeypatch, [json.dumps(DEGENERATE)])
    wrapper = _wrapper(tmp_path, monkeypatch)

    results = run_coro_blocking(wrapper.batch_json_complete(
        prompts=["p"], model="fake/llm", cache_keys=["score_pair"],
    ))

    assert results[0] == DEGENERATE
    assert (wrapper.cache_dir / "score_pair.json").exists()


# ---------------------------------------------------------------------------
# The cache read: poisoned entries self-heal
# ---------------------------------------------------------------------------

def test_poisoned_cache_entry_is_treated_as_a_miss(tmp_path):
    wrapper = LLMWrapper(cache_dir=str(tmp_path))
    (wrapper.cache_dir / "intro_pair.json").write_text(json.dumps(DEGENERATE))

    assert wrapper._load_cached("intro_pair", INTRODUCTION_KEY_SETS) is None
    # …and is still served to a phase that declares no requirement.
    assert wrapper._load_cached("intro_pair", None) == DEGENERATE


def test_stringified_cache_entry_is_recovered_without_respending(tmp_path):
    """Exactly Dorien's cached file: the prose is there, under an empty key."""
    wrapper = LLMWrapper(cache_dir=str(tmp_path))
    (wrapper.cache_dir / "intro_pair.json").write_text(json.dumps(STRINGIFIED_ENVELOPE))

    cached = wrapper._load_cached("intro_pair", INTRODUCTION_KEY_SETS)
    assert cached["intro_for_b"] == "What Dorien can offer Lorin."


def test_unreadable_cache_entry_is_a_miss_not_a_crash(tmp_path):
    wrapper = LLMWrapper(cache_dir=str(tmp_path))
    (wrapper.cache_dir / "intro_pair.json").write_text("{not json")

    assert wrapper._load_cached("intro_pair", INTRODUCTION_KEY_SETS) is None


def test_no_cache_dir_is_always_a_miss():
    wrapper = LLMWrapper(cache_dir=None)
    assert wrapper._load_cached("intro_pair", INTRODUCTION_KEY_SETS) is None


# ---------------------------------------------------------------------------
# End to end through generate_introductions_for_matches
# ---------------------------------------------------------------------------

class _CannedLLM:
    """Minimal batch_json_complete stand-in returning one canned response."""

    def __init__(self, response):
        self.response = response
        self.cache_dir = None
        self.received_key_sets = "unset"

    def set_component(self, component):
        pass

    async def batch_json_complete(self, prompts, model=None, cache_keys=None,
                                  schema_hints=None, required_key_sets=None, **kwargs):
        self.received_key_sets = required_key_sets
        return [self.response for _ in prompts]


SECTIONS = {
    "u1": {"skills": "backend"},
    "u2": {"skills": "design"},
}
NAMES = {"u1": "Ada", "u2": "Grace"}
TEMPLATE = "{goal} {instruction} {user_a_name} {user_b_name} {user1_text} {user2_text}"


def _run(response):
    llm = _CannedLLM(response)
    edge = Edge(user1="u1", user2="u2", pair_id="u1_u2",
                final_weight=0.8, embed_score=0.5, llm_score=0.9)
    intros = generate_introductions_for_matches(
        final_edges=[edge],
        sections_dict=SECTIONS,
        instruction="match them",
        goal="a goal",
        prompt_template=TEMPLATE,
        llm_wrapper=llm,
        model="fake/llm",
        display_names=NAMES,
    )
    return llm, intros["u1_u2"]


def test_the_gate_is_declared_to_the_llm_layer():
    llm, _ = _run(GOOD)
    assert llm.received_key_sets == INTRODUCTION_KEY_SETS


@pytest.mark.parametrize("response", [DEGENERATE, {}, None, "text", ValueError("boom")])
def test_a_bad_response_yields_the_honest_fallback_never_filler(response):
    """The published-card contract: an unusable response must read as "you've
    been matched", not as authored prose that says nothing."""
    _, intro = _run(response)
    assert "Great to meet you" not in intro.intro
    assert "Share your background" not in intro.starter_topics
    assert "You've been matched with Grace" in intro.intro
    assert "Ada" in intro.intro and "Grace" in intro.intro


def test_a_partial_response_falls_back_rather_than_half_rendering():
    """One directional half present is not an introduction — the old code
    dropped straight to filler here."""
    _, intro = _run({"intro_for_a": "A about B", "starter_topics": "- t"})
    assert "You've been matched with" in intro.intro


def test_a_stringified_response_still_renders_the_real_prose():
    """Belt and braces: even if the envelope reaches this layer un-unwrapped
    (a cached entry from before the gate, say), the card must not carry filler."""
    _, intro = _run(STRINGIFIED_ENVELOPE)
    assert "Great to meet you" not in intro.intro


def test_a_good_response_renders_both_directions():
    _, intro = _run(GOOD)
    assert intro.intro == "For Ada: A about B\n\nFor Grace: B about A"
    assert intro.starter_topics == "- topic one"


def test_a_list_of_starter_topics_is_joined_not_repr_d():
    _, intro = _run({**GOOD, "starter_topics": ["- one", "- two"]})
    assert intro.starter_topics == "- one\n- two"


def test_the_legacy_single_intro_format_still_works():
    _, intro = _run({"intro": "they should talk", "starter_topics": "- t"})
    assert intro.intro == "they should talk"
