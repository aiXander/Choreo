"""A HyDE descriptor is always a string — at the cache boundary, both ways.

Regression for a real outage (wintercircus, 2026-09-18 → 09-19). A model answered
`{"descriptors": ["...", false]}`; the raw JSON `false` was written into
`hyde/<cross_key>.jsonl`, and because that cache is content-addressed on the
SOURCE TEXT the bad entry replayed on every run forever. `hyde_content_hash` does
`"\x1f".join(descriptors)`, so every FULL run for the whole deployment died with
`TypeError: sequence item 1: expected str instance, bool found` — one member's
line taking down a 1,500-member nightly reconcile for two nights.

Both directions matter and are tested separately: sanitizing only on the way IN
leaves every already-poisoned cache file broken until someone wipes it (and pays
to regenerate every descriptor); sanitizing only on the way OUT keeps writing bad
entries. `hyde_content_hash` itself stays strict on purpose — it is the tripwire.
"""

import pytest

from choreo.hyde import _clean_descriptors
from choreo.schemas import hyde_content_hash


def test_a_non_string_descriptor_is_dropped_and_padded_with_the_source_text():
    """The exact prod shape: item 1 is a JSON bool."""
    out = _clean_descriptors(
        ["a real descriptor", False], n_descriptors=2, fallback="the user's own needs text"
    )
    assert out == ["a real descriptor", "the user's own needs text"]
    # The whole point: this is now hashable.
    assert hyde_content_hash(out)


@pytest.mark.parametrize(
    "raw",
    [
        ["ok", False],          # the observed failure
        ["ok", True],
        ["ok", None],
        ["ok", 3],
        ["ok", {"d": "x"}],
        ["ok", ["nested"]],
        None,                    # cache miss / `merged.get` returned nothing
        "a bare string",         # the LLM answered a string, not a list
        {"descriptors": "x"},    # not a list at all
        [],
    ],
)
def test_every_shape_yields_exactly_n_strings(raw):
    out = _clean_descriptors(raw, n_descriptors=2, fallback="fallback text")
    assert len(out) == 2
    assert all(isinstance(d, str) for d in out)
    assert hyde_content_hash(out)  # never raises


def test_a_bare_string_response_is_kept_as_the_first_descriptor():
    """Preserves the pre-existing `isinstance(descriptors, str)` behaviour."""
    assert _clean_descriptors("just one", n_descriptors=2, fallback="fb") == ["just one", "fb"]


def test_good_descriptors_pass_through_untouched_and_are_truncated_to_n():
    assert _clean_descriptors(["a", "b", "c"], n_descriptors=2, fallback="fb") == ["a", "b"]
    # Empty strings are legitimate (absent source -> zero vectors, masked downstream).
    assert _clean_descriptors(["", ""], n_descriptors=2, fallback="fb") == ["", ""]


def test_hyde_content_hash_stays_strict():
    """The tripwire is deliberately NOT softened: anything bypassing the sanitizer
    must still fail loudly rather than hash a silently-coerced value."""
    with pytest.raises(TypeError):
        hyde_content_hash(["ok", False])
