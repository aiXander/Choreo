"""Incremental-embedding tests: content-hash reuse, not roster reuse (WS1).

Locks the core economics: changing one profile's one section re-embeds exactly
that cell; adding a user re-embeds only the new user; unchanged vectors are
byte-identical across runs.
"""

import numpy as np

from choreo.schemas import sections_from_dict
from choreo.embed import embed_sections, create_section_embeddings_bundle
from choreo import embed as embed_mod


def _sections(d):
    return sections_from_dict(d)


BASE = {
    "alice": {"skills": "AGENTS dev", "needs": "VISUALS help"},
    "bob": {"skills": "VISUALS art", "needs": "AGENTS help"},
    "carol": {"skills": "MUSIC", "needs": "FOOD"},
}


def test_single_section_change_reembeds_only_that_cell(fake_embed_fn):
    bundle1 = embed_sections(_sections(BASE), "fake/model", embed_fn=fake_embed_fn)
    assert sum(len(c) for c in fake_embed_fn.calls) == 6  # 3 users × 2 sections

    changed = {**BASE, "bob": {"skills": "VISUALS art", "needs": "ROBOTS help"}}
    fake_embed_fn.calls.clear()
    bundle2 = embed_sections(_sections(changed), "fake/model",
                             existing=bundle1, embed_fn=fake_embed_fn)

    # exactly ONE text embedded: bob's changed needs
    assert [t for call in fake_embed_fn.calls for t in call] == ["ROBOTS help"]

    # every unchanged vector is byte-identical
    for u in ("alice", "carol"):
        i1, i2 = bundle1.user_ids.index(u), bundle2.user_ids.index(u)
        assert np.array_equal(bundle1.embeddings[i1], bundle2.embeddings[i2])
    b1, b2 = bundle1.user_ids.index("bob"), bundle2.user_ids.index("bob")
    s = bundle1.section_names.index("skills")
    n = bundle1.section_names.index("needs")
    assert np.array_equal(bundle1.embeddings[b1, s], bundle2.embeddings[b2, s])
    assert not np.array_equal(bundle1.embeddings[b1, n], bundle2.embeddings[b2, n])


def test_adding_user_does_not_reembed_everyone(fake_embed_fn):
    bundle1 = embed_sections(_sections(BASE), "fake/model", embed_fn=fake_embed_fn)
    fake_embed_fn.calls.clear()

    grown = {**BASE, "dave": {"skills": "FOOD chef", "needs": "MUSIC dj"}}
    bundle2 = embed_sections(_sections(grown), "fake/model",
                             existing=bundle1, embed_fn=fake_embed_fn)

    embedded = [t for call in fake_embed_fn.calls for t in call]
    assert sorted(embedded) == ["FOOD chef", "MUSIC dj"]  # only the new user
    assert bundle2.user_ids == ["alice", "bob", "carol", "dave"]
    for u in BASE:
        i1, i2 = bundle1.user_ids.index(u), bundle2.user_ids.index(u)
        assert np.array_equal(bundle1.embeddings[i1], bundle2.embeddings[i2])


def test_removing_user_keeps_remaining_vectors(fake_embed_fn):
    bundle1 = embed_sections(_sections(BASE), "fake/model", embed_fn=fake_embed_fn)
    fake_embed_fn.calls.clear()

    shrunk = {k: v for k, v in BASE.items() if k != "carol"}
    bundle2 = embed_sections(_sections(shrunk), "fake/model",
                             existing=bundle1, embed_fn=fake_embed_fn)
    assert not fake_embed_fn.calls           # nothing re-embedded
    assert bundle2.user_ids == ["alice", "bob"]


def test_model_change_invalidates_existing(fake_embed_fn):
    bundle1 = embed_sections(_sections(BASE), "fake/model", embed_fn=fake_embed_fn)
    fake_embed_fn.calls.clear()
    embed_sections(_sections(BASE), "other/model",
                   existing=bundle1, embed_fn=fake_embed_fn)
    assert sum(len(c) for c in fake_embed_fn.calls) == 6  # full re-embed


def test_hyde_incremental_reuse(fake_embed_fn, fake_llm):
    from choreo.hyde import hyde_descriptors_for_sections
    from choreo.utils import load_yaml, DEFAULT_PROMPT_PATHS

    sections = _sections(BASE)
    template = load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"]
    hyde = hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights={"needs_skills": 0.8},
        hyde_config={"n_descriptors": 1},
        prompt_template=template, goal="t",
        llm_wrapper=fake_llm, model="fake/llm",
    )
    bundle1 = embed_sections(sections, "fake/model",
                             hyde_descriptors=hyde, embed_fn=fake_embed_fn)
    fake_embed_fn.calls.clear()

    # same inputs again with existing -> zero embedding calls at all
    bundle2 = embed_sections(sections, "fake/model", hyde_descriptors=hyde,
                             existing=bundle1, embed_fn=fake_embed_fn)
    assert not fake_embed_fn.calls
    assert np.array_equal(bundle1.hyde["needs_skills"], bundle2.hyde["needs_skills"])


def test_filestore_wrapper_roundtrip(tmp_path, monkeypatch, fake_embed_fn):
    """create_section_embeddings_bundle: persist, then rerun with zero API calls."""
    monkeypatch.setattr(embed_mod, "get_embeddings",
                        lambda texts, model: fake_embed_fn(texts))

    sections = _sections(BASE)
    embeds_dir = tmp_path / "embeds"
    bundle1 = create_section_embeddings_bundle(
        sections, "fake/model", str(embeds_dir))
    assert (embeds_dir / "vectors.npz").exists()
    assert (embeds_dir / "bundle_meta.json").exists()

    fake_embed_fn.calls.clear()
    bundle2 = create_section_embeddings_bundle(
        sections, "fake/model", str(embeds_dir))
    assert not fake_embed_fn.calls           # disk reuse, no embedding calls
    assert np.array_equal(bundle1.embeddings, bundle2.embeddings)


def test_legacy_dir_adoption(tmp_path, monkeypatch, fake_embed_fn):
    """A pre-refactor embeds dir (no bundle_meta.json) is adopted for an
    unchanged roster instead of re-embedding everyone."""
    monkeypatch.setattr(embed_mod, "get_embeddings",
                        lambda texts, model: fake_embed_fn(texts))
    sections = _sections(BASE)
    embeds_dir = tmp_path / "embeds"
    bundle1 = create_section_embeddings_bundle(sections, "fake/model", str(embeds_dir))

    # strip the meta side-car -> legacy layout
    (embeds_dir / "bundle_meta.json").unlink()

    fake_embed_fn.calls.clear()
    bundle2 = create_section_embeddings_bundle(sections, "fake/model", str(embeds_dir))
    assert not fake_embed_fn.calls
    assert np.array_equal(bundle1.embeddings, bundle2.embeddings)
    # and the dir is upgraded with hashes for next time
    assert (embeds_dir / "bundle_meta.json").exists()
