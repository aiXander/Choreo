"""Mode B (query match) tests — offline, against the synthetic keyword pool."""

import numpy as np
import pytest

from choreo import embed as embed_mod
from choreo.query import build_query_atom, run_query_match, QUERY_ID
from conftest import FakeLLMWrapper, default_responder, keyword_embed


@pytest.fixture(autouse=True)
def _fake_query_embedding(monkeypatch):
    """run_query_match embeds the transient query atom internally — route that
    through the deterministic keyword embedder instead of the real API."""
    monkeypatch.setattr(
        embed_mod, "get_embeddings",
        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]),
    )


def _pool(synthetic_bundle):
    sections, _, bundle = synthetic_bundle
    return bundle, {s.id: s.sections for s in sections}


def test_explicit_mapping_query_ranks_keyword_match(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    result = run_query_match(
        query={"needs": "I need AGENTS engineering for my project"},
        pool=pool,
        config=test_config,
        pool_sections=pool_sections,
        llm_rerank=False,   # pure-embedding path first
        generate_intros=False,
        llm_wrapper=fake_llm,
    )
    # alice's skills carry the AGENTS keyword -> cosine 1.0 -> rank #1
    assert result.shortlist[0]["user_id"] == "alice"
    assert result.shortlist[0]["embed_score"] > 0.99
    assert not result.llm_rerank_applied
    assert result.pool_size == 4
    # scores ranked descending
    scores = [e["score"] for e in result.shortlist]
    assert scores == sorted(scores, reverse=True)


def test_llm_rerank_on_by_default_and_reorders(fake_llm, fake_embed_fn, test_config):
    # Two candidates share the AGENTS skill keyword (equal embed score 1.0);
    # the LLM re-rank is the tie-breaker that puts frank on top.
    from conftest import build_pool
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "carol": {"skills": "MUSIC", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    pool_sections = {s.id: s.sections for s in sections}

    def responder(component, prompt):
        if component == "query_rerank":
            import re
            keys = re.findall(r'"([^"]+)": "0\.\.1"', prompt)
            return {k: (0.95 if "frank" in k else 0.05) for k in keys}
        return default_responder(component, prompt)

    llm = FakeLLMWrapper(responder)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool,
        config=test_config,
        pool_sections=pool_sections,
        generate_intros=False,
        llm_wrapper=llm,
        # llm_rerank NOT passed -> defaults ON (config query.llm_rerank: true)
    )
    assert result.llm_rerank_applied
    assert "query_rerank" in llm.components_called()
    # equal embed scores; llm_weight 0.65 dominates -> frank jumps over alice
    ids = [e["user_id"] for e in result.shortlist]
    assert set(ids) >= {"alice", "frank"}
    assert result.shortlist[0]["user_id"] == "frank"
    assert result.shortlist[0]["llm_score"] == 0.95


def test_rerank_retries_missing_scores(fake_llm, fake_embed_fn, test_config):
    """A parsed-but-incomplete rerank response triggers a re-ask for ONLY the
    unscored candidates (transport retries don't cover this failure mode)."""
    from conftest import build_pool
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    pool_sections = {s.id: s.sections for s in sections}

    rerank_calls = {"n": 0}

    def responder(component, prompt):
        if component == "query_rerank":
            import re
            keys = re.findall(r'"([^"]+)": "0\.\.1"', prompt)
            rerank_calls["n"] += 1
            if rerank_calls["n"] == 1:
                keys = keys[1:]   # first response omits one pair's score
            return {k: 0.8 for k in keys}
        return default_responder(component, prompt)

    llm = FakeLLMWrapper(responder)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=llm,
    )
    assert rerank_calls["n"] == 2          # initial round + exactly one retry
    assert result.llm_rerank_applied
    # every shortlisted candidate ended up LLM-scored despite the dropped key
    assert all(e["llm_score"] is not None for e in result.shortlist)


def test_recipe_override_changes_results(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)

    base = run_query_match(
        query={"needs": "MUSIC composition help"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        llm_rerank=False, generate_intros=False, llm_wrapper=fake_llm,
    )
    # needs->skills: carol has MUSIC skills
    assert base.shortlist[0]["user_id"] == "carol"

    # Override the cross weight to target `needs` instead of `skills`:
    # now the match is whoever NEEDS music -> david.
    override = {"section_weights": {}, "cross_section_weights": {"needs_needs": 1.0}}
    flipped = run_query_match(
        query={"needs": "MUSIC composition help"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        recipe_override=override,
        llm_rerank=False, generate_intros=False, llm_wrapper=fake_llm,
    )
    assert flipped.recipe["cross_section_weights"] == {"needs_needs": 1.0}
    assert flipped.shortlist[0]["user_id"] == "david"


def test_multi_descriptor_query(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    config = {**test_config, "hyde": {"n_descriptors": 3}}
    result = run_query_match(
        query={"needs": "VISUALS for a stage show"},
        pool=pool, config=config, pool_sections=pool_sections,
        llm_rerank=False, generate_intros=False, llm_wrapper=fake_llm,
    )
    # max-pool over 3 descriptors still lands the keyword match
    assert result.shortlist[0]["user_id"] == "bob"
    assert result.shortlist[0]["embed_score"] > 0.99


def test_exclude_ids_and_top_k(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        exclude_ids={"alice"}, top_k=2,
        llm_rerank=False, generate_intros=False, llm_wrapper=fake_llm,
    )
    ids = [e["user_id"] for e in result.shortlist]
    assert "alice" not in ids
    assert len(ids) <= 2


def test_missing_pool_sections_skips_rerank_with_note(synthetic_bundle, fake_llm, test_config):
    pool, _ = _pool(synthetic_bundle)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config,
        pool_sections=None,   # no sections -> rerank + intros skipped
        llm_wrapper=fake_llm,
    )
    assert not result.llm_rerank_applied
    assert any("pool_sections" in n for n in result.notes)
    assert result.shortlist  # embedding-only ranking still returned


def test_intros_generated_for_shortlist(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        llm_rerank=False, top_k=2, llm_wrapper=fake_llm,
    )
    assert all(e["intro"] for e in result.shortlist)
    assert "introduction_generation" in fake_llm.components_called()


def test_auto_expand_raw_text_query(synthetic_bundle, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    llm = FakeLLMWrapper()
    result = run_query_match(
        query="find me someone great at AGENTS engineering",   # raw text
        pool=pool, config=test_config, pool_sections=pool_sections,
        llm_rerank=False, generate_intros=False, llm_wrapper=llm,
    )
    # extraction ran (auto-expand) and the keyword carried into the ranking
    assert "profile_extraction" in llm.components_called()
    assert result.shortlist[0]["user_id"] == "alice"
    # query sections follow the pool's section order
    assert list(result.query_sections.keys()) == pool.section_names


def test_build_query_atom_validation(synthetic_bundle):
    pool, _ = _pool(synthetic_bundle)
    with pytest.raises(ValueError):
        build_query_atom({"nonexistent_section": "x"}, pool.section_names)
    atom = build_query_atom({"needs": "AGENTS"}, pool.section_names)
    assert atom.id == QUERY_ID
    assert atom.sections["needs"] == "AGENTS"
    assert atom.sections["skills"] == ""   # unmapped -> absent (masked neutral)


def test_pool_model_mismatch_raises(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    config = {**test_config, "models": {**test_config["models"],
                                        "embedding": "different/model"}}
    with pytest.raises(ValueError, match="not comparable"):
        run_query_match(
            query={"needs": "AGENTS"},
            pool=pool, config=config, pool_sections=pool_sections,
            llm_wrapper=fake_llm,
        )


def test_reference_scores_normalization(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        llm_rerank=False, generate_intros=False, llm_wrapper=fake_llm,
        reference_scores=np.array([0.0, 1.0]),   # identity min-max mapping
    )
    top = result.shortlist[0]
    assert abs(top["embed_score_normalized"] - top["embed_score"]) < 1e-6
