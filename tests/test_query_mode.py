"""Mode B (query match) tests — offline, against the synthetic keyword pool."""

import numpy as np
import pytest

from choreo import embed as embed_mod
from choreo.query import build_query_atom, run_query_match, QUERY_ID
from conftest import (
    FakeLLMWrapper,
    default_responder,
    keyword_embed,
    profile_labels,
    scoring_hint_keys,
)


def _key_labels(key: str, labels: dict) -> list:
    """The human labels on either side of an alias pair key like 'Q_P2'."""
    return [labels.get(part, part) for part in key.split("_")]


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
            labels = profile_labels(prompt)
            return {k: (0.95 if "frank" in _key_labels(k, labels) else 0.05)
                    for k in scoring_hint_keys(prompt)}
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


def _omit_one_score_pool(fake_llm, fake_embed_fn):
    """A 3-candidate pool + a responder whose FIRST round omits one pair's
    score (the failure mode a re-ask exists to cover)."""
    from conftest import build_pool
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    calls = {"n": 0}

    def responder(component, prompt):
        if component == "query_rerank":
            keys = scoring_hint_keys(prompt)
            calls["n"] += 1
            if calls["n"] == 1:
                keys = keys[1:]   # first response omits one pair's score
            return {k: 0.8 for k in keys}
        return default_responder(component, prompt)

    return pool, {s.id: s.sections for s in sections}, calls, responder


def test_rerank_does_not_retry_by_default(fake_llm, fake_embed_fn, test_config):
    """A dropped score costs NO serial re-ask round by default: the retry is a
    whole extra round-trip after the wave is already dispatched, so
    `query.rerank_max_retries` defaults to 0 and the unscored candidate simply
    drops out (test_config carries no override — it mirrors config.yaml)."""
    pool, pool_sections, calls, responder = _omit_one_score_pool(fake_llm, fake_embed_fn)

    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=FakeLLMWrapper(responder),
        top_k=2,
    )
    assert calls["n"] == 1                 # one round, no re-ask
    assert result.llm_rerank_applied
    # The unscored candidate is DROPPED, never embed-ranked into the shortlist.
    assert all(e["llm_score"] is not None for e in result.shortlist)
    assert len(result.shortlist) == 2


def test_rerank_retries_missing_scores_when_configured(fake_llm, fake_embed_fn, test_config):
    """`query.rerank_max_retries` > 0 re-asks for ONLY the unscored candidates
    (transport retries don't cover a parsed-but-incomplete response)."""
    pool, pool_sections, calls, responder = _omit_one_score_pool(fake_llm, fake_embed_fn)
    config = {**test_config, "query": {**test_config.get("query", {}), "rerank_max_retries": 1}}

    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=FakeLLMWrapper(responder),
    )
    assert calls["n"] == 2                 # initial round + exactly one retry
    assert result.llm_rerank_applied
    # every shortlisted candidate ended up LLM-scored despite the dropped key
    assert all(e["llm_score"] is not None for e in result.shortlist)


def test_unscored_candidate_cannot_outrank_a_scored_one(fake_llm, fake_embed_fn, test_config):
    """The drop-unscored invariant. An unscored candidate must never ride its
    raw embed_norm into the shortlist: that scores it as though the LLM fully
    endorsed its embedding rank, so it would beat candidates the LLM actually
    saw and judged poor — inverting the re-rank's whole purpose."""
    from conftest import build_pool
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    pool_sections = {s.id: s.sections for s in sections}

    def responder(component, prompt):
        if component == "query_rerank":
            keys = scoring_hint_keys(prompt)
            # Score everyone POORLY except the top embed candidate, which is
            # omitted entirely — the embed-only fallback would rank it first.
            return {k: 0.05 for k in keys[1:]}
        return default_responder(component, prompt)

    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=FakeLLMWrapper(responder),
        top_k=2,
    )
    assert len(result.shortlist) == 2
    assert all(e["llm_score"] is not None for e in result.shortlist)
    assert any("Dropped 1 unscored candidate" in n for n in result.notes)


def test_falls_back_to_embed_ranking_when_too_few_scored(fake_llm, fake_embed_fn, test_config):
    """Drop-unscored must not starve the shortlist: when fewer than top_k
    candidates came back scored, unscored ones keep embedding-only ranking
    rather than returning a short list."""
    from conftest import build_pool
    sections, _, pool = build_pool({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    pool_sections = {s.id: s.sections for s in sections}

    def responder(component, prompt):
        if component == "query_rerank":
            keys = scoring_hint_keys(prompt)
            return {keys[0]: 0.8}          # only ONE candidate ever scored
        return default_responder(component, prompt)

    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=FakeLLMWrapper(responder),
        top_k=3,
    )
    assert len(result.shortlist) == 3      # filled, not starved
    assert any("keep embedding-only ranking" in n for n in result.notes)


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


def test_sections_provider_fetches_once_scoped_to_rerank_pool(fake_llm, fake_embed_fn, test_config):
    """Lazy pool_sections (P1): with pool_sections=None + a sections_provider,
    the provider runs exactly once, with only the over-fetched re-rank
    candidate ids — never the whole roster — and the re-rank consumes it."""
    from conftest import build_pool
    pool_dict = {
        f"agent{i}": {"skills": f"AGENTS engineering flavor {i}", "needs": "VISUALS"}
        for i in range(5)
    }
    pool_dict["zed"] = {"skills": "MUSIC", "needs": "FOOD"}   # never a candidate
    sections, _, pool = build_pool(pool_dict, fake_llm, fake_embed_fn)
    all_sections = {s.id: s.sections for s in sections}

    provider_calls = []

    def provider(ids):
        provider_calls.append(list(ids))
        return {i: all_sections[i] for i in ids}

    query_llm = FakeLLMWrapper()   # fresh wrapper: pool-build HyDE stays out of its ledger
    result = run_query_match(
        query={"skills": "AGENTS engineering"},
        pool=pool,
        config=test_config,
        pool_sections=None,
        sections_provider=provider,
        # agent-leg style: same-section weights, empty cross ⇒ no HyDE call
        recipe_override={"section_weights": {"skills": 1.0}, "cross_section_weights": {}},
        top_k=1,
        generate_intros=False,
        llm_wrapper=query_llm,
    )
    assert len(provider_calls) == 1
    (ids,) = provider_calls
    # top_k(1) × rerank_pool_multiplier(3 default) = 3 of the 5 eligible —
    # a strict subset of the 6-user roster, zed never fetched
    assert len(ids) == 3
    assert set(ids) <= {f"agent{i}" for i in range(5)}
    assert result.llm_rerank_applied   # provider output fed the re-rank
    # the leg-style query path makes NO LLM call before the re-rank
    assert query_llm.components_called() == ["query_rerank"]


def test_sections_provider_ignored_when_pool_sections_given(synthetic_bundle, fake_llm, test_config):
    """Precedence: explicit pool_sections wins — the provider is never called."""
    pool, pool_sections = _pool(synthetic_bundle)

    def provider(ids):
        raise AssertionError("provider must not be consulted")

    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config,
        pool_sections=pool_sections, sections_provider=provider,
        generate_intros=False, llm_wrapper=fake_llm,
    )
    assert result.llm_rerank_applied


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


# ---------------------------------------------------------------------------
# query_scoring template — the directional Mode-B re-rank prompt
# ---------------------------------------------------------------------------

def test_query_rerank_uses_query_scoring_template(synthetic_bundle, fake_llm, test_config):
    """The re-rank renders the packaged `query_scoring` template (directional,
    no-reciprocity), not the mutual pair_scoring one."""
    pool, pool_sections = _pool(synthetic_bundle)
    llm = FakeLLMWrapper()
    run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=llm,
    )
    rerank_prompts = [p for c, p in llm.prompts_seen if c == "query_rerank"]
    assert rerank_prompts
    assert 'is a QUERY' in rerank_prompts[0]
    assert "Can person A's skills help person B" not in rerank_prompts[0]


def test_query_scoring_resolution_precedence():
    """Inline query text > scoring file's query_scoring key > pair template."""
    from choreo.config import resolve_prompt_templates

    # Packaged defaults: distinct query template resolves
    templates = resolve_prompt_templates()
    assert templates["query_scoring"] != templates["scoring"]
    assert "is a QUERY" in templates["query_scoring"]

    # A custom inline scoring prompt WITHOUT a query variant governs both
    # paths (pre-query_scoring behavior preserved for adopters)
    inline = "CUSTOM {user_profiles_xml_formatted} {json_format_hint}"
    templates = resolve_prompt_templates(config={"prompts": {"scoring_prompt_text": inline}})
    assert templates["scoring"] == inline
    assert templates["query_scoring"] == inline

    # An explicit inline query variant wins over everything
    templates = resolve_prompt_templates(config={"prompts": {
        "scoring_prompt_text": inline,
        "query_scoring_prompt_text": "QUERYONLY {user_profiles_xml_formatted} {json_format_hint}",
    }})
    assert templates["query_scoring"].startswith("QUERYONLY")


def test_pair_instruction_stays_out_of_query_prompts(synthetic_bundle, test_config):
    """The packaged query template renders {goal} only: recipe.instruction is
    pair-framed flavor and must not leak into the directional re-rank prompt.
    Different query framing = override the template wholesale
    (query_scoring_prompt_text)."""
    pool, pool_sections = _pool(synthetic_bundle)
    config = {**test_config,
              "recipe": {**test_config["recipe"], "instruction": "PAIR-ONLY-PROSE"},
              "instruction_prompt": {"goal": "COMMUNITY-GOAL"}}
    llm = FakeLLMWrapper()
    run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        generate_intros=False, llm_wrapper=llm,
    )
    rerank_prompts = [p for c, p in llm.prompts_seen if c == "query_rerank"]
    assert rerank_prompts
    assert "COMMUNITY-GOAL" in rerank_prompts[0]
    assert "PAIR-ONLY-PROSE" not in rerank_prompts[0]
