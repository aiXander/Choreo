"""Improvement-sprint Phase 0 + Track 4/5 tests (offline).

Covers: F1 re-rank over-fetch, F2 absent-text gate (is_absent), F3 HyDE
cache-key context fingerprint, F4 display_names threading, inline prompt-text
overrides (resolve_prompt_templates), language pinning, and generate_intros
top-N limiting.
"""

import numpy as np
import pytest

from choreo import embed as embed_mod
from choreo.config import load_config, resolve_prompt_templates
from choreo.embed import embed_sections
from choreo.extract import build_extraction_prompt
from choreo.hyde import hyde_cache_key, hyde_context_fingerprint, hyde_descriptors_for_sections
from choreo.query import run_query_match, QUERY_ID
from choreo.batch_match import run_batch_match
from choreo.schemas import EmbeddingsBundle, sections_from_dict
from choreo.score import build_batch_scoring_prompt
from choreo.utils import DEFAULT_PROMPT_PATHS, hash_text, is_absent, load_yaml

from conftest import DIM, FakeLLMWrapper, default_responder, keyword_embed


@pytest.fixture(autouse=True)
def _fake_query_embedding(monkeypatch):
    """Query-mode tests embed the transient atom internally — keep it offline."""
    monkeypatch.setattr(
        embed_mod, "get_embeddings",
        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]),
    )


HYDE_TEMPLATE = load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"]


# ---------------------------------------------------------------------------
# F2 — is_absent + embed/HyDE gating
# ---------------------------------------------------------------------------

def test_is_absent_cases():
    assert is_absent(None)
    assert is_absent("")
    assert is_absent("   \n ")
    assert is_absent("Not specified")
    assert is_absent("not specified")
    assert is_absent("  NOT SPECIFIED. ")
    assert not is_absent("Not specified, but I do have needs")
    assert not is_absent("python")


def test_embed_gates_not_specified_to_zero_vector(fake_embed_fn):
    sections = sections_from_dict({
        "u1": {"skills": "AGENTS engineering", "needs": "Not specified"},
        "u2": {"skills": "Not specified", "needs": "VISUALS please"},
    })
    bundle = embed_sections(
        extracted_sections=sections,
        embedding_model="fake/embedding-model",
        embed_fn=fake_embed_fn,
    )
    # "Not specified" cells are zero vectors and were never sent to the API
    assert np.allclose(bundle.embeddings[0, 1], 0)   # u1.needs
    assert np.allclose(bundle.embeddings[1, 0], 0)   # u2.skills
    assert np.linalg.norm(bundle.embeddings[0, 0]) > 0
    embedded_texts = [t for call in fake_embed_fn.calls for t in call]
    assert "Not specified" not in embedded_texts


def test_embed_zeroes_stale_phantom_vectors_from_existing_bundle(fake_embed_fn):
    """A pre-fix bundle embedded 'Not specified' as real text; on reload the
    absent gate must win over content-hash reuse and zero the cell."""
    sections = sections_from_dict({"u1": {"skills": "Not specified", "needs": "AGENTS"}})
    phantom = EmbeddingsBundle(
        user_ids=["u1"],
        section_names=["skills", "needs"],
        embeddings=np.ones((1, 2, DIM)),
        embedding_model="fake/embedding-model",
        section_hashes={"u1": {"skills": hash_text("Not specified"),
                               "needs": hash_text("AGENTS")}},
    )
    bundle = embed_sections(
        extracted_sections=sections,
        embedding_model="fake/embedding-model",
        existing=phantom,
        embed_fn=fake_embed_fn,
    )
    assert np.allclose(bundle.embeddings[0, 0], 0)        # phantom zeroed
    assert np.allclose(bundle.embeddings[0, 1], np.ones(DIM))  # real cell reused


def test_hyde_skips_absent_sources(fake_llm):
    sections = sections_from_dict({
        "u1": {"skills": "MUSIC", "needs": "Not specified"},
        "u2": {"skills": "FOOD", "needs": "AGENTS backend help"},
    })
    result = hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights={"needs_skills": 0.8},
        hyde_config={"n_descriptors": 2},
        prompt_template=HYDE_TEMPLATE,
        goal="test",
        llm_wrapper=fake_llm,
        model="fake/llm",
    )
    # u1 got empty descriptors without an LLM call; u2 got real ones
    assert result["needs_skills"][0].descriptors == ["", ""]
    assert all("AGENTS" in d for d in result["needs_skills"][1].descriptors)
    assert fake_llm.calls == [("hyde_generation", 1)]  # exactly one prompt sent


# ---------------------------------------------------------------------------
# F3 — HyDE cache key covers prompt template / goal / model / language
# ---------------------------------------------------------------------------

def test_hyde_cache_key_invalidates_on_context_change(fake_llm):
    sections = sections_from_dict({"u1": {"skills": "MUSIC", "needs": "AGENTS help"}})

    def run(existing=None):
        fake_llm.calls.clear()
        return hyde_descriptors_for_sections(
            extracted_sections=sections,
            cross_section_weights={"needs_skills": 0.8},
            hyde_config={"n_descriptors": 1},
            prompt_template=HYDE_TEMPLATE,
            goal="goal A",
            llm_wrapper=fake_llm,
            model="fake/llm",
            existing=existing,
        )

    first = run()
    fingerprint = hyde_context_fingerprint(HYDE_TEMPLATE, "goal A", "fake/llm", "needs_skills")
    key = hyde_cache_key("AGENTS help", 1, "needs_skills", fingerprint)
    existing = {"needs_skills": {key: first["needs_skills"][0].descriptors}}

    # Same context -> cache hit, zero LLM calls
    run(existing=existing)
    assert not fake_llm.calls

    # Changed goal -> different fingerprint -> the stale entry must NOT replay
    fake_llm.calls.clear()
    hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights={"needs_skills": 0.8},
        hyde_config={"n_descriptors": 1},
        prompt_template=HYDE_TEMPLATE,
        goal="goal B — completely different",
        llm_wrapper=fake_llm,
        model="fake/llm",
        existing=existing,
    )
    assert fake_llm.calls  # regenerated

    # Fingerprint differs on template, model and language too
    assert hyde_context_fingerprint(HYDE_TEMPLATE, "g", "m1", "needs_skills") != \
           hyde_context_fingerprint(HYDE_TEMPLATE, "g", "m2", "needs_skills")
    assert hyde_context_fingerprint(HYDE_TEMPLATE, "g", "m", "needs_skills") != \
           hyde_context_fingerprint(HYDE_TEMPLATE + "x", "g", "m", "needs_skills")
    assert hyde_context_fingerprint(HYDE_TEMPLATE, "g", "m", "needs_skills") != \
           hyde_context_fingerprint(HYDE_TEMPLATE, "g", "m", "needs_skills", language="Dutch")


# ---------------------------------------------------------------------------
# F4 — display_names threading
# ---------------------------------------------------------------------------

def test_scoring_prompt_uses_aliases_and_renders_names():
    prompt, alias_of = build_batch_scoring_prompt(
        user_profiles=["uuid-a", "uuid-b"],
        sections_dict={"uuid-a": {"skills": "AGENTS"}, "uuid-b": {"skills": "VISUALS"}},
        instruction="score",
        prompt_template="{instruction} {goal}\n{user_profiles_xml_formatted}\n{json_format_hint}",
        goal="g",
        display_names={"uuid-a": "Alice Anderson"},
    )
    assert alias_of == {"uuid-a": "P1", "uuid-b": "P2"}
    assert '<profile id="P1" name="Alice Anderson">' in prompt
    assert "Profile of Alice Anderson (P1):" in prompt
    # No display name -> the raw id doubles as the profile's name label
    assert '<profile id="P2" name="uuid-b">' in prompt
    # The JSON hint is keyed by aliases with an explicit numeric range —
    # raw ids never appear in the requested keys.
    assert '"P1_P2": <score 0.0-1.0>' in prompt
    assert "uuid-a_uuid-b" not in prompt


def test_scoring_prompt_query_pseudo_user_gets_q_alias():
    from choreo.utils import QUERY_ID
    prompt, alias_of = build_batch_scoring_prompt(
        user_profiles=[QUERY_ID, "alice", "bob"],
        sections_dict={QUERY_ID: {"needs": "AGENTS"},
                       "alice": {"skills": "AGENTS"}, "bob": {"skills": "VISUALS"}},
        instruction="score",
        prompt_template="{user_profiles_xml_formatted}\n{json_format_hint}",
        goal="g",
        pairs=[(QUERY_ID, "alice"), (QUERY_ID, "bob")],
    )
    assert alias_of == {QUERY_ID: "Q", "alice": "P1", "bob": "P2"}
    # The unnamed query renders alias-only (never the __query__ sentinel);
    # candidates fall back to their raw id as name.
    assert '<profile id="Q">' in prompt
    assert "Profile of Q:" in prompt
    assert QUERY_ID not in prompt
    assert '"Q_P1": <score 0.0-1.0>' in prompt
    assert '"Q_P2": <score 0.0-1.0>' in prompt
    # Query-mode disambiguation: the query pseudo-user's sections are search
    # TARGETS, rendered "Looking for (<Section>): …" so the model can't read
    # them as attributes the asker possesses. Candidate sections stay bare.
    assert "Looking for (Needs): AGENTS" in prompt
    assert "  Skills: AGENTS" in prompt  # alice (candidate) — unchanged
    assert "  Skills: VISUALS" in prompt  # bob (candidate) — unchanged
    assert "Looking for (Skills)" not in prompt  # never applied to candidates


def test_pair_scores_parse_back_through_aliases():
    """The fake model answers in alias keys; score_pairs_with_llm must hand
    back PairScores keyed by REAL ids/pair_ids."""
    from choreo.score import get_pair_score

    response = {"Q_P1": 0.7, "P2_P1": "0.4"}
    alias_of = {"__query__": "Q", "uuid-a": "P1", "uuid-b": "P2"}
    assert get_pair_score(response, "__query__", "uuid-a", alias_of) == 0.7
    # reversed alias order still resolves
    assert get_pair_score(response, "uuid-a", "uuid-b", alias_of) == "0.4"
    # raw-id fallback (model echoed ids instead of aliases)
    assert get_pair_score({"uuid-a_uuid-b": 0.2}, "uuid-a", "uuid-b", alias_of) == 0.2
    assert get_pair_score(response, "uuid-b", "nobody", alias_of) is None


def test_query_match_display_names_flow_into_intros(synthetic_bundle, test_config):
    sections, _, pool = synthetic_bundle
    pool_sections = {s.id: s.sections for s in sections}
    llm = FakeLLMWrapper()
    names = {"alice": "Alice Anderson", "bob": "Bob Builder",
             "carol": "Carol C", "david": "David D", QUERY_ID: "Xander"}
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        display_names=names,
        llm_wrapper=llm,
    )
    top = result.shortlist[0]
    assert "For Xander:" in top["intro"]
    assert QUERY_ID not in top["intro"]
    # the re-rank prompt carried the display names alongside the ids
    rerank_prompts = [p for c, p in llm.prompts_seen if c == "query_rerank"]
    assert rerank_prompts and 'name="Alice Anderson"' in rerank_prompts[0]


def test_batch_match_display_names_flow_into_intros(synthetic_bundle, test_config):
    sections, _, pool = synthetic_bundle
    pool_sections = {s.id: s.sections for s in sections}
    llm = FakeLLMWrapper()
    names = {"alice": "Alice Anderson", "bob": "Bob Builder"}
    result = run_batch_match(
        member_ids=["alice"],
        pool=pool,
        config=test_config,
        pool_sections=pool_sections,
        display_names=names,
        llm_wrapper=llm,
    )
    assert result.edges
    intros = " ".join(e.intro for e in result.edges)
    assert "Alice Anderson" in intros
    scoring_prompts = [p for c, p in llm.prompts_seen if c == "batch_pair_scoring"]
    assert scoring_prompts and 'name="Alice Anderson"' in scoring_prompts[0]


# ---------------------------------------------------------------------------
# F1 — re-rank over-fetch recovers below-the-cut candidates
# ---------------------------------------------------------------------------

def _agents_pool(fake_llm, fake_embed_fn):
    """3 candidates with identical embedding signal; the LLM prefers gina."""
    from conftest import build_pool as _bp
    sections, _, pool = _bp({
        "alice": {"skills": "AGENTS engineering", "needs": "VISUALS"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC"},
        "gina": {"skills": "AGENTS research", "needs": "FOOD"},
    }, fake_llm, fake_embed_fn)
    return pool, {s.id: s.sections for s in sections}


def _gina_responder(component, prompt):
    if component == "query_rerank":
        from conftest import profile_labels, scoring_hint_keys
        labels = profile_labels(prompt)
        return {
            k: (0.95 if "gina" in [labels.get(p, p) for p in k.split("_")] else 0.05)
            for k in scoring_hint_keys(prompt)
        }
    return default_responder(component, prompt)


def test_overfetch_recovers_candidate_below_embedding_cut(fake_llm, fake_embed_fn, test_config):
    pool, pool_sections = _agents_pool(fake_llm, fake_embed_fn)
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        top_k=1, generate_intros=False,
        llm_wrapper=FakeLLMWrapper(_gina_responder),
        # rerank_pool_multiplier defaults to 3 -> all 3 candidates re-ranked
    )
    assert len(result.shortlist) == 1                      # truncated to top_k
    assert result.shortlist[0]["user_id"] == "gina"        # recovered from rank 3
    assert any("over-fetch" in n for n in result.notes)


def test_multiplier_one_reproduces_legacy_reorder_only(fake_llm, fake_embed_fn, test_config):
    pool, pool_sections = _agents_pool(fake_llm, fake_embed_fn)
    config = {**test_config, "query": {**test_config["query"], "rerank_pool_multiplier": 1}}
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        top_k=1, generate_intros=False,
        llm_wrapper=FakeLLMWrapper(_gina_responder),
    )
    # only the embedding top-1 was re-ranked -> gina cannot be recovered
    assert result.shortlist[0]["user_id"] == "alice"


# ---------------------------------------------------------------------------
# Track 5 — generate_intros accepts an int (top-N)
# ---------------------------------------------------------------------------

def test_generate_intros_top_n(synthetic_bundle, test_config):
    sections, _, pool = synthetic_bundle
    pool_sections = {s.id: s.sections for s in sections}
    llm = FakeLLMWrapper()
    result = run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=test_config, pool_sections=pool_sections,
        top_k=3, llm_rerank=False, generate_intros=1,
        llm_wrapper=llm,
    )
    assert result.shortlist[0]["intro"]
    assert all(not e["intro"] for e in result.shortlist[1:])
    intro_calls = [n for c, n in llm.calls if c == "introduction_generation"]
    assert sum(intro_calls) == 1


# ---------------------------------------------------------------------------
# Track 4 — inline prompt-text overrides + language pinning
# ---------------------------------------------------------------------------

def test_resolve_prompt_templates_defaults():
    templates = resolve_prompt_templates()
    assert "sections" in templates and isinstance(templates["sections"], dict)
    assert "{json_format_hint}" in templates["scoring"]
    assert "{user_a_name}" in templates["introduction"]
    assert "{source_text}" in templates["hyde"]


def test_resolve_prompt_templates_inline_text_wins(tmp_path):
    # inline text beats both packaged defaults and explicit paths
    custom_scoring = tmp_path / "scoring_prompt.yaml"
    custom_scoring.write_text("pair_scoring: 'FROM_FILE {json_format_hint}'\n")
    config = load_config(overrides={"prompts": {
        "scoring_prompt_text": "INLINE_SCORING {json_format_hint}",
        "section_prompt_text": {"sections": {"skills": {"active": True,
                                                        "guideline": "g", "max_words": 10}},
                                "sections_prompt": "X {profile_text} {sections_list}"},
    }})
    templates = resolve_prompt_templates(
        config=config, prompt_paths={"scoring": str(custom_scoring)})
    assert templates["scoring"].startswith("INLINE_SCORING")
    assert list(templates["sections"]["sections"]) == ["skills"]
    # without inline text the explicit path wins over the packaged default
    templates2 = resolve_prompt_templates(prompt_paths={"scoring": str(custom_scoring)})
    assert templates2["scoring"].startswith("FROM_FILE")


def test_inline_intro_template_reaches_the_llm(synthetic_bundle, test_config):
    sections, _, pool = synthetic_bundle
    pool_sections = {s.id: s.sections for s in sections}
    llm = FakeLLMWrapper()
    config = {**test_config, "prompts": {
        "introduction_prompt_text":
            "CUSTOM_INTRO {user_a_name} + {user_b_name}\n{user1_text}\n{user2_text}",
    }}
    run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        top_k=1, llm_rerank=False, llm_wrapper=llm,
    )
    intro_prompts = [p for c, p in llm.prompts_seen if c == "introduction_generation"]
    assert intro_prompts and intro_prompts[0].startswith("CUSTOM_INTRO")


def test_language_pinning_in_extraction_and_hyde(synthetic_bundle, test_config):
    # unit: extraction prompt line
    sections_config = load_yaml(DEFAULT_PROMPT_PATHS["sections"])
    prompt = build_extraction_prompt("hello", sections_config, goal="g", language="Dutch")
    assert "Write the extracted sections in Dutch." in prompt
    default_prompt = build_extraction_prompt("hello", sections_config, goal="g")
    assert "the same language as the profile text" in default_prompt

    # end-to-end: config language reaches the HyDE prompt on the query path
    sections, _, pool = synthetic_bundle
    pool_sections = {s.id: s.sections for s in sections}
    llm = FakeLLMWrapper()
    config = {**test_config,
              "instruction_prompt": {"goal": "g", "language": "Dutch"}}
    run_query_match(
        query={"needs": "AGENTS engineering"},
        pool=pool, config=config, pool_sections=pool_sections,
        top_k=1, llm_rerank=False, generate_intros=False, llm_wrapper=llm,
    )
    hyde_prompts = [p for c, p in llm.prompts_seen if c == "hyde_generation"]
    assert hyde_prompts and "Write the descriptors in Dutch." in hyde_prompts[0]
