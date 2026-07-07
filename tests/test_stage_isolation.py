"""Per-stage isolation tests on the synthetic 4-user fixture.

Each stage is exercised on the previous stage's output — both as in-memory
objects and through its dump/load disk format — proving the two chaining
styles compose. Includes the WS2 regression check: rectangular similarity with
source == target reduces EXACTLY to the historical square behavior.
"""

import numpy as np

from choreo.utils import cosine_matrix, load_yaml, parse_cross_key, stable_pair_id, DEFAULT_PROMPT_PATHS
from choreo.schemas import sections_from_dict
from choreo.extract import extract_sections
from choreo.hyde import hyde_descriptors_for_sections
from choreo.embed import embed_sections
from choreo.candidate import (
    compute_fused_similarity_matrix,
    generate_rectangular_similarity,
)
from choreo.score import score_pairs_with_llm, create_sections_dict
from choreo.match import greedy_b_matching, create_matches
from choreo.candidate import CandidatePair
from choreo.introduction import generate_introductions_for_matches
from choreo.report import build_report_data
from choreo.ingest import Profile
from choreo.utils import hash_text
from choreo import stages


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def _profiles(sections_dict):
    return [
        Profile(id=uid, text=" / ".join(s.values()), hash=hash_text(" / ".join(s.values())))
        for uid, s in sections_dict.items()
    ]


def test_extract_stage(synthetic_sections_dict, fake_llm):
    sections_config = {
        "sections": {name: {"guideline": "g", "max_words": 50}
                     for name in ("skills", "vision", "project", "needs")},
        "sections_prompt": "Context: {goal}\n<profile>\n{profile_text}\n</profile>\n{sections_list}",
    }
    profiles = _profiles(synthetic_sections_dict)
    extracted = extract_sections(
        profiles=profiles,
        sections_config=sections_config,
        model="fake/llm",
        llm_wrapper=fake_llm,
        goal="test",
    )
    assert [e.id for e in extracted] == list(synthetic_sections_dict)
    assert fake_llm.call_count == 4
    # every section populated by the fake responder
    assert all(e.sections["skills"] for e in extracted)

    # `existing` reuse: same profiles again -> zero LLM calls
    existing = {e.hash: e.sections for e in extracted}
    fake_llm.calls.clear()
    again = extract_sections(
        profiles=profiles,
        sections_config=sections_config,
        model="fake/llm",
        llm_wrapper=fake_llm,
        goal="test",
        existing=existing,
    )
    assert not fake_llm.calls
    assert [e.sections for e in again] == [e.sections for e in extracted]


# ---------------------------------------------------------------------------
# hyde — n_descriptors 1 AND > 1 (§3.3 lock-in)
# ---------------------------------------------------------------------------

def _gen_hyde(sections, fake_llm, n_descriptors):
    return hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights={"needs_skills": 0.8},
        hyde_config={"n_descriptors": n_descriptors},
        prompt_template=load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"],
        goal="test",
        llm_wrapper=fake_llm,
        model="fake/llm",
    )


def test_hyde_stage_single_and_multi_descriptor(synthetic_sections_dict, fake_llm):
    sections = sections_from_dict(synthetic_sections_dict)

    hyde1 = _gen_hyde(sections, fake_llm, 1)
    assert set(hyde1) == {"needs_skills"}
    assert all(len(hd.descriptors) == 1 for hd in hyde1["needs_skills"])
    assert [hd.user_id for hd in hyde1["needs_skills"]] == [s.id for s in sections]

    hyde3 = _gen_hyde(sections, fake_llm, 3)
    assert all(len(hd.descriptors) == 3 for hd in hyde3["needs_skills"])
    # fake responder echoes the source text -> keyword survives in EVERY variant
    alice_hyde = hyde3["needs_skills"][0]
    assert all("VISUALS" in d for d in alice_hyde.descriptors)

    # `existing` reuse keyed by content hash + prompt-context fingerprint
    # -> zero LLM calls
    from choreo.hyde import hyde_cache_key, hyde_context_fingerprint
    template = load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"]
    fingerprint = hyde_context_fingerprint(template, "test", "fake/llm", "needs_skills")
    existing = {"needs_skills": {
        hyde_cache_key(s.sections["needs"], 3, "needs_skills", fingerprint):
            hyde3["needs_skills"][i].descriptors
        for i, s in enumerate(sections)
    }}
    fake_llm.calls.clear()
    again = hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights={"needs_skills": 0.8},
        hyde_config={"n_descriptors": 3},
        prompt_template=load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"],
        goal="test",
        llm_wrapper=fake_llm,
        model="fake/llm",
        existing=existing,
    )
    assert not fake_llm.calls
    assert again["needs_skills"][0].descriptors == hyde3["needs_skills"][0].descriptors


# ---------------------------------------------------------------------------
# embed — bundle shape + provenance; multi-descriptor HyDE shape
# ---------------------------------------------------------------------------

def test_embed_stage_bundle_shape_and_provenance(synthetic_sections_dict, fake_llm, fake_embed_fn):
    sections = sections_from_dict(synthetic_sections_dict)
    hyde3 = _gen_hyde(sections, fake_llm, 3)
    bundle = embed_sections(
        extracted_sections=sections,
        embedding_model="fake/embedding-model",
        hyde_descriptors=hyde3,
        embed_fn=fake_embed_fn,
    )
    assert bundle.embeddings.shape == (4, 4, 32)
    assert bundle.hyde["needs_skills"].shape == (4, 3, 32)   # n_descriptors=3 preserved
    assert bundle.embedding_model == "fake/embedding-model"
    assert bundle.dim == 32
    assert set(bundle.section_hashes) == {s.id for s in sections}
    assert set(bundle.hyde_hashes["needs_skills"]) == {s.id for s in sections}


# ---------------------------------------------------------------------------
# similarity — WS2 regression: rectangular(source==target) == legacy square
# ---------------------------------------------------------------------------

def _legacy_square_fused(embeddings, section_names, section_weights,
                         cross_section_weights, hyde_embeddings):
    """Verbatim re-implementation of the PRE-refactor square algorithm."""
    n_users = embeddings.shape[0]
    section_present, section_matrices = {}, {}
    for idx, name in enumerate(section_names):
        sec = embeddings[:, idx, :]
        section_present[name] = np.linalg.norm(sec, axis=1) > 1e-8
        section_matrices[name] = cosine_matrix(sec)

    weighted_sum = np.zeros((n_users, n_users))
    weight_mass = np.zeros((n_users, n_users))
    for name, weight in section_weights.items():
        if name not in section_matrices:
            continue
        present = section_present[name]
        mask = np.outer(present, present).astype(float)
        weighted_sum += weight * mask * section_matrices[name]
        weight_mass += abs(weight) * mask

    for cross_key, weight in (cross_section_weights or {}).items():
        src_section, tgt_section = parse_cross_key(cross_key)
        src_emb = hyde_embeddings[cross_key]
        tgt_idx = section_names.index(tgt_section)
        tgt_emb = embeddings[:, tgt_idx:tgt_idx + 1, :]
        src_norm = src_emb / (np.linalg.norm(src_emb, axis=2, keepdims=True) + 1e-8)
        tgt_norm = tgt_emb / (np.linalg.norm(tgt_emb, axis=2, keepdims=True) + 1e-8)
        cross = np.full((n_users, n_users), -np.inf)
        for sd in range(src_emb.shape[1]):
            for td in range(tgt_emb.shape[1]):
                cross = np.maximum(cross, src_norm[:, sd, :] @ tgt_norm[:, td, :].T)
        src_present = np.linalg.norm(src_emb, axis=2).max(axis=1) > 1e-8
        mask = np.outer(src_present, section_present[tgt_section]).astype(float)
        weighted_sum += weight * mask * cross
        weight_mass += abs(weight) * mask

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight_mass > 0, weighted_sum / weight_mass, 0.0)


def test_rectangular_reduces_to_square_exactly():
    rng = np.random.default_rng(7)
    n, dim = 6, 16
    section_names = ["skills", "vision", "needs"]
    embeddings = rng.normal(size=(n, len(section_names), dim))
    embeddings[2, 1] = 0.0  # one absent section -> masking semantics covered
    hyde = {"needs_skills": rng.normal(size=(n, 2, dim))}  # n_descriptors=2
    section_weights = {"skills": -0.1, "vision": 0.3}
    cross_weights = {"needs_skills": 0.8}

    legacy = _legacy_square_fused(embeddings, section_names, section_weights,
                                  cross_weights, hyde)

    # (a) square mode (no target passed)
    fused_square, _ = compute_fused_similarity_matrix(
        embeddings=embeddings, section_names=section_names,
        section_weights=section_weights, cross_section_weights=cross_weights,
        hyde_embeddings=hyde,
    )
    # (b) rectangular mode with target == source
    fused_rect, _ = compute_fused_similarity_matrix(
        embeddings=embeddings, section_names=section_names,
        section_weights=section_weights, cross_section_weights=cross_weights,
        hyde_embeddings=hyde,
        target_embeddings=embeddings, target_section_names=section_names,
    )

    # Square mode (no target passed) preserves the legacy path BIT-EXACTLY.
    assert np.array_equal(fused_square, legacy)
    # Explicit rectangular with target == source agrees to numerical noise
    # (BLAS uses a symmetric kernel for A @ A.T that the rectangular
    # A @ B.T path reproduces only to ~1 ULP).
    assert np.allclose(fused_rect, legacy, rtol=0, atol=1e-12)


def test_rectangular_subset_consistency(synthetic_bundle):
    """A rectangular source-subset must equal the matching rows of the square run."""
    sections, hyde, bundle = synthetic_bundle
    recipe = {"section_weights": {"vision": 0.2},
              "cross_section_weights": {"needs_skills": 0.8}}

    square = generate_rectangular_similarity(bundle, bundle, recipe_config=recipe)
    assert square.sym_matrix is not None  # square run gets the symmetric matrix

    sub = bundle.subset(["bob", "david"])
    rect = generate_rectangular_similarity(sub, bundle, recipe_config=recipe)
    assert rect.sym_matrix is None        # rectangular: never symmetrized
    assert rect.dir_matrix.shape == (2, 4)
    bob_idx = bundle.user_ids.index("bob")
    david_idx = bundle.user_ids.index("david")
    assert np.allclose(rect.dir_matrix[0], square.dir_matrix[bob_idx])
    assert np.allclose(rect.dir_matrix[1], square.dir_matrix[david_idx])


def test_keyword_pairing_signal(synthetic_bundle):
    """The engineered needs↔skills keywords produce the intended directional hits."""
    _, _, bundle = synthetic_bundle
    recipe = {"section_weights": {}, "cross_section_weights": {"needs_skills": 1.0}}
    res = generate_rectangular_similarity(bundle, bundle, recipe_config=recipe)
    ids = bundle.user_ids
    d = res.dir_matrix
    # alice needs VISUALS -> bob's skills (cos 1); carol needs FOOD -> david
    assert d[ids.index("alice"), ids.index("bob")] > 0.99
    assert d[ids.index("carol"), ids.index("david")] > 0.99
    assert d[ids.index("alice"), ids.index("carol")] < 0.01


# ---------------------------------------------------------------------------
# score — excluded_pairs honored; selected_pairs path
# ---------------------------------------------------------------------------

def test_score_stage_with_exclusions(synthetic_bundle, fake_llm, test_config):
    sections, _, bundle = synthetic_bundle
    recipe = {"section_weights": {"vision": 0.2},
              "cross_section_weights": {"needs_skills": 0.8}}
    res = generate_rectangular_similarity(bundle, bundle, recipe_config=recipe)

    excluded = {stable_pair_id("alice", "bob")}
    scores = score_pairs_with_llm(
        similarity_matrix=res.sym_matrix,
        user_ids=bundle.user_ids,
        sections_dict=create_sections_dict(sections),
        instruction="score",
        goal="test",
        prompts_config_path=DEFAULT_PROMPT_PATHS["scoring"],
        llm_wrapper=fake_llm,
        model="fake/llm",
        max_n_llm_evaluations_per_profile=8,
        global_cap=50,
        n_profiles_to_score_together=4,
        excluded_pairs=excluded,
    )
    assert scores  # something was scored
    assert "alice_bob" not in scores
    assert all(0.0 <= s.score <= 1.0 for s in scores.values())


def test_score_retry_recovers_dropped_scores(synthetic_bundle, test_config):
    """A parsed-but-incomplete scoring response (valid JSON, missing pair keys)
    triggers a retry pass that re-asks ONLY the missing pairs."""
    from conftest import FakeLLMWrapper, default_responder
    sections, _, bundle = synthetic_bundle
    recipe = {"section_weights": {"vision": 0.2},
              "cross_section_weights": {"needs_skills": 0.8}}
    res = generate_rectangular_similarity(bundle, bundle, recipe_config=recipe)

    def _score(llm):
        return score_pairs_with_llm(
            similarity_matrix=res.sym_matrix,
            user_ids=bundle.user_ids,
            sections_dict=create_sections_dict(sections),
            instruction="score",
            goal="test",
            prompts_config_path=DEFAULT_PROMPT_PATHS["scoring"],
            llm_wrapper=llm,
            model="fake/llm",
            max_n_llm_evaluations_per_profile=8,
            global_cap=50,
            n_profiles_to_score_together=4,
        )

    baseline = _score(FakeLLMWrapper())          # nothing dropped

    state = {"first": True}

    def dropping_responder(component, prompt):
        if component == "batch_pair_scoring" and state["first"]:
            state["first"] = False
            return {}        # valid JSON, but EVERY pair key omitted
        return default_responder(component, prompt)

    recovered = _score(FakeLLMWrapper(dropping_responder))
    assert set(recovered) == set(baseline)       # retry recovered every pair
    assert recovered                             # and there was something to score


def test_profile_grouping_deterministic_across_hash_seeds():
    """Grouping must not depend on set iteration order (hash-salted per
    process): different PYTHONHASHSEED values must yield identical groups,
    otherwise scoring cache keys change every run and nothing ever hits."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "from choreo.score import create_profile_groups_from_pairs\n"
        "from choreo.candidate import CandidatePair\n"
        "users = [f'u{i:02d}' for i in range(13)]\n"
        "pairs = [CandidatePair.create(users[i], users[j], 0.5)\n"
        "         for i in range(13) for j in range(i + 1, 13)]\n"
        "groups = create_profile_groups_from_pairs(pairs, 5)\n"
        "print([','.join(sorted(g)) for g in groups])\n"
    )

    def run(seed):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        out = subprocess.run([sys.executable, "-c", script], env=env,
                             capture_output=True, text=True, check=True,
                             cwd=str(Path(__file__).parent.parent))
        return out.stdout.splitlines()[-1]

    assert run(1) == run(42) == run(999)


# ---------------------------------------------------------------------------
# match — asymmetric caps + exclusions
# ---------------------------------------------------------------------------

def _edge(u1, u2, w):
    from choreo.schemas import Edge
    return Edge(user1=u1, user2=u2, pair_id=stable_pair_id(u1, u2),
                final_weight=w, embed_score=w, llm_score=w)


def test_greedy_b_matching_symmetric_unchanged():
    edges = [_edge("a", "b", 0.9), _edge("a", "c", 0.8),
             _edge("b", "c", 0.7), _edge("c", "d", 0.6), _edge("b", "d", 0.5)]
    selected = greedy_b_matching(edges, b_min=1, b_max=2,
                                 all_users={"a", "b", "c", "d"})
    pair_ids = {e.pair_id for e in selected}
    # Phase 1 picks the top edges greedily…
    assert {"a_b", "a_c", "b_c"} <= pair_ids
    degrees = {}
    for e in selected:
        degrees[e.user1] = degrees.get(e.user1, 0) + 1
        degrees[e.user2] = degrees.get(e.user2, 0) + 1
    # …and 'd' (below b_min) gets force-filled in Phase 3, which legitimately
    # relaxes the partner's b_max — the legacy semantics.
    assert all(degrees.get(u, 0) >= 1 for u in "abcd")
    assert all(degrees[u] <= 2 for u in "ad")  # caps hold where no force-fill needed


def test_greedy_b_matching_asymmetric_pool_cap():
    # Both members' best partner is pool user "p1"; pool_b_max=1 forces the
    # second member onto p2 instead of saturating p1.
    edges = [
        _edge("m1", "p1", 0.9), _edge("m2", "p1", 0.85),
        _edge("m1", "p2", 0.5), _edge("m2", "p2", 0.45),
    ]
    selected = greedy_b_matching(
        edges, b_min=1, b_max=1,
        all_users={"m1", "m2", "p1", "p2"},
        member_ids={"m1", "m2"},
        pool_b_max=1,
    )
    pair_ids = sorted(e.pair_id for e in selected)
    assert pair_ids == ["m1_p1", "m2_p2"]


def test_greedy_b_matching_exclusions():
    edges = [_edge("m1", "p1", 0.9), _edge("m1", "p2", 0.5)]
    selected = greedy_b_matching(
        edges, b_min=1, b_max=2, all_users={"m1", "p1", "p2"},
        excluded_pairs={"m1_p1"},
    )
    assert [e.pair_id for e in selected] == ["m1_p2"]


def test_create_matches_with_explicit_reference_scores():
    from choreo.schemas import PairScore
    cands = [CandidatePair.create("a", "b", 0.6), CandidatePair.create("a", "c", 0.4)]
    llm_scores = {
        "a_b": PairScore("a_b", "a", "b", 0.6, 0.9),
        "a_c": PairScore("a_c", "a", "c", 0.4, 0.2),
    }
    edges, norm_embed, _ = create_matches(
        candidates=cands,
        llm_scores=llm_scores,
        all_user_ids=["a", "b", "c"],
        matching_config={"b_min": 1, "b_max": 2},
        blending_config={"embed_weight": 0.5, "llm_weight": 0.5},
        reference_scores=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    )
    # explicit reference [0,1]: embed 0.6 -> 0.6 normalized, 0.4 -> 0.4
    assert abs(norm_embed["a_b"] - 0.6) < 1e-9
    assert abs(norm_embed["a_c"] - 0.4) < 1e-9
    assert edges


# ---------------------------------------------------------------------------
# introduce + report
# ---------------------------------------------------------------------------

def test_introduce_and_report_stages(synthetic_bundle, fake_llm):
    sections, _, _ = synthetic_bundle
    edges = [_edge("alice", "bob", 0.9), _edge("carol", "david", 0.8)]
    intros = generate_introductions_for_matches(
        final_edges=edges,
        sections_dict=create_sections_dict(sections),
        instruction="intro",
        goal="test",
        introduction_config_path=DEFAULT_PROMPT_PATHS["introduction"],
        llm_wrapper=fake_llm,
        model="fake/llm",
    )
    assert set(intros) == {"alice_bob", "carol_david"}
    assert "For alice:" in intros["alice_bob"].intro

    for e in edges:
        e.intro = intros[e.pair_id].intro
        e.starter_topics = intros[e.pair_id].starter_topics

    report = build_report_data(edges, sections, top_matches_per_user=2)
    assert set(report["user_reports"]) == {"alice", "bob", "carol", "david"}
    assert report["cohort_summary"]["overview"]["total_users"] == 4

    # scope restriction (subset batch mode shape)
    scoped = build_report_data(edges, sections, top_matches_per_user=2,
                               scope_user_ids=["alice"])
    assert set(scoped["user_reports"]) == {"alice"}
    assert scoped["cohort_summary"]["overview"]["total_users"] == 1
    # degree stats are scoped too: pool-side endpoints must not pollute them
    assert scoped["cohort_summary"]["overview"]["average_degree"] == 1.0
    assert scoped["cohort_summary"]["degree_distribution"] == {1: 1}


# ---------------------------------------------------------------------------
# disk chaining: stage N's dump feeds stage N+1's load
# ---------------------------------------------------------------------------

def test_stages_chain_via_disk(synthetic_sections_dict, fake_llm, fake_embed_fn, tmp_path):
    sections = sections_from_dict(synthetic_sections_dict)
    stages.dump_sections(sections, tmp_path / "sections.jsonl")

    sections2 = stages.load_sections(tmp_path / "sections.jsonl")
    hyde = _gen_hyde(sections2, fake_llm, 1)
    stages.dump_hyde(hyde, tmp_path / "hyde.json")

    bundle = embed_sections(
        extracted_sections=sections2,
        embedding_model="fake/embedding-model",
        hyde_descriptors=stages.load_hyde(tmp_path / "hyde.json"),
        embed_fn=fake_embed_fn,
    )
    stages.dump_embeddings(bundle, tmp_path / "embeds")

    bundle2 = stages.load_embeddings(tmp_path / "embeds")
    res = generate_rectangular_similarity(
        bundle2, bundle2,
        recipe_config={"section_weights": {"vision": 0.2},
                       "cross_section_weights": {"needs_skills": 0.8}},
    )
    stages.dump_similarity(res, tmp_path / "sim")
    res2 = stages.load_similarity(tmp_path / "sim")

    # in-memory result == disk-roundtripped result
    assert np.allclose(res.dir_matrix, res2.dir_matrix)
    assert res2.source_ids == bundle.user_ids
