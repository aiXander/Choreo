"""Mode C (subset batch match) tests — offline, with seeded exclusions."""

import numpy as np

from choreo.utils import stable_pair_id
from choreo.batch_match import run_batch_match, select_pairs_rectangular
from choreo.store import FileStore


def _pool(synthetic_bundle):
    sections, _, bundle = synthetic_bundle
    return bundle, {s.id: s.sections for s in sections}


def test_batch_match_excludes_history_pairs(fake_llm, fake_embed_fn, test_config):
    # 6-user pool where each member has a BACKUP keyword match, so excluding
    # their best historical pair still leaves novel candidates:
    #   alice needs VISUALS -> bob (excluded) or eve
    #   bob   needs AGENTS  -> alice (excluded, same pair) or frank
    from conftest import build_pool
    pool_dict = {
        "alice": {"skills": "AGENTS dev", "needs": "VISUALS for my show"},
        "bob": {"skills": "VISUALS shader art", "needs": "AGENTS backend"},
        "eve": {"skills": "VISUALS stage design", "needs": "FOOD catering"},
        "frank": {"skills": "AGENTS automation", "needs": "MUSIC for a bar"},
        "carol": {"skills": "MUSIC composition", "needs": "FOOD pop-up"},
        "david": {"skills": "FOOD fermentation", "needs": "MUSIC dj"},
    }
    sections, _, pool = build_pool(pool_dict, fake_llm, fake_embed_fn,
                                   cross_weights={"needs_skills": 0.8})
    pool_sections = {s.id: s.sections for s in sections}
    config = {**test_config,
              "recipe": {**test_config["recipe"], "section_weights": {}}}
    excluded = {stable_pair_id("alice", "bob")}

    result = run_batch_match(
        member_ids=["alice", "bob"],
        pool=pool,
        config=config,
        excluded_pairs=excluded,
        pool_sections=pool_sections,
        llm_wrapper=fake_llm,
    )

    surfaced = {e.pair_id for e in result.edges}
    assert "alice_bob" not in surfaced
    assert all(p["pair_id"] != "alice_bob" for p in result.new_pairs)
    assert result.excluded_count == 1

    # members get NOVEL matches (the backup keyword candidates)
    member_degrees = {m: 0 for m in ("alice", "bob")}
    for e in result.edges:
        for u in (e.user1, e.user2):
            if u in member_degrees:
                member_degrees[u] += 1
    assert all(d >= config["matching"]["b_min"] for d in member_degrees.values())
    assert {"alice_eve", "bob_frank"} <= {e.pair_id for e in result.edges}
    assert set(result.report_data["user_reports"]) == {"alice", "bob"}

    # every edge touches at least one member (member × pool restriction)
    assert all({"alice", "bob"} & {e.user1, e.user2} for e in result.edges)

    # intros attached
    assert all(e.intro for e in result.edges)


def test_batch_match_unknown_member_raises(synthetic_bundle, fake_llm, test_config):
    pool, pool_sections = _pool(synthetic_bundle)
    try:
        run_batch_match(
            member_ids=["ghost"],
            pool=pool, config=test_config, excluded_pairs=set(),
            pool_sections=pool_sections, llm_wrapper=fake_llm,
        )
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "ghost" in str(exc)


def test_select_pairs_rectangular_dedup_and_self_skip():
    # 2 members vs pool of 3 (members included in pool)
    member_ids = ["a", "b"]
    pool_ids = ["a", "b", "c"]
    # dir[i][j]: member i x pool j
    dir_matrix = np.array([
        [9.0, 0.8, 0.6],   # a: self, a->b 0.8, a->c 0.6
        [0.4, 9.0, 0.5],   # b: b->a 0.4, self, b->c 0.5
    ])
    pairs = select_pairs_rectangular(
        dir_matrix, member_ids, pool_ids,
        max_n_llm_evaluations_per_profile=8, global_cap=50,
    )
    by_id = {p.pair_id: p for p in pairs}
    # self pairs (9.0 entries) never appear
    assert all("_" in pid and pid.split("_")[0] != pid.split("_")[1] for pid in by_id)
    # a_b appears ONCE with the two directions averaged: (0.8 + 0.4) / 2
    assert abs(by_id["a_b"].similarity_score - 0.6) < 1e-9
    # member->pool-only pairs keep their single directional value
    assert abs(by_id["a_c"].similarity_score - 0.6) < 1e-9
    assert abs(by_id["b_c"].similarity_score - 0.5) < 1e-9


def test_select_pairs_rectangular_exclusions_and_caps():
    member_ids = ["a"]
    pool_ids = ["b", "c", "d"]
    dir_matrix = np.array([[0.9, 0.8, 0.7]])
    pairs = select_pairs_rectangular(
        dir_matrix, member_ids, pool_ids,
        max_n_llm_evaluations_per_profile=2, global_cap=50,
        excluded_pairs={"a_b"},
    )
    ids = [p.pair_id for p in pairs]
    assert "a_b" not in ids          # excluded
    assert len(ids) == 2             # per-profile cap on the member


def test_batch_sections_provider_scoped_to_selected_pairs_and_members(
    fake_llm, fake_embed_fn, test_config
):
    """Lazy pool_sections (P1): the provider runs once, with union(users in
    selected pairs, member_ids) — a pool user whose only candidate pair is
    history-excluded is never fetched, and a member with zero surviving pairs
    still rides along so their report renders."""
    from conftest import build_pool
    pool_dict = {
        "alice": {"skills": "AGENTS dev", "needs": "VISUALS for my show"},
        "bob": {"skills": "VISUALS shader art", "needs": "AGENTS backend"},
        "loner": {"skills": "MUSIC composition", "needs": "FOOD catering"},
        "stranger": {"skills": "FOOD fermentation", "needs": "MUSIC dj"},
    }
    sections, _, pool = build_pool(pool_dict, fake_llm, fake_embed_fn,
                                   cross_weights={"needs_skills": 0.8})
    all_sections = {s.id: s.sections for s in sections}
    config = {**test_config,
              "recipe": {**test_config["recipe"], "section_weights": {}}}

    provider_calls = []

    def provider(ids):
        provider_calls.append(list(ids))
        return {i: all_sections[i] for i in ids}

    result = run_batch_match(
        member_ids=["alice", "loner"],
        pool=pool,
        config=config,
        excluded_pairs={stable_pair_id("loner", "stranger")},  # loner's only pair
        pool_sections=None,
        sections_provider=provider,
        llm_wrapper=fake_llm,
    )

    assert len(provider_calls) == 1
    (ids,) = provider_calls
    # selected-pair users (alice, bob) ∪ members (alice, loner); stranger's
    # only candidate pair was history-excluded ⇒ never fetched
    assert set(ids) == {"alice", "bob", "loner"}
    # the zero-pair member still gets a rendered report
    assert set(result.report_data["user_reports"]) == {"alice", "loner"}
    assert {e.pair_id for e in result.edges} == {"alice_bob"}


def test_batch_requires_sections_or_provider(synthetic_bundle, fake_llm, test_config):
    pool, _ = _pool(synthetic_bundle)
    try:
        run_batch_match(
            member_ids=["alice"], pool=pool, config=test_config,
            excluded_pairs=set(), pool_sections=None, llm_wrapper=fake_llm,
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sections_provider" in str(exc)


def test_filestore_match_history_window(tmp_path):
    from choreo.schemas import Edge

    store = FileStore(tmp_path)
    e_old = Edge("a", "b", "a_b", 0.9, 0.5, 0.5)
    e_new = Edge("a", "c", "a_c", 0.8, 0.5, 0.5)
    store.put_matches([e_old], matched_at="2020-01-01T00:00:00+00:00")
    store.put_matches([e_new])  # now

    assert store.get_match_history() == {"a_b", "a_c"}            # no window: all
    assert store.get_match_history(window_months=6) == {"a_c"}    # old one expired
    assert store.get_match_history(ids=["b"]) == {"a_b"}          # id filter


def test_batch_then_history_roundtrip(synthetic_bundle, fake_llm, test_config, tmp_path):
    """Full novelty loop: run, persist history, rerun -> only novel pairs."""
    pool, pool_sections = _pool(synthetic_bundle)
    store = FileStore(tmp_path)

    first = run_batch_match(
        member_ids=["alice"], pool=pool, config=test_config,
        excluded_pairs=store.get_match_history(window_months=6),
        pool_sections=pool_sections, llm_wrapper=fake_llm,
    )
    assert first.edges
    store.put_matches(first.edges)

    second = run_batch_match(
        member_ids=["alice"], pool=pool, config=test_config,
        excluded_pairs=store.get_match_history(window_months=6),
        pool_sections=pool_sections, llm_wrapper=fake_llm,
    )
    first_ids = {e.pair_id for e in first.edges}
    second_ids = {e.pair_id for e in second.edges}
    assert not first_ids & second_ids   # strictly novel matches
