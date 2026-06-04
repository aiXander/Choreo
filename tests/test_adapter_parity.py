"""Adapter parity: FileStore-driven run == in-memory-objects-driven run.

Proves the two chaining styles (disk vs Python objects) agree, and that the
full pipeline composes end-to-end offline through every entry stage.
"""

import numpy as np

from choreo.ingest import Profile
from choreo.utils import hash_text
from choreo.schemas import sections_from_dict
from choreo.store import FileStore
from choreo.runners import run_full_match
from choreo import embed as embed_mod
from conftest import FakeLLMWrapper, keyword_embed


def _edge_map(edges):
    return {e.pair_id: round(e.final_weight, 9) for e in edges}


def test_filestore_vs_inmemory_parity(synthetic_sections_dict, test_config,
                                      tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "get_embeddings",
                        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]))
    sections = sections_from_dict(synthetic_sections_dict)

    # (a) pure in-memory: no store at all
    run_mem = run_full_match(sections, test_config, llm_wrapper=FakeLLMWrapper())

    # (b) FileStore-backed: same inputs, disk caches + persistence
    store = FileStore(tmp_path / "group")
    run_store = run_full_match(sections, test_config, store=store,
                               llm_wrapper=FakeLLMWrapper())

    # identical matching outcomes
    assert _edge_map(run_mem["edges"]) == _edge_map(run_store["edges"])
    assert run_mem["report_data"]["cohort_summary"]["overview"] == \
           run_store["report_data"]["cohort_summary"]["overview"]
    assert np.allclose(run_mem["similarity"]["sym_matrix"],
                       run_store["similarity"]["sym_matrix"])

    # the store actually persisted: a fresh FileStore can serve the bundle back
    bundle = FileStore(tmp_path / "group").get_embeddings()
    assert bundle.user_ids == run_store["embeddings"].user_ids
    assert np.array_equal(bundle.embeddings, run_store["embeddings"].embeddings)

    # rerunning store-backed is fully cached: zero embedding API calls
    calls = []

    def counting_embed(texts, model):
        calls.append(texts)
        return np.vstack([keyword_embed(t) for t in texts])

    monkeypatch.setattr(embed_mod, "get_embeddings", counting_embed)
    run_again = run_full_match(sections, test_config, store=store,
                               llm_wrapper=FakeLLMWrapper())
    assert not calls
    assert _edge_map(run_again["edges"]) == _edge_map(run_store["edges"])


def test_entry_at_any_stage_equivalence(synthetic_sections_dict, test_config, monkeypatch):
    """Raw profiles, pre-extracted sections and a pre-built bundle must all
    converge to the same matching outcome."""
    monkeypatch.setattr(embed_mod, "get_embeddings",
                        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]))

    # Entry 1: pre-extracted sections
    sections = sections_from_dict(synthetic_sections_dict)
    run_sections = run_full_match(sections, test_config, llm_wrapper=FakeLLMWrapper())

    # Entry 2: pre-built embeddings bundle (skip extract/hyde/embed entirely)
    bundle = run_sections["embeddings"]
    run_bundle = run_full_match(bundle, test_config, sections=sections,
                                llm_wrapper=FakeLLMWrapper())
    assert _edge_map(run_sections["edges"]) == _edge_map(run_bundle["edges"])

    # Entry 3: raw profiles (the fake extractor echoes the raw text into every
    # section, so outcomes differ from entry 1 — just assert the flow completes
    # and produces the full result shape)
    profiles = [
        Profile(id=uid, text=" / ".join(s.values()),
                hash=hash_text(" / ".join(s.values())))
        for uid, s in synthetic_sections_dict.items()
    ]
    llm = FakeLLMWrapper()
    run_profiles = run_full_match(profiles, test_config, llm_wrapper=llm)
    assert "profile_extraction" in llm.components_called()
    for key in ("edges", "report_data", "embeddings", "llm_scores",
                "similarity", "introductions"):
        assert key in run_profiles
    assert set(run_profiles["report_data"]["user_reports"]) == set(synthetic_sections_dict)


def test_bundle_entry_requires_sections(synthetic_bundle, test_config):
    _, _, bundle = synthetic_bundle
    try:
        run_full_match(bundle, test_config, llm_wrapper=FakeLLMWrapper())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "sections" in str(exc)


def test_excluded_pairs_thread_through_full_match(synthetic_sections_dict,
                                                  test_config, monkeypatch):
    monkeypatch.setattr(embed_mod, "get_embeddings",
                        lambda texts, model: np.vstack([keyword_embed(t) for t in texts]))
    sections = sections_from_dict(synthetic_sections_dict)
    run = run_full_match(sections, test_config, llm_wrapper=FakeLLMWrapper(),
                         excluded_pairs={"alice_bob"})
    assert "alice_bob" not in {e.pair_id for e in run["edges"]}
    assert "alice_bob" not in run["llm_scores"]
