"""Schema round-trip tests: to_dict/from_dict identity and dump→load fidelity.

Locks the §3.1 stage contracts and the canonical disk formats.
"""

import numpy as np

from choreo.schemas import (
    Edge,
    EmbeddingsBundle,
    ExtractedSections,
    HydeDescriptors,
    Introduction,
    PairScore,
    SimilarityResult,
    sections_from_dict,
)
from choreo import stages


def _bundle(n=3) -> EmbeddingsBundle:
    rng = np.random.default_rng(42)
    users = [f"u{i}" for i in range(n)]
    return EmbeddingsBundle(
        user_ids=users,
        section_names=["skills", "needs"],
        embeddings=rng.normal(size=(n, 2, 8)),
        hyde={"needs_skills": rng.normal(size=(n, 2, 8))},
        embedding_model="fake/model",
        dim=8,
        section_hashes={u: {"skills": f"h{u}s", "needs": f"h{u}n"} for u in users},
        hyde_hashes={"needs_skills": {u: f"hy{u}" for u in users}},
    )


def _bundles_equal(a: EmbeddingsBundle, b: EmbeddingsBundle) -> bool:
    return (
        a.user_ids == b.user_ids
        and a.section_names == b.section_names
        and np.allclose(a.embeddings, b.embeddings)
        and set(a.hyde) == set(b.hyde)
        and all(np.allclose(a.hyde[k], b.hyde[k]) for k in a.hyde)
        and a.embedding_model == b.embedding_model
        and a.dim == b.dim
        and a.section_hashes == b.section_hashes
        and a.hyde_hashes == b.hyde_hashes
    )


# ---------------------------------------------------------------------------
# to_dict / from_dict identity
# ---------------------------------------------------------------------------

def test_extracted_sections_roundtrip():
    es = ExtractedSections(id="alice", sections={"skills": "x", "needs": "y"}, hash="abc")
    assert ExtractedSections.from_dict(es.to_dict()) == es


def test_hyde_descriptors_roundtrip():
    hd = HydeDescriptors(user_id="a", source_section="needs",
                         target_section="skills", descriptors=["d1", "d2"])
    assert HydeDescriptors.from_dict(hd.to_dict()) == hd


def test_pair_score_roundtrip():
    ps = PairScore(pair_id="a_b", user1="a", user2="b", embed_score=0.5, score=0.8)
    assert PairScore.from_dict(ps.to_dict()) == ps


def test_edge_roundtrip():
    e = Edge(user1="a", user2="b", pair_id="a_b", final_weight=0.7,
             embed_score=0.5, llm_score=0.8, embed_score_normalized=0.6,
             llm_score_normalized=0.9, intro="hi", starter_topics="• t")
    e2 = Edge.from_dict(e.to_dict())
    # to_dict rounds floats to 3 decimals — compare at that precision
    assert (e2.user1, e2.user2, e2.pair_id, e2.intro, e2.starter_topics) == \
           ("a", "b", "a_b", "hi", "• t")
    assert abs(e2.final_weight - 0.7) < 1e-9


def test_introduction_roundtrip():
    intro = Introduction(pair_id="a_b", user1="a", user2="b",
                         intro="hello", starter_topics="• x")
    assert Introduction.from_dict(intro.to_dict()) == intro


def test_embeddings_bundle_roundtrip():
    b = _bundle()
    assert _bundles_equal(EmbeddingsBundle.from_dict(b.to_dict()), b)


def test_similarity_result_roundtrip():
    rng = np.random.default_rng(0)
    res = SimilarityResult(
        source_ids=["a"],
        target_ids=["a", "b"],
        dir_matrix=rng.normal(size=(1, 2)),
        sym_matrix=None,
        matrices_dict={
            "section_matrices": {"skills": rng.normal(size=(1, 2))},
            "cross_section_matrices": {"needs_skills": rng.normal(size=(1, 2))},
            "section_weights": {"skills": 0.3},
            "cross_section_weights": {"needs_skills": 0.8},
            "combined_matrix": rng.normal(size=(1, 2)),
        },
    )
    back = SimilarityResult.from_dict(res.to_dict())
    assert back.source_ids == res.source_ids and back.target_ids == res.target_ids
    assert np.allclose(back.dir_matrix, res.dir_matrix)
    assert back.sym_matrix is None
    assert np.allclose(back.matrices_dict["combined_matrix"],
                       res.matrices_dict["combined_matrix"])
    assert np.allclose(back.matrices_dict["section_matrices"]["skills"],
                       res.matrices_dict["section_matrices"]["skills"])


def test_sections_from_dict_hash_stability():
    a = sections_from_dict({"u": {"skills": "x", "needs": "y"}})[0]
    b = sections_from_dict({"u": {"needs": "y", "skills": "x"}})[0]  # key order flipped
    assert a.hash == b.hash
    c = sections_from_dict({"u": {"skills": "x!", "needs": "y"}})[0]
    assert c.hash != a.hash


def test_bundle_subset_order_and_missing():
    b = _bundle(4)
    sub = b.subset(["u2", "u0"])
    assert sub.user_ids == ["u2", "u0"]
    assert np.allclose(sub.embeddings[0], b.embeddings[2])
    assert np.allclose(sub.embeddings[1], b.embeddings[0])
    assert np.allclose(sub.hyde["needs_skills"][0], b.hyde["needs_skills"][2])
    assert sub.section_hashes["u2"] == b.section_hashes["u2"]
    try:
        b.subset(["u0", "ghost"])
        assert False, "expected KeyError"
    except KeyError as exc:
        assert "ghost" in str(exc)


# ---------------------------------------------------------------------------
# dump → load fidelity (the canonical disk formats)
# ---------------------------------------------------------------------------

def test_sections_dump_load(tmp_path):
    sections = sections_from_dict({"a": {"skills": "x"}, "b": {"skills": "y"}})
    path = tmp_path / "sections.jsonl"
    stages.dump_sections(sections, path)
    assert stages.load_sections(path) == sections


def test_hyde_dump_load(tmp_path):
    hyde = {"needs_skills": [
        HydeDescriptors("a", "needs", "skills", ["d1", "d2"]),
        HydeDescriptors("b", "needs", "skills", ["d3"]),
    ]}
    path = tmp_path / "hyde.json"
    stages.dump_hyde(hyde, path)
    assert stages.load_hyde(path) == hyde


def test_embeddings_dump_load(tmp_path):
    b = _bundle()
    b.dump(tmp_path / "embeds")
    assert _bundles_equal(EmbeddingsBundle.load(tmp_path / "embeds"), b)


def test_dump_clears_stale_hyde_file(tmp_path):
    """A hyde-less dump must remove a previous run's hyde_vectors.npz —
    otherwise load() resurrects HyDE arrays that no longer match the roster."""
    _bundle(2).dump(tmp_path / "embeds")                  # writes hyde_vectors.npz
    no_hyde = _bundle(3)
    no_hyde.hyde, no_hyde.hyde_hashes = {}, {}
    no_hyde.dump(tmp_path / "embeds")                     # hyde disabled this run
    assert not (tmp_path / "embeds" / "hyde_vectors.npz").exists()
    assert EmbeddingsBundle.load(tmp_path / "embeds").hyde == {}


def test_similarity_dump_load(tmp_path):
    rng = np.random.default_rng(1)
    res = SimilarityResult(
        source_ids=["a", "b"],
        target_ids=["a", "b"],
        dir_matrix=rng.normal(size=(2, 2)),
        sym_matrix=rng.normal(size=(2, 2)),
        matrices_dict={
            "section_matrices": {"skills": rng.normal(size=(2, 2))},
            "cross_section_matrices": {},
            "section_weights": {"skills": 0.3},
            "cross_section_weights": {},
            "combined_matrix": rng.normal(size=(2, 2)),
        },
    )
    stages.dump_similarity(res, tmp_path / "sim")
    back = stages.load_similarity(tmp_path / "sim")
    assert back.source_ids == res.source_ids
    assert np.allclose(back.dir_matrix, res.dir_matrix)
    assert np.allclose(back.sym_matrix, res.sym_matrix)
    assert np.allclose(back.matrices_dict["section_matrices"]["skills"],
                       res.matrices_dict["section_matrices"]["skills"])
    assert back.matrices_dict["section_weights"] == {"skills": 0.3}


def test_scores_edges_intros_report_dump_load(tmp_path):
    scores = {"a_b": PairScore("a_b", "a", "b", 0.5, 0.8)}
    stages.dump_scores(scores, tmp_path / "scores.json")
    assert stages.load_scores(tmp_path / "scores.json") == scores

    edges = [Edge("a", "b", "a_b", 0.7, 0.5, 0.8, 0.6, 0.9, "hi", "• t")]
    stages.dump_edges(edges, tmp_path / "edges.json")
    loaded = stages.load_edges(tmp_path / "edges.json")
    assert loaded[0].pair_id == "a_b" and loaded[0].intro == "hi"

    intros = {"a_b": Introduction("a_b", "a", "b", "hello", "• x")}
    stages.dump_introductions(intros, tmp_path / "intros.json")
    assert stages.load_introductions(tmp_path / "intros.json") == intros

    report = {"user_reports": {"a": {"profile": "p", "matches": "m"}},
              "cohort_summary": {"overview": {"total_users": 1}}}
    stages.dump_report_data(report, tmp_path / "report.json")
    assert stages.load_report_data(tmp_path / "report.json") == report


def test_stage_registry_introspection():
    names = stages.list_stages()
    assert names == ["extract", "hyde", "embed", "similarity",
                     "score", "match", "introduce", "report"]
    for name in names:
        desc = stages.describe_stage(name)
        assert desc["input_schema"] and desc["output_schema"]
        assert desc["supports_disk_chaining"]
    # JSON-serializable contract
    import json
    json.dumps(stages.describe_all_stages())
