"""End-to-end golden-file regression on the data/test4 fixture (LIVE, gated).

Run with::

    RUN_LLM_TESTS=1 uv run pytest tests/test_e2e_regression.py -s

Requires OPENROUTER_API_KEY in .env and the data/test4 caches (extraction,
HyDE, embeddings and the stable-keyed LLM score/intro caches). With warm
caches the run makes zero LLM calls and the outputs are byte-stable, so the
compare is strict modulo float jitter.

Golden provenance: the goldens were captured from a post-refactor cached run
that was itself validated for structural equivalence (identical match sets,
byte-identical embedding scores) against the pre-refactor baseline run.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden" / "test4"

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_LLM_TESTS"),
    reason="live-LLM e2e test; set RUN_LLM_TESTS=1 to run",
)


def _approx_equal(a, b, tol=0.05, path="$"):
    """Recursive compare, tolerant to small float jitter in scores."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ: {a.keys()} vs {b.keys()}"
        for k in a:
            _approx_equal(a[k], b[k], tol, f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: list length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _approx_equal(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        assert abs(a - b) <= tol, f"{path}: {a} vs {b}"
    elif path.endswith(".intro") or "matches" in path:
        # LLM prose can re-sample if a cache entry is missing — only require
        # presence, not byte equality.
        assert bool(a) == bool(b), f"{path}: presence differs"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def test_full_txt_run_matches_golden(tmp_path, monkeypatch):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    monkeypatch.chdir(ROOT)

    import main as main_mod
    rc = main_mod.main(group_name="test4", pipeline_name="matching")
    assert rc == 0

    new = json.loads((ROOT / "data/test4/outputs/cohort.json").read_text())
    golden = json.loads((GOLDEN_DIR / "cohort.json").read_text())

    # Structural core: same users, same match sets
    assert new["overview"] == golden["overview"]
    for user, gdata in golden["users"].items():
        ndata = new["users"][user]
        assert sorted(m["partner"] for m in ndata["matches"]) == \
               sorted(m["partner"] for m in gdata["matches"]), f"match set for {user}"
        assert ndata["degree"] == gdata["degree"]

    # Numeric: embedding stats are cache-deterministic; final weights tolerate
    # jitter from any re-sampled LLM scores.
    _approx_equal(new["score_statistics"], golden["score_statistics"], tol=0.05)

    # Per-user report files exist and carry both fields
    for user in golden["users"]:
        report = json.loads((ROOT / f"data/test4/outputs/{user}.json").read_text())
        assert set(report) == {"profile", "matches"}
        assert report["profile"]
