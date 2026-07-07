"""Shared fixtures: deterministic fake embedder + fake LLM wrapper.

The suite is offline-first: every stage runs against these fakes, so
`uv run pytest` needs no API key. Live-LLM tests are gated behind
RUN_LLM_TESTS=1 (see test_e2e_regression.py).
"""

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))          # for `import main` / `import choreo` from the checkout


DIM = 32

# Keyword -> basis-vector index. Texts containing a keyword embed to that basis
# vector, so cross-section cosine is exactly 1.0 for a keyword match and 0.0
# otherwise — making ranking assertions deterministic.
KEYWORDS = ["AGENTS", "VISUALS", "MUSIC", "FOOD", "ROBOTS"]


def keyword_embed(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding: keyword basis vector or content hash."""
    if not text or not text.strip():
        return np.zeros(DIM)
    for idx, kw in enumerate(KEYWORDS):
        if kw in text:
            v = np.zeros(DIM)
            v[idx] = 1.0
            return v
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM)
    return v / np.linalg.norm(v)


@pytest.fixture
def fake_embed_fn():
    """Embedding callable matching embed_fn's contract; records every call."""
    calls = []

    def fn(texts):
        calls.append(list(texts))
        return np.vstack([keyword_embed(t) for t in texts])

    fn.calls = calls
    return fn


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch):
    """Run from the repo root so relative data paths (tmp stores etc.) resolve."""
    monkeypatch.chdir(ROOT)


# ---------------------------------------------------------------------------
# Fake LLM wrapper
# ---------------------------------------------------------------------------

SECTION_NAMES = ["skills", "vision", "project", "needs"]


def default_responder(component: str, prompt: str):
    """Canned, prompt-derived JSON per pipeline component."""
    if component == "profile_extraction":
        m = re.search(r"<profile>\s*(.*?)\s*</profile>", prompt, re.S)
        text = m.group(1) if m else "unknown"
        # Echo the raw profile text into every section so keyword vectors
        # survive extraction.
        return {s: f"{s} {text}" for s in SECTION_NAMES}

    if component == "hyde_generation":
        m = re.search(r"<source>\s*(.*?)\s*</source>", prompt, re.S)
        source = m.group(1) if m else "unknown"
        n = int(re.search(r"Generate (\d+) descriptor", prompt).group(1))
        # Echo the source text so its keyword carries into the HyDE embedding;
        # extra descriptors get a suffix (still containing the keyword).
        return {"descriptors": [f"{source} (variant {i})" for i in range(n)]}

    if component in ("batch_pair_scoring", "query_rerank"):
        # Pair keys come from the json.dumps format hint: {"a_b": "0..1", ...}
        keys = re.findall(r'"([^"]+)": "0\.\.1"', prompt)
        # Deterministic per-pair score in [0.30, 0.90]
        return {
            k: 0.30 + (int(hashlib.sha256(k.encode()).hexdigest()[:4], 16) % 61) / 100.0
            for k in keys
        }

    if component == "introduction_generation":
        # Captures the display name (or id) from the profile headers — display
        # names may contain spaces.
        m = re.findall(r"Profile of ([^:\n]+):", prompt)
        a, b = (m + ["A", "B"])[:2]
        return {
            "intro_for_a": f"intro for {a} about {b}",
            "intro_for_b": f"intro for {b} about {a}",
            "starter_topics": "• topic one • topic two",
        }

    raise AssertionError(f"FakeLLMWrapper: unexpected component '{component}'")


class FakeLLMWrapper:
    """Mimics llm.LLMWrapper.batch_json_complete with canned JSON responses."""

    def __init__(self, responder=default_responder):
        self.responder = responder
        self.cache_dir = None
        self.call_count = 0
        self.calls = []          # (component, n_prompts) per batch call
        self.prompts_seen = []   # (component, prompt) per individual prompt
        self.component = None
        self.reasoning_effort = "low"

    def set_component(self, component):
        self.component = component

    async def batch_json_complete(self, prompts, model=None, cache_keys=None,
                                  schema_hints=None, **kwargs):
        self.calls.append((self.component, len(prompts)))
        out = []
        for prompt in prompts:
            self.call_count += 1
            self.prompts_seen.append((self.component, prompt))
            out.append(self.responder(self.component, prompt))
        return out

    def get_stats(self):
        return {"total_calls": self.call_count}

    def components_called(self):
        return [c for c, _ in self.calls]


@pytest.fixture
def fake_llm():
    return FakeLLMWrapper()


# ---------------------------------------------------------------------------
# Synthetic 4-user fixture (keyword-engineered natural pairs)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sections_dict():
    """4 users whose needs↔skills keywords pair alice↔bob and carol↔david."""
    return {
        "alice": {
            "skills": "AGENTS engineering and backend infrastructure",
            "vision": "build useful tools",
            "project": "an agent PLATFORM thing",
            "needs": "VISUALS for my live show",
        },
        "bob": {
            "skills": "VISUALS projection mapping and shaders",
            "vision": "audiovisual art everywhere",
            "project": "a touring AV show",
            "needs": "AGENTS backend help for interactivity",
        },
        "carol": {
            "skills": "MUSIC composition and sound design",
            "vision": "sound in public space",
            "project": "a generative radio",
            "needs": "FOOD pop-up partner for events",
        },
        "david": {
            "skills": "FOOD fermentation and pop-up catering",
            "vision": "local food culture",
            "project": "a supper club",
            "needs": "MUSIC for dinner events",
        },
    }


@pytest.fixture
def test_config():
    """Minimal config mirroring config.yaml's shape, fake-model friendly."""
    return {
        "models": {
            "embedding": "fake/embedding-model",
            "embedding_dimensions": None,   # fake model is not MRL-capable
            "extraction_llm": "fake/llm",
            "pair_llm": "fake/llm",
            "reasoning_effort": "low",
        },
        "instruction_prompt": {"goal": "test goal: match people"},
        "budgets": {
            "extraction_llm_calls": 100,
            "max_pair_llm_calls": 50,
            "max_n_llm_evaluations_per_profile": 8,
            "n_profiles_to_score_together": 4,
        },
        "hyde": {"n_descriptors": 1},
        "recipe": {
            "instruction": "score collaboration potential",
            "section_weights": {"vision": 0.2},
            "cross_section_weights": {"needs_skills": 0.8},
        },
        "blending": {"embed_weight": 0.35, "llm_weight": 0.65},
        "matching": {
            "b_min": 1,
            "b_max": 2,
            "min_profiles_required": 2,
            "pool_b_max": None,
            "novelty_window_months": 6,
        },
        "query": {
            "top_k": 3,
            "llm_rerank": True,
            "generate_intros": True,
            "recipe": {
                "section_weights": {},
                "cross_section_weights": {"needs_skills": 1.0},
            },
        },
    }


def build_pool(sections_dict, fake_llm, fake_embed_fn,
               cross_weights=None, n_descriptors=1):
    """Sections + HyDE + embeddings bundle for an arbitrary synthetic pool."""
    from choreo.schemas import sections_from_dict
    from choreo.hyde import hyde_descriptors_for_sections
    from choreo.embed import embed_sections
    from choreo.utils import load_yaml, DEFAULT_PROMPT_PATHS

    sections = sections_from_dict(sections_dict)
    hyde = hyde_descriptors_for_sections(
        extracted_sections=sections,
        cross_section_weights=cross_weights or {"needs_skills": 0.8},
        hyde_config={"n_descriptors": n_descriptors},
        prompt_template=load_yaml(DEFAULT_PROMPT_PATHS["hyde"])["hyde_generation"],
        goal="test",
        llm_wrapper=fake_llm,
        model="fake/llm",
    )
    bundle = embed_sections(
        extracted_sections=sections,
        embedding_model="fake/embedding-model",
        hyde_descriptors=hyde,
        embed_fn=fake_embed_fn,
    )
    return sections, hyde, bundle


@pytest.fixture
def synthetic_bundle(synthetic_sections_dict, fake_embed_fn, fake_llm, test_config):
    """Sections + HyDE + embeddings for the synthetic 4-user pool."""
    return build_pool(
        synthetic_sections_dict, fake_llm, fake_embed_fn,
        cross_weights=test_config["recipe"]["cross_section_weights"],
        n_descriptors=test_config["hyde"]["n_descriptors"],
    )
