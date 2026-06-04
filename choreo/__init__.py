"""Choreo — a library of AI matchmaking compute.

Extracts structured sections from free-text profiles, embeds them, computes
directional "who can help whom" similarity, and refines with LLM pair scoring.
Every stage is a pure transform with a declared IO schema; persistence lives in
adapters (the bundled FileStore, or whatever store your app implements).

Public API (the names most integrations need):

    from choreo import run_full_match, run_query_match, run_batch_match
    from choreo import load_config, resolve_prompt_paths
    from choreo import EmbeddingsBundle, sections_from_dict, FileStore, Store

Plotting helpers (``choreo.tsne``, ``choreo.visualize_similarity``,
``choreo.score_correlation``) are NOT imported here — they need the optional
``choreo[plots]`` extra (matplotlib/seaborn/scikit-learn).
"""

from .config import deep_merge, load_config, resolve_prompt_paths
from .ingest import Profile, load_profiles
from .llm import LLMWrapper
from .runners import run_batch_match, run_full_match, run_query_match
from .query import run_query_match_json
from .schemas import (
    Edge,
    EmbeddingsBundle,
    ExtractedSections,
    HydeDescriptors,
    Introduction,
    PairScore,
    SimilarityResult,
    sections_from_dict,
)
from .store import FileStore, Store
from .utils import DEFAULT_PROMPT_PATHS, DEFAULTS_DIR, is_stale, stable_pair_id, utc_now_iso

__all__ = [
    "DEFAULT_PROMPT_PATHS",
    "DEFAULTS_DIR",
    "Edge",
    "EmbeddingsBundle",
    "ExtractedSections",
    "FileStore",
    "HydeDescriptors",
    "Introduction",
    "LLMWrapper",
    "PairScore",
    "Profile",
    "SimilarityResult",
    "Store",
    "deep_merge",
    "is_stale",
    "load_config",
    "load_profiles",
    "resolve_prompt_paths",
    "run_batch_match",
    "run_full_match",
    "run_query_match",
    "run_query_match_json",
    "sections_from_dict",
    "stable_pair_id",
    "utc_now_iso",
]
