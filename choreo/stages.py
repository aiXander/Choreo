"""Stage registry: every pipeline stage with its declared IO schema.

The central organizing idea (docs/01_todo.md §3.1): each stage is a pure
transform with a fixed, discoverable input/output contract. An external caller
(e.g. the community platform's Neon adapter) never has to read Choreo's source
to learn what a stage needs — it calls ``describe_stage("similarity")``,
formats its data into that shape, invokes the stage, and consumes the declared
output. Data can be handed over in-memory (Python objects) or via disk using
each stage's ``load``/``dump`` helpers — both chaining styles are first-class.

The schemas here are intentionally lightweight, JSON-serializable descriptions
of the dataclasses in ``schemas.py`` (the actual typed currency).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .utils import ensure_dir, load_json, save_json, load_jsonl, save_jsonl
from .schemas import (
    Edge,
    EmbeddingsBundle,
    ExtractedSections,
    HydeDescriptors,
    Introduction,
    PairScore,
    SimilarityResult,
)
from . import extract as extract_mod
from . import hyde as hyde_mod
from . import embed as embed_mod
from . import candidate as candidate_mod
from . import score as score_mod
from . import match as match_mod
from . import introduction as introduction_mod
from . import report as report_mod


# ---------------------------------------------------------------------------
# Disk helpers per stage (the canonical formats, used by FileStore/adapters)
# ---------------------------------------------------------------------------

PathLike = Union[str, Path]


def dump_sections(sections: List[ExtractedSections], path: PathLike) -> None:
    save_jsonl([s.to_dict() for s in sections], path)


def load_sections(path: PathLike) -> List[ExtractedSections]:
    return [ExtractedSections.from_dict(item) for item in load_jsonl(path)]


def dump_hyde(hyde: Dict[str, List[HydeDescriptors]], path: PathLike) -> None:
    save_json({k: [hd.to_dict() for hd in v] for k, v in hyde.items()}, path)


def load_hyde(path: PathLike) -> Dict[str, List[HydeDescriptors]]:
    data = load_json(path)
    return {k: [HydeDescriptors.from_dict(d) for d in v] for k, v in data.items()}


def dump_embeddings(bundle: EmbeddingsBundle, path: PathLike) -> None:
    bundle.dump(path)  # directory format (vectors.npz + side-cars)


def load_embeddings(path: PathLike) -> EmbeddingsBundle:
    return EmbeddingsBundle.load(path)


def dump_similarity(result: SimilarityResult, path: PathLike) -> None:
    """Directory format: matrices in similarity.npz, ids/weights in meta json."""
    out = ensure_dir(path)
    arrays: Dict[str, np.ndarray] = {"dir_matrix": result.dir_matrix}
    if result.sym_matrix is not None:
        arrays["sym_matrix"] = result.sym_matrix
    md = result.matrices_dict or {}
    for name, mat in md.get("section_matrices", {}).items():
        arrays[f"section__{name}"] = mat
    for key, mat in md.get("cross_section_matrices", {}).items():
        arrays[f"cross__{key}"] = mat
    if "combined_matrix" in md:
        arrays["combined"] = md["combined_matrix"]
    np.savez_compressed(out / "similarity.npz", **arrays)
    save_json(
        {
            "source_ids": result.source_ids,
            "target_ids": result.target_ids,
            "section_weights": md.get("section_weights", {}),
            "cross_section_weights": md.get("cross_section_weights", {}),
        },
        out / "similarity_meta.json",
    )


def load_similarity(path: PathLike) -> SimilarityResult:
    base = Path(path)
    meta = load_json(base / "similarity_meta.json")
    with np.load(base / "similarity.npz") as data:
        arrays = {k: data[k] for k in data.files}
    matrices_dict: Dict[str, Any] = {
        "section_matrices": {
            k[len("section__"):]: v for k, v in arrays.items() if k.startswith("section__")
        },
        "cross_section_matrices": {
            k[len("cross__"):]: v for k, v in arrays.items() if k.startswith("cross__")
        },
        "section_weights": meta.get("section_weights", {}),
        "cross_section_weights": meta.get("cross_section_weights", {}),
    }
    if "combined" in arrays:
        matrices_dict["combined_matrix"] = arrays["combined"]
    return SimilarityResult(
        source_ids=meta["source_ids"],
        target_ids=meta["target_ids"],
        dir_matrix=arrays["dir_matrix"],
        sym_matrix=arrays.get("sym_matrix"),
        matrices_dict=matrices_dict,
    )


def dump_scores(scores: Dict[str, PairScore], path: PathLike) -> None:
    save_json({k: v.to_dict() for k, v in scores.items()}, path)


def load_scores(path: PathLike) -> Dict[str, PairScore]:
    return {k: PairScore.from_dict(v) for k, v in load_json(path).items()}


def dump_edges(edges: List[Edge], path: PathLike) -> None:
    save_json([e.to_dict() for e in edges], path)


def load_edges(path: PathLike) -> List[Edge]:
    return [Edge.from_dict(item) for item in load_json(path)]


def dump_introductions(intros: Dict[str, Introduction], path: PathLike) -> None:
    save_json({k: v.to_dict() for k, v in intros.items()}, path)


def load_introductions(path: PathLike) -> Dict[str, Introduction]:
    return {k: Introduction.from_dict(v) for k, v in load_json(path).items()}


def dump_report_data(report_data: Dict[str, Any], path: PathLike) -> None:
    save_json(report_data, path)


def load_report_data(path: PathLike) -> Dict[str, Any]:
    return load_json(path)


# ---------------------------------------------------------------------------
# Stage specs
# ---------------------------------------------------------------------------

@dataclass
class StageSpec:
    """A pipeline stage: pure transform + declared IO schema + disk helpers."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    run: Callable[..., Any]
    load: Optional[Callable[[PathLike], Any]] = None
    dump: Optional[Callable[[Any, PathLike], None]] = None
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


_SECTIONS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "ExtractedSections",
        "fields": {"id": "str", "sections": "{section_name: text}", "hash": "str (content hash)"},
    },
}

_HYDE_SCHEMA = {
    "type": "object",
    "description": "{cross_key: [HydeDescriptors per user, same order as sections]}",
    "values": {
        "type": "HydeDescriptors",
        "fields": {
            "user_id": "str",
            "source_section": "str",
            "target_section": "str",
            "descriptors": "list[str] (n_descriptors HyDE phrasings)",
        },
    },
}

_EMBEDDINGS_SCHEMA = {
    "type": "EmbeddingsBundle",
    "fields": {
        "user_ids": "list[str] (row order)",
        "section_names": "list[str] (column order)",
        "embeddings": "float[n_users, n_sections, dim] (full native size)",
        "hyde": "{cross_key: float[n_users, n_descriptors, dim]}",
        "embedding_model": "str (provenance)",
        "dim": "int (native dim)",
        "section_hashes": "{user_id: {section: content_hash}}",
        "hyde_hashes": "{cross_key: {user_id: content_hash}}",
    },
}

_SIMILARITY_SCHEMA = {
    "type": "SimilarityResult",
    "fields": {
        "source_ids": "list[str]",
        "target_ids": "list[str]",
        "dir_matrix": "float[n_source, n_target] — dir[i][j] = how well target j helps source i",
        "sym_matrix": "float[n,n] | null — only for square cohort runs",
        "matrices_dict": "per-section + cross-section component matrices and weights",
    },
}

_SCORES_SCHEMA = {
    "type": "object",
    "description": "{pair_id: PairScore}",
    "values": {
        "type": "PairScore",
        "fields": {"pair_id": "str", "user1": "str", "user2": "str",
                   "embed_score": "float", "score": "float (0..1 LLM)"},
    },
}

_EDGES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "Edge",
        "fields": {
            "user1": "str", "user2": "str", "pair_id": "str",
            "final_weight": "float (blended)", "embed_score": "float",
            "llm_score": "float", "embed_score_normalized": "float|null",
            "llm_score_normalized": "float|null", "intro": "str",
            "starter_topics": "str",
        },
    },
}

_INTROS_SCHEMA = {
    "type": "object",
    "description": "{pair_id: Introduction}",
    "values": {
        "type": "Introduction",
        "fields": {"pair_id": "str", "user1": "str", "user2": "str",
                   "intro": "str (directional, both sides)", "starter_topics": "str"},
    },
}

_REPORT_SCHEMA = {
    "type": "object",
    "fields": {
        "user_reports": "{user_id: {profile: markdown, matches: markdown}}",
        "cohort_summary": "overview/degree_distribution/score_statistics/users",
    },
}


STAGE_REGISTRY: Dict[str, StageSpec] = {}


def _register(spec: StageSpec) -> None:
    STAGE_REGISTRY[spec.name] = spec


_register(StageSpec(
    name="extract",
    description="LLM section extraction from raw profile text.",
    input_schema={
        "profiles": {"type": "array", "items": {"type": "Profile", "fields": {
            "id": "str", "text": "raw profile text", "hash": "str (content hash)"}}},
        "sections_config": "active-filtered section config (guidelines, max_words, prompt template)",
        "existing": "optional {profile_hash: sections_dict} reuse map — cached profiles skip the LLM",
    },
    output_schema=_SECTIONS_SCHEMA,
    run=extract_mod.extract_sections,
    load=load_sections,
    dump=dump_sections,
    notes="Entry-at-stage alternative: schemas.sections_from_dict({user_id: {section: text}}) "
          "bypasses extraction entirely for pre-sectioned input.",
))

_register(StageSpec(
    name="hyde",
    description="HyDE descriptors: rewrite source sections into target vocabulary.",
    input_schema={
        "extracted_sections": _SECTIONS_SCHEMA,
        "cross_section_weights": "{'<source>_<target>': weight} — one HyDE set per key",
        "hyde_config": "{n_descriptors: int}",
        "existing": "optional {cross_key: {cache_key: descriptors}} reuse map",
    },
    output_schema=_HYDE_SCHEMA,
    run=hyde_mod.hyde_descriptors_for_sections,
    load=load_hyde,
    dump=dump_hyde,
))

_register(StageSpec(
    name="embed",
    description="Embed sections + HyDE descriptors; content-hash incremental reuse.",
    input_schema={
        "extracted_sections": _SECTIONS_SCHEMA,
        "embedding_model": "str (OpenRouter slug)",
        "hyde_descriptors": _HYDE_SCHEMA,
        "existing": "optional EmbeddingsBundle — only changed/new cells are re-embedded",
    },
    output_schema=_EMBEDDINGS_SCHEMA,
    run=embed_mod.embed_sections,
    load=load_embeddings,
    dump=dump_embeddings,
    notes="Embedding always happens inside this repo; external stores only hold "
          "the bundle and hand it back as `existing`.",
))

_register(StageSpec(
    name="similarity",
    description="Rectangular fused similarity: source set × target set (directional).",
    input_schema={
        "source": _EMBEDDINGS_SCHEMA,
        "target": _EMBEDDINGS_SCHEMA,
        "recipe_config": "{section_weights, cross_section_weights}",
    },
    output_schema=_SIMILARITY_SCHEMA,
    run=candidate_mod.generate_rectangular_similarity,
    load=load_similarity,
    dump=dump_similarity,
    notes="Square cohort mode = pass the same bundle twice (reduces exactly to "
          "the legacy behavior; symmetrization only happens on the legacy path).",
))

_register(StageSpec(
    name="score",
    description="Batched, budgeted LLM pair scoring.",
    input_schema={
        "similarity_matrix": "float[n,n] symmetric (or caller-selected pairs via selected_pairs)",
        "user_ids": "list[str]",
        "sections_dict": "{user_id: sections}",
        "instruction / goal / prompt_template": "scoring prompt inputs",
        "budgets": "max_n_llm_evaluations_per_profile, global_cap, n_profiles_to_score_together",
        "excluded_pairs": "optional set[pair_id] — never scored (novelty)",
    },
    output_schema=_SCORES_SCHEMA,
    run=score_mod.score_pairs_with_llm,
    load=load_scores,
    dump=dump_scores,
))

_register(StageSpec(
    name="match",
    description="Greedy b-matching on blended embed+LLM scores.",
    input_schema={
        "candidates": "list[CandidatePair]",
        "llm_scores": _SCORES_SCHEMA,
        "all_user_ids": "list[str]",
        "matching_config": "{b_min, b_max} (+ member_ids/pool_b_max for asymmetric subset mode)",
        "blending_config": "{embed_weight, llm_weight}",
        "excluded_pairs": "optional set[pair_id]",
    },
    output_schema=_EDGES_SCHEMA,
    run=match_mod.create_matches,
    load=load_edges,
    dump=dump_edges,
))

_register(StageSpec(
    name="introduce",
    description="Directional introductions + starter topics per matched pair.",
    input_schema={
        "final_edges": _EDGES_SCHEMA,
        "sections_dict": "{user_id: sections}",
        "instruction / goal / prompt_template": "introduction prompt inputs",
    },
    output_schema=_INTROS_SCHEMA,
    run=introduction_mod.generate_introductions_for_matches,
    load=load_introductions,
    dump=dump_introductions,
))

_register(StageSpec(
    name="report",
    description="Per-user report data + cohort summary (returned, not written).",
    input_schema={
        "all_edges": _EDGES_SCHEMA,
        "extracted_sections": _SECTIONS_SCHEMA,
        "top_matches_per_user": "int",
        "scope_user_ids": "optional list[str] — which users get reports (batch mode: members only)",
    },
    output_schema=_REPORT_SCHEMA,
    run=report_mod.build_report_data,
    load=load_report_data,
    dump=dump_report_data,
))


# ---------------------------------------------------------------------------
# Introspection API
# ---------------------------------------------------------------------------

def list_stages() -> List[str]:
    """Names of all registered stages, in pipeline order."""
    return list(STAGE_REGISTRY.keys())


def get_stage(name: str) -> StageSpec:
    try:
        return STAGE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown stage '{name}'. Available: {', '.join(STAGE_REGISTRY)}"
        ) from exc


def describe_stage(name: str) -> Dict[str, Any]:
    """JSON-serializable description of a stage's IO contract."""
    spec = get_stage(name)
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
        "output_schema": spec.output_schema,
        "notes": spec.notes,
        "supports_disk_chaining": spec.load is not None and spec.dump is not None,
    }


def describe_all_stages() -> Dict[str, Dict[str, Any]]:
    return {name: describe_stage(name) for name in STAGE_REGISTRY}


if __name__ == "__main__":
    print(json.dumps(describe_all_stages(), indent=2))
