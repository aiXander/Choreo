"""Mode runners — the public orchestration API external apps import.

Three importable functions, all taking schema objects in and returning schema
objects out (callers may still chain via disk using each stage's load/dump):

  - ``run_full_match(inputs, config, ...)``   — Mode "cohort": the classic
    all-to-all run. ``inputs`` may enter at ANY stage: raw ``Profile`` objects
    (everything runs), pre-extracted ``ExtractedSections`` (skip extraction),
    or an ``EmbeddingsBundle`` (skip extraction + HyDE + embedding).
  - ``run_query_match(query, pool, config, ...)``  — Mode B (1×M), query.py.
  - ``run_batch_match(member_ids, pool, config, ...)`` — Mode C (M×N),
    batch_match.py.

Each accepts an optional ``FileStore`` for the convenience/standalone case but
never requires one; with no store everything happens in memory and the caller
persists the returned objects itself (e.g. into Neon).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from .utils import filter_active_sections
from .config import resolve_prompt_templates
from .llm import LLMWrapper
from .ingest import Profile
from .schemas import EmbeddingsBundle, ExtractedSections, sections_from_dict  # noqa: F401
from .extract import extract_sections, extract_sections_from_profiles
from .hyde import generate_hyde_descriptors
from .embed import (
    create_section_embeddings_bundle,
    embed_sections,
    supports_mrl,
    truncate_embeddings,
)
from .candidate import CandidatePair, generate_similarity_matrix
from .score import create_sections_dict, score_pairs_with_llm
from .match import create_matches
from .introduction import attach_fallback_intro, generate_introductions_for_matches
from .report import build_report_data
from .store import FileStore

# Re-exported mode runners (defined in their own modules)
from .query import run_query_match, run_query_match_json  # noqa: F401
from .batch_match import run_batch_match  # noqa: F401

FullMatchInputs = Union[Sequence[Profile], Sequence[ExtractedSections], EmbeddingsBundle]


def run_full_match(
    inputs: FullMatchInputs,
    config: Dict[str, Any],
    *,
    sections: Optional[List[ExtractedSections]] = None,
    store: Optional[FileStore] = None,
    llm_wrapper: Optional[LLMWrapper] = None,
    prompt_paths: Optional[Dict[str, str]] = None,
    excluded_pairs: Optional[Set[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the full all-to-all matching flow, entering at whatever stage
    matches the input type.

    Args:
        inputs: One of
            - ``list[Profile]`` (raw text): extract → HyDE → embed → match,
            - ``list[ExtractedSections]`` (pre-sectioned, e.g. from
              ``sections_from_dict``): HyDE → embed → match,
            - ``EmbeddingsBundle`` (pre-embedded): straight to similarity
            (requires ``sections`` for LLM scoring / intros / reports).
        config: Full pipeline config dict.
        sections: Sections per user — required only with an EmbeddingsBundle.
        store: Optional FileStore. When given, extraction/HyDE/embeddings use
            its disk caches and the refreshed embeddings are persisted back.
            Without it everything runs purely in memory.
        llm_wrapper: Optional LLM wrapper (default: store's cache dir, or none).
        prompt_paths: Optional prompt-file overrides
            ({"sections"/"scoring"/"introduction"/"hyde": path}). Inline
            prompt text in the config (``prompts.<name>_prompt_text``) takes
            precedence over paths.
        excluded_pairs: Optional pair_ids never to score/match.
        display_names: Optional {user_id: human name} map threaded into
            scoring + intro prompts (prose speaks names, score JSON stays
            keyed by id) — pass it when ids are uuids.
        force: Re-run every step, ignoring caches.

    Returns:
        Dict with: ``edges``, ``report_data``, ``embeddings`` (full-size
        bundle), ``extracted_sections``, ``introductions``, ``llm_scores``,
        ``similarity`` (dir/sym matrices + user order + component matrices),
        ``normalized_embed_scores``, ``normalized_llm_scores``, and the
        MRL-truncated ``working_embeddings``/``working_hyde`` used for the
        similarity math (handy for plots).
    """
    templates = resolve_prompt_templates(config=config, prompt_paths=prompt_paths)
    models_cfg = config.get("models", {})
    budgets = config.get("budgets", {})
    goal = config.get("instruction_prompt", {}).get("goal")
    language = config.get("instruction_prompt", {}).get("language") or ""

    if llm_wrapper is None:
        cache_dir = str(store.cache_dir) if store is not None and store.cache_dir else None
        llm_wrapper = LLMWrapper(
            cache_dir=cache_dir,
            reasoning_effort=models_cfg.get("reasoning_effort", "low"),
            max_concurrent_llm_calls=config.get("concurrency", {}).get("max_concurrent_llm_calls", 16),
        )

    # ---- Stages 2 / 2.5 / 3: extract → HyDE → embed (or skip ahead) ----------
    if isinstance(inputs, EmbeddingsBundle):
        if sections is None:
            raise ValueError(
                "run_full_match(EmbeddingsBundle) needs `sections` for LLM "
                "scoring, intros and reports."
            )
        cfg_model = models_cfg.get("embedding")
        if inputs.embedding_model and cfg_model and inputs.embedding_model != cfg_model:
            raise ValueError(
                f"Embeddings bundle was created with '{inputs.embedding_model}' "
                f"but the config asks for '{cfg_model}' — vectors are not "
                "comparable. Re-embed the bundle or fix models.embedding."
            )
        bundle = inputs
        extracted_sections = sections
    else:
        items = list(inputs)
        if not items:
            raise ValueError("No inputs provided")

        if isinstance(items[0], Profile):
            print("\n🧠 Step 2: Extracting sections with LLM...")
            try:
                if store is not None and store.processed_dir is not None:
                    extracted_sections = extract_sections_from_profiles(
                        profiles=items,
                        sections_config=templates["sections"],
                        model=models_cfg.get("extraction_llm"),
                        llm_wrapper=llm_wrapper,
                        processed_dir=str(store.processed_dir),
                        budgets=budgets,
                        goal=goal,
                        force=force,
                        language=language,
                    )
                else:
                    sections_config = filter_active_sections(templates["sections"])
                    extracted_sections = extract_sections(
                        profiles=items,
                        sections_config=sections_config,
                        model=models_cfg.get("extraction_llm"),
                        llm_wrapper=llm_wrapper,
                        goal=goal,
                        max_calls=budgets.get("extraction_llm_calls", 300),  # matches defaults/config.yaml
                        use_llm_cache=not force,
                        language=language,
                    )
                print(f"✅ Extracted sections for {len(extracted_sections)} profiles")
            except Exception as exc:
                raise RuntimeError(f"Error extracting sections: {exc}") from exc
        elif isinstance(items[0], ExtractedSections):
            extracted_sections = items
        else:
            raise TypeError(
                f"Unsupported input type {type(items[0]).__name__}: expected "
                "Profile, ExtractedSections or EmbeddingsBundle"
            )

        # HyDE generation step (only runs when cross_section_weights are configured)
        cross_section_weights = config.get("recipe", {}).get("cross_section_weights", {}) or {}
        hyde_descriptors = {}
        if cross_section_weights:
            print("\n🔮 Step 2.5: Generating HyDE descriptors...")
            try:
                hyde_descriptors = generate_hyde_descriptors(
                    extracted_sections=extracted_sections,
                    cross_section_weights=cross_section_weights,
                    hyde_config=config.get("hyde", {}),
                    prompt_template=templates["hyde"],
                    goal=goal,
                    llm_wrapper=llm_wrapper,
                    model=models_cfg.get("extraction_llm"),
                    cache_dir=Path(store.processed_dir) if store is not None and store.processed_dir else None,
                    sections_config=templates["sections"],
                    force=force,
                    language=language,
                )
                print(f"✅ Generated HyDE descriptors for {len(hyde_descriptors)} cross-section pairs")
            except Exception as exc:
                raise RuntimeError(f"Error generating HyDE descriptors: {exc}") from exc

        print("\n🔢 Step 3: Creating embeddings...")
        if store is not None and store.embeds_dir is not None:
            bundle = create_section_embeddings_bundle(
                extracted_sections=extracted_sections,
                embedding_model=models_cfg.get("embedding"),
                embeds_dir=str(store.embeds_dir),
                hyde_descriptors=hyde_descriptors if hyde_descriptors else None,
                force=force,
            )
        else:
            bundle = embed_sections(
                extracted_sections=extracted_sections,
                embedding_model=models_cfg.get("embedding"),
                hyde_descriptors=hyde_descriptors if hyde_descriptors else None,
            )
        print(f"✅ Created embeddings: {bundle.embeddings.shape}")

    # ---- MRL truncation of the working copies --------------------------------
    # Full-size vectors stay in the bundle (and on disk); only the similarity
    # math runs on truncated copies. Skipped (with a warning) on models not
    # known to be MRL-trained, since truncating those would corrupt similarity.
    working_embeddings = bundle.embeddings
    working_hyde = dict(bundle.hyde)
    embedding_dimensions = models_cfg.get("embedding_dimensions")
    embedding_model = models_cfg.get("embedding")
    if embedding_dimensions:
        if supports_mrl(embedding_model):
            working_embeddings = truncate_embeddings(working_embeddings, embedding_dimensions)
            working_hyde = {k: truncate_embeddings(v, embedding_dimensions)
                            for k, v in working_hyde.items()}
            print(f"   MRL-truncated to {embedding_dimensions} dims: {working_embeddings.shape}")
        else:
            print(f"⚠️  embedding_dimensions={embedding_dimensions} is set, but model "
                  f"'{embedding_model}' is not known to support Matryoshka (MRL) "
                  f"truncation — keeping full {working_embeddings.shape[-1]} dims. Add it to "
                  f"MRL_CAPABLE_MODELS in choreo/embed.py if it does, or unset "
                  f"embedding_dimensions to silence this warning.")

    if working_hyde:
        for k, v in working_hyde.items():
            print(f"   HyDE embeddings [{k}]: {v.shape}")

    # ---- Stage 4: similarity (square cohort path, symmetrized) ----------------
    print("\n🎯 Step 4: Generating similarity matrix...")
    try:
        dir_matrix, sym_matrix, user_ids_sorted, matrices_dict = generate_similarity_matrix(
            embeddings=working_embeddings,
            user_ids=bundle.user_ids,
            section_names=bundle.section_names,
            recipe_config=config.get("recipe", {}),
            hyde_embeddings=working_hyde if working_hyde else None,
        )
        print(f"✅ Generated similarity matrix for {len(user_ids_sorted)} users")
    except Exception as exc:
        raise RuntimeError(f"Error generating similarity matrix: {exc}") from exc

    # ---- Stage 5: LLM pair scoring ---------------------------------------------
    print("\n⚡ Step 5: LLM pair scoring...")
    sections_dict = create_sections_dict(extracted_sections)
    instruction = config.get("recipe", {}).get("instruction", "find good matches")
    unscored_pairs: List[CandidatePair] = []
    try:
        llm_scores = score_pairs_with_llm(
            similarity_matrix=sym_matrix,
            user_ids=user_ids_sorted,
            sections_dict=sections_dict,
            instruction=instruction,
            goal=goal,
            prompt_template=templates["scoring"],
            llm_wrapper=llm_wrapper,
            model=models_cfg.get("pair_llm"),
            max_n_llm_evaluations_per_profile=budgets.get("max_n_llm_evaluations_per_profile"),
            global_cap=budgets.get("max_pair_llm_calls"),
            n_profiles_to_score_together=budgets.get("n_profiles_to_score_together"),
            force=force,
            excluded_pairs=excluded_pairs,
            reasoning_effort=models_cfg.get("pair_reasoning_effort", "medium"),
            unscored_out=unscored_pairs,
            display_names=display_names,
        )
        print(f"✅ Scored {len(llm_scores)} pairs with LLM")
    except Exception as exc:
        raise RuntimeError(f"Error scoring pairs: {exc}") from exc

    # ---- Stage 6: b-matching ------------------------------------------------------
    print("\n🔗 Step 6: Greedy b-matching...")
    try:
        scored_candidates = [
            CandidatePair.create(score.user1, score.user2, score.embed_score)
            for score in llm_scores.values()
        ]
        # Selected pairs the LLM never scored stay in the candidate set with
        # their embedding-only weight (compute_final_weights blends embed-only
        # when no PairScore exists) instead of being dropped from matching.
        scored_candidates.extend(unscored_pairs)
        final_edges, normalized_embed_scores, normalized_llm_scores = create_matches(
            candidates=scored_candidates,
            llm_scores=llm_scores,
            all_user_ids=user_ids_sorted,
            matching_config=config.get("matching", {}),
            blending_config=config.get("blending", {}),
            similarity_matrix=sym_matrix,
            excluded_pairs=excluded_pairs,
        )
        print(f"✅ Created {len(final_edges)} final matches")
    except Exception as exc:
        raise RuntimeError(f"Error creating matches: {exc}") from exc

    # ---- Stage 7: introductions -----------------------------------------------------
    print("\n💬 Step 7: Generating introductions for matches...")
    try:
        introductions = generate_introductions_for_matches(
            final_edges=final_edges,
            sections_dict=sections_dict,
            instruction=instruction,
            goal=goal,
            prompt_template=templates["introduction"],
            llm_wrapper=llm_wrapper,
            model=models_cfg.get("pair_llm"),
            force=force,
            display_names=display_names,
        )
        for edge in final_edges:
            intro_obj = introductions.get(edge.pair_id)
            if intro_obj:
                edge.intro = intro_obj.intro
                edge.starter_topics = intro_obj.starter_topics
            else:
                attach_fallback_intro(edge, display_names=display_names)
        print(f"✅ Generated introductions for {len(introductions)} matches")
    except Exception as exc:
        raise RuntimeError(f"Error generating introductions: {exc}") from exc

    # ---- Stage 8: report data (returned; the adapter persists it) ----------------------
    print("\n📝 Step 8: Building report data...")
    try:
        report_data = build_report_data(
            all_edges=final_edges,
            extracted_sections=extracted_sections,
            top_matches_per_user=config.get("matching", {}).get("b_max"),
        )
    except Exception as exc:
        raise RuntimeError(f"Error generating reports: {exc}") from exc

    return {
        "edges": final_edges,
        "report_data": report_data,
        "embeddings": bundle,
        "extracted_sections": extracted_sections,
        "introductions": introductions,
        "llm_scores": llm_scores,
        "similarity": {
            "dir_matrix": dir_matrix,
            "sym_matrix": sym_matrix,
            "user_ids": user_ids_sorted,
            "matrices_dict": matrices_dict,
        },
        "normalized_embed_scores": normalized_embed_scores,
        "normalized_llm_scores": normalized_llm_scores,
        "working_embeddings": working_embeddings,
        "working_hyde": working_hyde,
    }
