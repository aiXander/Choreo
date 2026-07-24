"""Mode B: query match (1 × M) — the hot path.

A query is treated as a *partial profile*: a pseudo-user (``__query__``) with
only some sections populated. It drops straight into the existing directional
machinery — the per-pair fusion already treats an absent section as neutral,
not as similarity 0 — so a query with only some sections filled matches on
exactly those and ignores everything else.

The recommended query shape is an explicit section mapping authored by the
caller (e.g. an agent decomposing the ask into per-section legs) combined with
a ``recipe_override`` of same-section weights and empty cross weights — that
path embeds the legs directly against the matching pool sections and never
calls an LLM before the re-rank (no extraction, no HyDE). Raw-text queries
(extract-LLM auto-expansion) and cross-term recipes (which HyDE-expand the
filled source sections) remain supported for standalone use.

The candidate pool always comes in as an argument (an EmbeddingsBundle pulled
from whatever store the caller owns); the pool is NEVER re-embedded here. At
most the one-row query atom is embedded (and HyDE'd, if cross weights ask for
it) per call.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np

from .utils import (
    QUERY_ID,
    filter_active_sections,
    hash_text,
    is_absent,
    parse_cross_key,
    stable_pair_id,
)
from .config import resolve_prompt_templates
from .llm import LLMWrapper, run_coro_blocking
from .ingest import Profile
from .schemas import Edge, EmbeddingsBundle, ExtractedSections
from .extract import extract_sections
from .hyde import hyde_descriptors_for_sections
from .embed import embed_sections, supports_mrl, truncate_embeddings
from .candidate import generate_rectangular_similarity
from .score import build_batch_scoring_prompt, get_pair_score
from .introduction import generate_introductions_for_matches


# Re-exported from utils for backward compatibility (`from choreo.query
# import QUERY_ID` keeps working); the constant lives in utils so score.py
# can alias the query pseudo-user without a circular import.
__all__ = ["QUERY_ID", "QueryMatchResult", "build_query_atom",
           "run_query_match", "run_query_match_json"]


@dataclass
class QueryMatchResult:
    """Ranked shortlist for one query — returned, never written."""
    query_sections: Dict[str, str]
    shortlist: List[Dict[str, Any]]   # ranked: user_id, scores, intro, ...
    recipe: Dict[str, Any]
    llm_rerank_applied: bool
    pool_size: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_sections": dict(self.query_sections),
            "shortlist": list(self.shortlist),
            "recipe": dict(self.recipe),
            "llm_rerank_applied": self.llm_rerank_applied,
            "pool_size": self.pool_size,
            "notes": list(self.notes),
        }


def build_query_atom(
    query: Union[str, Dict[str, str]],
    section_names: List[str],
    *,
    sections_config: Optional[Dict[str, Any]] = None,
    llm_wrapper: Optional[LLMWrapper] = None,
    model: Optional[str] = None,
    goal: str = "",
    language: str = "",
) -> ExtractedSections:
    """Build the transient query atom: a partial ExtractedSections.

    Two supported input shapes (docs/01_todo.md §3.2):
      (a) explicit section mapping (default, cheapest): ``{"needs": "..."}``
          — no extraction LLM call; unmapped sections stay empty (= absent,
          masked as neutral downstream).
      (b) raw text: routed through the extract stage to auto-populate sections
          (requires sections_config + llm_wrapper + model).

    The atom's sections dict is constructed in ``section_names`` order so its
    embedding axes line up with the pool bundle.
    """
    if isinstance(query, dict):
        unknown = [k for k in query if k not in section_names]
        if unknown:
            print(f"Warning: query sections {unknown} not in pool sections "
                  f"{section_names} — ignoring them")
        sections = {name: (query.get(name, "") or "").strip() for name in section_names}
        if not any(sections.values()):
            raise ValueError(
                "Query maps to no known section — provide at least one of "
                f"{section_names} (e.g. {{'needs': '<query text>'}})."
            )
        return ExtractedSections(
            id=QUERY_ID,
            sections=sections,
            hash=hash_text(json.dumps(sections, ensure_ascii=False, sort_keys=True)),
        )

    # Raw text -> auto-expand via the extract stage
    if not (sections_config and llm_wrapper and model):
        raise ValueError(
            "Raw-text queries need sections_config, llm_wrapper and model for "
            "auto-expansion — or pass an explicit section mapping instead."
        )
    profile = Profile(id=QUERY_ID, text=query.strip(), hash=hash_text(query.strip()))
    extracted = extract_sections(
        profiles=[profile],
        sections_config=sections_config,
        model=model,
        llm_wrapper=llm_wrapper,
        goal=goal,
        language=language,
    )[0]
    # Re-shape to pool section order; treat "Not specified" as absent (empty)
    # so unfilled sections are masked as neutral rather than embedded literally.
    sections = {}
    for name in section_names:
        text = (extracted.sections.get(name) or "").strip()
        sections[name] = "" if is_absent(text) else text
    extracted.sections = sections
    return extracted


def _normalize_against_reference(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Min-max normalize values against a reference distribution (pool-relative
    by default; callers may pass stable cross-run reference stats instead)."""
    ref = np.asarray(reference, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return values
    ref_min, ref_max = float(ref.min()), float(ref.max())
    if ref_max <= ref_min:
        return np.full_like(values, 0.5)
    return (values - ref_min) / (ref_max - ref_min)


def _llm_rerank_query_candidates(
    query_atom: ExtractedSections,
    candidate_ids: List[str],
    pool_sections: Dict[str, Dict[str, str]],
    config: Dict[str, Any],
    llm_wrapper: LLMWrapper,
    prompt_template: str,
    display_names: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Score query↔candidate pairs with the LLM (no set-cover, no b-matching).

    Reuses the batch scoring prompt framing: each call presents the query
    profile plus a chunk of candidates and asks ONLY for the query↔candidate
    scores. Returns {candidate_id: score 0..1} for every candidate the LLM
    answered — candidates absent from the result went unscored, and
    `run_query_match` drops them rather than ranking them on embeddings alone.

    Two latency knobs, both under `query:` and both defaulting to the fast
    setting, because this runs inside a live agent tool call:

    - `rerank_max_retries` (default 0): serial re-ask rounds for candidates the
      model silently dropped from a chunk response. The wave is already
      dispatched by then, so each round is a whole extra round-trip — measured
      at ~20% of query wall clock to rescue a single candidate out of
      top_k*multiplier. The over-fetch pool already covers that gap.
    - `rerank_deadline_s` (default None): wall-clock budget for the wave.
      Cancels stragglers instead of waiting out the model's tail.
    """
    # Note: the packaged query_scoring template deliberately does NOT render
    # `{instruction}` — recipe.instruction is pair-framed flavor for the
    # cohort/batch paths, and community context reaches the query prompt via
    # `{goal}`. The kwarg is still passed for custom templates that want it.
    instruction = config.get("recipe", {}).get("instruction", "find good matches")
    goal = config.get("instruction_prompt", {}).get("goal", "")
    query_cfg = config.get("query", {}) or {}

    sections_dict = {**pool_sections, QUERY_ID: query_atom.sections}

    # Chunk candidates so each call stays at the configured batch width
    # (query + chunk = one prompt). Floor the CHUNK at 2 — flooring before the
    # -1 would degrade n_profiles_to_score_together=2 to one-candidate calls.
    chunk_size = max(2, config.get("budgets", {}).get("n_profiles_to_score_together", 5) - 1)
    max_retry_rounds = max(0, int(query_cfg.get("rerank_max_retries", 0) or 0))
    deadline_s = query_cfg.get("rerank_deadline_s")

    llm_wrapper.set_component("query_rerank")
    llm_scores: Dict[str, float] = {}
    remaining = list(candidate_ids)

    for round_idx in range(max_retry_rounds + 1):
        if not remaining:
            break
        if round_idx:
            print(f"Retrying query re-rank for {len(remaining)} unscored "
                  f"candidate(s) (retry {round_idx}/{max_retry_rounds})")

        chunks = [remaining[i:i + chunk_size] for i in range(0, len(remaining), chunk_size)]
        prompts, cache_keys, chunk_alias_maps = [], [], []
        for chunk in chunks:
            prompt, alias_of = build_batch_scoring_prompt(
                user_profiles=[QUERY_ID] + chunk,
                sections_dict=sections_dict,
                instruction=instruction,
                prompt_template=prompt_template,
                goal=goal,
                pairs=[(QUERY_ID, cand) for cand in chunk],
                display_names=display_names,
            )
            prompts.append(prompt)
            # Cache key = hash of the full prompt: covers the query content AND
            # every candidate's section content (a roster-only key would replay
            # stale scores after a profile edit). The retry round stays in the
            # key so a cached-but-incomplete response can never short-circuit
            # its own retry.
            suffix = f"_retry{round_idx}" if round_idx else ""
            cache_keys.append(f"query_score_{hash_text(prompt)}{suffix}")
            chunk_alias_maps.append(alias_of)

        try:
            responses = run_coro_blocking(llm_wrapper.batch_json_complete(
                prompts=prompts,
                model=config.get("models", {}).get("pair_llm"),
                cache_keys=cache_keys,
                reasoning_effort=config.get("models", {}).get("pair_reasoning_effort", "medium"),
                progress_label="query_rerank",
                deadline_s=deadline_s,
            ))
        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: query re-rank round failed entirely ({e})")
            continue  # the same remaining candidates go into the next round

        for chunk, alias_of, response in zip(chunks, chunk_alias_maps, responses):
            if response is None:
                # Cancelled at the deadline (or never dispatched) — an expected
                # outcome, not a failure. The chunk's candidates stay unscored.
                continue
            if isinstance(response, Exception):
                print(f"Warning: query re-rank chunk failed: {response}")
                continue
            if not isinstance(response, dict):
                print(f"Warning: query re-rank chunk returned non-dict JSON "
                      f"({type(response).__name__}); skipping")
                continue
            for cand in chunk:
                score = get_pair_score(response, QUERY_ID, cand, alias_of)
                if score is None:
                    continue
                try:
                    llm_scores[cand] = max(0.0, min(1.0, float(score)))
                except (TypeError, ValueError):
                    print(f"Warning: unparsable re-rank score for {cand}: {score!r}")
        remaining = [c for c in candidate_ids if c not in llm_scores]

    if remaining:
        print(f"No re-rank score for {len(remaining)}/{len(candidate_ids)} candidate(s) "
              f"after {max_retry_rounds} retry round(s) — they drop out of the "
              f"shortlist (the over-fetch pool covers the gap)")
    return llm_scores


def run_query_match(
    query: Union[str, Dict[str, str]],
    pool: EmbeddingsBundle,
    config: Dict[str, Any],
    *,
    pool_sections: Optional[Dict[str, Dict[str, str]]] = None,
    sections_provider: Optional[Callable[[List[str]], Dict[str, Dict[str, str]]]] = None,
    recipe_override: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    llm_rerank: Optional[bool] = None,
    generate_intros: Optional[Union[bool, int]] = None,
    exclude_ids: Optional[Set[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
    llm_wrapper: Optional[LLMWrapper] = None,
    prompt_paths: Optional[Dict[str, str]] = None,
    reference_scores: Optional[np.ndarray] = None,
) -> QueryMatchResult:
    """Rank a community pool against one transient query. Returns, never writes.

    Args:
        query: Either an explicit section mapping (``{"needs": "..."}``,
            default — no extraction call) or raw query text (auto-expanded via
            the extract stage).
        pool: Pre-built community embeddings (the caller pulls these from its
            store; never re-embedded here).
        config: Full pipeline config. ``query:`` keys provide defaults for
            top_k / llm_rerank / generate_intros / recipe /
            rerank_pool_multiplier.
        pool_sections: ``{user_id: sections}`` for the pool — required for LLM
            re-rank and intros (skipped with a note if absent). Explicit
            ``pool_sections`` wins over ``sections_provider``.
        sections_provider: Lazy fallback for ``pool_sections``: a callable
            ``(user_ids) -> {user_id: sections}`` invoked ONCE with the
            over-fetched re-rank candidate ids after the embedding cut, so the
            caller only materializes section text for the ~top_k×multiplier
            survivors instead of the whole pool. Only consulted when
            ``pool_sections`` is None and an LLM hop (re-rank or intros)
            actually needs sections.
        recipe_override: Per-call recipe (section_weights/cross_section_weights).
            Precedence: argument > config["query"]["recipe"] > config["recipe"].
        top_k: Shortlist size (default config query.top_k, else 5).
        llm_rerank: LLM re-rank. Defaults ON (config query.llm_rerank). The
            re-rank pool is over-fetched to ``top_k *
            query.rerank_pool_multiplier`` (default 3) embedding candidates so
            the LLM can *recover* good matches the embedding stage ranked just
            below the cut, not merely reorder the embedding top-K; the final
            shortlist is the re-ranked top ``top_k``.
        generate_intros: Per-candidate intro generation for the final
            shortlist. ``True`` = all shortlist rows, an int N = only the top
            N rows (cheaper hot path when the adapter renders fewer), ``False``
            = none.
        exclude_ids: Pool users to skip (e.g. the asker themself, or users
            already matched to this need). Mode-B novelty exclusion maps here:
            the adapter turns the asker's recent match history into candidate
            ids and passes them in.
        display_names: Optional {user_id: human name} map threaded into the
            re-rank and intro prompts, so prose speaks names even when ids are
            uuids. Include a ``{"__query__": <asker name>}`` entry to name the
            query side.
        llm_wrapper: Optional LLM wrapper (defaults to a cache-less one).
        prompt_paths: Optional {"sections"/"scoring"/"introduction"/"hyde": path}.
            Inline prompt text in the config (``prompts.<name>_prompt_text``)
            takes precedence over paths.
        reference_scores: Optional stable reference distribution for the
            embed-score normalization (defaults to this pool's own row values).

    Returns:
        QueryMatchResult with the ranked shortlist (id, scores, why/intro).
    """
    notes: List[str] = []
    query_cfg = config.get("query", {}) or {}
    templates = resolve_prompt_templates(config=config, prompt_paths=prompt_paths)
    language = config.get("instruction_prompt", {}).get("language") or ""

    top_k = top_k if top_k is not None else query_cfg.get("top_k", 5)
    llm_rerank = llm_rerank if llm_rerank is not None else query_cfg.get("llm_rerank", True)
    generate_intros = (
        generate_intros if generate_intros is not None
        else query_cfg.get("generate_intros", True)
    )

    recipe = recipe_override or query_cfg.get("recipe") or config.get("recipe", {})

    models_cfg = config.get("models", {})
    embedding_model = models_cfg.get("embedding")
    if pool.embedding_model and embedding_model and pool.embedding_model != embedding_model:
        raise ValueError(
            f"Pool embeddings were created with '{pool.embedding_model}' but the "
            f"config asks for '{embedding_model}' — vectors are not comparable. "
            "Re-embed the pool or fix models.embedding."
        )

    if llm_wrapper is None:
        llm_wrapper = LLMWrapper(
            cache_dir=None,
            reasoning_effort=models_cfg.get("reasoning_effort", "low"),
            max_concurrent_llm_calls=config.get("concurrency", {}).get("max_concurrent_llm_calls", 16),
        )

    # ---- 1. Build the query atom (partial profile) -------------------------
    sections_config = None
    if isinstance(query, str):
        sections_config = filter_active_sections(templates["sections"])
    query_atom = build_query_atom(
        query,
        section_names=pool.section_names,
        sections_config=sections_config,
        llm_wrapper=llm_wrapper,
        model=models_cfg.get("extraction_llm"),
        goal=config.get("instruction_prompt", {}).get("goal", ""),
        language=language,
    )

    # ---- 2. HyDE the populated source section(s) ---------------------------
    # Only cross keys whose source section the query actually filled make
    # sense; the rest are dropped for this call.
    cross_weights = {
        k: w for k, w in (recipe.get("cross_section_weights") or {}).items()
        if query_atom.sections.get(parse_cross_key(k)[0], "").strip()
    }
    dropped = set(recipe.get("cross_section_weights") or {}) - set(cross_weights)
    if dropped:
        notes.append(f"Dropped cross weights with empty query source sections: {sorted(dropped)}")

    effective_recipe = {
        "section_weights": recipe.get("section_weights", {}) or {},
        "cross_section_weights": cross_weights,
    }

    hyde_descriptors = {}
    if cross_weights:
        hyde_descriptors = hyde_descriptors_for_sections(
            extracted_sections=[query_atom],
            cross_section_weights=cross_weights,
            hyde_config=config.get("hyde", {}),
            prompt_template=templates["hyde"],
            goal=config.get("instruction_prompt", {}).get("goal", ""),
            llm_wrapper=llm_wrapper,
            model=models_cfg.get("extraction_llm"),
            sections_config=sections_config or templates["sections"],
            language=language,
        )

    # ---- 3. Embed the query atom (the pool is NEVER re-embedded) -----------
    query_bundle = embed_sections(
        extracted_sections=[query_atom],
        embedding_model=embedding_model,
        hyde_descriptors=hyde_descriptors or None,
    )

    # MRL truncation: mirror the cohort pipeline so query and pool vectors live
    # in the same working dimensionality.
    embedding_dimensions = models_cfg.get("embedding_dimensions")
    pool_working = pool
    if embedding_dimensions and supports_mrl(embedding_model):
        query_bundle.embeddings = truncate_embeddings(query_bundle.embeddings, embedding_dimensions)
        query_bundle.hyde = {k: truncate_embeddings(v, embedding_dimensions)
                             for k, v in query_bundle.hyde.items()}
        # Truncating + renormalizing the WHOLE pool is the dominant per-query
        # CPU cost and the pool is immutable between upserts — memoize the
        # working copy on the bundle object so warm adapters that reuse the
        # same instance (e.g. a host adapter's warm pool cache) pay it once.
        cached = getattr(pool, "_truncated_working_copy", None)
        if cached is not None and cached[0] == embedding_dimensions:
            pool_working = cached[1]
        else:
            pool_working = EmbeddingsBundle(
                user_ids=pool.user_ids,
                section_names=pool.section_names,
                embeddings=truncate_embeddings(pool.embeddings, embedding_dimensions),
                hyde={k: truncate_embeddings(v, embedding_dimensions) for k, v in pool.hyde.items()},
                embedding_model=pool.embedding_model,
                dim=pool.dim,
            )
            pool._truncated_working_copy = (embedding_dimensions, pool_working)

    # ---- 4. 1×M directional similarity --------------------------------------
    similarity = generate_rectangular_similarity(
        source=query_bundle,
        target=pool_working,
        recipe_config=effective_recipe,
    )
    row = similarity.dir_matrix[0].astype(float).copy()

    if reference_scores is None:
        reference_scores = row[np.isfinite(row)]

    exclude_ids = set(exclude_ids or ())
    eligible = [
        (user_id, row[j])
        for j, user_id in enumerate(pool.user_ids)
        if user_id not in exclude_ids and np.isfinite(row[j]) and row[j] > 0
    ]
    eligible.sort(key=lambda t: t[1], reverse=True)

    # Over-fetch for the re-rank: give the LLM a pool of top_k * multiplier
    # embedding candidates so it can RECOVER a good match the embedding stage
    # ranked just below the cut — a re-rank over exactly top_k candidates can
    # only reorder, never recover. The final shortlist is truncated back to
    # top_k after re-ranking; without re-rank the fetch stays at top_k.
    rerank_pool_multiplier = query_cfg.get("rerank_pool_multiplier", 3) or 1
    fetch_n = max(top_k, int(top_k * rerank_pool_multiplier)) if llm_rerank else top_k
    shortlist_pairs = eligible[:fetch_n]

    if not shortlist_pairs:
        notes.append("No pool candidate had positive overlapping signal with the query.")
        return QueryMatchResult(
            query_sections=query_atom.sections,
            shortlist=[],
            recipe=effective_recipe,
            llm_rerank_applied=False,
            pool_size=len(pool.user_ids),
            notes=notes,
        )

    candidate_ids = [u for u, _ in shortlist_pairs]

    # Lazy pool_sections: fetch section text for ONLY the over-fetched
    # candidates, and only when an LLM hop will actually read it. Explicit
    # pool_sections always wins; absent both, the hops below skip with a note.
    if (
        pool_sections is None
        and sections_provider is not None
        and (llm_rerank or generate_intros)
    ):
        pool_sections = sections_provider(candidate_ids)

    embed_scores = {u: float(s) for u, s in shortlist_pairs}
    embed_norm = dict(zip(
        candidate_ids,
        _normalize_against_reference(
            np.array([embed_scores[u] for u in candidate_ids]), reference_scores
        ).tolist(),
    ))

    # ---- 5. LLM re-rank (ON by default) -------------------------------------
    llm_scores: Dict[str, float] = {}
    rerank_applied = False
    if llm_rerank:
        if pool_sections is None:
            notes.append("llm_rerank requested but pool_sections not provided — skipped.")
        else:
            llm_scores = _llm_rerank_query_candidates(
                query_atom=query_atom,
                candidate_ids=candidate_ids,
                pool_sections=pool_sections,
                config=config,
                llm_wrapper=llm_wrapper,
                # Query re-rank gets the DIRECTIONAL template (candidate →
                # query need; no reciprocity) — the pair template's mutual
                # framing fights a query that has no skills section. Falls
                # back to the pair template for custom scoring prompts that
                # don't define a query variant (config.py resolves this).
                prompt_template=templates.get("query_scoring") or templates["scoring"],
                display_names=display_names,
            )
            rerank_applied = bool(llm_scores)
            if rerank_applied and len(candidate_ids) > top_k:
                notes.append(
                    f"Re-ranked {len(candidate_ids)} embedding candidates "
                    f"(over-fetch x{rerank_pool_multiplier}) down to top {top_k}."
                )

    blending = config.get("blending", {})
    embed_w = blending.get("embed_weight", 0.35)
    llm_w = blending.get("llm_weight", 0.65)

    # NOTE — deliberate difference from the cohort/batch paths: the raw 0..1
    # LLM score is blended here, NOT remapped through
    # utils.normalize_scores_with_reference_distribution. The cohort remap
    # compresses LLM scores into the spread of the scored candidates' embed
    # scores (an anchoring-noise damper for N×N edge selection); a query
    # shortlist's top-K embed scores are typically near-identical, so that
    # remap would collapse the LLM signal and make the re-rank a no-op. Query
    # final scores rank candidates WITHIN one shortlist — they are not
    # comparable with cohort/batch final_weight values.
    def final_score(user_id: str) -> float:
        if user_id in llm_scores:
            return embed_w * embed_norm[user_id] + llm_w * llm_scores[user_id]
        return embed_norm[user_id]

    # Drop-unscored. An unscored candidate is NOT neutral: falling back to
    # embed_norm scores it as though the LLM had fully endorsed its embedding
    # rank, so it outranks candidates the LLM saw and judged mediocre (embed
    # 0.85 unscored = 0.85, vs 0.35*0.85 + 0.65*0.5 = 0.62 scored). That
    # inverts the re-rank's entire purpose — demoting embedding-plausible but
    # actually-bad matches — and it biases hardest toward exactly the
    # candidates we skipped. So once the re-rank has run, rank ONLY scored
    # candidates; the over-fetch pool is the buffer that makes dropping the
    # rest affordable, and it's what lets `rerank_deadline_s` cancel
    # stragglers without corrupting the ranking.
    #
    # The embed-only fallback still governs when the re-rank was skipped
    # entirely, or when too few candidates came back scored to fill top_k — a
    # short shortlist is worse than an embedding-ranked one.
    scored_ids = [u for u in candidate_ids if u in llm_scores]
    unscored_n = len(candidate_ids) - len(scored_ids)
    if rerank_applied and len(scored_ids) >= top_k:
        rank_pool = scored_ids
        if unscored_n:
            notes.append(
                f"Dropped {unscored_n} unscored candidate(s); the {len(scored_ids)} "
                f"scored candidates cover top {top_k}."
            )
    else:
        rank_pool = candidate_ids
        if rerank_applied and unscored_n:
            notes.append(
                f"Only {len(scored_ids)} of {len(candidate_ids)} candidates were "
                f"re-rank scored — fewer than top_k={top_k}, so unscored candidates "
                "keep embedding-only ranking to fill the shortlist."
            )

    # Truncate the (possibly over-fetched) re-ranked pool to the final top_k.
    ranked = sorted(rank_pool, key=final_score, reverse=True)[:top_k]

    # ---- 6. Intros for the final shortlist ----------------------------------
    # generate_intros: True = all shortlist rows; int N = only the top N
    # (cheaper hot path when the adapter renders fewer); False/0 = none.
    if generate_intros is True:
        intro_ids = list(ranked)
    elif generate_intros:
        intro_ids = list(ranked[:int(generate_intros)])
    else:
        intro_ids = []

    intros = {}
    if intro_ids:
        if pool_sections is None:
            notes.append("generate_intros requested but pool_sections not provided — skipped.")
        else:
            pseudo_edges = [
                Edge(
                    user1=QUERY_ID,
                    user2=user_id,
                    pair_id=stable_pair_id(QUERY_ID, user_id),
                    final_weight=final_score(user_id),
                    embed_score=embed_scores[user_id],
                    llm_score=llm_scores.get(user_id, 0.0),
                )
                for user_id in intro_ids
            ]
            # Intro cache keys hash the full prompt (query text + candidate
            # content), so distinct queries can never collide on the __query__
            # pair id — and identical repeat queries get their intros for free.
            intros = generate_introductions_for_matches(
                final_edges=pseudo_edges,
                sections_dict={**pool_sections, QUERY_ID: query_atom.sections},
                instruction=config.get("recipe", {}).get("instruction", "find good matches"),
                goal=config.get("instruction_prompt", {}).get("goal", ""),
                prompt_template=templates["introduction"],
                llm_wrapper=llm_wrapper,
                model=models_cfg.get("pair_llm"),
                display_names=display_names,
            )

    shortlist = []
    for rank, user_id in enumerate(ranked, 1):
        intro = intros.get(stable_pair_id(QUERY_ID, user_id))
        shortlist.append({
            "rank": rank,
            "user_id": user_id,
            "score": round(final_score(user_id), 4),
            "embed_score": round(embed_scores[user_id], 4),
            "embed_score_normalized": round(embed_norm[user_id], 4),
            "llm_score": round(llm_scores[user_id], 4) if user_id in llm_scores else None,
            "intro": intro.intro if intro else "",
            "starter_topics": intro.starter_topics if intro else "",
        })

    return QueryMatchResult(
        query_sections=query_atom.sections,
        shortlist=shortlist,
        recipe=effective_recipe,
        llm_rerank_applied=rerank_applied,
        pool_size=len(pool.user_ids),
        notes=notes,
    )


def run_query_match_json(
    payload: Dict[str, Any],
    config: Dict[str, Any],
    llm_wrapper: Optional[LLMWrapper] = None,
) -> Dict[str, Any]:
    """Thin JSON-in/JSON-out wrapper for agent tool-calls.

    Payload shape::

        {
          "query": "raw text"  |  {"needs": "..."},      # required
          "store_dir": "data/<group>",                    # pool source (a)...
          "pool": {<EmbeddingsBundle.to_dict()>},         # ...or inline (b)
          "pool_sections": {user_id: {section: text}},    # optional with (b)
          "top_k": 5, "llm_rerank": true,                 # optional overrides
          "generate_intros": true | <int top-N>,
          "recipe_override": {...}, "exclude_ids": [...],
          "display_names": {user_id: "Name", "__query__": "Asker"}
        }

    With ``store_dir`` the pool bundle + sections are read via FileStore (the
    standalone path). A production caller (Neon adapter) passes the pool
    inline instead. Returns ``QueryMatchResult.to_dict()`` (plus
    ``success: false`` + ``error`` on failure).
    """
    from .store import FileStore  # local import to keep module deps one-way

    try:
        if "query" not in payload:
            raise ValueError("payload must contain 'query'")

        pool_sections = payload.get("pool_sections")
        pool_val = payload.get("pool")
        if pool_val is not None:
            # Inline pool: either the serialized dict shape or an
            # already-constructed bundle (lets warm adapters skip the
            # to_dict/from_dict round-trip for warm pool caches).
            pool = (pool_val if isinstance(pool_val, EmbeddingsBundle)
                    else EmbeddingsBundle.from_dict(pool_val))
        elif payload.get("store_dir"):
            fstore = FileStore(payload["store_dir"])
            pool = fstore.get_embeddings()
            if pool_sections is None:
                pool_sections = {s.id: s.sections for s in fstore.get_sections()}
        else:
            raise ValueError("payload must contain either 'store_dir' or 'pool'")

        result = run_query_match(
            query=payload["query"],
            pool=pool,
            config=config,
            pool_sections=pool_sections,
            recipe_override=payload.get("recipe_override"),
            top_k=payload.get("top_k"),
            llm_rerank=payload.get("llm_rerank"),
            generate_intros=payload.get("generate_intros"),
            exclude_ids=set(payload.get("exclude_ids") or ()),
            display_names=payload.get("display_names"),
            llm_wrapper=llm_wrapper,
        )
        return {"success": True, **result.to_dict()}
    except Exception as exc:  # pylint: disable=broad-except
        return {"success": False, "error": str(exc)}
