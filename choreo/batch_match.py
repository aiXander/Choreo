"""Mode C: subset batch match (M members × N pool) with novelty exclusions.

Runs the full matching machinery — directional similarity, budgeted LLM pair
scoring, blended b-matching, intros, reports — but over an explicit member
subset as the "who needs matches" side against the full community pool as the
candidate side, excluding pairs already surfaced in prior runs.

Everything is caller-supplied: the member list, the pool embeddings, the pool
sections and the ``excluded_pairs`` history set. Choreo never reads a ``tier``
flag, never decides who is a member, and never owns the match history — the
adapter builds ``excluded_pairs`` from its own store (FileStore reads
``match_history.jsonl`` honoring ``matching.novelty_window_months``; an
external app applies the same window to its ``past_matches`` table).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .utils import DEFAULT_PROMPT_PATHS, stable_pair_id
from .llm import LLMWrapper
from .schemas import Edge, EmbeddingsBundle, ExtractedSections, Introduction
from .candidate import CandidatePair, generate_rectangular_similarity
from .embed import supports_mrl, truncate_embeddings
from .score import score_pairs_with_llm, create_sections_dict  # noqa: F401
from .match import create_matches
from .introduction import attach_fallback_intro, generate_introductions_for_matches
from .report import build_report_data


@dataclass
class BatchMatchResult:
    """Outcome of one subset batch run — returned, persistence is the caller's."""
    edges: List[Edge]
    report_data: Dict[str, Any]           # user_reports (members only) + cohort_summary
    new_pairs: List[Dict[str, Any]]       # pairs surfaced THIS run, for history append
    member_ids: List[str]
    excluded_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "report_data": self.report_data,
            "new_pairs": list(self.new_pairs),
            "member_ids": list(self.member_ids),
            "excluded_count": self.excluded_count,
        }


def select_pairs_rectangular(
    dir_matrix: np.ndarray,
    member_ids: List[str],
    pool_ids: List[str],
    max_n_llm_evaluations_per_profile: int,
    global_cap: int,
    excluded_pairs: Optional[Set[str]] = None,
) -> List[CandidatePair]:
    """Greedy per-member pair selection over a rectangular member×pool matrix.

    The rectangular counterpart of ``select_pairs_for_llm_scoring_optimal``:
    candidate pairs are restricted to (member × pool), self-pairs and
    history-excluded pairs are skipped, and each pair gets ONE score — when
    both users are members the two directional entries are averaged, otherwise
    the single member→pool direction is used.

    Selection round-robins over members so every member gets its best
    available pairs before anyone exhausts the budget; per-profile caps apply
    to every participant (members and pool users alike).
    """
    excluded_pairs = excluded_pairs or set()

    # One score per unordered pair: collect available directional values.
    pair_values: Dict[str, List[float]] = {}
    pair_users: Dict[str, tuple] = {}
    for i, member in enumerate(member_ids):
        for j, candidate in enumerate(pool_ids):
            if candidate == member:
                continue  # self-pair
            value = dir_matrix[i, j]
            if not np.isfinite(value):
                continue
            pair_id = stable_pair_id(member, candidate)
            if pair_id in excluded_pairs:
                continue
            pair_values.setdefault(pair_id, []).append(float(value))
            pair_users[pair_id] = (min(member, candidate), max(member, candidate))

    all_pairs: List[CandidatePair] = []
    for pair_id, values in pair_values.items():
        score = sum(values) / len(values)
        if score > 0:  # only positive similarities (consistent with square mode)
            user1, user2 = pair_users[pair_id]
            all_pairs.append(CandidatePair(
                user1=user1, user2=user2, similarity_score=score, pair_id=pair_id,
            ))

    all_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
    print(f"Rectangular selection: {len(all_pairs)} eligible member×pool pairs "
          f"({len(excluded_pairs)} pair ids excluded by history)")

    # Round-robin greedy over members, mirroring the square-mode behavior.
    selected: List[CandidatePair] = []
    used: Set[str] = set()
    counts: Dict[str, int] = {}

    member_order = list(member_ids)
    idx = 0
    consecutive_skips = 0
    while consecutive_skips < len(member_order) and len(selected) < global_cap:
        member = member_order[idx]
        idx = (idx + 1) % len(member_order)

        if counts.get(member, 0) >= max_n_llm_evaluations_per_profile:
            consecutive_skips += 1
            continue

        best = None
        for pair in all_pairs:
            if pair.pair_id in used:
                continue
            if member not in (pair.user1, pair.user2):
                continue
            other = pair.user2 if pair.user1 == member else pair.user1
            if counts.get(other, 0) >= max_n_llm_evaluations_per_profile:
                continue
            best = pair
            break

        if best is None:
            consecutive_skips += 1
            continue

        selected.append(best)
        used.add(best.pair_id)
        counts[best.user1] = counts.get(best.user1, 0) + 1
        counts[best.user2] = counts.get(best.user2, 0) + 1
        consecutive_skips = 0

    member_counts = [counts.get(m, 0) for m in member_order]
    print(f"Selected {len(selected)}/{len(all_pairs)} pairs for LLM scoring "
          f"(per-member evals min/max: {min(member_counts) if member_counts else 0}"
          f"/{max(member_counts) if member_counts else 0})")
    return selected


def run_batch_match(
    member_ids: List[str],
    pool: EmbeddingsBundle,
    config: Dict[str, Any],
    excluded_pairs: Optional[Set[str]] = None,
    *,
    pool_sections: Dict[str, Dict[str, str]],
    llm_wrapper: Optional[LLMWrapper] = None,
    prompt_paths: Optional[Dict[str, str]] = None,
) -> BatchMatchResult:
    """Match an explicit member subset against the community pool, novel pairs only.

    Args:
        member_ids: The "who needs matches" side (must all be present in
            ``pool`` — members are community users too). Caller-supplied,
            always: Choreo never decides who is a member.
        pool: Community embeddings bundle (the candidate side; never
            re-embedded here).
        config: Full pipeline config. Degree targets bind on members
            (``matching.b_min``/``b_max``); pool users get the optional looser
            ``matching.pool_b_max`` cap.
        excluded_pairs: pair_ids already surfaced in prior runs (the novelty
            input — built by the adapter from its match history, honoring
            ``matching.novelty_window_months``).
        pool_sections: ``{user_id: sections}`` for everyone in the pool
            (needed for LLM scoring, intros and reports).
        llm_wrapper: Optional LLM wrapper (defaults to a cache-less one).
        prompt_paths: Optional prompt-file overrides.

    Returns:
        BatchMatchResult: final edges, report data for the members only, and
        the new pairs surfaced this run (for the caller to append to its
        history store).
    """
    excluded_pairs = set(excluded_pairs or ())
    prompt_paths = {**DEFAULT_PROMPT_PATHS, **(prompt_paths or {})}

    if not member_ids:
        raise ValueError("member_ids is empty — nothing to match")
    # Dedupe (order-preserving): a duplicated member id would create duplicate
    # similarity rows, double-counting that member in the reference
    # distribution below and skewing every normalized score.
    member_ids = list(dict.fromkeys(member_ids))
    members_bundle = pool.subset(member_ids)  # raises on unknown members

    models_cfg = config.get("models", {})
    if llm_wrapper is None:
        llm_wrapper = LLMWrapper(
            cache_dir=None,
            reasoning_effort=models_cfg.get("reasoning_effort", "low"),
            max_concurrent_llm_calls=config.get("concurrency", {}).get("max_concurrent_llm_calls", 16),
        )

    # MRL truncation working copies (mirrors the cohort pipeline)
    embedding_model = models_cfg.get("embedding")
    embedding_dimensions = models_cfg.get("embedding_dimensions")
    pool_working, members_working = pool, members_bundle
    if embedding_dimensions and supports_mrl(embedding_model):
        def _truncate(b: EmbeddingsBundle) -> EmbeddingsBundle:
            return EmbeddingsBundle(
                user_ids=b.user_ids,
                section_names=b.section_names,
                embeddings=truncate_embeddings(b.embeddings, embedding_dimensions),
                hyde={k: truncate_embeddings(v, embedding_dimensions) for k, v in b.hyde.items()},
                embedding_model=b.embedding_model,
                dim=b.dim,
            )
        pool_working, members_working = _truncate(pool), _truncate(members_bundle)

    # ---- rectangular member × pool similarity -------------------------------
    similarity = generate_rectangular_similarity(
        source=members_working,
        target=pool_working,
        recipe_config=config.get("recipe", {}),
    )

    # Reference distribution for cross-run-stable normalization: every
    # member×pool entry except self-pairs (explicit input, not the square
    # upper triangle — see utils.prepare_normalized_scores).
    self_mask = np.array([
        [pool_id == member_id for pool_id in pool.user_ids]
        for member_id in member_ids
    ])
    reference_scores = similarity.dir_matrix[~self_mask].ravel()

    # ---- budgeted pair selection (member × pool, novelty-excluded) ----------
    # Fallback literals match defaults/config.yaml (always present via
    # load_config; only a hand-built partial config dict ever hits them).
    budgets = config.get("budgets", {})
    selected_pairs = select_pairs_rectangular(
        dir_matrix=similarity.dir_matrix,
        member_ids=member_ids,
        pool_ids=pool.user_ids,
        max_n_llm_evaluations_per_profile=budgets.get("max_n_llm_evaluations_per_profile", 24),
        global_cap=budgets.get("max_pair_llm_calls", 1200),
        excluded_pairs=excluded_pairs,
    )

    # ---- LLM pair scoring ----------------------------------------------------
    unscored_pairs: List[CandidatePair] = []
    llm_scores = score_pairs_with_llm(
        similarity_matrix=None,
        user_ids=[],
        sections_dict=pool_sections,
        instruction=config.get("recipe", {}).get("instruction", "find good matches"),
        goal=config.get("instruction_prompt", {}).get("goal", ""),
        prompts_config_path=prompt_paths["scoring"],
        llm_wrapper=llm_wrapper,
        model=models_cfg.get("pair_llm"),
        max_n_llm_evaluations_per_profile=budgets.get("max_n_llm_evaluations_per_profile"),
        global_cap=budgets.get("max_pair_llm_calls"),
        n_profiles_to_score_together=budgets.get("n_profiles_to_score_together"),
        selected_pairs=selected_pairs,
        excluded_pairs=excluded_pairs,
        reasoning_effort=models_cfg.get("pair_reasoning_effort", "medium"),
        unscored_out=unscored_pairs,
    )

    # ---- asymmetric b-matching ------------------------------------------------
    scored_candidates = [
        CandidatePair.create(s.user1, s.user2, s.embed_score)
        for s in llm_scores.values()
    ]
    # Selected-but-unscored pairs keep their embedding-only weight instead of
    # being dropped from matching (mirrors the cohort runner).
    scored_candidates.extend(unscored_pairs)
    matching_config = config.get("matching", {})
    final_edges, _, _ = create_matches(
        candidates=scored_candidates,
        llm_scores=llm_scores,
        all_user_ids=list(pool.user_ids),
        matching_config=matching_config,
        blending_config=config.get("blending", {}),
        reference_scores=reference_scores,
        member_ids=set(member_ids),
        excluded_pairs=excluded_pairs,
    )

    # ---- intros ---------------------------------------------------------------
    introductions: Dict[str, Introduction] = generate_introductions_for_matches(
        final_edges=final_edges,
        sections_dict=pool_sections,
        instruction=config.get("recipe", {}).get("instruction", "find good matches"),
        goal=config.get("instruction_prompt", {}).get("goal", ""),
        introduction_config_path=prompt_paths["introduction"],
        llm_wrapper=llm_wrapper,
        model=models_cfg.get("pair_llm"),
    )
    for edge in final_edges:
        intro = introductions.get(edge.pair_id)
        if intro:
            edge.intro = intro.intro
            edge.starter_topics = intro.starter_topics
        else:
            attach_fallback_intro(edge)

    # ---- reports (members only) + new-pair residue ------------------------------
    extracted = [
        ExtractedSections(id=user_id, sections=sections, hash="")
        for user_id, sections in pool_sections.items()
    ]
    report_data = build_report_data(
        all_edges=final_edges,
        extracted_sections=extracted,
        top_matches_per_user=matching_config.get("b_max", 4),  # matches defaults/config.yaml
        scope_user_ids=list(member_ids),
    )

    new_pairs = [
        {
            "pair_id": edge.pair_id,
            "user1": edge.user1,
            "user2": edge.user2,
            "final_weight": round(edge.final_weight, 4),
        }
        for edge in final_edges
    ]

    return BatchMatchResult(
        edges=final_edges,
        report_data=report_data,
        new_pairs=new_pairs,
        member_ids=list(member_ids),
        excluded_count=len(excluded_pairs),
    )
