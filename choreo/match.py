"""Greedy b-matching algorithm to create final matches."""

from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from .candidate import CandidatePair
from .utils import prepare_normalized_scores
from .schemas import Edge, PairScore  # noqa: F401 — Edge re-exported


def compute_final_weights(
    candidates: List[CandidatePair],
    llm_scores: Dict[str, PairScore],
    embed_weight: float,
    llm_weight: float,
    full_similarity_matrix: np.ndarray = None,
    all_user_ids: List[str] = None,
    reference_scores: np.ndarray = None,
) -> List[Edge]:
    """
    Compute final weights blending embedding and LLM scores with proper normalization.

    Args:
        candidates: All candidate pairs
        llm_scores: LLM scores by pair_id
        embed_weight: Weight for embedding score in final blend
        llm_weight: Weight for LLM score in final blend
        full_similarity_matrix: Full similarity matrix for reference distribution (optional)
        all_user_ids: All user IDs corresponding to similarity matrix (optional)
        reference_scores: Explicit flat reference distribution (optional; takes
            precedence over the square-matrix derivation — see
            utils.prepare_normalized_scores)

    Returns:
        List of Edge objects with final weights
    """
    # Get normalized scores using reference distribution
    normalized_embed_lookup, normalized_llm_lookup, normalization_applied = prepare_normalized_scores(
        candidates=candidates,
        llm_scores=llm_scores,
        full_similarity_matrix=full_similarity_matrix,
        all_user_ids=all_user_ids,
        reference_scores=reference_scores,
    )
    
    edges = []
    
    for candidate in candidates:
        pair_id = candidate.pair_id
        
        # Get normalized embedding score
        embed_score_normalized = normalized_embed_lookup.get(pair_id, candidate.similarity_score)
        
        # Get LLM score if available
        llm_score_obj = llm_scores.get(pair_id)
        
        if llm_score_obj:
            # Get normalized LLM score
            llm_score_normalized = normalized_llm_lookup.get(pair_id, llm_score_obj.score)
            
            # Blend normalized scores
            final_weight = embed_weight * embed_score_normalized + llm_weight * llm_score_normalized
            llm_score_raw = llm_score_obj.score
        else:
            # Use only embedding score
            llm_score_raw = 0.0
            final_weight = embed_score_normalized  # Only embedding score
        
        edge = Edge(
            user1=candidate.user1,
            user2=candidate.user2,
            pair_id=pair_id,
            final_weight=final_weight,
            embed_score=candidate.similarity_score,  # Keep original for reference
            llm_score=llm_score_raw,  # Keep original for reference
            embed_score_normalized=embed_score_normalized,  # Store normalized for display
            llm_score_normalized=llm_score_normalized if llm_score_obj else 0.0  # Store normalized for display
        )
        edges.append(edge)
    
    return edges


def greedy_b_matching(
    edges: List[Edge],
    b_min: int,
    b_max: int,
    all_users: Set[str],
    member_ids: Optional[Set[str]] = None,
    pool_b_max: Optional[int] = None,
    excluded_pairs: Optional[Set[str]] = None,
) -> List[Edge]:
    """
    Greedy b-matching algorithm, optionally ASYMMETRIC.

    Symmetric (legacy cohort) mode — the default: every user is a "member" and
    ``b_min``/``b_max`` bind on everyone. Asymmetric (subset batch) mode: pass
    ``member_ids`` and the degree targets bind only on the member side; users
    outside ``member_ids`` are pool candidates whose degree is capped by
    ``pool_b_max`` instead (None = uncapped) so the constraint is looser — but
    still prevents one popular pool person from saturating every member when
    set.

    Args:
        edges: All candidate edges with final weights
        b_min: Minimum degree per member
        b_max: Maximum degree per member
        all_users: Set of all user IDs appearing on the member side
        member_ids: Users the degree targets bind on (None = all_users)
        pool_b_max: Optional degree cap for non-member (pool) endpoints
        excluded_pairs: Optional pair_ids never to select (novelty exclusions;
            defense-in-depth — selection upstream should already skip them)

    Returns:
        Selected edges forming the b-matching
    """
    members = set(member_ids) if member_ids is not None else set(all_users)

    if excluded_pairs:
        n_before = len(edges)
        edges = [e for e in edges if e.pair_id not in excluded_pairs]
        if len(edges) < n_before:
            print(f"Excluded {n_before - len(edges)} edges from match history")

    def cap_of(user: str) -> Optional[int]:
        """Max degree allowed for this endpoint (None = uncapped pool user)."""
        return b_max if user in members else pool_b_max

    def under_cap(user: str) -> bool:
        cap = cap_of(user)
        return cap is None or user_degrees[user] < cap

    # Sort edges by final weight (descending)
    sorted_edges = sorted(edges, key=lambda e: e.final_weight, reverse=True)

    # Track degree of each user
    user_degrees = defaultdict(int)
    selected_edges = []
    selected_set = set()

    print(f"Starting greedy b-matching with {len(sorted_edges)} edges")
    print(f"Target degree range: [{b_min}, {b_max}] per member"
          + (f" (pool cap: {pool_b_max})" if member_ids is not None else ""))

    # Phase 1: Greedy selection while respecting per-endpoint caps
    for edge in sorted_edges:
        if under_cap(edge.user1) and under_cap(edge.user2):
            selected_edges.append(edge)
            selected_set.add(id(edge))
            user_degrees[edge.user1] += 1
            user_degrees[edge.user2] += 1

    print(f"Phase 1: Selected {len(selected_edges)} edges")

    # Phase 2: Backfill members below b_min (respecting partner caps).
    # sorted() — set iteration is hash-salted per process; backfill order
    # decides edge selection under contention, so keep it deterministic.
    users_below_min = [
        user for user in sorted(members)
        if user_degrees[user] < b_min
    ]

    if users_below_min:
        print(f"Phase 2: Backfilling {len(users_below_min)} users below minimum degree")

        for user in users_below_min:
            needed = b_min - user_degrees[user]

            # Find best available edges for this user
            candidates = [
                edge for edge in sorted_edges
                if id(edge) not in selected_set
                and (edge.user1 == user or edge.user2 == user)
            ]
            candidates.sort(key=lambda e: e.final_weight, reverse=True)

            for edge in candidates:
                if needed <= 0:
                    break
                other_user = edge.user2 if edge.user1 == user else edge.user1
                if under_cap(other_user):
                    selected_edges.append(edge)
                    selected_set.add(id(edge))
                    user_degrees[edge.user1] += 1
                    user_degrees[edge.user2] += 1
                    needed -= 1

    # Phase 3: Force-fill members still below b_min (relaxing caps for partners)
    users_still_below = [
        user for user in sorted(members)
        if user_degrees[user] < b_min
    ]

    if users_still_below:
        print(f"Phase 3: Force-filling {len(users_still_below)} users still below b_min (relaxing b_max)")

        for user in users_still_below:
            needed = b_min - user_degrees[user]

            # Find best available edges, ignoring the cap for the partner
            candidates = [
                edge for edge in sorted_edges
                if id(edge) not in selected_set
                and (edge.user1 == user or edge.user2 == user)
            ]
            # Prefer partners with lowest current degree (least overloaded)
            candidates.sort(key=lambda e, u=user: (
                user_degrees[e.user2 if e.user1 == u else e.user1],
                -e.final_weight
            ))

            for edge in candidates:
                if needed <= 0:
                    break
                other_user = edge.user2 if edge.user1 == user else edge.user1
                selected_edges.append(edge)
                selected_set.add(id(edge))
                user_degrees[edge.user1] += 1
                user_degrees[edge.user2] += 1
                needed -= 1
                print(f"  {user}: force-added edge with {other_user} "
                      f"(partner degree now {user_degrees[other_user]})")

    # Final statistics (over the member side, where the targets bind)
    final_degrees = {user: user_degrees[user] for user in members}
    avg_degree = sum(final_degrees.values()) / len(members) if members else 0

    users_at_min = sum(1 for d in final_degrees.values() if d >= b_min)
    users_at_max = sum(1 for d in final_degrees.values() if d == b_max)

    print("Final matching:")
    print(f"  Selected {len(selected_edges)} edges")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Users at/above b_min ({b_min}): {users_at_min}/{len(members)}")
    print(f"  Users at b_max ({b_max}): {users_at_max}/{len(members)}")

    return selected_edges


def create_matches(
    candidates: List[CandidatePair],
    llm_scores: Dict[str, PairScore],
    all_user_ids: List[str],
    matching_config: Dict,
    blending_config: Dict,
    similarity_matrix: np.ndarray = None,
    reference_scores: np.ndarray = None,
    member_ids: Optional[Set[str]] = None,
    excluded_pairs: Optional[Set[str]] = None,
) -> Tuple[List[Edge], Dict[str, float], Dict[str, float]]:
    """
    Full pipeline to create final matches.

    Args:
        candidates: All candidate pairs
        llm_scores: LLM scores by pair_id
        all_user_ids: All user IDs in the system
        matching_config: Matching configuration (b_min, b_max; optional
            pool_b_max for asymmetric subset mode)
        blending_config: Blending configuration (weights)
        similarity_matrix: Full square similarity matrix for normalization (optional)
        reference_scores: Explicit flat reference distribution for
            normalization (optional; rectangular modes pass their member×pool
            matrix values here — see utils.prepare_normalized_scores)
        member_ids: Degree targets bind on these users only (None = everyone;
            see greedy_b_matching)
        excluded_pairs: Pair_ids never to match (novelty exclusions)

    Returns:
        Tuple of (final_edges, normalized_embed_scores, normalized_llm_scores)
    """
    if excluded_pairs:
        candidates = [c for c in candidates if c.pair_id not in excluded_pairs]

    # Get normalized scores for plotting
    normalized_embed_lookup, normalized_llm_lookup, _ = prepare_normalized_scores(
        candidates=candidates,
        llm_scores=llm_scores,
        full_similarity_matrix=similarity_matrix,
        all_user_ids=all_user_ids,
        reference_scores=reference_scores,
    )

    # Compute final weights with normalization
    edges = compute_final_weights(
        candidates=candidates,
        llm_scores=llm_scores,
        embed_weight=blending_config['embed_weight'],
        llm_weight=blending_config['llm_weight'],
        full_similarity_matrix=similarity_matrix,
        all_user_ids=all_user_ids,
        reference_scores=reference_scores,
    )

    print(f"Computed final weights for {len(edges)} edges")

    # Run greedy b-matching
    selected_edges = greedy_b_matching(
        edges=edges,
        b_min=matching_config['b_min'],
        b_max=matching_config['b_max'],
        all_users=set(all_user_ids),
        member_ids=member_ids,
        pool_b_max=matching_config.get('pool_b_max'),
        excluded_pairs=excluded_pairs,
    )

    return selected_edges, normalized_embed_lookup, normalized_llm_lookup
