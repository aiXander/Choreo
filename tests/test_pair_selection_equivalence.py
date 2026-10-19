"""Pair selection + scoring-group packing: vectorized rewrites == legacy loops.

``select_pairs_rectangular`` (batch mode), ``select_pairs_for_llm_scoring_optimal``
(square cohort mode) and ``create_profile_groups_from_pairs`` were rewritten
from O(picks × N²) / O(groups × users) Python loops into ranked-row walks and
an adjacency index (a ~1.6k-member community run spent ~10 min of pure CPU in
the old rectangular selector before its first LLM call). Their output feeds
the LLM scoring calls AND keys the LLM response cache (via the groups), so the
contract is exact equality with the old code: same pairs, same order, same
scores (bit-for-bit), same groups, same log lines — ties included.

The legacy implementations below are VERBATIM copies of the pre-rewrite code
(only renamed to ``_reference_*``); they are the oracle, do not "fix" them.
"""

import contextlib
import io
import time
from typing import Dict, List, Optional, Set

import numpy as np
import pytest

from choreo.batch_match import blend_directional, select_pairs_rectangular
from choreo.candidate import CandidatePair
from choreo.score import create_profile_groups_from_pairs, select_pairs_for_llm_scoring_optimal
from choreo.utils import stable_pair_id


# ---------------------------------------------------------------------------
# Reference oracle: verbatim pre-rewrite implementations
# ---------------------------------------------------------------------------

def _reference_select_pairs_rectangular(
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
    both users are members the two directional entries are blended as
    ``0.7 * max + 0.3 * mean``, otherwise the single member→pool direction is
    used.

    Why a max-leaning blend and not the mean: the fused matrix is DIRECTIONAL
    on purpose (``cross[i][j]`` = "j can help i"), and the pair-scoring prompt
    tells the LLM that strong one-directional help is very valuable — but this
    shortlist runs first and used to average the two directions, so "A can
    solve B's problem, B is useless to A" scored the same as "both vaguely
    help each other" and often never reached the LLM at all. ``max`` alone
    over-rewards (0.8/0.1 would tie with 0.8/0.8); the 0.3 mean share keeps a
    strongly mutual pair ahead of a strongly one-directional one.

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
        score = blend_directional(values)
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


def _reference_select_pairs_for_llm_scoring_optimal(
    similarity_matrix: np.ndarray,
    user_ids: List[str],
    max_n_llm_evaluations_per_profile: int,
    global_cap: int,
    excluded_pairs: Optional[Set[str]] = None,
) -> List[CandidatePair]:
    """
    Select optimal subset of pairs for LLM scoring using greedy per-profile selection.
    Works directly with similarity matrix instead of pre-filtered candidates.

    Args:
        similarity_matrix: Full similarity matrix (n_users, n_users)
        user_ids: List of user IDs
        max_n_llm_evaluations_per_profile: Maximum number of evaluations per profile
        global_cap: Global max pairs to score (for backward compatibility)
        excluded_pairs: Optional set of pair_ids to skip entirely (novelty
            exclusions — pairs already surfaced in prior runs)

    Returns:
        Optimally selected pairs for scoring
    """
    n_users = len(user_ids)
    if n_users == 0:
        return []

    print(f"Greedy pair selection for {n_users} users with max {max_n_llm_evaluations_per_profile} evaluations per profile")

    # Create all possible pairs with their scores (upper triangle only to avoid duplicates)
    excluded_pairs = excluded_pairs or set()
    all_pairs = []
    for i in range(n_users):
        for j in range(i + 1, n_users):
            score = similarity_matrix[i, j]
            if score > 0:  # Only positive similarities
                pair = CandidatePair.create(user_ids[i], user_ids[j], score)
                if pair.pair_id in excluded_pairs:
                    continue
                all_pairs.append(pair)
    
    # Sort by similarity score (highest first)
    all_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
    
    # Greedy per-profile selection
    selected_pairs = []
    user_pair_counts = {user: 0 for user in user_ids}
    used_pair_ids = set()
    
    # Round-robin through users, adding their best available pair each time
    user_idx = 0
    consecutive_skips = 0
    
    while consecutive_skips < n_users and len(selected_pairs) < global_cap:
        current_user = user_ids[user_idx]
        
        # Skip this user if they've reached their maximum
        if user_pair_counts[current_user] >= max_n_llm_evaluations_per_profile:
            user_idx = (user_idx + 1) % n_users
            consecutive_skips += 1
            continue
        
        # Find the best available pair involving this user
        best_pair = None
        for pair in all_pairs:
            # Skip if already selected
            if pair.pair_id in used_pair_ids:
                continue
                
            # Skip if this user is not involved in the pair
            if pair.user1 != current_user and pair.user2 != current_user:
                continue
                
            # Skip if the other user in the pair has reached their maximum
            other_user = pair.user2 if pair.user1 == current_user else pair.user1
            if user_pair_counts[other_user] >= max_n_llm_evaluations_per_profile:
                continue
            
            # This is the best available pair for this user
            best_pair = pair
            break
        
        if best_pair is not None:
            # Add the pair
            selected_pairs.append(best_pair)
            used_pair_ids.add(best_pair.pair_id)
            user_pair_counts[best_pair.user1] += 1
            user_pair_counts[best_pair.user2] += 1
            consecutive_skips = 0  # Reset skip counter
        else:
            # No available pair for this user
            consecutive_skips += 1
        
        # Move to next user
        user_idx = (user_idx + 1) % n_users
    
    # Print statistics
    total_score = sum(pair.similarity_score for pair in selected_pairs)
    user_counts = list(user_pair_counts.values())
    min_count, max_count = (min(user_counts), max(user_counts)) if user_counts else (0, 0)
    avg_count = sum(user_counts) / len(user_counts) if user_counts else 0
    
    print(f"Selected {len(selected_pairs)}/{len(all_pairs)} pairs for LLM scoring")
    print(f"Total similarity score: {total_score:.4f}")
    print(f"Per-user evaluation counts - Min: {min_count}, Max: {max_count}, Avg: {avg_count:.1f}")
    
    return selected_pairs


def _reference_create_profile_groups_from_pairs(
    candidate_pairs: List[CandidatePair],
    n_profiles_to_score_together: int,
    max_groups: Optional[int] = None,
) -> List[Set[str]]:
    """Pack profiles into groups so every candidate pair lands in at least one group.

    Each group of up to ``n_profiles_to_score_together`` users becomes one batched
    LLM call that scores all C(n, 2) pairs among its members. To get a pair scored
    its two users must co-occur in some group, so this is a set-cover problem:
    cover all candidate pairs with as few groups (LLM calls) as possible.

    Greedy strategy: seed each new group from the highest-priority still-uncovered
    pair (``candidate_pairs`` is pre-sorted by similarity), then grow the group by
    repeatedly adding the user that covers the most *additional* uncovered pairs
    with the current members. Every iteration covers at least its seed pair, so the
    loop is guaranteed to terminate (≤ one group per pair in the worst case).

    Unlike the previous implementation this never evicts users from consideration,
    so it cannot orphan pairs and leave half the candidates unscored.

    ``max_groups`` (the LLM-call budget) optionally caps the number of groups; if
    pairs remain uncovered when the cap is hit, that is logged loudly rather than
    silently dropped.
    """
    if not candidate_pairs:
        return []

    n = max(2, n_profiles_to_score_together)

    # All users, and the set of candidate pairs to cover (keyed by an unordered
    # frozenset so user1/user2 ordering never matters). Preserve input priority.
    all_users: Set[str] = set()
    pair_keys: List[frozenset] = []
    pair_key_set: Set[frozenset] = set()
    for pair in candidate_pairs:
        all_users.add(pair.user1)
        all_users.add(pair.user2)
        key = frozenset((pair.user1, pair.user2))
        if key not in pair_key_set:
            pair_key_set.add(key)
            pair_keys.append(key)

    # Deterministic iteration order: set iteration is hash-salted per process,
    # which would make grouping (and thus LLM cache keys) vary between runs.
    users_sorted = sorted(all_users)

    covered: Set[frozenset] = set()
    groups: List[Set[str]] = []

    def uncovered_gain(user: str, group: Set[str]) -> int:
        """Number of still-uncovered candidate pairs adding `user` to `group` covers."""
        gain = 0
        for member in group:
            key = frozenset((user, member))
            if key in pair_key_set and key not in covered:
                gain += 1
        return gain

    total_pairs = len(pair_key_set)
    while len(covered) < total_pairs:
        if max_groups is not None and len(groups) >= max_groups:
            break

        # Seed from the highest-priority pair that is still uncovered.
        seed = next((key for key in pair_keys if key not in covered), None)
        if seed is None:
            break
        group: Set[str] = set(seed)

        # Grow greedily to size n by maximum uncovered-pair gain.
        while len(group) < n:
            best_user, best_gain = None, 0
            for user in users_sorted:
                if user in group:
                    continue
                gain = uncovered_gain(user, group)
                if gain > best_gain:
                    best_user, best_gain = user, gain
            if best_user is None:  # no remaining user covers a new pair
                break
            group.add(best_user)

        # Mark every candidate pair inside this group as covered.
        members = sorted(group)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = frozenset((members[a], members[b]))
                if key in pair_key_set:
                    covered.add(key)
        groups.append(group)

    n_covered = len(covered)
    print(f"Created {len(groups)} profile groups covering {n_covered}/{total_pairs} candidate pairs")
    if n_covered < total_pairs:
        print(
            f"⚠️  WARNING: {total_pairs - n_covered} candidate pair(s) left UNSCORED "
            f"after hitting the max_groups={max_groups} budget. Raise "
            f"budgets.max_pair_llm_calls to score them all."
        )
    for i, group in enumerate(groups):
        members = sorted(group)
        pairs_in_group = sum(
            1 for a in range(len(members)) for b in range(a + 1, len(members))
            if frozenset((members[a], members[b])) in pair_key_set
        )
        print(f"  Group {i+1}: {len(group)} profiles, {pairs_in_group} pairs")

    return groups


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _quiet(fn, *args, **kwargs):
    """Run fn, returning (result, captured stdout) — the log lines are part of
    the contract (they show up in Modal logs and get grepped)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


def _ids(rng: np.random.Generator, n: int, prefix: str = "u") -> List[str]:
    """Unique ids whose string order is unrelated to their index order (so
    min/max user ordering and positional tie-breaks are both exercised)."""
    return [f"{prefix}{v:05d}" for v in rng.permutation(max(10 * n, 10))[:n]]


def _pair_rows(pairs: List[CandidatePair]):
    return [(p.pair_id, p.user1, p.user2, p.similarity_score, type(p.similarity_score))
            for p in pairs]


def _sprinkle(rng, m: np.ndarray, *, nonfinite: float = 0.0, zeros: float = 0.0) -> np.ndarray:
    m = m.copy()
    if nonfinite:
        mask = rng.random(m.shape) < nonfinite
        m[mask] = rng.choice([np.nan, np.inf, -np.inf], size=int(mask.sum()))
    if zeros:
        m[rng.random(m.shape) < zeros] = 0.0
    return m


def _random_excluded(rng, users_a: List[str], users_b: List[str], frac: float) -> Set[str]:
    excluded = {stable_pair_id(a, b) for a in users_a for b in users_b
                if a != b and rng.random() < frac}
    excluded.add("ghost_pair")          # ids of pairs that don't exist are inert
    excluded.add(stable_pair_id(users_a[0], "nobody"))
    return excluded


def _assert_rect_equal(dir_matrix, member_ids, pool_ids, per, global_cap, excluded=None):
    kwargs = dict(max_n_llm_evaluations_per_profile=per, global_cap=global_cap,
                  excluded_pairs=excluded)
    ref, ref_log = _quiet(_reference_select_pairs_rectangular,
                          dir_matrix, list(member_ids), list(pool_ids), **kwargs)
    new, new_log = _quiet(select_pairs_rectangular,
                          dir_matrix, list(member_ids), list(pool_ids), **kwargs)
    assert _pair_rows(new) == _pair_rows(ref)
    assert new_log == ref_log
    return new


def _assert_square_equal(matrix, user_ids, per, global_cap, excluded=None):
    kwargs = dict(max_n_llm_evaluations_per_profile=per, global_cap=global_cap,
                  excluded_pairs=excluded)
    ref, ref_log = _quiet(_reference_select_pairs_for_llm_scoring_optimal,
                          matrix, list(user_ids), **kwargs)
    new, new_log = _quiet(select_pairs_for_llm_scoring_optimal,
                          matrix, list(user_ids), **kwargs)
    assert _pair_rows(new) == _pair_rows(ref)
    assert new_log == ref_log
    return new


def _rect_case(seed, n_members, n_pool, *, mode="subset", nonfinite=0.0, zeros=0.0,
               loc=0.3, scale=0.3, quantize=None, dtype=np.float64):
    """members×pool matrix. mode: 'equal' (members == pool, shuffled order),
    'subset' (members ⊂ pool) or 'outside' (some members missing from pool)."""
    rng = np.random.default_rng(seed)
    if mode == "equal":
        pool = _ids(rng, n_pool)
        members = list(rng.permutation(pool))[:n_pool]
    elif mode == "subset":
        pool = _ids(rng, n_pool)
        members = list(rng.choice(pool, size=n_members, replace=False))
    else:  # 'outside'
        everyone = _ids(rng, n_pool + n_members)
        pool = everyone[:n_pool]
        inside = list(rng.choice(pool, size=n_members // 2, replace=False))
        members = inside + everyone[n_pool:n_pool + n_members - len(inside)]
        members = list(rng.permutation(members))
    m = rng.normal(loc, scale, size=(len(members), len(pool)))
    if quantize:
        m = np.round(m / quantize) * quantize
    m = _sprinkle(rng, m, nonfinite=nonfinite, zeros=zeros).astype(dtype)
    return rng, m, [str(x) for x in members], [str(x) for x in pool]


# ---------------------------------------------------------------------------
# select_pairs_rectangular == legacy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["equal", "subset", "outside"])
@pytest.mark.parametrize("per,global_cap", [(5, 10_000), (2, 10_000), (1, 10_000), (4, 37)])
def test_rectangular_matches_legacy(mode, per, global_cap):
    n_members = 50 if mode == "equal" else 25
    seed = ["equal", "subset", "outside"].index(mode) * 1000 + per * 100 + global_cap % 97
    _, m, members, pool = _rect_case(seed, n_members, 50, mode=mode)
    _assert_rect_equal(m, members, pool, per, global_cap)


@pytest.mark.parametrize("mode", ["equal", "subset", "outside"])
def test_rectangular_matches_legacy_with_exclusions_nonfinite_and_nonpositive(mode):
    rng, m, members, pool = _rect_case(7, 30, 45, mode=mode, nonfinite=0.08, zeros=0.05,
                                       loc=0.05, scale=0.3)
    excluded = _random_excluded(rng, members, pool, 0.15)
    pairs = _assert_rect_equal(m, members, pool, 4, 10_000, excluded)
    assert pairs and not ({p.pair_id for p in pairs} & excluded)


def test_rectangular_one_finite_direction_passes_through_unblended():
    # a<->b: a→b is NaN, so only b→a (0.3) exists and must pass through as-is.
    members = pool = ["a", "b", "c"]
    m = np.array([[9.0, np.nan, 0.2],
                  [0.3, 9.0, 0.1],
                  [0.5, 0.4, 9.0]])
    pairs = _assert_rect_equal(m, members, pool, 8, 50)
    assert {p.pair_id: p.similarity_score for p in pairs}["a_b"] == 0.3


def test_rectangular_random_sweep():
    """Many small random configurations, every knob at once."""
    for seed in range(120):
        rng = np.random.default_rng(1000 + seed)
        mode = ["equal", "subset", "outside"][seed % 3]
        n_pool = int(rng.integers(2, 30))
        n_members = n_pool if mode == "equal" else int(rng.integers(1, n_pool + 1))
        rng, m, members, pool = _rect_case(
            2000 + seed, n_members, n_pool, mode=mode,
            nonfinite=float(rng.choice([0.0, 0.1])), zeros=float(rng.choice([0.0, 0.1])),
            loc=float(rng.choice([0.3, 0.0])),
            dtype=np.float32 if seed % 5 == 0 else np.float64,
        )
        excluded = _random_excluded(rng, members, pool, 0.1) if seed % 2 else None
        _assert_rect_equal(m, members, pool, int(rng.integers(0, 6)),
                           int(rng.choice([3, 10_000])), excluded)


def test_rectangular_tiny_inputs():
    _assert_rect_equal(np.array([[0.5, 0.2]]), ["a"], ["a", "b"], 3, 10)        # 1 member
    _assert_rect_equal(np.array([[9.0, 0.4], [0.6, 9.0]]), ["a", "b"], ["a", "b"], 3, 10)
    assert _assert_rect_equal(np.array([[0.9]]), ["a"], ["a"], 3, 10) == []      # self only
    assert _assert_rect_equal(np.zeros((0, 3)), [], ["a", "b", "c"], 3, 10) == []


def test_rectangular_member_with_no_eligible_pair():
    rng, m, members, pool = _rect_case(11, 12, 12, mode="equal")
    lonely = 3
    m[lonely, :] = -0.5                       # nobody helps them …
    m[:, pool.index(members[lonely])] = -0.5  # … and they help nobody
    pairs = _assert_rect_equal(m, members, pool, 3, 10_000)
    assert all(members[lonely] not in (p.user1, p.user2) for p in pairs)


def test_rectangular_ids_with_underscores_and_exclusions():
    # Fixed underscore count per id keeps pair ids unambiguous (the documented
    # precondition); the excluded-pair count must still parse them right.
    rng, m, members, pool = _rect_case(21, 15, 25, mode="subset")
    rename = {u: f"user_{u}" for u in pool}
    members, pool = [rename[u] for u in members], [rename[u] for u in pool]
    excluded = _random_excluded(rng, members, pool, 0.2)
    _assert_rect_equal(m, members, pool, 3, 10_000, excluded)


def test_rectangular_ties_resolve_exactly_like_legacy():
    """Tie contract: equal blended scores keep the legacy order — the pair's
    first member-major (i, j) position — so even heavily tied matrices select
    identically (quantized values ⇒ massive ties, incl. blended ties)."""
    for seed, mode in [(31, "equal"), (32, "subset"), (33, "outside")]:
        rng, m, members, pool = _rect_case(seed, 30, 40 if mode != "equal" else 30,
                                           mode=mode, quantize=0.1, zeros=0.1)
        _assert_rect_equal(m, members, pool, 3, 10_000)
    # All-equal matrix: pure tie order.
    members = pool = [f"p{k}" for k in (5, 2, 9, 1, 7, 3)]
    _assert_rect_equal(np.full((6, 6), 0.5), members, pool, 2, 10_000)


def test_rectangular_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        select_pairs_rectangular(np.ones((2, 2)), ["a", "a"], ["a", "b"], 3, 10)


# ---------------------------------------------------------------------------
# select_pairs_for_llm_scoring_optimal == legacy
# ---------------------------------------------------------------------------

def _square_case(seed, n, *, symmetric=True, nonfinite=0.0, zeros=0.0, loc=0.3,
                 quantize=None, dtype=np.float64):
    rng = np.random.default_rng(seed)
    users = _ids(rng, n)
    m = rng.normal(loc, 0.3, size=(n, n))
    if symmetric:
        m = (m + m.T) / 2
    if quantize:
        m = np.round(m / quantize) * quantize
    m = _sprinkle(rng, m, nonfinite=nonfinite, zeros=zeros).astype(dtype)
    return rng, m, users


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("per,global_cap", [(5, 10_000), (1, 10_000), (4, 23)])
def test_square_matches_legacy(symmetric, per, global_cap):
    _, m, users = _square_case(per * 100 + global_cap % 97, 45, symmetric=symmetric)
    _assert_square_equal(m, users, per, global_cap)


def test_square_matches_legacy_with_exclusions_nonfinite_ties_and_float32():
    for seed in range(40):
        rng, m, users = _square_case(
            500 + seed, int(np.random.default_rng(seed).integers(1, 35)),
            symmetric=bool(seed % 2), nonfinite=0.08 if seed % 3 else 0.0, zeros=0.05,
            loc=0.1, quantize=0.1 if seed % 4 == 0 else None,
            dtype=np.float32 if seed % 5 == 0 else np.float64,
        )
        excluded = _random_excluded(rng, users, users, 0.15) if seed % 2 else None
        _assert_square_equal(m, users, int(rng.integers(0, 5)),
                             int(rng.choice([4, 10_000])), excluded)


def test_square_empty_and_duplicate_ids():
    assert _assert_square_equal(np.zeros((0, 0)), [], 3, 10) == []
    with pytest.raises(ValueError, match="duplicate"):
        select_pairs_for_llm_scoring_optimal(np.ones((2, 2)), ["a", "a"], 3, 10)


# ---------------------------------------------------------------------------
# create_profile_groups_from_pairs == legacy (groups key the LLM cache)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_together", [2, 3, 5, 8])
@pytest.mark.parametrize("max_groups", [None, 6])
def test_profile_groups_match_legacy(n_together, max_groups):
    for seed in range(6):
        rng, m, members, pool = _rect_case(800 + seed, 30, 40, mode="subset")
        pairs, _ = _quiet(select_pairs_rectangular, m, members, pool, 4, 10_000)
        # Also a raw, unsorted pair list with a duplicate and a flipped order.
        raw = list(rng.permutation(pairs))
        if raw:
            p = raw[0]
            raw.append(CandidatePair(p.user2, p.user1, p.similarity_score, p.pair_id))
        for candidate_pairs in (pairs, raw):
            ref, ref_log = _quiet(_reference_create_profile_groups_from_pairs,
                                  candidate_pairs, n_together, max_groups)
            new, new_log = _quiet(create_profile_groups_from_pairs,
                                  candidate_pairs, n_together, max_groups)
            assert new == ref
            assert new_log == ref_log


# ---------------------------------------------------------------------------
# Coarse perf guard (the reason for the rewrite)
# ---------------------------------------------------------------------------

def test_selection_and_grouping_scale_to_a_whole_community():
    """1,500 × 1,500 at per-profile cap 16: the legacy rectangular selector
    took ~10 min here; the rewrite is ~1 s. The bound is deliberately loose
    (CI noise) — it only has to catch a regression back to O(picks × N²)."""
    n = 1500
    rng = np.random.default_rng(42)
    users = [f"user-{k:05d}" for k in rng.permutation(n)]
    m = rng.normal(0.3, 0.2, size=(n, n))

    t0 = time.perf_counter()
    pairs, _ = _quiet(select_pairs_rectangular, m, users, users, 16, 100_000)
    t_rect = time.perf_counter() - t0
    assert len(pairs) > 10_000
    assert t_rect < 30, f"select_pairs_rectangular took {t_rect:.1f}s"

    t0 = time.perf_counter()
    _quiet(select_pairs_for_llm_scoring_optimal, (m + m.T) / 2, users, 16, 100_000)
    t_square = time.perf_counter() - t0
    assert t_square < 30, f"select_pairs_for_llm_scoring_optimal took {t_square:.1f}s"

    t0 = time.perf_counter()
    groups, _ = _quiet(create_profile_groups_from_pairs, pairs, 5)
    t_groups = time.perf_counter() - t0
    assert groups
    assert t_groups < 30, f"create_profile_groups_from_pairs took {t_groups:.1f}s"
