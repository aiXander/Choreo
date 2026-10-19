"""LLM-based pair scoring for candidate pairs."""

from typing import Any, Callable, List, Dict, Set, Tuple, Optional
import numpy as np
from itertools import combinations

from .candidate import CandidatePair
from .llm import LLMWrapper, run_coro_blocking
from .utils import QUERY_ID, load_yaml, hash_text, is_absent, stable_pair_id
from .schemas import ExtractedSections, PairScore  # noqa: F401 — PairScore re-exported


# Extra re-ask rounds for pair scores that come back missing/unparsable in an
# otherwise-successful response. (Transport-level failures — rate limits, 5xx,
# malformed JSON — are already retried per call inside LLMWrapper; this covers
# the other failure mode: valid JSON that omits some of the requested keys.)
MAX_SCORE_RETRIES = 3


def create_profile_groups_from_pairs(
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

    # Adjacency over still-UNCOVERED candidate pairs: a user's gain for a group
    # is how many group members it has an uncovered pair with, so scanning the
    # members' adjacency touches only real candidates instead of every user ×
    # every member per growth step (the old O(users) scan took seconds at
    # ~1.5k users). Same gains, same tie-break (lowest id in sorted order,
    # matching the old first-strictly-greater scan over ``users_sorted``), so
    # the groups — and therefore the LLM cache keys — are unchanged.
    uncovered_adj: Dict[str, Set[str]] = {user: set() for user in users_sorted}
    for key in pair_keys:
        if len(key) == 2:
            u, v = sorted(key)
            uncovered_adj[u].add(v)
            uncovered_adj[v].add(u)

    total_pairs = len(pair_key_set)
    seed_idx = 0  # covered only grows, so the first-uncovered seed scan is monotone
    while len(covered) < total_pairs:
        if max_groups is not None and len(groups) >= max_groups:
            break

        # Seed from the highest-priority pair that is still uncovered.
        while seed_idx < len(pair_keys) and pair_keys[seed_idx] in covered:
            seed_idx += 1
        if seed_idx >= len(pair_keys):
            break
        group: Set[str] = set(pair_keys[seed_idx])

        # Grow greedily to size n by maximum uncovered-pair gain.
        while len(group) < n:
            gains: Dict[str, int] = {}
            for member in group:
                for user in uncovered_adj[member]:
                    if user not in group:
                        gains[user] = gains.get(user, 0) + 1
            if not gains:  # no remaining user covers a new pair
                break
            best_gain = max(gains.values())
            group.add(min(user for user, gain in gains.items() if gain == best_gain))

        # Mark every candidate pair inside this group as covered.
        members = sorted(group)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = frozenset((members[a], members[b]))
                if key in pair_key_set and key not in covered:
                    covered.add(key)
                    uncovered_adj[members[a]].discard(members[b])
                    uncovered_adj[members[b]].discard(members[a])
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


def build_batch_scoring_prompt(
    user_profiles: List[str],  # list of user_ids
    sections_dict: Dict[str, Dict[str, str]],
    instruction: str,
    prompt_template: str,
    goal: str,
    pairs: Optional[List[Tuple[str, str]]] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Build prompt for batch LLM scoring of multiple profiles.

    By default every C(n,2) pair among ``user_profiles`` is requested. Pass
    ``pairs`` to request only a subset — query mode uses this to score
    query↔candidate pairs without asking for candidate↔candidate scores.

    The prompt — and the JSON the model returns — is keyed by short
    per-prompt ALIASES (``Q`` for the query pseudo-user, ``P1``/``P2``/… in
    ``user_profiles`` order), never by raw user ids: long ids (uuids) in the
    pair keys are attention noise for the model, and a single transcription
    slip makes a score unparsable (which in query mode silently drops the
    candidate). Returns ``(prompt, alias_of)`` where ``alias_of`` maps real
    user id → alias; parse responses through :func:`get_pair_score` with that
    map. Everything outside the prompt stays keyed by real ids.

    ``display_names`` maps user ids to the human names woven into the prompt
    (``<profile id="P1" name="…">``). Ids without an entry fall back to the
    raw id as the name — readable for file-derived ids, noisy for uuids, so
    pass display_names whenever ids are opaque. The query pseudo-user without
    a name renders alias-only.
    """
    display_names = display_names or {}

    # Deterministic per-prompt aliases in user_profiles order (callers pass
    # sorted/deterministic rosters, so prompt bytes — and LLM cache keys —
    # are stable across runs).
    alias_of: Dict[str, str] = {}
    for user_id in user_profiles:
        if user_id == QUERY_ID:
            alias_of[user_id] = "Q"
        else:
            alias_of[user_id] = f"P{len(alias_of) + (0 if QUERY_ID in alias_of else 1)}"

    def label_of(user_id: str) -> Optional[str]:
        name = display_names.get(user_id)
        if name:
            return name
        return None if user_id == QUERY_ID else user_id

    # Format all user profiles
    def format_sections(sections: Dict[str, str], user_id: str) -> str:
        label = label_of(user_id)
        alias = alias_of[user_id]
        lines = [f"Profile of {label} ({alias}):" if label else f"Profile of {alias}:"]
        # The query pseudo-user's sections are SEARCH TARGETS aimed at each
        # candidate's matching section, not attributes the asker possesses —
        # render them as "Looking for (<Section>): …" so the model can never
        # misread the query as a profile of skills the asker already has.
        # `QUERY_ID` only ever appears in query mode, so candidate-side and
        # default/pair-mode rendering (bare "<Section>: …") stays untouched.
        is_query = user_id == QUERY_ID
        for section_name, content in sections.items():
            if not is_absent(content):
                if is_query:
                    lines.append(f"  Looking for ({section_name.title()}): {content}")
                else:
                    lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)

    # Create XML formatted profiles
    profiles_xml = "<profiles>\n"
    for user_id in user_profiles:
        user_sections = sections_dict.get(user_id, {})
        profile_text = format_sections(user_sections, user_id)
        label = label_of(user_id)
        alias = alias_of[user_id]
        tag_attrs = f"id=\"{alias}\" name=\"{label}\"" if label else f"id=\"{alias}\""
        profiles_xml += f"  <profile {tag_attrs}>\n    {profile_text.replace(chr(10), chr(10) + '    ')}\n  </profile>\n"
    profiles_xml += "</profiles>"

    # Generate requested pairs and create the JSON format hint — alias-keyed,
    # with an explicit numeric-range placeholder so the model knows the viable
    # range regardless of the surrounding template text.
    if pairs is None:
        pairs = list(combinations(user_profiles, 2))
    hint_keys = [f"{alias_of[u1]}_{alias_of[u2]}" for u1, u2 in pairs]
    json_format_hint = "{" + ", ".join(f'"{key}": <score 0.0-1.0>' for key in hint_keys) + "}"

    # Build the complete prompt
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_profiles_xml_formatted=profiles_xml,
        json_format_hint=json_format_hint
    )

    return prompt, alias_of


def get_pair_score(
    response: Dict[str, Any],
    user1: str,
    user2: str,
    alias_of: Optional[Dict[str, str]] = None,
):
    """Look up one pair's score in a scoring response: alias keys first (the
    format the prompt requested), raw-id keys as fallback (cheap insurance
    against a model echoing profile ids or a caller-supplied template that
    still renders ids) — both orders each. Returns None when absent."""
    alias_of = alias_of or {}
    alias1, alias2 = alias_of.get(user1), alias_of.get(user2)
    keys = []
    if alias1 and alias2:
        keys += [f"{alias1}_{alias2}", f"{alias2}_{alias1}"]
    keys += [f"{user1}_{user2}", f"{user2}_{user1}"]
    for key in keys:
        if key in response:
            return response[key]
    return None


def _run_scoring_batch(
    llm_wrapper: LLMWrapper,
    prompts: List[str],
    cache_keys: List[Optional[str]],
    model: str,
    reasoning_effort: str = "medium",
) -> List:
    """Run one parallel batch of scoring prompts (async-host safe)."""
    return run_coro_blocking(llm_wrapper.batch_json_complete(
        prompts=prompts,
        model=model,
        cache_keys=cache_keys,
        reasoning_effort=reasoning_effort,
        print_reasoning_summary=False,
        verbosity=0,
        progress_label="score",
    ))


def _merge_scores_from_response(
    response,
    pairs: List[CandidatePair],
    pair_scores: Dict[str, PairScore],
    alias_of: Optional[Dict[str, str]] = None,
) -> List[CandidatePair]:
    """Parse one batch response into ``pair_scores``; return the pairs whose
    score came back missing or unparsable (the retry input). ``alias_of`` is
    the prompt's real-id → alias map (see ``build_batch_scoring_prompt``)."""
    if isinstance(response, Exception) or not isinstance(response, dict):
        print(f"Warning: unusable scoring response "
              f"({response if isinstance(response, Exception) else type(response).__name__})")
        return list(pairs)

    missing: List[CandidatePair] = []
    for pair in pairs:
        score = get_pair_score(response, pair.user1, pair.user2, alias_of)
        if score is None:
            missing.append(pair)
            continue
        try:
            score_val = float(score)
        except (ValueError, TypeError):
            print(f"Error parsing score for pair {pair.pair_id}: {score!r}")
            missing.append(pair)
            continue
        pair_scores[pair.pair_id] = PairScore(
            pair_id=pair.pair_id,
            user1=pair.user1,
            user2=pair.user2,
            embed_score=pair.similarity_score,
            score=max(0.0, min(1.0, score_val)),
        )
    return missing


def _require_unique_ids(ids: List[str], label: str) -> None:
    """The vectorized selectors index users by position; a repeated id would
    silently merge two rows the legacy dict-keyed code treated differently."""
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} contains duplicate user ids — dedupe before pair selection")


def _pairs_matching_pair_id(pair_id: str, index: Dict[str, int]) -> List[Tuple[str, str]]:
    """Every ``(x, y)`` of known users with ``stable_pair_id(x, y) == pair_id``.

    Pair ids are ``f"{min}_{max}"`` and user ids may themselves contain ``_``,
    so try every split point instead of parsing. Only used to count how many
    eligible pairs an ``excluded_pairs`` set removes (for the log line).
    """
    out: List[Tuple[str, str]] = []
    cut = pair_id.find("_")
    while cut >= 0:
        x, y = pair_id[:cut], pair_id[cut + 1:]
        if x < y and x in index and y in index:
            out.append((x, y))
        cut = pair_id.find("_", cut + 1)
    return out


class _RankedCandidates:
    """Each picker's candidates in exact legacy greedy order, ranked lazily.

    The legacy selectors kept ONE globally sorted pair list and, for every
    single pick, rescanned it from the top for "the first unused pair
    involving this picker whose partner is under the cap" — O(picks × N²).
    That first hit is exactly the next eligible entry of the picker's OWN
    candidate list in global order, so each picker gets its own ranked row
    and a pointer. Every skip reason (pair already used, partner at the cap,
    pair history-excluded) is permanent — counts only grow — so the pointer
    never has to look back: total work ≈ picks + skipped entries.

    ``scores[a, c]`` is picker ``a``'s pair score with column ``c`` (NaN = not
    a candidate). Global order is score desc, then ``tiebreak(a, cols)`` asc —
    the pair's insertion index in the legacy list, so equal scores resolve
    exactly as Python's stable ``sort(reverse=True)`` did. A row is ranked in
    growing top-K chunks cut at a score THRESHOLD (every entry tied with the
    K-th score joins the chunk), so ties never straddle a chunk boundary.
    """

    def __init__(
        self,
        scores: np.ndarray,
        tiebreak: Callable[[int, np.ndarray], np.ndarray],
        first_chunk: int,
    ):
        n = scores.shape[0]
        self._scores = scores
        self._tiebreak = tiebreak
        self._buf: List[List[int]] = [[] for _ in range(n)]
        self._ptr = [0] * n
        self._floor: List[Optional[float]] = [None] * n  # rank only scores < floor next
        self._chunk = [max(1, first_chunk)] * n
        self._drained = [False] * n

    def pop(self, a: int) -> Optional[int]:
        """Next not-yet-visited candidate column of picker ``a`` (None = none left)."""
        ptr = self._ptr[a]
        if ptr >= len(self._buf[a]):
            if not self._rank_next_chunk(a):
                return None
            ptr = 0
        self._ptr[a] = ptr + 1
        return self._buf[a][ptr]

    def _rank_next_chunk(self, a: int) -> bool:
        if self._drained[a]:
            return False
        row = self._scores[a]
        floor = self._floor[a]
        cols = np.flatnonzero(~np.isnan(row) if floor is None else row < floor)
        if cols.size == 0:
            self._drained[a] = True
            return False
        vals = row[cols]
        k = self._chunk[a]
        if cols.size > k:
            cut = np.partition(vals, cols.size - k)[cols.size - k]  # K-th largest
            keep = vals >= cut
            cols, vals = cols[keep], vals[keep]
            self._floor[a] = cut
            self._chunk[a] = 2 * k
        else:
            self._drained[a] = True  # this chunk holds the rest of the row
        order = np.lexsort((self._tiebreak(a, cols), -vals))
        self._buf[a] = cols[order].tolist()
        self._ptr[a] = 0
        return True


def select_pairs_for_llm_scoring_optimal(
    similarity_matrix: np.ndarray,
    user_ids: List[str],
    max_n_llm_evaluations_per_profile: int,
    global_cap: int,
    excluded_pairs: Optional[Set[str]] = None,
) -> List[CandidatePair]:
    """
    Select optimal subset of pairs for LLM scoring using greedy per-profile selection.
    Works directly with similarity matrix instead of pre-filtered candidates.

    Pairs are the upper triangle (``i < j``) with ``score > 0``, minus
    ``excluded_pairs``, sorted by score desc (ties: row-major ``(i, j)``
    order). Round-robin over users: each takes its best still-unused pair
    whose partner is under the per-profile cap, until every user is skipped
    in a row or ``global_cap`` is hit.

    Implemented with per-user ranked rows (``_RankedCandidates``), which
    selects exactly the pairs, in exactly the order, of the legacy
    rescan-the-global-list loop — including ties — in ~O(picks + skipped)
    instead of O(picks × N²). Requires unique ``user_ids``.

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
    _require_unique_ids(user_ids, "user_ids")

    print(f"Greedy pair selection for {n_users} users with max {max_n_llm_evaluations_per_profile} evaluations per profile")

    excluded_pairs = excluded_pairs or set()
    cap = max_n_llm_evaluations_per_profile

    # Symmetric view of the upper triangle: scores[a, b] = M[min(a,b), max(a,b)],
    # NaN where the pair is not eligible (diagonal, NaN, score <= 0).
    upper_vals = np.asarray(similarity_matrix)[:n_users, :n_users].astype(np.float64)
    rows, cols = np.indices((n_users, n_users))
    with np.errstate(invalid="ignore"):
        scores = np.where(cols > rows, upper_vals, upper_vals.T)
        scores[~(scores > 0)] = np.nan
    np.fill_diagonal(scores, np.nan)
    del rows, cols

    index = {user: i for i, user in enumerate(user_ids)}
    n_excluded_eligible = sum(
        1
        for pair_id in excluded_pairs
        for x, y in _pairs_matching_pair_id(pair_id, index)
        if not np.isnan(scores[index[x], index[y]])
    )
    n_all_pairs = int(np.count_nonzero(~np.isnan(scores))) // 2 - n_excluded_eligible

    def tiebreak(a: int, partners: np.ndarray) -> np.ndarray:
        lo, hi = np.minimum(partners, a), np.maximum(partners, a)
        return lo.astype(np.int64) * n_users + hi

    ranked = _RankedCandidates(scores, tiebreak, first_chunk=max(64, 4 * cap))

    # Greedy per-profile selection
    selected_pairs = []
    user_pair_counts = [0] * n_users
    used_pair_ids = set()

    # Round-robin through users, adding their best available pair each time
    user_idx = 0
    consecutive_skips = 0

    while consecutive_skips < n_users and len(selected_pairs) < global_cap:
        current_user = user_ids[user_idx]

        # Skip this user if they've reached their maximum
        if user_pair_counts[user_idx] >= max_n_llm_evaluations_per_profile:
            user_idx = (user_idx + 1) % n_users
            consecutive_skips += 1
            continue

        # Best available pair = next eligible entry of this user's ranked row.
        best_pair = None
        while True:
            other_idx = ranked.pop(user_idx)
            if other_idx is None:
                break
            if user_pair_counts[other_idx] >= max_n_llm_evaluations_per_profile:
                continue
            pair_id = stable_pair_id(current_user, user_ids[other_idx])
            if pair_id in used_pair_ids or pair_id in excluded_pairs:
                continue
            i, j = min(user_idx, other_idx), max(user_idx, other_idx)
            best_pair = CandidatePair.create(user_ids[i], user_ids[j], similarity_matrix[i, j])
            break

        if best_pair is not None:
            # Add the pair
            selected_pairs.append(best_pair)
            used_pair_ids.add(best_pair.pair_id)
            user_pair_counts[user_idx] += 1
            user_pair_counts[other_idx] += 1
            consecutive_skips = 0  # Reset skip counter
        else:
            # No available pair for this user
            consecutive_skips += 1

        # Move to next user
        user_idx = (user_idx + 1) % n_users

    # Print statistics
    total_score = sum(pair.similarity_score for pair in selected_pairs)
    user_counts = list(user_pair_counts)
    min_count, max_count = (min(user_counts), max(user_counts)) if user_counts else (0, 0)
    avg_count = sum(user_counts) / len(user_counts) if user_counts else 0
    
    print(f"Selected {len(selected_pairs)}/{n_all_pairs} pairs for LLM scoring")
    print(f"Total similarity score: {total_score:.4f}")
    print(f"Per-user evaluation counts - Min: {min_count}, Max: {max_count}, Avg: {avg_count:.1f}")
    
    return selected_pairs


def score_pairs_with_llm(
    similarity_matrix: np.ndarray,
    user_ids: List[str],
    sections_dict: Dict[str, Dict[str, str]],  # user_id -> sections
    instruction: str,
    goal: str,
    prompts_config_path: Optional[str] = None,
    llm_wrapper: LLMWrapper = None,
    model: str = None,
    max_n_llm_evaluations_per_profile: int = None,
    global_cap: int = None,
    n_profiles_to_score_together: int = None,
    force: bool = False,
    prompt_template: Optional[str] = None,
    excluded_pairs: Optional[Set[str]] = None,
    selected_pairs: Optional[List[CandidatePair]] = None,
    reasoning_effort: str = "medium",
    unscored_out: Optional[List[CandidatePair]] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> Dict[str, PairScore]:
    """
    Score pairs using LLM with batch scoring approach.

    Args:
        similarity_matrix: Full similarity matrix (n_users, n_users)
        user_ids: List of user IDs
        sections_dict: Dictionary mapping user_id to their sections
        instruction: Instruction for what kind of matching to do
        goal: Goal instruction for matching
        prompts_config_path: Path to prompts.yaml (alternative: prompt_template)
        llm_wrapper: LLM wrapper instance
        model: LLM model name
        max_n_llm_evaluations_per_profile: Max number of evaluations per profile
        global_cap: Global max pairs to score
        n_profiles_to_score_together: Number of profiles to score together in each batch
        force: Force re-evaluation
        prompt_template: The pair_scoring template string itself (keeps the
            transform free of file IO; takes precedence over prompts_config_path)
        excluded_pairs: Optional set of pair_ids to never score (novelty
            exclusions from match history)
        selected_pairs: Optional pre-selected candidate pairs. When given,
            the internal square-matrix selection is skipped entirely — this is
            how rectangular (member × pool) modes feed their own selection in.
        reasoning_effort: Reasoning effort for the scoring calls
            (config ``models.pair_reasoning_effort``; default "medium" — this
            is the quality-critical step).
        unscored_out: Optional list that collects every SELECTED pair that
            ends up without an LLM score (budget-capped grouping, exhausted
            retries, or a failed batch). Callers append these to the matching
            candidates so they keep their embedding-only weight instead of
            being dropped entirely (mirrors ``extract_sections(failed_out=)``).
        display_names: Optional {user_id: human name} map threaded into the
            scoring prompts so the model reasons over names instead of opaque
            ids (uuids); score JSON stays keyed by id.

    Returns:
        Dictionary mapping pair_id to PairScore
    """
    # Load prompt template
    if prompt_template is None:
        prompts_config = load_yaml(prompts_config_path)
        prompt_template = prompts_config['pair_scoring']

    # Select pairs for scoring based on budget
    if selected_pairs is None:
        selected_pairs = select_pairs_for_llm_scoring_optimal(
            similarity_matrix=similarity_matrix,
            user_ids=user_ids,
            max_n_llm_evaluations_per_profile=max_n_llm_evaluations_per_profile,
            global_cap=global_cap,
            excluded_pairs=excluded_pairs,
        )
    elif excluded_pairs:
        selected_pairs = [p for p in selected_pairs if p.pair_id not in excluded_pairs]
    
    if not selected_pairs:
        print("No pairs selected for LLM scoring")
        return {}
    
    # Create profile groups that cover all selected pairs. Each group is one LLM
    # call, so the call budget (global_cap = budgets.max_pair_llm_calls) caps the
    # number of groups.
    profile_groups = create_profile_groups_from_pairs(
        candidate_pairs=selected_pairs,
        n_profiles_to_score_together=n_profiles_to_score_together,
        max_groups=global_cap,
    )
    
    pair_scores = {}
    
    # Prepare all group prompts for parallel processing
    group_prompts = []
    group_cache_keys = []
    group_metadata = []  # Store (group_idx, pairs_to_score, alias_of)
    
    for group_idx, profile_group in enumerate(profile_groups):
        user_profiles = sorted(list(profile_group))
        
        # Find pairs in this group that we need to score
        pairs_to_score = [
            pair for pair in selected_pairs
            if pair.user1 in profile_group and pair.user2 in profile_group
        ]
        
        if not pairs_to_score:
            continue
        
        print(f"Preparing group {group_idx + 1}/{len(profile_groups)}: {len(user_profiles)} profiles, {len(pairs_to_score)} pairs")
        
        # Build batch scoring prompt
        prompt, alias_of = build_batch_scoring_prompt(
            user_profiles=user_profiles,
            sections_dict=sections_dict,
            instruction=instruction,
            prompt_template=prompt_template,
            goal=goal,
            display_names=display_names,
        )
        
        # Cache key = hash of the full prompt: it embeds the roster, every
        # profile's section CONTENT, the instruction, goal and template — so an
        # edited profile (or any prompt change) invalidates automatically. A
        # roster-only key would silently replay stale scores after a profile
        # edit. hash_text (sha256) — NOT the builtin hash(), which is salted
        # per process and would never hit across runs.
        cache_key = None if force else f"batch_score_{hash_text(prompt)}"
        
        group_prompts.append(prompt)
        group_cache_keys.append(cache_key)
        group_metadata.append((group_idx, pairs_to_score, alias_of))
    
    if not group_prompts:
        print("No groups to process")
        if unscored_out is not None:
            unscored_out.extend(selected_pairs)
        return {}

    print(f"Processing {len(group_prompts)} groups in parallel...")

    # Run all group scoring in parallel using batch_json_complete
    llm_wrapper.set_component("batch_pair_scoring")
    try:
        responses = _run_scoring_batch(
            llm_wrapper, group_prompts, group_cache_keys, model, reasoning_effort
        )
    except Exception as e:
        print(f"Error in parallel batch scoring: {e}")
        if unscored_out is not None:
            unscored_out.extend(selected_pairs)
        return {}

    missing: List[CandidatePair] = []
    for (group_idx, pairs_to_score, alias_of), response in zip(group_metadata, responses):
        still_missing = _merge_scores_from_response(response, pairs_to_score, pair_scores, alias_of)
        if still_missing:
            print(f"Warning: group {group_idx + 1} response missing "
                  f"{len(still_missing)}/{len(pairs_to_score)} pair score(s)")
        missing.extend(still_missing)

    # ---- Retry pass: re-ask ONLY the pairs whose scores came back missing or
    # unparsable. The transport layer already retries failed calls; this covers
    # the parsed-but-incomplete case. Note the retry round is baked into the
    # cache key — an earlier cached-but-incomplete response must never
    # short-circuit its own retry.
    group_width = max(2, n_profiles_to_score_together or 2)
    for retry_round in range(1, MAX_SCORE_RETRIES + 1):
        missing = [p for p in {p.pair_id: p for p in missing}.values()
                   if p.pair_id not in pair_scores]
        if not missing:
            break
        print(f"Retrying {len(missing)} unscored pair(s) "
              f"(retry {retry_round}/{MAX_SCORE_RETRIES})")

        # Pack the missing pairs into prompt-sized groups (≤ group_width users).
        retry_groups: List[List[CandidatePair]] = []
        current: List[CandidatePair] = []
        current_users: Set[str] = set()
        for pair in missing:
            pair_users = {pair.user1, pair.user2}
            if current and len(current_users | pair_users) > group_width:
                retry_groups.append(current)
                current, current_users = [], set()
            current.append(pair)
            current_users |= pair_users
        if current:
            retry_groups.append(current)

        retry_prompts, retry_keys, retry_pair_lists, retry_alias_maps = [], [], [], []
        for group_pairs in retry_groups:
            users = sorted({u for p in group_pairs for u in (p.user1, p.user2)})
            prompt, retry_alias_of = build_batch_scoring_prompt(
                user_profiles=users,
                sections_dict=sections_dict,
                instruction=instruction,
                prompt_template=prompt_template,
                goal=goal,
                pairs=[(p.user1, p.user2) for p in group_pairs],
                display_names=display_names,
            )
            retry_prompts.append(prompt)
            # Prompt hash covers roster + content + requested pairs; the retry
            # round stays in the key so a cached-but-incomplete response can
            # never short-circuit its own retry.
            retry_keys.append(
                None if force else
                f"batch_score_retry{retry_round}_{hash_text(prompt)}"
            )
            retry_pair_lists.append(group_pairs)
            retry_alias_maps.append(retry_alias_of)

        try:
            retry_responses = _run_scoring_batch(
                llm_wrapper, retry_prompts, retry_keys, model, reasoning_effort
            )
        except Exception as e:
            print(f"Error in scoring retry round {retry_round}: {e}")
            continue  # the same missing set goes into the next round

        next_missing: List[CandidatePair] = []
        for group_pairs, retry_alias_of, response in zip(
            retry_pair_lists, retry_alias_maps, retry_responses
        ):
            next_missing.extend(
                _merge_scores_from_response(response, group_pairs, pair_scores, retry_alias_of)
            )
        missing = next_missing

    missing = [p for p in {p.pair_id: p for p in missing}.values()
               if p.pair_id not in pair_scores]
    if missing:
        print(f"⚠️  WARNING: {len(missing)} pair(s) still unscored after "
              f"{MAX_SCORE_RETRIES} retries (they fall back to embedding-only "
              f"weight): {sorted(p.pair_id for p in missing)[:10]}")

    # Report EVERY selected-but-unscored pair (retry-exhausted AND pairs that
    # never made it into a group under the max_groups budget) so callers can
    # keep them as embedding-only matching candidates.
    if unscored_out is not None:
        unscored_out.extend(
            p for p in selected_pairs if p.pair_id not in pair_scores
        )

    print(f"Successfully scored {len(pair_scores)} pairs with batch LLM scoring")
    return pair_scores


def create_sections_dict(extracted_sections: List[ExtractedSections]) -> Dict[str, Dict[str, str]]:
    """Convert extracted sections list to dictionary for easy lookup."""
    return {
        profile.id: profile.sections
        for profile in extracted_sections
    }