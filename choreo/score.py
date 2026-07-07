"""LLM-based pair scoring for candidate pairs."""

import json
from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from itertools import combinations

from .candidate import CandidatePair
from .llm import LLMWrapper, run_coro_blocking
from .utils import load_yaml, hash_text, is_absent
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


def build_batch_scoring_prompt(
    user_profiles: List[str],  # list of user_ids
    sections_dict: Dict[str, Dict[str, str]],
    instruction: str,
    prompt_template: str,
    goal: str,
    pairs: Optional[List[Tuple[str, str]]] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, float]]:
    """Build prompt for batch LLM scoring of multiple profiles.

    By default every C(n,2) pair among ``user_profiles`` is requested. Pass
    ``pairs`` to request only a subset — query mode uses this to score
    query↔candidate pairs without asking for candidate↔candidate scores.

    ``display_names`` maps user ids to human names for the prompt prose
    (profiles render as ``<profile id="…" name="…">``); the returned score
    JSON stays keyed by id either way. Without it (or for ids not in the map)
    the prompt is byte-identical to the pre-display_names shape, so existing
    caches stay warm.
    """
    display_names = display_names or {}

    # Format all user profiles
    def format_sections(sections: Dict[str, str], user_id: str) -> str:
        name = display_names.get(user_id)
        if name and name != user_id:
            lines = [f"Profile of {name} (id: {user_id}):"]
        else:
            lines = [f"Profile of {user_id}:"]
        for section_name, content in sections.items():
            if not is_absent(content):
                lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)

    # Create XML formatted profiles
    profiles_xml = "<profiles>\n"
    for user_id in user_profiles:
        user_sections = sections_dict.get(user_id, {})
        profile_text = format_sections(user_sections, user_id)
        name = display_names.get(user_id)
        tag_attrs = f"id=\"{user_id}\" name=\"{name}\"" if name and name != user_id else f"id=\"{user_id}\""
        profiles_xml += f"  <profile {tag_attrs}>\n    {profile_text.replace(chr(10), chr(10) + '    ')}\n  </profile>\n"
    profiles_xml += "</profiles>"

    # Generate requested pairs and create JSON format hint
    if pairs is None:
        pairs = list(combinations(user_profiles, 2))
    pair_scores = {}

    for user1, user2 in pairs:
        pair_key = f"{user1}_{user2}"
        pair_scores[pair_key] = 0.0  # placeholder

    json_format_hint = json.dumps({pair_key: "0..1" for pair_key in pair_scores})

    # Build the complete prompt
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_profiles_xml_formatted=profiles_xml,
        json_format_hint=json_format_hint
    )

    return prompt, pair_scores


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
) -> List[CandidatePair]:
    """Parse one batch response into ``pair_scores``; return the pairs whose
    score came back missing or unparsable (the retry input)."""
    if isinstance(response, Exception) or not isinstance(response, dict):
        print(f"Warning: unusable scoring response "
              f"({response if isinstance(response, Exception) else type(response).__name__})")
        return list(pairs)

    missing: List[CandidatePair] = []
    for pair in pairs:
        score = response.get(f"{pair.user1}_{pair.user2}",
                             response.get(f"{pair.user2}_{pair.user1}"))
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
    group_metadata = []  # Store (group_idx, profile_group, pairs_to_score)
    
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
        prompt, pair_template = build_batch_scoring_prompt(
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
        group_metadata.append((group_idx, profile_group, pairs_to_score))
    
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
    for (group_idx, profile_group, pairs_to_score), response in zip(group_metadata, responses):
        still_missing = _merge_scores_from_response(response, pairs_to_score, pair_scores)
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

        retry_prompts, retry_keys, retry_pair_lists = [], [], []
        for group_pairs in retry_groups:
            users = sorted({u for p in group_pairs for u in (p.user1, p.user2)})
            prompt, _ = build_batch_scoring_prompt(
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

        try:
            retry_responses = _run_scoring_batch(
                llm_wrapper, retry_prompts, retry_keys, model, reasoning_effort
            )
        except Exception as e:
            print(f"Error in scoring retry round {retry_round}: {e}")
            continue  # the same missing set goes into the next round

        next_missing: List[CandidatePair] = []
        for group_pairs, response in zip(retry_pair_lists, retry_responses):
            next_missing.extend(_merge_scores_from_response(response, group_pairs, pair_scores))
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