"""LLM-based pair scoring for candidate pairs."""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import random
import asyncio
import numpy as np
from itertools import combinations

from candidate import CandidatePair
from extract import ExtractedSections
from llm import LLMWrapper
from utils import load_yaml, stable_pair_id


@dataclass 
class PairScore:
    """LLM score for a user pair."""
    pair_id: str
    user1: str
    user2: str
    embed_score: float
    score: float


def create_profile_groups_from_pairs(
    candidate_pairs: List[CandidatePair],
    n_profiles_to_score_together: int
) -> List[Set[str]]:
    """Create groups of user profiles to score together, ensuring all candidate pairs are covered."""
    
    # Extract all unique users from candidate pairs
    all_users = set()
    for pair in candidate_pairs:
        all_users.add(pair.user1)
        all_users.add(pair.user2)
    
    users_list = sorted(list(all_users))  # Sort for consistent ordering
    
    groups = []
    covered_pairs = set()
    
    # Create groups greedily - each group should cover as many uncovered pairs as possible
    while len(covered_pairs) < len(candidate_pairs):
        if len(users_list) <= n_profiles_to_score_together:
            # Add remaining users as final group
            groups.append(set(users_list))
            break
        
        # For each possible group of n_profiles_to_score_together users,
        # calculate how many uncovered candidate pairs it would score
        best_group = None
        best_new_pairs_count = 0
        
        # Try different starting points to avoid always picking the same users
        start_idx = len(groups) % len(users_list)
        
        for i in range(len(users_list) - n_profiles_to_score_together + 1):
            group_start = (start_idx + i) % (len(users_list) - n_profiles_to_score_together + 1)
            test_group = set(users_list[group_start:group_start + n_profiles_to_score_together])
            
            # Count how many uncovered pairs this group would score
            new_pairs_count = 0
            for pair in candidate_pairs:
                if pair.pair_id not in covered_pairs:
                    if pair.user1 in test_group and pair.user2 in test_group:
                        new_pairs_count += 1
            
            if new_pairs_count > best_new_pairs_count:
                best_new_pairs_count = new_pairs_count
                best_group = test_group
        
        # If no group covers any new pairs, take users involved in most uncovered pairs
        if best_group is None or best_new_pairs_count == 0:
            user_pair_counts = {}
            for pair in candidate_pairs:
                if pair.pair_id not in covered_pairs:
                    user_pair_counts[pair.user1] = user_pair_counts.get(pair.user1, 0) + 1
                    user_pair_counts[pair.user2] = user_pair_counts.get(pair.user2, 0) + 1
            
            # Take top n_profiles_to_score_together users by uncovered pair count
            sorted_users = sorted(user_pair_counts.items(), key=lambda x: x[1], reverse=True)
            best_group = set([user for user, _ in sorted_users[:n_profiles_to_score_together]])
        
        groups.append(best_group)
        
        # Mark pairs in this group as covered
        for pair in candidate_pairs:
            if pair.user1 in best_group and pair.user2 in best_group:
                covered_pairs.add(pair.pair_id)
        
        # Remove users from the list if they've been heavily used
        # This helps ensure good coverage across all users
        users_in_multiple_groups = set()
        for user in best_group:
            user_group_count = sum(1 for group in groups if user in group)
            if user_group_count >= 2:  # User has been in multiple groups
                users_in_multiple_groups.add(user)
        
        for user in users_in_multiple_groups:
            if user in users_list and len(users_list) > n_profiles_to_score_together:
                users_list.remove(user)
    
    print(f"Created {len(groups)} profile groups covering {len(covered_pairs)}/{len(candidate_pairs)} candidate pairs")
    for i, group in enumerate(groups):
        pairs_in_group = sum(1 for pair in candidate_pairs 
                           if pair.user1 in group and pair.user2 in group)
        print(f"  Group {i+1}: {len(group)} profiles, {pairs_in_group} pairs")
    
    return groups


def build_batch_scoring_prompt(
    user_profiles: List[str],  # list of user_ids
    sections_dict: Dict[str, Dict[str, str]],
    instruction: str,
    prompt_template: str,
    goal: str
) -> Tuple[str, Dict[str, float]]:
    """Build prompt for batch LLM scoring of multiple profiles."""
    
    # Format all user profiles
    def format_sections(sections: Dict[str, str], user_id: str) -> str:
        lines = [f"Profile of {user_id}:"]
        for section_name, content in sections.items():
            if content and content.strip() and content != "Not specified":
                lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)
    
    # Create XML formatted profiles
    profiles_xml = "<profiles>\n"
    for user_id in user_profiles:
        user_sections = sections_dict.get(user_id, {})
        profile_text = format_sections(user_sections, user_id)
        profiles_xml += f"  <profile id=\"{user_id}\">\n    {profile_text.replace(chr(10), chr(10) + '    ')}\n  </profile>\n"
    profiles_xml += "</profiles>"
    
    # Generate all possible pairs and create JSON format hint
    pairs = list(combinations(user_profiles, 2))
    pair_scores = {}
    json_format = {}
    
    for user1, user2 in pairs:
        pair_key = f"{user1}_{user2}"
        pair_scores[pair_key] = 0.0  # placeholder
        json_format[pair_key] = 0.0
    
    json_format_hint = str(json_format).replace("0.0", "0..1")
    
    # Build the complete prompt
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_profiles_xml_formatted=profiles_xml,
        json_format_hint=json_format_hint
    )
    
    return prompt, pair_scores


def select_pairs_for_llm_scoring_optimal(
    similarity_matrix: np.ndarray,
    user_ids: List[str],
    max_n_llm_evaluations_per_profile: int,
    global_cap: int
) -> List[CandidatePair]:
    """
    Select optimal subset of pairs for LLM scoring using greedy per-profile selection.
    Works directly with similarity matrix instead of pre-filtered candidates.
    
    Args:
        similarity_matrix: Full similarity matrix (n_users, n_users)
        user_ids: List of user IDs
        max_n_llm_evaluations_per_profile: Maximum number of evaluations per profile
        global_cap: Global max pairs to score (for backward compatibility)
        
    Returns:
        Optimally selected pairs for scoring
    """
    n_users = len(user_ids)
    if n_users == 0:
        return []
    
    print(f"Greedy pair selection for {n_users} users with max {max_n_llm_evaluations_per_profile} evaluations per profile")
    
    # Create all possible pairs with their scores (upper triangle only to avoid duplicates)
    all_pairs = []
    for i in range(n_users):
        for j in range(i + 1, n_users):
            score = similarity_matrix[i, j]
            if score > 0:  # Only positive similarities
                pair = CandidatePair.create(user_ids[i], user_ids[j], score)
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
    prompts_config_path: str,
    llm_wrapper: LLMWrapper,
    model: str,
    max_n_llm_evaluations_per_profile: int,
    global_cap: int,
    n_profiles_to_score_together: int,
    force: bool = False
) -> Dict[str, PairScore]:
    """
    Score pairs using LLM with batch scoring approach.
    
    Args:
        similarity_matrix: Full similarity matrix (n_users, n_users)
        user_ids: List of user IDs
        sections_dict: Dictionary mapping user_id to their sections
        instruction: Instruction for what kind of matching to do
        goal: Goal instruction for matching
        prompts_config_path: Path to prompts.yaml
        llm_wrapper: LLM wrapper instance
        model: LLM model name
        max_n_llm_evaluations_per_profile: Max number of evaluations per profile
        global_cap: Global max pairs to score
        n_profiles_to_score_together: Number of profiles to score together in each batch
        force: Force re-evaluation
        
    Returns:
        Dictionary mapping pair_id to PairScore
    """
    # Load prompt template
    prompts_config = load_yaml(prompts_config_path)
    prompt_template = prompts_config['pair_scoring']
    
    # Select pairs for scoring based on budget
    selected_pairs = select_pairs_for_llm_scoring_optimal(
        similarity_matrix=similarity_matrix,
        user_ids=user_ids,
        max_n_llm_evaluations_per_profile=max_n_llm_evaluations_per_profile,
        global_cap=global_cap
    )
    
    if not selected_pairs:
        print("No pairs selected for LLM scoring")
        return {}
    
    # Create profile groups that cover all selected pairs
    profile_groups = create_profile_groups_from_pairs(
        candidate_pairs=selected_pairs,
        n_profiles_to_score_together=n_profiles_to_score_together
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
            goal=goal
        )
        
        # Create cache key for this group
        group_signature = "_".join(sorted(user_profiles))
        cache_key = None if force else f"batch_score_{hash(group_signature)}_{hash(instruction)}"
        
        group_prompts.append(prompt)
        group_cache_keys.append(cache_key)
        group_metadata.append((group_idx, profile_group, pairs_to_score))
    
    if not group_prompts:
        print("No groups to process")
        return {}
    
    print(f"Processing {len(group_prompts)} groups in parallel...")
    
    # Run all group scoring in parallel using batch_json_complete
    llm_wrapper.set_component("batch_pair_scoring")
    
    async def _async_score_all_groups():
        """Score all groups in parallel with proper async cleanup."""
        try:
            responses = await llm_wrapper.batch_json_complete(
                prompts=group_prompts,
                model=model,
                cache_keys=group_cache_keys,
                verbosity=2
            )
            return responses
        finally:
            # Force cleanup of any remaining tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    
    try:
        responses = asyncio.run(_async_score_all_groups())
        
        # Process all group responses
        for (group_idx, profile_group, pairs_to_score), response in zip(group_metadata, responses):
            try:
                if isinstance(response, Exception):
                    print(f"Error in group {group_idx + 1}: {response}")
                    continue
                
                # Extract scores for pairs in this group
                for pair in pairs_to_score:
                    pair_key1 = f"{pair.user1}_{pair.user2}"
                    pair_key2 = f"{pair.user2}_{pair.user1}"
                    
                    score = None
                    if pair_key1 in response:
                        score = response[pair_key1]
                    elif pair_key2 in response:
                        score = response[pair_key2]
                    
                    if score is not None:
                        try:
                            score_val = float(score)
                            pair_score = PairScore(
                                pair_id=pair.pair_id,
                                user1=pair.user1,
                                user2=pair.user2,
                                embed_score=pair.similarity_score,
                                score=max(0.0, min(1.0, score_val))
                            )
                            pair_scores[pair.pair_id] = pair_score
                        except (ValueError, TypeError) as e:
                            print(f"Error parsing score for pair {pair.pair_id}: {e}")
                            continue
                    else:
                        print(f"Warning: No score found for pair {pair.pair_id} in group {group_idx + 1} response")
            
            except Exception as e:
                print(f"Error processing group {group_idx + 1}: {e}")
                continue
    
    except Exception as e:
        print(f"Error in parallel batch scoring: {e}")
        return {}
    
    print(f"Successfully scored {len(pair_scores)} pairs with batch LLM scoring")
    return pair_scores


def create_sections_dict(extracted_sections: List[ExtractedSections]) -> Dict[str, Dict[str, str]]:
    """Convert extracted sections list to dictionary for easy lookup."""
    return {
        profile.id: profile.sections
        for profile in extracted_sections
    }