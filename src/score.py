"""LLM-based pair scoring for candidate pairs."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import asyncio
import numpy as np

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
    intro: str
    starter_topics: str


def build_pair_scoring_prompt(
    user1_sections: Dict[str, str],
    user2_sections: Dict[str, str],
    user1_id: str,
    user2_id: str,
    instruction: str,
    prompt_template: str,
    goal: str
) -> str:
    """Build prompt for LLM pair scoring."""
    
    # Format sections nicely
    def format_sections(sections: Dict[str, str], user_id: str) -> str:
        lines = [f"Profile of {user_id}:"]
        for section_name, content in sections.items():
            if content and content.strip() and content != "Not specified":
                lines.append(f"  {section_name.title()}: {content}")
        return "\n".join(lines)
    
    user1_text = format_sections(user1_sections, user1_id)
    user2_text = format_sections(user2_sections, user2_id)
    
    # Build the complete prompt using template with all variables
    prompt = prompt_template.format(
        instruction=instruction,
        goal=goal,
        user_a_name=user1_id,
        user_b_name=user2_id,
        user1_text=user1_text,
        user2_text=user2_text
    )
    
    return prompt


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
    force: bool = False
) -> Dict[str, PairScore]:
    """
    Score pairs using LLM, selecting optimal subset from full similarity matrix.
    
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
    
    pair_scores = {}
    
    if selected_pairs:
        # Prepare batch data
        prompts = []
        cache_keys = []
        valid_pairs = []
        
        for pair in selected_pairs:
            # Get sections for both users
            user1_sections = sections_dict.get(pair.user1, {})
            user2_sections = sections_dict.get(pair.user2, {})
            
            if not user1_sections or not user2_sections:
                print(f"Warning: Missing sections for pair {pair.pair_id}")
                continue
            
            # Build scoring prompt
            prompt = build_pair_scoring_prompt(
                user1_sections=user1_sections,
                user2_sections=user2_sections,
                user1_id=pair.user1,
                user2_id=pair.user2,
                instruction=instruction,
                prompt_template=prompt_template,
                goal=goal
            )
            prompts.append(prompt)
            
            cache_key = None if force else f"score_{pair.pair_id}_{hash(instruction)}"
            cache_keys.append(cache_key)
            
            valid_pairs.append(pair)
        
        if valid_pairs:
            # Run batch scoring
            llm_wrapper.set_component("pair_scoring")
            
            try:
                responses = asyncio.run(
                    llm_wrapper.batch_json_complete(
                        prompts=prompts,
                        model=model,
                        cache_keys=cache_keys,
                        batch_size=16
                    )
                )
                
                # Process batch responses
                for pair, response in zip(valid_pairs, responses):
                    try:
                        if isinstance(response, Exception):
                            raise response
                        
                        # Validate response
                        score = float(response.get('score', 0.0))
                        intro = str(response.get('intro', ''))
                        starter_topics = str(response.get('starter_topics', ''))
                        
                        # Create PairScore
                        pair_score = PairScore(
                            pair_id=pair.pair_id,
                            user1=pair.user1,
                            user2=pair.user2,
                            embed_score=pair.similarity_score,
                            score=max(0.0, min(1.0, score)),
                            intro=intro.strip(),
                            starter_topics=starter_topics
                        )
                        
                        pair_scores[pair.pair_id] = pair_score
                        
                    except Exception as e:
                        print(f"Error processing response for pair {pair.pair_id}: {e}")
                        continue
                
            except Exception as e:
                print(f"Error in batch scoring: {e}")
                return {}
    
    print(f"Successfully scored {len(pair_scores)} pairs with LLM")
    return pair_scores


def create_sections_dict(extracted_sections: List[ExtractedSections]) -> Dict[str, Dict[str, str]]:
    """Convert extracted sections list to dictionary for easy lookup."""
    return {
        profile.id: profile.sections
        for profile in extracted_sections
    }