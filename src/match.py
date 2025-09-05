"""Greedy b-matching algorithm to create final matches."""

from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

from candidate import CandidatePair
from score import PairScore


@dataclass
class Edge:
    """Final matched edge between two users."""
    user1: str
    user2: str
    pair_id: str
    final_weight: float
    embed_score: float
    llm_score: float
    intro: str
    starter_topics: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'user1': self.user1,
            'user2': self.user2,
            'pair_id': self.pair_id,
            'final_weight': self.final_weight,
            'embed_score': self.embed_score,
            'llm_score': self.llm_score,
            'intro': self.intro,
            'starter_topics': self.starter_topics
        }


def compute_final_weights(
    candidates: List[CandidatePair],
    llm_scores: Dict[str, PairScore],
    embed_weight: float,
    llm_weight: float
) -> List[Edge]:
    """
    Compute final weights blending embedding and LLM scores.
    
    Args:
        candidates: All candidate pairs
        llm_scores: LLM scores by pair_id
        embed_weight: Weight for embedding score in final blend
        llm_weight: Weight for LLM score in final blend
        
    Returns:
        List of Edge objects with final weights
    """
    edges = []
    
    for candidate in candidates:
        pair_id = candidate.pair_id
        embed_score = candidate.similarity_score
        
        # Get LLM score if available
        llm_score_obj = llm_scores.get(pair_id)
        
        if llm_score_obj:
            # Use LLM score
            llm_score = llm_score_obj.score
            intro = llm_score_obj.intro
            starter_topics = llm_score_obj.starter_topics
            
            # Blend scores
            final_weight = embed_weight * embed_score + llm_weight * llm_score
        else:
            # Use only embedding score
            llm_score = 0.0
            intro = f"Suggested match based on profile similarity"
            starter_topics = "• Discuss shared interests • Talk about goals • Share experiences"
            
            final_weight = embed_score  # Only embedding score
        
        edge = Edge(
            user1=candidate.user1,
            user2=candidate.user2,
            pair_id=pair_id,
            final_weight=final_weight,
            embed_score=embed_score,
            llm_score=llm_score,
            intro=intro,
            starter_topics=starter_topics
        )
        edges.append(edge)
    
    return edges


def greedy_b_matching(
    edges: List[Edge],
    b_min: int,
    b_max: int,
    all_users: Set[str]
) -> List[Edge]:
    """
    Greedy b-matching algorithm.
    
    Args:
        edges: All candidate edges with final weights
        b_min: Minimum degree per user
        b_max: Maximum degree per user
        all_users: Set of all user IDs
        
    Returns:
        Selected edges forming b-matching
    """
    # Sort edges by final weight (descending)
    sorted_edges = sorted(edges, key=lambda e: e.final_weight, reverse=True)
    
    # Track degree of each user
    user_degrees = defaultdict(int)
    selected_edges = []
    
    print(f"Starting greedy b-matching with {len(sorted_edges)} edges")
    print(f"Target degree range: [{b_min}, {b_max}] per user")
    
    # Phase 1: Greedy selection while respecting b_max
    for edge in sorted_edges:
        user1_degree = user_degrees[edge.user1]
        user2_degree = user_degrees[edge.user2]
        
        # Check if adding this edge would violate b_max
        if user1_degree < b_max and user2_degree < b_max:
            selected_edges.append(edge)
            user_degrees[edge.user1] += 1
            user_degrees[edge.user2] += 1
    
    print(f"Phase 1: Selected {len(selected_edges)} edges")
    
    # Phase 2: Backfill users below b_min
    users_below_min = [
        user for user in all_users 
        if user_degrees[user] < b_min
    ]
    
    if users_below_min:
        print(f"Phase 2: Backfilling {len(users_below_min)} users below minimum degree")
        
        # Create a mapping of available edges for each user
        available_edges = defaultdict(list)
        for edge in sorted_edges:
            if edge not in selected_edges:
                available_edges[edge.user1].append(edge)
                available_edges[edge.user2].append(edge)
        
        for user in users_below_min:
            current_degree = user_degrees[user]
            needed = b_min - current_degree
            
            # Find best available edges for this user
            candidates = [
                edge for edge in available_edges[user]
                if edge not in selected_edges
            ]
            
            # Sort by weight and try to add
            candidates.sort(key=lambda e: e.final_weight, reverse=True)
            
            added = 0
            for edge in candidates:
                if added >= needed:
                    break
                    
                other_user = edge.user2 if edge.user1 == user else edge.user1
                
                # Check if other user still has capacity
                if user_degrees[other_user] < b_max:
                    selected_edges.append(edge)
                    user_degrees[edge.user1] += 1
                    user_degrees[edge.user2] += 1
                    added += 1
    
    # Final statistics
    final_degrees = {user: user_degrees[user] for user in all_users}
    avg_degree = sum(final_degrees.values()) / len(all_users) if all_users else 0
    
    users_at_min = sum(1 for d in final_degrees.values() if d >= b_min)
    users_at_max = sum(1 for d in final_degrees.values() if d == b_max)
    
    print(f"Final matching:")
    print(f"  Selected {len(selected_edges)} edges")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Users at/above b_min ({b_min}): {users_at_min}/{len(all_users)}")
    print(f"  Users at b_max ({b_max}): {users_at_max}/{len(all_users)}")
    
    return selected_edges


def create_matches(
    candidates: List[CandidatePair],
    llm_scores: Dict[str, PairScore],
    all_user_ids: List[str],
    matching_config: Dict,
    blending_config: Dict
) -> List[Edge]:
    """
    Full pipeline to create final matches.
    
    Args:
        candidates: All candidate pairs
        llm_scores: LLM scores by pair_id
        all_user_ids: All user IDs in the system
        matching_config: Matching configuration (b_min, b_max)
        blending_config: Blending configuration (weights)
        
    Returns:
        Final selected edges
    """
    # Compute final weights
    edges = compute_final_weights(
        candidates=candidates,
        llm_scores=llm_scores,
        embed_weight=blending_config['embed_weight'],
        llm_weight=blending_config['llm_weight']
    )
    
    print(f"Computed final weights for {len(edges)} edges")
    
    # Run greedy b-matching
    selected_edges = greedy_b_matching(
        edges=edges,
        b_min=matching_config['b_min'],
        b_max=matching_config['b_max'],
        all_users=set(all_user_ids)
    )
    
    return selected_edges