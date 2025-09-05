"""Generate candidate pairs using fused similarity from embeddings."""

import numpy as np
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

from utils import cosine_matrix, stable_pair_id


@dataclass
class CandidatePair:
    """A candidate user pair with similarity score."""
    user1: str
    user2: str
    similarity_score: float
    pair_id: str
    
    @classmethod
    def create(cls, user1: str, user2: str, score: float) -> 'CandidatePair':
        """Create candidate pair with stable ID."""
        return cls(
            user1=user1,
            user2=user2,
            similarity_score=score,
            pair_id=stable_pair_id(user1, user2)
        )


def compute_fused_similarity_matrix(
    embeddings: np.ndarray,  # shape: (n_users, n_sections, embedding_dim)
    section_names: List[str],
    section_weights: Dict[str, float]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute fused similarity matrix across all sections.
    
    Args:
        embeddings: User embeddings array
        section_names: Names of sections
        section_weights: Weights for each section in final score (negative weights for dissimilarity)
        
    Returns:
        Tuple of (fused_matrix, section_matrices_dict)
        - fused_matrix: Fused similarity matrix of shape (n_users, n_users)
        - section_matrices_dict: Dict with individual section matrices and weights
    """
    n_users, n_sections, embedding_dim = embeddings.shape
    
    # Compute similarity matrix for each section
    section_matrices = {}
    
    for section_idx, section_name in enumerate(section_names):
        # Extract embeddings for this section
        section_embeddings = embeddings[:, section_idx, :]  # (n_users, embedding_dim)
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_matrix(section_embeddings)
        
        section_matrices[section_name] = similarity_matrix
    
    # Fuse matrices using weighted combination
    fused_matrix = np.zeros((n_users, n_users))
    total_weight = 0.0
    
    for section_name, weight in section_weights.items():
        if section_name in section_matrices:
            fused_matrix += weight * section_matrices[section_name]
            total_weight += weight
        else:
            print(f"Warning: Section '{section_name}' not found in embeddings")
    
    # Normalize by total weight
    if total_weight > 0:
        fused_matrix /= total_weight
    
    # Return both fused matrix and individual matrices with weights
    matrices_dict = {
        'section_matrices': section_matrices,
        'section_weights': section_weights,
        'combined_matrix': fused_matrix
    }
    
    return fused_matrix, matrices_dict


def get_top_k_candidates_per_user(
    similarity_matrix: np.ndarray,
    user_ids: List[str],
    k: int
) -> List[CandidatePair]:
    """
    Get top-K candidates for each user and create symmetric candidate set.
    Optimized to avoid redundant computations by processing upper triangle only.
    
    Args:
        similarity_matrix: Similarity matrix (n_users, n_users)
        user_ids: List of user IDs
        k: Number of top candidates per user
        
    Returns:
        List of unique candidate pairs
    """
    n_users = len(user_ids)
    
    # Pre-compute top-K candidates for all users efficiently
    # Set diagonal to -1 to exclude self-similarity
    similarity_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_no_diag, -1.0)
    
    # Get top-K indices for all users at once
    top_k_indices_all = np.argsort(similarity_no_diag, axis=1)[:, -k:]
    
    # Process only upper triangle to avoid duplicates
    candidate_pairs = []
    
    for i in range(n_users):
        user_i = user_ids[i]
        top_k_i = set(top_k_indices_all[i])
        
        for j in range(i + 1, n_users):  # Only upper triangle
            user_j = user_ids[j]
            top_k_j = set(top_k_indices_all[j])
            
            # Check if either user selects the other as a top-K candidate
            if j in top_k_i or i in top_k_j:
                score = similarity_matrix[i, j]
                if score > 0:  # Only positive similarities
                    pair = CandidatePair.create(user_i, user_j, score)
                    candidate_pairs.append(pair)
    
    # Sort by similarity score (descending)
    candidate_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
    
    print(f"Generated {len(candidate_pairs)} unique candidate pairs (top-{k} per user)")
    return candidate_pairs


def apply_recipe(
    embeddings: np.ndarray,
    section_names: List[str],
    recipe_config: Dict
) -> Tuple[np.ndarray, Dict]:
    """
    Apply a specific recipe to compute similarity matrix.
    
    Args:
        embeddings: User embeddings
        section_names: Section names
        recipe_config: Recipe configuration from config
        
    Returns:
        Tuple of (similarity_matrix, matrices_dict)
    """
    recipe_type = 'custom'
    section_weights = recipe_config.get('section_weights', {})
    
    print(f"Applying recipe: {recipe_type}")
    print(f"Section weights: {section_weights}")
    
    # Apply the recipe
    similarity_matrix, matrices_dict = compute_fused_similarity_matrix(
        embeddings=embeddings,
        section_names=section_names,
        section_weights=section_weights
    )
    
    return similarity_matrix, matrices_dict

def generate_similarity_matrix(
    embeddings: np.ndarray,
    user_ids: List[str],
    section_names: List[str],
    recipe_config: Dict
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Generate similarity matrix for all user pairs.
    
    Args:
        embeddings: User embeddings
        user_ids: User IDs
        section_names: Section names
        recipe_config: Recipe configuration
        
    Returns:
        Tuple of (similarity_matrix, user_ids, matrices_dict)
    """
    # Apply recipe to get similarity matrix
    similarity_matrix, matrices_dict = apply_recipe(
        embeddings=embeddings,
        section_names=section_names,
        recipe_config=recipe_config
    )
    
    print(f"Generated similarity matrix of shape {similarity_matrix.shape}")
    
    return similarity_matrix, user_ids, matrices_dict