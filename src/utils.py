"""Utility functions for prompt-mesh matching system."""

import json
import hashlib
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a set of vectors using vectorized operations."""
    # Normalize vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Handle zero vectors
    norms = np.where(norms == 0, 1, norms)
    normalized_vectors = vectors / norms
    
    # Compute similarity matrix with single matrix multiplication
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    
    return similarity_matrix


def stable_pair_id(u: str, v: str) -> str:
    """Create a stable pair ID regardless of order."""
    return f"{min(u, v)}_{max(u, v)}"


def hash_text(text: str) -> str:
    """Create a stable hash for text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def parse_cross_key(cross_key: str) -> Tuple[str, str]:
    """Split a cross-section key into (source, target).

    Preferred form uses ``->`` so multi-word section names parse correctly
    (e.g. ``"needs->final_project"``). The legacy ``source_target`` form is
    still accepted, but only when it splits cleanly into exactly two parts —
    multi-word names under the legacy form are ambiguous and rejected.
    """
    parts = cross_key.split("->") if "->" in cross_key else cross_key.split("_")
    parts = [p.strip() for p in parts]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Invalid cross_section_weights key '{cross_key}': expected "
            "'source->target' (or legacy 'source_target')"
        )
    return parts[0], parts[1]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    words = len(text.split())
    return int(words * 1.3)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save data to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def truncate_words(text: str, max_words: int) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def get_cache_path(cache_dir: Path, key: str, suffix: str = '.json') -> Path:
    """Generate cache file path from key."""
    return cache_dir / f"{key}{suffix}"


def filter_active_sections(sections_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of sections_config with only active sections."""
    filtered = {k: v for k, v in sections_config['sections'].items() if v.get('active', True)}
    return {**sections_config, 'sections': filtered}


def generate_schema_hint_from_sections(sections_config: Dict[str, Any]) -> str:
    """Generate schema hint JSON string from sections config."""
    schema_dict = {}
    for section_name in sections_config['sections'].keys():
        schema_dict[section_name] = "..."
    return json.dumps(schema_dict)


def generate_json_structure_from_sections(sections_config: Dict[str, Any]) -> str:
    """Generate JSON structure string for prompts from sections config."""
    lines = ["{"]
    section_names = list(sections_config['sections'].keys())
    for i, section_name in enumerate(section_names):
        comma = "," if i < len(section_names) - 1 else ""
        lines.append(f'  "{section_name}": "extracted {section_name} text"{comma}')
    lines.append("}")
    return '\n'.join(lines)


def get_section_names_list(sections_config: Dict[str, Any]) -> List[str]:
    """Get ordered list of section names from sections config."""
    return list(sections_config['sections'].keys())


def normalize_scores_with_reference_distribution(
    reference_scores: np.ndarray,
    target_scores: np.ndarray,
    selected_reference_scores: np.ndarray
) -> np.ndarray:
    """
    Normalize target scores using reference distribution as the baseline.
    
    This function implements a statistically principled score normalization:
    1. Normalize reference scores to 0-1 range
    2. Find the range that selected reference scores occupy in normalized space
    3. Map target scores to occupy the same range as selected reference scores
    
    Args:
        reference_scores: Full distribution of reference scores (e.g., all matrix scores)
        target_scores: Scores to be normalized (e.g., LLM scores)
        selected_reference_scores: Subset of reference scores that correspond to target scores
        
    Returns:
        Normalized target scores mapped to the same range as selected reference scores
    """
    # Step 1: Normalize reference scores to 0-1
    ref_min, ref_max = reference_scores.min(), reference_scores.max()
    if ref_max == ref_min:
        # Handle edge case where all reference scores are identical
        return np.full_like(target_scores, 0.5)
    
    # Step 2: Find range of selected reference scores in normalized space
    selected_normalized = (selected_reference_scores - ref_min) / (ref_max - ref_min)
    selected_min, selected_max = selected_normalized.min(), selected_normalized.max()
    
    # Step 3: Map target scores to occupy the same range as selected reference scores
    target_min, target_max = target_scores.min(), target_scores.max()
    if target_max == target_min:
        # Handle edge case where all target scores are identical
        target_normalized = np.full_like(target_scores, (selected_min + selected_max) / 2)
    else:
        # Linear mapping: target_scores [target_min, target_max] -> [selected_min, selected_max]
        target_normalized = selected_min + (target_scores - target_min) / (target_max - target_min) * (selected_max - selected_min)
    
    return target_normalized


def get_score_normalization_stats(
    reference_scores: np.ndarray,
    target_scores: np.ndarray,
    selected_reference_scores: np.ndarray,
    normalized_target_scores: np.ndarray
) -> Dict[str, Any]:
    """
    Get statistics about the score normalization process for debugging/analysis.
    
    Returns:
        Dictionary with normalization statistics
    """
    ref_min, ref_max = reference_scores.min(), reference_scores.max()
    selected_normalized = (selected_reference_scores - ref_min) / (ref_max - ref_min)
    
    return {
        'reference_range': (float(ref_min), float(ref_max)),
        'reference_normalized_range': (0.0, 1.0),
        'selected_reference_range': (float(selected_reference_scores.min()), float(selected_reference_scores.max())),
        'selected_normalized_range': (float(selected_normalized.min()), float(selected_normalized.max())),
        'target_original_range': (float(target_scores.min()), float(target_scores.max())),
        'target_normalized_range': (float(normalized_target_scores.min()), float(normalized_target_scores.max())),
        'n_reference_scores': len(reference_scores),
        'n_selected_scores': len(selected_reference_scores),
        'n_target_scores': len(target_scores)
    }


def prepare_normalized_scores(
    candidates: List,  # List[CandidatePair] - avoiding import here
    llm_scores: Dict[str, Any],  # Dict[str, PairScore] - avoiding import here
    full_similarity_matrix: np.ndarray = None,
    all_user_ids: List[str] = None
) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    """
    Prepare normalized embedding and LLM scores for final weight computation.
    
    Args:
        candidates: All candidate pairs
        llm_scores: LLM scores by pair_id
        full_similarity_matrix: Full similarity matrix for reference distribution (optional)
        all_user_ids: All user IDs corresponding to similarity matrix (optional)
        
    Returns:
        Tuple of (normalized_embed_lookup, normalized_llm_lookup, normalization_applied)
    """
    # Check if we can apply normalization
    should_normalize = (
        full_similarity_matrix is not None and 
        all_user_ids is not None and 
        len(llm_scores) > 0
    )
    
    normalized_embed_lookup = {}
    normalized_llm_lookup = {}
    
    if not should_normalize:
        # Return original scores without normalization
        for candidate in candidates:
            normalized_embed_lookup[candidate.pair_id] = candidate.similarity_score
            if candidate.pair_id in llm_scores:
                normalized_llm_lookup[candidate.pair_id] = llm_scores[candidate.pair_id].score
        return normalized_embed_lookup, normalized_llm_lookup, False
    
    print("Applying reference distribution normalization...")
    
    # Extract all scores from full similarity matrix (upper triangle only)
    n_users = len(all_user_ids)
    all_matrix_scores = []
    for i in range(n_users):
        for j in range(i + 1, n_users):
            all_matrix_scores.append(full_similarity_matrix[i, j])
    all_matrix_scores = np.array(all_matrix_scores)
    
    # Get reference range
    ref_min, ref_max = all_matrix_scores.min(), all_matrix_scores.max()
    
    if ref_max <= ref_min:
        print("Warning: All matrix scores identical, skipping normalization")
        # Fallback to original scores
        for candidate in candidates:
            normalized_embed_lookup[candidate.pair_id] = candidate.similarity_score
            if candidate.pair_id in llm_scores:
                normalized_llm_lookup[candidate.pair_id] = llm_scores[candidate.pair_id].score
        return normalized_embed_lookup, normalized_llm_lookup, False
    
    # Normalize all embedding scores to 0-1
    for candidate in candidates:
        embed_score_normalized = (candidate.similarity_score - ref_min) / (ref_max - ref_min)
        normalized_embed_lookup[candidate.pair_id] = embed_score_normalized
    
    # Process LLM scores if available
    if llm_scores:
        # Extract embedding and LLM scores for candidates that have both
        candidates_with_llm = [c for c in candidates if c.pair_id in llm_scores]
        
        if candidates_with_llm:
            selected_embed_scores = np.array([c.similarity_score for c in candidates_with_llm])
            actual_llm_scores = np.array([llm_scores[c.pair_id].score for c in candidates_with_llm])
            
            # Normalize LLM scores to match the selected embedding score range
            normalized_llm_scores = normalize_scores_with_reference_distribution(
                reference_scores=all_matrix_scores,
                target_scores=actual_llm_scores,
                selected_reference_scores=selected_embed_scores
            )
            
            # Create lookup for normalized LLM scores
            for candidate, norm_llm_score in zip(candidates_with_llm, normalized_llm_scores):
                normalized_llm_lookup[candidate.pair_id] = float(norm_llm_score)
            
            # Print normalization statistics
            stats = get_score_normalization_stats(
                reference_scores=all_matrix_scores,
                target_scores=actual_llm_scores,
                selected_reference_scores=selected_embed_scores,
                normalized_target_scores=normalized_llm_scores
            )
            print("📊 Score normalization stats:")
            print(f"   Matrix range: [{stats['reference_range'][0]:.3f}, {stats['reference_range'][1]:.3f}]")
            print(f"   Selected matrix range (normalized): [{stats['selected_normalized_range'][0]:.3f}, {stats['selected_normalized_range'][1]:.3f}]")
            print(f"   LLM original range: [{stats['target_original_range'][0]:.3f}, {stats['target_original_range'][1]:.3f}]")
            print(f"   LLM normalized range: [{stats['target_normalized_range'][0]:.3f}, {stats['target_normalized_range'][1]:.3f}]")
    
    return normalized_embed_lookup, normalized_llm_lookup, True