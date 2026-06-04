"""Generate candidate pairs using fused similarity from embeddings.

Similarity is RECTANGULAR at its core: a source user set scored against a
target user set (``fused[i][j]`` = "how well can target j help source i").
The legacy square cohort path is just the special case source == target —
``compute_fused_similarity_matrix`` reduces bit-exactly to the historical
behavior when no target is passed.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .utils import cosine_matrix, cosine_rect, stable_pair_id, parse_cross_key
from .schemas import EmbeddingsBundle, SimilarityResult


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
    embeddings: np.ndarray,  # SOURCE embeddings: (n_source, n_sections, embedding_dim)
    section_names: List[str],
    section_weights: Dict[str, float],
    cross_section_weights: Optional[Dict[str, float]] = None,
    hyde_embeddings: Optional[Dict[str, np.ndarray]] = None,
    target_embeddings: Optional[np.ndarray] = None,  # (n_target, n_sections, dim); None = square
    target_section_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute fused similarity between a source user set and a target user set.

    Square mode (the historical cohort path) is the default: omit
    ``target_embeddings`` and the target side IS the source side — this reduces
    bit-exactly to the pre-rectangular behavior.

    Rectangular mode: pass ``target_embeddings`` (and, if its section order
    differs, ``target_section_names``). Same-section terms become
    ``source_sec @ target_sec.T``; the directional cross term becomes
    ``source_hyde @ target_sec.T``. Nothing is symmetrized.

    The fused matrix is ASYMMETRIC whenever cross weights are present:
    fused[i][j] = "how well can target j help source i" != fused[j][i].

    Args:
        embeddings: Source embeddings array (n_source, n_sections, embedding_dim)
        section_names: Section names of the source embeddings (axis order)
        section_weights: Weights for same-section similarity
        cross_section_weights: Weights for cross-section similarity (e.g., {"needs_skills": 0.85})
        hyde_embeddings: SOURCE-side HyDE embeddings {cross_key: (n_source, n_descriptors, dim)}
        target_embeddings: Optional target embeddings (n_target, n_sections, dim)
        target_section_names: Section names of the target embeddings (defaults
            to ``section_names``)

    Returns:
        Tuple of (fused_matrix [n_source, n_target], matrices_dict)
    """
    n_source = embeddings.shape[0]

    # Square (legacy cohort) mode = no target passed. Kept as a real flag, not
    # just target=source, because the same-section term must call the exact
    # legacy cosine_matrix: BLAS uses a symmetric kernel for A @ A.T that
    # differs from the rectangular A @ B.T path by ~1 ULP, and the legacy
    # behavior is preserved bit-exactly.
    square_mode = target_embeddings is None
    if target_embeddings is None:
        target_embeddings = embeddings
        target_section_names = section_names
    elif target_section_names is None:
        target_section_names = section_names
    n_target = target_embeddings.shape[0]

    # Presence of each section per user, per side. A section a profile didn't
    # fill is stored as a zero vector (see embed.get_embeddings); treat that as
    # "absent" so it can be masked out of the fusion rather than counted as
    # similarity 0.
    shared_sections = [s for s in section_names if s in target_section_names]
    src_present = {}   # name -> bool array (n_source,)
    tgt_present = {}   # name -> bool array (n_target,)
    section_matrices = {}
    for section_name in shared_sections:
        src_sec = embeddings[:, section_names.index(section_name), :]
        tgt_sec = target_embeddings[:, target_section_names.index(section_name), :]
        src_present[section_name] = np.linalg.norm(src_sec, axis=1) > 1e-8
        tgt_present[section_name] = np.linalg.norm(tgt_sec, axis=1) > 1e-8
        if square_mode:
            section_matrices[section_name] = cosine_matrix(src_sec)
        else:
            section_matrices[section_name] = cosine_rect(src_sec, tgt_sec)

    cross_weights = cross_section_weights or {}
    hyde = hyde_embeddings or {}

    valid_section_weights = {k: v for k, v in section_weights.items() if k in section_matrices}
    for missing in set(section_weights) - set(valid_section_weights):
        print(f"Warning: Section '{missing}' not found in embeddings")

    # Per-pair fusion with masking: a section contributes to a pair only when the
    # relevant side(s) are present, and the denominator is the weight mass that
    # was *actually present* for that pair. This makes a missing section neutral
    # (ignored) instead of an extreme low that drags a sparse profile's scores.
    # When every section is present for a pair, this reduces exactly to the old
    # global-normalization behavior (denominator == sum of all abs weights).
    weighted_sum = np.zeros((n_source, n_target))
    weight_mass = np.zeros((n_source, n_target))

    for section_name, weight in valid_section_weights.items():
        mask = np.outer(src_present[section_name], tgt_present[section_name]).astype(float)
        weighted_sum += weight * mask * section_matrices[section_name]
        weight_mass += abs(weight) * mask

    # Cross-section similarity — DIRECTIONAL (not symmetrized)
    cross_section_matrices = {}

    for cross_key, weight in cross_weights.items():
        src_section, tgt_section = parse_cross_key(cross_key)

        if cross_key not in hyde:
            print(f"Warning: No HyDE embeddings for '{cross_key}', skipping")
            continue

        if tgt_section not in target_section_names:
            print(f"Warning: Target section '{tgt_section}' not found in embeddings, skipping")
            continue

        # Source side: HyDE embeddings (vocabulary-bridged toward target)
        # Shape: (n_source, n_descriptors, dim)
        src_emb = hyde[cross_key]

        # Target side: regular section embeddings
        # Shape: (n_target, 1, dim)
        tgt_idx = target_section_names.index(tgt_section)
        tgt_emb = target_embeddings[:, tgt_idx:tgt_idx+1, :]

        n_src_desc = src_emb.shape[1]
        n_tgt_desc = tgt_emb.shape[1]  # always 1 for regular sections

        # Normalize
        src_norm = src_emb / (np.linalg.norm(src_emb, axis=2, keepdims=True) + 1e-8)
        tgt_norm = tgt_emb / (np.linalg.norm(tgt_emb, axis=2, keepdims=True) + 1e-8)

        # Max-pooled cross-similarity (ASYMMETRIC)
        # cross_matrix[i][j] = max over descriptor pairs (s, t) of cos_sim(src_i_s, tgt_j_t)
        # With n_descriptors=1, this is just a single matmul.
        cross_matrix = np.full((n_source, n_target), -np.inf)
        for src_d in range(n_src_desc):
            for tgt_d in range(n_tgt_desc):
                pair_sim = src_norm[:, src_d, :] @ tgt_norm[:, tgt_d, :].T
                cross_matrix = np.maximum(cross_matrix, pair_sim)

        # *** DO NOT SYMMETRIZE ***
        # cross_matrix[i][j] = "j can help i" (j's skills match i's HyDE-bridged needs)
        # cross_matrix[j][i] = "i can help j" (i's skills match j's HyDE-bridged needs)

        cross_section_matrices[cross_key] = cross_matrix

        # Directional presence: source i (HyDE of i's needs) and target j (j's section).
        src_hyde_present = np.linalg.norm(src_emb, axis=2).max(axis=1) > 1e-8
        tgt_sec_present = np.linalg.norm(tgt_emb[:, 0, :], axis=1) > 1e-8
        mask = np.outer(src_hyde_present, tgt_sec_present).astype(float)
        weighted_sum += weight * mask * cross_matrix
        weight_mass += abs(weight) * mask

    # Per-pair normalization. Pairs with no overlapping present signal get 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        fused_matrix = np.where(weight_mass > 0, weighted_sum / weight_mass, 0.0)

    matrices_dict = {
        'section_matrices': section_matrices,
        'cross_section_matrices': cross_section_matrices,
        'section_weights': valid_section_weights,
        'cross_section_weights': cross_weights,
        'combined_matrix': fused_matrix,
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
    """
    n_users = len(user_ids)

    similarity_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_no_diag, -1.0)

    top_k_indices_all = np.argsort(similarity_no_diag, axis=1)[:, -k:]

    candidate_pairs = []
    for i in range(n_users):
        user_i = user_ids[i]
        top_k_i = set(top_k_indices_all[i])

        for j in range(i + 1, n_users):
            user_j = user_ids[j]
            top_k_j = set(top_k_indices_all[j])

            if j in top_k_i or i in top_k_j:
                score = similarity_matrix[i, j]
                if score > 0:
                    pair = CandidatePair.create(user_i, user_j, score)
                    candidate_pairs.append(pair)

    candidate_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
    print(f"Generated {len(candidate_pairs)} unique candidate pairs (top-{k} per user)")
    return candidate_pairs


def apply_recipe(
    embeddings: np.ndarray,
    section_names: List[str],
    recipe_config: Dict,
    hyde_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply a specific recipe to compute similarity matrix.

    Args:
        embeddings: User embeddings
        section_names: Section names
        recipe_config: Recipe configuration from config
        hyde_embeddings: HyDE embeddings dict

    Returns:
        Tuple of (similarity_matrix, matrices_dict)
    """
    section_weights = recipe_config.get('section_weights', {})
    cross_section_weights = recipe_config.get('cross_section_weights', {})

    print("Applying recipe:")
    print(f"  Section weights: {section_weights}")
    if cross_section_weights:
        print(f"  Cross-section weights: {cross_section_weights}")

    similarity_matrix, matrices_dict = compute_fused_similarity_matrix(
        embeddings=embeddings,
        section_names=section_names,
        section_weights=section_weights,
        cross_section_weights=cross_section_weights,
        hyde_embeddings=hyde_embeddings,
    )

    return similarity_matrix, matrices_dict


def generate_similarity_matrix(
    embeddings: np.ndarray,
    user_ids: List[str],
    section_names: List[str],
    recipe_config: Dict,
    hyde_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Generate similarity matrix for all user pairs.

    When cross-section weights are present, returns both directional (asymmetric)
    and symmetric matrices.

    Args:
        embeddings: User embeddings
        user_ids: User IDs
        section_names: Section names
        recipe_config: Recipe configuration
        hyde_embeddings: HyDE embeddings dict

    Returns:
        Tuple of (dir_similarity_matrix, sym_similarity_matrix, user_ids, matrices_dict)
        - dir_similarity_matrix: raw (possibly asymmetric) fused matrix
        - sym_similarity_matrix: (dir + dir.T) / 2, for b-matching and candidate selection
    """
    dir_matrix, matrices_dict = apply_recipe(
        embeddings=embeddings,
        section_names=section_names,
        recipe_config=recipe_config,
        hyde_embeddings=hyde_embeddings,
    )

    # Symmetrize for downstream use (b-matching, candidate selection)
    sym_matrix = (dir_matrix + dir_matrix.T) / 2

    has_cross = bool(recipe_config.get('cross_section_weights', {}))
    if has_cross:
        asymmetry = np.abs(dir_matrix - dir_matrix.T)
        mean_asym = np.mean(asymmetry[np.triu_indices(len(user_ids), k=1)])
        print(f"Directional matrix asymmetry (mean abs diff): {mean_asym:.4f}")

    print(f"Generated similarity matrices of shape {dir_matrix.shape}")

    return dir_matrix, sym_matrix, user_ids, matrices_dict


def generate_rectangular_similarity(
    source: EmbeddingsBundle,
    target: EmbeddingsBundle,
    recipe_config: Dict,
) -> SimilarityResult:
    """Source × target fused similarity over two embeddings bundles.

    The rectangular core primitive for query (1×M) and subset batch (M×N)
    modes. The directional matrix is NEVER symmetrized here; ``sym_matrix`` is
    only filled when source and target are the identical user list (the legacy
    square path goes through ``generate_similarity_matrix`` instead).
    """
    dir_matrix, matrices_dict = compute_fused_similarity_matrix(
        embeddings=source.embeddings,
        section_names=source.section_names,
        section_weights=recipe_config.get('section_weights', {}),
        cross_section_weights=recipe_config.get('cross_section_weights', {}),
        hyde_embeddings=source.hyde or None,
        target_embeddings=target.embeddings,
        target_section_names=target.section_names,
    )

    sym_matrix = None
    if source.user_ids == target.user_ids:
        sym_matrix = (dir_matrix + dir_matrix.T) / 2

    return SimilarityResult(
        source_ids=list(source.user_ids),
        target_ids=list(target.user_ids),
        dir_matrix=dir_matrix,
        sym_matrix=sym_matrix,
        matrices_dict=matrices_dict,
    )
