"""Generate embeddings for profile sections via OpenRouter."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from utils import ensure_dir, save_json, load_json
from extract import ExtractedSections
from hyde import HydeDescriptors
from cost_tracker import get_cost_tracker
from llm import get_openrouter_client, extract_usage, DEFAULT_EMBEDDING_MODEL


# Embedding models known to be Matryoshka (MRL) trained — i.e. their leading
# dimensions can be safely kept + renormalized to produce a shorter, still-valid
# embedding. Truncating any other model would silently corrupt similarity, so
# MRL truncation (embedding_dimensions in config) is skipped unless the active
# model is listed here. Add slugs as you verify support.
MRL_CAPABLE_MODELS = {
    "google/gemini-embedding-2-preview",
}


def supports_mrl(model: str) -> bool:
    """Whether `model` is known to support Matryoshka (MRL) truncation."""
    return model in MRL_CAPABLE_MODELS


def get_embeddings(texts: List[str], model: str) -> np.ndarray:
    """
    Get embeddings for a list of texts via OpenRouter's embeddings endpoint.

    Always fetches the model's full native dimensionality. Matryoshka (MRL)
    truncation to a smaller size is applied later, at computation time, via
    truncate_embeddings() — so the on-disk vectors stay full and the truncation
    size can be re-tuned without re-embedding.

    Args:
        texts: List of text strings to embed
        model: Embedding model name (e.g. "google/gemini-embedding-2-preview")

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = model or DEFAULT_EMBEDDING_MODEL
    try:
        client = get_openrouter_client()
        # The OpenAI SDK defaults to encoding_format="base64", which OpenRouter's
        # Google AI Studio embedding provider rejects (400 → empty data → the SDK
        # raises "No embedding data received"). Force "float" to stay compatible.
        response = client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )

        # Track cost using OpenRouter's native usage accounting (real cost in USD
        # credits, returned automatically). Falls back to 0 if not reported.
        cost_tracker = get_cost_tracker()
        try:
            usage = extract_usage(response)
            input_tokens = usage["prompt_tokens"] or len(texts)

            cost_tracker.record_call(
                component="embeddings",
                call_type="embedding",
                model=model,
                input_tokens=input_tokens,
                output_tokens=0,  # Embeddings don't have output tokens
                cost=usage["cost"] or 0.0,
            )
        except (AttributeError, KeyError, TypeError):
            # If cost tracking fails, continue without it
            print(f"Warning: Could not track cost for embedding call with model {model}")

        embeddings = np.array([item.embedding for item in response.data])
        print(f"Created embeddings with {model} of shape {embeddings.shape}")

        return embeddings

    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise


def truncate_embeddings(arr: np.ndarray, dimensions: int) -> np.ndarray:
    """
    Matryoshka (MRL) truncation of stored full-size embeddings, applied at
    computation time.

    gemini-embedding-2 is MRL-trained: the most important information is packed
    into the leading dimensions, so keeping the first `dimensions` components and
    L2-renormalizing reproduces what the API returns for an equivalent
    output_dimensionality request (verified to ~1e-7). This lets us store full
    3072-dim vectors once and re-tune the working size for free.

    Truncates along the last axis, so it works for both the section embeddings
    (users × sections × dims) and HyDE embeddings (users × descriptors × dims).

    Args:
        arr: Embedding array with the embedding dimension as the last axis
        dimensions: Target size. None or >= current size returns arr unchanged.

    Returns:
        Truncated, unit-normalized array (or arr unchanged).
    """
    if not dimensions or dimensions >= arr.shape[-1]:
        return arr
    sliced = arr[..., :dimensions]
    norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
    return sliced / np.clip(norms, 1e-12, None)


def create_section_embeddings(
    extracted_sections: List[ExtractedSections],
    embedding_model: str,
    embeds_dir: str,
    hyde_descriptors: Dict[str, List[HydeDescriptors]] = None,
    force: bool = False
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Create embeddings for all sections of all users, plus HyDE descriptors.

    Args:
        extracted_sections: List of ExtractedSections
        embedding_model: Name of embedding model
        embeds_dir: Directory to save embeddings
        hyde_descriptors: Dict mapping cross_key to list of HydeDescriptors per user
        force: Force regeneration

    Returns:
        Tuple of (user_ids, section_names, embeddings_array, hyde_embeddings)
        - embeddings_array shape: (n_users, n_sections, embedding_dim)
        - hyde_embeddings: dict of cross_key -> (n_users, n_descriptors, embedding_dim)
    """
    embeds_path = ensure_dir(embeds_dir)

    if not extracted_sections:
        raise ValueError("No extracted sections provided")

    # Get section names from first profile (assuming all have same structure)
    section_names = list(extracted_sections[0].sections.keys())
    user_ids = [profile.id for profile in extracted_sections]

    print(f"Creating embeddings for {len(user_ids)} users, {len(section_names)} sections each")

    # Check if embeddings already exist (unless force flag is set)
    vectors_file = embeds_path / "vectors.npz"
    ids_file = embeds_path / "ids.json"
    sections_file = embeds_path / "section_names.json"
    hyde_vectors_file = embeds_path / "hyde_vectors.npz"

    has_hyde = hyde_descriptors and len(hyde_descriptors) > 0

    if not force and vectors_file.exists() and ids_file.exists() and sections_file.exists():
        try:
            existing_ids = load_json(ids_file)
            existing_sections = load_json(sections_file)

            # Check if we have the same users and sections
            if (set(existing_ids) == set(user_ids) and
                existing_sections == section_names):

                # Try to load HyDE embeddings too
                hyde_embeddings = {}
                if has_hyde and hyde_vectors_file.exists():
                    hyde_data = np.load(hyde_vectors_file)
                    for cross_key in hyde_descriptors:
                        if cross_key in hyde_data:
                            hyde_embeddings[cross_key] = hyde_data[cross_key]

                    if set(hyde_embeddings.keys()) == set(hyde_descriptors.keys()):
                        print("Loading existing embeddings (including HyDE)...")
                        vectors = np.load(vectors_file)['vectors']
                        return existing_ids, existing_sections, vectors, hyde_embeddings
                elif not has_hyde:
                    print("Loading existing embeddings...")
                    vectors = np.load(vectors_file)['vectors']
                    return existing_ids, existing_sections, vectors, {}
        except Exception as e:
            print(f"Warning: Could not load existing embeddings: {e}")
    elif force:
        print("Force flag set - regenerating all embeddings")

    # Collect all texts to embed (flatten across users and sections)
    all_texts = []
    text_indices = []  # (user_idx, section_idx) for each text

    for user_idx, profile in enumerate(extracted_sections):
        for section_idx, section_name in enumerate(section_names):
            text = profile.sections.get(section_name, "")
            all_texts.append(text)
            text_indices.append((user_idx, section_idx))

    print(f"Getting embeddings for {len(all_texts)} text segments...")

    # Get embeddings in batches to avoid API limits
    batch_size = 64
    all_embeddings = []

    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_embeddings = get_embeddings(batch_texts, embedding_model)
        all_embeddings.append(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}")

    # Concatenate all embeddings
    flat_embeddings = np.vstack(all_embeddings)
    embedding_dim = flat_embeddings.shape[1]

    # Reshape into (n_users, n_sections, embedding_dim)
    n_users = len(user_ids)
    n_sections = len(section_names)
    embeddings_array = np.zeros((n_users, n_sections, embedding_dim))

    for i, (user_idx, section_idx) in enumerate(text_indices):
        embeddings_array[user_idx, section_idx] = flat_embeddings[i]

    # Save main embeddings and metadata
    np.savez_compressed(vectors_file, vectors=embeddings_array)
    save_json(user_ids, ids_file)
    save_json(section_names, sections_file)

    print(f"Saved embeddings to {embeds_path}")
    print(f"Shape: {embeddings_array.shape} (users, sections, dims)")

    # Embed HyDE descriptors
    hyde_embeddings = {}
    if has_hyde:
        print(f"Embedding HyDE descriptors for {len(hyde_descriptors)} cross-section pairs...")
        hyde_save_dict = {}

        for cross_key, user_descs in hyde_descriptors.items():
            n_desc = len(user_descs[0].descriptors)
            print(f"  {cross_key}: {n_users} users x {n_desc} descriptors")

            section_embeds = np.zeros((n_users, n_desc, embedding_dim))

            for d in range(n_desc):
                variant_texts = [ud.descriptors[d] for ud in user_descs]
                desc_embeddings = get_embeddings(variant_texts, embedding_model)
                section_embeds[:, d, :] = desc_embeddings

            hyde_embeddings[cross_key] = section_embeds
            hyde_save_dict[cross_key] = section_embeds

        np.savez_compressed(hyde_vectors_file, **hyde_save_dict)
        print(f"Saved HyDE embeddings to {hyde_vectors_file}")

    return user_ids, section_names, embeddings_array, hyde_embeddings


def load_embeddings(embeds_dir: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load embeddings from disk.
    
    Returns:
        Tuple of (user_ids, section_names, embeddings_array)
    """
    embeds_path = Path(embeds_dir)
    
    vectors_file = embeds_path / "vectors.npz"
    ids_file = embeds_path / "ids.json"
    sections_file = embeds_path / "section_names.json"
    
    if not all(f.exists() for f in [vectors_file, ids_file, sections_file]):
        raise FileNotFoundError("Embedding files not found. Run embedding generation first.")
    
    user_ids = load_json(ids_file)
    section_names = load_json(sections_file)
    embeddings_array = np.load(vectors_file)['vectors']
    
    return user_ids, section_names, embeddings_array