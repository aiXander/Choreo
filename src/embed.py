"""Generate embeddings for profile sections."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import litellm
from litellm import embedding

from utils import ensure_dir, save_json, load_json
from extract import ExtractedSections
from cost_tracker import get_cost_tracker


def get_embeddings(texts: List[str], model: str) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model name
        
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    try:
        response = embedding(model=model, input=texts)
        
        # Track cost using LiteLLM's response_cost
        cost_tracker = get_cost_tracker()
        try:
            cost = response._hidden_params["response_cost"]
            input_tokens = getattr(response.usage, 'prompt_tokens', len(texts)) if hasattr(response, 'usage') else len(texts)
            
            cost_tracker.record_call(
                component="embeddings",
                call_type="embedding",
                model=model,
                input_tokens=input_tokens,
                output_tokens=0,  # Embeddings don't have output tokens
                cost=cost
            )
        except (AttributeError, KeyError):
            # If cost tracking fails, continue without it
            print(f"Warning: Could not track cost for embedding call with model {model}")
        
        embeddings = []
        for item in response.data:
            embeddings.append(item["embedding"])

        embeddings = np.array(embeddings)
        print(f"Created embeddings with {model} of shape {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        raise


def create_section_embeddings(
    extracted_sections: List[ExtractedSections],
    embedding_model: str,
    embeds_dir: str,
    force: bool = False
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Create embeddings for all sections of all users.
    
    Args:
        extracted_sections: List of ExtractedSections
        embedding_model: Name of embedding model
        embeds_dir: Directory to save embeddings
        
    Returns:
        Tuple of (user_ids, section_names, embeddings_array)
        embeddings_array shape: (n_users, n_sections, embedding_dim)
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
    
    if not force and vectors_file.exists() and ids_file.exists() and sections_file.exists():
        try:
            existing_ids = load_json(ids_file)
            existing_sections = load_json(sections_file)
            
            # Check if we have the same users and sections
            if (set(existing_ids) == set(user_ids) and 
                existing_sections == section_names):
                print("Loading existing embeddings...")
                vectors = np.load(vectors_file)['vectors']
                return existing_ids, existing_sections, vectors
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
    
    # Save embeddings and metadata
    np.savez_compressed(vectors_file, vectors=embeddings_array)
    save_json(user_ids, ids_file)
    save_json(section_names, sections_file)
    
    print(f"Saved embeddings to {embeds_path}")
    print(f"Shape: {embeddings_array.shape} (users, sections, dims)")
    
    return user_ids, section_names, embeddings_array


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