"""Utility functions for prompt-mesh matching system."""

import json
import hashlib
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
from numba import jit


@jit(nopython=True)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    # Ensure contiguous arrays for better performance
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


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