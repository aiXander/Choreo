"""Ingest user profile text files."""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from utils import hash_text, ensure_dir


@dataclass
class Profile:
    """User profile data."""
    id: str
    text: str
    hash: str
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Profile':
        """Create profile from text file."""
        text = file_path.read_text(encoding='utf-8').strip()
        user_id = file_path.stem  # filename without extension
        text_hash = hash_text(text)
        
        return cls(id=user_id, text=text, hash=text_hash)


def load_profiles(raw_dir: str) -> List[Profile]:
    """
    Load all .txt files from raw_dir as Profile objects.
    
    Args:
        raw_dir: Directory containing .txt files (one per user)
    
    Returns:
        List of Profile objects
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    profiles = []
    txt_files = list(raw_path.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(f"No .txt files found in {raw_dir}")
    
    for file_path in sorted(txt_files):
        try:
            profile = Profile.from_file(file_path)
            profiles.append(profile)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    if not profiles:
        raise ValueError(f"No valid profiles loaded from {raw_dir}")
    
    print(f"Loaded {len(profiles)} profiles from {raw_dir}")
    return profiles


def profiles_to_dict(profiles: List[Profile]) -> Dict[str, Dict[str, Any]]:
    """Convert profiles to dictionary format for serialization."""
    return {
        profile.id: {
            'text': profile.text,
            'hash': profile.hash
        }
        for profile in profiles
    }