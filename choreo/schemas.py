"""Typed IO schemas for every pipeline stage.

Every stage of the matching pipeline consumes and produces one of the
dataclasses below (see `stages.py` for the runtime-introspectable registry).
Each type supports ``to_dict``/``from_dict`` so a caller can move stage data
across process boundaries as plain JSON, and the heavyweight bundle types also
ship ``dump``/``load`` disk helpers that define the canonical on-disk format
used by the FileStore adapter.

Design rules (docs/01_todo.md §3.1):
- Transforms never do IO; ``dump``/``load`` are adapter helpers.
- Embeddings carry provenance (model + native dim) and per-cell content hashes
  so reuse is content-addressed, not roster-addressed.
- Arrays keep an explicit ``user_ids``/``section_names`` order so subsets
  pulled from an external store line up with array axes deterministically.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .utils import hash_text, load_json, save_json, ensure_dir


# ---------------------------------------------------------------------------
# extract stage
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSections:
    """Extracted profile sections (one per user).

    ``last_updated_at`` carries the source profile's timestamp (ISO-8601, UTC)
    — i.e. "this extraction reflects the profile as of T". Optional: content
    hashes drive internal reuse; the timestamp is the adapter-level freshness
    signal (see utils.is_stale) and round-trips through stores untouched.
    """
    id: str
    sections: Dict[str, str]
    hash: str
    last_updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sections": dict(self.sections),
            "hash": self.hash,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedSections":
        return cls(
            id=data["id"],
            sections=dict(data["sections"]),
            hash=data["hash"],
            last_updated_at=data.get("last_updated_at"),
        )


def sections_content_hash(sections: Dict[str, str]) -> str:
    """Stable content hash over a sections mapping (key-order independent)."""
    canonical = json.dumps(
        {k: sections[k] for k in sorted(sections)}, ensure_ascii=False
    )
    return hash_text(canonical)


def sections_from_dict(
    profiles_sections: Dict[str, Dict[str, str]],
    last_updated_at: Optional[Union[str, Dict[str, str]]] = None,
) -> List[ExtractedSections]:
    """Entry-at-any-stage helper: ingest pre-sectioned input.

    Builds ``ExtractedSections`` directly from ``{user_id: {section: text}}``,
    bypassing ``load_profiles``/LLM extraction entirely. The content hash is
    derived from the section texts so downstream hash-based reuse (HyDE cache,
    embedding deltas) works exactly as for extracted profiles.

    ``last_updated_at`` optionally attaches source timestamps (ISO-8601):
    either one value for all users or a ``{user_id: timestamp}`` mapping —
    this is how an external store's ``updated_at`` column enters the pipeline.
    """
    def _ts(user_id: str) -> Optional[str]:
        if isinstance(last_updated_at, dict):
            return last_updated_at.get(user_id)
        return last_updated_at

    return [
        ExtractedSections(
            id=user_id,
            sections=dict(sections),
            hash=sections_content_hash(sections),
            last_updated_at=_ts(user_id),
        )
        for user_id, sections in profiles_sections.items()
    ]


# ---------------------------------------------------------------------------
# hyde stage
# ---------------------------------------------------------------------------

@dataclass
class HydeDescriptors:
    """HyDE descriptors for a user's section, bridging to target vocabulary."""
    user_id: str
    source_section: str     # e.g., "needs"
    target_section: str     # e.g., "skills"
    descriptors: List[str]  # Always a list, even when n_descriptors=1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "source_section": self.source_section,
            "target_section": self.target_section,
            "descriptors": list(self.descriptors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HydeDescriptors":
        return cls(
            user_id=data["user_id"],
            source_section=data["source_section"],
            target_section=data["target_section"],
            descriptors=list(data["descriptors"]),
        )


def hyde_content_hash(descriptors: List[str]) -> str:
    """Stable content hash over a user's HyDE descriptor list."""
    return hash_text("\x1f".join(descriptors))


# ---------------------------------------------------------------------------
# embed stage
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingsBundle:
    """Dense embeddings for a set of users, with provenance and content hashes.

    The single currency for everything downstream of the embed stage. Axes are
    pinned by ``user_ids`` (rows) and ``section_names`` (columns); HyDE arrays
    share the user axis. ``section_hashes``/``hyde_hashes`` record the content
    hash of the text behind each vector so the embed stage can recompute only
    deltas, regardless of who else is in the set (content-addressed reuse).

    Full-size native vectors are always stored; MRL truncation is applied by
    callers at computation time (see embed.truncate_embeddings).
    """
    user_ids: List[str]
    section_names: List[str]
    embeddings: np.ndarray                      # (n_users, n_sections, dim)
    hyde: Dict[str, np.ndarray] = field(default_factory=dict)   # cross_key -> (n_users, n_desc, dim)
    embedding_model: Optional[str] = None
    dim: Optional[int] = None                   # native embedding dim
    # user_id -> section_name -> hash_text(section text)
    section_hashes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # cross_key -> user_id -> hyde_content_hash(descriptors)
    hyde_hashes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # user_id -> last_updated_at of the source data this user's vectors were
    # computed from (ISO-8601). Adapter-level freshness signal (utils.is_stale);
    # content hashes above remain the internal invalidation mechanism.
    user_timestamps: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dim is None and getattr(self.embeddings, "ndim", 0) == 3:
            self.dim = int(self.embeddings.shape[-1])

    # -- subset assembly ----------------------------------------------------

    def index_of(self, user_id: str) -> int:
        try:
            return self.user_ids.index(user_id)
        except ValueError as exc:
            raise KeyError(f"User '{user_id}' not in embeddings bundle") from exc

    def subset(self, user_ids: List[str]) -> "EmbeddingsBundle":
        """Return a bundle for an arbitrary subset of users, in the given order.

        This is the "get_embeddings(ids)" primitive: query (1×M) and subset
        (M×N) modes pull slices of a community bundle without touching disk.
        Raises KeyError if any requested id is missing.
        """
        index = {u: i for i, u in enumerate(self.user_ids)}
        missing = [u for u in user_ids if u not in index]
        if missing:
            raise KeyError(f"Users not in embeddings bundle: {missing}")
        idx = [index[u] for u in user_ids]
        return EmbeddingsBundle(
            user_ids=list(user_ids),
            section_names=list(self.section_names),
            embeddings=self.embeddings[idx],
            hyde={k: v[idx] for k, v in self.hyde.items()},
            embedding_model=self.embedding_model,
            dim=self.dim,
            section_hashes={u: dict(self.section_hashes.get(u, {})) for u in user_ids},
            hyde_hashes={
                k: {u: h[u] for u in user_ids if u in h}
                for k, h in self.hyde_hashes.items()
            },
            user_timestamps={
                u: self.user_timestamps[u] for u in user_ids if u in self.user_timestamps
            },
        )

    # -- JSON round-trip ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_ids": list(self.user_ids),
            "section_names": list(self.section_names),
            "embeddings": self.embeddings.tolist(),
            "hyde": {k: v.tolist() for k, v in self.hyde.items()},
            "embedding_model": self.embedding_model,
            "dim": self.dim,
            "section_hashes": {u: dict(s) for u, s in self.section_hashes.items()},
            "hyde_hashes": {k: dict(h) for k, h in self.hyde_hashes.items()},
            "user_timestamps": dict(self.user_timestamps),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingsBundle":
        return cls(
            user_ids=list(data["user_ids"]),
            section_names=list(data["section_names"]),
            embeddings=np.asarray(data["embeddings"], dtype=float),
            hyde={k: np.asarray(v, dtype=float) for k, v in data.get("hyde", {}).items()},
            embedding_model=data.get("embedding_model"),
            dim=data.get("dim"),
            section_hashes={u: dict(s) for u, s in data.get("section_hashes", {}).items()},
            hyde_hashes={k: dict(h) for k, h in data.get("hyde_hashes", {}).items()},
            user_timestamps=dict(data.get("user_timestamps", {})),
        )

    # -- disk format (canonical; used by FileStore) --------------------------
    # Backwards-compatible with the pre-refactor layout: vectors.npz, ids.json,
    # section_names.json, hyde_vectors.npz. Provenance + hashes live in the new
    # bundle_meta.json side-car (absent on legacy dirs; tolerated on load).

    def dump(self, embeds_dir: Union[str, Path]) -> None:
        path = ensure_dir(embeds_dir)
        np.savez_compressed(path / "vectors.npz", vectors=self.embeddings)
        save_json(list(self.user_ids), path / "ids.json")
        save_json(list(self.section_names), path / "section_names.json")
        hyde_file = path / "hyde_vectors.npz"
        if self.hyde:
            np.savez_compressed(hyde_file, **self.hyde)
        elif hyde_file.exists():
            # Don't leave a stale file behind (e.g. after switching to a
            # cross-weight-free recipe): the next load() would resurrect HyDE
            # arrays that no longer match the roster.
            hyde_file.unlink()
        save_json(
            {
                "embedding_model": self.embedding_model,
                "dim": self.dim,
                "section_hashes": self.section_hashes,
                "hyde_hashes": self.hyde_hashes,
                "user_timestamps": self.user_timestamps,
            },
            path / "bundle_meta.json",
        )

    @classmethod
    def load(cls, embeds_dir: Union[str, Path]) -> "EmbeddingsBundle":
        path = Path(embeds_dir)
        vectors_file = path / "vectors.npz"
        ids_file = path / "ids.json"
        sections_file = path / "section_names.json"
        if not (vectors_file.exists() and ids_file.exists() and sections_file.exists()):
            raise FileNotFoundError(f"No embeddings bundle found in {embeds_dir}")

        embeddings = np.load(vectors_file)["vectors"]
        user_ids = load_json(ids_file)
        section_names = load_json(sections_file)

        hyde: Dict[str, np.ndarray] = {}
        hyde_file = path / "hyde_vectors.npz"
        if hyde_file.exists():
            with np.load(hyde_file) as hyde_data:
                hyde = {k: hyde_data[k] for k in hyde_data.files}

        meta: Dict[str, Any] = {}
        meta_file = path / "bundle_meta.json"
        if meta_file.exists():
            try:
                meta = load_json(meta_file)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: could not read {meta_file}: {exc}")

        return cls(
            user_ids=user_ids,
            section_names=section_names,
            embeddings=embeddings,
            hyde=hyde,
            embedding_model=meta.get("embedding_model"),
            dim=meta.get("dim"),
            section_hashes=meta.get("section_hashes", {}),
            hyde_hashes=meta.get("hyde_hashes", {}),
            user_timestamps=meta.get("user_timestamps", {}),
        )


# ---------------------------------------------------------------------------
# similarity stage
# ---------------------------------------------------------------------------

def _matrices_dict_to_jsonable(matrices_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in matrices_dict.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, dict):
            out[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


def _matrices_dict_from_jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if key in ("section_matrices", "cross_section_matrices"):
            out[key] = {k: np.asarray(v, dtype=float) for k, v in value.items()}
        elif key == "combined_matrix":
            out[key] = np.asarray(value, dtype=float)
        else:
            out[key] = value
    return out


@dataclass
class SimilarityResult:
    """Fused similarity between a source user set and a target user set.

    ``dir_matrix[i][j]`` = "how well can target j help source i" — DIRECTIONAL,
    never symmetrized for rectangular use. ``sym_matrix`` ((dir+dir.T)/2) is only
    populated on the legacy square cohort path where source == target.
    """
    source_ids: List[str]
    target_ids: List[str]
    dir_matrix: np.ndarray                      # (n_source, n_target)
    sym_matrix: Optional[np.ndarray] = None     # square cohort runs only
    matrices_dict: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_square(self) -> bool:
        return self.source_ids == self.target_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ids": list(self.source_ids),
            "target_ids": list(self.target_ids),
            "dir_matrix": self.dir_matrix.tolist(),
            "sym_matrix": self.sym_matrix.tolist() if self.sym_matrix is not None else None,
            "matrices_dict": _matrices_dict_to_jsonable(self.matrices_dict),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimilarityResult":
        sym = data.get("sym_matrix")
        return cls(
            source_ids=list(data["source_ids"]),
            target_ids=list(data["target_ids"]),
            dir_matrix=np.asarray(data["dir_matrix"], dtype=float),
            sym_matrix=np.asarray(sym, dtype=float) if sym is not None else None,
            matrices_dict=_matrices_dict_from_jsonable(data.get("matrices_dict", {})),
        )


# ---------------------------------------------------------------------------
# score stage
# ---------------------------------------------------------------------------

@dataclass
class PairScore:
    """LLM score for a user pair."""
    pair_id: str
    user1: str
    user2: str
    embed_score: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "user1": self.user1,
            "user2": self.user2,
            "embed_score": self.embed_score,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairScore":
        return cls(
            pair_id=data["pair_id"],
            user1=data["user1"],
            user2=data["user2"],
            embed_score=data["embed_score"],
            score=data["score"],
        )


# ---------------------------------------------------------------------------
# match stage
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """Final matched edge between two users."""
    user1: str
    user2: str
    pair_id: str
    final_weight: float
    embed_score: float
    llm_score: float
    embed_score_normalized: Optional[float] = None
    llm_score_normalized: Optional[float] = None
    intro: str = ""
    starter_topics: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'user1': self.user1,
            'user2': self.user2,
            'pair_id': self.pair_id,
            'final_weight': round(self.final_weight, 3),
            'embed_score': round(self.embed_score, 3),
            'llm_score': round(self.llm_score, 3),
            'embed_score_normalized': round(self.embed_score_normalized, 3) if self.embed_score_normalized is not None else None,
            'llm_score_normalized': round(self.llm_score_normalized, 3) if self.llm_score_normalized is not None else None,
            'intro': self.intro,
            'starter_topics': self.starter_topics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        return cls(
            user1=data["user1"],
            user2=data["user2"],
            pair_id=data["pair_id"],
            final_weight=data["final_weight"],
            embed_score=data["embed_score"],
            llm_score=data["llm_score"],
            embed_score_normalized=data.get("embed_score_normalized"),
            llm_score_normalized=data.get("llm_score_normalized"),
            intro=data.get("intro", ""),
            starter_topics=data.get("starter_topics", ""),
        )


# ---------------------------------------------------------------------------
# introduce stage
# ---------------------------------------------------------------------------

@dataclass
class Introduction:
    """Generated introduction for a matched pair."""
    pair_id: str
    user1: str
    user2: str
    intro: str
    starter_topics: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "user1": self.user1,
            "user2": self.user2,
            "intro": self.intro,
            "starter_topics": self.starter_topics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Introduction":
        return cls(
            pair_id=data["pair_id"],
            user1=data["user1"],
            user2=data["user2"],
            intro=data["intro"],
            starter_topics=data["starter_topics"],
        )
