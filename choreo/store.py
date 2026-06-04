"""Storage adapter protocol + the filesystem reference implementation.

Choreo is a library of matchmaking compute, not a database. Adapters own all
IO: they fetch existing data (sections, embeddings, match history), hand it to
the pure stage transforms as plain arguments, and persist whatever comes back.

This module ships the one adapter that lives in this repo — ``FileStore``,
which wraps the historical ``data/<group>/{raw,processed,embeds,outputs}``
layout. An external app (e.g. a Neon/Postgres wrapper) implements the same
``Store`` protocol outside this repo; nothing in the core ever requires one.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Set, Union

from .utils import ensure_dir, load_jsonl
from .schemas import Edge, EmbeddingsBundle, ExtractedSections
from .ingest import Profile, load_profiles


class Store(Protocol):
    """The contract an external persistence adapter must satisfy.

    All methods deal in schema objects (see ``schemas.py``); none of the core
    stages call a Store directly — runners/adapters do, and only when one is
    provided.
    """

    def get_sections(self, ids: Optional[List[str]] = None) -> List[ExtractedSections]: ...
    def put_sections(self, sections: List[ExtractedSections]) -> None: ...
    def get_embeddings(self, ids: Optional[List[str]] = None) -> EmbeddingsBundle: ...
    def put_embeddings(self, bundle: EmbeddingsBundle) -> None: ...
    def get_match_history(
        self,
        ids: Optional[Iterable[str]] = None,
        window_months: Optional[float] = None,
    ) -> Set[str]: ...
    def put_matches(self, edges: List[Edge], matched_at: Optional[str] = None) -> None: ...


class FileStore:
    """Filesystem reference adapter (the standalone ``.txt`` workflow).

    Owns the canonical disk formats of the stage outputs:
      - sections        -> ``<processed_dir>/sections.jsonl``
      - HyDE cache      -> ``<processed_dir>/hyde/<cross_key>.jsonl``
      - embeddings      -> ``<embeds_dir>/`` (see EmbeddingsBundle.dump)
      - match history   -> ``<base>/match_history.jsonl`` (base = parent of
        outputs_dir), one append-only row per surfaced pair with a timestamp.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        embeds_dir: Optional[Union[str, Path]] = None,
        outputs_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        base = Path(base_dir) if base_dir else None

        def _resolve(explicit, default_subdir: str) -> Optional[Path]:
            if explicit:
                return Path(explicit)
            if base:
                return base / default_subdir
            return None

        self.raw_dir = _resolve(raw_dir, "raw")
        self.processed_dir = _resolve(processed_dir, "processed")
        self.embeds_dir = _resolve(embeds_dir, "embeds")
        self.outputs_dir = _resolve(outputs_dir, "outputs")
        self.cache_dir = _resolve(cache_dir, "cache")

        if base:
            self.history_path = base / "match_history.jsonl"
        elif self.outputs_dir is not None:
            self.history_path = Path(self.outputs_dir).parent / "match_history.jsonl"
        else:
            self.history_path = None

    @classmethod
    def from_io_config(cls, io_config: Dict[str, str]) -> "FileStore":
        """Build from a config ``io:`` mapping (group, folder, or Modal mode)."""
        return cls(
            raw_dir=io_config.get("raw_dir"),
            processed_dir=io_config.get("processed_dir"),
            embeds_dir=io_config.get("embeds_dir"),
            outputs_dir=io_config.get("outputs_dir"),
            cache_dir=io_config.get("cache_dir"),
        )

    # -- profiles (raw .txt) -------------------------------------------------

    def get_profiles(self) -> List[Profile]:
        if self.raw_dir is None:
            raise ValueError("FileStore has no raw_dir configured")
        return load_profiles(str(self.raw_dir))

    # -- sections -------------------------------------------------------------

    @property
    def sections_file(self) -> Path:
        if self.processed_dir is None:
            raise ValueError("FileStore has no processed_dir configured")
        return Path(self.processed_dir) / "sections.jsonl"

    def get_sections(self, ids: Optional[List[str]] = None) -> List[ExtractedSections]:
        """Load sections. The jsonl is append-only, so the LAST row per user id
        wins (it is the most recent extraction for that user)."""
        if not self.sections_file.exists():
            return []
        by_id: Dict[str, ExtractedSections] = {}
        for item in load_jsonl(self.sections_file):
            by_id[item["id"]] = ExtractedSections.from_dict(item)
        if ids is None:
            return list(by_id.values())
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"No stored sections for users: {missing}")
        return [by_id[i] for i in ids]

    def put_sections(self, sections: List[ExtractedSections]) -> None:
        """Upsert sections by user id (rewrites the jsonl)."""
        by_id: Dict[str, ExtractedSections] = {s.id: s for s in self.get_sections()}
        for s in sections:
            by_id[s.id] = s
        ensure_dir(self.sections_file.parent)
        with open(self.sections_file, "w") as f:
            for s in by_id.values():
                f.write(json.dumps(s.to_dict()) + "\n")

    # -- embeddings -----------------------------------------------------------

    def get_embeddings(self, ids: Optional[List[str]] = None) -> EmbeddingsBundle:
        if self.embeds_dir is None:
            raise ValueError("FileStore has no embeds_dir configured")
        bundle = EmbeddingsBundle.load(self.embeds_dir)
        return bundle.subset(ids) if ids is not None else bundle

    def put_embeddings(self, bundle: EmbeddingsBundle) -> None:
        if self.embeds_dir is None:
            raise ValueError("FileStore has no embeds_dir configured")
        bundle.dump(self.embeds_dir)

    # -- match history (novelty input for batch mode) --------------------------

    def get_match_history(
        self,
        ids: Optional[Iterable[str]] = None,
        window_months: Optional[float] = None,
    ) -> Set[str]:
        """Return pair_ids surfaced in prior runs (the ``excluded_pairs`` input).

        Args:
            ids: If given, only pairs touching at least one of these users.
            window_months: If given, only pairs surfaced within the last N
                months (the configured novelty window); older matches become
                eligible again.
        """
        if self.history_path is None or not self.history_path.exists():
            return set()

        cutoff = None
        if window_months is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=window_months * 30.44)

        id_set = set(ids) if ids is not None else None
        excluded: Set[str] = set()
        for item in load_jsonl(self.history_path):
            if cutoff is not None:
                try:
                    matched_at = datetime.fromisoformat(item["matched_at"])
                    if matched_at.tzinfo is None:
                        matched_at = matched_at.replace(tzinfo=timezone.utc)
                    if matched_at < cutoff:
                        continue
                except (KeyError, ValueError):
                    pass  # unparsable timestamp -> keep the exclusion (safe side)
            if id_set is not None and not (
                item.get("user1") in id_set or item.get("user2") in id_set
            ):
                continue
            excluded.add(item["pair_id"])
        return excluded

    def put_matches(self, edges: List[Edge], matched_at: Optional[str] = None) -> None:
        """Append surfaced pairs to the history log (one row per edge)."""
        if self.history_path is None:
            raise ValueError("FileStore has no outputs/base dir to anchor match history")
        if not edges:
            return
        stamp = matched_at or datetime.now(timezone.utc).isoformat()
        ensure_dir(self.history_path.parent)
        with open(self.history_path, "a") as f:
            for edge in edges:
                f.write(json.dumps({
                    "pair_id": edge.pair_id,
                    "user1": edge.user1,
                    "user2": edge.user2,
                    "matched_at": stamp,
                }) + "\n")
