"""Crash-safe dumping of the raw data underlying each plot.

Every plot in ``<outputs_dir>/plots/`` is rendered from in-memory arrays that
are otherwise thrown away after the run (similarity matrices, the stochastic
t-SNE layouts, the per-pair scores). These helpers persist that raw data to
``<outputs_dir>/plots/raw_data/`` as ``.npz`` archives (plus a small JSON
sidecar with axis/label semantics) so the images can be edited and re-exported
later — drop outliers, rename labels, retune colours — without re-embedding or
re-running t-SNE.

Tracing back to users: every datapoint carries a label.
  * matrices  → rows/cols are indexed by ``user_ids`` (stored in the archive).
  * t-SNE     → each 2D point's row matches ``user_ids`` (or ``section_names``).
  * scores    → each point carries its ``pair_id`` (and best-effort user_a/user_b).

CRITICAL: these functions must NEVER crash the main pipeline. Every one is
wrapped so that on any error it prints a warning and returns ``None``.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _raw_data_dir(output_dir: str) -> Path:
    """Resolve (and create) ``<output_dir>/plots/raw_data/``."""
    path = Path(output_dir) / "plots" / "raw_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_key(name: str) -> str:
    """Turn an arbitrary section/cross-section name into a safe npz array key."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_") or "unnamed"


def _save_meta(path: Path, meta: dict) -> None:
    """Write a JSON sidecar; failures here are swallowed by the caller's guard."""
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def save_tsne_raw_data(
    output_dir: str,
    section_coords: Optional[Dict[str, np.ndarray]] = None,
    combined_coords: Optional[np.ndarray] = None,
    user_ids: Optional[List[str]] = None,
    section_relationship_coords: Optional[np.ndarray] = None,
    section_names: Optional[List[str]] = None,
    metric: str = "cosine",
    perplexity: Optional[int] = None,
    filename: str = "tsne_coords",
) -> Optional[str]:
    """Persist t-SNE 2D coordinates so layouts can be re-plotted/edited later.

    Args:
        section_coords: ``{section_name: (n_users, 2) array}`` per-section layouts.
        combined_coords: ``(n_users, 2)`` layout from the combined distance matrix.
        user_ids: row labels for the per-user coordinate arrays.
        section_relationship_coords: ``(n_sections, 2)`` centroid layout.
        section_names: labels for ``section_relationship_coords``.
        filename: base name (without extension) of the ``.npz`` to write.

    Returns the saved ``.npz`` path, or ``None`` on any failure.
    """
    try:
        raw_dir = _raw_data_dir(output_dir)
        arrays: Dict[str, np.ndarray] = {}
        key_to_name: Dict[str, str] = {}

        if user_ids is not None:
            arrays["user_ids"] = np.array([str(u) for u in user_ids])

        for section_name, coords in (section_coords or {}).items():
            if coords is None:
                continue
            key = f"section_{_safe_key(section_name)}"
            arrays[key] = np.asarray(coords)
            key_to_name[key] = section_name

        if combined_coords is not None:
            arrays["combined"] = np.asarray(combined_coords)
            key_to_name["combined"] = "combined (all sections)"

        if section_relationship_coords is not None:
            arrays["section_relationships"] = np.asarray(section_relationship_coords)
            key_to_name["section_relationships"] = "section centroids"
            arrays["section_names"] = np.array([str(s) for s in (section_names or [])])

        if not arrays:
            return None

        npz_path = raw_dir / f"{filename}.npz"
        np.savez(npz_path, **arrays)
        _save_meta(
            raw_dir / f"{filename}.meta.json",
            {
                "description": "t-SNE 2D coordinates. Per-user arrays are row-aligned "
                "to 'user_ids'; 'section_relationships' is row-aligned to "
                "'section_names'. Columns are [tsne_1, tsne_2].",
                "key_to_name": key_to_name,
                "metric": metric,
                "perplexity": perplexity,
            },
        )
        print(f"💾 Saved t-SNE raw data: {npz_path}")
        return str(npz_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️ Skipped saving t-SNE raw data (non-fatal): {exc}")
        return None


def save_similarity_raw_data(
    output_dir: str,
    matrices_dict: dict,
    user_ids: List[str],
) -> Optional[str]:
    """Persist the similarity matrices behind the similarity heatmaps.

    Saves every per-section matrix, every cross-section (directional) matrix and
    the combined matrix. All are ``(n_users, n_users)`` and indexed on both axes
    by ``user_ids``.

    Returns the saved ``.npz`` path, or ``None`` on any failure.
    """
    try:
        raw_dir = _raw_data_dir(output_dir)
        arrays: Dict[str, np.ndarray] = {
            "user_ids": np.array([str(u) for u in user_ids])
        }
        key_to_name: Dict[str, str] = {}

        for section_name, matrix in (matrices_dict.get("section_matrices") or {}).items():
            if matrix is None:
                continue
            key = f"section_{_safe_key(section_name)}"
            arrays[key] = np.asarray(matrix)
            key_to_name[key] = section_name

        for cross_key, matrix in (matrices_dict.get("cross_section_matrices") or {}).items():
            if matrix is None:
                continue
            key = f"cross_{_safe_key(cross_key)}"
            arrays[key] = np.asarray(matrix)
            key_to_name[key] = cross_key

        combined = matrices_dict.get("combined_matrix")
        if combined is not None:
            arrays["combined"] = np.asarray(combined)
            key_to_name["combined"] = "combined (fused) matrix"

        npz_path = raw_dir / "similarity_matrices.npz"
        np.savez(npz_path, **arrays)
        _save_meta(
            raw_dir / "similarity_matrices.meta.json",
            {
                "description": "Cosine similarity matrices. Each matrix is "
                "(n_users, n_users) with both axes indexed by 'user_ids'. "
                "Section/combined matrices are SYMMETRIC; cross_* matrices are "
                "DIRECTIONAL: entry[i][j] = how well user_ids[j] addresses "
                "user_ids[i]'s needs.",
                "key_to_name": key_to_name,
                "section_weights": matrices_dict.get("section_weights", {}),
                "cross_section_weights": matrices_dict.get("cross_section_weights", {}),
            },
        )
        print(f"💾 Saved similarity matrix raw data: {npz_path}")
        return str(npz_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️ Skipped saving similarity raw data (non-fatal): {exc}")
        return None


def save_score_correlation_raw_data(
    output_dir: str,
    normalized_embed_scores: Dict[str, float],
    normalized_llm_scores: Dict[str, float],
    group_name: Optional[str] = None,
) -> Optional[str]:
    """Persist the per-pair embedding/LLM scores behind the correlation plots.

    Each row is one scored pair, labelled by its stable ``pair_id`` (plus a
    best-effort split into ``user_a``/``user_b``).

    Returns the saved ``.npz`` path, or ``None`` on any failure.
    """
    try:
        if not normalized_embed_scores or not normalized_llm_scores:
            return None

        raw_dir = _raw_data_dir(output_dir)
        common = sorted(set(normalized_embed_scores) & set(normalized_llm_scores))
        if not common:
            return None

        pair_ids = np.array([str(p) for p in common])
        embed = np.array([float(normalized_embed_scores[p]) for p in common], dtype=float)
        llm = np.array([float(normalized_llm_scores[p]) for p in common], dtype=float)
        # pair_id is stable_pair_id ("min_max"); split mirrors the plotting code.
        # pair_ids stays the authoritative label since user IDs may contain "_".
        user_a = np.array([p.split("_", 1)[0] if "_" in p else p for p in common])
        user_b = np.array([p.split("_", 1)[1] if "_" in p else "" for p in common])

        filename = "score_correlation"
        if group_name:
            filename += f"_{group_name}"
        npz_path = raw_dir / f"{filename}.npz"
        np.savez(
            npz_path,
            pair_ids=pair_ids,
            user_a=user_a,
            user_b=user_b,
            normalized_embed_score=embed,
            normalized_llm_score=llm,
        )
        _save_meta(
            raw_dir / f"{filename}.meta.json",
            {
                "description": "Per-pair normalized scores behind the correlation "
                "plots. Row i corresponds to pair_ids[i]; user_a/user_b are a "
                "best-effort split of the pair_id (authoritative label is pair_ids).",
                "n_pairs": len(common),
                "group_name": group_name,
            },
        )
        print(f"💾 Saved score correlation raw data: {npz_path}")
        return str(npz_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"⚠️ Skipped saving score correlation raw data (non-fatal): {exc}")
        return None
