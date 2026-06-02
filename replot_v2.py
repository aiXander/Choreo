#!/usr/bin/env python3
"""Re-export every plot into a fresh ``plots_v2/`` from on-disk data.

This is a *re-plotting* helper, not a re-run of the pipeline. It loads the data
that already sits on disk for a finished run, lets you tweak the inputs
(currently: cleaning up the user labels), and then calls the **exact same
plotting functions** the pipeline uses — so the visuals stay identical except
for whatever you change here.

Data sources (all read-only):
  * similarity heatmaps      ← ``outputs/plots/raw_data/similarity_matrices.npz``
  * score-correlation plots  ← ``outputs/plots/raw_data/score_correlation_*.npz``
  * t-SNE scatter plots      ← ``embeds/`` (the saved t-SNE coords can't be fed
                                to ``create_tsne_plots``, which takes raw
                                embeddings; with ``random_state=42`` t-SNE is
                                deterministic, so re-running reproduces the same
                                layout — just with the new labels).

The existing ``plots/`` and ``plots/raw_data/`` are NEVER touched. Everything is
rendered into a staging dir and then moved into ``outputs/plots_v2/`` (which is
wiped and rebuilt on every run, so re-running is idempotent).

Usage:
    uv run python replot_v2.py            # uses EXPORT_DIR below
    uv run python replot_v2.py /path/to/<export>   # or pass the export dir
"""

import shutil
import sys
from pathlib import Path

import numpy as np

# --- make src/ importable so we can reuse the pipeline's plotting functions ---
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import load_yaml  # noqa: E402
from embed import load_embeddings, supports_mrl, truncate_embeddings  # noqa: E402
from tsne import create_tsne_plots  # noqa: E402
from visualize_similarity import create_similarity_plots  # noqa: E402
from score_correlation import (  # noqa: E402
    create_normalized_score_correlation_plot,
    create_normalized_detailed_score_analysis,
)

# ======================================================================
# CONFIG — tweak these at will, then re-run.
# ======================================================================

# The export folder containing {outputs/, embeds/, ...}. Overridable via argv[1].
EXPORT_DIR = Path(
    "/Users/xandersteenbrugge/Documents/GitHub/agent_frameworks/motherbrain/"
    "packages/db/transcript_exports_wintercircus_success"
)

# Group name used in score-correlation plot titles + filenames.
GROUP_NAME = "wintercircus"

# Prefix stripped off every label. Set to "" to disable.
LABEL_PREFIX = "wintercircus_"


def relabel(label: str) -> str:
    """Transform a single user-id label. Edit me to rename/clean labels.

    Currently: drop the ``LABEL_PREFIX`` wherever it appears so
    ``wintercircus_Alexander_Landser`` → ``Alexander_Landser``.
    """
    if not LABEL_PREFIX:
        return label
    return label.replace(LABEL_PREFIX, "")


# ======================================================================
# Re-plotting (reuses the pipeline's functions verbatim, with new labels)
# ======================================================================


def replot_similarity(raw_data_dir: Path, staging_dir: Path) -> None:
    """Re-render the similarity heatmaps from the saved matrices npz."""
    npz_path = raw_data_dir / "similarity_matrices.npz"
    meta = load_yaml(raw_data_dir / "similarity_matrices.meta.json")  # JSON is valid YAML
    if not npz_path.exists():
        print(f"⏭  No similarity matrices at {npz_path}; skipping.")
        return

    data = np.load(npz_path, allow_pickle=True)
    key_to_name = meta.get("key_to_name", {})

    user_ids = [relabel(str(u)) for u in data["user_ids"]]

    section_matrices = {}
    cross_section_matrices = {}
    combined_matrix = None
    for key in data.files:
        if key == "user_ids":
            continue
        name = key_to_name.get(key, key)
        if key == "combined":
            combined_matrix = data[key]
        elif key.startswith("section_"):
            section_matrices[name] = data[key]
        elif key.startswith("cross_"):
            cross_section_matrices[name] = data[key]

    matrices_dict = {
        "section_matrices": section_matrices,
        "cross_section_matrices": cross_section_matrices,
        "combined_matrix": combined_matrix,
        "section_weights": meta.get("section_weights", {}),
        "cross_section_weights": meta.get("cross_section_weights", {}),
    }

    print(f"🎨 Similarity: {len(section_matrices)} sections, "
          f"{len(user_ids)} users → re-plotting")
    create_similarity_plots(
        matrices_dict=matrices_dict,
        user_ids=user_ids,
        recipe_config={},
        output_dir=str(staging_dir),
        group_name=GROUP_NAME,
    )


def replot_tsne(embeds_dir: Path, staging_dir: Path) -> None:
    """Re-render the t-SNE scatters from the saved embeddings (deterministic)."""
    if not (embeds_dir / "vectors.npz").exists():
        print(f"⏭  No embeddings at {embeds_dir}; skipping t-SNE.")
        return

    user_ids, section_names, embeddings = load_embeddings(str(embeds_dir))

    # Mirror main.py: optionally MRL-truncate the working copy so the layout
    # matches what the pipeline produced.
    config = load_yaml(REPO_ROOT / "config" / "config.yaml")
    dims = config.get("models", {}).get("embedding_dimensions")
    model = config.get("models", {}).get("embedding")
    if dims and supports_mrl(model):
        embeddings = truncate_embeddings(embeddings, dims)
        print(f"   MRL-truncated to {dims} dims: {embeddings.shape}")

    labels = [relabel(str(u)) for u in user_ids]

    print(f"🎨 t-SNE: {embeddings.shape[0]} users, {len(section_names)} sections "
          f"→ re-plotting")
    create_tsne_plots(
        embeddings=embeddings,
        user_ids=labels,
        section_names=section_names,
        output_dir=str(staging_dir),
        metric="cosine",
        perplexity=7,
    )


def replot_score_correlation(raw_data_dir: Path, staging_dir: Path) -> None:
    """Re-render the embed-vs-LLM score correlation plots from the saved npz."""
    matches = sorted(raw_data_dir.glob("score_correlation*.npz"))
    if not matches:
        print(f"⏭  No score-correlation npz in {raw_data_dir}; skipping.")
        return
    npz_path = matches[0]

    data = np.load(npz_path, allow_pickle=True)
    # pair_id is the authoritative key; user_a/user_b can't be split reliably
    # because the user IDs themselves contain "_". Relabel the pair_id directly.
    pair_ids = [relabel(str(p)) for p in data["pair_ids"]]
    embed_scores = {pid: float(s) for pid, s in zip(pair_ids, data["normalized_embed_score"])}
    llm_scores = {pid: float(s) for pid, s in zip(pair_ids, data["normalized_llm_score"])}

    print(f"🎨 Score correlation: {len(pair_ids)} pairs → re-plotting")
    create_normalized_score_correlation_plot(
        normalized_embed_scores=embed_scores,
        normalized_llm_scores=llm_scores,
        output_dir=str(staging_dir),
        group_name=GROUP_NAME,
    )
    create_normalized_detailed_score_analysis(
        normalized_embed_scores=embed_scores,
        normalized_llm_scores=llm_scores,
        output_dir=str(staging_dir),
        group_name=GROUP_NAME,
    )


def main() -> None:
    export_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else EXPORT_DIR
    outputs_dir = export_dir / "outputs"
    embeds_dir = export_dir / "embeds"
    raw_data_dir = outputs_dir / "plots" / "raw_data"
    dest_dir = outputs_dir / "plots_v2"

    if not outputs_dir.exists():
        sys.exit(f"❌ outputs/ not found under {export_dir}")

    # Stage into a sibling temp dir; the plotting functions hardcode a "plots"
    # subfolder, so we let them write to <staging>/plots and move it to plots_v2.
    # This guarantees the existing plots/ and plots/raw_data/ are never touched.
    staging_dir = export_dir / ".plots_v2_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    print(f"📂 Export:  {export_dir}")
    print(f"📤 Writing: {dest_dir}\n")

    try:
        replot_similarity(raw_data_dir, staging_dir)
        replot_tsne(embeds_dir, staging_dir)
        replot_score_correlation(raw_data_dir, staging_dir)

        staged_plots = staging_dir / "plots"
        if not staged_plots.exists():
            sys.exit("❌ Nothing was plotted — no output produced.")

        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(staged_plots), str(dest_dir))
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    n_imgs = len(list(dest_dir.glob("*.jpg")))
    print(f"\n✅ Done. {n_imgs} plots written to {dest_dir}")
    print("   (existing plots/ and plots/raw_data/ left untouched)")


if __name__ == "__main__":
    main()
