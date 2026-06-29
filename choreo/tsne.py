"""t-SNE visualization for user embeddings.

Plots render in a slide-ready **dark, 16:9** style: dots are coloured by an
on-the-fly KMeans cluster of the 2D layout (so the cohort reads as
self-organising neighbourhoods rather than one undifferentiated blob), each
point gets a soft glow, and labels are haloed for legibility over dense
regions. Output files are exact 16:9 (3840×2160 at the default DPI) so they
drop straight onto a slide with no cropping. ``adjustText`` (an optional
``choreo[plots]`` dependency) is used to de-collide labels when available;
without it labels still render, just unmoved.
"""

import numpy as np
from pathlib import Path
from typing import List, Sequence

from .utils import ensure_dir
from .raw_data import save_tsne_raw_data


# --- Slide-ready dark theme (16:9) -------------------------------------------
# Per-figure facecolors are set explicitly (never via global rcParams) so this
# styling never leaks into the similarity / score-correlation plots that may be
# drawn in the same process.
_DARK_THEME = {
    "bg": "#0d1117",     # figure + axes background
    "fg": "#e6edf3",     # titles
    "sub": "#8b98a8",    # subtitle, axis labels, ticks, leader lines
    "grid": "#1c2430",   # faint grid
    "label": "#e6edf3",  # point labels
    "halo": "#0d1117",   # stroke behind labels (matches bg)
}
# Curated modern qualitative palette — vibrant on dark without being garish.
_CLUSTER_PALETTE = [
    "#4cc9f0", "#4895ef", "#7b6cf6", "#b15cff", "#f72585",
    "#ff7b9c", "#ffb703", "#2ec4b6", "#7ae582", "#ff8c42",
    "#5e60ce", "#56cfe1",
]
_FIGSIZE_16x9 = (16, 9)
_SAVE_DPI = 240          # 16in × 240 = 3840 px wide → 3840×2160 (4K, 16:9)
_SUBTITLE = "t-SNE map of community embeddings"


def _cluster_colors(coords: np.ndarray, k: int = 8) -> List[str]:
    """Colour each point by a KMeans cluster of the 2D layout (cosmetic grouping)."""
    from sklearn.cluster import KMeans

    n = len(coords)
    if n < 2:
        return [_CLUSTER_PALETTE[0]] * n
    k = max(1, min(k, n))
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(coords)
    return [_CLUSTER_PALETTE[int(c) % len(_CLUSTER_PALETTE)] for c in labels]


def _glow_scatter(ax, x, y, colors: Sequence[str], base: float) -> None:
    """Plot points as a soft halo → mid → solid core stack for a subtle glow."""
    ax.scatter(x, y, s=base * 6.0, c=colors, alpha=0.18, linewidths=0, zorder=2)
    ax.scatter(x, y, s=base * 2.6, c=colors, alpha=0.30, linewidths=0, zorder=3)
    ax.scatter(x, y, s=base, c=colors, alpha=0.95, linewidths=0.5,
               edgecolors=_DARK_THEME["bg"], zorder=4)


def _place_labels(ax, coords: np.ndarray, labels: Sequence[str], fontsize: float) -> None:
    """Draw haloed labels, de-colliding with adjustText when it's installed."""
    import matplotlib.patheffects as pe

    theme = _DARK_THEME
    texts = [
        ax.text(x, y, lab, fontsize=fontsize, color=theme["label"], zorder=6,
                path_effects=[pe.withStroke(linewidth=1.6, foreground=theme["halo"])])
        for (x, y), lab in zip(coords, labels)
    ]
    try:
        from adjustText import adjust_text
    except ImportError:
        return
    adjust_text(
        texts, ax=ax, expand=(1.05, 1.15),
        arrowprops=dict(arrowstyle="-", color=theme["sub"], lw=0.4, alpha=0.5),
        min_arrow_len=4,
    )


def _new_dark_axes():
    """Create a 16:9 dark figure/axes pair (facecolor set per-figure, not global)."""
    import matplotlib.pyplot as plt

    theme = _DARK_THEME
    fig, ax = plt.subplots(figsize=_FIGSIZE_16x9, facecolor=theme["bg"])
    ax.set_facecolor(theme["bg"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def _style_dark_axes(ax, title: str, subtitle: str = _SUBTITLE) -> None:
    """Apply the shared title/subtitle/axis treatment to a dark axes."""
    theme = _DARK_THEME
    ax.set_title(title, fontsize=20, fontweight="bold", color=theme["fg"], loc="left", pad=16)
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
            fontsize=10.5, color=theme["sub"], ha="left", va="bottom")
    ax.set_xlabel("t-SNE 1", fontsize=11, color=theme["sub"])
    ax.set_ylabel("t-SNE 2", fontsize=11, color=theme["sub"])
    ax.tick_params(colors=theme["sub"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_dark_16x9(fig, path) -> None:
    """Save without tight-crop so the file stays an exact 16:9 frame for slides."""
    import matplotlib.pyplot as plt

    # Manual margins leave room for the left-aligned title band; no bbox_inches
    # (which would crop to content and break the 16:9 ratio).
    fig.subplots_adjust(left=0.05, right=0.985, top=0.88, bottom=0.075)
    fig.savefig(path, dpi=_SAVE_DPI, facecolor=_DARK_THEME["bg"])
    plt.close(fig)


def create_tsne_plots(
    embeddings: np.ndarray,
    user_ids: List[str], 
    section_names: List[str],
    output_dir: str,
    metric: str = 'cosine',
    perplexity: int = 7
) -> dict:
    """
    Create t-SNE plots for each section and a combined distance matrix plot.
    
    Args:
        embeddings: Array of shape (n_users, n_sections, dim)
        user_ids: List of user identifiers
        section_names: List of section names
        output_dir: Directory to save plots
        metric: Distance metric for t-SNE
        perplexity: Perplexity parameter for t-SNE
        
    Returns:
        Dictionary with plot paths and metadata
    """
    # Lazy imports for heavy dependencies
    from sklearn.manifold import TSNE

    plots_dir = ensure_dir(Path(output_dir) / "plots")

    n_users, n_sections, dim = embeddings.shape
    print(f"Creating t-SNE plots for {n_users} users, {n_sections} sections, {dim} dimensions")

    # Collect the (stochastic) 2D layouts so they can be persisted for re-plotting.
    section_coords_by_name = {}

    results = {
        'plots_dir': str(plots_dir),
        'section_plots': {},
        'combined_plot': None,
        'metadata': {
            'n_users': n_users,
            'n_sections': n_sections,
            'embedding_dim': dim,
            'metric': metric,
            'perplexity': perplexity
        }
    }
    
    # Create t-SNE plot for each section
    for section_idx, section_name in enumerate(section_names):
        print(f"Creating t-SNE plot for section: {section_name}")
        
        # Extract embeddings for this section
        section_embeddings = embeddings[:, section_idx, :]  # Shape: (n_users, dim)
        
        # Create t-SNE
        tsne = TSNE(
            n_components=2,
            metric=metric,
            perplexity=min(perplexity, n_users - 1),  # Ensure perplexity is valid
            random_state=42
        )
        
        tsne_coords = tsne.fit_transform(section_embeddings)
        section_coords_by_name[section_name] = tsne_coords

        # Slide-ready dark 16:9 plot.
        fig, ax = _new_dark_axes()
        _glow_scatter(ax, tsne_coords[:, 0], tsne_coords[:, 1],
                      _cluster_colors(tsne_coords), base=95)
        _place_labels(ax, tsne_coords, user_ids, fontsize=6.0)
        _style_dark_axes(ax, section_name)

        plot_path = plots_dir / f"tsne_{section_name.lower().replace(' ', '_')}.jpg"
        _save_dark_16x9(fig, plot_path)

        results['section_plots'][section_name] = str(plot_path)
        print(f"Saved t-SNE plot for {section_name}: {plot_path}")
    
    # Create combined distance matrix t-SNE
    print("Creating combined distance matrix t-SNE plot")
    
    # Compute pairwise distances across all sections
    combined_distances = compute_combined_distances(embeddings, metric=metric)
    
    # Create t-SNE with precomputed distances
    tsne_combined = TSNE(
        n_components=2,
        metric='precomputed',
        init='random',  # Required when using precomputed distances
        perplexity=min(perplexity, n_users - 1),
        random_state=42
    )
    
    tsne_coords_combined = tsne_combined.fit_transform(combined_distances)

    # Slide-ready dark 16:9 combined plot.
    fig, ax = _new_dark_axes()
    _glow_scatter(ax, tsne_coords_combined[:, 0], tsne_coords_combined[:, 1],
                  _cluster_colors(tsne_coords_combined), base=120)
    _place_labels(ax, tsne_coords_combined, user_ids, fontsize=6.75)
    _style_dark_axes(ax, "Combined (all sections)")

    combined_plot_path = plots_dir / "tsne_combined_all_sections.jpg"
    _save_dark_16x9(fig, combined_plot_path)

    results['combined_plot'] = str(combined_plot_path)
    print(f"Saved combined t-SNE plot: {combined_plot_path}")

    # Persist the raw 2D coordinates (crash-safe: never propagates errors).
    save_tsne_raw_data(
        output_dir=output_dir,
        section_coords=section_coords_by_name,
        combined_coords=tsne_coords_combined,
        user_ids=user_ids,
        metric=metric,
        perplexity=perplexity,
    )

    return results


def compute_combined_distances(embeddings: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    Compute combined pairwise distances across all sections.
    
    Args:
        embeddings: Array of shape (n_users, n_sections, dim)
        metric: Distance metric to use
        
    Returns:
        Combined distance matrix of shape (n_users, n_users)
    """
    # Lazy import for sklearn
    from sklearn.metrics.pairwise import cosine_distances
    
    n_users, n_sections, dim = embeddings.shape
    
    # Initialize combined distance matrix
    combined_distances = np.zeros((n_users, n_users))
    
    # Compute distances for each section and aggregate
    for section_idx in range(n_sections):
        section_embeddings = embeddings[:, section_idx, :]
        
        if metric == 'cosine':
            section_distances = cosine_distances(section_embeddings)
        else:
            # For other metrics, you could add support here
            raise ValueError(f"Metric {metric} not supported yet")
        
        combined_distances += section_distances
    
    # Average across sections
    combined_distances /= n_sections
    
    return combined_distances


def visualize_section_relationships(
    embeddings: np.ndarray,
    user_ids: List[str],
    section_names: List[str],
    output_dir: str
) -> str:
    """
    Create a visualization showing how sections relate to each other.
    
    Args:
        embeddings: Array of shape (n_users, n_sections, dim)
        user_ids: List of user identifiers  
        section_names: List of section names
        output_dir: Directory to save plots
        
    Returns:
        Path to the saved plot
    """
    # Lazy imports for heavy dependencies
    from sklearn.manifold import TSNE

    plots_dir = ensure_dir(Path(output_dir) / "plots")

    n_users, n_sections, dim = embeddings.shape

    # Compute average embedding per section across all users
    section_centroids = np.mean(embeddings, axis=0)  # Shape: (n_sections, dim)

    # Create t-SNE for sections. Perplexity must stay below the sample count,
    # and here there are only n_sections points (often just a handful).
    tsne = TSNE(
        n_components=2, random_state=42, metric='cosine',
        perplexity=max(1, min(30, n_sections - 1)),
    )
    section_tsne = tsne.fit_transform(section_centroids)

    # Slide-ready dark 16:9 plot (one larger dot per section centroid).
    fig, ax = _new_dark_axes()
    _glow_scatter(ax, section_tsne[:, 0], section_tsne[:, 1],
                  _cluster_colors(section_tsne, k=n_sections), base=240)
    _place_labels(ax, section_tsne, section_names, fontsize=11)
    _style_dark_axes(ax, "Section relationships",
                     subtitle="t-SNE of section centroids")

    plot_path = plots_dir / "tsne_section_relationships.jpg"
    _save_dark_16x9(fig, plot_path)

    print(f"Saved section relationships plot: {plot_path}")

    # Persist the raw centroid coordinates (crash-safe).
    save_tsne_raw_data(
        output_dir=output_dir,
        section_relationship_coords=section_tsne,
        section_names=section_names,
        metric="cosine",
        filename="tsne_section_relationships",
    )

    return str(plot_path)