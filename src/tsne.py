"""t-SNE visualization for user embeddings."""

import numpy as np
from pathlib import Path
from typing import List

from utils import ensure_dir
from raw_data import save_tsne_raw_data


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
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import seaborn as sns
    
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    
    n_users, n_sections, dim = embeddings.shape
    print(f"Creating t-SNE plots for {n_users} users, {n_sections} sections, {dim} dimensions")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
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

        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne_coords[:, 0], 
            tsne_coords[:, 1], 
            alpha=0.7,
            s=100,
            c=range(n_users),
            cmap='tab20'
        )
        
        # Add user labels
        for i, user_id in enumerate(user_ids):
            plt.annotate(
                user_id, 
                (tsne_coords[i, 0], tsne_coords[i, 1]),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
        
        plt.title(f't-SNE Visualization: {section_name}', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = plots_dir / f"tsne_{section_name.lower().replace(' ', '_')}.jpg"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
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
    
    # Create combined plot
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        tsne_coords_combined[:, 0], 
        tsne_coords_combined[:, 1], 
        alpha=0.7,
        s=120,
        c=range(n_users),
        cmap='tab20'
    )
    
    # Add user labels
    for i, user_id in enumerate(user_ids):
        plt.annotate(
            user_id, 
            (tsne_coords_combined[i, 0], tsne_coords_combined[i, 1]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            fontweight='bold'
        )
    
    plt.title('t-SNE Visualization: Combined Distance Matrix (All Sections)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save combined plot
    combined_plot_path = plots_dir / "tsne_combined_all_sections.jpg"
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    
    n_users, n_sections, dim = embeddings.shape
    
    # Compute average embedding per section across all users
    section_centroids = np.mean(embeddings, axis=0)  # Shape: (n_sections, dim)
    
    # Create t-SNE for sections
    tsne = TSNE(n_components=2, random_state=42, metric='cosine')
    section_tsne = tsne.fit_transform(section_centroids)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(section_tsne[:, 0], section_tsne[:, 1], s=200, alpha=0.7)
    
    for i, section_name in enumerate(section_names):
        plt.annotate(
            section_name,
            (section_tsne[i, 0], section_tsne[i, 1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.title('Section Relationships (t-SNE of Section Centroids)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)

    plot_path = plots_dir / "tsne_section_relationships.jpg"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

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