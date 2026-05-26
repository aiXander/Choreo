#!/usr/bin/env python3
"""
Visualize similarity matrices from the candidate generation algorithm.
Creates individual section plots and a combined visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import ensure_dir


def create_similarity_plots(
    matrices_dict: dict,
    user_ids: list,
    recipe_config: dict,
    output_dir: str,
    group_name: str = None
):
    """Create and save similarity matrix visualizations using precomputed matrices."""
    
    # Lazy import for matplotlib
    import matplotlib.pyplot as plt
    
    # Create output directory
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(plots_dir)
    
    # Extract matrices and weights from the dict
    section_matrices = matrices_dict['section_matrices']
    section_weights = matrices_dict['section_weights']
    combined_matrix = matrices_dict['combined_matrix']
    
    # Create individual plots for each section
    for section_name, similarity_matrix in section_matrices.items():
        weight = section_weights.get(section_name, 0.0)
        
        # Create individual plot
        plt.figure(figsize=(10, 10))  # Square figure
        plt.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
        plt.colorbar(label='Similarity Score')
        plt.title(f'{section_name.capitalize()} Similarity Matrix\n(Weight: {weight:.2f})')
        
        # Add user labels if reasonable number
        if len(user_ids) <= 20:
            plt.xticks(range(len(user_ids)), user_ids, rotation=45, ha='right')
            plt.yticks(range(len(user_ids)), user_ids)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{section_name}_similarity.jpg'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {section_name} similarity plot: {plot_path}")
    
    # Create individual combined plot
    plt.figure(figsize=(10, 10))  # Square figure
    plt.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
    plt.colorbar(label='Combined Similarity Score')
    recipe_type = 'custom'
    
    # Add weights summary to title
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in section_weights.items()])
    plt.title(f'Combined Similarity Matrix\n{weights_str}')
    
    if len(user_ids) <= 20:
        plt.xticks(range(len(user_ids)), user_ids, rotation=45, ha='right')
        plt.yticks(range(len(user_ids)), user_ids)
    
    plt.tight_layout()
    combined_path = plots_dir / 'combined_similarity.jpg'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create final combined visualization
    n_sections = len(section_matrices)
    grid_cols = 2
    grid_rows = (n_sections + 1) // 2  # +1 for ceiling division
    
    fig = plt.figure(figsize=(16, 4 * grid_rows + 2))
    
    # Create grid layout: sections on left, combined on right
    gs = fig.add_gridspec(grid_rows, 3, width_ratios=[1, 1, 1.2])
    
    # Plot individual sections in grid
    for idx, (section_name, matrix) in enumerate(section_matrices.items()):
        row = idx // 2
        col = idx % 2
        
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
        weight = section_weights.get(section_name, 0.0)
        ax.set_title(f'{section_name.capitalize()}\n(w={weight:.2f})', 
                    fontsize=12)
        
        if len(user_ids) <= 10:
            ax.set_xticks(range(len(user_ids)))
            ax.set_xticklabels(user_ids, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(user_ids)))
            ax.set_yticklabels(user_ids, fontsize=8)
    
    # Plot combined matrix on the right
    ax_combined = fig.add_subplot(gs[:, 2])
    im_combined = ax_combined.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
    weights_str = ", ".join([f"{k[:8]}: {v:.2f}" for k, v in section_weights.items()])
    ax_combined.set_title(f'Combined Matrix\n{weights_str}', fontsize=14)
    
    if len(user_ids) <= 15:
        ax_combined.set_xticks(range(len(user_ids)))
        ax_combined.set_xticklabels(user_ids, rotation=45, ha='right', fontsize=10)
        ax_combined.set_yticks(range(len(user_ids)))
        ax_combined.set_yticklabels(user_ids, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im_combined, ax=ax_combined, shrink=0.8)
    cbar.set_label('Similarity Score', fontsize=12)
    
    # Add overall title
    group_str = f" ({group_name})" if group_name else ""
    fig.suptitle(f'Similarity Matrix Analysis{group_str}', fontsize=16, y=0.98)
    
    plt.tight_layout()
    final_path = plots_dir / 'similarity_analysis.jpg'
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final combined visualization: {final_path}")
    
    return {
        'section_matrices': section_matrices,
        'combined_matrix': combined_matrix,
        'plots_dir': str(plots_dir)
    }