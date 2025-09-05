#!/usr/bin/env python3
"""
Visualize similarity matrices from the candidate generation algorithm.
Creates individual section plots and a combined visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import argparse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_yaml, ensure_dir
from ingest import load_profiles
from extract import extract_sections_from_profiles
from embed import create_section_embeddings
from candidate import generate_similarity_matrix
from llm import LLMWrapper


def create_similarity_plots(
    matrices_dict: dict,
    user_ids: list,
    recipe_config: dict,
    output_dir: str,
    group_name: str = None
):
    """Create and save similarity matrix visualizations using precomputed matrices."""
    
    # Create output directory
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(plots_dir)
    
    # Extract matrices and weights from the dict
    section_matrices = matrices_dict['section_matrices']
    section_weights = matrices_dict['section_weights']
    combined_matrix = matrices_dict['combined_matrix']
    
    print("Creating individual section similarity plots...")
    
    # Create individual plots for each section
    for section_name, similarity_matrix in section_matrices.items():
        weight = section_weights.get(section_name, 0.0)
        
        # Create individual plot
        plt.figure(figsize=(10, 10))  # Square figure
        plt.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
        plt.colorbar(label='Similarity Score')
        plt.title(f'{section_name.capitalize()} Similarity Matrix\n(Weight: {weight:.2f})')
        plt.xlabel('User Index')
        plt.ylabel('User Index')
        
        # Add user labels if reasonable number
        if len(user_ids) <= 20:
            plt.xticks(range(len(user_ids)), user_ids, rotation=45, ha='right')
            plt.yticks(range(len(user_ids)), user_ids)
        
        plt.tight_layout()
        plot_path = plots_dir / f'{section_name}_similarity.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {section_name} similarity plot: {plot_path}")
    
    # Create individual combined plot
    print("Creating combined similarity plot...")
    
    # Create individual combined plot
    plt.figure(figsize=(10, 10))  # Square figure
    plt.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
    plt.colorbar(label='Combined Similarity Score')
    recipe_type = 'custom'
    
    # Add weights summary to title
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in section_weights.items()])
    plt.title(f'Combined Similarity Matrix\n(Recipe: {recipe_type})\nWeights: {weights_str}')
    plt.xlabel('User Index')
    plt.ylabel('User Index')
    
    if len(user_ids) <= 20:
        plt.xticks(range(len(user_ids)), user_ids, rotation=45, ha='right')
        plt.yticks(range(len(user_ids)), user_ids)
    
    plt.tight_layout()
    combined_path = plots_dir / 'combined_similarity.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined similarity plot: {combined_path}")
    
    # Create final combined visualization
    print("Creating final combined visualization...")
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
        ax.set_xlabel('User Index', fontsize=10)
        ax.set_ylabel('User Index', fontsize=10)
        
        if len(user_ids) <= 10:
            ax.set_xticks(range(len(user_ids)))
            ax.set_xticklabels(user_ids, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(user_ids)))
            ax.set_yticklabels(user_ids, fontsize=8)
    
    # Plot combined matrix on the right
    ax_combined = fig.add_subplot(gs[:, 2])
    im_combined = ax_combined.imshow(combined_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')  # Equal aspect ratio
    weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in section_weights.items()])
    ax_combined.set_title(f'Combined Matrix\n(Recipe: {recipe_type})\nWeights: {weights_str}', fontsize=14)
    ax_combined.set_xlabel('User Index', fontsize=12)
    ax_combined.set_ylabel('User Index', fontsize=12)
    
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
    final_path = plots_dir / 'similarity_analysis.png'
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final combined visualization: {final_path}")
    
    return {
        'section_matrices': section_matrices,
        'combined_matrix': combined_matrix,
        'plots_dir': plots_dir
    }


def main(group_name: str = None, force: bool = False):
    """Main function to generate similarity visualizations."""
    
    load_dotenv()
    
    print("🎨 Starting similarity matrix visualization...")
    
    # Load configurations
    config = load_yaml("config/config.yaml")
    
    # Update paths for group-specific data if group_name provided
    if group_name:
        print(f"📁 Using group-specific data: {group_name}")
        base_data_dir = f"data/{group_name}"
        config['io']['raw_dir'] = f"{base_data_dir}/raw"
        config['io']['processed_dir'] = f"{base_data_dir}/processed"
        config['io']['embeds_dir'] = f"{base_data_dir}/embeds"
        config['io']['outputs_dir'] = f"{base_data_dir}/outputs"
        config['io']['cache_dir'] = f"{base_data_dir}/cache"
    
    sections_config_path = "config/section_prompt.yaml"
    
    # Initialize LLM wrapper
    llm_wrapper = LLMWrapper(cache_dir=config['io']['cache_dir'])
    
    print("\n📁 Step 1: Loading profiles...")
    profiles = load_profiles(config['io']['raw_dir'])
    print(f"✅ Loaded {len(profiles)} profiles")
    
    print("\n🧠 Step 2: Loading/extracting sections...")
    try:
        goal = config['instruction_prompt']['goal']
        extracted_sections = extract_sections_from_profiles(
            profiles=profiles,
            sections_config_path=sections_config_path,
            model=config['models']['extraction_llm'],
            llm_wrapper=llm_wrapper,
            processed_dir=config['io']['processed_dir'],
            budgets=config['budgets'],
            goal=goal,
            force=force
        )
        print(f"✅ Loaded sections for {len(extracted_sections)} profiles")
    except Exception as e:
        print(f"❌ Error loading sections: {e}")
        return 1
    
    print("\n🔢 Step 3: Loading/creating embeddings...")
    user_ids, section_names, embeddings = create_section_embeddings(
        extracted_sections=extracted_sections,
        embedding_model=config['models']['embedding'],
        embeds_dir=config['io']['embeds_dir'],
        force=force
    )
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    
    print("\n🎨 Step 4: Creating similarity visualizations...")
    
    # Generate similarity matrices using the same logic as main pipeline
    _, user_ids_sorted, matrices_dict = generate_similarity_matrix(
        embeddings=embeddings,
        user_ids=user_ids,
        section_names=section_names,
        recipe_config=config['recipe']
    )
    
    results = create_similarity_plots(
        matrices_dict=matrices_dict,
        user_ids=user_ids_sorted,
        recipe_config=config['recipe'],
        output_dir=config['io']['outputs_dir'],
        group_name=group_name
    )
    
    print(f"\n🎉 Visualization completed successfully!")
    print(f"📁 Plots saved to: {results['plots_dir']}")
    print(f"📊 Section matrices: {len(results['section_matrices'])} plots")
    print(f"📈 Combined matrix shape: {results['combined_matrix'].shape}")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize similarity matrices")
    parser.add_argument("--group", type=str, help="Group name for data organization")
    parser.add_argument("--force", action="store_true", help="Force re-run all steps")
    
    args = parser.parse_args()
    exit_code = main(group_name=args.group, force=args.force)
    sys.exit(exit_code)