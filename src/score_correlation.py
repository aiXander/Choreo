"""Create correlation visualization between embedding scores and LLM scores."""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

from score import PairScore
from utils import ensure_dir


def create_normalized_score_correlation_plot(
    normalized_embed_scores: Dict[str, float],
    normalized_llm_scores: Dict[str, float],
    output_dir: str,
    group_name: Optional[str] = None
) -> str:
    """
    Create a scatterplot showing correlation between normalized embedding and LLM scores.
    
    Args:
        normalized_embed_scores: Normalized embedding scores by pair_id
        normalized_llm_scores: Normalized LLM scores by pair_id
        output_dir: Directory to save the plot
        group_name: Optional group name for plot title and filename
        
    Returns:
        Path to saved plot file
    """
    # Lazy imports for heavy dependencies
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if not normalized_embed_scores or not normalized_llm_scores:
        print("No normalized scores to plot")
        return ""
    
    # Extract normalized scores
    common_pair_ids = set(normalized_embed_scores.keys()) & set(normalized_llm_scores.keys())
    
    embed_scores = []
    llm_score_values = []
    pair_labels = []
    
    for pair_id in common_pair_ids:
        embed_scores.append(normalized_embed_scores[pair_id])
        llm_score_values.append(normalized_llm_scores[pair_id])
        # Extract user names from pair_id (assuming format: user1_user2)
        users = pair_id.split('_', 1)
        if len(users) == 2:
            pair_labels.append(f"{users[0]}-{users[1]}")
        else:
            pair_labels.append(pair_id)
    
    embed_scores = np.array(embed_scores)
    llm_score_values = np.array(llm_score_values)
    
    # Calculate correlation statistics
    correlation_coeff, p_value = stats.pearsonr(embed_scores, llm_score_values)
    r_squared = correlation_coeff ** 2
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(embed_scores, llm_score_values, alpha=0.7, s=50, color='blue', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(embed_scores, llm_score_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(embed_scores.min(), embed_scores.max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (R² = {r_squared:.3f})')
    
    # Labels and title
    plt.xlabel('Normalized Embedding Similarity Score', fontsize=12)
    plt.ylabel('Normalized LLM Score', fontsize=12)
    
    title = 'Normalized Embedding vs LLM Score Correlation'
    if group_name:
        title += f' - {group_name}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add statistics text box
    stats_text = f'Correlation: r = {correlation_coeff:.3f}\n'
    stats_text += f'R-squared: {r_squared:.3f}\n'
    stats_text += f'P-value: {p_value:.4f}\n'
    stats_text += f'N pairs: {len(embed_scores)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='monospace')
    
    # Grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits with some padding
    x_margin = (embed_scores.max() - embed_scores.min()) * 0.05
    y_margin = (llm_score_values.max() - llm_score_values.min()) * 0.05
    plt.xlim(embed_scores.min() - x_margin, embed_scores.max() + x_margin)
    plt.ylim(llm_score_values.min() - y_margin, llm_score_values.max() + y_margin)
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(plots_dir)
    
    filename = "normalized_embedding_llm_score_correlation"
    if group_name:
        filename += f"_{group_name}"
    filename += ".png"
    
    plot_path = plots_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved normalized score correlation plot: {plot_path}.  Pairs plotted: {len(embed_scores)}")
    
    return str(plot_path)


def create_normalized_detailed_score_analysis(
    normalized_embed_scores: Dict[str, float],
    normalized_llm_scores: Dict[str, float],
    output_dir: str,
    group_name: Optional[str] = None
) -> str:
    """
    Create a detailed analysis plot with histograms and scatter plot using normalized scores.
    
    Args:
        normalized_embed_scores: Normalized embedding scores by pair_id
        normalized_llm_scores: Normalized LLM scores by pair_id
        output_dir: Directory to save the plot
        group_name: Optional group name for plot title
        
    Returns:
        Path to saved plot file
    """
    # Lazy imports for heavy dependencies
    import matplotlib.pyplot as plt
    from scipy import stats
    
    if not normalized_embed_scores or not normalized_llm_scores:
        print("No normalized scores to analyze")
        return ""
    
    # Extract normalized data
    common_pair_ids = set(normalized_embed_scores.keys()) & set(normalized_llm_scores.keys())
    
    embed_scores = [normalized_embed_scores[pair_id] for pair_id in common_pair_ids]
    llm_score_values = [normalized_llm_scores[pair_id] for pair_id in common_pair_ids]
    
    embed_scores = np.array(embed_scores)
    llm_score_values = np.array(llm_score_values)
    
    # Calculate correlation
    correlation_coeff, p_value = stats.pearsonr(embed_scores, llm_score_values)
    r_squared = correlation_coeff ** 2
    
    # Create subplot figure
    fig = plt.figure(figsize=(15, 10))
    
    # Main scatter plot
    ax1 = plt.subplot(2, 2, (3, 4))  # Bottom row, full width
    ax1.scatter(embed_scores, llm_score_values, alpha=0.7, s=60, color='blue', edgecolors='black', linewidth=0.5)
    
    # Trend line
    z = np.polyfit(embed_scores, llm_score_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(embed_scores.min(), embed_scores.max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Normalized Embedding Similarity Score', fontsize=12)
    ax1.set_ylabel('Normalized LLM Score', fontsize=12)
    ax1.set_title(f'Normalized Score Correlation (R² = {r_squared:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of embedding scores
    ax2 = plt.subplot(2, 2, 1)
    ax2.hist(embed_scores, bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Normalized Embedding Score', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Normalized Embedding Distribution', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Histogram of LLM scores
    ax3 = plt.subplot(2, 2, 2)
    ax3.hist(llm_score_values, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_xlabel('Normalized LLM Score', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Normalized LLM Distribution', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Add overall title
    title = 'Normalized Score Analysis Dashboard'
    if group_name:
        title += f' - {group_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add statistics
    stats_text = f'Correlation: r = {correlation_coeff:.3f}\\n'
    stats_text += f'R-squared: {r_squared:.3f}\\n'
    stats_text += f'P-value: {p_value:.4f}\\n'
    stats_text += f'N pairs: {len(embed_scores)}\\n'
    stats_text += f'Embed range: [{embed_scores.min():.3f}, {embed_scores.max():.3f}]\\n'
    stats_text += f'LLM range: [{llm_score_values.min():.3f}, {llm_score_values.max():.3f}]'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9, fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(plots_dir)
    
    filename = "normalized_detailed_score_analysis"
    if group_name:
        filename += f"_{group_name}"
    filename += ".png"
    
    plot_path = plots_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved normalized detailed score analysis: {plot_path}")
    
    return str(plot_path)