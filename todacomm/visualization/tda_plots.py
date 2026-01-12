"""
TDA and Geometry Visualization Functions.

Provides plotting utilities for persistence diagrams, Betti curves,
layer-wise TDA metric comparisons, and geometry characterization plots.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def plot_persistence_diagram(
    diagrams: Dict[int, np.ndarray],
    title: str = "Persistence Diagram",
    ax: Optional[plt.Axes] = None,
    max_dim: int = 1,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot persistence diagram showing birth-death pairs.

    Args:
        diagrams: Dictionary mapping dimension to (n, 2) array of (birth, death) pairs
        title: Plot title
        ax: Matplotlib axes (creates new figure if None)
        max_dim: Maximum homology dimension to plot
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    # Find max value for diagonal
    max_val = 0
    for dim, pts in diagrams.items():
        if dim <= max_dim and len(pts) > 0:
            finite_deaths = pts[:, 1][np.isfinite(pts[:, 1])]
            if len(finite_deaths) > 0:
                max_val = max(max_val, np.max(finite_deaths))
            max_val = max(max_val, np.max(pts[:, 0]))

    if max_val == 0:
        max_val = 1

    # Plot diagonal
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, label='Diagonal')

    # Plot points for each dimension
    legend_handles = []
    for dim in range(max_dim + 1):
        if dim in diagrams and len(diagrams[dim]) > 0:
            pts = diagrams[dim]
            births = pts[:, 0]
            deaths = pts[:, 1]

            # Handle infinite deaths (plot at max_val * 1.05)
            deaths = np.where(np.isinf(deaths), max_val * 1.05, deaths)

            ax.scatter(births, deaths, c=colors[dim % len(colors)],
                      marker=markers[dim % len(markers)], s=50, alpha=0.7,
                      edgecolors='black', linewidths=0.5)

            legend_handles.append(mpatches.Patch(
                color=colors[dim % len(colors)],
                label=f'H{dim} ({len(pts)} features)'
            ))

    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(handles=legend_handles, loc='lower right')
    ax.set_xlim(-max_val * 0.05, max_val * 1.1)
    ax.set_ylim(-max_val * 0.05, max_val * 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def plot_layer_tda_comparison(
    tda_summaries: Dict[str, Dict[str, float]],
    metric: str = "total_persistence",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot TDA metrics across layers as grouped bar chart with interpretive annotations.

    Args:
        tda_summaries: Dictionary mapping layer names to TDA summary dicts
        metric: Which metric to plot ('count', 'total_persistence', 'max_lifetime')
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    layers = list(tda_summaries.keys())

    # Extract H0 and H1 values
    h0_values = [tda_summaries[layer].get(f'H0_{metric}', 0) for layer in layers]
    h1_values = [tda_summaries[layer].get(f'H1_{metric}', 0) for layer in layers]

    x = np.arange(len(layers))
    width = 0.6

    # Find peak layers for annotation
    h0_peak_idx = np.argmax(h0_values)
    h1_peak_idx = np.argmax(h1_values)

    # H0 plot
    colors_h0 = ['#1f77b4' if i != h0_peak_idx else '#d62728' for i in range(len(layers))]
    bars0 = axes[0].bar(x, h0_values, width, color=colors_h0, edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel(f'H0 {metric.replace("_", " ").title()}', fontsize=11)
    axes[0].set_title(f'H0 (Connected Components)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars0, h0_values):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(h0_values)*0.02,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Add peak annotation for H0
    if max(h0_values) > 0:
        axes[0].annotate(
            f'Peak: {layers[h0_peak_idx]}\n(max cluster spread)',
            xy=(h0_peak_idx, h0_values[h0_peak_idx]),
            xytext=(h0_peak_idx + 0.5, h0_values[h0_peak_idx] * 0.7),
            fontsize=9, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5)
        )

    # H1 plot
    colors_h1 = ['#ff7f0e' if i != h1_peak_idx else '#d62728' for i in range(len(layers))]
    bars1 = axes[1].bar(x, h1_values, width, color=colors_h1, edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Layer', fontsize=11)
    axes[1].set_ylabel(f'H1 {metric.replace("_", " ").title()}', fontsize=11)
    axes[1].set_title(f'H1 (Loops/Cycles)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, h1_values):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(h1_values)*0.02 if max(h1_values) > 0 else 0.1,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # Add peak annotation for H1
    if max(h1_values) > 0:
        axes[1].annotate(
            f'Peak: {layers[h1_peak_idx]}\n(strongest cycles)',
            xy=(h1_peak_idx, h1_values[h1_peak_idx]),
            xytext=(h1_peak_idx - 0.5 if h1_peak_idx > len(layers)/2 else h1_peak_idx + 0.5,
                    h1_values[h1_peak_idx] * 0.7),
            fontsize=9, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5)
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # Add interpretation note at bottom
    interpretation = _generate_persistence_interpretation(layers, h0_values, h1_values)
    fig.text(0.5, -0.02, interpretation, ha='center', fontsize=9, style='italic',
             wrap=True, transform=fig.transFigure)

    plt.tight_layout()
    return fig


def _generate_persistence_interpretation(layers: List[str], h0_values: List[float], h1_values: List[float]) -> str:
    """Generate a brief interpretation of the persistence values."""
    h0_peak_layer = layers[np.argmax(h0_values)]
    h1_peak_layer = layers[np.argmax(h1_values)]

    h0_spread = max(h0_values) / min(h0_values) if min(h0_values) > 0 else float('inf')

    parts = []
    if h0_spread > 5:
        parts.append(f"H0: {h0_spread:.0f}x variation across layers, peak at {h0_peak_layer} (maximum cluster separation)")
    else:
        parts.append(f"H0: Relatively uniform cluster spread across layers")

    if max(h1_values) > 0:
        parts.append(f"H1: Strongest cyclic structure at {h1_peak_layer}")
    else:
        parts.append("H1: Minimal cyclic structure detected")

    return " | ".join(parts)


def plot_betti_curves(
    tda_summaries: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    Plot Betti numbers (feature counts) across layers with interpretive annotations.

    Args:
        tda_summaries: Dictionary mapping layer names to TDA summary dicts
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = list(tda_summaries.keys())
    x = np.arange(len(layers))

    h0_counts = [tda_summaries[layer].get('H0_count', 0) for layer in layers]
    h1_counts = [tda_summaries[layer].get('H1_count', 0) for layer in layers]

    ax.plot(x, h0_counts, 'o-', color='#1f77b4', linewidth=2, markersize=10,
            label=f'H0 (Connected Components)', markeredgecolor='black')
    ax.plot(x, h1_counts, 's-', color='#ff7f0e', linewidth=2, markersize=10,
            label=f'H1 (Loops)', markeredgecolor='black')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Feature Count', fontsize=12)
    ax.set_title('Betti Numbers Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotations
    for i, (h0, h1) in enumerate(zip(h0_counts, h1_counts)):
        ax.annotate(f'{int(h0)}', (i, h0), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9, color='#1f77b4')
        ax.annotate(f'{int(h1)}', (i, h1), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=9, color='#ff7f0e')

    # Add interpretation box
    h0_constant = len(set(h0_counts)) == 1
    h1_variation = max(h1_counts) - min(h1_counts) if h1_counts else 0

    note_lines = []
    if h0_constant:
        note_lines.append(f"Note: H0={int(h0_counts[0])} is constant (equals sample count)")
        note_lines.append("Focus on persistence metrics for meaningful cluster analysis")
    else:
        note_lines.append("H0 varies - indicates different cluster structures per layer")

    if h1_variation > 0:
        h1_max_layer = layers[np.argmax(h1_counts)]
        note_lines.append(f"H1 peaks at {h1_max_layer} ({int(max(h1_counts))} loops)")

    note_text = "\n".join(note_lines)
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    return fig


def plot_tda_summary(
    tda_summaries: Dict[str, Dict[str, float]],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a comprehensive TDA summary figure with multiple subplots.

    Args:
        tda_summaries: Dictionary mapping layer names to TDA summary dicts
        model_name: Name of the model for title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    layers = list(tda_summaries.keys())
    x = np.arange(len(layers))

    # 1. Feature counts (Betti numbers)
    ax1 = fig.add_subplot(gs[0, 0])
    h0_counts = [tda_summaries[layer].get('H0_count', 0) for layer in layers]
    h1_counts = [tda_summaries[layer].get('H1_count', 0) for layer in layers]

    width = 0.35
    ax1.bar(x - width/2, h0_counts, width, label='H0', color='#1f77b4', edgecolor='black')
    ax1.bar(x + width/2, h1_counts, width, label='H1', color='#ff7f0e', edgecolor='black')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Count')
    ax1.set_title('Feature Counts', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Total persistence
    ax2 = fig.add_subplot(gs[0, 1])
    h0_persist = [tda_summaries[layer].get('H0_total_persistence', 0) for layer in layers]
    h1_persist = [tda_summaries[layer].get('H1_total_persistence', 0) for layer in layers]

    ax2.bar(x - width/2, h0_persist, width, label='H0', color='#1f77b4', edgecolor='black')
    ax2.bar(x + width/2, h1_persist, width, label='H1', color='#ff7f0e', edgecolor='black')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Total Persistence')
    ax2.set_title('Total Persistence', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Max lifetime
    ax3 = fig.add_subplot(gs[0, 2])
    h0_max = [tda_summaries[layer].get('H0_max_lifetime', 0) for layer in layers]
    h1_max = [tda_summaries[layer].get('H1_max_lifetime', 0) for layer in layers]

    ax3.bar(x - width/2, h0_max, width, label='H0', color='#1f77b4', edgecolor='black')
    ax3.bar(x + width/2, h1_max, width, label='H1', color='#ff7f0e', edgecolor='black')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Max Lifetime')
    ax3.set_title('Maximum Lifetime', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. H0 metrics line plot
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(x, h0_persist, 'o-', color='#1f77b4', linewidth=2, markersize=8,
             label='Total Persistence', markeredgecolor='black')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x, h0_max, 's--', color='#2ca02c', linewidth=2, markersize=8,
                  label='Max Lifetime', markeredgecolor='black')

    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Total Persistence', color='#1f77b4')
    ax4_twin.set_ylabel('Max Lifetime', color='#2ca02c')
    ax4.set_title('H0 Metrics Across Layers', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 5. H1 metrics line plot
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(x, h1_counts, 'o-', color='#ff7f0e', linewidth=2, markersize=8,
             label='Count', markeredgecolor='black')
    ax5_twin = ax5.twinx()
    ax5_twin.plot(x, h1_persist, 's--', color='#d62728', linewidth=2, markersize=8,
                  label='Total Persistence', markeredgecolor='black')

    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Count', color='#ff7f0e')
    ax5_twin.set_ylabel('Total Persistence', color='#d62728')
    ax5.set_title('H1 Metrics Across Layers', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    fig.suptitle(f'TDA Analysis Summary: {model_name}', fontsize=16, fontweight='bold', y=1.02)

    # Add interpretation summary at bottom
    h0_peak_layer = layers[np.argmax(h0_persist)]
    h1_peak_layer = layers[np.argmax(h1_persist)]
    h0_spread = max(h0_persist) / min(h0_persist) if min(h0_persist) > 0 else 0

    summary_text = (
        f"Key Insights: H0 persistence peaks at {h0_peak_layer} ({h0_spread:.0f}x spread), "
        f"H1 strongest at {h1_peak_layer}. "
        f"See interpretation report for detailed analysis."
    )
    fig.text(0.5, -0.02, summary_text, ha='center', fontsize=10, style='italic',
             wrap=True, transform=fig.transFigure)

    return fig


def generate_all_visualizations(
    tda_summaries: Dict[str, Dict[str, float]],
    output_dir: Path,
    model_name: str = "Model",
    diagrams: Optional[Dict[str, Dict[int, np.ndarray]]] = None
) -> List[Path]:
    """
    Generate all TDA visualizations and save to output directory.

    Args:
        tda_summaries: Dictionary mapping layer names to TDA summary dicts
        output_dir: Directory to save plots
        model_name: Name of the model for titles
        diagrams: Optional persistence diagrams for each layer

    Returns:
        List of paths to generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # 1. Summary figure
    fig = plot_tda_summary(tda_summaries, model_name)
    summary_path = output_dir / "tda_summary.png"
    fig.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    generated_files.append(summary_path)
    print(f"  Saved: {summary_path}")

    # 2. Layer comparison - Total Persistence
    fig = plot_layer_tda_comparison(
        tda_summaries,
        metric="total_persistence",
        title=f"{model_name}: Total Persistence Across Layers"
    )
    persist_path = output_dir / "layer_persistence.png"
    fig.savefig(persist_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    generated_files.append(persist_path)
    print(f"  Saved: {persist_path}")

    # 3. Betti curves
    fig = plot_betti_curves(tda_summaries)
    betti_path = output_dir / "betti_curves.png"
    fig.savefig(betti_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    generated_files.append(betti_path)
    print(f"  Saved: {betti_path}")

    # 4. Persistence diagrams (if provided)
    if diagrams:
        for layer_name, layer_diagrams in diagrams.items():
            fig = plot_persistence_diagram(
                layer_diagrams,
                title=f"Persistence Diagram: {layer_name}"
            )
            diagram_path = output_dir / f"persistence_diagram_{layer_name}.png"
            fig.savefig(diagram_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append(diagram_path)
            print(f"  Saved: {diagram_path}")

    return generated_files


# =============================================================================
# Geometry Visualization Functions
# =============================================================================

def plot_geometry_evolution(
    geometry_data: List[Dict[str, Any]],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot the evolution of geometry metrics across layers.

    Shows intrinsic dimensionality, hubness, and distance statistics
    to provide intuition for how representations change through the network.

    Args:
        geometry_data: List of geometry result dicts (one per layer)
        model_name: Name of the model for title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    layers = [g['layer'] for g in geometry_data]
    x = np.arange(len(layers))

    # Extract metrics
    mle_dims = [g.get('mle_intrinsic_dim', 0) for g in geometry_data]
    local_pca_dims = [g.get('local_pca_dim', 0) for g in geometry_data]
    hubness = [g.get('hubness', 0) for g in geometry_data]
    dist_means = [g.get('dist_mean', 0) for g in geometry_data]
    dist_stds = [g.get('dist_std', 0) for g in geometry_data]
    sparsity = [g.get('sparsity', 0) * 100 for g in geometry_data]  # Convert to %
    n_dims = [g.get('n_dims', 0) for g in geometry_data]

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))

    # 1. Intrinsic Dimensionality
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x, mle_dims, color='#2E86AB', edgecolor='black', alpha=0.8, label='MLE')
    ax1.plot(x, local_pca_dims, 'o-', color='#A23B72', linewidth=2, markersize=8,
             label='Local PCA', markeredgecolor='black')
    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Intrinsic Dimension', fontsize=11)
    ax1.set_title('Intrinsic Dimensionality\n(Lower = More Compressed)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Add trend annotation
    if len(mle_dims) > 1:
        dim_change = ((mle_dims[-1] - mle_dims[0]) / mle_dims[0]) * 100 if mle_dims[0] > 0 else 0
        trend_text = f"{dim_change:+.0f}% change" if dim_change != 0 else "No change"
        ax1.annotate(trend_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='right', va='top',
                    color='green' if dim_change < 0 else 'red',
                    fontweight='bold')

    # 2. Hubness Score
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = ['#E8505B' if h > 1 else '#34BE82' for h in hubness]
    bars = ax2.bar(x, hubness, color=bar_colors, edgecolor='black', alpha=0.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal (=1)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Hubness (k-occurrence skewness)', fontsize=11)
    ax2.set_title('Hubness Score\n(>1 = Hub Points Exist)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Distance Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.errorbar(x, dist_means, yerr=dist_stds, fmt='o-', color='#5C4B99',
                 linewidth=2, markersize=8, capsize=5, capthick=2,
                 markeredgecolor='black', label='Mean ± Std')
    ax3.fill_between(x, np.array(dist_means) - np.array(dist_stds),
                     np.array(dist_means) + np.array(dist_stds),
                     alpha=0.2, color='#5C4B99')
    ax3.set_xlabel('Layer', fontsize=11)
    ax3.set_ylabel('k-NN Distance', fontsize=11)
    ax3.set_title('Distance Distribution\n(Spread of Points)', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Dimension vs Intrinsic Dim (compression ratio)
    ax4 = fig.add_subplot(gs[1, 0])
    compression = [n / m if m > 0 else 0 for n, m in zip(n_dims, mle_dims)]
    ax4.bar(x, compression, color='#F18F01', edgecolor='black', alpha=0.8)
    ax4.set_xlabel('Layer', fontsize=11)
    ax4.set_ylabel('Compression Ratio (Ambient / Intrinsic)', fontsize=11)
    ax4.set_title('Compression Ratio\n(Higher = More Redundancy)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Sparsity (if applicable)
    ax5 = fig.add_subplot(gs[1, 1])
    if max(sparsity) > 0:
        ax5.bar(x, sparsity, color='#44BBA4', edgecolor='black', alpha=0.8)
        ax5.set_ylabel('Sparsity (%)', fontsize=11)
        ax5.set_title('Activation Sparsity\n(% Zero Values)', fontweight='bold')
    else:
        # If no sparsity, show ambient dimension
        ax5.bar(x, n_dims, color='#6B4C9A', edgecolor='black', alpha=0.8)
        ax5.set_ylabel('Dimensions', fontsize=11)
        ax5.set_title('Ambient Dimensionality', fontweight='bold')
    ax5.set_xlabel('Layer', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    # 6. Summary metrics text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Compute summary statistics
    initial_dim = mle_dims[0] if mle_dims else 0
    final_dim = mle_dims[-1] if mle_dims else 0
    initial_hubness = hubness[0] if hubness else 0
    final_hubness = hubness[-1] if hubness else 0

    summary_text = f"""
    Summary: {model_name}
    ─────────────────────────

    Intrinsic Dimension:
      • Initial: {initial_dim:.1f}
      • Final:   {final_dim:.1f}
      • Change:  {((final_dim - initial_dim) / initial_dim * 100) if initial_dim > 0 else 0:+.1f}%

    Hubness:
      • Initial: {initial_hubness:.2f}
      • Final:   {final_hubness:.2f}
      • Status:  {'✓ Normalized' if final_hubness <= 1.2 else '⚠ High hubness'}

    Interpretation:
      {'Representations compressed' if final_dim < initial_dim else 'Dim maintained'}
      {'Uniform k-NN structure' if final_hubness < 1.2 else 'Some hub points remain'}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    fig.suptitle(f'Geometry Evolution: {model_name}', fontsize=14, fontweight='bold', y=1.02)

    return fig


def plot_combined_layer_analysis(
    geometry_data: List[Dict[str, Any]],
    tda_summaries: Dict[str, Dict[str, float]],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive figure combining geometry and TDA metrics.

    This provides full layer-wise intuition for how the neural network
    transforms data geometry and topology.

    Args:
        geometry_data: List of geometry result dicts (one per layer)
        tda_summaries: Dictionary mapping layer names to TDA summary dicts
        model_name: Name of the model for title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    layers = [g['layer'] for g in geometry_data]
    x = np.arange(len(layers))

    # Extract geometry metrics
    mle_dims = [g.get('mle_intrinsic_dim', 0) for g in geometry_data]
    hubness = [g.get('hubness', 0) for g in geometry_data]

    # Extract TDA metrics (match layer order)
    h0_persist = [tda_summaries.get(layer, {}).get('H0_total_persistence', 0) for layer in layers]
    h1_persist = [tda_summaries.get(layer, {}).get('H1_total_persistence', 0) for layer in layers]
    h1_counts = [tda_summaries.get(layer, {}).get('H1_count', 0) for layer in layers]

    # Row 1: Key metrics overview
    # 1a. Intrinsic Dimension
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, mle_dims, color=plt.cm.Blues(0.6), edgecolor='black', alpha=0.9)
    ax1.set_ylabel('MLE Intrinsic Dim', fontsize=11)
    ax1.set_title('Dimensionality\n(representation complexity)', fontweight='bold', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    # Color gradient based on value
    norm = plt.Normalize(min(mle_dims), max(mle_dims))
    for bar, val in zip(bars, mle_dims):
        bar.set_facecolor(plt.cm.Blues(norm(val)))

    # 1b. Hubness
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#E8505B' if h > 1.5 else '#FFA500' if h > 1 else '#34BE82' for h in hubness]
    ax2.bar(x, hubness, color=colors, edgecolor='black', alpha=0.9)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Hubness Score', fontsize=11)
    ax2.set_title('Hubness\n(k-NN uniformity)', fontweight='bold', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 1c. H1 Count (loops)
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(x, h1_counts, color=plt.cm.Oranges(0.6), edgecolor='black', alpha=0.9)
    ax3.set_ylabel('H1 Count (loops)', fontsize=11)
    ax3.set_title('Topological Loops\n(cyclic structures)', fontweight='bold', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    norm = plt.Normalize(min(h1_counts), max(h1_counts))
    for bar, val in zip(bars, h1_counts):
        bar.set_facecolor(plt.cm.Oranges(norm(val)))

    # Row 2: Persistence metrics
    # 2a. H0 Total Persistence
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x, h0_persist, color='#1f77b4', edgecolor='black', alpha=0.8)
    ax4.set_ylabel('H0 Total Persistence', fontsize=11)
    ax4.set_title('Component Spread\n(cluster separation)', fontweight='bold', fontsize=11)
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    # Mark peak
    peak_idx = np.argmax(h0_persist)
    ax4.annotate('Peak', xy=(peak_idx, h0_persist[peak_idx]),
                xytext=(peak_idx, h0_persist[peak_idx] * 1.1),
                fontsize=9, ha='center', color='red', fontweight='bold')

    # 2b. H1 Total Persistence
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(x, h1_persist, color='#ff7f0e', edgecolor='black', alpha=0.8)
    ax5.set_ylabel('H1 Total Persistence', fontsize=11)
    ax5.set_title('Loop Strength\n(cycle persistence)', fontweight='bold', fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(layers, rotation=45, ha='right', fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    peak_idx = np.argmax(h1_persist)
    ax5.annotate('Peak', xy=(peak_idx, h1_persist[peak_idx]),
                xytext=(peak_idx, h1_persist[peak_idx] * 1.1),
                fontsize=9, ha='center', color='red', fontweight='bold')

    # 2c. Dim vs H1 scatter (relationship)
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(mle_dims, h1_counts, c=x, cmap='viridis',
                         s=100, edgecolors='black', linewidths=1)
    ax6.set_xlabel('Intrinsic Dimension', fontsize=11)
    ax6.set_ylabel('H1 Count', fontsize=11)
    ax6.set_title('Dim vs Topology\n(layer progression)', fontweight='bold', fontsize=11)
    ax6.grid(True, alpha=0.3)
    # Add layer labels
    for i, layer in enumerate(layers):
        ax6.annotate(layer.replace('layer_', 'L'), (mle_dims[i], h1_counts[i]),
                    fontsize=8, ha='left', va='bottom')

    # Row 3: Evolution plots
    # 3a-b. Combined evolution
    ax7 = fig.add_subplot(gs[2, 0:2])

    # Normalize for comparison
    mle_norm = np.array(mle_dims) / max(mle_dims) if max(mle_dims) > 0 else mle_dims
    hub_norm = np.array(hubness) / max(hubness) if max(hubness) > 0 else hubness
    h0_norm = np.array(h0_persist) / max(h0_persist) if max(h0_persist) > 0 else h0_persist
    h1_norm = np.array(h1_persist) / max(h1_persist) if max(h1_persist) > 0 else h1_persist

    ax7.plot(x, mle_norm, 'o-', linewidth=2.5, markersize=10, label='Intrinsic Dim',
             color='#2E86AB', markeredgecolor='black')
    ax7.plot(x, hub_norm, 's-', linewidth=2.5, markersize=10, label='Hubness',
             color='#E8505B', markeredgecolor='black')
    ax7.plot(x, h0_norm, '^-', linewidth=2.5, markersize=10, label='H0 Persistence',
             color='#1f77b4', markeredgecolor='black')
    ax7.plot(x, h1_norm, 'D-', linewidth=2.5, markersize=10, label='H1 Persistence',
             color='#ff7f0e', markeredgecolor='black')

    ax7.set_xlabel('Layer', fontsize=12)
    ax7.set_ylabel('Normalized Value', fontsize=12)
    ax7.set_title('Layer-wise Evolution (Normalized)', fontweight='bold', fontsize=12)
    ax7.set_xticks(x)
    ax7.set_xticklabels(layers, rotation=45, ha='right', fontsize=10)
    ax7.legend(loc='upper right', fontsize=10, ncol=2)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(-0.05, 1.15)

    # 3c. Interpretation panel
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Generate interpretation
    dim_trend = "decreasing" if mle_dims[-1] < mle_dims[0] else "stable/increasing"
    hub_trend = "normalizing" if hubness[-1] < hubness[0] else "increasing"
    h0_peak = layers[np.argmax(h0_persist)]
    h1_peak = layers[np.argmax(h1_persist)]

    interpretation = f"""
    Geometric Interpretation
    ════════════════════════

    Dimensionality: {dim_trend}
    • Start: {mle_dims[0]:.1f} → End: {mle_dims[-1]:.1f}
    • Network {'compresses' if dim_trend == 'decreasing' else 'preserves'} information

    Hubness: {hub_trend}
    • Start: {hubness[0]:.2f} → End: {hubness[-1]:.2f}
    • k-NN structure {'becomes uniform' if hub_trend == 'normalizing' else 'has hub points'}

    Topological Complexity:
    • H0 peaks at {h0_peak} (max cluster spread)
    • H1 peaks at {h1_peak} (max cyclic structure)
    • Final H1: {h1_counts[-1]:.0f} loops

    Overall: {'Compression + Simplification' if dim_trend == 'decreasing' and h1_counts[-1] < h1_counts[0] else 'Complex transformations'}
    """

    ax8.text(0.05, 0.95, interpretation, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff',
                      edgecolor='#4682b4', alpha=0.95))

    fig.suptitle(f'Combined Geometry + TDA Analysis: {model_name}',
                 fontsize=16, fontweight='bold', y=1.01)

    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Compare geometry and TDA metrics across multiple models.

    Args:
        results: Dictionary mapping model names to result dicts
                 Each result should have 'geometry' and 'tda' keys
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.6

    # Extract final layer metrics for each model
    final_dims = []
    final_hubness = []
    final_h1_counts = []
    final_h0_persist = []
    final_h1_persist = []
    param_counts = []

    for model_name, result in results.items():
        # Get final layer geometry
        geometry = result.get('geometry', [])
        if geometry:
            final_geom = geometry[-1]
            final_dims.append(final_geom.get('mle_intrinsic_dim', 0))
            final_hubness.append(final_geom.get('hubness', 0))
        else:
            final_dims.append(0)
            final_hubness.append(0)

        # Get final layer TDA
        tda = result.get('tda', {})
        final_layer = list(tda.keys())[-1] if tda else None
        if final_layer:
            final_h1_counts.append(tda[final_layer].get('H1_count', 0))
            final_h0_persist.append(tda[final_layer].get('H0_total_persistence', 0))
            final_h1_persist.append(tda[final_layer].get('H1_total_persistence', 0))
        else:
            final_h1_counts.append(0)
            final_h0_persist.append(0)
            final_h1_persist.append(0)

        # Get parameter count
        model_info = result.get('model_info', {})
        param_counts.append(model_info.get('num_params_millions', 0))

    # Plot 1: Intrinsic Dimension
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(models)))
    axes[0, 0].bar(x, final_dims, width, color=colors, edgecolor='black')
    axes[0, 0].set_ylabel('MLE Intrinsic Dim')
    axes[0, 0].set_title('Final Layer Dimension', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=30, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Plot 2: Hubness
    hub_colors = ['#34BE82' if h <= 1.2 else '#E8505B' for h in final_hubness]
    axes[0, 1].bar(x, final_hubness, width, color=hub_colors, edgecolor='black')
    axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
    axes[0, 1].set_ylabel('Hubness')
    axes[0, 1].set_title('Final Layer Hubness', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=30, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Plot 3: H1 Count
    axes[0, 2].bar(x, final_h1_counts, width, color='#ff7f0e', edgecolor='black')
    axes[0, 2].set_ylabel('H1 Count')
    axes[0, 2].set_title('Final Layer Loops', fontweight='bold')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models, rotation=30, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)

    # Plot 4: H0 Persistence
    axes[1, 0].bar(x, final_h0_persist, width, color='#1f77b4', edgecolor='black')
    axes[1, 0].set_ylabel('H0 Total Persistence')
    axes[1, 0].set_title('Component Spread', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=30, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 5: H1 Persistence
    axes[1, 1].bar(x, final_h1_persist, width, color='#ff7f0e', edgecolor='black')
    axes[1, 1].set_ylabel('H1 Total Persistence')
    axes[1, 1].set_title('Loop Strength', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=30, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Plot 6: Params vs Dim scatter
    if any(param_counts):
        scatter = axes[1, 2].scatter(param_counts, final_dims, c=final_h1_counts,
                                     s=200, cmap='Oranges', edgecolors='black',
                                     linewidths=1.5)
        axes[1, 2].set_xlabel('Parameters (M)')
        axes[1, 2].set_ylabel('Intrinsic Dim')
        axes[1, 2].set_title('Size vs Complexity', fontweight='bold')
        for i, model in enumerate(models):
            axes[1, 2].annotate(model, (param_counts[i], final_dims[i]),
                               fontsize=9, ha='left', va='bottom')
        plt.colorbar(scatter, ax=axes[1, 2], label='H1 Count')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')

    fig.suptitle('Model Comparison: Geometry & TDA', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def generate_ablation_visualizations(
    results: Dict[str, Any],
    output_dir: Path,
) -> List[Path]:
    """
    Generate all visualizations for an ablation study.

    Args:
        results: Full ablation results dictionary
        output_dir: Directory to save plots

    Returns:
        List of paths to generated plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    model_results = results.get('results', {})

    print(f"\nGenerating visualizations...")

    # 1. Per-model visualizations
    for model_name, result in model_results.items():
        if 'error' in result:
            continue

        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        geometry_data = result.get('geometry', [])
        tda_summaries = result.get('tda', {})

        if geometry_data:
            # Geometry evolution
            fig = plot_geometry_evolution(geometry_data, model_name)
            path = model_dir / "geometry_evolution.png"
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append(path)
            print(f"  Saved: {path}")

        if tda_summaries:
            # TDA summary
            fig = plot_tda_summary(tda_summaries, model_name)
            path = model_dir / "tda_summary.png"
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append(path)
            print(f"  Saved: {path}")

        if geometry_data and tda_summaries:
            # Combined analysis
            fig = plot_combined_layer_analysis(geometry_data, tda_summaries, model_name)
            path = model_dir / "combined_analysis.png"
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            generated_files.append(path)
            print(f"  Saved: {path}")

    # 2. Cross-model comparison (if multiple models)
    if len(model_results) > 1:
        fig = plot_model_comparison(model_results)
        path = output_dir / "model_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        generated_files.append(path)
        print(f"  Saved: {path}")

    print(f"\nGenerated {len(generated_files)} visualization files")
    return generated_files
