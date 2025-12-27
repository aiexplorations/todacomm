"""
TDA Visualization Functions.

Provides plotting utilities for persistence diagrams, Betti curves,
and layer-wise TDA metric comparisons.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
