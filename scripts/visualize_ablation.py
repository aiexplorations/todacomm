#!/usr/bin/env python
"""
Visualize Ablation Study Results

Generate layer-wise visualizations for geometry and TDA metrics from
existing ablation study results.

Usage:
    python scripts/visualize_ablation.py experiments/transformer_ablation_20251230_134136
    python scripts/visualize_ablation.py experiments/mlp_ablation_20251230_125115
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from todacomm.visualization.tda_plots import (
    generate_ablation_visualizations,
    plot_geometry_evolution,
    plot_tda_summary,
    plot_combined_layer_analysis,
    plot_model_comparison,
)


def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for ablation study results"
    )
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory containing results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for visualizations (default: <experiment_dir>/plots)",
    )

    args = parser.parse_args()

    # Find results file
    experiment_dir = Path(args.experiment_dir)
    results_file = experiment_dir / "results.json"

    if not results_file.exists():
        print(f"ERROR: results.json not found in {experiment_dir}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_file}")
    results = load_results(results_file)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = experiment_dir / "plots"

    print(f"Output directory: {output_dir}")

    # Generate visualizations
    generated_files = generate_ablation_visualizations(results, output_dir)

    print(f"\n{'='*60}")
    print(f"Generated {len(generated_files)} visualization files:")
    for f in generated_files:
        print(f"  - {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
