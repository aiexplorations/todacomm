"""
Meta-analysis module for comparing TDA results across multiple models.

Generates comparative reports highlighting cross-model patterns in H0/H1 features,
with embedded visualizations and links to data files.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def find_experiment_dirs(base_dir: str = "experiments", model_names: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Find experiment directories for specified models.

    Args:
        base_dir: Base experiments directory
        model_names: List of model names to find (e.g., ['gpt2', 'bert'])

    Returns:
        Dict mapping model names to their most recent experiment directories
    """
    experiments_path = Path(base_dir)
    if not experiments_path.exists():
        return {}

    model_dirs = {}

    for exp_dir in sorted(experiments_path.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        # Extract model name from directory (format: modelname_tda_TIMESTAMP)
        dir_name = exp_dir.name
        if "_tda_" not in dir_name:
            continue

        model_key = dir_name.split("_tda_")[0]

        # Skip if not in requested models (if specified)
        if model_names and model_key not in model_names:
            continue

        # Keep most recent experiment per model
        if model_key not in model_dirs:
            model_dirs[model_key] = exp_dir

    return model_dirs


def load_tda_data(exp_dir: Path) -> Optional[Dict]:
    """Load TDA summaries, metrics, and artifact paths from an experiment directory."""
    run_dir = exp_dir / "runs" / "run_0"
    tda_path = run_dir / "tda_summaries.json"
    metrics_path = run_dir / "metrics.json"
    viz_dir = run_dir / "visualizations"
    interpretation_path = run_dir / "tda_interpretation.md"

    if not tda_path.exists():
        return None

    try:
        with open(tda_path) as f:
            tda_data = json.load(f)

        metrics = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        # Collect visualization paths
        visualizations = {}
        if viz_dir.exists():
            for viz_file in viz_dir.glob("*.png"):
                viz_name = viz_file.stem
                visualizations[viz_name] = str(viz_file)

        # Collect data file paths
        data_files = {
            "tda_summaries": str(tda_path),
            "metrics": str(metrics_path) if metrics_path.exists() else None,
            "interpretation": str(interpretation_path) if interpretation_path.exists() else None,
        }

        # Check for artifacts
        artifacts_dir = exp_dir / "artifacts"
        if artifacts_dir.exists():
            for artifact in artifacts_dir.glob("*.csv"):
                data_files[artifact.stem] = str(artifact)

        return {
            "tda": tda_data,
            "metrics": metrics,
            "experiment_dir": str(exp_dir),
            "visualizations": visualizations,
            "data_files": data_files
        }
    except Exception as e:
        print(f"Error loading data from {exp_dir}: {e}")
        return None


def compute_model_summary(tda_data: Dict) -> Dict:
    """Compute summary statistics for a model's TDA data."""
    tda = tda_data["tda"]
    metrics = tda_data["metrics"]

    # Get embedding and peak H0
    embedding_h0 = tda.get("embedding", {}).get("H0_total_persistence", 0)

    # Find peak H0 layer
    peak_h0 = 0
    peak_h0_layer = None
    for layer, values in tda.items():
        if layer == "final":
            continue
        h0 = values.get("H0_total_persistence", 0)
        if h0 > peak_h0:
            peak_h0 = h0
            peak_h0_layer = layer

    # Calculate expansion ratio
    expansion_ratio = peak_h0 / embedding_h0 if embedding_h0 > 0 else 0

    # Find max H1
    max_h1 = 0
    max_h1_layer = None
    for layer, values in tda.items():
        h1 = values.get("H1_total_persistence", 0)
        if h1 > max_h1:
            max_h1 = h1
            max_h1_layer = layer

    # Count layers with H1 features
    h1_layers = sum(1 for v in tda.values() if v.get("H1_count", 0) > 0)

    return {
        "embedding_h0": embedding_h0,
        "peak_h0": peak_h0,
        "peak_h0_layer": peak_h0_layer,
        "expansion_ratio": expansion_ratio,
        "max_h1": max_h1,
        "max_h1_layer": max_h1_layer,
        "h1_layers_count": h1_layers,
        "total_layers": len(tda),
        "perplexity": metrics.get("perplexity"),
        "accuracy": metrics.get("accuracy")
    }


def make_relative_path(from_file: Path, to_file: Path) -> str:
    """Create a relative path from one file to another."""
    try:
        # Get the directory containing the source file
        from_dir = from_file.parent.resolve()
        to_path = Path(to_file).resolve()

        # Calculate relative path
        rel_path = os.path.relpath(to_path, from_dir)
        return rel_path
    except ValueError:
        # Different drives on Windows, return absolute
        return str(to_file)


def generate_meta_analysis(
    model_names: List[str],
    output_path: Optional[str] = None,
    base_dir: str = "experiments"
) -> str:
    """
    Generate a meta-analysis report comparing multiple models.

    Args:
        model_names: List of model names to compare
        output_path: Path to save the report (optional)
        base_dir: Base experiments directory

    Returns:
        The generated markdown report
    """
    # Find experiment directories
    model_dirs = find_experiment_dirs(base_dir, model_names)

    if not model_dirs:
        return "# Meta-Analysis\n\nNo experiment data found for the specified models."

    # Load data for each model
    model_data = {}
    for model_name, exp_dir in model_dirs.items():
        data = load_tda_data(exp_dir)
        if data:
            data["summary"] = compute_model_summary(data)
            model_data[model_name] = data

    if not model_data:
        return "# Meta-Analysis\n\nCould not load TDA data for any models."

    # Convert paths to be relative to output file if specified
    output_file = Path(output_path) if output_path else None
    if output_file:
        for model_name, data in model_data.items():
            # Convert visualization paths
            if "visualizations" in data:
                for viz_name, viz_path in data["visualizations"].items():
                    data["visualizations"][viz_name] = make_relative_path(output_file, viz_path)

            # Convert data file paths
            if "data_files" in data:
                for file_name, file_path in data["data_files"].items():
                    if file_path:
                        data["data_files"][file_name] = make_relative_path(output_file, file_path)

            # Convert experiment dir
            data["experiment_dir"] = make_relative_path(output_file, data["experiment_dir"])

    # Generate report
    report = generate_report_markdown(model_data)

    # Save if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\n✓ Meta-analysis saved to: {output_path}")

    return report


def generate_report_markdown(model_data: Dict, output_dir: Optional[Path] = None) -> str:
    """Generate the markdown report from model data with embedded visualizations."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Meta-Analysis: TDA Comparison Across Models",
        "",
        f"*Generated: {timestamp}*",
        "",
        "## Summary",
        "",
        f"Compared **{len(model_data)}** models using topological data analysis.",
        "",
        "## Models Analyzed",
        "",
        "| Model | Perplexity | Accuracy | Layers |",
        "|-------|------------|----------|--------|",
    ]

    # Sort by perplexity (best first)
    sorted_models = sorted(
        model_data.items(),
        key=lambda x: x[1]["summary"].get("perplexity") or float("inf")
    )

    for model_name, data in sorted_models:
        summary = data["summary"]
        ppl = summary.get("perplexity")
        acc = summary.get("accuracy")
        ppl_str = f"{ppl:.2f}" if ppl else "N/A"
        acc_str = f"{acc:.3f}" if acc else "N/A"
        lines.append(f"| {model_name} | {ppl_str} | {acc_str} | {summary['total_layers']} |")

    # H0 Analysis
    lines.extend([
        "",
        "## H0 (Cluster Structure) Analysis",
        "",
        "| Model | Embedding H0 | Peak H0 | Peak Layer | Expansion |",
        "|-------|--------------|---------|------------|-----------|",
    ])

    # Sort by expansion ratio
    sorted_by_expansion = sorted(
        model_data.items(),
        key=lambda x: x[1]["summary"]["expansion_ratio"],
        reverse=True
    )

    for model_name, data in sorted_by_expansion:
        s = data["summary"]
        lines.append(
            f"| {model_name} | {s['embedding_h0']:.1f} | {s['peak_h0']:.0f} | "
            f"{s['peak_h0_layer']} | **{s['expansion_ratio']:.0f}x** |"
        )

    # H1 Analysis
    lines.extend([
        "",
        "## H1 (Cyclic Structure) Analysis",
        "",
        "| Model | Max H1 | H1 Layer | Layers with H1 |",
        "|-------|--------|----------|----------------|",
    ])

    sorted_by_h1 = sorted(
        model_data.items(),
        key=lambda x: x[1]["summary"]["max_h1"],
        reverse=True
    )

    for model_name, data in sorted_by_h1:
        s = data["summary"]
        lines.append(
            f"| {model_name} | {s['max_h1']:.2f} | {s['max_h1_layer']} | "
            f"{s['h1_layers_count']}/{s['total_layers']} |"
        )

    # Key Findings
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Best perplexity model
    best_ppl_model = sorted_models[0][0]
    best_ppl = sorted_models[0][1]["summary"].get("perplexity")
    if best_ppl:
        lines.append(f"1. **Best perplexity**: {best_ppl_model} ({best_ppl:.2f})")

    # Highest expansion model
    highest_exp_model = sorted_by_expansion[0][0]
    highest_exp = sorted_by_expansion[0][1]["summary"]["expansion_ratio"]
    lines.append(f"2. **Highest H0 expansion**: {highest_exp_model} ({highest_exp:.0f}x)")

    # Lowest expansion model
    lowest_exp_model = sorted_by_expansion[-1][0]
    lowest_exp = sorted_by_expansion[-1][1]["summary"]["expansion_ratio"]
    lines.append(f"3. **Most constrained H0**: {lowest_exp_model} ({lowest_exp:.0f}x)")

    # Strongest H1
    strongest_h1_model = sorted_by_h1[0][0]
    strongest_h1 = sorted_by_h1[0][1]["summary"]["max_h1"]
    lines.append(f"4. **Strongest cyclic structure**: {strongest_h1_model} (H1={strongest_h1:.2f})")

    # Correlation analysis
    lines.extend([
        "",
        "## Expansion vs Performance",
        "",
        "```",
        "Model                 Expansion   Perplexity",
        "-" * 45,
    ])

    for model_name, data in sorted_by_expansion:
        s = data["summary"]
        ppl = s.get("perplexity")
        ppl_str = f"{ppl:.2f}" if ppl else "N/A"
        lines.append(f"{model_name:<20}  {s['expansion_ratio']:>6.0f}x    {ppl_str:>8}")

    lines.append("```")

    # Per-Model Visualizations Section
    lines.extend([
        "",
        "---",
        "",
        "## Per-Model Visualizations",
        "",
    ])

    for model_name, data in sorted_models:
        lines.append(f"### {model_name}")
        lines.append("")

        # Add visualizations if available
        viz = data.get("visualizations", {})
        exp_dir = data["experiment_dir"]

        if viz:
            # Create relative paths for visualizations
            if "tda_summary" in viz:
                lines.append(f"**TDA Summary**")
                lines.append(f"![TDA Summary]({viz['tda_summary']})")
                lines.append("")

            if "layer_persistence" in viz:
                lines.append(f"**Layer Persistence (H0/H1)**")
                lines.append(f"![Layer Persistence]({viz['layer_persistence']})")
                lines.append("")

            if "betti_curves" in viz:
                lines.append(f"**Betti Curves**")
                lines.append(f"![Betti Curves]({viz['betti_curves']})")
                lines.append("")
        else:
            lines.append("*No visualizations available*")
            lines.append("")

        # Link to data files
        data_files = data.get("data_files", {})
        if data_files:
            lines.append("**Data Files:**")
            if data_files.get("interpretation"):
                lines.append(f"- [TDA Interpretation]({data_files['interpretation']})")
            if data_files.get("tda_summaries"):
                lines.append(f"- [TDA Summaries (JSON)]({data_files['tda_summaries']})")
            if data_files.get("metrics"):
                lines.append(f"- [Metrics (JSON)]({data_files['metrics']})")
            if data_files.get("experiment_data"):
                lines.append(f"- [Experiment Data (CSV)]({data_files['experiment_data']})")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Methodology
    lines.extend([
        "## Methodology",
        "",
        "- Persistent homology computed on PCA-reduced activation samples",
        "- H0 tracks connected components (cluster spread)",
        "- H1 tracks 1-dimensional loops (cyclic structure)",
        "- Expansion ratio = Peak H0 / Embedding H0",
        "",
        "## Source Experiments",
        "",
    ])

    for model_name, data in sorted_models:
        lines.append(f"- **{model_name}**: `{data['experiment_dir']}`")

    return "\n".join(lines)


def run_meta_analysis_cli(model_names: List[str], output_name: Optional[str] = None) -> str:
    """
    CLI entry point for meta-analysis.

    Args:
        model_names: List of model names that were analyzed
        output_name: Optional name for the output file

    Returns:
        Path to the generated report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_name:
        output_path = f"experiments/{output_name}_meta_analysis.md"
    else:
        output_path = f"experiments/meta_analysis_{timestamp}.md"

    generate_meta_analysis(model_names, output_path)

    return output_path
