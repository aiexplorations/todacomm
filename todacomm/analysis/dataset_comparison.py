"""
Dataset comparison module for analyzing TDA results across different datasets.

Generates reports showing how a model's topological structure differs
when processing different types of data (e.g., WikiText-2 vs SQuAD).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def make_relative_path(from_file: Path, to_file: Path) -> str:
    """Create a relative path from one file to another."""
    try:
        from_dir = from_file.parent.resolve()
        to_path = Path(to_file).resolve()
        return os.path.relpath(to_path, from_dir)
    except ValueError:
        return str(to_file)


def find_dataset_experiments(
    model_name: str,
    datasets: List[str],
    base_dir: str = "experiments"
) -> Dict[str, Path]:
    """
    Find experiment directories for a model across different datasets.

    Args:
        model_name: Model key (e.g., 'gpt2')
        datasets: List of dataset names to find
        base_dir: Base experiments directory

    Returns:
        Dict mapping dataset names to experiment directories
    """
    experiments_path = Path(base_dir)
    if not experiments_path.exists():
        return {}

    dataset_dirs = {}

    # Look for experiments matching model_dataset_tda pattern
    for exp_dir in sorted(experiments_path.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        dir_name = exp_dir.name

        # Check if this experiment is for our model
        for dataset in datasets:
            pattern = f"{model_name}_{dataset}_tda"
            if dir_name.startswith(pattern):
                if dataset not in dataset_dirs:
                    dataset_dirs[dataset] = exp_dir
                break

    return dataset_dirs


def load_dataset_tda_data(exp_dir: Path) -> Optional[Dict]:
    """Load TDA summaries, metrics, and artifact paths from an experiment directory."""
    run_dir = exp_dir / "runs" / "run_0"
    tda_path = run_dir / "tda_summaries.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.json"
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

        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

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
            "config": config,
            "experiment_dir": str(exp_dir),
            "visualizations": visualizations,
            "data_files": data_files
        }
    except Exception as e:
        print(f"Error loading data from {exp_dir}: {e}")
        return None


def compute_dataset_summary(tda_data: Dict) -> Dict:
    """Compute summary statistics for a dataset's TDA data."""
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

    expansion_ratio = peak_h0 / embedding_h0 if embedding_h0 > 0 else 0

    # Find max H1
    max_h1 = 0
    max_h1_layer = None
    total_h1 = 0
    for layer, values in tda.items():
        h1 = values.get("H1_total_persistence", 0)
        total_h1 += h1
        if h1 > max_h1:
            max_h1 = h1
            max_h1_layer = layer

    # Count layers with H1 features
    h1_layers = sum(1 for v in tda.values() if v.get("H1_count", 0) > 0)

    # Average H0 across layers
    h0_values = [v.get("H0_total_persistence", 0) for v in tda.values()]
    avg_h0 = sum(h0_values) / len(h0_values) if h0_values else 0

    return {
        "embedding_h0": embedding_h0,
        "peak_h0": peak_h0,
        "peak_h0_layer": peak_h0_layer,
        "avg_h0": avg_h0,
        "expansion_ratio": expansion_ratio,
        "max_h1": max_h1,
        "max_h1_layer": max_h1_layer,
        "total_h1": total_h1,
        "h1_layers_count": h1_layers,
        "total_layers": len(tda),
        "perplexity": metrics.get("perplexity"),
        "accuracy": metrics.get("accuracy")
    }


def generate_dataset_comparison(
    model_name: str,
    datasets: List[str],
    output_path: Optional[str] = None,
    base_dir: str = "experiments"
) -> str:
    """
    Generate a comparison report for a model across multiple datasets.

    Args:
        model_name: Model key (e.g., 'gpt2')
        datasets: List of dataset names
        output_path: Path to save the report
        base_dir: Base experiments directory

    Returns:
        The generated markdown report
    """
    # Find experiment directories
    dataset_dirs = find_dataset_experiments(model_name, datasets, base_dir)

    if not dataset_dirs:
        return f"# Dataset Comparison\n\nNo experiment data found for {model_name} on {datasets}."

    # Load data for each dataset
    dataset_data = {}
    for dataset_name, exp_dir in dataset_dirs.items():
        data = load_dataset_tda_data(exp_dir)
        if data:
            data["summary"] = compute_dataset_summary(data)
            data["dataset"] = dataset_name
            dataset_data[dataset_name] = data

    if not dataset_data:
        return "# Dataset Comparison\n\nCould not load TDA data for any datasets."

    # Convert paths to be relative to output file if specified
    output_file = Path(output_path) if output_path else None
    if output_file:
        for dataset_name, data in dataset_data.items():
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
    report = generate_dataset_report_markdown(model_name, dataset_data)

    # Save if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)
        print(f"\n✓ Dataset comparison saved to: {output_path}")

    return report


def generate_dataset_report_markdown(model_name: str, dataset_data: Dict) -> str:
    """Generate the markdown report from dataset comparison data."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Dataset Comparison: {model_name}",
        "",
        f"*Generated: {timestamp}*",
        "",
        "## Summary",
        "",
        f"Compared TDA features for **{model_name}** across **{len(dataset_data)}** datasets.",
        "",
        "## Performance by Dataset",
        "",
        "| Dataset | Perplexity | Accuracy | Layers |",
        "|---------|------------|----------|--------|",
    ]

    for dataset_name, data in sorted(dataset_data.items()):
        summary = data["summary"]
        ppl = summary.get("perplexity")
        acc = summary.get("accuracy")
        ppl_str = f"{ppl:.2f}" if ppl else "N/A"
        acc_str = f"{acc:.3f}" if acc else "N/A"
        lines.append(f"| {dataset_name} | {ppl_str} | {acc_str} | {summary['total_layers']} |")

    # H0 Comparison
    lines.extend([
        "",
        "## H0 (Cluster Structure) by Dataset",
        "",
        "| Dataset | Embedding H0 | Peak H0 | Avg H0 | Expansion |",
        "|---------|--------------|---------|--------|-----------|",
    ])

    for dataset_name, data in sorted(dataset_data.items()):
        s = data["summary"]
        lines.append(
            f"| {dataset_name} | {s['embedding_h0']:.1f} | {s['peak_h0']:.0f} | "
            f"{s['avg_h0']:.1f} | {s['expansion_ratio']:.0f}x |"
        )

    # H1 Comparison
    lines.extend([
        "",
        "## H1 (Cyclic Structure) by Dataset",
        "",
        "| Dataset | Max H1 | Total H1 | H1 Layer | Layers with H1 |",
        "|---------|--------|----------|----------|----------------|",
    ])

    for dataset_name, data in sorted(dataset_data.items()):
        s = data["summary"]
        lines.append(
            f"| {dataset_name} | {s['max_h1']:.2f} | {s['total_h1']:.2f} | "
            f"{s['max_h1_layer']} | {s['h1_layers_count']}/{s['total_layers']} |"
        )

    # Key Differences
    lines.extend([
        "",
        "## Key Observations",
        "",
    ])

    datasets = list(dataset_data.keys())
    if len(datasets) >= 2:
        d1, d2 = datasets[0], datasets[1]
        s1, s2 = dataset_data[d1]["summary"], dataset_data[d2]["summary"]

        # Compare expansion ratios
        exp_diff = abs(s1["expansion_ratio"] - s2["expansion_ratio"])
        if exp_diff > 10:
            higher = d1 if s1["expansion_ratio"] > s2["expansion_ratio"] else d2
            lines.append(f"1. **H0 expansion differs by {exp_diff:.0f}x** - {higher} shows more cluster spread")

        # Compare H1
        h1_diff = abs(s1["total_h1"] - s2["total_h1"])
        if h1_diff > 0.5:
            higher = d1 if s1["total_h1"] > s2["total_h1"] else d2
            lines.append(f"2. **Cyclic structure differs** - {higher} has stronger H1 features ({h1_diff:.2f} difference)")

        # Compare performance
        if s1.get("perplexity") and s2.get("perplexity"):
            better = d1 if s1["perplexity"] < s2["perplexity"] else d2
            lines.append(f"3. **Better perplexity on {better}**")

    # Interpretation
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Different datasets may elicit different topological structures because:",
        "- **Domain specificity**: Q&A data (SQuAD) vs general text (WikiText) require different representations",
        "- **Sequence structure**: Question-answer pairs vs continuous prose",
        "- **Vocabulary distribution**: Technical terms vs general language",
        "",
        "---",
        "",
        "## Per-Dataset Visualizations",
        "",
    ])

    for dataset_name, data in sorted(dataset_data.items()):
        lines.append(f"### {dataset_name}")
        lines.append("")

        # Add visualizations if available
        viz = data.get("visualizations", {})

        if viz:
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

    # Source Experiments
    lines.extend([
        "## Source Experiments",
        "",
    ])

    for dataset_name, data in sorted(dataset_data.items()):
        lines.append(f"- **{dataset_name}**: `{data['experiment_dir']}`")

    return "\n".join(lines)


def run_dataset_comparison_cli(
    model_name: str,
    datasets: List[str],
    output_name: Optional[str] = None
) -> str:
    """
    CLI entry point for dataset comparison.

    Args:
        model_name: Model key
        datasets: List of dataset names
        output_name: Optional output filename

    Returns:
        Path to the generated report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_name:
        output_path = f"experiments/{output_name}_dataset_comparison.md"
    else:
        output_path = f"experiments/{model_name}_dataset_comparison_{timestamp}.md"

    generate_dataset_comparison(model_name, datasets, output_path)

    return output_path
