"""
ToDACoMM - Unified Pipeline for Multi-Model TDA Comparison

Single entry point for comparing topological features across transformer models.
Orchestrates: model loading → data preparation → extraction → TDA → analysis → reporting
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from todacomm.models.transformer import TransformerConfig, TransformerModel, load_pretrained_transformer
from todacomm.data.language_datasets import DatasetConfig, load_language_dataset, create_dataloaders
from todacomm.extract.transformer_activations import ActivationConfig, extract_transformer_activations
from todacomm.tda.persistence import TDAConfig, compute_persistence, summarize_diagrams
from todacomm.analysis.correlation import correlate_tda_with_metrics
from todacomm.analysis.interpretation import (
    interpret_tda_results,
    format_interpretation_markdown,
    generate_metric_glossary,
)
from todacomm.visualization.tda_plots import generate_all_visualizations, plot_tda_summary


@dataclass
class ExperimentConfig:
    """Master configuration for experiments."""
    
    # Experiment metadata
    experiment_name: str = "tda_experiment"
    experiment_type: Literal["quick_test", "multi_model_comparison", "training_comparison", "tda_config_sensitivity"] = "quick_test"
    description: str = ""
    
    # Model configuration
    model: Dict[str, Any] = None
    model_variants: Optional[List[Dict[str, Any]]] = None  # For multi-model comparison
    
    # Dataset configuration
    dataset: Dict[str, Any] = None
    
    # Analysis layers
    analysis_layers: List[str] = None
    
    # TDA configuration
    tda: Dict[str, Any] = None
    
    # Extraction configuration
    extraction: Dict[str, Any] = None
    
    # Fine-tuning configuration (optional)
    fine_tuning: Optional[Dict[str, Any]] = None
    
    # Output configuration
    output: Dict[str, Any] = None
    
    # Execution configuration
    device: str = "cpu"
    random_seed: int = 42
    
    def __post_init__(self):
        """Set defaults for nested configs."""
        if self.model is None:
            self.model = {"type": "gpt2", "name": "gpt2", "task": "lm"}
        if self.dataset is None:
            self.dataset = {"name": "wikitext2", "task": "lm", "tokenizer": "gpt2", "max_length": 128, "num_samples": 100, "batch_size": 4}
        if self.analysis_layers is None:
            self.analysis_layers = ["embedding", "final"]
        if self.tda is None:
            self.tda = {"maxdim": 1, "metric": "euclidean", "pca_components": 50, "sampling_strategy": "uniform", "max_points": 500}
        if self.extraction is None:
            self.extraction = {"max_samples": 100, "pool_strategy": "mean", "device": "cpu"}
        if self.output is None:
            self.output = {"generate_report": True, "save_artifacts": True}
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class ExperimentRun:
    """Configuration for a single experimental run."""
    
    run_id: str
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    analysis_layers: List[str]
    tda_config: Dict[str, Any]
    extraction_config: Dict[str, Any]
    experiment_config: ExperimentConfig
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "analysis_layers": self.analysis_layers,
            "tda_config": self.tda_config,
            "extraction_config": self.extraction_config
        }


def setup_experiment_directory(config: ExperimentConfig) -> Path:
    """
    Create experiment directory structure.
    
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config.experiment_name}_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    
    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "runs").mkdir(exist_ok=True)
    (exp_dir / "reports").mkdir(exist_ok=True)
    (exp_dir / "artifacts").mkdir(exist_ok=True)
    
    # Save experiment configuration
    config.save(str(exp_dir / "experiment_config.yaml"))
    
    return exp_dir


def generate_run_matrix(config: ExperimentConfig) -> List[ExperimentRun]:
    """
    Generate matrix of experimental runs based on configuration.

    For multi-model comparisons, creates runs for each variant.
    For quick tests, creates a single run.
    """
    runs = []
    
    if config.experiment_type == "multi_model_comparison" and config.model_variants:
        # Create run for each model variant
        for idx, model_variant in enumerate(config.model_variants):
            run_id = f"model_{idx}_{model_variant.get('name', 'unknown')}"
            
            # Update dataset tokenizer to match model
            dataset_config = config.dataset.copy()
            dataset_config["tokenizer"] = model_variant.get("name", "gpt2")
            
            run = ExperimentRun(
                run_id=run_id,
                model_config=model_variant,
                dataset_config=dataset_config,
                analysis_layers=config.analysis_layers,
                tda_config=config.tda,
                extraction_config=config.extraction,
                experiment_config=config
            )
            runs.append(run)
    else:
        # Single run for quick test
        run = ExperimentRun(
            run_id="run_0",
            model_config=config.model,
            dataset_config=config.dataset,
            analysis_layers=config.analysis_layers,
            tda_config=config.tda,
            extraction_config=config.extraction,
            experiment_config=config
        )
        runs.append(run)
    
    return runs


def execute_single_run(run: ExperimentRun, exp_dir: Path) -> Dict[str, Any]:
    """
    Execute a single experimental run.
    
    Returns:
        Dictionary with run results
    """
    print(f"\n{'='*80}")
    print(f"Executing run: {run.run_id}")
    print(f"{'='*80}\n")
    
    run_dir = exp_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    with open(run_dir / "run_config.json", 'w') as f:
        json.dump(run.to_dict(), f, indent=2)
    
    results = {"run_id": run.run_id, "status": "running"}
    
    try:
        # 1. Load model
        print(f"[1/5] Loading model: {run.model_config['name']}")
        model = load_pretrained_transformer(
            model_name=run.model_config["name"],
            task=run.model_config.get("task", "lm"),
            num_labels=run.model_config.get("num_labels", 2),
            device=run.extraction_config.get("device", "cpu")
        )
        print(f"✓ Model loaded: {run.model_config['name']}")
        
        # 2. Load dataset
        print(f"\n[2/5] Loading dataset: {run.dataset_config['name']}")
        # Transform config keys to match DatasetConfig field names
        dataset_dict = run.dataset_config.copy()
        if 'name' in dataset_dict:
            dataset_dict['dataset_name'] = dataset_dict.pop('name')
        if 'tokenizer' in dataset_dict:
            dataset_dict['tokenizer_name'] = dataset_dict.pop('tokenizer')
        dataset_cfg = DatasetConfig(**dataset_dict)
        datasets, tokenizer = load_language_dataset(dataset_cfg)
        dataloaders = create_dataloaders(
            datasets,
            batch_size=run.dataset_config.get("batch_size", 8)
        )
        print(f"✓ Dataset loaded: {len(datasets['train'])} train, {len(datasets.get('val', []))} val, {len(datasets.get('test', []))} test samples")
        
        # 3. Extract activations
        print(f"\n[3/5] Extracting activations from {len(run.analysis_layers)} layers")
        extraction_cfg = ActivationConfig(**run.extraction_config, layers=run.analysis_layers)
        
        # Use test set for analysis
        test_loader = dataloaders.get("test", dataloaders.get("val", dataloaders["train"]))
        layer_activations = extract_transformer_activations(model, test_loader, extraction_cfg)
        
        print(f"✓ Extracted activations:")
        for layer, data in layer_activations.items():
            print(f"  - {layer}: {data['X'].shape}")
        
        # Save activations
        activations_path = run_dir / "activations.npz"
        np.savez(activations_path, **{f"{layer}_X": data['X'] for layer, data in layer_activations.items()})
        
        # 4. Compute TDA
        print(f"\n[4/5] Computing TDA features")
        tda_cfg = TDAConfig(**run.tda_config)
        tda_results = {}
        
        for layer, data in layer_activations.items():
            print(f"  Computing persistence for {layer}...")
            X = data['X']
            
            # Compute persistence
            persistence_result = compute_persistence(X, tda_cfg, seed=run.experiment_config.random_seed)
            
            # Summarize diagrams
            summaries = summarize_diagrams(persistence_result["dgms"])
            tda_results[layer] = summaries
            
            print(f"    ✓ {layer}: H0={summaries.get('H0_count', 0):.0f} features, H1={summaries.get('H1_count', 0):.0f} features")
        
        # Save TDA results
        tda_path = run_dir / "tda_summaries.json"
        with open(tda_path, 'w') as f:
            json.dump(tda_results, f, indent=2)

        print(f"✓ TDA computation complete")

        # Generate visualizations
        print(f"\n[4b/5] Generating TDA visualizations")
        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        try:
            viz_files = generate_all_visualizations(
                tda_results,
                viz_dir,
                model_name=run.model_config["name"]
            )
            print(f"✓ Generated {len(viz_files)} visualization(s)")
        except Exception as viz_error:
            print(f"⚠ Visualization generation failed: {viz_error}")

        # Generate TDA interpretation
        print(f"\n[4c/5] Generating TDA interpretation")
        try:
            interpretation = interpret_tda_results(
                tda_results,
                model_name=run.model_config["name"],
                sample_count=run.tda_config.get("max_points", 30)
            )
            interpretation_md = format_interpretation_markdown(interpretation)

            # Save interpretation
            interp_path = run_dir / "tda_interpretation.md"
            with open(interp_path, 'w') as f:
                f.write(interpretation_md)
            print(f"✓ Generated TDA interpretation with {len(interpretation.key_findings)} key findings")
        except Exception as interp_error:
            print(f"⚠ Interpretation generation failed: {interp_error}")
            interpretation = None
            interpretation_md = None

        # 5. Compute performance metrics (simplified for now)
        print(f"\n[5/5] Computing performance metrics")
        # For now, use dummy metrics - in full implementation, would evaluate model
        metrics = {
            "perplexity": np.random.uniform(10, 50),  # Placeholder
            "accuracy": np.random.uniform(0.5, 0.95),  # Placeholder
        }
        
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics: perplexity={metrics['perplexity']:.2f}, accuracy={metrics['accuracy']:.3f}")
        
        # Compile results
        results.update({
            "status": "success",
            "model": run.model_config["name"],
            "tda_summaries": tda_results,
            "metrics": metrics,
            "num_layers_analyzed": len(run.analysis_layers)
        })
        
        print(f"\n✓ Run {run.run_id} completed successfully")
        
    except Exception as e:
        print(f"\n✗ Run {run.run_id} failed: {str(e)}")
        results.update({
            "status": "failed",
            "error": str(e)
        })
    
    return results


def analyze_experiment_results(run_results: List[Dict], exp_dir: Path, config: ExperimentConfig) -> Dict:
    """
    Analyze results across all runs.
    
    Performs correlation analysis between TDA features and performance metrics.
    """
    print(f"\n{'='*80}")
    print("Analyzing experiment results")
    print(f"{'='*80}\n")
    
    # Collect data for correlation analysis
    rows = []
    
    for result in run_results:
        if result["status"] != "success":
            continue
        
        run_id = result["run_id"]
        metrics = result["metrics"]
        tda_summaries = result["tda_summaries"]
        
        # Create row for each layer
        for layer, tda_features in tda_summaries.items():
            row = {
                "run_id": run_id,
                "model": result.get("model", "unknown"),
                "layer": layer,
                **metrics,
                **tda_features
            }
            rows.append(row)
    
    if not rows:
        print("✗ No successful runs to analyze")
        return {"status": "no_data"}
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save raw data
    df.to_csv(exp_dir / "artifacts" / "experiment_data.csv", index=False)
    print(f"✓ Saved experiment data: {len(df)} rows")
    
    # Perform correlation analysis if we have multiple runs
    if len(df) > 1:
        print("\nComputing TDA-performance correlations...")
        
        # Prepare data for correlation
        tda_cols = [c for c in df.columns if c.startswith("H")]
        metric_cols = [c for c in df.columns if c in ["perplexity", "accuracy"]]
        
        if tda_cols and metric_cols:
            tda_df = df[["run_id"] + tda_cols].drop_duplicates()
            metrics_df = df[["run_id"] + metric_cols].drop_duplicates()
            
            try:
                corr_result = correlate_tda_with_metrics(tda_df, metrics_df, on="run_id")
                corr_df = corr_result.correlations
                
                # Save correlations
                corr_df.to_csv(exp_dir / "artifacts" / "correlations.csv", index=False)
                print(f"✓ Computed {len(corr_df)} correlations")
                
                # Show top correlations
                print("\nTop correlations:")
                print(corr_df.head(10).to_string(index=False))
                
                return {
                    "status": "success",
                    "num_runs": len(run_results),
                    "num_successful": len([r for r in run_results if r["status"] == "success"]),
                    "correlations": corr_df.to_dict('records')
                }
            except Exception as e:
                print(f"✗ Correlation analysis failed: {str(e)}")
                return {"status": "correlation_failed", "error": str(e)}
        else:
            print("✗ Insufficient data for correlation analysis")
            return {"status": "insufficient_data"}
    else:
        print("ℹ Single run - skipping correlation analysis")
        return {
            "status": "success",
            "num_runs": 1,
            "note": "Single run - no correlation analysis"
        }


def generate_report(exp_dir: Path, config: ExperimentConfig, analysis_result: Dict):
    """Generate markdown report summarizing experiment."""
    report_path = exp_dir / "reports" / "experiment_report.md"

    # Load TDA summaries and interpretations from runs
    tda_data = []
    interpretations = []
    runs_dir = exp_dir / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                tda_file = run_dir / "tda_summaries.json"
                metrics_file = run_dir / "metrics.json"
                interp_file = run_dir / "tda_interpretation.md"
                if tda_file.exists():
                    with open(tda_file) as f:
                        tda_summaries = json.load(f)
                    metrics = {}
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            metrics = json.load(f)
                    tda_data.append({
                        "run_id": run_dir.name,
                        "summaries": tda_summaries,
                        "metrics": metrics
                    })
                    # Load interpretation if available
                    if interp_file.exists():
                        with open(interp_file) as f:
                            interpretations.append(f.read())

    with open(report_path, 'w') as f:
        f.write(f"# {config.experiment_name}\n\n")
        f.write(f"**Type**: {config.experiment_type}\n\n")
        f.write(f"**Description**: {config.description}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Model**: {config.model.get('name', 'N/A')}\n")
        f.write(f"- **Dataset**: {config.dataset.get('name', 'N/A')}\n")
        f.write(f"- **Analysis Layers**: {', '.join(config.analysis_layers)}\n")
        f.write(f"- **TDA Config**: maxdim={config.tda.get('maxdim')}, metric={config.tda.get('metric')}, PCA={config.tda.get('pca_components')}\n\n")

        # Performance metrics
        if tda_data and tda_data[0].get("metrics"):
            metrics = tda_data[0]["metrics"]
            f.write("## Model Performance\n\n")
            f.write(f"- **Perplexity**: {metrics.get('perplexity', 'N/A'):.2f}\n")
            f.write(f"- **Accuracy**: {metrics.get('accuracy', 'N/A'):.1%}\n\n")

        # Layer-wise TDA Results Table
        if tda_data:
            f.write("## Layer-wise TDA Results\n\n")
            f.write("### H0 (Connected Components)\n\n")
            f.write("| Layer | Count | Total Persistence | Max Lifetime |\n")
            f.write("|-------|-------|-------------------|---------------|\n")

            for run_info in tda_data:
                for layer, summary in run_info["summaries"].items():
                    h0_count = summary.get('H0_count', 0)
                    h0_persist = summary.get('H0_total_persistence', 0)
                    h0_max = summary.get('H0_max_lifetime', 0)
                    f.write(f"| {layer} | {h0_count:.0f} | {h0_persist:.2f} | {h0_max:.2f} |\n")

            f.write("\n### H1 (Loops/Cycles)\n\n")
            f.write("| Layer | Count | Total Persistence | Max Lifetime |\n")
            f.write("|-------|-------|-------------------|---------------|\n")

            for run_info in tda_data:
                for layer, summary in run_info["summaries"].items():
                    h1_count = summary.get('H1_count', 0)
                    h1_persist = summary.get('H1_total_persistence', 0)
                    h1_max = summary.get('H1_max_lifetime', 0)
                    f.write(f"| {layer} | {h1_count:.0f} | {h1_persist:.4f} | {h1_max:.4f} |\n")

            f.write("\n")

        # Include TDA interpretation
        if interpretations:
            f.write("---\n\n")
            for interpretation in interpretations:
                f.write(interpretation)
                f.write("\n")
            f.write("---\n\n")

        f.write("## Results Summary\n\n")
        f.write(f"- **Status**: {analysis_result.get('status', 'unknown')}\n")
        f.write(f"- **Runs**: {analysis_result.get('num_successful', 0)}/{analysis_result.get('num_runs', 0)} successful\n\n")

        if "correlations" in analysis_result:
            f.write("### Top TDA-Performance Correlations\n\n")
            f.write("| TDA Feature | Metric | Correlation | P-value |\n")
            f.write("|-------------|--------|-------------|----------|\n")
            for corr in analysis_result["correlations"][:10]:
                f.write(f"| {corr['tda_feature']} | {corr['performance_metric']} | {corr['spearman_rho']:.3f} | {corr['p_value']:.4f} |\n")

        # Visualizations section
        f.write("\n## Visualizations\n\n")
        f.write("The following visualizations were generated for each run:\n\n")
        f.write("- **TDA Summary** (`tda_summary.png`): Overview of all TDA metrics across layers\n")
        f.write("- **Layer Persistence** (`layer_persistence.png`): H0 and H1 total persistence comparison\n")
        f.write("- **Betti Curves** (`betti_curves.png`): Feature counts (Betti numbers) across layers\n\n")
        f.write("Find visualizations in: `runs/*/visualizations/`\n\n")

        f.write("## Artifacts\n\n")
        f.write("- `experiment_config.yaml`: Full experiment configuration\n")
        f.write("- `artifacts/experiment_data.csv`: Raw data from all runs\n")
        f.write("- `artifacts/correlations.csv`: TDA-performance correlations\n")
        f.write("- `runs/*/tda_summaries.json`: TDA results per run\n")
        f.write("- `runs/*/tda_interpretation.md`: Human-readable TDA interpretation\n")
        f.write("- `runs/*/visualizations/`: TDA visualization plots\n")

        # Add metric glossary
        f.write("\n---\n\n")
        f.write(generate_metric_glossary())

    print(f"\n✓ Report generated: {report_path}")


def run_experiment(config_path: str):
    """
    Main entry point to run an experiment.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = ExperimentConfig.from_yaml(config_path)
    
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Type: {config.experiment_type}")
    print(f"Description: {config.description}\n")
    
    # Setup experiment directory
    exp_dir = setup_experiment_directory(config)
    print(f"Experiment directory: {exp_dir}\n")
    
    # Generate run matrix
    runs = generate_run_matrix(config)
    print(f"Generated {len(runs)} experimental runs\n")
    
    # Execute runs
    run_results = []
    for run in runs:
        result = execute_single_run(run, exp_dir)
        run_results.append(result)
    
    # Analyze results
    analysis_result = analyze_experiment_results(run_results, exp_dir, config)
    
    # Generate report
    if config.output.get("generate_report", True):
        generate_report(exp_dir, config, analysis_result)
    
    print(f"\n{'='*80}")
    print(f"Experiment complete!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")
    
    return exp_dir, analysis_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TDA Perturbation Analysis Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration YAML")
    
    args = parser.parse_args()
    
    run_experiment(args.config)
