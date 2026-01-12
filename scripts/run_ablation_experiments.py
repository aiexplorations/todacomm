#!/usr/bin/env python
"""
Run ablation experiments and bootstrap CI analysis for the technical report.

This script:
1. Loads activation data from key models
2. Computes bootstrap confidence intervals
3. Runs sample size ablation
4. Runs PCA components ablation
5. Saves results to JSON for inclusion in the report
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from todacomm.tda.persistence import (
    TDAConfig,
    bootstrap_persistence_ci,
    compare_bootstrap_results,
    ablation_sample_size,
    ablation_pca_components,
    compute_persistence,
    summarize_diagrams
)


# Key models to analyze (representing encoder, decoder families)
MODELS_TO_ANALYZE = {
    "bert": "experiments/bert_tda_20251206_094654/runs/run_0/activations.npz",
    "gpt2": "experiments/gpt2_tda_20251206_094353/runs/run_0/activations.npz",
    "distilgpt2": "experiments/distilgpt2_tda_20251206_094343/runs/run_0/activations.npz",
    "smollm2-360m": "experiments/smollm2-360m_tda_20251206_094548/runs/run_0/activations.npz",
    "pythia-410m": "experiments/pythia-410m_tda_20251206_094528/runs/run_0/activations.npz",
}

# Layers to analyze for each model (peak layers from meta-analysis)
PEAK_LAYERS = {
    "bert": "layer_10_X",
    "gpt2": "layer_10_X",
    "distilgpt2": "layer_4_X",
    "smollm2-360m": "layer_30_X",
    "pythia-410m": "layer_23_X",
}


def load_activations(npz_path: str) -> Dict[str, np.ndarray]:
    """Load activation data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def run_bootstrap_analysis(
    models_data: Dict[str, Dict[str, np.ndarray]],
    config: TDAConfig,
    seed: int = 42
) -> Dict[str, Any]:
    """Run bootstrap CI analysis on peak layers for all models."""
    results = {}

    for model_name, activations in models_data.items():
        print(f"\n  Running bootstrap for {model_name}...")
        peak_layer = PEAK_LAYERS.get(model_name, "final_X")

        if peak_layer not in activations:
            # Find the highest available layer
            layer_keys = [k for k in activations.keys() if k.startswith("layer_")]
            if layer_keys:
                peak_layer = sorted(layer_keys, key=lambda x: int(x.split("_")[1]))[-1]
            else:
                peak_layer = "final_X"

        X = activations[peak_layer]
        print(f"    Layer: {peak_layer}, Shape: {X.shape}")

        # Run bootstrap
        boot_result = bootstrap_persistence_ci(
            X, config,
            n_bootstrap=100,
            ci_level=0.95,
            seed=seed
        )

        results[model_name] = {
            "layer": peak_layer,
            "n_samples": X.shape[0],
            "n_dims": X.shape[1],
            "point_estimate": boot_result.point_estimate,
            "ci_lower": boot_result.ci_lower,
            "ci_upper": boot_result.ci_upper,
            "std_error": boot_result.std_error,
            "n_bootstrap": boot_result.n_bootstrap,
            "ci_level": boot_result.ci_level
        }

        # Print summary
        h0 = boot_result.point_estimate["H0_total_persistence"]
        h0_ci = (boot_result.ci_lower["H0_total_persistence"],
                 boot_result.ci_upper["H0_total_persistence"])
        print(f"    H0: {h0:.2f} [{h0_ci[0]:.2f}, {h0_ci[1]:.2f}]")

    return results


def run_pairwise_comparisons(
    models_data: Dict[str, Dict[str, np.ndarray]],
    config: TDAConfig,
    seed: int = 42
) -> Dict[str, Any]:
    """Compare key model pairs for statistical significance."""

    # First compute bootstrap results
    boot_results = {}
    for model_name, activations in models_data.items():
        peak_layer = PEAK_LAYERS.get(model_name, "final_X")
        if peak_layer not in activations:
            layer_keys = [k for k in activations.keys() if k.startswith("layer_")]
            if layer_keys:
                peak_layer = sorted(layer_keys, key=lambda x: int(x.split("_")[1]))[-1]
            else:
                peak_layer = "final_X"

        X = activations[peak_layer]
        boot_results[model_name] = bootstrap_persistence_ci(
            X, config, n_bootstrap=100, seed=seed
        )

    # Compare pairs
    comparisons = {}
    pairs = [
        ("bert", "gpt2"),
        ("bert", "distilgpt2"),
        ("bert", "smollm2-360m"),
        ("gpt2", "distilgpt2"),
        ("gpt2", "smollm2-360m"),
    ]

    for m1, m2 in pairs:
        if m1 in boot_results and m2 in boot_results:
            comp = compare_bootstrap_results(
                {m1: boot_results[m1], m2: boot_results[m2]},
                metric="H0_total_persistence"
            )
            comparisons[f"{m1}_vs_{m2}"] = comp[f"{m1}_vs_{m2}"]

    return comparisons


def run_sample_size_ablation(
    models_data: Dict[str, Dict[str, np.ndarray]],
    config: TDAConfig,
    seed: int = 42
) -> Dict[str, Any]:
    """Run sample size ablation on selected models."""
    results = {}

    # Use GPT-2 and BERT as representative models
    for model_name in ["gpt2", "bert"]:
        if model_name not in models_data:
            continue

        print(f"\n  Running sample size ablation for {model_name}...")
        activations = models_data[model_name]
        peak_layer = PEAK_LAYERS.get(model_name, "final_X")

        if peak_layer not in activations:
            layer_keys = [k for k in activations.keys() if k.startswith("layer_")]
            peak_layer = sorted(layer_keys, key=lambda x: int(x.split("_")[1]))[-1] if layer_keys else "final_X"

        X = activations[peak_layer]
        n_available = X.shape[0]

        # Test sample sizes up to available data
        sample_sizes = [s for s in [25, 50, 75] if s <= n_available]

        ablation_result = ablation_sample_size(
            X, config,
            sample_sizes=sample_sizes,
            n_repeats=10,
            seed=seed
        )

        results[model_name] = {
            "layer": peak_layer,
            "available_samples": n_available,
            "tested_sizes": sample_sizes,
            "results": ablation_result
        }

        # Print summary
        for size in sample_sizes:
            h0_mean = ablation_result[size]["metrics"]["H0_total_persistence"]["mean"]
            h0_cv = ablation_result[size]["metrics"]["H0_total_persistence"]["cv"]
            print(f"    n={size}: H0={h0_mean:.2f} (CV={h0_cv:.3f})")

    return results


def run_pca_ablation(
    models_data: Dict[str, Dict[str, np.ndarray]],
    config: TDAConfig,
    seed: int = 42
) -> Dict[str, Any]:
    """Run PCA components ablation on selected models."""
    results = {}

    for model_name in ["gpt2", "bert"]:
        if model_name not in models_data:
            continue

        print(f"\n  Running PCA ablation for {model_name}...")
        activations = models_data[model_name]
        peak_layer = PEAK_LAYERS.get(model_name, "final_X")

        if peak_layer not in activations:
            layer_keys = [k for k in activations.keys() if k.startswith("layer_")]
            peak_layer = sorted(layer_keys, key=lambda x: int(x.split("_")[1]))[-1] if layer_keys else "final_X"

        X = activations[peak_layer]
        n_dims = X.shape[1]

        # Test PCA values up to available dimensions
        pca_values = [p for p in [10, 20, 50, 100] if p < n_dims]

        ablation_result = ablation_pca_components(
            X, config,
            pca_values=pca_values,
            n_repeats=5,
            seed=seed
        )

        results[model_name] = {
            "layer": peak_layer,
            "original_dims": n_dims,
            "tested_pca": pca_values,
            "results": ablation_result
        }

        # Print summary
        for pca_k in pca_values:
            key = str(pca_k)
            h0_mean = ablation_result[key]["metrics"]["H0_total_persistence"]["mean"]
            h0_cv = ablation_result[key]["metrics"]["H0_total_persistence"]["cv"]
            print(f"    PCA={pca_k}: H0={h0_mean:.2f} (CV={h0_cv:.3f})")

    return results


def main():
    print("=" * 60)
    print("ToDACoMM Ablation Experiments")
    print("=" * 60)

    # Configuration
    config = TDAConfig(
        maxdim=1,
        pca_components=50,
        max_points=500,
        n_bootstrap=100,
        ci_level=0.95
    )

    # Load all model activations
    print("\n1. Loading activation data...")
    models_data = {}
    for model_name, path in MODELS_TO_ANALYZE.items():
        if Path(path).exists():
            print(f"   Loading {model_name}...")
            models_data[model_name] = load_activations(path)
        else:
            print(f"   WARNING: {path} not found, skipping {model_name}")

    if not models_data:
        print("ERROR: No activation data found!")
        return

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "maxdim": config.maxdim,
            "pca_components": config.pca_components,
            "max_points": config.max_points,
            "n_bootstrap": config.n_bootstrap,
            "ci_level": config.ci_level
        },
        "models_analyzed": list(models_data.keys())
    }

    # Run bootstrap analysis
    print("\n2. Running bootstrap confidence intervals...")
    results["bootstrap_ci"] = run_bootstrap_analysis(models_data, config)

    # Run pairwise comparisons
    print("\n3. Running pairwise significance tests...")
    results["pairwise_comparisons"] = run_pairwise_comparisons(models_data, config)

    # Run sample size ablation
    print("\n4. Running sample size ablation...")
    results["sample_size_ablation"] = run_sample_size_ablation(models_data, config)

    # Run PCA ablation
    print("\n5. Running PCA components ablation...")
    results["pca_ablation"] = run_pca_ablation(models_data, config)

    # Save results
    output_path = Path("experiments/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nBootstrap 95% CIs for H0 Total Persistence:")
    print("-" * 50)
    for model, data in results["bootstrap_ci"].items():
        h0 = data["point_estimate"]["H0_total_persistence"]
        lo = data["ci_lower"]["H0_total_persistence"]
        hi = data["ci_upper"]["H0_total_persistence"]
        print(f"  {model:15s}: {h0:8.2f} [{lo:8.2f}, {hi:8.2f}]")

    print("\nPairwise Comparisons (H0):")
    print("-" * 50)
    for pair, data in results["pairwise_comparisons"].items():
        sig = "SIGNIFICANT" if data["significant"] else "not significant"
        print(f"  {pair:25s}: {sig} (d={data['cohens_d']:.2f})")

    return results


if __name__ == "__main__":
    main()
