#!/usr/bin/env python
"""
MLP Depth Ablation Study

Systematic comparison of activation topology across MLP architectures
of varying depths (2-6 layers).

Workflow:
1. Train MLPs of different depths on the same dataset
2. Extract activations (10k-50k samples per layer)
3. Characterize geometry (intrinsic dimension, k-NN structure)
4. Apply TDA (subsampled to 2000 for Ripser)
5. Compare results across depths

Usage:
    python scripts/run_mlp_ablation.py --dataset digits --samples 10000
    python scripts/run_mlp_ablation.py --dataset mnist --samples 20000 --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from todacomm.models.mlp import MLPModel, MLPConfig, PRESET_CONFIGS
from todacomm.data.standard_datasets import (
    StandardDatasetConfig,
    load_standard_dataset,
    create_dataloaders,
    get_dataset_info,
)
from todacomm.extract.mlp_activations import (
    MLPActivationConfig,
    extract_mlp_activations,
    get_activation_statistics,
    subsample_activations,
)
from todacomm.analysis.geometry import (
    GeometryConfig,
    characterize_geometry,
    summarize_geometry,
    recommend_pca_components,
)
from todacomm.training.mlp_trainer import (
    TrainingConfig,
    train_mlp,
)
from todacomm.tda.persistence import TDAConfig, compute_persistence, summarize_diagrams
from torch.utils.data import ConcatDataset, DataLoader


# Depth configurations for ablation
DEPTH_CONFIGS = {
    "shallow_2": [256],              # 2 layers total (1 hidden + output)
    "shallow_3": [256, 128],         # 3 layers total
    "medium_4": [256, 128, 64],      # 4 layers total
    "medium_5": [256, 192, 128, 64], # 5 layers total
    "deep_6": [256, 192, 128, 96, 64],  # 6 layers total
}


def run_single_depth(
    depth_name: str,
    hidden_dims: List[int],
    train_loader,
    val_loader,
    extraction_loader,
    input_dim: int,
    num_classes: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train and analyze a single depth configuration.

    Returns dictionary with training, geometry, and TDA results.
    """
    print(f"\n{'='*60}")
    print(f"Training {depth_name}: {hidden_dims}")
    print(f"{'='*60}")

    # 1. Create and train model
    mlp_config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        activation="relu",
        dropout=0.1,
    )
    model = MLPModel(mlp_config)

    train_config = TrainingConfig(
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        device=config["device"],
        optimizer="adam",
        early_stopping_patience=5,
    )

    train_result = train_mlp(model, train_loader, val_loader, train_config)
    print(f"  Training: val_acc={train_result.best_val_acc:.4f} @ epoch {train_result.best_epoch}")

    # 2. Extract activations from ALL data (train+val+test combined)
    extract_config = MLPActivationConfig(
        layers="all",
        max_samples=config["num_samples"],
        device=config["device"],
        show_progress=True,
    )

    activations = extract_mlp_activations(model, extraction_loader, extract_config)
    act_stats = get_activation_statistics(activations)

    print(f"  Extracted {act_stats[list(act_stats.keys())[0]]['n_samples']} samples")

    # 3. Geometry characterization
    geometry_config = GeometryConfig(
        k_neighbors=20,
        mle_k=10,
        max_samples_for_dim=5000,
        seed=config["seed"],
    )

    geometry_results = {}
    for layer_name, data in activations.items():
        geometry_results[layer_name] = characterize_geometry(
            data["X"], layer_name, geometry_config
        )

    geometry_df = summarize_geometry(list(geometry_results.values()))
    print(f"  Geometry: MLE dim range [{geometry_df['mle_intrinsic_dim'].min():.1f}, "
          f"{geometry_df['mle_intrinsic_dim'].max():.1f}]")

    # 4. TDA with subsampling
    recommended_pca = recommend_pca_components(list(geometry_results.values()))
    print(f"  Recommended PCA components: {recommended_pca}")

    # Subsample for TDA (Ripser is O(n³))
    tda_samples = min(2000, config["num_samples"])
    subsampled = subsample_activations(activations, tda_samples, seed=config["seed"])

    tda_config = TDAConfig(
        maxdim=1,
        pca_components=min(recommended_pca, 50),
        max_points=tda_samples,
        sampling_strategy="uniform",
    )

    tda_results = {}
    for layer_name, data in subsampled.items():
        persistence = compute_persistence(data["X"], tda_config, seed=config["seed"])
        tda_results[layer_name] = summarize_diagrams(persistence["dgms"])

    print(f"  TDA: H0 range [{min(r['H0_count'] for r in tda_results.values())}, "
          f"{max(r['H0_count'] for r in tda_results.values())}]")

    return {
        "depth_name": depth_name,
        "hidden_dims": hidden_dims,
        "num_layers": len(hidden_dims) + 1,
        "training": train_result.to_dict(),
        "geometry": geometry_df.to_dict(orient="records"),
        "tda": tda_results,
        "activation_stats": act_stats,
        "recommended_pca": recommended_pca,
    }


def run_ablation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full ablation study across all depth configurations.
    """
    print(f"\nMLP Depth Ablation Study")
    print(f"Dataset: {config['dataset']}")
    print(f"Requested samples: {config['num_samples']}")
    print(f"Device: {config['device']}")
    print(f"Depths: {list(DEPTH_CONFIGS.keys())}")

    # Load dataset
    dataset_info = get_dataset_info(config["dataset"])
    print(f"\nDataset info: {dataset_info}")

    dataset_config = StandardDatasetConfig(
        dataset_name=config["dataset"],
        batch_size=64,
        num_samples=config.get("max_train_samples"),
        normalize=True,
        seed=config["seed"],
    )

    datasets = load_standard_dataset(dataset_config)
    loaders = create_dataloaders(datasets, batch_size=64)

    # Create combined loader for activation extraction (train + val + test)
    # This gives us the maximum number of samples for geometry/TDA analysis
    all_datasets = []
    total_samples = 0
    for split_name, dataset in datasets.items():
        all_datasets.append(dataset)
        total_samples += len(dataset)

    combined_dataset = ConcatDataset(all_datasets)
    extraction_loader = DataLoader(
        combined_dataset,
        batch_size=256,  # Larger batch for extraction efficiency
        shuffle=False,
        num_workers=0,
    )

    print(f"Total samples available: {total_samples}")
    if total_samples < config["num_samples"]:
        print(f"  WARNING: Requested {config['num_samples']} samples but only {total_samples} available!")
        print(f"  TIP: Use MNIST (60k samples) for large-scale analysis")

    # Run ablation for each depth
    results = {}
    for depth_name, hidden_dims in DEPTH_CONFIGS.items():
        try:
            result = run_single_depth(
                depth_name=depth_name,
                hidden_dims=hidden_dims,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                extraction_loader=extraction_loader,
                input_dim=dataset_info["input_dim"],
                num_classes=dataset_info["num_classes"],
                config=config,
            )
            results[depth_name] = result
        except Exception as e:
            print(f"  ERROR in {depth_name}: {e}")
            import traceback
            traceback.print_exc()
            results[depth_name] = {"error": str(e)}

    return {
        "config": config,
        "dataset_info": dataset_info,
        "total_samples_available": total_samples,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def summarize_ablation(ablation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table of ablation results.
    """
    rows = []
    for depth_name, result in ablation_results["results"].items():
        if "error" in result:
            continue

        row = {
            "depth": depth_name,
            "num_layers": result["num_layers"],
            "val_acc": result["training"]["best_val_acc"],
            "best_epoch": result["training"]["best_epoch"],
            "recommended_pca": result["recommended_pca"],
        }

        # Add geometry metrics (from last hidden layer)
        geometry = result["geometry"]
        if geometry:
            last_hidden = [g for g in geometry if g["layer"].startswith("hidden_")][-1]
            row["mle_dim"] = last_hidden.get("mle_intrinsic_dim")
            row["hubness"] = last_hidden.get("hubness")

        # Add TDA metrics (from last hidden layer)
        tda = result["tda"]
        hidden_layers = [k for k in tda.keys() if k.startswith("hidden_")]
        if hidden_layers:
            last_hidden_tda = tda[hidden_layers[-1]]
            row["H0_count"] = last_hidden_tda.get("H0_count")
            row["H1_count"] = last_hidden_tda.get("H1_count")
            row["H0_persistence"] = last_hidden_tda.get("H0_total_persistence")

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="MLP Depth Ablation Study for TDA Analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digits",
        choices=["digits", "iris", "wine", "breast_cancer", "mnist", "fashion_mnist"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of activation samples to extract",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: experiments/mlp_ablation_<timestamp>)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        "dataset": args.dataset,
        "num_samples": args.samples,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": args.device,
        "seed": args.seed,
        "max_train_samples": None,  # Use full training set
    }

    # Run ablation
    results = run_ablation(config)

    # Create summary
    summary_df = summarize_ablation(results)
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/mlp_ablation_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    with open(output_dir / "results.json", "w") as f:
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        json.dump(results, f, indent=2, default=convert)

    # Save summary CSV
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
