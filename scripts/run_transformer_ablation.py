#!/usr/bin/env python
"""
Transformer Depth/Size Ablation Study

Systematic comparison of activation topology across transformer architectures
of varying sizes (e.g., Pythia 70M → 410M, GPT-2 → GPT-2-medium).

Workflow:
1. Load pretrained transformers of different sizes
2. Extract activations from Wikitext (10k-50k samples per layer)
3. Characterize geometry (intrinsic dimension, k-NN structure)
4. Apply TDA (subsampled to 2000 for Ripser)
5. Compare results across model sizes

Usage:
    python scripts/run_transformer_ablation.py --family pythia --samples 10000
    python scripts/run_transformer_ablation.py --family gpt2 --samples 20000 --device mps
    python scripts/run_transformer_ablation.py --models gpt2,gpt2-medium --samples 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from todacomm.models.transformer import TransformerModel, TransformerConfig, load_pretrained_transformer
from todacomm.data.language_datasets import (
    DatasetConfig,
    load_wikitext2,
    create_dataloaders,
)
from todacomm.extract.transformer_activations import (
    ActivationConfig,
    extract_transformer_activations,
)
from todacomm.analysis.geometry import (
    GeometryConfig,
    characterize_geometry,
    summarize_geometry,
    recommend_pca_components,
)
from todacomm.tda.persistence import TDAConfig, compute_persistence, summarize_diagrams


# Model families for ablation
MODEL_FAMILIES = {
    "pythia": {
        "pythia-70m": "EleutherAI/pythia-70m",
        "pythia-160m": "EleutherAI/pythia-160m",
        "pythia-410m": "EleutherAI/pythia-410m",
    },
    "gpt2": {
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
    },
    "smollm": {
        "smollm-135m": "HuggingFaceTB/SmolLM-135M",
        "smollm-360m": "HuggingFaceTB/SmolLM-360M",
    },
    "opt": {
        "opt-125m": "facebook/opt-125m",
        "opt-350m": "facebook/opt-350m",
    },
}

# Layers to extract for each model (will auto-detect if not specified)
DEFAULT_LAYERS = ["embedding", "final"]  # + intermediate layers


def get_model_info(model: TransformerModel) -> Dict[str, Any]:
    """Extract model metadata."""
    num_layers = model._get_num_layers()

    # Try to get hidden size from model config
    hidden_size = None
    if hasattr(model.model, "config"):
        cfg = model.model.config
        for attr in ["hidden_size", "n_embd", "d_model"]:
            if hasattr(cfg, attr):
                hidden_size = getattr(cfg, attr)
                break

    # Count parameters
    num_params = sum(p.numel() for p in model.model.parameters())

    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_params": num_params,
        "num_params_millions": num_params / 1e6,
    }


def select_layers_to_extract(model: TransformerModel, strategy: str = "sparse") -> List[str]:
    """
    Select which layers to extract activations from.

    Args:
        model: TransformerModel instance
        strategy:
            - "all": Extract all layers
            - "sparse": Extract embedding, every 4th layer, and final
            - "key": Extract embedding, 25%, 50%, 75%, and final

    Returns:
        List of layer names to extract
    """
    num_layers = model._get_num_layers()

    if strategy == "all":
        layers = ["embedding"]
        layers += [f"layer_{i}" for i in range(num_layers)]
        layers.append("final")
    elif strategy == "sparse":
        layers = ["embedding"]
        # Every 4th layer, or at least 3 intermediate layers
        step = max(1, num_layers // 4)
        for i in range(0, num_layers, step):
            layers.append(f"layer_{i}")
        if f"layer_{num_layers-1}" not in layers:
            layers.append(f"layer_{num_layers-1}")
        layers.append("final")
    elif strategy == "key":
        layers = ["embedding"]
        # 25%, 50%, 75% depth
        for pct in [0.25, 0.5, 0.75]:
            idx = int(num_layers * pct)
            layers.append(f"layer_{idx}")
        layers.append(f"layer_{num_layers-1}")
        layers.append("final")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Remove duplicates while preserving order
    seen = set()
    unique_layers = []
    for layer in layers:
        if layer not in seen:
            seen.add(layer)
            unique_layers.append(layer)

    return unique_layers


def subsample_activations(
    activations: Dict[str, Dict[str, np.ndarray]],
    n_samples: int,
    seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """Subsample activations to n_samples for TDA."""
    rng = np.random.default_rng(seed)
    result = {}

    for layer_name, data in activations.items():
        X = data["X"]
        y = data["y"]

        if X.shape[0] <= n_samples:
            result[layer_name] = data
        else:
            indices = rng.choice(X.shape[0], size=n_samples, replace=False)
            result[layer_name] = {
                "X": X[indices],
                "y": y[indices],
            }

    return result


def run_single_model(
    model_name: str,
    model_path: str,
    dataloader,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load model, extract activations, characterize geometry, and run TDA.

    Returns dictionary with model info, geometry, and TDA results.
    """
    print(f"\n{'='*60}")
    print(f"Processing {model_name}: {model_path}")
    print(f"{'='*60}")

    # 1. Load model
    print(f"  Loading model...")
    try:
        model = load_pretrained_transformer(
            model_path,
            task="lm",
            device=config["device"]
        )
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {"error": str(e)}

    model_info = get_model_info(model)
    print(f"  Model: {model_info['num_params_millions']:.1f}M params, "
          f"{model_info['num_layers']} layers, "
          f"hidden_size={model_info['hidden_size']}")

    # 2. Select layers to extract
    layers_to_extract = select_layers_to_extract(model, strategy=config["layer_strategy"])
    print(f"  Extracting layers: {layers_to_extract}")

    # 3. Extract activations
    extract_config = ActivationConfig(
        layers=layers_to_extract,
        max_samples=config["num_samples"],
        pool_strategy="mean",  # Mean pool over sequence
        device=config["device"],
        show_progress=True,
    )

    print(f"  Extracting activations (max {config['num_samples']} samples)...")
    activations = extract_transformer_activations(model, dataloader, extract_config)

    first_layer = list(activations.keys())[0]
    n_extracted = activations[first_layer]["X"].shape[0]
    print(f"  Extracted {n_extracted} samples")

    # Get activation statistics
    activation_stats = {}
    for layer_name, data in activations.items():
        X = data["X"]
        activation_stats[layer_name] = {
            "n_samples": X.shape[0],
            "n_dims": X.shape[1],
            "mean": float(np.mean(X)),
            "std": float(np.std(X)),
            "min": float(np.min(X)),
            "max": float(np.max(X)),
            "sparsity": float(np.mean(X == 0)),
            "memory_mb": X.nbytes / (1024 * 1024),
        }

    # 4. Geometry characterization
    print(f"  Characterizing geometry...")
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

    # 5. TDA with subsampling
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

    print(f"  Computing TDA on {tda_samples} samples...")
    tda_results = {}
    for layer_name, data in subsampled.items():
        persistence = compute_persistence(data["X"], tda_config, seed=config["seed"])
        tda_results[layer_name] = summarize_diagrams(persistence["dgms"])

    print(f"  TDA: H0 range [{min(r['H0_count'] for r in tda_results.values())}, "
          f"{max(r['H0_count'] for r in tda_results.values())}]")

    # Free GPU memory
    del model
    if config["device"] != "cpu":
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return {
        "model_name": model_name,
        "model_path": model_path,
        "model_info": model_info,
        "layers_extracted": layers_to_extract,
        "geometry": geometry_df.to_dict(orient="records"),
        "tda": tda_results,
        "activation_stats": activation_stats,
        "recommended_pca": recommended_pca,
    }


def run_ablation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full ablation study across all specified models.
    """
    print(f"\nTransformer Ablation Study")
    print(f"Models: {list(config['models'].keys())}")
    print(f"Requested samples: {config['num_samples']}")
    print(f"Device: {config['device']}")

    # Load Wikitext dataset
    print(f"\nLoading Wikitext-2 dataset...")
    dataset_config = DatasetConfig(
        dataset_name="wikitext2",
        tokenizer_name=config.get("tokenizer", "gpt2"),
        max_length=config.get("max_length", 512),
        batch_size=config.get("batch_size", 8),
        num_samples=None,  # Use full dataset
        seed=config["seed"],
    )

    datasets = load_wikitext2(dataset_config)
    loaders = create_dataloaders(datasets, batch_size=config.get("batch_size", 8))

    # Use train + val for extraction
    from torch.utils.data import ConcatDataset, DataLoader
    combined_dataset = ConcatDataset([datasets["train"], datasets["val"]])
    extraction_loader = DataLoader(
        combined_dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=False,
        num_workers=0,
    )

    total_samples = len(combined_dataset)
    print(f"Total samples available: {total_samples}")
    if total_samples < config["num_samples"]:
        print(f"  WARNING: Requested {config['num_samples']} samples but only {total_samples} available!")

    # Run ablation for each model
    results = {}
    for model_name, model_path in config["models"].items():
        try:
            result = run_single_model(
                model_name=model_name,
                model_path=model_path,
                dataloader=extraction_loader,
                config=config,
            )
            results[model_name] = result
        except Exception as e:
            print(f"  ERROR in {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"error": str(e)}

    return {
        "config": {k: v for k, v in config.items() if k != "models"},
        "models": list(config["models"].keys()),
        "total_samples_available": total_samples,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def summarize_ablation(ablation_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table of ablation results.
    """
    rows = []
    for model_name, result in ablation_results["results"].items():
        if "error" in result:
            continue

        row = {
            "model": model_name,
            "params_M": result["model_info"]["num_params_millions"],
            "num_layers": result["model_info"]["num_layers"],
            "hidden_size": result["model_info"]["hidden_size"],
            "recommended_pca": result["recommended_pca"],
        }

        # Add geometry metrics (from final layer)
        geometry = result["geometry"]
        if geometry:
            final = [g for g in geometry if g["layer"] == "final"]
            if final:
                final = final[0]
                row["mle_dim"] = final.get("mle_intrinsic_dim")
                row["hubness"] = final.get("hubness")

        # Add TDA metrics (from final layer)
        tda = result["tda"]
        if "final" in tda:
            final_tda = tda["final"]
            row["H0_count"] = final_tda.get("H0_count")
            row["H1_count"] = final_tda.get("H1_count")
            row["H0_persistence"] = final_tda.get("H0_total_persistence")

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by parameter count
    if "params_M" in df.columns:
        df = df.sort_values("params_M")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Transformer Ablation Study for TDA Analysis"
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=list(MODEL_FAMILIES.keys()),
        help="Model family to ablate (pythia, gpt2, smollm, opt)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of specific models (e.g., gpt2,gpt2-medium)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of activation samples to extract",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--layer-strategy",
        type=str,
        default="sparse",
        choices=["all", "sparse", "key"],
        help="Which layers to extract (all, sparse, key)",
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
        help="Output directory (default: experiments/transformer_ablation_<timestamp>)",
    )

    args = parser.parse_args()

    # Determine which models to run
    if args.models:
        # Parse comma-separated model list
        model_names = [m.strip() for m in args.models.split(",")]
        models = {}
        for name in model_names:
            # Check if it's a known model in any family
            found = False
            for family_name, family_models in MODEL_FAMILIES.items():
                if name in family_models:
                    models[name] = family_models[name]
                    found = True
                    break
            if not found:
                # Assume it's a HuggingFace model path
                models[name] = name
    elif args.family:
        models = MODEL_FAMILIES[args.family]
    else:
        # Default to GPT-2 family
        print("No family or models specified, defaulting to GPT-2 family")
        models = MODEL_FAMILIES["gpt2"]

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        "models": models,
        "num_samples": args.samples,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "layer_strategy": args.layer_strategy,
        "device": args.device,
        "seed": args.seed,
        "tokenizer": "gpt2",  # Use GPT-2 tokenizer as default
    }

    # Run ablation
    results = run_ablation(config)

    # Create summary
    summary_df = summarize_ablation(results)
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No successful results to summarize")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"experiments/transformer_ablation_{timestamp}")

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
