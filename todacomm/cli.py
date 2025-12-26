#!/usr/bin/env python3
"""
ToDACoMM Command Line Interface

Usage:
    todacomm run --config configs/gpt2_demo.yaml
    todacomm run --model gpt2 --samples 200
    todacomm run --models gpt2,bert,distilgpt2 --samples 200
    todacomm compare gpt2,bert,pythia-70m
    todacomm list-models
    todacomm list-configs
    todacomm init my_experiment.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import yaml


# Supported models with their configurations
# Models under 500M parameters for efficient TDA analysis
SUPPORTED_MODELS = {
    # === GPT-2 Family ===
    "gpt2": {
        "type": "gpt2",
        "name": "gpt2",
        "task": "lm",
        "tokenizer": "gpt2",
        "description": "GPT-2 Small (117M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "distilgpt2": {
        "type": "gpt2",
        "name": "distilgpt2",
        "task": "lm",
        "tokenizer": "distilgpt2",
        "description": "DistilGPT-2 (82M parameters)",
        "num_layers": 6,
        "default_layers": ["embedding", "layer_0", "layer_3", "layer_5", "final"]
    },
    # === BERT Family ===
    "bert": {
        "type": "bert",
        "name": "bert-base-uncased",
        "task": "classification",
        "tokenizer": "bert-base-uncased",
        "description": "BERT Base Uncased (110M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "distilbert": {
        "type": "distilbert",
        "name": "distilbert-base-uncased",
        "task": "classification",
        "tokenizer": "distilbert-base-uncased",
        "description": "DistilBERT Base Uncased (66M parameters)",
        "num_layers": 6,
        "default_layers": ["embedding", "layer_0", "layer_3", "layer_5", "final"]
    },
    # === OPT Family (Meta) ===
    "opt-125m": {
        "type": "opt",
        "name": "facebook/opt-125m",
        "task": "lm",
        "tokenizer": "facebook/opt-125m",
        "description": "OPT-125M (Meta, 125M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "opt-350m": {
        "type": "opt",
        "name": "facebook/opt-350m",
        "task": "lm",
        "tokenizer": "facebook/opt-350m",
        "description": "OPT-350M (Meta, 350M parameters)",
        "num_layers": 24,
        "default_layers": ["embedding", "layer_0", "layer_11", "layer_23", "final"]
    },
    # === Pythia Family (EleutherAI) ===
    "pythia-70m": {
        "type": "pythia",
        "name": "EleutherAI/pythia-70m",
        "task": "lm",
        "tokenizer": "EleutherAI/pythia-70m",
        "description": "Pythia-70M (EleutherAI, 70M parameters)",
        "num_layers": 6,
        "default_layers": ["embedding", "layer_0", "layer_3", "layer_5", "final"]
    },
    "pythia-160m": {
        "type": "pythia",
        "name": "EleutherAI/pythia-160m",
        "task": "lm",
        "tokenizer": "EleutherAI/pythia-160m",
        "description": "Pythia-160M (EleutherAI, 160M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "pythia-410m": {
        "type": "pythia",
        "name": "EleutherAI/pythia-410m",
        "task": "lm",
        "tokenizer": "EleutherAI/pythia-410m",
        "description": "Pythia-410M (EleutherAI, 410M parameters)",
        "num_layers": 24,
        "default_layers": ["embedding", "layer_0", "layer_11", "layer_23", "final"]
    },
    # === SmolLM Family (HuggingFace) ===
    "smollm-135m": {
        "type": "smollm",
        "name": "HuggingFaceTB/SmolLM-135M",
        "task": "lm",
        "tokenizer": "HuggingFaceTB/SmolLM-135M",
        "description": "SmolLM-135M (HuggingFace, 135M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "smollm-360m": {
        "type": "smollm",
        "name": "HuggingFaceTB/SmolLM-360M",
        "task": "lm",
        "tokenizer": "HuggingFaceTB/SmolLM-360M",
        "description": "SmolLM-360M (HuggingFace, 360M parameters)",
        "num_layers": 32,
        "default_layers": ["embedding", "layer_0", "layer_15", "layer_31", "final"]
    },
    # === Qwen2 Family (Alibaba) ===
    "qwen2-0.5b": {
        "type": "qwen2",
        "name": "Qwen/Qwen2-0.5B",
        "task": "lm",
        "tokenizer": "Qwen/Qwen2-0.5B",
        "description": "Qwen2-0.5B (Alibaba, 500M parameters)",
        "num_layers": 24,
        "default_layers": ["embedding", "layer_0", "layer_11", "layer_23", "final"]
    },
    # === GPT-Neo Family (EleutherAI) ===
    "gpt-neo-125m": {
        "type": "gpt-neo",
        "name": "EleutherAI/gpt-neo-125m",
        "task": "lm",
        "tokenizer": "EleutherAI/gpt-neo-125m",
        "description": "GPT-Neo-125M (EleutherAI, 125M parameters)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    # === 2024-2025 Frontier Small Models ===
    "smollm2-135m": {
        "type": "smollm",
        "name": "HuggingFaceTB/SmolLM2-135M",
        "task": "lm",
        "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        "description": "SmolLM2-135M (HuggingFace, 135M params, 2024)",
        "num_layers": 12,
        "default_layers": ["embedding", "layer_0", "layer_5", "layer_11", "final"]
    },
    "smollm2-360m": {
        "type": "smollm",
        "name": "HuggingFaceTB/SmolLM2-360M",
        "task": "lm",
        "tokenizer": "HuggingFaceTB/SmolLM2-360M",
        "description": "SmolLM2-360M (HuggingFace, 360M params, 2024)",
        "num_layers": 32,
        "default_layers": ["embedding", "layer_0", "layer_15", "layer_31", "final"]
    },
    "qwen2.5-0.5b": {
        "type": "qwen2",
        "name": "Qwen/Qwen2.5-0.5B",
        "task": "lm",
        "tokenizer": "Qwen/Qwen2.5-0.5B",
        "description": "Qwen2.5-0.5B (Alibaba, 494M params, 2024)",
        "num_layers": 24,
        "default_layers": ["embedding", "layer_0", "layer_11", "layer_23", "final"]
    },
    "qwen2.5-coder-0.5b": {
        "type": "qwen2",
        "name": "Qwen/Qwen2.5-Coder-0.5B",
        "task": "lm",
        "tokenizer": "Qwen/Qwen2.5-Coder-0.5B",
        "description": "Qwen2.5-Coder-0.5B (Alibaba, 494M params, 2024)",
        "num_layers": 24,
        "default_layers": ["embedding", "layer_0", "layer_11", "layer_23", "final"]
    },
    "gemma2-2b": {
        "type": "gemma",
        "name": "google/gemma-2-2b",
        "task": "lm",
        "tokenizer": "google/gemma-2-2b",
        "description": "Gemma-2-2B (Google, 2.6B params, 2024)",
        "num_layers": 26,
        "default_layers": ["embedding", "layer_0", "layer_12", "layer_25", "final"]
    },
}


def get_all_layers(model_key: str) -> list:
    """Generate list of all layers for a model."""
    model_info = SUPPORTED_MODELS[model_key]
    num_layers = model_info["num_layers"]
    layers = ["embedding"]
    layers.extend([f"layer_{i}" for i in range(num_layers)])
    layers.append("final")
    return layers


def get_pool_strategy(model_type: str) -> str:
    """
    Determine appropriate pooling strategy based on model architecture.

    For decoder-only models (GPT-2, Pythia, etc.), uses 'last' token pooling
    as the final token contains aggregated sequence information and is used
    for next-token prediction.

    For encoder models (BERT, DistilBERT), uses 'cls' token pooling as the
    [CLS] token is specifically trained to aggregate sequence-level information.

    References:
    - Persistence Topological Features in LLMs (arXiv:2410.11042): Uses last token
      for autoregressive models as it "captures information about the entire sequence"
    - BERT paper: [CLS] token designed for sequence-level classification

    Args:
        model_type: Model type string (e.g., 'gpt2', 'bert', 'pythia')

    Returns:
        Pooling strategy: 'last' for decoders, 'cls' for encoders
    """
    # Encoder models use [CLS] token
    encoder_types = {"bert", "distilbert", "roberta", "electra"}

    if model_type.lower() in encoder_types:
        return "cls"
    else:
        # Decoder models use last token
        return "last"


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="todacomm",
        description="ToDACoMM - Topological Data Analysis Comparison of Multiple Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  todacomm run --config configs/gpt2_demo.yaml
  todacomm run --model gpt2 --samples 200
  todacomm run --model gpt2 --layers all              # Analyze ALL 14 layers
  todacomm run --models gpt2,bert,pythia-70m          # Analyze multiple models
  todacomm run --model gpt2 --layers embedding,layer_5,final
  todacomm compare gpt2,bert,pythia-70m               # Compare existing results
  todacomm list-models
  todacomm list-configs
  todacomm init my_config.yaml --model gpt2
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run TDA analysis pipeline")
    run_parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "-m", "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Single model to analyze (see 'todacomm list-models')"
    )
    run_parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to analyze (e.g., 'gpt2,bert,pythia-70m'). Defaults to --layers all."
    )
    run_parser.add_argument(
        "--hf-model",
        type=str,
        help="Any HuggingFace model name (e.g., 'microsoft/phi-1_5')"
    )
    run_parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer layers (required with --hf-model if auto-detection fails)"
    )
    run_parser.add_argument(
        "-n", "--samples",
        type=int,
        default=200,
        help="Number of samples to use (default: 200)"
    )
    run_parser.add_argument(
        "-l", "--layers",
        type=str,
        help="Layers to analyze: 'all' for every layer, or comma-separated list (e.g., embedding,layer_5,final)"
    )
    run_parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "squad"],
        help="Dataset to use (default: wikitext2)"
    )
    run_parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets (e.g., 'wikitext2,squad'). Analyzes same model on each dataset."
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Experiment name (default: model_name_tda)"
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu)"
    )
    run_parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    run_parser.add_argument(
        "--pca",
        type=int,
        default=50,
        help="PCA components for dimensionality reduction (default: 50)"
    )

    # List models command
    subparsers.add_parser("list-models", help="List supported models")

    # List configs command
    subparsers.add_parser("list-configs", help="List available configuration files")

    # Compare command (meta-analysis of existing results)
    compare_parser = subparsers.add_parser("compare", help="Generate meta-analysis comparing existing model results")
    compare_parser.add_argument(
        "models",
        type=str,
        help="Comma-separated list of models to compare (e.g., 'gpt2,bert,pythia-70m')"
    )
    compare_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output filename for the meta-analysis report"
    )

    # Init command
    init_parser = subparsers.add_parser("init", help="Create a new configuration file")
    init_parser.add_argument(
        "filename",
        type=str,
        help="Name for the new config file"
    )
    init_parser.add_argument(
        "-m", "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        default="gpt2",
        help="Model to configure (default: gpt2)"
    )
    init_parser.add_argument(
        "-n", "--samples",
        type=int,
        default=200,
        help="Number of samples (default: 200)"
    )

    return parser


def cmd_list_models(args: argparse.Namespace) -> int:
    """List supported models."""
    print("\n" + "=" * 60)
    print("Supported Models")
    print("=" * 60 + "\n")

    for key, model in SUPPORTED_MODELS.items():
        print(f"  {key}")
        print(f"    Name: {model['name']}")
        print(f"    Type: {model['type']}")
        print(f"    Task: {model['task']}")
        print(f"    Description: {model['description']}")
        print(f"    Total layers: {model['num_layers']} transformer blocks")
        print(f"    Default analysis: {', '.join(model['default_layers'])}")
        print()

    print("Usage:")
    print("  todacomm run --model gpt2                    # Analyze default layers")
    print("  todacomm run --model gpt2 --layers all       # Analyze ALL layers")
    print("  todacomm run --model bert --samples 500")
    print()

    return 0


def cmd_list_configs(args: argparse.Namespace) -> int:
    """List available configuration files."""
    configs_dir = Path("configs")

    print("\n" + "=" * 60)
    print("Available Configuration Files")
    print("=" * 60 + "\n")

    if not configs_dir.exists():
        print("  No configs directory found.")
        print("  Run 'todacomm init <name>' to create one.")
        return 0

    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    if not yaml_files:
        print("  No configuration files found in configs/")
        print("  Run 'todacomm init <name>' to create one.")
        return 0

    for config_file in sorted(yaml_files):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            name = config.get("experiment_name", "unnamed")
            desc = config.get("description", "No description")
            model = config.get("model", {}).get("name", "N/A")
            print(f"  {config_file}")
            print(f"    Experiment: {name}")
            print(f"    Model: {model}")
            print(f"    Description: {desc[:50]}...")
            print()
        except Exception as e:
            print(f"  {config_file} (error reading: {e})")

    print("Usage:")
    print("  todacomm run --config configs/<filename>.yaml")
    print()

    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Create a new configuration file."""
    filename = args.filename
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"

    filepath = Path("configs") / filename
    filepath.parent.mkdir(exist_ok=True)

    if filepath.exists():
        print(f"Error: {filepath} already exists.")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != "y":
            return 1

    model_info = SUPPORTED_MODELS[args.model]
    experiment_name = filepath.stem

    config = {
        "experiment_name": experiment_name,
        "experiment_type": "quick_test",
        "description": f"TDA analysis of {model_info['description']}",
        "model": {
            "type": model_info["type"],
            "name": model_info["name"],
            "task": model_info["task"]
        },
        "dataset": {
            "name": "wikitext2",
            "task": "lm",
            "tokenizer": model_info["tokenizer"],
            "max_length": 128,
            "num_samples": args.samples,
            "batch_size": 8
        },
        "analysis_layers": model_info["default_layers"],
        "tda": {
            "maxdim": 1,
            "metric": "euclidean",
            "pca_components": 50,
            "sampling_strategy": "uniform",
            "max_points": 500
        },
        "extraction": {
            "max_samples": args.samples,
            "pool_strategy": get_pool_strategy(model_info["type"]),
            "device": "cpu"
        },
        "output": {
            "generate_report": True,
            "save_artifacts": True
        },
        "device": "cpu",
        "random_seed": 42
    }

    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Created configuration file: {filepath}")
    print(f"\nRun your experiment with:")
    print(f"  todacomm run --config {filepath}")
    print()

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Generate meta-analysis comparing existing model results."""
    from todacomm.analysis.meta_analysis import run_meta_analysis_cli

    model_names = [m.strip() for m in args.models.split(",")]

    print(f"\n{'=' * 60}")
    print("ToDACoMM - Meta-Analysis")
    print(f"{'=' * 60}")
    print(f"\nComparing models: {', '.join(model_names)}")

    output_path = run_meta_analysis_cli(model_names, args.output)

    print(f"\n{'=' * 60}")
    print(f"Meta-analysis complete: {output_path}")
    print(f"{'=' * 60}\n")

    return 0


def run_single_model(
    model_key: str,
    args: argparse.Namespace,
    use_all_layers: bool = False,
    dataset_override: Optional[str] = None
) -> Optional[str]:
    """
    Run TDA analysis on a single model.

    Args:
        model_key: Model key from SUPPORTED_MODELS
        args: Parsed command line arguments
        use_all_layers: Whether to analyze all layers
        dataset_override: Override the dataset from args

    Returns:
        Experiment directory path on success, None on failure
    """
    from pipeline.unified_pipeline import run_experiment

    model_info = SUPPORTED_MODELS[model_key]
    model_name = model_info["name"]
    model_type = model_info["type"]
    tokenizer = model_info["tokenizer"]
    num_layers = model_info["num_layers"]
    description = model_info["description"]
    default_layers = model_info["default_layers"]

    # Use override dataset or fall back to args
    dataset = dataset_override or args.dataset

    # Determine layers to analyze
    if use_all_layers or (args.layers and args.layers.lower() == "all"):
        layers = get_all_layers(model_key)
    elif args.layers:
        layers = [l.strip() for l in args.layers.split(",")]
    else:
        layers = default_layers

    # Include dataset in experiment name if using multiple datasets
    if dataset_override:
        experiment_name = f"{model_key}_{dataset}_tda"
    else:
        experiment_name = f"{model_key}_tda"

    config = {
        "experiment_name": experiment_name,
        "experiment_type": "quick_test",
        "description": f"Quick TDA analysis of {description} on {dataset}",
        "model": {
            "type": model_type,
            "name": model_name,
            "task": "lm"
        },
        "dataset": {
            "name": dataset,
            "task": "lm" if dataset == "wikitext2" else "qa",
            "tokenizer": tokenizer,
            "max_length": args.max_length,
            "num_samples": args.samples,
            "batch_size": 8
        },
        "analysis_layers": layers,
        "tda": {
            "maxdim": 1,
            "metric": "euclidean",
            "pca_components": args.pca,
            "sampling_strategy": "uniform",
            "max_points": 500
        },
        "extraction": {
            "max_samples": args.samples,
            "pool_strategy": get_pool_strategy(model_type),
            "device": args.device
        },
        "output": {
            "generate_report": True,
            "save_artifacts": True
        },
        "device": args.device,
        "random_seed": 42
    }

    # Save temporary config
    temp_config_path = Path("configs") / f".temp_{experiment_name}.yaml"
    temp_config_path.parent.mkdir(exist_ok=True)

    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'-' * 60}")
    print(f"Model: {model_name} ({description})")
    print(f"Dataset: {dataset}")
    print(f"Layers: {len(layers)} layers")
    print(f"{'-' * 60}")

    try:
        run_experiment(str(temp_config_path))
        return experiment_name
    except Exception as e:
        print(f"Error analyzing {model_key}: {e}")
        return None
    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()


def cmd_run(args: argparse.Namespace) -> int:
    """Run the TDA analysis pipeline."""
    # Import here to avoid slow startup
    from pipeline.unified_pipeline import run_experiment

    if args.config:
        # Run with config file
        config_path = args.config
        if not Path(config_path).exists():
            print(f"Error: Config file not found: {config_path}")
            return 1

        print(f"\nRunning pipeline with config: {config_path}\n")
        run_experiment(config_path)
        return 0

    elif hasattr(args, 'models') and args.models:
        # Multi-model analysis
        model_names = [m.strip() for m in args.models.split(",")]

        # Validate all models
        invalid_models = [m for m in model_names if m not in SUPPORTED_MODELS]
        if invalid_models:
            print(f"Error: Unknown model(s): {', '.join(invalid_models)}")
            print(f"Run 'todacomm list-models' to see supported models.")
            return 1

        print(f"\n{'=' * 60}")
        print("ToDACoMM - Multi-Model Analysis")
        print(f"{'=' * 60}")
        print(f"\nModels to analyze: {', '.join(model_names)}")
        print(f"Samples: {args.samples}")
        print(f"Layers: ALL (default for multi-model analysis)")
        print(f"Device: {args.device}")

        # Run analysis on each model (default to all layers)
        successful_models = []
        for i, model_key in enumerate(model_names, 1):
            print(f"\n[{i}/{len(model_names)}] Analyzing {model_key}...")
            result = run_single_model(model_key, args, use_all_layers=True)
            if result:
                successful_models.append(model_key)

        # Generate meta-analysis
        if len(successful_models) > 1:
            from todacomm.analysis.meta_analysis import run_meta_analysis_cli

            print(f"\n{'=' * 60}")
            print("Generating Meta-Analysis")
            print(f"{'=' * 60}")

            output_name = args.output or "multi_model_comparison"
            output_path = run_meta_analysis_cli(successful_models, output_name)

            print(f"\n{'=' * 60}")
            print("Multi-Model Analysis Complete!")
            print(f"{'=' * 60}")
            print(f"\nModels analyzed: {len(successful_models)}/{len(model_names)}")
            print(f"Meta-analysis: {output_path}")
        else:
            print(f"\nCompleted {len(successful_models)} model(s).")

        return 0

    elif hasattr(args, 'datasets') and args.datasets and args.model:
        # Multi-dataset analysis for a single model
        datasets = [d.strip() for d in args.datasets.split(",")]

        # Validate datasets
        valid_datasets = {"wikitext2", "squad"}
        invalid_datasets = [d for d in datasets if d not in valid_datasets]
        if invalid_datasets:
            print(f"Error: Unknown dataset(s): {', '.join(invalid_datasets)}")
            print(f"Supported datasets: {', '.join(valid_datasets)}")
            return 1

        print(f"\n{'=' * 60}")
        print("ToDACoMM - Multi-Dataset Analysis")
        print(f"{'=' * 60}")
        print(f"\nModel: {args.model}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Samples: {args.samples}")
        print(f"Layers: ALL (default for multi-dataset analysis)")
        print(f"Device: {args.device}")

        # Run analysis on each dataset (default to all layers)
        successful_datasets = []
        for i, dataset in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Analyzing on {dataset}...")
            result = run_single_model(args.model, args, use_all_layers=True, dataset_override=dataset)
            if result:
                successful_datasets.append(dataset)

        # Generate dataset comparison
        if len(successful_datasets) > 1:
            from todacomm.analysis.dataset_comparison import run_dataset_comparison_cli

            print(f"\n{'=' * 60}")
            print("Generating Dataset Comparison")
            print(f"{'=' * 60}")

            output_name = args.output or f"{args.model}_datasets"
            output_path = run_dataset_comparison_cli(args.model, successful_datasets, output_name)

            print(f"\n{'=' * 60}")
            print("Multi-Dataset Analysis Complete!")
            print(f"{'=' * 60}")
            print(f"\nDatasets analyzed: {len(successful_datasets)}/{len(datasets)}")
            print(f"Comparison report: {output_path}")
        else:
            print(f"\nCompleted {len(successful_datasets)} dataset(s).")

        return 0

    elif args.model or args.hf_model:
        # Quick run - create temporary config
        if args.model:
            # Use preset model
            model_info = SUPPORTED_MODELS[args.model]
            model_name = model_info["name"]
            model_type = model_info["type"]
            tokenizer = model_info["tokenizer"]
            num_layers = model_info["num_layers"]
            description = model_info["description"]
            default_layers = model_info["default_layers"]
        else:
            # Custom HuggingFace model
            model_name = args.hf_model
            model_type = "custom"
            tokenizer = args.hf_model
            num_layers = args.num_layers or 12  # Default, will be auto-detected
            description = f"Custom model: {args.hf_model}"
            # Generate default layers based on num_layers
            if num_layers <= 6:
                default_layers = ["embedding", "layer_0", f"layer_{num_layers//2}", f"layer_{num_layers-1}", "final"]
            elif num_layers <= 12:
                default_layers = ["embedding", "layer_0", f"layer_{num_layers//2}", f"layer_{num_layers-1}", "final"]
            else:
                default_layers = ["embedding", "layer_0", f"layer_{num_layers//2}", f"layer_{num_layers-1}", "final"]

        # Parse layers if provided
        if args.layers:
            if args.layers.lower() == "all":
                if args.model:
                    layers = get_all_layers(args.model)
                else:
                    # Generate all layers for custom model
                    layers = ["embedding"] + [f"layer_{i}" for i in range(num_layers)] + ["final"]
            else:
                layers = [l.strip() for l in args.layers.split(",")]
        else:
            layers = default_layers

        experiment_name = args.output or f"{(args.model or args.hf_model.split('/')[-1])}_tda"

        config = {
            "experiment_name": experiment_name,
            "experiment_type": "quick_test",
            "description": f"Quick TDA analysis of {description}",
            "model": {
                "type": model_type,
                "name": model_name,
                "task": "lm"
            },
            "dataset": {
                "name": args.dataset,
                "task": "lm" if args.dataset == "wikitext2" else "qa",
                "tokenizer": tokenizer,
                "max_length": args.max_length,
                "num_samples": args.samples,
                "batch_size": 8
            },
            "analysis_layers": layers,
            "tda": {
                "maxdim": 1,
                "metric": "euclidean",
                "pca_components": args.pca,
                "sampling_strategy": "uniform",
                "max_points": 500
            },
            "extraction": {
                "max_samples": args.samples,
                "pool_strategy": get_pool_strategy(model_type),
                "device": args.device
            },
            "output": {
                "generate_report": True,
                "save_artifacts": True
            },
            "device": args.device,
            "random_seed": 42
        }

        # Save temporary config
        temp_config_path = Path("configs") / f".temp_{experiment_name}.yaml"
        temp_config_path.parent.mkdir(exist_ok=True)

        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n{'=' * 60}")
        print(f"ToDACoMM - Quick Run")
        print(f"{'=' * 60}")
        print(f"\nModel: {model_name} ({description})")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {args.samples}")
        print(f"Layers: {', '.join(layers)}")
        print(f"Device: {args.device}")
        print()

        try:
            run_experiment(str(temp_config_path))
        finally:
            # Clean up temp config
            if temp_config_path.exists():
                temp_config_path.unlink()

        return 0

    else:
        print("Error: Must specify --config, --model, or --hf-model")
        print("\nUsage:")
        print("  todacomm run --config configs/gpt2_demo.yaml")
        print("  todacomm run --model gpt2 --samples 200")
        print("  todacomm run --hf-model microsoft/phi-1_5 --num-layers 24")
        return 1


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "compare": cmd_compare,
        "list-models": cmd_list_models,
        "list-configs": cmd_list_configs,
        "init": cmd_init,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
