# ToDACoMM

**Topological Data Analysis Comparison of Multiple Models**

A framework for characterizing and comparing the topological signatures of language model representations using persistent homology.

## Core Contribution

ToDACoMM provides **systematic, reproducible characterization** of how different transformer architectures transform representations geometrically. The contribution is *descriptive and comparative*, not predictive.

**What we found** (analyzing 10 models with 500 samples each):

| Finding | Observation |
|---------|-------------|
| **Encoder-Decoder Divide** | BERT shows 2x expansion; decoders show 55-694x |
| **Architecture Fingerprints** | Model families have consistent topological signatures |
| **H1 Universality** | All models show cyclic structure in all layers at scale |

### What This Is

- A **measurement tool** for topological properties of neural network activations
- A **comparative framework** revealing architecture-specific signatures
- A **descriptive analysis** of how representations evolve through layers
- **Reproducible methodology** using standard persistent homology (Ripser)

### What This Is NOT

| Not This | Why |
|----------|-----|
| A predictive model | We cannot predict perplexity from topology |
| A novel TDA method | We use standard persistent homology |
| A causal theory | Topology describes geometry, doesn't explain behavior |
| A benchmark | N=10 models is insufficient for statistical claims |

### One-Sentence Summary

> ToDACoMM demonstrates that persistent homology reveals consistent, architecture-specific topological signatures in language models—most notably the stark encoder-decoder divide—providing a new descriptive lens for understanding representation geometry.

---

## Key Findings

From our analysis of 10 models across 5 architecture families:

### The Encoder-Decoder Topological Divide

| Architecture | Model | Expansion Ratio | Interpretation |
|--------------|-------|-----------------|----------------|
| **Encoder** | BERT | 2x | Bidirectional attention captures structure early |
| **Decoder** | DistilGPT-2 | 55x | Progressive buildup through causal attention |
| **Decoder** | GPT-2 | 95x | |
| **Decoder** | SmolLM2-360M | 694x | Extreme geometric transformation |

### Architecture Family Signatures

| Family | Expansion Range | H1 Characteristics |
|--------|-----------------|-------------------|
| GPT-2 | 55-95x | Moderate, correlates with perplexity |
| Pythia | 143-189x | Stable, scales with model size |
| Qwen | 629-673x | Consistent across variants |
| SmolLM2 | 298-694x | Extreme expansion and H1 |

See `experiments/technical_report.md` for full analysis and theoretical framework.

---

## Quick Start

### Installation

```bash
git clone https://github.com/aiexplorations/todacomm.git
cd todacomm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"  # Include dev dependencies for testing
```

### Verify Installation

```bash
# Check CLI is available
todacomm --help

# List supported models
todacomm list-models

# Run a quick test (no model download needed)
pytest tests/ -v -x --ignore=tests/test_transformer_extraction.py -k "not slow"
```

### Run Your First Analysis

```bash
# Quick analysis with GPT-2
todacomm run --model gpt2 --samples 200

# Full analysis (500 samples, all layers)
todacomm run --model gpt2 --samples 500 --layers all

# Compare multiple models
todacomm run --models gpt2,bert,pythia-70m --samples 500
```

This will:
1. Load the model and extract activations from WikiText-2
2. Compute persistent homology (H0/H1) for each layer
3. Generate visualizations and interpretation report
4. (For multi-model) Generate comparative meta-analysis

---

## TDA Metrics Computed

| Metric | What It Measures | Interpretation |
|--------|------------------|----------------|
| **H0 Total Persistence** | Cluster separation across scales | Higher = more spread-out representations |
| **H0 Max Lifetime** | Most persistent cluster | Dominant structure in layer |
| **H1 Total Persistence** | Cyclic structure strength | Higher = more loop/hole topology |
| **H1 Count** | Number of cycles | Complexity of cyclic patterns |
| **Expansion Ratio** | Peak H0 / Embedding H0 | Geometric transformation magnitude |

### What These Metrics Reveal

- **H0 (Connected Components)**: How representations cluster and separate through layers
- **H1 (Loops/Cycles)**: Cyclic dependencies in the representation manifold
- **Expansion Ratio**: How dramatically the model transforms input geometry

### What These Metrics Do NOT Reveal

- Causal mechanisms of model behavior
- Predictive relationship to downstream performance
- Why one architecture outperforms another

---

## CLI Reference

```bash
todacomm <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `run` | Run TDA analysis (single or multiple models) |
| `compare` | Generate meta-analysis from existing results |
| `list-models` | Show supported models |
| `init` | Create a new configuration file |

### Examples

```bash
# Single model analysis
todacomm run --model gpt2 --samples 500 --layers all

# Multi-model comparison (generates meta-analysis)
todacomm run --models gpt2,bert,distilgpt2,pythia-70m --samples 500

# Multi-dataset comparison
todacomm run --model gpt2 --datasets wikitext2,squad --samples 500

# Use GPU
todacomm run --model gpt2 --device mps  # or cuda

# Custom HuggingFace model
todacomm run --hf-model microsoft/phi-1_5 --num-layers 24 --samples 200
```

### Options

```
-m, --model MODEL       Preset model (gpt2, bert, pythia-70m, etc.)
-n, --samples N         Number of samples (recommend 500 for robust H1)
-l, --layers LAYERS     'all' or comma-separated list
-d, --dataset DATASET   wikitext2 or squad
-o, --output NAME       Experiment name
--device DEVICE         cpu, cuda, or mps
--pca N                 PCA components (default: 50)
```

---

## Supported Models

### Preset Models (<1B parameters)

| Family | Models | Parameters |
|--------|--------|------------|
| GPT-2 | `gpt2`, `distilgpt2` | 117M, 82M |
| BERT | `bert`, `distilbert` | 110M, 66M |
| Pythia | `pythia-70m`, `pythia-160m`, `pythia-410m` | 70M-410M |
| SmolLM2 | `smollm2-135m`, `smollm2-360m` | 135M, 360M |
| Qwen2/2.5 | `qwen2-0.5b`, `qwen2.5-0.5b`, `qwen2.5-coder-0.5b` | 500M |
| OPT | `opt-125m`, `opt-350m` | 125M, 350M |

### Custom Models

Any HuggingFace causal language model:
```bash
todacomm run --hf-model <model-name> --num-layers <N>
```

---

## Output Structure

```
experiments/<model>_tda_<timestamp>/
├── runs/run_0/
│   ├── tda_summaries.json       # H0/H1 metrics per layer
│   ├── metrics.json             # Model performance (perplexity)
│   ├── tda_interpretation.md    # Human-readable analysis
│   └── visualizations/
│       ├── tda_summary.png      # 6-panel overview
│       ├── layer_persistence.png # H0/H1 comparison
│       └── betti_curves.png     # Feature evolution
├── artifacts/
│   └── experiment_data.csv      # Combined results
└── reports/
    └── experiment_report.md     # Full report
```

---

## Methodology

### Pipeline

```
Input Text → Tokenize → Extract Activations → Pool Sequences → PCA → Ripser → Metrics
```

### Key Methodological Choices

| Step | Choice | Rationale |
|------|--------|-----------|
| **Pooling** | Last token (decoders), CLS (encoders) | Architecture-appropriate aggregation |
| **Dimensionality** | PCA to 50 components | Enables efficient homology computation |
| **Homology** | Vietoris-Rips via Ripser | Standard, efficient persistent homology |
| **Sample Size** | 500 recommended | Statistical stability for H1 detection |

See `experiments/technical_report.md` Section 3-5 for detailed methodology and theoretical grounding.

---

## Theoretical Framework

The framework connects TDA metrics to representation theory:

- **Superposition Hypothesis**: Models encode more features than dimensions using near-orthogonal directions. Expansion ratio may reflect encoding efficiency.
- **Linear Representation Hypothesis**: Features are linear directions in activation space. H0 persistence measures how spread these directions become.
- **Intrinsic Dimension Dynamics**: Representations expand then compress through layers. Our H0 trajectory tracks this pattern.

See `experiments/technical_report.md` Section 3 for full theoretical framework with references.

---

## Limitations

1. **Descriptive, not predictive**: Topology describes geometry but doesn't explain or predict performance
2. **Sample size**: 10 models is insufficient for statistical generalization
3. **Single dataset**: WikiText-2 results may not generalize
4. **Correlation ≠ causation**: Patterns don't imply mechanisms

---

## Future Directions

The framework enables research questions like:

- Does topology change during training or fine-tuning?
- Do different tasks (QA, classification) induce different signatures?
- Can topology detect model drift or degradation?
- How do larger models (1B+) compare?

---

## Running Tests

```bash
# Fast unit tests (~50s, no model downloads)
pytest tests/ -v

# Integration tests with real models (~6min, downloads models)
pytest tests/ -v --run-slow

# With coverage report
pytest tests/ --cov=todacomm --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=todacomm --cov-report=html
open htmlcov/index.html
```

### Current Test Coverage

| Module | Coverage |
|--------|----------|
| `cli.py` | 97% |
| `tda/persistence.py` | 97% |
| `visualization/tda_plots.py` | 99% |
| `analysis/interpretation.py` | 99% |
| `analysis/meta_analysis.py` | 95% |
| `analysis/dataset_comparison.py` | 85% |
| `analysis/correlation.py` | 85% |
| **Overall** | **82%** |

Note: Lower coverage in `models/transformer.py` and `extract/` modules is expected—these require model downloads which are skipped in fast tests.

---

## Project Structure

```
todacomm/
├── todacomm/              # Core library
│   ├── models/            # Transformer wrappers
│   ├── tda/               # Persistent homology (Ripser)
│   ├── extract/           # Activation extraction
│   ├── analysis/          # Interpretation, meta-analysis
│   └── visualization/     # TDA plotting
├── configs/               # Experiment configurations
├── experiments/           # Output directory
│   └── technical_report.md  # Full analysis report
└── tests/                 # Test suite
```

---

## Citation

```bibtex
@software{sampathkumar2025todacomm,
  title={ToDACoMM: Topological Data Analysis Comparison of Multiple Models},
  author={Sampathkumar, Rajesh},
  year={2025},
  url={https://github.com/aiexplorations/todacomm}
}
```

---

## License

MIT

## Contact

- **Issues**: GitHub Issues
- **Email**: rexplorations@gmail.com
