# ToDACoMM - Quick Start Guide

Get started with topological analysis of transformer representations.

## What ToDACoMM Does

ToDACoMM is a **descriptive framework** for characterizing how different transformer architectures transform representations geometrically. Using persistent homology, it reveals:

- **Architecture-specific topological signatures** (e.g., encoder vs decoder divide)
- **How representations evolve** through transformer layers
- **Comparative patterns** across model families

**Important**: This is a measurement and comparison tool, not a predictive model.

---

## Installation

```bash
# Clone repository
git clone https://github.com/aiexplorations/todacomm.git
cd todacomm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

---

## Quick Test

Run a minimal experiment to verify everything works:

```bash
todacomm run --model gpt2 --samples 100
```

This will:
1. Load GPT-2 and extract activations from WikiText-2
2. Compute persistent homology (H0/H1) for each layer
3. Generate visualizations and interpretation report

**Output location**: `experiments/gpt2_tda_TIMESTAMP/`

---

## Multi-Model Comparison

Compare topology across different transformer architectures:

```bash
todacomm run --models gpt2,bert,distilgpt2,pythia-70m --samples 500
```

This generates:
- Individual analysis for each model
- Meta-analysis comparing topological signatures
- Visualizations showing architecture differences

---

## Understanding Results

### Output Structure

```
experiments/your_experiment_TIMESTAMP/
├── runs/run_0/
│   ├── tda_summaries.json      # H0/H1 metrics per layer
│   ├── metrics.json            # Model performance (perplexity)
│   ├── tda_interpretation.md   # Human-readable analysis
│   └── visualizations/
│       ├── tda_summary.png     # 6-panel overview
│       ├── layer_persistence.png
│       └── betti_curves.png
├── artifacts/
│   └── experiment_data.csv     # Combined results
└── reports/
    └── experiment_report.md    # Full report
```

### Key Metrics

| Metric | What It Measures |
|--------|------------------|
| **H0 Total Persistence** | Cluster separation across scales |
| **H0 Max Lifetime** | Most persistent cluster structure |
| **H1 Total Persistence** | Cyclic/loop structure strength |
| **H1 Count** | Number of cycles detected |
| **Expansion Ratio** | Peak H0 / Embedding H0 |

### Interpreting Results

These metrics provide **descriptive comparisons**:

| You CAN observe | You CANNOT conclude |
|-----------------|---------------------|
| Topology differs between architectures | Why one model outperforms another |
| Topology evolves across layers | Causal mechanisms of behavior |
| Consistent patterns within model families | Predictions about new models |
| Encoder-decoder topological divide | That topology causes performance |

---

## Custom Experiments

### Using CLI Options

```bash
# Full analysis (500 samples, all layers)
todacomm run --model gpt2 --samples 500 --layers all

# Use GPU acceleration
todacomm run --model gpt2 --device mps  # or cuda

# Custom HuggingFace model
todacomm run --hf-model microsoft/phi-1_5 --num-layers 24 --samples 200
```

### Using Configuration Files

Create a YAML configuration:

```yaml
experiment_name: "my_experiment"
experiment_type: "quick_test"

model:
  type: "gpt2"
  name: "gpt2"
  task: "lm"

dataset:
  name: "wikitext2"
  num_samples: 500
  batch_size: 8

analysis_layers: ["embedding", "layer_3", "layer_6", "final"]

tda:
  maxdim: 1
  pca_components: 50
  max_points: 1000
```

Save as `configs/my_experiment.yaml` and run:

```bash
python -m pipeline.unified_pipeline --config configs/my_experiment.yaml
```

---

## Running Tests

```bash
# Fast tests only (~45s)
pytest tests/ -v -m "not slow"

# All tests including model loading (~6min)
pytest tests/ -v --run-slow

# With coverage
pytest --cov=todacomm --cov-report=html
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory** | Reduce `--samples` or `--pca` value |
| **Slow execution** | Use smaller model (`distilgpt2`) or fewer samples |
| **Import errors** | Ensure venv is activated: `source .venv/bin/activate` |
| **CUDA not found** | Use `--device cpu` or `--device mps` for Apple Silicon |

---

## Key Findings Reference

From our analysis of 10 models:

| Architecture | Expansion Range | Interpretation |
|--------------|-----------------|----------------|
| **BERT (encoder)** | 2x | Bidirectional attention captures structure early |
| **GPT-2 (decoder)** | 55-95x | Progressive buildup through causal attention |
| **SmolLM2** | 298-694x | Extreme geometric transformation |

See `experiments/technical_report.md` for full analysis.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/aiexplorations/todacomm/issues)
- **Email**: rexplorations@gmail.com
