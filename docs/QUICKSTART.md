# TDA Perturbation Analysis - Quick Start Guide

This guide will walk you through running your first TDA analysis on a small language model.

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/tda_perturbation_analysis.git
cd tda_perturbation_analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Run a minimal experiment to verify everything works:

```bash
python -m pipeline.unified_pipeline --config configs/test_tiny_lm.yaml
```

This will:
1. Load GPT-2 small model
2. Load WikiText-2 dataset (100 samples)
3. Extract activations from 4 layers
4. Compute TDA features (H0, H1 persistence)
5. Generate analysis report

**Expected output**: `experiments/test_tiny_lm_TIMESTAMP/`

## Architecture Perturbation Study

Compare topology across different transformer architectures:

```bash
python -m pipeline.unified_pipeline --config configs/architecture_perturbation_lm.yaml
```

This compares:
- GPT-2 small (117M params)
- DistilGPT-2 (82M params)
- BERT-base (110M params)
- DistilBERT (66M params)

## Understanding Results

After running an experiment, check:

```
experiments/your_experiment_TIMESTAMP/
├── experiment_config.yaml      # Full configuration
├── runs/                        # Individual model runs
│   └── run_0/
│       ├── activations.npz      # Extracted features
│       ├── tda_summaries.json   # TDA features
│       └── metrics.json         # Performance metrics
├── artifacts/
│   ├── experiment_data.csv      # Combined data
│   └── correlations.csv         # TDA-performance correlations
└── reports/
    └── experiment_report.md     # Analysis summary
```

### Key TDA Features

- **H0_count**: Number of connected components
- **H0_total_persistence**: Total persistence of H0 features
- **H1_count**: Number of loops/holes
- **H1_total_persistence**: Total persistence of H1 features (key metric!)
- **H1_max_lifetime**: Longest-lived loop

### Interpreting Correlations

> **⚠️ Note (December 2025)**: Earlier claims about "H1 overfitting signatures" from the parent tda-dnn project were found to be based on circular reasoning. Interpret correlations cautiously as exploratory findings, not validated theory.

Correlations between H1 features and performance may suggest various relationships, but **causal interpretation requires careful validation**:
- Correlations can arise from confounding factors
- Statistical significance does not imply mechanistic understanding
- Use these as exploratory signals, not definitive conclusions

## Custom Experiments

Create your own configuration:

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
  pca_components: 100
  max_points: 1000
```

Save as `configs/my_experiment.yaml` and run:

```bash
python -m pipeline.unified_pipeline --config configs/my_experiment.yaml
```

## Running Tests

```bash
# Fast tests only
pytest tests/ -v -m "not slow"

# All tests (including slow model loading)
pytest tests/ -v --run-slow

# Specific test file
pytest tests/test_pipeline.py -v
```

## Next Steps

- See `examples/` for Jupyter notebooks
- Read `docs/language_model_tda.md` for detailed analysis guide
- Check `docs/api_reference.md` for API documentation

## Troubleshooting

**Out of memory**: Reduce `num_samples`, `max_points`, or `pca_components` in config

**Slow execution**: Use smaller model (distilgpt2) or reduce `num_samples`

**Import errors**: Ensure virtual environment is activated and dependencies installed

## Support

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Research questions
- Email: rexplorations@gmail.com
