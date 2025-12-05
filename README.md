# TDA Perturbation Analysis for Small Language Models

> **⚠️ DISCLAIMER (December 2025)**
>
> This repository builds on [tda-dnn](https://github.com/aiexplorations/tda-dnn), which contained claims about "H1 Paradox" and "H1 Overfitting Signature" that were later found to be based on **circular reasoning**. The pipeline infrastructure here remains useful for TDA experimentation, but any specific claims about H1-performance relationships from the parent project should not be relied upon. Use this as an exploratory tool, not as validation of particular topological hypotheses.

A comprehensive, reproducible pipeline for analyzing topological properties of transformer-based language models using persistent homology and systematic perturbation studies.

## 🎯 Overview

This pipeline applies **Topological Data Analysis (TDA)** to understand the geometry and topology of learned representations in small language models (10M-500M parameters). By analyzing persistent homology features across different model architectures, training regimes, and TDA configurations, we can discover topology-performance relationships that provide insights into model behavior.

**Key Focus**: Transformer architectures (GPT-2, BERT) with attention mechanism analysis, optimized for Apple Silicon (M4 Pro) using MLX.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/tda_perturbation_analysis.git
cd tda_perturbation_analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Quick test with tiny model (fast validation)
python -m pipeline.unified_pipeline --config configs/test_tiny_lm.yaml

# Architecture perturbation study (GPT-2 variants)
python -m pipeline.unified_pipeline --config configs/architecture_perturbation_lm.yaml

# Training regime perturbation (fine-tuning analysis)
python -m pipeline.unified_pipeline --config configs/training_perturbation_lm.yaml

# TDA methodology robustness testing
python -m pipeline.unified_pipeline --config configs/tda_config_perturbation_lm.yaml
```

### View Results

```bash
# Experiments create timestamped directories:
ls experiments/
# └── architecture_perturbation_lm_20251124_220000/
#     ├── runs/           # Individual model runs
#     ├── reports/        # Analysis reports
#     ├── artifacts/      # Correlation data
#     └── experiment_config.yaml

# View report
open experiments/architecture_perturbation_lm_*/reports/final_report.md
```

## 📊 Supported Models

### Transformer Architectures (Primary Focus)
- **GPT-2**: Small (117M), Medium (345M)
- **BERT**: Base (110M), DistilBERT (66M)
- **Custom Transformers**: Configurable 6-12 layers, 512-768 hidden dims

### Language Tasks
- ✅ **Next Token Prediction**: Perplexity analysis on WikiText-2, Penn Treebank
- ✅ **Question Answering**: Accuracy/F1 on SQuAD 2.0
- ✅ **Reasoning**: Chain-of-thought on GSM8K, ARC

## 🔬 Perturbation Framework

### 1. Architecture Perturbation
Systematic analysis across model configurations:
- Model size: 10M-500M parameters
- Depth: 6-12 transformer layers
- Width: 512-1024 hidden dimensions
- Attention heads: 8-12 heads

### 2. Training Regime Perturbation
Fine-tuning dynamics analysis:
- Epochs: 1-5 (undertrained → overtrained)
- Learning rates: 1e-5 to 5e-5
- Dataset sizes: 1K-10K samples
- Regularization: 0.0-0.1

### 3. TDA Configuration Perturbation
Methodological robustness testing:
- PCA dimensions: 50-200 components
- Sampling strategies: uniform, multiscale, adaptive
- Distance metrics: euclidean, cosine
- Max points: 1K-5K samples

## 🏗 Architecture

```
tda_perturbation_analysis/
├── tda_perturbation/          # Core library
│   ├── models/                # Transformer models
│   ├── tda/                   # TDA computation
│   ├── data/                  # Language datasets
│   ├── extract/               # Activation extraction
│   ├── analysis/              # Correlation analysis
│   └── utils/                 # Utilities
├── pipeline/                  # Unified pipeline
├── configs/                   # Experiment configs
├── tests/                     # Unit & integration tests
├── examples/                  # Jupyter notebooks
└── docs/                      # Documentation
```

## 💻 Hardware Requirements

**Optimized for Apple Silicon (M4 Pro)**:
- MLX framework for efficient inference
- Memory-efficient batching and gradient checkpointing
- PCA reduction for high-dimensional representations
- Works on CPU/MPS backends

**Strategy**: Use pre-trained models + fine-tuning (no expensive training from scratch)

## 📚 Documentation

- **[Language Model TDA Guide](docs/language_model_tda.md)**: Transformer-specific analysis
- **[API Reference](docs/api_reference.md)**: Core library documentation
- **[Examples](examples/)**: Jupyter notebooks with tutorials

## 🤝 Contributing

Contributions welcome! This is an open-source research project.

See `CONTRIBUTING.md` for guidelines.

## 📄 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{sampathkumar2025tdaperturbation,
  title={TDA Perturbation Analysis for Small Language Models},
  author={Sampathkumar, Rajesh},
  year={2025},
  url={https://github.com/your-username/tda_perturbation_analysis}
}
```

## 📞 Contact

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for research questions
- **Email**: rexplorations@gmail.com

---

**Built on the proven architecture of [tda-dnn](https://github.com/aiexplorations/tda-dnn)**
