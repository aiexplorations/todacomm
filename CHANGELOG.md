# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-31

### Added

- Initial release of ToDACoMM (Topological Data Analysis Comparison of Multiple Models)
- Core TDA analysis using persistent homology (Ripser)
  - H0 (connected components) and H1 (loops/cycles) computation
  - Expansion ratio metrics
  - Layer-by-layer topological characterization
- Support for 15+ transformer models:
  - GPT-2 family (gpt2, distilgpt2)
  - BERT family (bert, distilbert)
  - Pythia family (70m, 160m, 410m)
  - SmolLM2 family (135m, 360m)
  - Qwen2/2.5 family (0.5b variants)
  - OPT family (125m, 350m)
- MLP model support with configurable architectures
- CLI interface (`todacomm` command):
  - `run` - Single/multi-model TDA analysis
  - `compare` - Meta-analysis across experiments
  - `list-models` - Show supported models
  - `init` - Generate configuration files
- Visualization suite:
  - 6-panel TDA summary plots
  - Layer persistence diagrams
  - Betti curve visualizations
  - Multi-model comparison charts
- Analysis modules:
  - Automated interpretation with key findings
  - Meta-analysis for model comparison
  - Dataset comparison reports
  - Geometry characterization (intrinsic dimension, PCA analysis)
- WikiText-2 and SQuAD dataset support
- Comprehensive test suite (82% coverage on core modules)

### Technical Details

- Built on PyTorch and HuggingFace Transformers
- Uses Ripser for efficient persistent homology computation
- PCA dimensionality reduction (default: 50 components)
- Support for CPU, CUDA, and MPS devices

[0.1.0]: https://github.com/aiexplorations/todacomm/releases/tag/v0.1.0
