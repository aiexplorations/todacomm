# ToDACoMM + deep-lyapunov Integration Plan

## Overview

This plan outlines the integration of [deep-lyapunov](https://pypi.org/project/deep-lyapunov/) with ToDACoMM to provide a combined framework for analyzing both the **geometric structure** (topology) and **dynamical stability** (Lyapunov exponents) of neural network representations.

## Conceptual Framework

| Framework | Measures | Perspective |
|-----------|----------|-------------|
| **ToDACoMM** | H0/H1 persistence, expansion ratio | Static geometry of learned representations |
| **deep-lyapunov** | Lyapunov exponents, convergence ratio | Dynamic sensitivity to training perturbations |

### Why Combine Them?

1. **Complementary Views**: Topology describes *what* the representation looks like; Lyapunov describes *how stable* it is to reach that state
2. **Richer Architecture Fingerprints**: Extend ToDACoMM's architecture signatures with stability profiles
3. **Training Insights**: Correlate topological development with training stability
4. **Model Selection**: Combined metrics for informed architecture decisions

## Implementation Plan

### Phase 1: Add deep-lyapunov as Optional Dependency

**Files to modify:**
- `pyproject.toml`

```toml
[project.optional-dependencies]
stability = [
    "deep-lyapunov>=0.1.0",
]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "jupyter>=1.0.0",
    "deep-lyapunov>=0.1.0",  # Include in dev for testing
]
```

**Installation:**
```bash
pip install todacomm[stability]
```

### Phase 2: Create Stability Analysis Module

**New file:** `todacomm/analysis/stability.py`

```python
"""
Training stability analysis using Lyapunov exponents.

Integrates deep-lyapunov for measuring sensitivity to initial conditions
during neural network training.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

try:
    from deep_lyapunov import LyapunovAnalyzer
    DEEP_LYAPUNOV_AVAILABLE = True
except ImportError:
    DEEP_LYAPUNOV_AVAILABLE = False


@dataclass
class StabilityResult:
    """Results from Lyapunov stability analysis."""
    model_name: str
    lyapunov_exponent: float  # Negative = stable, Positive = chaotic
    convergence_ratio: float  # < 1 = convergent, > 1 = divergent
    effective_dimensionality: float
    trajectory_spread: Dict[str, float]  # initial, final spreads
    interpretation: str


def check_stability_available() -> bool:
    """Check if deep-lyapunov is installed."""
    return DEEP_LYAPUNOV_AVAILABLE


def analyze_training_stability(
    model,
    train_loader,
    num_perturbations: int = 5,
    perturbation_scale: float = 1e-4,
    training_steps: int = 100,
    device: str = "cpu",
) -> StabilityResult:
    """
    Analyze training stability using Lyapunov exponents.

    Args:
        model: PyTorch model to analyze
        train_loader: DataLoader for training data
        num_perturbations: Number of perturbed copies to track
        perturbation_scale: Scale of initial weight perturbations
        training_steps: Number of training steps to track
        device: Device to run on

    Returns:
        StabilityResult with Lyapunov metrics
    """
    if not DEEP_LYAPUNOV_AVAILABLE:
        raise ImportError(
            "deep-lyapunov is required for stability analysis. "
            "Install with: pip install todacomm[stability]"
        )

    # Implementation using deep-lyapunov API
    # ... (to be implemented based on deep-lyapunov's actual API)
    pass


def interpret_stability(lyapunov_exponent: float) -> str:
    """Interpret Lyapunov exponent value."""
    if lyapunov_exponent < -0.1:
        return "Highly stable: training converges reliably"
    elif lyapunov_exponent < 0:
        return "Stable: training shows mild convergence"
    elif lyapunov_exponent < 0.1:
        return "Neutral: training is sensitive to initialization"
    else:
        return "Unstable: training is chaotic, sensitive to initial conditions"
```

### Phase 3: Extend CLI with Stability Commands

**Modify:** `todacomm/cli.py`

Add new CLI command:
```bash
todacomm stability --model gpt2 --training-steps 100
```

Add `--with-stability` flag to `run` command:
```bash
todacomm run --model gpt2 --samples 500 --with-stability
```

### Phase 4: Create Combined Analysis Module

**New file:** `todacomm/analysis/combined.py`

```python
"""
Combined topology + stability analysis.

Provides unified view of geometric structure and training dynamics.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import pandas as pd

from .interpretation import TDAInterpretation
from .stability import StabilityResult


@dataclass
class CombinedAnalysis:
    """Combined topology and stability analysis."""
    model_name: str
    topology: TDAInterpretation
    stability: Optional[StabilityResult]
    correlation_insights: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for comparison."""
        return pd.DataFrame([{
            "model": self.model_name,
            "expansion_ratio": self._get_expansion_ratio(),
            "h1_max": self._get_h1_max(),
            "lyapunov_exponent": self.stability.lyapunov_exponent if self.stability else None,
            "convergence_ratio": self.stability.convergence_ratio if self.stability else None,
        }])


def correlate_topology_stability(
    topology_results: List[TDAInterpretation],
    stability_results: List[StabilityResult],
) -> Dict[str, float]:
    """
    Compute correlations between topology and stability metrics.

    Returns:
        Dict mapping metric pairs to correlation coefficients
    """
    # Correlate expansion_ratio with lyapunov_exponent
    # Correlate h1_persistence with convergence_ratio
    # etc.
    pass
```

### Phase 5: Visualization Extensions

**Modify:** `todacomm/visualization/tda_plots.py`

Add new plotting functions:
- `plot_topology_stability_scatter()`: Scatter plot of expansion ratio vs Lyapunov
- `plot_combined_dashboard()`: Multi-panel dashboard with both metrics
- `plot_training_trajectory()`: Visualize topology evolution during training

### Phase 6: Testing

**New file:** `tests/test_stability.py`

```python
"""Tests for stability analysis integration."""

import pytest
from todacomm.analysis.stability import (
    check_stability_available,
    interpret_stability,
)


class TestStabilityAvailability:
    def test_check_returns_bool(self):
        result = check_stability_available()
        assert isinstance(result, bool)


class TestInterpretStability:
    def test_negative_is_stable(self):
        result = interpret_stability(-0.5)
        assert "stable" in result.lower()

    def test_positive_is_unstable(self):
        result = interpret_stability(0.5)
        assert "unstable" in result.lower() or "chaotic" in result.lower()
```

## Research Questions to Address

### Immediate Questions

1. **Architecture Correlation**: Do architectures with consistent topological signatures (e.g., GPT-2 family) also have consistent Lyapunov profiles?

2. **Expansion-Stability Relationship**: Is there correlation between expansion ratio and Lyapunov exponent?

3. **H1 and Stability**: Do models with stronger H1 (cyclic structure) show different training dynamics?

### Extended Research

4. **Fine-tuning Dynamics**: Track both metrics during fine-tuning to identify phase transitions

5. **Checkpoint Analysis**: At what training checkpoint does topology "lock in"? Does this correlate with stability?

6. **Reproducibility Prediction**: Can combined metrics predict training reproducibility?

## Output Format

### Combined Report

```
=== ToDACoMM + Stability Analysis ===

Model: gpt2
Dataset: wikitext-2

TOPOLOGY METRICS
  Expansion Ratio: 95x
  Peak H0 Layer: layer_8
  H1 Present: Yes (layers 6-11)

STABILITY METRICS
  Lyapunov Exponent: -0.12 (Stable)
  Convergence Ratio: 0.85
  Effective Dimensionality: 42.3

COMBINED INSIGHTS
  - Moderate geometry with stable training
  - H1 emergence coincides with training stabilization
  - Recommended for reproducible deployments

ARCHITECTURE COMPARISON
  Model         | Expansion | Lyapunov | Stability
  gpt2          | 95x       | -0.12    | Stable
  SmolLM2-360M  | 694x      | +0.08    | Neutral
  BERT          | 2x        | -0.31    | Very Stable
```

## Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Add optional dependency | Small |
| 2 | Create stability module | Medium |
| 3 | Extend CLI | Small |
| 4 | Combined analysis | Medium |
| 5 | Visualization | Medium |
| 6 | Testing | Small |

## Dependencies

- `deep-lyapunov>=0.1.0` (optional, for stability analysis)
- Existing todacomm dependencies

## Success Criteria

1. `pip install todacomm[stability]` works
2. `todacomm run --model gpt2 --with-stability` produces combined report
3. Correlation analysis between topology and stability is computable
4. Documentation updated with stability analysis examples
5. Test coverage maintained at 80%+ for new code
