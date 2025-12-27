"""Analysis utilities for TDA-performance correlations and interpretation."""

from .interpretation import (
    interpret_tda_results,
    format_interpretation_markdown,
    generate_metric_glossary,
    TDAInterpretation,
    LayerInsight,
    PatternInsight,
)

__all__ = [
    "correlation",
    "interpret_tda_results",
    "format_interpretation_markdown",
    "generate_metric_glossary",
    "TDAInterpretation",
    "LayerInsight",
    "PatternInsight",
]
