"""Analysis utilities for TDA-performance correlations and interpretation."""

from .interpretation import (
    interpret_tda_results,
    format_interpretation_markdown,
    generate_metric_glossary,
    TDAInterpretation,
    LayerInsight,
    PatternInsight,
)
from .geometry import (
    GeometryConfig,
    GeometryResult,
    characterize_geometry,
    summarize_geometry,
    recommend_pca_components,
)

__all__ = [
    "correlation",
    "interpret_tda_results",
    "format_interpretation_markdown",
    "generate_metric_glossary",
    "TDAInterpretation",
    "LayerInsight",
    "PatternInsight",
    "geometry",
    "GeometryConfig",
    "GeometryResult",
    "characterize_geometry",
    "summarize_geometry",
    "recommend_pca_components",
]
