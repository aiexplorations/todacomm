from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class CorrelationResult:
    correlations: pd.DataFrame
    skipped_reason: Optional[str] = None


def correlate_tda_with_metrics(
    tda_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    on: str = "run_id",
    verbose: bool = True
) -> CorrelationResult:
    """
    Compute correlations between TDA features and performance metrics.

    Note: For single-model runs, metrics are constant across layers,
    making within-model correlation meaningless. Use meta-analysis
    for cross-model comparisons.
    """

    if verbose:
        print(f"TDA columns: {tda_df.columns.tolist()}")
        print(f"Metrics columns: {metrics_df.columns.tolist()}")

    # Ensure the merge column exists
    if on not in tda_df.columns:
        raise ValueError(f"'{on}' not found in TDA dataframe. Available: {tda_df.columns.tolist()}")
    if on not in metrics_df.columns:
        raise ValueError(f"'{on}' not found in metrics dataframe. Available: {metrics_df.columns.tolist()}")

    # Merge dataframes
    df = tda_df.merge(metrics_df, on=on, how='inner', suffixes=("_tda", "_metrics"))
    if verbose:
        print(f"Merged dataframe shape: {df.shape}")
        print(f"Merged columns: {df.columns.tolist()}")

    if df.empty:
        raise ValueError("No matching rows found after merge")

    # Find TDA and metric columns more robustly
    tda_cols = [c for c in df.columns if c.startswith("H") and c.endswith(("_count", "_persistence", "_lifetime"))]
    # Look for common metric column names
    known_metrics = {"perplexity", "accuracy", "loss", "f1", "precision", "recall", "bleu", "rouge"}
    metric_cols = [c for c in df.columns if c.endswith(("_acc", "_loss", "_accuracy")) or c.lower() in known_metrics]

    if verbose:
        print(f"TDA columns found: {tda_cols}")
        print(f"Metric columns found: {metric_cols}")

    if not tda_cols:
        raise ValueError("No TDA columns found with expected patterns (H*_count, H*_persistence, H*_lifetime)")
    if not metric_cols:
        raise ValueError("No metric columns found with expected patterns (*_acc, *_loss, *_accuracy)")

    # Check if metrics have variance (single-run = no variance)
    metrics_have_variance = False
    for mcol in metric_cols:
        if df[mcol].nunique() > 1:
            metrics_have_variance = True
            break

    if not metrics_have_variance:
        # Single-run case: metrics are constant across layers
        # Skip correlation computation as it's not meaningful
        if verbose:
            print("Note: Metrics are constant (single model run). Skipping within-model correlation.")
            print("      Use 'todacomm compare' for cross-model TDA-performance analysis.")

        # Return empty result with explanation
        return CorrelationResult(
            correlations=pd.DataFrame(columns=[
                "tda_feature", "performance_metric", "spearman_rho", "p_value", "n_samples"
            ]),
            skipped_reason="Metrics constant within single model run. Use meta-analysis for cross-model comparison."
        )

    rows = []
    for tcol in tda_cols:
        for mcol in metric_cols:
            # Skip if either column has all NaN values
            if df[tcol].isna().all() or df[mcol].isna().all():
                if verbose:
                    print(f"Skipping {tcol} vs {mcol}: contains all NaN values")
                continue

            # Skip if no variance in the metric column
            if df[mcol].nunique() <= 1:
                continue

            # Compute correlation (suppress ConstantInputWarning)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", message=".*constant.*")
                    rho, p = spearmanr(df[tcol], df[mcol], nan_policy='omit')

                # Skip NaN results
                if np.isnan(rho):
                    continue

                rows.append({
                    "tda_feature": tcol,
                    "performance_metric": mcol,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n_samples": len(df.dropna(subset=[tcol, mcol]))
                })
                if verbose:
                    print(f"Correlation: {tcol} vs {mcol}: ρ={rho:.3f}, p={p:.3f}, n={len(df.dropna(subset=[tcol, mcol]))}")
            except Exception as e:
                if verbose:
                    print(f"Error computing correlation {tcol} vs {mcol}: {e}")
                continue

    if not rows:
        return CorrelationResult(
            correlations=pd.DataFrame(columns=[
                "tda_feature", "performance_metric", "spearman_rho", "p_value", "n_samples"
            ]),
            skipped_reason="No valid correlations computed (insufficient variance in data)."
        )

    result_df = pd.DataFrame(rows).sort_values(by="spearman_rho", key=abs, ascending=False)
    return CorrelationResult(correlations=result_df)
