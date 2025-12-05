from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class CorrelationResult:
    correlations: pd.DataFrame


def correlate_tda_with_metrics(tda_df: pd.DataFrame, metrics_df: pd.DataFrame, on: str = "run_id") -> CorrelationResult:
    """Compute correlations between TDA features and performance metrics"""
    
    # Debug: Print column names to understand the data
    print(f"TDA columns: {tda_df.columns.tolist()}")
    print(f"Metrics columns: {metrics_df.columns.tolist()}")
    
    # Ensure the merge column exists
    if on not in tda_df.columns:
        raise ValueError(f"'{on}' not found in TDA dataframe. Available: {tda_df.columns.tolist()}")
    if on not in metrics_df.columns:
        raise ValueError(f"'{on}' not found in metrics dataframe. Available: {metrics_df.columns.tolist()}")
    
    # Merge dataframes
    df = tda_df.merge(metrics_df, on=on, how='inner', suffixes=("_tda", "_metrics"))
    print(f"Merged dataframe shape: {df.shape}")
    print(f"Merged columns: {df.columns.tolist()}")
    
    if df.empty:
        raise ValueError("No matching rows found after merge")
    
    # Find TDA and metric columns more robustly
    tda_cols = [c for c in df.columns if c.startswith("H") and c.endswith(("_count", "_persistence", "_lifetime"))]
    metric_cols = [c for c in df.columns if c.endswith(("_acc", "_loss", "_accuracy"))]
    
    print(f"TDA columns found: {tda_cols}")
    print(f"Metric columns found: {metric_cols}")
    
    if not tda_cols:
        raise ValueError("No TDA columns found with expected patterns (H*_count, H*_persistence, H*_lifetime)")
    if not metric_cols:
        raise ValueError("No metric columns found with expected patterns (*_acc, *_loss, *_accuracy)")

    rows = []
    for tcol in tda_cols:
        for mcol in metric_cols:
            # Skip if either column has all NaN values
            if df[tcol].isna().all() or df[mcol].isna().all():
                print(f"Skipping {tcol} vs {mcol}: contains all NaN values")
                continue
                
            # Compute correlation
            try:
                rho, p = spearmanr(df[tcol], df[mcol], nan_policy='omit')
                rows.append({
                    "tda_feature": tcol, 
                    "performance_metric": mcol, 
                    "spearman_rho": rho, 
                    "p_value": p,
                    "n_samples": len(df.dropna(subset=[tcol, mcol]))
                })
                print(f"Correlation: {tcol} vs {mcol}: ρ={rho:.3f}, p={p:.3f}, n={len(df.dropna(subset=[tcol, mcol]))}")
            except Exception as e:
                print(f"Error computing correlation {tcol} vs {mcol}: {e}")
                continue
    
    if not rows:
        raise ValueError("No valid correlations could be computed")
    
    result_df = pd.DataFrame(rows).sort_values(by="spearman_rho", key=abs, ascending=False)
    return CorrelationResult(correlations=result_df)
