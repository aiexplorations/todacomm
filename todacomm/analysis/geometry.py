"""
Pre-TDA geometry characterization module.

Provides comprehensive geometric analysis of activation point clouds
before applying expensive topological data analysis. This helps:
1. Inform PCA component selection
2. Validate sample size requirements
3. Detect potential issues (high hubness, low intrinsic dim)
4. Understand activation structure before TDA

Key metrics:
- Basic statistics: mean, variance, sparsity, correlation
- Intrinsic dimensionality: MLE, correlation dimension, local PCA
- k-NN structure: distance distributions, hubness
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


@dataclass
class GeometryConfig:
    """
    Configuration for pre-TDA geometry characterization.

    Attributes:
        compute_basic_stats: Compute mean, variance, correlation
        compute_intrinsic_dim: Compute dimensionality estimates
        compute_knn_structure: Compute k-NN distance analysis
        k_neighbors: Number of neighbors for k-NN analysis
        mle_k: k value for MLE intrinsic dimension estimator
        local_pca_k: k for local PCA neighborhoods
        local_pca_samples: Number of random points for local PCA
        correlation_dim_scales: Number of scales for correlation dimension
        variance_threshold: Threshold for local PCA dimension estimate
        max_samples_for_dim: Subsample for expensive dimension estimates
        max_dims_for_correlation: Max dimensions for correlation matrix
        seed: Random seed for reproducibility
    """
    compute_basic_stats: bool = True
    compute_intrinsic_dim: bool = True
    compute_knn_structure: bool = True

    # k-NN parameters
    k_neighbors: int = 20
    distance_metric: str = "euclidean"

    # Intrinsic dimension parameters
    mle_k: int = 10
    local_pca_k: int = 50
    local_pca_samples: int = 100
    correlation_dim_scales: int = 10
    variance_threshold: float = 0.95

    # Efficiency parameters
    max_samples_for_dim: int = 5000
    max_dims_for_correlation: int = 100

    seed: int = 42


@dataclass
class GeometryResult:
    """
    Results from pre-TDA geometry characterization.

    Contains comprehensive metrics for understanding activation geometry
    before applying topological data analysis.
    """
    layer_name: str
    n_samples: int
    n_dims: int

    # Basic statistics
    mean: Optional[np.ndarray] = None
    variance: Optional[np.ndarray] = None
    total_variance: Optional[float] = None
    sparsity: Optional[float] = None
    correlation_matrix: Optional[np.ndarray] = None

    # Intrinsic dimensionality estimates
    mle_intrinsic_dim: Optional[float] = None
    correlation_dim: Optional[float] = None
    local_pca_intrinsic_dim: Optional[float] = None
    pca_explained_variance_ratio: Optional[np.ndarray] = None
    pca_cumulative_variance: Optional[np.ndarray] = None
    pca_dim_for_95_var: Optional[int] = None

    # k-NN structure
    knn_distances: Optional[np.ndarray] = None
    distance_distribution: Optional[Dict[str, float]] = None
    hubness_score: Optional[float] = None
    k_occurrences: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding large arrays."""
        result = {
            "layer_name": self.layer_name,
            "n_samples": self.n_samples,
            "n_dims": self.n_dims,
            "total_variance": self.total_variance,
            "sparsity": self.sparsity,
            "mle_intrinsic_dim": self.mle_intrinsic_dim,
            "correlation_dim": self.correlation_dim,
            "local_pca_intrinsic_dim": self.local_pca_intrinsic_dim,
            "pca_dim_for_95_var": self.pca_dim_for_95_var,
            "hubness_score": self.hubness_score,
        }
        if self.distance_distribution:
            result["distance_distribution"] = self.distance_distribution
        return result


# =============================================================================
# Basic Statistics
# =============================================================================

def compute_basic_stats(X: np.ndarray) -> Dict[str, Any]:
    """
    Compute basic statistics of activation matrix.

    Args:
        X: Activation matrix (n_samples, n_dims)

    Returns:
        Dictionary with mean, variance, total_variance, sparsity
    """
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)
    total_variance = float(np.var(X))

    # Sparsity: fraction of near-zero activations
    sparsity = float(np.mean(np.abs(X) < 1e-6))

    return {
        "mean": mean,
        "variance": variance,
        "total_variance": total_variance,
        "sparsity": sparsity,
    }


def compute_correlation_matrix(
    X: np.ndarray,
    max_dims: int = 100
) -> np.ndarray:
    """
    Compute correlation matrix of features.

    Args:
        X: Activation matrix (n_samples, n_dims)
        max_dims: Maximum dimensions to include (subsamples if exceeded)

    Returns:
        Correlation matrix (n_dims, n_dims) or (max_dims, max_dims)
    """
    import warnings

    if X.shape[1] > max_dims:
        # Subsample dimensions uniformly
        np.random.seed(42)
        idx = np.random.choice(X.shape[1], max_dims, replace=False)
        idx = np.sort(idx)
        X = X[:, idx]

    # Handle constant columns (zero variance)
    std = np.std(X, axis=0)
    valid_cols = std > 1e-8

    if np.sum(valid_cols) < 2:
        return np.eye(X.shape[1])

    X_valid = X[:, valid_cols]

    # Suppress warnings from near-zero variance columns
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr = np.corrcoef(X_valid.T)

    # Handle any NaN values that slipped through
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Expand back to full size if needed
    if not np.all(valid_cols):
        full_corr = np.eye(len(valid_cols))
        valid_idx = np.where(valid_cols)[0]
        for i, vi in enumerate(valid_idx):
            for j, vj in enumerate(valid_idx):
                full_corr[vi, vj] = corr[i, j]
        return full_corr

    return corr


# =============================================================================
# Intrinsic Dimensionality Estimation
# =============================================================================

def estimate_mle_intrinsic_dim(
    X: np.ndarray,
    k: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    MLE estimator for intrinsic dimensionality.

    Based on Levina & Bickel (2004): "Maximum Likelihood Estimation
    of Intrinsic Dimension".

    The MLE estimate uses the ratio of k-nearest neighbor distances
    to estimate the local dimensionality around each point.

    Args:
        X: Point cloud (n_samples, n_dims)
        k: Number of nearest neighbors to use
        seed: Random seed (unused, for interface consistency)

    Returns:
        Estimated intrinsic dimension
    """
    n_samples = len(X)
    if n_samples < k + 1:
        return float(X.shape[1])  # Return ambient dimension if too few samples

    # Fit k-NN
    k_actual = min(k + 1, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Exclude self (distance = 0)
    distances = distances[:, 1:]

    # Remove any zero distances to avoid log(0)
    distances = np.maximum(distances, 1e-10)

    # MLE estimate: d = (k-1) / sum(log(max_dist / dist_i))
    max_dists = distances[:, -1:]  # k-th neighbor distance
    log_ratios = np.log(max_dists / distances[:, :-1])

    # Avoid division by zero
    log_ratio_sums = np.sum(log_ratios, axis=1)
    valid_mask = log_ratio_sums > 0

    if np.sum(valid_mask) == 0:
        return float(X.shape[1])

    mle_dims = (k - 1) / log_ratio_sums[valid_mask]

    # Return mean estimate
    return float(np.mean(mle_dims))


def estimate_correlation_dim(
    X: np.ndarray,
    n_scales: int = 10,
    seed: Optional[int] = None
) -> float:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.

    The correlation dimension measures how the number of point pairs
    within distance r scales with r. For a d-dimensional manifold,
    C(r) ~ r^d as r -> 0.

    Args:
        X: Point cloud (n_samples, n_dims)
        n_scales: Number of distance scales to use
        seed: Random seed (unused)

    Returns:
        Estimated correlation dimension
    """
    from scipy.spatial.distance import pdist

    n_samples = len(X)
    if n_samples < 10:
        return np.nan

    # Subsample for efficiency if needed
    if n_samples > 2000:
        np.random.seed(seed if seed else 42)
        idx = np.random.choice(n_samples, 2000, replace=False)
        X = X[idx]

    # Compute pairwise distances
    distances = pdist(X, metric='euclidean')

    if len(distances) == 0:
        return np.nan

    # Compute correlation integral at different scales
    r_min, r_max = np.percentile(distances, [5, 95])

    if r_min <= 0 or r_max <= r_min:
        return np.nan

    scales = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    # C(r) = fraction of pairs with distance < r
    C_r = np.array([np.mean(distances < r) for r in scales])

    # Filter valid points for log-log fit
    valid = C_r > 0
    if np.sum(valid) < 3:
        return np.nan

    log_r = np.log(scales[valid])
    log_C = np.log(C_r[valid])

    # Linear fit in log-log space: log(C) = d * log(r) + const
    try:
        slope, _ = np.polyfit(log_r, log_C, 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def estimate_local_pca_dim(
    X: np.ndarray,
    k: int = 50,
    n_samples: int = 100,
    variance_threshold: float = 0.95,
    seed: Optional[int] = None
) -> float:
    """
    Estimate intrinsic dimension via local PCA neighborhoods.

    For each sample point, finds k-nearest neighbors and computes
    PCA on that local neighborhood. The local dimension is the number
    of components needed to explain variance_threshold of variance.

    Args:
        X: Point cloud (n_samples, n_dims)
        k: Number of neighbors for local neighborhoods
        n_samples: Number of random points to sample
        variance_threshold: Variance threshold for dimension counting
        seed: Random seed

    Returns:
        Mean local intrinsic dimension
    """
    n_points = len(X)
    if n_points < k + 1:
        return float(X.shape[1])

    rng = np.random.RandomState(seed)

    # Fit k-NN
    k_actual = min(k, n_points - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Sample random center points
    n_centers = min(n_samples, n_points)
    center_idx = rng.choice(n_points, n_centers, replace=False)

    local_dims = []
    for i in center_idx:
        # Get local neighborhood
        local_X = X[indices[i]]

        if local_X.shape[0] < 3:
            continue

        # Fit PCA
        try:
            n_components = min(local_X.shape[0] - 1, local_X.shape[1])
            if n_components < 1:
                continue

            pca = PCA(n_components=n_components).fit(local_X)
            cumsum = np.cumsum(pca.explained_variance_ratio_)

            # Find dimension for threshold
            local_dim = np.argmax(cumsum >= variance_threshold) + 1
            local_dims.append(local_dim)
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(local_dims) == 0:
        return float(X.shape[1])

    return float(np.mean(local_dims))


def compute_pca_analysis(
    X: np.ndarray,
    n_components: int = 50
) -> Dict[str, Any]:
    """
    Compute PCA analysis for dimensionality understanding.

    Args:
        X: Point cloud (n_samples, n_dims)
        n_components: Maximum components to compute

    Returns:
        Dictionary with explained variance ratios and cumulative variance
    """
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])

    if n_components < 1:
        return {
            "explained_variance_ratio": np.array([1.0]),
            "cumulative_variance": np.array([1.0]),
            "dim_for_95_var": 1,
        }

    pca = PCA(n_components=n_components).fit(X)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Find dimension for 95% variance
    dim_95 = int(np.argmax(cumulative >= 0.95) + 1)
    if cumulative[-1] < 0.95:
        dim_95 = len(cumulative)

    return {
        "explained_variance_ratio": explained,
        "cumulative_variance": cumulative,
        "dim_for_95_var": dim_95,
    }


# =============================================================================
# k-NN Structure Analysis
# =============================================================================

def compute_knn_structure(
    X: np.ndarray,
    k: int = 20,
    metric: str = "euclidean"
) -> Dict[str, Any]:
    """
    Analyze k-NN structure of the point cloud.

    Computes:
    - Distance distribution statistics
    - Hubness score (skewness of k-occurrence distribution)

    High hubness indicates presence of "hub" points that appear
    as neighbors of many other points, which can affect TDA results.

    Args:
        X: Point cloud (n_samples, n_dims)
        k: Number of neighbors
        metric: Distance metric

    Returns:
        Dictionary with distances, distribution stats, hubness
    """
    n_samples = len(X)
    k_actual = min(k + 1, n_samples - 1)

    if k_actual < 2:
        return {
            "distances": np.array([]),
            "distance_distribution": {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "iqr": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "hubness_score": 0.0,
            "k_occurrences": np.array([]),
        }

    nbrs = NearestNeighbors(n_neighbors=k_actual, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Exclude self
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Distance distribution statistics
    all_distances = distances.flatten()
    dist_stats = {
        "mean": float(np.mean(all_distances)),
        "std": float(np.std(all_distances)),
        "median": float(np.median(all_distances)),
        "iqr": float(np.percentile(all_distances, 75) - np.percentile(all_distances, 25)),
        "min": float(np.min(all_distances)),
        "max": float(np.max(all_distances)),
    }

    # Hubness: how often each point appears as a neighbor
    k_occurrences = np.bincount(indices.flatten(), minlength=n_samples)
    hubness = float(skew(k_occurrences))

    return {
        "distances": distances,
        "distance_distribution": dist_stats,
        "hubness_score": hubness,
        "k_occurrences": k_occurrences,
    }


# =============================================================================
# Main Characterization Function
# =============================================================================

def characterize_geometry(
    X: np.ndarray,
    layer_name: str,
    config: Optional[GeometryConfig] = None
) -> GeometryResult:
    """
    Comprehensive pre-TDA geometry characterization.

    Args:
        X: Activation matrix (n_samples, n_dims)
        layer_name: Name of the layer being analyzed
        config: Geometry configuration (uses defaults if None)

    Returns:
        GeometryResult with all computed metrics
    """
    if config is None:
        config = GeometryConfig()

    rng = np.random.RandomState(config.seed)
    n_samples, n_dims = X.shape

    result = GeometryResult(
        layer_name=layer_name,
        n_samples=n_samples,
        n_dims=n_dims
    )

    # Subsample for expensive computations
    if n_samples > config.max_samples_for_dim:
        idx = rng.choice(n_samples, config.max_samples_for_dim, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Basic statistics
    if config.compute_basic_stats:
        stats = compute_basic_stats(X)
        result.mean = stats["mean"]
        result.variance = stats["variance"]
        result.total_variance = stats["total_variance"]
        result.sparsity = stats["sparsity"]

        result.correlation_matrix = compute_correlation_matrix(
            X, max_dims=config.max_dims_for_correlation
        )

    # Intrinsic dimensionality
    if config.compute_intrinsic_dim:
        result.mle_intrinsic_dim = estimate_mle_intrinsic_dim(
            X_sub, k=config.mle_k, seed=config.seed
        )

        result.correlation_dim = estimate_correlation_dim(
            X_sub, n_scales=config.correlation_dim_scales, seed=config.seed
        )

        result.local_pca_intrinsic_dim = estimate_local_pca_dim(
            X_sub,
            k=config.local_pca_k,
            n_samples=config.local_pca_samples,
            variance_threshold=config.variance_threshold,
            seed=config.seed
        )

        pca_result = compute_pca_analysis(X_sub)
        result.pca_explained_variance_ratio = pca_result["explained_variance_ratio"]
        result.pca_cumulative_variance = pca_result["cumulative_variance"]
        result.pca_dim_for_95_var = pca_result["dim_for_95_var"]

    # k-NN structure
    if config.compute_knn_structure:
        knn_result = compute_knn_structure(
            X_sub, k=config.k_neighbors, metric=config.distance_metric
        )
        result.knn_distances = knn_result["distances"]
        result.distance_distribution = knn_result["distance_distribution"]
        result.hubness_score = knn_result["hubness_score"]
        result.k_occurrences = knn_result["k_occurrences"]

    return result


def summarize_geometry(results: List[GeometryResult]) -> pd.DataFrame:
    """
    Create summary DataFrame from multiple geometry results.

    Args:
        results: List of GeometryResult objects (e.g., from different layers)

    Returns:
        DataFrame with one row per layer
    """
    rows = []
    for r in results:
        row = {
            "layer": r.layer_name,
            "n_samples": r.n_samples,
            "n_dims": r.n_dims,
            "total_variance": r.total_variance,
            "sparsity": r.sparsity,
            "mle_intrinsic_dim": r.mle_intrinsic_dim,
            "correlation_dim": r.correlation_dim,
            "local_pca_dim": r.local_pca_intrinsic_dim,
            "pca_dim_95var": r.pca_dim_for_95_var,
            "hubness": r.hubness_score,
        }
        if r.distance_distribution:
            for k, v in r.distance_distribution.items():
                row[f"dist_{k}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def recommend_pca_components(results: List[GeometryResult]) -> int:
    """
    Recommend PCA components based on geometry analysis.

    Heuristic: Use the maximum of:
    - 2x the average MLE intrinsic dimension
    - Dimension needed for 95% variance
    - At least 20 (for TDA stability)

    Args:
        results: List of GeometryResult objects

    Returns:
        Recommended number of PCA components
    """
    mle_dims = [r.mle_intrinsic_dim for r in results if r.mle_intrinsic_dim is not None]
    pca_dims = [r.pca_dim_for_95_var for r in results if r.pca_dim_for_95_var is not None]

    recommendations = []

    if mle_dims:
        # 2x average MLE dimension
        recommendations.append(int(2 * np.mean(mle_dims)))

    if pca_dims:
        # Max dimension for 95% variance across layers
        recommendations.append(max(pca_dims))

    # Minimum of 20 for TDA stability
    recommendations.append(20)

    return max(recommendations)
