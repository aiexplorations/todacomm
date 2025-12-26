from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import numpy as np
from ripser import ripser
from persim import PersistenceImager
from sklearn.neighbors import NearestNeighbors


@dataclass
class TDAConfig:
    maxdim: int = 1
    metric: str = "euclidean"
    pca_components: Optional[int] = 20
    # Multi-scale sampling configuration
    sampling_strategy: Literal["uniform", "multiscale", "adaptive"] = "uniform"
    max_points: int = 2000
    global_sample_ratio: float = 0.5  # Fraction for global sampling in multiscale
    local_neighborhoods: int = 10     # Number of local neighborhoods
    local_neighbors: int = 50         # K-neighbors per local region
    # Bootstrap configuration for confidence intervals
    n_bootstrap: int = 100            # Number of bootstrap replicates
    ci_level: float = 0.95            # Confidence interval level (e.g., 0.95 for 95% CI)


def _maybe_pca(X: np.ndarray, k: Optional[int]) -> np.ndarray:
    if k is None or X.shape[1] <= (k or 0):
        return X
    # simple PCA via SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:k].T).astype(np.float32)


def uniform_sample(X: np.ndarray, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    """Uniform random sampling of points"""
    if len(X) <= n_samples:
        return X
    
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    return X[indices]


def multiscale_sample(X: np.ndarray, config: TDAConfig, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Multi-scale sampling strategy combining global and local sampling.
    
    Args:
        X: Point cloud data (n_points, n_dims)
        config: TDAConfig with sampling parameters
        seed: Random seed for reproducibility
        
    Returns:
        Dict containing sampled points and metadata
    """
    rng = np.random.RandomState(seed)
    n_total = len(X)
    
    if n_total <= config.max_points:
        return {"points": X, "method": "no_sampling", "n_global": n_total, "n_local": 0}
    
    # Calculate sampling sizes
    n_global = int(config.max_points * config.global_sample_ratio)
    n_local_per_neighborhood = (config.max_points - n_global) // config.local_neighborhoods
    
    # Global uniform sampling
    global_indices = rng.choice(n_total, size=n_global, replace=False)
    global_sample = X[global_indices]
    
    # Local neighborhood sampling
    local_samples = []
    local_center_indices = rng.choice(n_total, size=config.local_neighborhoods, replace=False)
    
    # Use KNN to find local neighborhoods
    nbrs = NearestNeighbors(n_neighbors=min(config.local_neighbors, n_total), 
                           algorithm='auto').fit(X)
    
    for center_idx in local_center_indices:
        # Find neighbors of this center point
        distances, neighbor_indices = nbrs.kneighbors([X[center_idx]])
        neighbor_indices = neighbor_indices[0]  # Remove batch dimension
        
        # Sample from local neighborhood
        local_sample_size = min(n_local_per_neighborhood, len(neighbor_indices))
        if local_sample_size > 0:
            local_subset_indices = rng.choice(neighbor_indices, size=local_sample_size, replace=False)
            local_samples.append(X[local_subset_indices])
    
    # Combine global and local samples
    if local_samples:
        local_combined = np.vstack(local_samples)
        combined_sample = np.vstack([global_sample, local_combined])
    else:
        combined_sample = global_sample
    
    # Remove duplicates (can occur if global and local sampling overlap)
    _, unique_indices = np.unique(combined_sample, axis=0, return_index=True)
    final_sample = combined_sample[unique_indices]
    
    return {
        "points": final_sample,
        "method": "multiscale",
        "n_global": len(global_sample),
        "n_local": len(local_combined) if local_samples else 0,
        "n_final": len(final_sample),
        "n_duplicates_removed": len(combined_sample) - len(final_sample)
    }


def adaptive_sample(X: np.ndarray, config: TDAConfig, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Adaptive sampling based on local density.
    
    Higher density regions get more samples to preserve local topology.
    """
    rng = np.random.RandomState(seed)
    n_total = len(X)
    
    if n_total <= config.max_points:
        return {"points": X, "method": "no_sampling", "n_samples": n_total}
    
    # Estimate local density using k-nearest neighbors
    k_density = min(20, n_total // 10)  # Use 20 neighbors or 10% of data
    nbrs = NearestNeighbors(n_neighbors=k_density, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Density estimate: inverse of average distance to k-nearest neighbors
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Skip self (distance=0)
    densities = 1.0 / (avg_distances + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Normalize densities to probabilities
    sampling_probs = densities / np.sum(densities)
    
    # Sample according to density
    sample_indices = rng.choice(n_total, size=config.max_points, replace=False, p=sampling_probs)
    sampled_points = X[sample_indices]
    
    return {
        "points": sampled_points,
        "method": "adaptive",
        "n_samples": len(sampled_points),
        "density_stats": {
            "min_density": float(np.min(densities)),
            "max_density": float(np.max(densities)),
            "mean_density": float(np.mean(densities))
        }
    }


def apply_sampling(X: np.ndarray, config: TDAConfig, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Apply the configured sampling strategy"""
    if config.sampling_strategy == "uniform":
        sampled_points = uniform_sample(X, config.max_points, seed)
        return {"points": sampled_points, "method": "uniform", "n_samples": len(sampled_points)}
    elif config.sampling_strategy == "multiscale":
        return multiscale_sample(X, config, seed)
    elif config.sampling_strategy == "adaptive":
        return adaptive_sample(X, config, seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {config.sampling_strategy}")


def compute_persistence(X: np.ndarray, config: TDAConfig, seed: Optional[int] = None) -> Dict:
    """
    Compute persistent homology with configurable sampling and preprocessing.
    
    Args:
        X: Input point cloud (n_points, n_dims)
        config: TDA configuration including sampling strategy
        seed: Random seed for reproducible sampling
        
    Returns:
        Dict containing persistence diagrams and metadata
    """
    # Apply PCA preprocessing
    Xproc = _maybe_pca(X, config.pca_components)
    
    # Apply sampling strategy
    sampling_result = apply_sampling(Xproc, config, seed)
    Xsampled = sampling_result["points"]
    
    # Compute persistent homology
    # Suppress ripser warning about square matrices (we're passing point clouds, not distance matrices)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*distance_matrix.*")
        result = ripser(Xsampled, maxdim=config.maxdim, metric=config.metric)
    dgms = result["dgms"]  # list: H0, H1, ...
    
    return {
        "dgms": dgms,
        "X_proc": Xproc,
        "X_sampled": Xsampled,
        "sampling_info": sampling_result,
        "original_shape": X.shape,
        "processed_shape": Xproc.shape,
        "sampled_shape": Xsampled.shape
    }


def summarize_diagrams(dgms: List[np.ndarray]) -> Dict[str, float]:
    summaries: Dict[str, float] = {}
    for dim, dgm in enumerate(dgms):
        if dgm.size == 0:
            summaries[f"H{dim}_count"] = 0.0
            summaries[f"H{dim}_total_persistence"] = 0.0
            summaries[f"H{dim}_max_lifetime"] = 0.0
            continue
        births = dgm[:, 0]
        deaths = np.where(np.isinf(dgm[:, 1]), births.max() + 1.0, dgm[:, 1])
        lifetimes = deaths - births
        summaries[f"H{dim}_count"] = float(len(lifetimes))
        summaries[f"H{dim}_total_persistence"] = float(np.sum(lifetimes))
        summaries[f"H{dim}_max_lifetime"] = float(np.max(lifetimes))
    return summaries


def persistence_image(dgm: np.ndarray, pixels: int = 20) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros((pixels, pixels), dtype=np.float32)
    pimgr = PersistenceImager(pixel_size=1.0, birth_range=None, pers_range=None, pixels=[pixels, pixels])
    pimgr.fit(dgm)
    img = pimgr.transform(dgm)
    return img.astype(np.float32)


@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval computation."""
    point_estimate: Dict[str, float]
    ci_lower: Dict[str, float]
    ci_upper: Dict[str, float]
    std_error: Dict[str, float]
    n_bootstrap: int
    ci_level: float
    bootstrap_samples: Optional[List[Dict[str, float]]] = None  # Raw bootstrap values if requested


def bootstrap_persistence_ci(
    X: np.ndarray,
    config: TDAConfig,
    n_bootstrap: Optional[int] = None,
    ci_level: Optional[float] = None,
    seed: Optional[int] = None,
    return_samples: bool = False
) -> BootstrapResult:
    """
    Compute bootstrap confidence intervals for TDA summary statistics.

    Uses percentile bootstrap method: resample data with replacement,
    compute TDA on each resample, and take percentiles of the distribution.

    Args:
        X: Input point cloud (n_points, n_dims)
        config: TDA configuration
        n_bootstrap: Number of bootstrap replicates (default: config.n_bootstrap)
        ci_level: Confidence level, e.g., 0.95 for 95% CI (default: config.ci_level)
        seed: Random seed for reproducibility
        return_samples: If True, include all bootstrap sample values in result

    Returns:
        BootstrapResult containing point estimates and confidence intervals
    """
    n_boot = n_bootstrap if n_bootstrap is not None else config.n_bootstrap
    ci = ci_level if ci_level is not None else config.ci_level

    rng = np.random.RandomState(seed)
    n_points = len(X)

    # Collect bootstrap samples
    bootstrap_summaries: List[Dict[str, float]] = []

    for b in range(n_boot):
        # Resample with replacement
        boot_indices = rng.choice(n_points, size=n_points, replace=True)
        X_boot = X[boot_indices]

        # Compute persistence on bootstrap sample
        # Use a different seed for each bootstrap to vary sampling if applicable
        boot_seed = rng.randint(0, 2**31) if seed is not None else None
        result = compute_persistence(X_boot, config, seed=boot_seed)
        summary = summarize_diagrams(result["dgms"])
        bootstrap_summaries.append(summary)

    # Compute point estimate on original data
    original_result = compute_persistence(X, config, seed=seed)
    point_estimate = summarize_diagrams(original_result["dgms"])

    # Extract metric names
    metric_names = list(point_estimate.keys())

    # Compute confidence intervals using percentile method
    alpha = 1 - ci
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower: Dict[str, float] = {}
    ci_upper: Dict[str, float] = {}
    std_error: Dict[str, float] = {}

    for metric in metric_names:
        values = np.array([s[metric] for s in bootstrap_summaries])
        ci_lower[metric] = float(np.percentile(values, lower_percentile))
        ci_upper[metric] = float(np.percentile(values, upper_percentile))
        std_error[metric] = float(np.std(values, ddof=1))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_boot,
        ci_level=ci,
        bootstrap_samples=bootstrap_summaries if return_samples else None
    )


def compare_bootstrap_results(
    results: Dict[str, BootstrapResult],
    metric: str = "H0_total_persistence"
) -> Dict[str, Dict]:
    """
    Compare bootstrap results across multiple conditions/models.

    Checks for statistically significant differences by examining
    whether confidence intervals overlap.

    Args:
        results: Dict mapping condition names to BootstrapResult
        metric: Which TDA metric to compare

    Returns:
        Dict with pairwise comparisons and effect sizes
    """
    conditions = list(results.keys())
    comparisons = {}

    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            r1, r2 = results[cond1], results[cond2]

            # Check CI overlap (conservative test)
            ci_overlap = not (r1.ci_upper[metric] < r2.ci_lower[metric] or
                             r2.ci_upper[metric] < r1.ci_lower[metric])

            # Effect size (Cohen's d approximation using pooled SE)
            diff = r1.point_estimate[metric] - r2.point_estimate[metric]
            pooled_se = np.sqrt((r1.std_error[metric]**2 + r2.std_error[metric]**2) / 2)
            cohens_d = diff / pooled_se if pooled_se > 0 else 0.0

            comparisons[f"{cond1}_vs_{cond2}"] = {
                "diff": diff,
                "ci_overlap": ci_overlap,
                "significant": not ci_overlap,
                "cohens_d": cohens_d,
                f"{cond1}_estimate": r1.point_estimate[metric],
                f"{cond2}_estimate": r2.point_estimate[metric],
            }

    return comparisons


def ablation_sample_size(
    X: np.ndarray,
    config: TDAConfig,
    sample_sizes: Optional[List[int]] = None,
    n_repeats: int = 10,
    seed: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Ablation study: How do TDA metrics vary with sample size?

    Args:
        X: Full input point cloud
        config: TDA configuration
        sample_sizes: List of sample sizes to test (default: [50, 100, 200, 500, 1000])
        n_repeats: Number of repeated measurements per sample size
        seed: Random seed

    Returns:
        Dict mapping sample sizes to summary statistics with mean/std
    """
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 500, 1000]

    # Filter to valid sample sizes
    n_points = len(X)
    sample_sizes = [s for s in sample_sizes if s <= n_points]

    rng = np.random.RandomState(seed)
    results = {}

    for size in sample_sizes:
        size_summaries = []

        for rep in range(n_repeats):
            # Subsample
            indices = rng.choice(n_points, size=size, replace=False)
            X_sub = X[indices]

            # Compute TDA
            rep_seed = rng.randint(0, 2**31) if seed is not None else None
            result = compute_persistence(X_sub, config, seed=rep_seed)
            summary = summarize_diagrams(result["dgms"])
            size_summaries.append(summary)

        # Aggregate across repeats
        metric_names = list(size_summaries[0].keys())
        results[size] = {
            "n_repeats": n_repeats,
            "metrics": {}
        }

        for metric in metric_names:
            values = np.array([s[metric] for s in size_summaries])
            results[size]["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv": float(np.std(values, ddof=1) / np.mean(values)) if np.mean(values) > 0 else 0.0
            }

    return results


def ablation_pca_components(
    X: np.ndarray,
    config: TDAConfig,
    pca_values: Optional[List[int]] = None,
    n_repeats: int = 5,
    seed: Optional[int] = None
) -> Dict[str, Dict]:
    """
    Ablation study: How do TDA metrics vary with PCA components?

    Args:
        X: Input point cloud
        config: Base TDA configuration
        pca_values: List of PCA component counts to test (default: [10, 20, 50, 100, None])
        n_repeats: Number of repeated measurements per setting
        seed: Random seed

    Returns:
        Dict mapping PCA components to summary statistics
    """
    if pca_values is None:
        pca_values = [10, 20, 50, 100, None]  # None = no PCA

    # Filter to valid PCA values (must be < n_dims)
    n_dims = X.shape[1]
    pca_values = [p for p in pca_values if p is None or p < n_dims]

    rng = np.random.RandomState(seed)
    results = {}

    for pca_k in pca_values:
        # Create modified config
        test_config = TDAConfig(
            maxdim=config.maxdim,
            metric=config.metric,
            pca_components=pca_k,
            sampling_strategy=config.sampling_strategy,
            max_points=config.max_points,
            global_sample_ratio=config.global_sample_ratio,
            local_neighborhoods=config.local_neighborhoods,
            local_neighbors=config.local_neighbors,
            n_bootstrap=config.n_bootstrap,
            ci_level=config.ci_level
        )

        pca_summaries = []
        for rep in range(n_repeats):
            rep_seed = rng.randint(0, 2**31) if seed is not None else None
            result = compute_persistence(X, test_config, seed=rep_seed)
            summary = summarize_diagrams(result["dgms"])
            pca_summaries.append(summary)

        # Aggregate
        key = str(pca_k) if pca_k is not None else "None"
        metric_names = list(pca_summaries[0].keys())
        results[key] = {
            "pca_components": pca_k,
            "n_repeats": n_repeats,
            "metrics": {}
        }

        for metric in metric_names:
            values = np.array([s[metric] for s in pca_summaries])
            results[key]["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "cv": float(np.std(values, ddof=1) / np.mean(values)) if np.mean(values) > 0 else 0.0
            }

    return results
