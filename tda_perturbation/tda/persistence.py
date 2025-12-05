from __future__ import annotations

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
