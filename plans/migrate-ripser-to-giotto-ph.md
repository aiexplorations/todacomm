# Migration Plan: ripser → giotto-ph

## Objective

Migrate from `ripser` to `giotto-ph` for persistent homology computation to leverage multicore CPU parallelization on Apple Silicon.

## Rationale

- **Performance**: giotto-ph's parallel implementation can match or exceed GPU-accelerated Ripser++ using 5-10 CPU cores
- **Apple Silicon**: Native ARM wheels available; no CUDA dependency
- **API Compatibility**: Nearly identical API (`ripser_parallel` vs `ripser`)
- **Active maintenance**: giotto-ph is actively maintained by giotto-ai

## Current State

### Files to modify
| File | Change |
|------|--------|
| `todacomm/tda/persistence.py` | Replace ripser import and function call |
| `pyproject.toml` | Replace ripser dependency with giotto-ph |
| `requirements.txt` | Replace ripser dependency with giotto-ph |
| `tests/test_tda_persistence.py` | No changes needed (tests use public API) |

### Current implementation (`persistence.py:8,190`)
```python
from ripser import ripser
# ...
result = ripser(Xsampled, maxdim=config.maxdim, metric=config.metric)
```

## Implementation Steps

### Step 1: Update dependencies

**pyproject.toml** (line 34):
```diff
-    "ripser>=0.6.4",
+    "giotto-ph>=0.2.2",
```

**requirements.txt** (line 19):
```diff
-ripser>=0.6.4
+giotto-ph>=0.2.2
```

### Step 2: Update TDAConfig

Add `n_threads` parameter to `TDAConfig` for controlling parallelization:

```python
@dataclass
class TDAConfig:
    maxdim: int = 1
    metric: str = "euclidean"
    pca_components: Optional[int] = 20
    n_threads: int = -1  # -1 = use all available cores
    # ... rest unchanged
```

### Step 3: Update persistence.py imports

```diff
-from ripser import ripser
+from gph import ripser_parallel
```

### Step 4: Update compute_persistence function

```diff
-result = ripser(Xsampled, maxdim=config.maxdim, metric=config.metric)
+result = ripser_parallel(
+    Xsampled,
+    maxdim=config.maxdim,
+    metric=config.metric,
+    n_threads=config.n_threads
+)
```

### Step 5: Remove ripser warning suppression

The warning suppression for ripser can be removed since giotto-ph handles this differently:

```diff
-    # Suppress ripser warning about square matrices
-    with warnings.catch_warnings():
-        warnings.filterwarnings("ignore", message=".*distance_matrix.*")
-        result = ripser(Xsampled, maxdim=config.maxdim, metric=config.metric)
+    result = ripser_parallel(
+        Xsampled,
+        maxdim=config.maxdim,
+        metric=config.metric,
+        n_threads=config.n_threads
+    )
```

## API Differences

| Feature | ripser | giotto-ph |
|---------|--------|-----------|
| Import | `from ripser import ripser` | `from gph import ripser_parallel` |
| Function | `ripser(X, maxdim=1)` | `ripser_parallel(X, maxdim=1, n_threads=-1)` |
| Parallelism | Single-threaded | Multi-threaded (H1+ dimensions) |
| Return format | `{"dgms": [...]}` | `{"dgms": [...]}` (identical) |
| Metric support | euclidean, cosine, precomputed | euclidean, cosine, precomputed |

## Testing

1. **Run existing test suite** - all tests should pass without modification:
   ```bash
   pytest tests/test_tda_persistence.py -v
   ```

2. **Verify parallelization works**:
   ```python
   import os
   print(f"Available cores: {os.cpu_count()}")
   # Run compute_persistence and verify it uses multiple cores
   ```

3. **Performance comparison** (optional):
   ```python
   import time
   # Compare runtime with n_threads=1 vs n_threads=-1
   ```

## Rollback Plan

If issues arise, revert the three files to use ripser:
```bash
git checkout HEAD -- todacomm/tda/persistence.py pyproject.toml requirements.txt
pip install -e .
```

## Notes

- giotto-ph parallelizes H1 and higher dimensions; H0 remains single-threaded
- The `collapse_edges` option in giotto-ph can further speed up computation for large datasets (not enabled by default)
- DTM (Distance to Measure) weighted filtrations available via `weights="DTM"` parameter

## References

- [giotto-ph GitHub](https://github.com/giotto-ai/giotto-ph)
- [giotto-ph PyPI](https://pypi.org/project/giotto-ph/)
- [giotto-ph Paper](https://arxiv.org/abs/2107.05412)
- [ripser_parallel API docs](https://giotto-ai.github.io/giotto-ph/build/html/modules/ripser_parallel.html)
