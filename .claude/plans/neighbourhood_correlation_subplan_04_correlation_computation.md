# Sub-Plan 04: Correlation Computation (Approaches A + B + Derived Metrics)

**Parent plan**: `neighbourhood_correlation_plan.md` (Computation: Two Approaches)

## Review Corrections (applied)

1. **Keep X SPARSE throughout** — marker subsets can be 1000s of genes (data-driven selection). Dense would be 416k × 2000 genes × 4 bytes = 3.3 GB. Sparse at ~10-30% density = 330 MB–1 GB. Use float32.
2. **Pearson WITHOUT centering** — use the alternative formula that works on sparse data:
   ```
   r = (Σ(x·y)/n - x̄·ȳ) / (σ_x · σ_y)
   ```
   Pre-compute per-row stats ONCE from sparse (all O(nnz)):
   ```python
   n = X.shape[1]  # n_markers
   sum_x = np.asarray(X.sum(axis=1)).flatten()             # (n_cells,)
   sum_x2 = np.asarray(X.power(2).sum(axis=1)).flatten()   # (n_cells,)
   mean_x = sum_x / n
   var_x = sum_x2 / n - mean_x**2
   std_x = np.sqrt(np.clip(var_x, 1e-12, None))
   ```
3. **Approach B — sparse throughout**:
   ```python
   weighted_sum = mask_csr @ X  # sparse × sparse = sparse (n_cells, n_markers)
   row_sums = np.asarray(mask_csr.sum(axis=1)).flatten()
   # avg_profiles is dense (weighted mean fills zeros) but computed from sparse
   avg_profiles = np.asarray(weighted_sum.todense()) / np.where(row_sums > 0, row_sums, 1)[:, None]
   # Row-wise Pearson via cross-term (X stays sparse):
   cross = np.asarray(X.multiply(avg_profiles).sum(axis=1)).flatten()  # sparse × dense → sparse → sum
   mean_a = avg_profiles.mean(axis=1)
   std_a = avg_profiles.std(axis=1).clip(min=1e-12)
   corr_avg = (cross / n - mean_x * mean_a) / (std_x * std_a)
   corr_avg[row_sums == 0] = np.nan
   ```
   Note: `avg_profiles` is dense (unavoidable — weighted mean is dense) but the cross-term uses `X.multiply(dense)` which stays sparse (only X's non-zero pattern).
4. **Approach A — fully sparse block processing**:
   ```python
   # Pre-computed: mean_x, std_x for all cells (from step 2)
   # Per block [b0:b1]:
   block_csr = mask_csr[b0:b1].tocoo()
   rows = block_csr.row + b0  # absolute cell indices
   cols = block_csr.col        # neighbour indices
   weights = block_csr.data

   # Sparse row extraction for all pairs
   X_i = X[rows]  # (nnz_block, n_markers) sparse — fancy indexing
   X_j = X[cols]  # (nnz_block, n_markers) sparse

   # Cross terms via sparse element-wise multiply (non-zero only where BOTH non-zero)
   cross = np.asarray(X_i.multiply(X_j).sum(axis=1)).flatten()  # (nnz_block,)

   # Pearson from pre-computed stats
   r_flat = (cross / n - mean_x[rows] * mean_x[cols]) / (std_x[rows] * std_x[cols])

   # Aggregate per source row via np.add.reduceat (rows already sorted)
   ```
   Memory per block: `batch_size=2000` → ~100k pairs × n_markers sparse → ~160 MB at 10% density. Adjust batch_size based on n_markers.
5. **Weighted median explicit implementation** (no scipy):
   ```python
   def weighted_median(values, weights):
       order = np.argsort(values)
       v, w = values[order], weights[order]
       cw = np.cumsum(w)
       return v[np.searchsorted(cw, 0.5 * cw[-1])]
   ```
   Per-row Python loop unavoidable — bottleneck of Approach A; acceptable.
6. **NaN handling**: use `np.where(row_sums > 0, numer/denom, np.nan)` to avoid RuntimeWarnings.
7. **Add `corr_weighted_mean_all_neighbours`** to outputs (main plan lists it; previously missing from sub-plan).
8. **Output column for `n_neighbours_within_technical_cross_dataset`**: add this when both technical and dataset keys are provided — sub-plan 08 needs it for H14.
9. **Document orphan cascade**: cells with `n_neighbours_same_library == 0` get NaN in ALL normalised metrics by design (matches WL-0 leaf).
10. **Adaptive batch_size**: `batch_size = max(500, min(10000, 100_000_000 // (n_markers * 4)))` — auto-scale to keep intermediate arrays under ~400 MB.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD)

## Dependencies
- Sub-plans 01, 03 complete (`normalise_counts`, `construct_neighbour_masks`, `list_active_masks`)

## Tasks

### 1. Main entry point: `compute_marker_correlation`

```python
def compute_marker_correlation(
    adata,
    connectivities,                 # sparse CSR
    marker_genes,                   # list or pd.Index of var_names
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
    layer: str | None = None,       # use adata.X if None; else adata.layers[layer]
    batch_size: int = 10000,        # cells per block for Approach A
) -> pd.DataFrame:
    """Compute per-cell neighbourhood marker gene correlation metrics.

    Returns DataFrame indexed by adata.obs_names with columns:
        n_neighbours_{mask}, frac_neighbours_{mask}
        corr_avg_{mask}                     (Approach B)
        corr_mean_{mask}, corr_median_{mask}
        corr_weighted_mean_{mask}, corr_weighted_median_{mask}
        corr_std_{mask}, corr_cv_{mask}
        corr_discrepancy_{mask}             (A vs B)
        corr_norm_by_library_{mask}, corr_norm_by_all_{mask}  (derived)
    Plus mask-independent:
        corr_avg_all_neighbours, corr_weighted_mean_all_neighbours
        n_neighbours_total, marker_gene_total_expression
    """
```

### 2. Workflow inside function

1. Extract `X = adata[:, marker_genes].X` (possibly via `layer`) — stays SPARSE
2. Normalise: `X = normalise_counts(X, n_vars=adata.n_vars)` — stays sparse; use full adata.n_vars for scaling
3. Pre-compute per-row stats from sparse (O(nnz)):
   ```python
   n = X.shape[1]
   sum_x = np.asarray(X.sum(axis=1)).flatten()
   sum_x2 = np.asarray(X.power(2).sum(axis=1)).flatten()
   mean_x = sum_x / n
   std_x = np.sqrt(np.clip(sum_x2 / n - mean_x**2, 1e-12, None))
   marker_gene_total_expression = sum_x  # total normalised expression
   ```
4. Construct masks: `masks = construct_neighbour_masks(adata, connectivities, ...)`
5. Also compute all-neighbours base: use `connectivities` (no mask, self-loops removed)
6. Compute adaptive batch_size: `batch_size = max(500, min(10000, 100_000_000 // (n * 4)))`
7. For each mask + `all_neighbours`:
   - Call `_approach_B_per_mask(X, mask, mean_x, std_x)` → `corr_avg_{mask}` (X stays sparse)
   - Call `_approach_A_per_mask(X, mask, mean_x, std_x, batch_size)` → `corr_mean/median/weighted_mean/weighted_median/std/cv`
   - Compute `n_neighbours_{mask} = np.asarray(mask.getnnz(axis=1)).flatten()` (count non-zero entries)
9. After all masks done, compute `frac_neighbours_{mask} = n_neighbours_{mask} / n_neighbours_total`
10. Compute `corr_discrepancy_{mask} = corr_avg_{mask} - corr_mean_{mask}`
11. Compute derived normalised metrics:
    - `corr_norm_by_library_{mask} = corr_avg_{mask} / corr_avg_same_library` (NaN-safe division)
    - `corr_norm_by_all_{mask} = corr_avg_{mask} / corr_avg_all_neighbours`
12. Assemble final DataFrame, index by `adata.obs_names`

### 3. Approach B: `_approach_B_per_mask`

```python
def _approach_B_per_mask(X, mask_csr, mean_x, std_x):
    """Weighted-average-of-neighbours then Pearson(cell, average).

    X: SPARSE (n_cells, n_markers). NOT densified.
    mean_x, std_x: pre-computed per-row stats (n_cells,).

    Vectorised (no per-cell loop):

    weighted_sum = mask_csr @ X              # sparse × sparse = sparse
    row_sums = np.asarray(mask_csr.sum(axis=1)).flatten()
    # avg_profiles STAYS SPARSE (weighted mean of sparse vectors is sparse)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    avg_profiles = weighted_sum.multiply(1.0 / safe_sums[:, None])  # sparse

    # Row-wise Pearson via non-centering formula (everything sparse):
    # r = (Σ(x·a)/n - x̄·ā) / (σ_x · σ_a)
    n = X.shape[1]
    cross = np.asarray(X.multiply(avg_profiles).sum(axis=1)).flatten()  # sparse × sparse → sum
    sum_a = np.asarray(avg_profiles.sum(axis=1)).flatten()
    sum_a2 = np.asarray(avg_profiles.power(2).sum(axis=1)).flatten()
    mean_a = sum_a / n
    std_a = np.sqrt(np.clip(sum_a2 / n - mean_a**2, 1e-12, None))
    corr_avg = (cross / n - mean_x * mean_a) / (std_x * std_a)
    corr_avg[row_sums == 0] = np.nan
    return corr_avg  # (n_cells,)
    """
```

Note: `avg_profiles` is SPARSE (`sparse @ sparse = sparse`; per-row scaling preserves sparsity). Entire Approach B pipeline stays sparse. The non-centering Pearson formula is custom (no package provides sparse Pearson) — **MUST be validated against `np.corrcoef` in tests**.

### 4. Approach A: `_approach_A_per_mask`

```python
def _approach_A_per_mask(X, mask_csr, mean_x, std_x, batch_size=2000):
    """Pairwise correlations then aggregate (weighted + unweighted).

    X: SPARSE (n_cells, n_markers).
    mean_x, std_x: pre-computed per-row stats.

    Pearson without centering:
        r[i,j] = (Σ(x_i · x_j) / n - mean_x[i]*mean_x[j]) / (std_x[i]*std_x[j])

    Vectorised per block:

    for b0 in range(0, n_cells, batch_size):
        block_csr = mask_csr[b0:b1].tocoo()
        rows = block_csr.row + b0   # absolute indices
        cols = block_csr.col
        weights = block_csr.data

        # Sparse row extraction for all pairs
        X_i = X[rows]    # (nnz_block, n_markers) sparse
        X_j = X[cols]    # (nnz_block, n_markers) sparse

        # Cross terms: element-wise sparse × sparse (non-zero only at intersection)
        cross = np.asarray(X_i.multiply(X_j).sum(axis=1)).flatten()

        # Pearson from pre-computed stats
        r_flat = (cross / n - mean_x[rows] * mean_x[cols]) / (
            std_x[rows] * std_x[cols]
        )

        # Aggregate per source row (rows already sorted in COO-from-CSR):
        # use np.add.reduceat for mean, np.split for median/std/etc.

    Returns dict of (n_cells,) arrays keyed by
        "mean", "median", "weighted_mean", "weighted_median", "std", "cv"
    """
```

**Weighted median** (per-row Python loop — unavoidable bottleneck; acceptable):
```python
def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, 0.5 * cw[-1])]
```

**Memory**: `batch_size=2000` → ~100k pairs × n_markers sparse at ~10% density → ~160 MB. Auto-scale batch_size based on n_markers (correction #10).

### 5. Handling zero neighbours

- `n_neighbours_{mask} == 0` → all correlation columns for that mask = NaN
- Division operations use `np.where(denom != 0, numer / denom, np.nan)`

### 6. Add to `__all__`

Export `compute_marker_correlation`.

## Test cases

### MANDATORY: Sparse Pearson validation against np.corrcoef
```python
# Generate small test data (100 cells × 50 genes, ~30% sparse)
rng = np.random.default_rng(42)
X_dense = rng.poisson(2, size=(100, 50)).astype(np.float32)
X_dense[rng.random(X_dense.shape) < 0.3] = 0
X_sparse = sp.csr_matrix(X_dense)

# Test _approach_B: compare sparse Pearson against np.corrcoef for each cell vs its avg
for cell_i in range(min(20, 100)):
    # manual dense Pearson
    avg_dense = X_dense[neighbours_of_i].mean(axis=0)
    expected = np.corrcoef(X_dense[cell_i], avg_dense)[0, 1]
    # sparse Pearson from our implementation
    actual = corr_avg_from_sparse[cell_i]
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

# Test _approach_A: compare per-neighbour correlations
for cell_i in range(min(20, 100)):
    for j in neighbours_of_i:
        expected = np.corrcoef(X_dense[cell_i], X_dense[j])[0, 1]
        actual = r_flat_from_sparse[pair_index]
        np.testing.assert_allclose(actual, expected, rtol=1e-5)
```

### Other tests
- Zero-neighbour cells get NaN for all correlation columns but non-NaN for `n_neighbours`
- `corr_discrepancy` = 0 when all neighbours are identical to each other
- `corr_norm_by_library_same_library` ≈ 1.0 by construction (trivial)
- Zero-variance rows (constant expression): correlation should be NaN (division by zero in std)
- All-zero rows: correlation should be NaN
- Sparsity preserved: intermediate matrices checked with `sp.issparse()`
- Runtime on immune integration (416k cells, 2000 markers, 4 masks): sanity check <60 min total

## Verification
- Compute on one trained immune model, verify output DataFrame has expected columns and sensible value ranges
- Spot-check a few cells manually
- Memory profile: sparse matrix never densified except for marker-subset (~300 MB)
