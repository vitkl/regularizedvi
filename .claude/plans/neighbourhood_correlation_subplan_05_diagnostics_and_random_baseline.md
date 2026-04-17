# Sub-Plan 05: Part 1 Diagnostics (All 3 Covariates) + Random KNN Baseline

**Parent plan**: `neighbourhood_correlation_plan.md` (Part 1 Neighbour Distribution Diagnostics + Random KNN baseline section)

## Review Corrections (applied)

1. **Vectorised composition via sparse matmul** (primary implementation):
   ```python
   codes = adata.obs[level_key].astype("category").cat.codes.to_numpy()
   one_hot = sp.csr_matrix(
       (np.ones(n_cells), (np.arange(n_cells), codes)),
       shape=(n_cells, n_unique),
   )
   composition = connectivities @ one_hot              # (n_cells, n_unique)
   totals = np.asarray(composition.sum(axis=1)).clip(min=1e-12)
   composition = composition / totals                  # per-row normalisation
   ```
   O(nnz × n_unique) — ~5 s for library × 50 uniques × 20M nnz.
2. **Vectorised penetration**:
   ```python
   n_mask = np.asarray(mask_csr.getnnz(axis=1)).flatten()
   penetration = pd.Series(n_mask >= threshold).groupby(adata.obs[stratify_key]).mean()
   ```
3. **Analytical isolation formula fix** — self-exclusion:
   - For mask `cross_dataset` (no different-dataset neighbours): `P(isolated) = ((n_same_dataset_i - 1) / (n_total - 1))^k_i`
   - For mask `cross_library`: `P(isolated) = ((n_same_library_i - 1) / (n_total - 1))^k_i`
   - Main plan's shorthand `(n_same_group / n_total)^k_i` is approximate; sub-plan uses the corrected self-excluded form.
4. **Bulk random graph sampling** (replaces per-cell `rng.choice`):
   ```python
   # Pre-sort degrees; sample in bulk
   max_k = degree_per_cell.max()
   samples = rng.integers(0, n_cells, size=(n_cells, max_k))
   # Post-hoc: reject self-hits by re-sampling only those positions
   self_mask = samples == np.arange(n_cells)[:, None]
   while self_mask.any():
       samples[self_mask] = rng.integers(0, n_cells, size=self_mask.sum())
       self_mask = samples == np.arange(n_cells)[:, None]
   # Truncate each row to its degree
   ```
   Construct sparse graph directly: `sp.csr_matrix((data, indices, indptr))` from sampled arrays — no dense intermediate.
5. **Parameterise `high_degree_multiplier: float = 1.5`** in `compute_neighbourhood_diagnostics` signature.
6. **Return dict keys** explicit for all cross-level masks: `penetration_cross_library`, `penetration_cross_dataset`, `penetration_between_{tech_name}` (one per technical covariate).

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD)

## Dependencies
- Sub-plans 01, 03 complete
- Sub-plan 04 not strictly required but random baseline uses same correlation code — import `_approach_B_per_mask` for reuse

## Tasks

### 1. `compute_neighbourhood_diagnostics`

```python
def compute_neighbourhood_diagnostics(
    adata,
    connectivities,
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
    k_reference: int = 50,
    penetration_thresholds: tuple[int, int] = (10, 25),
    high_degree_multiplier: float = 1.5,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Part 1 diagnostics on the KNN connectivity graph.

    Returns dict with:
        'degree': pd.Series (per-cell nnz count)
        'high_degree_obs_means': pd.DataFrame (numeric obs means for high-degree vs normal cells)
        'composition_{level}': pd.DataFrame (per-cell fraction of neighbours per covariate value)
                               for level in {library, dataset, technical_<name>}
        'penetration_{level}': pd.DataFrame (fraction of cells with >= threshold neighbours)
    """
```

### 2. Diagnostic details

**(1) Per-cell degree distribution**
- `degree = np.asarray(connectivities.getnnz(axis=1)).flatten()`
- Return as Series indexed by `adata.obs_names`

**(2) High-degree vs normal cells obs comparison**
- Define "high degree" as `degree > k_reference * 1.5` (configurable via parameter)
- Compute mean of each numeric obs column for high-degree cells vs rest
- Return DataFrame: columns = obs columns, rows = ["high_degree_mean", "normal_mean", "diff"]

**(3) Per-cell composition — apply to ALL 3 covariate levels**

For each level (library, dataset, each technical covariate):
- Get covariate values, one-hot encode per unique value
- For each cell i: `composition[i, c] = sum(connectivities[i, j] for j where obs[j, level] == c) / total_weight[i]`
- Return DataFrame (n_cells × n_unique_values_at_level)
- Produces separate DataFrame per level

**(4) Integration penetration — apply to ALL 3 covariate levels**

For each cross-level mask (cross_library, cross_dataset, between_{technical}):
- Use masks from `construct_neighbour_masks`
- For each cell: count `n_neighbours_{cross_level}`
- Compute fraction of cells with `n_neighbours >= threshold` for each threshold in `penetration_thresholds`
- Stratify by the cell's own value at that level (per-dataset penetration, per-library penetration, etc.)
- Return DataFrame: rows = stratification values, columns = thresholds

### 3. Random KNN baseline

```python
def compute_random_knn_baseline(
    adata,
    X_normalised_markers,   # dense (n_cells, n_markers) — reuse from main compute
    degree_per_cell,        # (n_cells,) — match actual per-cell neighbour counts
    library_key: str,
    dataset_key: str | None = None,
    n_random_graphs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Compute expected correlation under random neighbour assignment.

    For each cell i with degree k_i, sample k_i random cells (excluding self),
    then compute correlation metrics on this random graph using same Approach B.

    Returns per-cell random-baseline correlation values matching the output
    columns of compute_marker_correlation (for each mask).

    For isolation fraction, analytical:
        P(isolated_cross_dataset | k_i) = (n_same_dataset / n_total)^k_i
    """
```

**Implementation** (bulk vectorised sampling per correction #4):
```python
rng = np.random.default_rng(random_state)
max_k = degree_per_cell.max()
# Bulk sample: (n_cells, max_k) random neighbour indices
samples = rng.integers(0, n_cells, size=(n_cells, max_k))
# Reject self-hits by re-sampling those positions
self_mask = samples == np.arange(n_cells)[:, None]
while self_mask.any():
    samples[self_mask] = rng.integers(0, n_cells, size=self_mask.sum())
    self_mask = samples == np.arange(n_cells)[:, None]
# Truncate each row to its actual degree: build CSR directly
indptr = np.zeros(n_cells + 1, dtype=np.int64)
indptr[1:] = np.cumsum(degree_per_cell)
indices = np.concatenate([samples[i, :degree_per_cell[i]] for i in range(n_cells)])
data = np.ones(len(indices), dtype=np.float32)  # uniform weights
random_conn = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))
```
- Apply `construct_neighbour_masks` on `random_conn` to get random masks
- Run `_approach_B_per_mask` to get `corr_avg_random_{mask}`
- Return DataFrame

### 4. Analytical isolation baseline

```python
def compute_analytical_isolation_baseline(
    adata,
    degree_per_cell,
    covariate_key: str,
) -> pd.Series:
    """Analytical expected isolation fraction under random KNN.

    For each cell: P(all k_i random neighbours have same covariate value as self)
                 = ((n_same_group - 1) / (n_total - 1))^k_i  # self-exclusion on both sides

    Returns per-cell probability of being isolated from cross-{covariate} neighbours
    under random assignment.
    """
```

### 5. Add to `__all__`

Export: `compute_neighbourhood_diagnostics`, `compute_random_knn_baseline`, `compute_analytical_isolation_baseline`.

## Test cases
- Degree of self-loops excluded (diagonal = 0 after `connectivities.setdiag(0)`)
- Composition sums to 1 per cell (for each level, fractions sum to 1)
- Penetration: at threshold 0, fraction = 1 (trivial)
- Random baseline correlation should be near 0 on average (random gene vectors have low mean correlation)
- Analytical isolation matches empirical random baseline within Monte Carlo error

## Verification
- Run on immune integration: produces composition tables for library (50+ columns), dataset (7 columns), no technical
- Run on embryo: adds technical composition columns for Embryo and Experiment
- Run random baseline: `corr_avg_random_cross_dataset` should be lower than model's `corr_avg_cross_dataset` for a good integration model
