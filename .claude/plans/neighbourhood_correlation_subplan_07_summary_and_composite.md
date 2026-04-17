# Sub-Plan 07: Per-Model Summary + Headline Metrics + Composite Score

**Parent plan**: `neighbourhood_correlation_plan.md` (Headline Metrics, Composite score, Isolation score normalisation)

## Review Corrections (applied)

1. **Isolation normalisation side-of-hierarchy fix**:
   - Mask `cross_library` (different library, same dataset): "isolated from cross_library" means no neighbours with different library but same dataset. Expected random: `P = (n_same_library_i / (n_total - 1))^k_i` — chance all random picks are same library (thus no cross-library).
   - Mask `cross_dataset`: "isolated from cross_dataset" means no neighbours with different dataset. Expected: `P = (n_same_dataset_i / (n_total - 1))^k_i`.
   - Mask `between_libraries`: `P = ((n_same_library_i - 1) / (n_total - 1))^k_i` (exclude self from same-library count).
   - Clarify per-mask formulas in docstring; remove contradictory parenthetical from original.
2. **OVL normalisation**: compute `h_x / h_x.sum()` and `h_y / h_y.sum()` separately (each a proper probability distribution), then `OVL = sum(min(p_x, p_y))`. Do NOT divide by `sum(hist_x)` — that is wrong when sample sizes differ:
   ```python
   h_x, _ = np.histogram(x[~np.isnan(x)], bins=n_bins, range=range_)
   h_y, _ = np.histogram(y[~np.isnan(y)], bins=n_bins, range=range_)
   p_x = h_x / h_x.sum()
   p_y = h_y / h_y.sum()
   ovl = np.minimum(p_x, p_y).sum()
   ```
3. **`min_neighbours` clarification**: default 1 matches main plan's "cells with >=1 cross-X neighbours" filter for H3/H7. Distinct from penetration thresholds (10/25) in sub-plan 05 — different purposes. Add docstring note.
4. **`stratify_by` as list[str]**: allow passing multiple obs columns. Either loop over columns returning dict[str, DataFrame], or accept a single column and document callers to invoke multiple times.
5. **Stratified summary uses NaN-safe aggregation**: `groupby(...).agg(lambda g: np.nanmedian(g))` or `pd.DataFrame.agg(np.nanmedian)`. Percentiles via `np.nanpercentile(g, [10,25,50,75,90])`.
6. **H5/H9 can exceed 1**: document that `clip(H5, 0, 2)` in the composite handles values > 1 (model isolates MORE than random → severe under-integration). H5/H9 > 1 is a red flag.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD)

## Dependencies
- Sub-plans 04 (metrics), 05 (random baseline), 06 (leaves)
- Read-only: `src/regularizedvi/plt/_integration_metrics.py` (look at `_classify_metric_col` + heatmap color grouping conventions to match)

## Tasks

### 1. Per-model summary function

```python
def summarise_marker_correlation(
    metrics_df: pd.DataFrame,       # from compute_marker_correlation
    random_baseline_df: pd.DataFrame | None = None,
    leaf_df: pd.DataFrame | None = None,
    min_neighbours: int = 1,        # cells with >=1 mask neighbours for cross-metrics
) -> pd.Series:
    """Compute per-model headline metrics H1-H14.

    Returns pd.Series indexed by metric name:
        corr_within_library, corr_consistency
        corr_cross_library, corr_gap_library, isolation_norm_cross_library, discrepancy_cross_library
        corr_cross_dataset, corr_gap_dataset, isolation_norm_cross_dataset, discrepancy_cross_dataset
        distrib_overlap_library, distrib_overlap_dataset
        (H13, H14 added later by model comparison sub-plan)

    See plan.md "Headline Metrics" for definitions.
    """
```

### 2. Metric formulas

Implement each H1–H12 per plan:

- **H1 `corr_within_library`**: `np.nanmedian(metrics_df['corr_avg_same_library'])`
- **H2 `corr_consistency`**: `np.nanmedian(metrics_df['corr_std_same_library'])`
- **H3 `corr_cross_library`**: median of `corr_avg_cross_library` restricted to cells where `n_neighbours_cross_library >= min_neighbours`
- **H4 `corr_gap_library`**: `H1 - H3`
- **H5 `isolation_norm_cross_library`**: `isolation_frac / expected_random_isolation_frac`
- **H6 `discrepancy_cross_library`**: `np.nanmedian(metrics_df['corr_discrepancy_cross_library'])`
- **H7-H10**: same pattern for cross_dataset
- **H11 `distrib_overlap_library`**: OVL of distributions of `corr_avg_same_library` vs `corr_avg_cross_library` — see section 4
- **H12 `distrib_overlap_dataset`**: OVL of `corr_avg_same_library` vs `corr_avg_cross_dataset`

### 3. Isolation normalisation

```python
def compute_isolation_norm(
    metrics_df: pd.DataFrame,
    adata,
    mask_name: str,              # e.g. "cross_library", "cross_dataset"
    covariate_key: str,          # which obs column defines the group
) -> float:
    """Observed isolation / expected random isolation.

    Observed: fraction of cells with n_neighbours_{mask} == 0.
    Expected random: analytical P(all k_i random neighbours same group)
                    = (n_same_group_cell_i / n_total)^k_i averaged over cells.
    """
```

Per-mask expected random isolation (see correction #1 for exact formulas):
- `cross_library`: P = `(n_same_library_i / (n_total - 1))^k_i` — chance random picks are all same library (= no cross-library neighbours)
- `cross_dataset`: P = `(n_same_dataset_i / (n_total - 1))^k_i` — all same dataset
- `between_libraries`: P = `((n_same_library_i - 1) / (n_total - 1))^k_i` — self-excluded from same-library count

### 4. Distribution overlap (OVL)

```python
def compute_distribution_overlap(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 50,
    range_: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Overlap coefficient between two distributions.

    OVL = integral of min(f_x(v), f_y(v)) dv
        ≈ sum(min(p_x, p_y))  where p_x = h_x/h_x.sum(), p_y = h_y/h_y.sum() (normalised separately)

    Uses histogram approximation on a fixed range.
    Drops NaN values before computing.
    """
```

Return value in [0, 1]: 1 = identical distributions, 0 = no overlap.

### 5. Composite score

```python
def compute_composite_score(
    headline: pd.Series,
    has_dataset: bool = True,
) -> pd.Series:
    """Compute total, bio_conservation, batch_correction (mirrors scIB 60/40)."""
```

Formulas from plan:
```
bio_conservation = H1 (corr_within_library)

library_integration = 0.4 * H3 + 0.3 * (1 - clip(H5, 0, 2)) + 0.3 * H11
dataset_integration = 0.4 * H7 + 0.3 * (1 - clip(H9, 0, 2)) + 0.3 * H12

if has_dataset:
    batch_correction = 0.5 * library_integration + 0.5 * dataset_integration
else:
    batch_correction = library_integration

total = 0.6 * bio_conservation + 0.4 * batch_correction
```

### 6. Stratified summary

```python
def stratified_summary(
    metrics_df: pd.DataFrame,
    stratify_by: str | list[str],  # obs column name(s) — if list, produce separate summary per column
    metrics_to_report: list[str] | None = None,
) -> pd.DataFrame:
    """Per-stratum (per dataset / per library / per leaf) summary of headline metrics.

    Returns DataFrame: rows = stratum values, columns = metric name, values = median.
    Also adds percentile columns (10, 25, 50, 75, 90) for each corr_avg_{mask}.
    """
```

### 7. Add to `__all__`

Export: `summarise_marker_correlation`, `compute_isolation_norm`, `compute_distribution_overlap`, `compute_composite_score`, `stratified_summary`.

## Test cases
- Single-dataset input: H7-H10 are NaN, composite uses only library integration
- Isolation norm ≈ 1 when model and random perform equally
- Isolation norm < 1 for a well-integrating model
- OVL of identical distributions = 1; OVL of disjoint = 0
- Composite score is deterministic given same headline inputs

## Verification
- Run on one immune model, produce headline Series with 12 entries
- Run composite score, verify value in reasonable range [0, 1]
- Compare H1 ranking across 8 immune models — should correlate with existing scIB bio_conservation ranking
