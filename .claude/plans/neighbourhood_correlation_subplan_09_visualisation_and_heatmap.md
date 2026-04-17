# Sub-Plan 09: Visualisation + Benchmarker Heatmap Extension

**Parent plan**: `neighbourhood_correlation_plan.md` (Visualisations section)

## Review Corrections (applied)

1. **V1 default columns expanded**: include all `corr_avg_{mask}`, `corr_std_{mask}`, `corr_discrepancy_{mask}`, `n_neighbours_{mask}`, plus derived `corr_norm_by_library_{mask}`, `corr_norm_by_all_{mask}`. Optional reduced set via parameter.
2. **V5 per-mask grid**: `plot_failure_mode_scatter` must be a grid with one subplot per mask for avg×mean (3 plots) plus the standalone comparisons (sb×xb, sb×xd, xb×xd, discrepancy×avg_xd, model×random). Total ~8 panels.
3. **V14 random baseline source**: `plot_metric_hist2d(metrics_df_combined, "corr_avg_same_library", "corr_avg_random_same_library")` where random columns come from sub-plan 05 `compute_random_knn_baseline` output (merge into `metrics_df` upstream).
4. **Heatmap extension — all required edits to `_integration_metrics.py`**:
   - Add `_NEIGHBOURHOOD_METRICS` set and `_NEIGHBOURHOOD_PREFIXES` tuple
   - Register new colour in `_GROUP_COLORS` (suggest `"#1f77b4"` blue)
   - Add branch to `_classify_metric_col` (around line 460)
   - Insert `"neighbourhood"` in `ordered_metric_cols` (line 445), `group_positions` (line 562), `group_labels` (line 566) — between `"batch"` and hyperparameters
   - No new parameter needed; auto-detect via `_classify_metric_col`
   - H13/H14 are per-model scalars — flow into `scib_df` alongside other metrics
5. **Rasterisation**: add `rasterized=True` on scatter/UMAP calls (400k points → prevents SVG bloat).
6. **Diverging norm for correlations**: use `matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax_abs, vmax=vmax_abs)` with `vmax_abs = max(|min|, |max|)` of finite values — ensures colour symmetry around 0.
7. **Hist2d log scaling**: replace `log_counts: bool` flag with `norm=LogNorm()` from `matplotlib.colors` — cleaner.
8. **NaN handling in UMAP**: draw NaN cells first in light grey (`#dddddd`, smaller size), then overlay valid cells. Use `np.isfinite` mask.
9. **V9 per-library layout**: 7 datasets × ~10 libraries = 70 subplots is too many. Use one row per dataset with libraries as overlaid KDE lines (coloured by library) — cuts to 7 panels.
10. **Test cases added**: heatmap renders with neighbourhood columns only (no scIB); symmetric diverging norm verified; rasterised SVG size sanity check.

## Primary files
- `src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD: UMAP and hist2d plotting)
- `src/regularizedvi/plt/_integration_metrics.py` (EXTEND: benchmarker heatmap with new columns)

## Dependencies
- Sub-plans 06 (leaf labels), 07 (headline metrics), 08 (cross-model)
- Read-only: `src/regularizedvi/plt/_integration_metrics.py` — study `_classify_metric_col`, `_GROUP_COLORS`, `plot_integration_heatmap` to add new column group

## Tasks

### 1. UMAP plotting (V1, V12)

```python
def plot_marker_correlation_umap(
    adata,
    metrics_df: pd.DataFrame,
    columns: list[str] | None = None,       # default: all corr_avg_{mask} + discrepancy
    leaf_df: pd.DataFrame | None = None,    # if provided, add leaf panel
    umap_key: str = "X_umap",
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    cmap_divergent: str = "RdBu_r",
    cmap_sequential: str = "viridis",
) -> "Figure":
    """Grid of UMAPs colored by each per-cell metric.

    - Correlation metrics: diverging colormap centered at 0
    - Counts (n_neighbours): sequential colormap
    - Leaf labels: categorical palette
    """
```

Reuse existing `plot_umap_comparison` pattern from `_integration_metrics.py` if available.

### 2. hist2d plots (V2–V6, V14)

```python
def plot_metric_hist2d(
    metrics_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 100,
    range_: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cmap: str = "viridis",
    log_counts: bool = True,
    title: str | None = None,
    ax=None,
) -> "Axes":
    """2D histogram of two metrics (one per cell)."""

def plot_failure_mode_scatter(
    metrics_df: pd.DataFrame,
    figsize: tuple[float, float] = (16, 12),
) -> "Figure":
    """6-panel grid: sb×xb, sb×xd, xb×xd, avg×mean per mask, discrepancy×avg_xd, model×random."""
```

### 3. Overlaid distribution histograms (V7, V8)

```python
def plot_distribution_overlap(
    metrics_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str | None = None,
    label_b: str | None = None,
    bins: int = 50,
    range_: tuple[float, float] = (-1.0, 1.0),
    ax=None,
) -> "Axes":
    """Overlay two histograms for visual distribution overlap comparison."""
```

### 4. Per-library distribution (V9)

```python
def plot_per_library_distributions(
    metrics_df: pd.DataFrame,
    adata,
    metric_col: str = "corr_avg_cross_library",
    library_key: str = "batch",
    dataset_key: str = "dataset",
) -> "Figure":
    """Histogram of metric per library, faceted by dataset."""
```

### 5. Isolation bar chart (V10)

```python
def plot_isolation_bars(
    per_model_headlines: dict[str, pd.Series],  # model_name -> headline series
    metric: str = "isolation_norm_cross_dataset",
    stratify_by_dataset: bool = True,
) -> "Figure":
    """Bar chart: normalised isolation per dataset × model."""
```

### 6. Leaf distribution stacked bar (V13)

```python
def plot_leaf_distribution(
    per_model_leaves: dict[str, pd.Series],  # model -> per-cell leaf series
) -> "Figure":
    """Stacked bar: fraction of cells per leaf, per model."""
```

### 7. Extend benchmarker heatmap (V11) — `_integration_metrics.py`

- Study `_classify_metric_col` (around line 460 per exploration). It has sets `_BIO_METRICS`, `_BATCH_METRICS`. ADD a new set `_NEIGHBOURHOOD_METRICS` containing the H1–H14 metric names
- Add a new column group color for neighbourhood correlation (distinct from bio/batch)
- Update `plot_integration_heatmap` to handle the new group label and color
- The heatmap should show the new H1-H14 columns alongside existing silhouette/LISI/ARI/NMI columns

**Backward compatibility**: existing callers should continue to work — only ADD new metric names, don't rename existing ones.

### 8. Add to `__all__`

`_neighbourhood_correlation.py`: Export all plotting functions.
`_integration_metrics.py`: No new exports needed (extending existing functions).

## Test cases
- `plot_marker_correlation_umap` produces a figure without errors on test data
- Heatmap renders with new columns included
- Hist2d handles NaN values (drop before binning)
- Per-library plots handle datasets with only 1 library (no cross-library — skip or show empty)

## Verification
- Generate all plots for one immune model; visually inspect
- Generate heatmap across all 8 immune models with new H1-H14 columns
- Verify colour groups are visually distinguishable (bio / batch / neighbourhood / hyperparameters)
