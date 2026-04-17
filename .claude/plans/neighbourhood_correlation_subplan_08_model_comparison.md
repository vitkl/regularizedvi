# Sub-Plan 08: Cross-Model Comparison + Integration Failure Rate

**Parent plan**: `neighbourhood_correlation_plan.md` (Model Comparison for Integration Failure Detection)

## Review Corrections (applied)

1. **Explicit `np.nanmax` for best-achievable envelope**:
   ```python
   stack = np.stack([per_model_metrics[m][metric_name].to_numpy() for m in models], axis=0)
   best_achievable = np.nanmax(stack, axis=0)  # NaN-skipped max
   ```
   When ALL models have NaN for a cell, result is NaN (no model integrates it) — consensus isolated.
2. **Integration failure rate handles NaN best_achievable**:
   ```python
   best = compute_best_achievable(cross_model_df, metric_name)
   model_values = cross_model_df[(model_name, metric_name)].to_numpy()
   # failure: best is finite AND > threshold_high AND model is NaN or < threshold_low
   failure = np.isfinite(best) & (best > threshold_high) & (
       np.isnan(model_values) | (model_values < threshold_low)
   )
   rate = failure.sum() / len(failure)
   ```
   NaN `best_achievable` cells are NOT counted as failures (they're consensus-isolated, handled separately).
3. **`subset_mask` application order**: applied BEFORE histogram binning. Add to docstring:
   > "Subset filter is applied to both distributions before computing OVL, so the result reflects the subset's integration quality."
   Provide helper `make_cell_subset_mask(adata, column, values)` for TEA-seq-style subsetting.
4. **Contingency category naming** aligned with main plan 3×3 table (symmetrical):
   - `(A-hi, B-hi)`: `both_succeed`
   - `(A-hi, B-low)`: `A_ok_B_wrong_pairing`
   - `(A-hi, B-no)`: `A_ok_B_isolates`
   - `(A-low, B-hi)`: `B_ok_A_wrong_pairing`
   - `(A-low, B-low)`: `both_wrong_pairing`
   - `(A-low, B-no)`: `A_wrong_B_isolates`
   - `(A-no, B-hi)`: `B_ok_A_isolates`
   - `(A-no, B-low)`: `A_isolates_B_wrong`
   - `(A-no, B-no)`: `both_isolate_ambiguous`
5. **Tissue-group integration interface**: requires `n_neighbours_within_{tech}_cross_dataset` column produced by sub-plan 04. Add column requirement to sub-plan 04's Output Columns section (already noted in sub-plan 04 corrections).
6. **Interface to sub-plan 06**: add explicit section "Integration with Decision Tree":
   > "Output of `flag_consensus_isolated(cross_model_df)` (boolean per cell) is passed to `classify_failure_modes(..., model_comparison_result=consensus_flag)`. Inside `classify_failure_modes`, cells with `n_neighbours_cross_dataset == 0` branch: True (consensus isolated, no model connects) → XD-0a; False (some model connects) → XD-0b."
7. **Vectorisation note**: all cross-model operations use stacked arrays (n_models × n_cells) and `np.nanmax`/`np.nanmin`/boolean ops. No per-cell Python loops.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD)

## Dependencies
- Sub-plans 04 (metrics_df per model), 07 (distribution overlap helper)

## Tasks

### 1. Cross-model DataFrame assembly

```python
def assemble_cross_model_metrics(
    per_model_metrics: dict[str, pd.DataFrame],
    shared_cell_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Combine per-model metrics into MultiIndex DataFrame (cell × model).

    Parameters
    ----------
    per_model_metrics : dict[model_name, metrics_df]
        Output of compute_marker_correlation for each model.
        All DataFrames must share the same cell index (shared_cell_index).

    Returns
    -------
    DataFrame with MultiIndex columns: (model_name, metric_name).
    Rows indexed by shared_cell_index.
    """
```

### 2. Best-achievable envelope

```python
def compute_best_achievable(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
) -> pd.Series:
    """For each cell, compute max over models of the given metric.

    If ANY model achieves high correlation, matching is possible.
    """
```

### 3. Integration failure rate per model

```python
def compute_integration_failure_rate(
    cross_model_df: pd.DataFrame,
    model_name: str,
    metric_name: str = "corr_avg_cross_dataset",
    threshold_high: float = 0.4,
    threshold_low: float = 0.2,
) -> float:
    """Fraction of cells that CAN be integrated (per ensemble) but this model fails.

    Failure condition:
        best_achievable[cell] > threshold_high
        AND (model_value[cell] is NaN OR model_value[cell] < threshold_low)
    """
```

### 4. Pairwise distribution overlap

```python
def compute_model_pair_overlaps(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
    subset_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute pairwise OVL between models' metric distributions.

    Parameters
    ----------
    subset_mask : optional per-cell boolean (e.g., restrict to TEA-seq cells or
                  cells in a given tissue group).

    Returns
    -------
    Square DataFrame (n_models × n_models), OVL values.
    """
```

Uses `compute_distribution_overlap` from sub-plan 07.

### 5. 3×3 contingency per cell

```python
def compute_contingency_per_cell(
    cross_model_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_name: str = "corr_avg_cross_dataset",
    threshold_high: float = 0.4,
    threshold_low: float = 0.2,
) -> pd.DataFrame:
    """Per-cell contingency classification for a model pair.

    Returns DataFrame with one column 'category' per cell (symmetrical names):
        'both_succeed', 'A_ok_B_wrong_pairing', 'A_ok_B_isolates',
        'B_ok_A_wrong_pairing', 'both_wrong_pairing', 'A_wrong_B_isolates',
        'B_ok_A_isolates', 'A_isolates_B_wrong', 'both_isolate_ambiguous'
    """
```

### 6. Tissue-group integration (pure technical baseline)

```python
def compute_tissue_group_integration(
    cross_model_df: pd.DataFrame,
    adata,
    technical_covariate_key: str,  # e.g. "tissue"
    model_name: str,
    metric_name: str = "corr_avg_cross_dataset",
) -> float:
    """Mean cross-dataset correlation for cells in same technical group.

    Restricts to cells where ANY cross-dataset neighbour is in the same
    technical covariate value (same tissue, same experiment, etc.).

    This is the unambiguous integration metric — within a tissue group,
    the same cell types MUST exist across datasets.
    """
```

### 7. Update `summarise_marker_correlation` with H13, H14

Extend sub-plan 07 output series with:
- **H13 `integration_failure_rate`**: computed if cross_model_df provided
- **H14 `tissue_group_integration`**: computed if technical_covariate_key provided

### 8. Consensus isolated cells flag

```python
def flag_consensus_isolated(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
    n_neighbours_col: str = "n_neighbours_cross_dataset",
    min_corr: float = 0.3,
) -> pd.Series:
    """Boolean flag: cells that NO model integrates cross-dataset.

    True when for ALL models: n_neighbours == 0 OR corr < min_corr.
    These are candidate dataset-specific populations (label-free detection).
    """
```

### 9. Add to `__all__`

Export: `assemble_cross_model_metrics`, `compute_best_achievable`, `compute_integration_failure_rate`, `compute_model_pair_overlaps`, `compute_contingency_per_cell`, `compute_tissue_group_integration`, `flag_consensus_isolated`.

## Test cases
- With 2 identical models: `integration_failure_rate = 0`, `OVL = 1`
- With 1 model + 1 random-permuted model: OVL low, failure rate high for the worse model
- Consensus isolated flag: cells that fail in all models are correctly flagged
- 3×3 contingency categories sum to n_cells

## Verification
- Run on immune models (scvi_baseline vs regularizedVI) restricted to TEA-seq cells
  - Expect low OVL between scvi_baseline and regularizedVI
  - Expect high integration_failure_rate for scvi_baseline
- Check consensus_isolated set is small fraction of cells for a healthy integration
