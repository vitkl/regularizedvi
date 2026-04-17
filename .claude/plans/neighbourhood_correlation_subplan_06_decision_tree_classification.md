# Sub-Plan 06: Decision Tree Leaf Assignment

**Parent plan**: `neighbourhood_correlation_plan.md` (Decision Tree: Failure Mode Classification)

## Review Corrections (applied)

1. **Leaf count: 25 total** (WL: 6, XL: 7, XD: 12). Previous doc said 21.
2. **All XD leaves must be implemented** (currently stubbed). Add:
   - `XD-0c` Cascaded under-integration (cross-library was under-integrated)
   - `XD-0d` Compounded failure (cross-library failed)
   - `XD-2` Spurious high correlation (HIGH but at/below random via DIM 5)
   - `XD-3` Partial cross-dataset integration (HIGH + above-random + MIXED)
   - `XD-4b` Systematic positive failure (HIGH WL, LOW XD, LOW XL, HOMOGENEOUS)
   - `XD-5b` Semi-random cross-dataset pairing (MIXED + both gene groups LOW)
3. **Vectorised classification via `np.select`** (replace per-row `df.apply`):
   ```python
   has_sl = df['n_neighbours_same_library'].to_numpy() > 0
   hi     = df['corr_avg_same_library'].to_numpy() >= th_high
   homog  = df['corr_std_same_library'].to_numpy() <= th_std
   above_random = df['corr_avg_same_library'].to_numpy() > random_corr_sl
   conditions = [
       ~has_sl,
       has_sl & hi & homog,
       has_sl & hi & ~homog,
       has_sl & ~hi & ~above_random,
       has_sl & ~hi & above_random & homog,
       has_sl & ~hi & above_random & ~homog,
   ]
   choices = ['WL-0_orphan', 'WL-1_ideal', 'WL-2_merged_related',
              'WL-3_noisy', 'WL-4_false_merge_confident', 'WL-5_false_merge_partial']
   wl_leaves = np.select(conditions, choices, default='WL-unknown')
   ```
   Repeat for XL and XD. 100-1000× faster than `df.apply(axis=1)` for n>100k cells.
4. **Dimension 5 in XD branch** for XD-1 vs XD-2: add `above_random_xd = corr_avg_cross_dataset > random_corr_xd` check.
5. **Gene group comparison helper** (Dimension 6) — explicit pseudocode:
   ```python
   def _compare_gene_groups(cell_idx, mask_name, broad_df, specific_df, thresholds):
       broad_hi    = broad_df.loc[cell_idx, f'corr_avg_{mask_name}'] >= thresholds['high']
       specific_lo = specific_df.loc[cell_idx, f'corr_avg_{mask_name}'] < thresholds['high']
       if broad_hi and specific_lo:
           return 'broad_high_specific_low'
       # ... other cases
   ```
   Apply only at branching leaves (WL-4, XL-3, XD-4a, XD-5a). Produces sub-label column `leaf_{level}_sublabel`.
6. **Model comparison interface**: add `model_comparison_result: pd.Series | None = None` parameter (cell-indexed bool: "other model connects cross-dataset"). When None, emit `XD-0_isolated_unknown`. When provided: True → XD-0b (under-integration), False → XD-0a (dataset-enriched).
7. **Severity ordering** for combined `failure_mode` — `XD-0a` is NOT a failure (dataset-enriched cell type is biologically informative, not a model problem). Exclude from failure severity.
   - WL severity (worst to best): WL-0 > WL-3 > WL-4 > WL-5 > WL-2 > WL-1
   - XL severity: XL-0b > XL-5 > XL-0a > XL-3 > XL-4 > XL-2 > XL-1
   - XD severity: XD-0d > XD-6 > XD-4b > XD-4a > XD-0b > XD-5a > XD-5b > XD-0c > XD-2 > XD-3 > XD-1
   - **Non-failure leaves** (report as `failure_mode = 'not_a_failure'` with sub-label): XD-0a (dataset-enriched), XL-0a with no shared cell type context (if model comparison later reveals this). These are legitimate biological findings, not model problems.
   - Combined `failure_mode` column: take max severity across levels. If all levels are non-failure (XD-1, XL-1, WL-1) → `'ideal'`. If any level is non-failure but others have failures → report the failure.
   - The subjective orderings above are TENTATIVE — open to adjustment based on what correlates with model rank in scIB benchmarks.
8. **Handle `dataset_key=None`**: skip XL and XD classification entirely; return DataFrame with only `leaf_within_library` column.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD)

## Dependencies
- Sub-plans 04 (correlation output DataFrame), 05 (random baseline)

## Tasks

### 1. Main classification function

```python
def classify_failure_modes(
    metrics_df: pd.DataFrame,       # output of compute_marker_correlation
    random_baseline_df: pd.DataFrame | None = None,  # output of compute_random_knn_baseline
    gene_group_metrics: dict[str, pd.DataFrame] | None = None,  # {group_name: metrics_df} per gene group
    threshold_high: float | None = None,   # if None, use 25th percentile of within_library
    threshold_low: float | None = None,    # if None, use 5th percentile
    std_threshold: float | None = None,    # if None, use median of corr_std
) -> pd.DataFrame:
    """Assign each cell to a decision tree leaf.

    Leaves (from plan):
        WL-0 (Orphan), WL-1..WL-5 (within-library variants)
        XL-0a..XL-5 (cross-library variants, conditioned on WL result)
        XD-0a..XD-6 (cross-dataset variants, conditioned on WL + XL)

    Returns DataFrame indexed by cell with columns:
        'leaf_within_library'    (WL-0 through WL-5)
        'leaf_cross_library'     (XL-0a through XL-5 or NaN)
        'leaf_cross_dataset'     (XD-0a through XD-6 or NaN)
        'failure_mode'           (combined label, most severe)
    """
```

### 2. Vectorised classification via `np.select` (NOT per-row `df.apply`)

**Within-library level** (all conditions as boolean arrays):

```python
# Pre-compute boolean arrays (vectorised)
has_sl = df['n_neighbours_same_library'].to_numpy() > 0
corr_sl = df['corr_avg_same_library'].to_numpy()
std_sl = df['corr_std_same_library'].to_numpy()
hi = corr_sl >= th_high
homog = std_sl <= th_std
above_random = corr_sl > random_corr_sl if random_corr_sl is not None else np.ones(n, dtype=bool)

conditions = [
    ~has_sl,
    has_sl & hi & homog,
    has_sl & hi & ~homog,
    has_sl & ~hi & ~above_random,
    has_sl & ~hi & above_random & homog,
    has_sl & ~hi & above_random & ~homog,
]
choices = ['WL-0_orphan', 'WL-1_ideal', 'WL-2_merged_related',
           'WL-3_noisy', 'WL-4_false_merge_confident', 'WL-5_false_merge_partial']
wl_leaves = np.select(conditions, choices, default='WL-unknown')
```

**Cross-library level** (conditioned on WL result, same vectorised pattern):

```python
has_xl = df['n_neighbours_cross_library'].to_numpy() > 0
corr_xl = df['corr_avg_cross_library'].to_numpy()
std_xl = df['corr_std_cross_library'].to_numpy()
hi_xl = corr_xl >= th_high
wl_was_ideal = np.isin(wl_leaves, ['WL-1_ideal', 'WL-2_merged_related'])

conditions_xl = [
    ~has_xl & wl_was_ideal,
    ~has_xl & ~wl_was_ideal,
    has_xl & hi_xl & (std_xl <= th_std),
    has_xl & hi_xl & (std_xl > th_std),
    has_xl & ~hi_xl & wl_was_ideal & (std_xl <= th_std),
    has_xl & ~hi_xl & wl_was_ideal & (std_xl > th_std),
    has_xl & ~hi_xl & ~wl_was_ideal,
]
choices_xl = ['XL-0a_under_integration', 'XL-0b_compounded_failure',
              'XL-1_ideal', 'XL-2_partial', 'XL-3_wrong_pairing',
              'XL-4_forced_distinct', 'XL-5_poor_model']
xl_leaves = np.select(conditions_xl, choices_xl, default='XL-unknown')
```

**Cross-dataset level**: same pattern with 12 leaves. Key additions per correction #2:
- `XD-0c`: `~has_xd & xl_is('XL-0a_under_integration')`
- `XD-0d`: `~has_xd & xl_is_failure`
- XD-0a vs XD-0b: requires `model_comparison_result` boolean. When None, emit `XD-0_isolated_unknown`.
- `XD-2`: `has_xd & hi_xd & ~above_random_xd` (spurious high correlation, DIM 5)
- `XD-3`: `has_xd & hi_xd & above_random_xd & ~homog_xd` (partial)
- `XD-4a`: `has_xd & ~hi_xd & wl_was_ideal & xl_was_ideal & homog_xd`
- `XD-4b`: `has_xd & ~hi_xd & wl_was_ideal & ~xl_was_ideal & homog_xd`
- `XD-5a`: `has_xd & ~hi_xd & wl_was_ideal & ~homog_xd & broad_high & specific_low` (gene groups)
- `XD-5b`: `has_xd & ~hi_xd & wl_was_ideal & ~homog_xd & ~broad_high`
- `XD-6`: `has_xd & ~hi_xd & ~wl_was_ideal`

This is 100-1000× faster than `df.apply(axis=1)` for 416k cells.

### 3. Threshold selection

Defaults when user doesn't specify:
- `threshold_high = np.nanpercentile(metrics_df['corr_avg_same_library'], 25)`
- `threshold_low = np.nanpercentile(metrics_df['corr_avg_same_library'], 5)`
- `std_threshold = np.nanmedian(metrics_df['corr_std_same_library'])`

Thresholds are adaptive: "high" means above the lower quartile of the within-library distribution (best-case scenario for that model).

### 4. Gene group comparison (DIM 6)

For leaves that branch on gene group agreement (WL-4, XL-3, XD-4a, XD-5a):
- Compare `corr_avg_{mask}_broad_markers` vs `corr_avg_{mask}_subtype_markers` (requires running `compute_marker_correlation` separately per gene group)
- If gene group metrics provided: add sub-label `_broad_high_specific_low` or `_both_low` etc.

### 5. Combined `failure_mode` column

A single consolidated label per cell combining WL + XL + XD results. Priority: deepest failure wins (XD > XL > WL). If XD leaf is "ideal", report WL/XL as "ideal" too.

### 6. Summary function

```python
def summarize_failure_modes(
    leaf_df: pd.DataFrame,
    stratify_by: list[str] | None = None,
) -> pd.DataFrame:
    """Count cells per leaf, optionally stratified by obs column(s).

    Returns DataFrame with leaf counts and fractions.
    """
```

### 7. Add to `__all__`

Export `classify_failure_modes`, `summarize_failure_modes`.

## Test cases
- Cell with all-NaN cross-dataset metrics gets `leaf_cross_dataset = 'XD-0_isolated'`
- Cell with high within-library, high cross-library, high cross-dataset → WL-1 + XL-1 + XD-1
- Cell with zero same-library neighbours → WL-0 and everything downstream is NaN/cascaded
- Thresholds computed from data are reproducible across runs

## Verification
- Run on one immune model, print per-leaf counts
- UMAP colored by leaf (sub-plan 09) shows spatial coherence
- Compare leaf distribution across models — varying models should have different leaf distributions
