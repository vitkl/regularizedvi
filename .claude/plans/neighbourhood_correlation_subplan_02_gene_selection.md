# Sub-Plan 02: Gene Selection (`select_marker_genes`)

**Parent plan**: `neighbourhood_correlation_plan.md` (Part 2 → Gene Selection)

## Review Corrections (applied)

1. **Specificity axis clarification**: with `label_averages` shape `(n_genes, n_labels)`, use `(label_averages.T / label_averages.sum(axis=1)).T` — this produces per-gene specificity (each row sums to 1). Matches user's intent: "what fraction of this gene's total expression across labels comes from each label". User's `sum(0, keep_dims=True)` notation assumed transposed orientation. Document this in the docstring with a small worked example.
2. **Curated genes MUST be included in `union`**: compute `union = data_driven_union ∪ curated_genes`, not just data_driven. User explicitly said both sets used simultaneously.
3. **Named gene groups required**: return dict must include the 4 main-plan groups:
   - `all_markers` = union (above)
   - `broad_lineage_markers` = specific at `level_2` OR `level_3`
   - `cell_type_markers` = specific at `harmonized_annotation` (NOT level_1 — per main plan correction)
   - `subtype_markers` = specific at `harmonized_annotation` at strictest threshold
   - `per_category_markers` = dict keyed by `category` column from curated CSV (one gene set per category)
4. **Avoid layer mutation**: `adata_ds` may be a view; writing `.layers[...]` forces materialisation. Refactor `compute_cluster_averages` call via a thin matrix helper `_cluster_averages_from_matrix(X_norm, labels, var_names, gene_names)` that takes the normalised matrix directly (no layer roundtrip).
5. **Optional vectorised one-hot path** (performance note, NOT a reimplementation): for many label columns, one-hot encode labels as sparse `I (n_cells, n_clusters)`, compute `(I.T @ X_norm) / I.sum(0)` — single sparse matmul replaces the per-cluster loop. Document as alternative for large label sets.
6. **Extended signature**: add `harmonized_annotation_col: str = "harmonized_annotation"`, `category_col: str = "category"`, and `broad_level_cols: list[str] = ["level_2", "level_3"]` parameters.

## Primary file
`src/regularizedvi/plt/_neighbourhood_correlation.py` (ADD to module)

## Dependencies
- Sub-plan 01 must be complete (imports `normalise_counts`, `compute_cluster_averages` from same module)
- Read-only: `src/regularizedvi/plt/_dotplot.py` lines 69-72 (gene symbol matching pattern)
- Read-only: `docs/notebooks/known_marker_genes.csv`

## Tasks

### 1. Implement `select_marker_genes()` function

```python
def select_marker_genes(
    adata,
    label_columns: list[str],
    dataset_col: str = "dataset",
    layer: str | None = None,
    mean_threshold: float = 1.0,
    specificity_threshold: float = 0.1,
    curated_marker_csv: str | "Path" | None = None,
    symbol_col: str = "SYMBOL",
    harmonized_annotation_col: str = "harmonized_annotation",
    category_col: str = "category",
    broad_level_cols: list[str] | None = None,  # default: ["level_2", "level_3"]
    subtype_specificity_threshold: float = 0.3,
    per_dataset: bool = True,
    return_per_level: bool = True,
) -> dict[str, pd.Index]:
    """Select marker genes by data-driven specificity, per dataset.

    Applied per dataset separately (datasets may disagree on labelling).
    Uses ``compute_cluster_averages`` on normalised data for per-label means,
    then filters by absolute mean AND specificity thresholds.

    See plan.md section "Gene Selection" for full spec.
    """
```

### 2. Implementation steps (in function)

1. **Gene symbol matching**: if `curated_marker_csv` provided, load and map symbols to `adata.var_names` via `symbol_col`. Reuse `_dotplot.py:69-72` pattern. Store as `curated_genes: pd.Index`.

2. **Helper: `_cluster_averages_from_matrix(X_norm, labels, var_names)`** — thin wrapper that computes per-cluster means WITHOUT mutating adata.layers (avoids view materialisation). Takes normalised matrix directly:
   ```python
   def _cluster_averages_from_matrix(X_norm, labels, var_names):
       # X_norm: (n_cells, n_genes) sparse or dense; labels: (n_cells,) array
       cats = pd.Categorical(labels)
       result = {}
       for cat in cats.categories:
           mask = (labels == cat)
           if mask.sum() == 0: continue
           # Mean along axis=0 over rows in this cluster
           sub = X_norm[mask]
           result[cat] = np.asarray(sub.mean(axis=0)).flatten()
       return pd.DataFrame(result, index=var_names)  # (n_genes, n_clusters)
   ```
   Alternatively: use sparse one-hot + matmul for vectorised version (see correction #5).

3. **Per-dataset loop** (if `per_dataset=True`):
   - Subset `adata_ds = adata[adata.obs[dataset_col] == ds]`
   - Normalise: `X_norm = normalise_counts(adata_ds.X, n_vars=adata.n_vars)` — stays sparse; NO layer mutation
   - For each `label_col` in `label_columns`:
     - Skip if `<=1` unique non-null value in this dataset
     - `averages_col = _cluster_averages_from_matrix(X_norm, adata_ds.obs[label_col].to_numpy(), adata.var_names)`
     - Prefix column names: `averages_col.columns = [f"{label_col}:{c}" for c in averages_col.columns]`
   - Concatenate all `averages_col` DataFrames column-wise → `label_averages` (n_genes × total_n_labels)
   - Specificity: `specificity = (label_averages.T / label_averages.sum(axis=1)).T`, fill NaN with 0 (per-gene specificity — each gene's row sums to 1)
   - Filter: `(label_averages.max(axis=1) > mean_threshold) & (specificity.max(axis=1) > specificity_threshold)`
   - Collect selected gene indices for this dataset

4. **Per-level gene sets** (if `return_per_level=True`):
   - Run the specificity filter separately per label column (not concatenated)
   - Map to named gene groups per main-plan table:
     - `broad_lineage_markers` = union of genes selected at any of `broad_level_cols` (default `["level_2", "level_3"]`)
     - `cell_type_markers` = genes selected at `harmonized_annotation_col` (NOT level_1)
     - `subtype_markers` = genes selected at `harmonized_annotation_col` with stricter specificity_threshold (e.g., 0.3)
     - `per_category_markers` = dict keyed by `category_col` values from curated CSV, genes from curated CSV subset per category

5. **Union across datasets**: `data_driven_union = reduce(lambda a, b: a.union(b), per_dataset_genes.values())`. **INCLUDE CURATED**: `union = data_driven_union.union(curated_genes)`.

6. **Overlap with curated**: intersection sizes for summary.

7. **Summary DataFrame**:
   ```
   columns: dataset, label_column, n_genes_passing_mean, n_genes_passing_specificity,
            n_selected, n_overlap_with_curated
   ```

8. **Return dict**:
   ```python
   {
       "union": pd.Index,                       # all_markers = curated ∪ data_driven
       "curated": pd.Index,                     # curated marker genes found in adata
       "data_driven": pd.Index,                 # data-driven only
       "broad_lineage_markers": pd.Index,       # specific at level_2/level_3
       "cell_type_markers": pd.Index,           # specific at harmonized_annotation
       "subtype_markers": pd.Index,             # harmonized_annotation, strict threshold
       "per_category_markers": dict[str, pd.Index],  # from curated CSV category column
       "per_level": dict[str, pd.Index],        # gene set per label column
       "per_dataset": dict[str, pd.Index],      # gene set per dataset
       "summary": pd.DataFrame,
   }
   ```

### 3. Print summary table
Console output: per-dataset gene counts and overlap statistics.

## Test cases
- Single dataset, single label column: returns expected gene count
- Multiple label columns: union is larger than any single column
- Per-dataset vs pooled: per_dataset union >= pooled result
- Curated overlap: known marker genes present in `union` when curated CSV provided
- Threshold sensitivity: lower thresholds → more genes selected

## Verification
- Run on immune integration adata with 5 label columns, 7 datasets
- Report gene counts per dataset and union
- Compare overlap with `known_marker_genes.csv`
- Import check: `from regularizedvi.plt._neighbourhood_correlation import select_marker_genes`
