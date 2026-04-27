"""Cell-level neighbourhood marker gene correlation metrics.

Per-cell marker gene expression correlation with KNN neighbours, stratified by
library / dataset / technical covariate relationships. Distinguishes positive
vs negative integration failure modes. Label-free analysis.

See ``.claude/plans/neighbourhood_correlation_plan.md`` for design rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    from anndata import AnnData

__all__ = [
    "normalise_counts",
    "compute_cluster_averages",
    "select_marker_genes",
    "validate_covariate_hierarchy",
    "construct_neighbour_masks",
    "list_active_masks",
    "compute_marker_correlation",
    "compute_neighbourhood_diagnostics",
    "compute_random_knn_baseline",
    "compute_analytical_isolation_baseline",
    "classify_failure_modes",
    "summarize_failure_modes",
    "compute_distribution_overlap",
    "compute_isolation_norm",
    "summarise_marker_correlation",
    "compute_composite_score",
    "stratified_summary",
    "assemble_cross_model_metrics",
    "compute_best_achievable",
    "compute_integration_failure_rate",
    "compute_model_pair_overlaps",
    "compute_contingency_per_cell",
    "compute_cross_technical_correlation",
    "flag_consensus_isolated",
    "classify_cell_quality",
    "plot_marker_correlation_umap",
    "plot_metric_hist2d",
    "plot_failure_mode_scatter",
    "plot_distribution_overlap",
    "plot_per_library_distributions",
    "plot_isolation_bars",
    "plot_leaf_distribution",
]


def normalise_counts(X, n_vars: int | None = None, total_counts=None):
    """Normalise counts so per-gene average equals 1: count * (n_vars / total_count).

    Parameters
    ----------
    X
        Cells x genes matrix (sparse or dense). May be a marker-gene subset.
    n_vars
        Number of genes used in the scale factor. Defaults to ``X.shape[1]``.
        Pass the **full** gene count when ``X`` is a marker subset so the
        scale reflects whole-cell total counts.
    total_counts
        Optional per-cell total counts (length ``X.shape[0]``). If given,
        skips recomputing from ``X`` — used to pre-compute totals on the
        full gene matrix before subsetting to markers.
    """
    if n_vars is None:
        n_vars = X.shape[1]
    if total_counts is None:
        total_counts = np.asarray(X.sum(axis=1)).flatten()
    else:
        total_counts = np.asarray(total_counts).flatten()
        if total_counts.shape[0] != X.shape[0]:
            raise ValueError(f"total_counts length {total_counts.shape[0]} != X.shape[0] {X.shape[0]}")
    scale = np.zeros(len(total_counts), dtype=np.float32)
    mask = total_counts > 0
    scale[mask] = n_vars / total_counts[mask]
    if sp.issparse(X):
        return X.multiply(scale[:, None]).tocsr()
    return X * scale[:, None]


# Copied from cell2location.cluster_averages.cluster_averages
def compute_cluster_averages(adata, labels, use_raw=True, layer=None):
    """Per-cluster mean expression (genes x clusters)."""
    if layer is not None:
        x = adata.layers[layer]
        var_names = adata.var_names
    else:
        if not use_raw:
            x = adata.X
            var_names = adata.var_names
        else:
            if not adata.raw:
                raise ValueError("AnnData object has no raw data, change `use_raw=True, layer=None` or fix your object")
            x = adata.raw.X
            var_names = adata.raw.var_names

    if sum(adata.obs.columns == labels) != 1:
        raise ValueError("`labels` is absent in adata_ref.obs or not unique")

    all_clusters = np.unique(adata.obs[labels])
    averages_mat = np.zeros((1, x.shape[1]))

    for c in all_clusters:
        sparse_subset = csr_matrix(x[np.isin(adata.obs[labels], c), :])
        aver = sparse_subset.mean(0)
        averages_mat = np.concatenate((averages_mat, aver))
    averages_mat = averages_mat[1:, :].T
    averages_df = pd.DataFrame(data=averages_mat, index=var_names, columns=all_clusters)

    return averages_df


def _cluster_averages_from_matrix(
    X_norm,
    labels: np.ndarray,
    var_names: pd.Index,
) -> pd.DataFrame:
    """Per-cluster mean expression via sparse one-hot matmul, safe on AnnData views."""
    cats = pd.Categorical(labels)
    categories = cats.categories
    codes = cats.codes

    valid_mask = codes >= 0
    n_cells = X_norm.shape[0]
    n_clusters = len(categories)

    row_idx = np.arange(n_cells)[valid_mask]
    col_idx = codes[valid_mask]
    data = np.ones(len(row_idx), dtype=np.float32)
    indicator = csr_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_clusters))

    counts = np.asarray(indicator.sum(axis=0)).flatten()

    if sp.issparse(X_norm):
        sums = np.asarray((indicator.T @ X_norm).todense())
    else:
        sums = np.asarray(indicator.T @ X_norm)

    safe_counts = counts.copy()
    safe_counts[safe_counts == 0] = 1
    averages = (sums / safe_counts[:, None]).T

    keep = counts > 0
    return pd.DataFrame(
        data=averages[:, keep],
        index=var_names,
        columns=categories[keep],
    )


def select_marker_genes(
    adata: AnnData,
    label_columns: list[str],
    dataset_col: str = "dataset",
    layer: str | None = None,
    mean_threshold: float = 1.0,
    specificity_threshold: float = 0.1,
    curated_marker_csv: str | Path | None = None,
    symbol_col: str = "SYMBOL",
    harmonized_annotation_col: str = "harmonized_annotation",
    category_col: str = "category",
    broad_level_cols: list[str] | None = None,
    subtype_specificity_threshold: float = 0.3,
    top_n_per_label: int | None = 500,
    per_dataset: bool = True,
    return_per_level: bool = True,
) -> dict[str, pd.Index | pd.DataFrame | dict]:
    """Select marker genes by data-driven specificity per dataset, union with curated CSV.

    For each column of the specificity table (one per label value in each
    label column, per dataset), keep the top ``top_n_per_label`` genes by
    specificity — restricted to genes that already pass ``mean_threshold``
    and ``specificity_threshold``. Set ``top_n_per_label=None`` (or 0) to
    disable. Union across columns and datasets.
    """
    if broad_level_cols is None:
        broad_level_cols = ["level_2", "level_3"]

    curated_genes = pd.Index([])
    per_category: dict[str, pd.Index] = {}

    if curated_marker_csv is not None:
        marker_df = pd.read_csv(curated_marker_csv)

        if symbol_col in adata.var.columns:
            symbol_to_var = dict(zip(adata.var[symbol_col], adata.var_names, strict=False))
        else:
            symbol_to_var = {v: v for v in adata.var_names}

        mapped = [symbol_to_var[g] for g in marker_df["gene"].unique() if g in symbol_to_var]
        curated_genes = pd.Index(mapped).unique()

        if category_col in marker_df.columns:
            for cat_val, grp in marker_df.groupby(category_col):
                cat_mapped = [symbol_to_var[g] for g in grp["gene"].unique() if g in symbol_to_var]
                if cat_mapped:
                    per_category[cat_val] = pd.Index(cat_mapped).unique()

    per_dataset_genes: dict[str, pd.Index] = {}
    per_level_genes: dict[str, pd.Index] = {}
    summary_rows: list[dict] = []

    if per_dataset and dataset_col in adata.obs.columns:
        datasets = adata.obs[dataset_col].unique()
    else:
        datasets = ["__all__"]

    for ds in datasets:
        if ds == "__all__":
            adata_ds = adata
        else:
            adata_ds = adata[adata.obs[dataset_col] == ds]

        if adata_ds.n_obs == 0:
            continue

        if layer is not None:
            X_raw = adata_ds.layers[layer]
        else:
            X_raw = adata_ds.X

        X_norm = normalise_counts(X_raw, n_vars=adata.n_vars)

        all_averages_list: list[pd.DataFrame] = []
        ds_selected_genes = pd.Index([])

        for label_col in label_columns:
            if label_col not in adata_ds.obs.columns:
                continue

            labels_arr = adata_ds.obs[label_col].to_numpy()
            n_unique = pd.Series(labels_arr).dropna().nunique()
            if n_unique <= 1:
                continue

            averages_col = _cluster_averages_from_matrix(X_norm, labels_arr, adata.var_names)

            averages_col.columns = [f"{label_col}:{c}" for c in averages_col.columns]
            all_averages_list.append(averages_col)

            if return_per_level:
                _run_specificity_filter_per_level(
                    averages_col=averages_col,
                    label_col=label_col,
                    ds=ds,
                    mean_threshold=mean_threshold,
                    specificity_threshold=specificity_threshold,
                    subtype_specificity_threshold=subtype_specificity_threshold,
                    top_n_per_label=top_n_per_label,
                    harmonized_annotation_col=harmonized_annotation_col,
                    per_level_genes=per_level_genes,
                    summary_rows=summary_rows,
                    curated_genes=curated_genes,
                )

        if not all_averages_list:
            continue

        label_averages = pd.concat(all_averages_list, axis=1)

        row_sums = label_averages.sum(axis=1)
        safe_row_sums = row_sums.copy()
        safe_row_sums[safe_row_sums == 0] = 1.0
        specificity = label_averages.div(safe_row_sums, axis=0)

        passes_mean = label_averages.max(axis=1) > mean_threshold
        passes_spec = specificity.max(axis=1) > specificity_threshold
        selected_mask = passes_mean & passes_spec
        n_pre_top_n = int(selected_mask.sum())

        if top_n_per_label is not None and top_n_per_label > 0 and selected_mask.any():
            candidates = specificity.loc[selected_mask]
            top_mask = pd.Series(False, index=specificity.index)
            for col in candidates.columns:
                top_idx = candidates[col].nlargest(top_n_per_label).index
                top_mask.loc[top_idx] = True
            selected_mask = selected_mask & top_mask

        ds_selected_genes = adata.var_names[selected_mask]
        per_dataset_genes[ds] = ds_selected_genes

        summary_rows.append(
            {
                "dataset": ds,
                "label_column": "<all_concatenated>",
                "n_genes_passing_mean": int(passes_mean.sum()),
                "n_genes_passing_specificity": int(passes_spec.sum()),
                "n_selected": n_pre_top_n,
                "top_n_per_label": top_n_per_label,
                "n_selected_after_top_n": int(selected_mask.sum()),
                "n_overlap_with_curated": int(ds_selected_genes.intersection(curated_genes).size),
            }
        )

    if per_dataset_genes:
        from functools import reduce

        data_driven_union = reduce(lambda a, b: a.union(b), per_dataset_genes.values())
    else:
        data_driven_union = pd.Index([])

    union = data_driven_union.union(curated_genes)

    broad_lineage_markers = pd.Index([])
    cell_type_markers = pd.Index([])
    subtype_markers = pd.Index([])

    if return_per_level:
        broad_parts = [per_level_genes[col] for col in broad_level_cols if col in per_level_genes]
        if broad_parts:
            from functools import reduce

            broad_lineage_markers = reduce(lambda a, b: a.union(b), broad_parts)

        if harmonized_annotation_col in per_level_genes:
            cell_type_markers = per_level_genes[harmonized_annotation_col]

        subtype_key = f"{harmonized_annotation_col}__subtype"
        if subtype_key in per_level_genes:
            subtype_markers = per_level_genes[subtype_key]

    summary_df = pd.DataFrame(summary_rows)

    n_curated = len(curated_genes)
    n_dd = len(data_driven_union)
    n_union = len(union)
    print("Gene selection summary:")
    print(f"  Curated markers (in adata): {n_curated}")
    print(f"  Data-driven markers:        {n_dd}")
    print(f"  Union (curated + DD):       {n_union}")
    if n_curated > 0:
        overlap = data_driven_union.intersection(curated_genes)
        print(f"  Overlap (DD ∩ curated):     {len(overlap)}")
    for ds, genes in per_dataset_genes.items():
        print(f"  Dataset '{ds}': {len(genes)} genes")
    if return_per_level:
        print(f"  Broad lineage markers:      {len(broad_lineage_markers)}")
        print(f"  Cell type markers:          {len(cell_type_markers)}")
        print(f"  Subtype markers:            {len(subtype_markers)}")
        print(f"  Per-category groups:        {len(per_category)}")

    return {
        "union": union,
        "curated": curated_genes,
        "data_driven": data_driven_union,
        "broad_lineage_markers": broad_lineage_markers,
        "cell_type_markers": cell_type_markers,
        "subtype_markers": subtype_markers,
        "per_category_markers": per_category,
        "per_level": per_level_genes,
        "per_dataset": per_dataset_genes,
        "summary": summary_df,
    }


def _run_specificity_filter_per_level(
    *,
    averages_col: pd.DataFrame,
    label_col: str,
    ds: str,
    mean_threshold: float,
    specificity_threshold: float,
    subtype_specificity_threshold: float,
    top_n_per_label: int | None,
    harmonized_annotation_col: str,
    per_level_genes: dict[str, pd.Index],
    summary_rows: list[dict],
    curated_genes: pd.Index,
) -> None:
    """Run specificity filter for one label column, mutating per_level_genes and summary_rows."""
    row_sums = averages_col.sum(axis=1)
    safe_row_sums = row_sums.copy()
    safe_row_sums[safe_row_sums == 0] = 1.0
    spec = averages_col.div(safe_row_sums, axis=0)

    passes_mean = averages_col.max(axis=1) > mean_threshold
    passes_spec = spec.max(axis=1) > specificity_threshold
    selected = passes_mean & passes_spec
    n_pre_top_n = int(selected.sum())

    if top_n_per_label is not None and top_n_per_label > 0 and selected.any():
        candidates = spec.loc[selected]
        top_mask = pd.Series(False, index=spec.index)
        for col in candidates.columns:
            top_idx = candidates[col].nlargest(top_n_per_label).index
            top_mask.loc[top_idx] = True
        selected = selected & top_mask

    selected_genes = averages_col.index[selected]

    if label_col in per_level_genes:
        per_level_genes[label_col] = per_level_genes[label_col].union(selected_genes)
    else:
        per_level_genes[label_col] = pd.Index(selected_genes)

    if label_col == harmonized_annotation_col:
        passes_spec_strict = spec.max(axis=1) > subtype_specificity_threshold
        subtype_selected = passes_mean & passes_spec_strict
        subtype_genes = averages_col.index[subtype_selected]
        subtype_key = f"{harmonized_annotation_col}__subtype"
        if subtype_key in per_level_genes:
            per_level_genes[subtype_key] = per_level_genes[subtype_key].union(subtype_genes)
        else:
            per_level_genes[subtype_key] = pd.Index(subtype_genes)

    summary_rows.append(
        {
            "dataset": ds,
            "label_column": label_col,
            "n_genes_passing_mean": int(passes_mean.sum()),
            "n_genes_passing_specificity": int(passes_spec.sum()),
            "n_selected": n_pre_top_n,
            "top_n_per_label": top_n_per_label,
            "n_selected_after_top_n": int(selected.sum()),
            "n_overlap_with_curated": int(selected_genes.intersection(curated_genes).size),
        }
    )


def validate_covariate_hierarchy(
    adata,
    library_key: str,
    dataset_key: str | None = None,
) -> None:
    """Verify each library maps to exactly one dataset, raise ValueError otherwise.

    Also emits a warning (does not raise) if any cells have NaN in
    ``library_key`` or ``dataset_key`` — downstream isolation/correlation
    metrics will return NaN for those cells.
    """
    for key in (library_key, dataset_key):
        if key is None:
            continue
        n_nan = int(adata.obs[key].isna().sum())
        if n_nan > 0:
            _logger.warning(
                "validate_covariate_hierarchy: %d/%d cells have NaN '%s'; "
                "downstream per-cell metrics for those cells will be NaN.",
                n_nan,
                adata.n_obs,
                key,
            )

    if dataset_key is None:
        return

    pairs = adata.obs[[library_key, dataset_key]].dropna().drop_duplicates()
    datasets_per_library = pairs.groupby(library_key)[dataset_key].nunique()
    bad = datasets_per_library[datasets_per_library > 1]

    if len(bad) > 0:
        details = []
        for lib in bad.index:
            ds_values = pairs.loc[pairs[library_key] == lib, dataset_key].unique().tolist()
            details.append(f"  library '{lib}' -> datasets {ds_values}")
        detail_str = "\n".join(details)
        raise ValueError(
            f"Covariate hierarchy violation: {len(bad)} library value(s) span "
            f"multiple datasets. Each library must map to exactly one dataset.\n"
            f"{detail_str}"
        )

    libraries_per_dataset = pairs.groupby(dataset_key)[library_key].nunique()
    _logger.info(
        "Covariate hierarchy validated: %d libraries across %d datasets. Libraries per dataset: %s",
        len(pairs),
        libraries_per_dataset.shape[0],
        libraries_per_dataset.to_dict(),
    )


def construct_neighbour_masks(
    adata,
    connectivities: sp.csr_matrix,
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
) -> dict[str, sp.csr_matrix]:
    """Build masked connectivity matrices for each covariate relationship (same/cross library/dataset)."""
    validate_covariate_hierarchy(adata, library_key, dataset_key)

    conn = connectivities.copy().tocsr()
    conn.setdiag(0)
    conn.eliminate_zeros()

    i_idx, j_idx = conn.nonzero()
    n = conn.shape[0]

    masks: dict[str, sp.csr_matrix] = {}

    lib_codes = adata.obs[library_key].astype("category").cat.codes.to_numpy()
    same_lib_bool = lib_codes[i_idx] == lib_codes[j_idx]

    same_lib_mask = sp.csr_matrix((same_lib_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
    masks["same_library"] = conn.multiply(same_lib_mask)

    between_lib_bool = ~same_lib_bool
    between_lib_mask = sp.csr_matrix((between_lib_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
    masks["between_libraries"] = conn.multiply(between_lib_mask)

    if dataset_key is not None:
        ds_codes = adata.obs[dataset_key].astype("category").cat.codes.to_numpy()
        same_ds_bool = ds_codes[i_idx] == ds_codes[j_idx]

        cross_lib_bool = between_lib_bool & same_ds_bool
        cross_lib_mask = sp.csr_matrix((cross_lib_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
        masks["cross_library"] = conn.multiply(cross_lib_mask)

        cross_ds_bool = ~same_ds_bool
        cross_ds_mask = sp.csr_matrix((cross_ds_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
        masks["cross_dataset"] = conn.multiply(cross_ds_mask)

    any_between_tech_bool: np.ndarray | None = None
    for tech_key in technical_covariate_keys or []:
        t_codes = adata.obs[tech_key].astype("category").cat.codes.to_numpy()
        same_t_bool = t_codes[i_idx] == t_codes[j_idx]

        within_mask = sp.csr_matrix((same_t_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
        masks[f"within_{tech_key}"] = conn.multiply(within_mask)

        between_bool = ~same_t_bool
        between_mask = sp.csr_matrix((between_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
        masks[f"between_{tech_key}"] = conn.multiply(between_mask)

        if any_between_tech_bool is None:
            any_between_tech_bool = between_bool.copy()
        else:
            any_between_tech_bool = any_between_tech_bool | between_bool

    # H14: "cross_technical" = neighbours that differ in ANY technical covariate value.
    # This is the union over all technical keys of the per-key between_{tech} masks.
    if any_between_tech_bool is not None:
        cross_tech_mask = sp.csr_matrix((any_between_tech_bool, (i_idx, j_idx)), shape=(n, n), dtype=bool)
        masks["cross_technical"] = conn.multiply(cross_tech_mask)

    return masks


def list_active_masks(
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
) -> list[str]:
    """Return ordered list of mask names based on which covariate keys are provided."""
    names: list[str] = ["same_library", "between_libraries"]

    if dataset_key is not None:
        names.extend(["cross_library", "cross_dataset"])

    for tech_key in technical_covariate_keys or []:
        names.extend([f"within_{tech_key}", f"between_{tech_key}"])

    if technical_covariate_keys:
        names.append("cross_technical")

    return names


def _sparse_pearson_row_stats(X):
    """Per-row mean and std from a sparse matrix, O(nnz) with sparsity preserved.

    Uses the identity ``var = E[X^2] - E[X]^2`` so the centring step
    (which would destroy sparsity on highly-sparse scRNA-seq data) is
    avoided. No existing package provides sparse Pearson correlation:
    ``sklearn.metrics.pairwise.cosine_similarity`` is cosine, not Pearson;
    ``scipy.stats.pearsonr`` and ``sklearn.metrics.pairwise_distances
    (metric='correlation')`` require dense arrays.

    Zero-variance rows are floored at ``1e-12`` in the variance before
    sqrt to prevent divide-by-zero downstream (callers treat cells with
    no marker expression as NaN-correlation — see
    :func:`_approach_B_per_mask`).

    Matches :func:`numpy.corrcoef` (within float32 tolerance) for all
    non-zero-variance rows — validated in
    ``TestApproachA.test_matches_np_corrcoef`` and
    ``TestSparsePearsonHighSparsity``.
    """
    n = X.shape[1]
    sum_x = np.asarray(X.sum(axis=1)).flatten()
    sum_x2 = np.asarray(X.power(2).sum(axis=1)).flatten()
    mean_x = sum_x / n
    var_x = sum_x2 / n - mean_x**2
    std_x = np.sqrt(np.clip(var_x, 1e-12, None))
    return mean_x, std_x


def _approach_B_per_mask(X, mask_csr, mean_x, std_x):
    """Pearson(cell, weighted-average-of-neighbours) per cell, fully sparse."""
    n = X.shape[1]

    weighted_sum = mask_csr @ X
    row_sums = np.asarray(mask_csr.sum(axis=1)).flatten()

    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    avg_profiles = weighted_sum.multiply(1.0 / safe_sums[:, None])

    cross = np.asarray(X.multiply(avg_profiles).sum(axis=1)).flatten()
    sum_a = np.asarray(avg_profiles.sum(axis=1)).flatten()
    sum_a2 = np.asarray(avg_profiles.power(2).sum(axis=1)).flatten()
    mean_a = sum_a / n
    std_a = np.sqrt(np.clip(sum_a2 / n - mean_a**2, 1e-12, None))

    corr_avg = (cross / n - mean_x * mean_a) / (std_x * std_a)
    corr_avg[row_sums == 0] = np.nan

    zero_var_mask = (std_x < 1e-6) | (std_a < 1e-6)
    corr_avg[zero_var_mask] = np.nan

    return corr_avg


def weighted_median(values, weights):
    """Value at which cumulative weight reaches 50%."""
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    half = 0.5 * cw[-1]
    idx = np.searchsorted(cw, half)
    return float(v[min(idx, len(v) - 1)])


def _approach_A_per_mask(X, mask_csr, mean_x, std_x, batch_size=2000):
    """Pairwise Pearson(cell, neighbour) aggregated per cell: mean/median/weighted/std/cv."""
    n_cells = X.shape[0]
    n = X.shape[1]

    result_mean = np.full(n_cells, np.nan, dtype=np.float64)
    result_median = np.full(n_cells, np.nan, dtype=np.float64)
    result_wmean = np.full(n_cells, np.nan, dtype=np.float64)
    result_wmedian = np.full(n_cells, np.nan, dtype=np.float64)
    result_std = np.full(n_cells, np.nan, dtype=np.float64)
    result_cv = np.full(n_cells, np.nan, dtype=np.float64)

    for b0 in range(0, n_cells, batch_size):
        b1 = min(b0 + batch_size, n_cells)
        block_csr = mask_csr[b0:b1]
        block_coo = block_csr.tocoo()
        if block_coo.nnz == 0:
            continue

        rows_local = block_coo.row
        rows_abs = rows_local + b0
        cols = block_coo.col
        weights_arr = block_coo.data.astype(np.float64)

        X_i = X[rows_abs]
        X_j = X[cols]

        cross = np.asarray(X_i.multiply(X_j).sum(axis=1)).flatten()

        # Individual std check: constant rows yield NaN even when neighbour has variance
        std_i = std_x[rows_abs]
        std_j = std_x[cols]
        valid_pair = (std_i > 1e-6) & (std_j > 1e-6)
        denom = std_i * std_j
        r_flat = np.where(
            valid_pair,
            (cross / n - mean_x[rows_abs] * mean_x[cols]) / denom,
            np.nan,
        )

        unique_rows_local, row_starts = np.unique(rows_local, return_index=True)
        row_ends = np.append(row_starts[1:], len(rows_local))

        for _k, (rl, start, end) in enumerate(zip(unique_rows_local, row_starts, row_ends, strict=False)):
            r_vals = r_flat[start:end]
            w_vals = weights_arr[start:end]
            cell_idx = rl + b0

            valid = ~np.isnan(r_vals)
            if not np.any(valid):
                continue

            r_valid = r_vals[valid]
            w_valid = w_vals[valid]

            result_mean[cell_idx] = np.mean(r_valid)
            result_median[cell_idx] = np.median(r_valid)

            w_sum = w_valid.sum()
            if w_sum > 0:
                result_wmean[cell_idx] = np.average(r_valid, weights=w_valid)
                result_wmedian[cell_idx] = weighted_median(r_valid, w_valid)
            else:
                result_wmean[cell_idx] = np.nan
                result_wmedian[cell_idx] = np.nan

            if len(r_valid) > 1:
                result_std[cell_idx] = np.std(r_valid, ddof=0)
                if abs(result_mean[cell_idx]) > 1e-12:
                    result_cv[cell_idx] = result_std[cell_idx] / abs(result_mean[cell_idx])
                else:
                    result_cv[cell_idx] = np.nan
            else:
                result_std[cell_idx] = 0.0
                result_cv[cell_idx] = np.nan

    return {
        "mean": result_mean,
        "median": result_median,
        "weighted_mean": result_wmean,
        "weighted_median": result_wmedian,
        "std": result_std,
        "cv": result_cv,
    }


def compute_marker_correlation(
    adata,
    connectivities,
    marker_genes,
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
    layer: str | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Per-cell marker gene correlation with KNN neighbours, stratified by covariate masks."""
    marker_idx = adata.var_names.isin(marker_genes)
    if marker_idx.sum() == 0:
        raise ValueError("No marker genes found in adata.var_names")

    # Compute total counts on the FULL gene matrix BEFORE subsetting to markers,
    # so the normalisation scale reflects whole-cell sequencing depth.
    X_full = adata.layers[layer] if layer is not None else adata.X
    total_counts_full = np.asarray(X_full.sum(axis=1)).flatten()

    if layer is not None:
        X = adata[:, marker_idx].layers[layer]
    else:
        X = adata[:, marker_idx].X

    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    X = normalise_counts(X, n_vars=adata.n_vars, total_counts=total_counts_full)
    X = X.astype(np.float32)

    mean_x, std_x = _sparse_pearson_row_stats(X)
    marker_gene_total_expression = np.asarray(X.sum(axis=1)).flatten()

    n_markers = X.shape[1]

    if batch_size is None:
        batch_size = max(500, min(10000, 100_000_000 // (n_markers * 4)))

    masks = construct_neighbour_masks(
        adata,
        connectivities,
        library_key=library_key,
        dataset_key=dataset_key,
        technical_covariate_keys=technical_covariate_keys,
    )

    conn_all = connectivities.copy().tocsr()
    conn_all.setdiag(0)
    conn_all.eliminate_zeros()

    result = {}
    n_cells = X.shape[0]

    n_neighbours_total = np.asarray(conn_all.getnnz(axis=1)).flatten()
    result["n_neighbours_total"] = n_neighbours_total
    result["marker_gene_total_expression"] = marker_gene_total_expression

    _logger.info("Computing all-neighbours correlations...")
    corr_avg_all = _approach_B_per_mask(X, conn_all, mean_x, std_x)
    result["corr_avg_all_neighbours"] = corr_avg_all

    agg_all = _approach_A_per_mask(X, conn_all, mean_x, std_x, batch_size)
    result["corr_weighted_mean_all_neighbours"] = agg_all["weighted_mean"]

    mask_names = list_active_masks(
        library_key=library_key,
        dataset_key=dataset_key,
        technical_covariate_keys=technical_covariate_keys,
    )

    for mask_name in mask_names:
        _logger.info("Computing correlations for mask '%s'...", mask_name)
        mask = masks[mask_name]

        n_neigh = np.asarray(mask.getnnz(axis=1)).flatten()
        result[f"n_neighbours_{mask_name}"] = n_neigh

        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(
                n_neighbours_total > 0,
                n_neigh / n_neighbours_total,
                np.nan,
            )
        result[f"frac_neighbours_{mask_name}"] = frac

        corr_avg = _approach_B_per_mask(X, mask, mean_x, std_x)
        result[f"corr_avg_{mask_name}"] = corr_avg

        agg = _approach_A_per_mask(X, mask, mean_x, std_x, batch_size)
        result[f"corr_mean_{mask_name}"] = agg["mean"]
        result[f"corr_median_{mask_name}"] = agg["median"]
        result[f"corr_weighted_mean_{mask_name}"] = agg["weighted_mean"]
        result[f"corr_weighted_median_{mask_name}"] = agg["weighted_median"]
        result[f"corr_std_{mask_name}"] = agg["std"]
        result[f"corr_cv_{mask_name}"] = agg["cv"]

        result[f"corr_discrepancy_{mask_name}"] = corr_avg - agg["mean"]

    corr_avg_same_library = result.get("corr_avg_same_library")

    for mask_name in mask_names:
        corr_avg = result[f"corr_avg_{mask_name}"]

        if corr_avg_same_library is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                norm_lib = np.where(
                    np.abs(corr_avg_same_library) > 1e-12,
                    corr_avg / corr_avg_same_library,
                    np.nan,
                )
            result[f"corr_norm_by_library_{mask_name}"] = norm_lib
        else:
            result[f"corr_norm_by_library_{mask_name}"] = np.full(n_cells, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            norm_all = np.where(
                np.abs(corr_avg_all) > 1e-12,
                corr_avg / corr_avg_all,
                np.nan,
            )
        result[f"corr_norm_by_all_{mask_name}"] = norm_all

    df = pd.DataFrame(result, index=adata.obs_names)
    return df


def compute_neighbourhood_diagnostics(
    adata,
    connectivities,
    library_key: str,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
    k_reference: int = 50,
    penetration_thresholds: tuple[int, ...] = (10, 25),
    high_degree_multiplier: float = 1.5,
) -> dict[str, pd.DataFrame | pd.Series]:
    """KNN graph diagnostics: degree distribution, high-degree comparison, composition, penetration."""
    conn = connectivities.copy().tocsr()
    conn.setdiag(0)
    conn.eliminate_zeros()

    n_cells = conn.shape[0]
    diagnostics: dict[str, pd.DataFrame | pd.Series] = {}

    degree_arr = np.asarray(conn.getnnz(axis=1)).flatten()
    degree = pd.Series(degree_arr, index=adata.obs_names, name="degree")
    diagnostics["degree"] = degree

    threshold = k_reference * high_degree_multiplier
    is_high_degree = degree_arr > threshold

    numeric_cols = adata.obs.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 0 and is_high_degree.any() and (~is_high_degree).any():
        high_means = adata.obs.loc[is_high_degree, numeric_cols].mean()
        normal_means = adata.obs.loc[~is_high_degree, numeric_cols].mean()
        diff = high_means - normal_means
        high_degree_df = pd.DataFrame(
            {
                "high_degree_mean": high_means,
                "normal_mean": normal_means,
                "diff": diff,
            }
        )
    else:
        high_degree_df = pd.DataFrame(columns=["high_degree_mean", "normal_mean", "diff"])
    diagnostics["high_degree_obs_means"] = high_degree_df

    composition_keys: dict[str, str] = {}
    composition_keys["library"] = library_key
    if dataset_key is not None:
        composition_keys["dataset"] = dataset_key
    for tech_key in technical_covariate_keys or []:
        composition_keys[f"technical_{tech_key}"] = tech_key

    for display_name, obs_col in composition_keys.items():
        cat = adata.obs[obs_col].astype("category")
        codes = cat.cat.codes.to_numpy()
        categories = cat.cat.categories
        n_unique = len(categories)

        one_hot = sp.csr_matrix(
            (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), codes)),
            shape=(n_cells, n_unique),
        )
        composition = conn @ one_hot
        composition_dense = np.asarray(composition.todense())
        totals = composition_dense.sum(axis=1, keepdims=True)
        totals = np.clip(totals, 1e-12, None)
        composition_dense = composition_dense / totals

        diagnostics[f"composition_{display_name}"] = pd.DataFrame(
            composition_dense,
            index=adata.obs_names,
            columns=categories,
        )

    masks = construct_neighbour_masks(
        adata,
        connectivities,
        library_key=library_key,
        dataset_key=dataset_key,
        technical_covariate_keys=technical_covariate_keys,
    )

    penetration_masks: dict[str, str] = {}
    if "cross_library" in masks:
        penetration_masks["cross_library"] = library_key
    if "cross_dataset" in masks:
        penetration_masks["cross_dataset"] = dataset_key
    penetration_masks["between_libraries"] = library_key
    for tech_key in technical_covariate_keys or []:
        mask_name = f"between_{tech_key}"
        if mask_name in masks:
            penetration_masks[mask_name] = tech_key

    for mask_name, stratify_key in penetration_masks.items():
        mask_csr = masks[mask_name]
        n_cross = np.asarray(mask_csr.getnnz(axis=1)).flatten()

        pen_data = {}
        for thr in penetration_thresholds:
            has_enough = n_cross >= thr
            pen_series = pd.Series(has_enough, index=adata.obs_names).groupby(adata.obs[stratify_key]).mean()
            pen_data[f"threshold_{thr}"] = pen_series

        diagnostics[f"penetration_{mask_name}"] = pd.DataFrame(pen_data)

    return diagnostics


def compute_random_knn_baseline(
    adata,
    X_normalised_markers,
    degree_per_cell,
    library_key: str,
    dataset_key: str | None = None,
    n_random_graphs: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Expected correlation under random KNN (degree-matched, uniform random neighbours)."""
    rng = np.random.default_rng(random_state)
    n_cells = adata.n_obs
    degree_per_cell = np.asarray(degree_per_cell, dtype=np.int64)

    if not sp.issparse(X_normalised_markers):
        X = sp.csr_matrix(X_normalised_markers.astype(np.float32))
    else:
        X = X_normalised_markers.tocsr().astype(np.float32)

    mean_x, std_x = _sparse_pearson_row_stats(X)

    mask_names = list_active_masks(
        library_key=library_key,
        dataset_key=dataset_key,
    )

    accumulators: dict[str, np.ndarray] = {}
    accumulators["corr_avg_random_all_neighbours"] = np.zeros(n_cells, dtype=np.float64)
    for mask_name in mask_names:
        accumulators[f"corr_avg_random_{mask_name}"] = np.zeros(n_cells, dtype=np.float64)

    max_k = int(degree_per_cell.max())
    if max_k == 0:
        result = {k: np.full(n_cells, np.nan) for k in accumulators}
        return pd.DataFrame(result, index=adata.obs_names)

    for _iter in range(n_random_graphs):
        samples = rng.integers(0, n_cells, size=(n_cells, max_k))

        self_mask = samples == np.arange(n_cells)[:, None]
        while self_mask.any():
            samples[self_mask] = rng.integers(0, n_cells, size=int(self_mask.sum()))
            self_mask = samples == np.arange(n_cells)[:, None]

        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(degree_per_cell)
        total_nnz = int(indptr[-1])

        indices = np.empty(total_nnz, dtype=np.int64)
        for i in range(n_cells):
            start = indptr[i]
            end = indptr[i + 1]
            indices[start:end] = samples[i, : degree_per_cell[i]]

        data = np.ones(total_nnz, dtype=np.float32)
        random_conn = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

        corr_all = _approach_B_per_mask(X, random_conn, mean_x, std_x)
        corr_all = np.nan_to_num(corr_all, nan=0.0)
        accumulators["corr_avg_random_all_neighbours"] += corr_all

        random_masks = construct_neighbour_masks(
            adata,
            random_conn,
            library_key=library_key,
            dataset_key=dataset_key,
        )

        for mask_name in mask_names:
            if mask_name in random_masks:
                corr_mask = _approach_B_per_mask(X, random_masks[mask_name], mean_x, std_x)
                corr_mask = np.nan_to_num(corr_mask, nan=0.0)
                accumulators[f"corr_avg_random_{mask_name}"] += corr_mask

    result = {k: v / n_random_graphs for k, v in accumulators.items()}

    return pd.DataFrame(result, index=adata.obs_names)


def compute_analytical_isolation_baseline(
    adata,
    degree_per_cell,
    covariate_key: str,
) -> pd.Series:
    """Per-cell P(all k random neighbours same covariate group) = ((n_same-1)/(n_total-1))^k.

    Cells whose ``obs[covariate_key]`` is NaN are excluded: a warning is logged
    and the returned probability for those cells is NaN.
    """
    degree_per_cell = np.asarray(degree_per_cell, dtype=np.int64)
    n_total = adata.n_obs
    cov_series = adata.obs[covariate_key]
    nan_mask = cov_series.isna().to_numpy()
    n_nan = int(nan_mask.sum())
    if n_nan > 0:
        _logger.warning(
            "compute_analytical_isolation_baseline: %d/%d cells have NaN '%s'; "
            "returning NaN isolation probability for those cells.",
            n_nan,
            adata.n_obs,
            covariate_key,
        )

    group_counts = cov_series.value_counts(dropna=True)
    n_same = cov_series.map(group_counts).to_numpy(dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        p_same = (n_same - 1) / (n_total - 1)
    p_isolated = p_same**degree_per_cell
    p_isolated = np.where(nan_mask, np.nan, p_isolated)

    return pd.Series(
        p_isolated,
        index=adata.obs_names,
        name=f"isolation_prob_{covariate_key}",
    )


_WL_SEVERITY: dict[str, int] = {
    "WL-unknown": -1,
    "WL-0_orphan": 0,
    "WL-3_noisy": 1,
    "WL-4_false_merge_confident": 2,
    "WL-5_false_merge_partial": 3,
    "WL-2_merged_related": 4,
    "WL-1_ideal": 5,
}

_XL_SEVERITY: dict[str, int] = {
    "XL-unknown": -1,
    "XL-0b_compounded_failure": 0,
    "XL-5_poor_model": 1,
    "XL-0a_under_integration": 2,
    "XL-3_wrong_pairing": 3,
    "XL-4_forced_distinct": 4,
    "XL-2_partial": 5,
    "XL-1_ideal": 6,
}

_XD_SEVERITY: dict[str, int] = {
    "XD-unknown": -1,
    "XD-0d_compounded": 0,
    "XD-6_poor_model": 1,
    "XD-4b_systematic_failure": 2,
    "XD-4a_wrong_pairing": 3,
    "XD-0b_under_integration": 4,
    "XD-5a_forced_merge": 5,
    "XD-5b_semi_random": 6,
    "XD-0c_cascaded": 7,
    "XD-2_spurious": 8,
    "XD-3_partial": 9,
    "XD-1_ideal": 10,
    "XD-0a_dataset_enriched": 11,
    "XD-0_isolated_unknown": 4,
}

_ALL_SEVERITY: dict[str, int] = {}
_ALL_SEVERITY.update(_WL_SEVERITY)
_ALL_SEVERITY.update(_XL_SEVERITY)
_ALL_SEVERITY.update(_XD_SEVERITY)

_NON_FAILURE_LEAVES = frozenset({"WL-1_ideal", "XL-1_ideal", "XD-1_ideal", "XD-0a_dataset_enriched"})


def classify_failure_modes(
    metrics_df: pd.DataFrame,
    random_baseline_df: pd.DataFrame | None = None,
    gene_group_metrics: dict[str, pd.DataFrame] | None = None,
    model_comparison_result: pd.Series | None = None,
    threshold_high: float | None = None,
    std_threshold: float | None = None,
) -> pd.DataFrame:
    """Assign each cell to a decision tree leaf (WL/XL/XD) via vectorised np.select."""
    n = len(metrics_df)

    corr_sl_raw = metrics_df["corr_avg_same_library"].to_numpy(dtype=np.float64)

    if threshold_high is None:
        threshold_high = float(np.nanpercentile(corr_sl_raw, 25))
    if std_threshold is None:
        std_sl_raw = metrics_df["corr_std_same_library"].to_numpy(dtype=np.float64)
        std_threshold = float(np.nanmedian(std_sl_raw))

    th_high = threshold_high
    th_std = std_threshold

    has_sl = metrics_df["n_neighbours_same_library"].to_numpy() > 0
    corr_sl = metrics_df["corr_avg_same_library"].to_numpy(dtype=np.float64)
    std_sl = metrics_df["corr_std_same_library"].to_numpy(dtype=np.float64)

    hi_sl = corr_sl >= th_high
    homog_sl = std_sl <= th_std

    if random_baseline_df is not None and "corr_avg_random_same_library" in random_baseline_df.columns:
        random_corr_sl = (
            random_baseline_df["corr_avg_random_same_library"].reindex(metrics_df.index).to_numpy(dtype=np.float64)
        )
        above_random_sl = corr_sl > random_corr_sl
    else:
        above_random_sl = np.ones(n, dtype=bool)

    wl_conditions = [
        ~has_sl,
        has_sl & hi_sl & homog_sl,
        has_sl & hi_sl & ~homog_sl,
        has_sl & ~hi_sl & ~above_random_sl,
        has_sl & ~hi_sl & above_random_sl & homog_sl,
        has_sl & ~hi_sl & above_random_sl & ~homog_sl,
    ]
    wl_choices = [
        "WL-0_orphan",
        "WL-1_ideal",
        "WL-2_merged_related",
        "WL-3_noisy",
        "WL-4_false_merge_confident",
        "WL-5_false_merge_partial",
    ]
    wl_leaves = np.select(wl_conditions, wl_choices, default="WL-unknown")

    has_xl_cols = "n_neighbours_cross_library" in metrics_df.columns
    if has_xl_cols:
        has_xl = metrics_df["n_neighbours_cross_library"].to_numpy() > 0
        corr_xl = metrics_df["corr_avg_cross_library"].to_numpy(dtype=np.float64)
        std_xl = metrics_df["corr_std_cross_library"].to_numpy(dtype=np.float64)

        hi_xl = corr_xl >= th_high
        homog_xl = std_xl <= th_std

        wl_was_ideal = np.isin(wl_leaves, ["WL-1_ideal", "WL-2_merged_related"])

        xl_conditions = [
            ~has_xl & wl_was_ideal,
            ~has_xl & ~wl_was_ideal,
            has_xl & hi_xl & homog_xl,
            has_xl & hi_xl & ~homog_xl,
            has_xl & ~hi_xl & wl_was_ideal & homog_xl,
            has_xl & ~hi_xl & wl_was_ideal & ~homog_xl,
            has_xl & ~hi_xl & ~wl_was_ideal,
        ]
        xl_choices = [
            "XL-0a_under_integration",
            "XL-0b_compounded_failure",
            "XL-1_ideal",
            "XL-2_partial",
            "XL-3_wrong_pairing",
            "XL-4_forced_distinct",
            "XL-5_poor_model",
        ]
        xl_leaves = np.select(xl_conditions, xl_choices, default="XL-unknown")
    else:
        xl_leaves = None

    has_xd_cols = "n_neighbours_cross_dataset" in metrics_df.columns
    if has_xd_cols:
        has_xd = metrics_df["n_neighbours_cross_dataset"].to_numpy() > 0
        corr_xd = metrics_df["corr_avg_cross_dataset"].to_numpy(dtype=np.float64)
        std_xd = metrics_df["corr_std_cross_dataset"].to_numpy(dtype=np.float64)

        hi_xd = corr_xd >= th_high
        homog_xd = std_xd <= th_std

        wl_was_ideal_xd = np.isin(wl_leaves, ["WL-1_ideal", "WL-2_merged_related"])

        if xl_leaves is not None:
            xl_was_ideal = np.isin(xl_leaves, ["XL-1_ideal", "XL-2_partial"])
            xl_was_under = xl_leaves == "XL-0a_under_integration"
            xl_was_failure = ~np.isin(
                xl_leaves,
                ["XL-0a_under_integration", "XL-1_ideal", "XL-2_partial"],
            )
        else:
            xl_was_ideal = np.ones(n, dtype=bool)
            xl_was_under = np.zeros(n, dtype=bool)
            xl_was_failure = np.zeros(n, dtype=bool)

        if random_baseline_df is not None and "corr_avg_random_cross_dataset" in random_baseline_df.columns:
            random_corr_xd = (
                random_baseline_df["corr_avg_random_cross_dataset"].reindex(metrics_df.index).to_numpy(dtype=np.float64)
            )
            above_random_xd = corr_xd > random_corr_xd
        else:
            above_random_xd = np.ones(n, dtype=bool)

        if gene_group_metrics is not None and "broad_lineage" in gene_group_metrics and "subtype" in gene_group_metrics:
            broad_df = gene_group_metrics["broad_lineage"]
            specific_df = gene_group_metrics["subtype"]
            if "corr_avg_cross_dataset" in broad_df.columns and "corr_avg_cross_dataset" in specific_df.columns:
                broad_high = (
                    broad_df["corr_avg_cross_dataset"].reindex(metrics_df.index).to_numpy(dtype=np.float64) >= th_high
                )
                specific_low = (
                    specific_df["corr_avg_cross_dataset"].reindex(metrics_df.index).to_numpy(dtype=np.float64) < th_high
                )
                gene_group_available = True
            else:
                gene_group_available = False
        else:
            gene_group_available = False

        if not gene_group_available:
            broad_high = np.ones(n, dtype=bool)
            specific_low = np.ones(n, dtype=bool)

        if model_comparison_result is not None:
            mc = model_comparison_result.reindex(metrics_df.index).to_numpy()
            mc_bool = np.asarray(mc, dtype=bool)
            mc_valid = ~pd.isna(model_comparison_result.reindex(metrics_df.index))
            other_connects = mc_bool & mc_valid
            no_model_connects = ~mc_bool & mc_valid
            mc_unknown = ~mc_valid
        else:
            other_connects = np.zeros(n, dtype=bool)
            no_model_connects = np.zeros(n, dtype=bool)
            mc_unknown = np.ones(n, dtype=bool)

        cond_xd_0a = ~has_xd & xl_was_ideal & no_model_connects
        cond_xd_0b = ~has_xd & xl_was_ideal & other_connects
        cond_xd_0_unk = ~has_xd & xl_was_ideal & mc_unknown
        cond_xd_0c = ~has_xd & xl_was_under
        cond_xd_0d = ~has_xd & xl_was_failure

        cond_xd_1 = has_xd & hi_xd & above_random_xd & homog_xd
        cond_xd_3 = has_xd & hi_xd & above_random_xd & ~homog_xd
        cond_xd_2 = has_xd & hi_xd & ~above_random_xd

        cond_xd_4a = has_xd & ~hi_xd & wl_was_ideal_xd & xl_was_ideal & homog_xd
        cond_xd_4b = has_xd & ~hi_xd & wl_was_ideal_xd & ~xl_was_ideal & homog_xd
        cond_xd_5a = has_xd & ~hi_xd & wl_was_ideal_xd & ~homog_xd & broad_high & specific_low
        cond_xd_5b = has_xd & ~hi_xd & wl_was_ideal_xd & ~homog_xd & ~(broad_high & specific_low)

        cond_xd_6 = has_xd & ~hi_xd & ~wl_was_ideal_xd

        xd_conditions = [
            cond_xd_0a,
            cond_xd_0b,
            cond_xd_0_unk,
            cond_xd_0c,
            cond_xd_0d,
            cond_xd_1,
            cond_xd_3,
            cond_xd_2,
            cond_xd_4a,
            cond_xd_4b,
            cond_xd_5a,
            cond_xd_5b,
            cond_xd_6,
        ]
        xd_choices = [
            "XD-0a_dataset_enriched",
            "XD-0b_under_integration",
            "XD-0_isolated_unknown",
            "XD-0c_cascaded",
            "XD-0d_compounded",
            "XD-1_ideal",
            "XD-3_partial",
            "XD-2_spurious",
            "XD-4a_wrong_pairing",
            "XD-4b_systematic_failure",
            "XD-5a_forced_merge",
            "XD-5b_semi_random",
            "XD-6_poor_model",
        ]
        xd_leaves = np.select(xd_conditions, xd_choices, default="XD-unknown")
    else:
        xd_leaves = None

    result = pd.DataFrame(index=metrics_df.index)
    result["leaf_within_library"] = wl_leaves

    if xl_leaves is not None:
        result["leaf_cross_library"] = xl_leaves
    if xd_leaves is not None:
        result["leaf_cross_dataset"] = xd_leaves

    failure_mode = _compute_combined_failure_mode(wl_leaves, xl_leaves, xd_leaves)
    result["failure_mode"] = failure_mode

    return result


def _compute_combined_failure_mode(
    wl_leaves: np.ndarray,
    xl_leaves: np.ndarray | None,
    xd_leaves: np.ndarray | None,
) -> np.ndarray:
    """Most severe (lowest severity number) leaf across WL/XL/XD levels per cell."""
    n = len(wl_leaves)
    result = np.empty(n, dtype=object)

    leaf_arrays = [wl_leaves]
    sev_maps = [_WL_SEVERITY]

    if xl_leaves is not None:
        leaf_arrays.append(xl_leaves)
        sev_maps.append(_XL_SEVERITY)
    if xd_leaves is not None:
        leaf_arrays.append(xd_leaves)
        sev_maps.append(_XD_SEVERITY)

    worst_sev = np.full(n, 999, dtype=np.int32)
    worst_label = np.full(n, "unknown", dtype=object)

    for leaves, sev_map in zip(leaf_arrays, sev_maps, strict=False):
        for label, sev_num in sev_map.items():
            mask = leaves == label
            if not np.any(mask):
                continue
            if label in _NON_FAILURE_LEAVES:
                continue
            update = mask & (sev_num < worst_sev)
            worst_sev[update] = sev_num
            worst_label[update] = "unknown" if label.endswith("-unknown") else label

    # Cells with all-ideal leaves never get their sev updated → worst_sev stays 999.
    # Relabel them as "ideal" rather than the default "unknown".
    any_valid = np.zeros(n, dtype=bool)
    for leaves, sev_map in zip(leaf_arrays, sev_maps, strict=False):
        for label in sev_map:
            if label in _NON_FAILURE_LEAVES:
                any_valid |= leaves == label
    ideal_mask = (worst_sev == 999) & any_valid
    worst_label[ideal_mask] = "ideal"

    result[:] = worst_label
    return result


def summarize_failure_modes(
    leaf_df: pd.DataFrame,
    stratify_by: list[str] | None = None,
) -> pd.DataFrame:
    """Count cells per decision tree leaf, optionally stratified by additional columns."""
    leaf_columns = [
        c
        for c in ["leaf_within_library", "leaf_cross_library", "leaf_cross_dataset", "failure_mode"]
        if c in leaf_df.columns
    ]

    if not leaf_columns:
        raise ValueError("No leaf columns found in leaf_df")

    all_summaries = []
    for leaf_col in leaf_columns:
        if stratify_by:
            group_cols = [leaf_col] + [c for c in stratify_by if c in leaf_df.columns]
        else:
            group_cols = [leaf_col]

        counts = leaf_df.groupby(group_cols, observed=True).size().reset_index(name="count")
        counts["fraction"] = counts["count"] / len(leaf_df)
        counts["level"] = leaf_col
        counts = counts.rename(columns={leaf_col: "leaf"})
        all_summaries.append(counts)

    return pd.concat(all_summaries, ignore_index=True)


def compute_distribution_overlap(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 50,
    range_: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Histogram-based overlap coefficient (OVL) between two 1-D distributions."""
    x_clean = np.asarray(x, dtype=np.float64)
    y_clean = np.asarray(y, dtype=np.float64)
    x_clean = x_clean[~np.isnan(x_clean)]
    y_clean = y_clean[~np.isnan(y_clean)]

    if len(x_clean) == 0 or len(y_clean) == 0:
        return 0.0

    h_x, _ = np.histogram(x_clean, bins=n_bins, range=range_)
    h_y, _ = np.histogram(y_clean, bins=n_bins, range=range_)

    sum_x = h_x.sum()
    sum_y = h_y.sum()

    if sum_x == 0 or sum_y == 0:
        return 0.0

    p_x = h_x / sum_x
    p_y = h_y / sum_y

    ovl = float(np.minimum(p_x, p_y).sum())
    return ovl


def compute_isolation_norm(
    metrics_df: pd.DataFrame,
    adata,
    mask_name: str,
    library_key: str | None = None,
    dataset_key: str | None = None,
    technical_key: str | None = None,
) -> float:
    """Observed isolation fraction / analytical expected isolation under random KNN.

    Per-mask ``P(random neighbour qualifies)``:

    ===================  =====================================  =============
    mask                 p_match                                keys used
    ===================  =====================================  =============
    same_library         ``(n_lib - 1) / (n_total - 1)``          library_key
    between_libraries    ``(n_total - n_lib) / (n_total - 1)``    library_key
    cross_library        ``(n_dataset - n_lib) / (n_total - 1)``  library_key + dataset_key
    cross_dataset        ``(n_total - n_dataset) / (n_total-1)``  dataset_key
    within_{tech}        ``(n_tech - 1) / (n_total - 1)``         technical_key
    between_{tech}       ``(n_total - n_tech) / (n_total - 1)``   technical_key
    cross_technical      ``(n_total - n_tech) / (n_total - 1)``   technical_key
    ===================  =====================================  =============

    ``P(isolated_in_mask) = (1 - p_match)^k`` per cell; the analytical
    expected isolation fraction is the mean over cells.

    Cells with NaN values in the relevant covariate column are excluded
    from both the observed and expected fractions; a warning is logged.
    """
    nn_col = f"n_neighbours_{mask_name}"
    if nn_col not in metrics_df.columns:
        return np.nan

    n_mask = metrics_df[nn_col].to_numpy()
    n_cells = len(n_mask)
    if n_cells == 0:
        return np.nan

    if "n_neighbours_total" in metrics_df.columns:
        degree = metrics_df["n_neighbours_total"].to_numpy(dtype=np.int64)
    else:
        degree = np.zeros(n_cells, dtype=np.int64)
        for c in ["n_neighbours_same_library", "n_neighbours_cross_library", "n_neighbours_cross_dataset"]:
            if c in metrics_df.columns:
                degree = degree + metrics_df[c].to_numpy(dtype=np.int64)

    n_total = adata.n_obs

    def _group_sizes(key: str) -> tuple[np.ndarray, np.ndarray]:
        col = adata.obs[key]
        nan = col.isna().to_numpy()
        if nan.any():
            _logger.warning(
                "compute_isolation_norm(%s): %d/%d cells have NaN '%s'; excluded from isolation baseline.",
                mask_name,
                int(nan.sum()),
                adata.n_obs,
                key,
            )
        counts = col.value_counts(dropna=True)
        return col.map(counts).to_numpy(dtype=np.float64), nan

    nan_mask = np.zeros(n_cells, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        if mask_name == "same_library":
            if library_key is None:
                return np.nan
            n_lib, lib_nan = _group_sizes(library_key)
            p_match = (n_lib - 1) / (n_total - 1)
            nan_mask |= lib_nan
        elif mask_name == "between_libraries":
            if library_key is None:
                return np.nan
            n_lib, lib_nan = _group_sizes(library_key)
            p_match = (n_total - n_lib) / (n_total - 1)
            nan_mask |= lib_nan
        elif mask_name == "cross_library":
            if library_key is None or dataset_key is None:
                return np.nan
            n_lib, lib_nan = _group_sizes(library_key)
            n_ds, ds_nan = _group_sizes(dataset_key)
            p_match = (n_ds - n_lib) / (n_total - 1)
            nan_mask |= lib_nan | ds_nan
        elif mask_name == "cross_dataset":
            if dataset_key is None:
                return np.nan
            n_ds, ds_nan = _group_sizes(dataset_key)
            p_match = (n_total - n_ds) / (n_total - 1)
            nan_mask |= ds_nan
        elif mask_name.startswith("within_"):
            if technical_key is None:
                return np.nan
            n_tech, t_nan = _group_sizes(technical_key)
            p_match = (n_tech - 1) / (n_total - 1)
            nan_mask |= t_nan
        elif mask_name == "cross_technical" or mask_name.startswith("between_"):
            if technical_key is None:
                return np.nan
            n_tech, t_nan = _group_sizes(technical_key)
            p_match = (n_total - n_tech) / (n_total - 1)
            nan_mask |= t_nan
        else:
            return np.nan

        p_match = np.clip(p_match, 0.0, 1.0)
        p_isolated = (1.0 - p_match) ** degree

    p_isolated = np.where(nan_mask, np.nan, p_isolated)

    expected_frac = float(np.nanmean(p_isolated))
    if not np.isfinite(expected_frac) or expected_frac == 0.0:
        return np.nan

    valid = ~nan_mask
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.nan
    observed_frac = float((n_mask[valid] == 0).sum()) / n_valid

    return observed_frac / expected_frac


def summarise_marker_correlation(
    metrics_df: pd.DataFrame,
    adata=None,
    library_key: str | None = None,
    dataset_key: str | None = None,
    technical_covariate_keys: list[str] | None = None,
    random_baseline_df: pd.DataFrame | None = None,
    min_neighbours: int = 1,
) -> pd.Series:
    """Per-model headline metrics (H1-H12, H14) for model comparison.

    When ``technical_covariate_keys`` is supplied and
    ``corr_avg_cross_technical`` exists in ``metrics_df`` (produced by
    :func:`compute_marker_correlation` with the same ``technical_covariate_keys``),
    the H14 headline ``cross_technical_correlation`` is included.
    """
    result = {}

    if "corr_avg_same_library" in metrics_df.columns:
        result["corr_within_library"] = float(np.nanmedian(metrics_df["corr_avg_same_library"].to_numpy()))
    else:
        result["corr_within_library"] = np.nan

    if "corr_std_same_library" in metrics_df.columns:
        result["corr_consistency"] = float(np.nanmedian(metrics_df["corr_std_same_library"].to_numpy()))
    else:
        result["corr_consistency"] = np.nan

    has_cross_library = "corr_avg_cross_library" in metrics_df.columns

    if has_cross_library:
        nn_xl = metrics_df["n_neighbours_cross_library"].to_numpy()
        mask_xl = nn_xl >= min_neighbours
        xl_vals = metrics_df["corr_avg_cross_library"].to_numpy()
        xl_filtered = xl_vals[mask_xl]
        result["corr_cross_library"] = float(np.nanmedian(xl_filtered)) if len(xl_filtered) > 0 else np.nan

        result["corr_gap_library"] = result["corr_within_library"] - result["corr_cross_library"]

        if adata is not None and library_key is not None and dataset_key is not None:
            result["isolation_norm_cross_library"] = compute_isolation_norm(
                metrics_df,
                adata,
                "cross_library",
                library_key=library_key,
                dataset_key=dataset_key,
            )
        else:
            result["isolation_norm_cross_library"] = np.nan

        if "corr_discrepancy_cross_library" in metrics_df.columns:
            result["discrepancy_cross_library"] = float(
                np.nanmedian(metrics_df["corr_discrepancy_cross_library"].to_numpy())
            )
        else:
            result["discrepancy_cross_library"] = np.nan

        sl_vals = metrics_df["corr_avg_same_library"].to_numpy()
        result["distrib_overlap_library"] = compute_distribution_overlap(sl_vals, xl_vals)
    else:
        result["corr_cross_library"] = np.nan
        result["corr_gap_library"] = np.nan
        result["isolation_norm_cross_library"] = np.nan
        result["discrepancy_cross_library"] = np.nan
        result["distrib_overlap_library"] = np.nan

    has_cross_dataset = "corr_avg_cross_dataset" in metrics_df.columns

    if has_cross_dataset and dataset_key is not None:
        nn_xd = metrics_df["n_neighbours_cross_dataset"].to_numpy()
        mask_xd = nn_xd >= min_neighbours
        xd_vals = metrics_df["corr_avg_cross_dataset"].to_numpy()
        xd_filtered = xd_vals[mask_xd]
        result["corr_cross_dataset"] = float(np.nanmedian(xd_filtered)) if len(xd_filtered) > 0 else np.nan

        result["corr_gap_dataset"] = result["corr_within_library"] - result["corr_cross_dataset"]

        if adata is not None:
            result["isolation_norm_cross_dataset"] = compute_isolation_norm(
                metrics_df,
                adata,
                "cross_dataset",
                dataset_key=dataset_key,
            )
        else:
            result["isolation_norm_cross_dataset"] = np.nan

        if "corr_discrepancy_cross_dataset" in metrics_df.columns:
            result["discrepancy_cross_dataset"] = float(
                np.nanmedian(metrics_df["corr_discrepancy_cross_dataset"].to_numpy())
            )
        else:
            result["discrepancy_cross_dataset"] = np.nan

        sl_vals_for_xd = metrics_df["corr_avg_same_library"].to_numpy()
        result["distrib_overlap_dataset"] = compute_distribution_overlap(sl_vals_for_xd, xd_vals)
    else:
        result["corr_cross_dataset"] = np.nan
        result["corr_gap_dataset"] = np.nan
        result["isolation_norm_cross_dataset"] = np.nan
        result["discrepancy_cross_dataset"] = np.nan
        result["distrib_overlap_dataset"] = np.nan

    if technical_covariate_keys and "corr_avg_cross_technical" in metrics_df.columns:
        result["cross_technical_correlation"] = compute_cross_technical_correlation(
            metrics_df,
            min_neighbours=min_neighbours,
        )
    else:
        result["cross_technical_correlation"] = np.nan

    return pd.Series(result)


def _nanweighted(values: list[float], weights: list[float]) -> float:
    """Weighted mean with NaN-skip: ``sum(v*w) / sum(w)`` over non-NaN entries.

    Returns ``np.nan`` if all inputs are NaN. Otherwise the remaining valid
    components are re-normalised so a partial score is still produced. This
    makes composite scores comparable even if one prerequisite metric is
    missing (e.g. ``isolation_norm_cross_dataset`` when there is only one
    dataset).
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = ~np.isnan(v)
    if not valid.any():
        return np.nan
    w_valid_sum = w[valid].sum()
    if w_valid_sum == 0:
        return np.nan
    return float(np.nansum(v * w) / w_valid_sum)


def compute_composite_score(
    headline: pd.Series,
    has_dataset: bool = True,
) -> pd.Series:
    """Composite integration score (scIB-style 60/40 bio/batch split).

    Each composite (``library_integration``, ``dataset_integration``,
    ``batch_correction``, ``total``) is a weighted mean of its inputs. NaN
    components are **skipped and the remaining weights are renormalised**
    (graceful degradation), so a partial score is produced as long as at
    least one input is finite. A single missing prerequisite metric (e.g.
    ``isolation_norm_cross_dataset`` in a single-dataset run) no longer
    poisons the total.

    ``_logger.info`` reports how many components were skipped per composite.
    """
    result = {}

    bio = headline.get("corr_within_library", np.nan)
    result["bio_conservation"] = bio

    h3 = headline.get("corr_cross_library", np.nan)
    h5 = headline.get("isolation_norm_cross_library", np.nan)
    h11 = headline.get("distrib_overlap_library", np.nan)

    h5_clipped = np.clip(h5, 0.0, 2.0) if not np.isnan(h5) else np.nan
    lib_inputs = [h3, 1.0 - h5_clipped if not np.isnan(h5_clipped) else np.nan, h11]
    lib_weights = [0.4, 0.3, 0.3]
    result["library_integration"] = _nanweighted(lib_inputs, lib_weights)
    n_lib_skipped = int(np.isnan(lib_inputs).sum())
    if n_lib_skipped > 0:
        _logger.info(
            "compute_composite_score: library_integration skipped %d/3 NaN components",
            n_lib_skipped,
        )

    if has_dataset:
        h7 = headline.get("corr_cross_dataset", np.nan)
        h9 = headline.get("isolation_norm_cross_dataset", np.nan)
        h12 = headline.get("distrib_overlap_dataset", np.nan)

        h9_clipped = np.clip(h9, 0.0, 2.0) if not np.isnan(h9) else np.nan
        ds_inputs = [h7, 1.0 - h9_clipped if not np.isnan(h9_clipped) else np.nan, h12]
        ds_weights = [0.4, 0.3, 0.3]
        result["dataset_integration"] = _nanweighted(ds_inputs, ds_weights)
        n_ds_skipped = int(np.isnan(ds_inputs).sum())
        if n_ds_skipped > 0:
            _logger.info(
                "compute_composite_score: dataset_integration skipped %d/3 NaN components",
                n_ds_skipped,
            )

        result["batch_correction"] = _nanweighted(
            [result["library_integration"], result["dataset_integration"]],
            [0.5, 0.5],
        )
    else:
        result["dataset_integration"] = np.nan
        result["batch_correction"] = result["library_integration"]

    result["total"] = _nanweighted([bio, result["batch_correction"]], [0.6, 0.4])

    return pd.Series(result)


def stratified_summary(
    metrics_df: pd.DataFrame,
    stratify_by: str | list[str],
    metrics_to_report: list[str] | None = None,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """Per-stratum median + percentiles of correlation metrics, grouped by stratify_by."""
    if isinstance(stratify_by, str):
        return _stratified_summary_single(metrics_df, stratify_by, metrics_to_report)

    result = {}
    for col in stratify_by:
        result[col] = _stratified_summary_single(metrics_df, col, metrics_to_report)
    return result


def _stratified_summary_single(
    metrics_df: pd.DataFrame,
    stratify_col: str,
    metrics_to_report: list[str] | None = None,
) -> pd.DataFrame:
    """Stratified summary for a single grouping column."""
    if stratify_col not in metrics_df.columns:
        raise ValueError(
            f"Column '{stratify_col}' not found in metrics_df. Available columns: {list(metrics_df.columns[:10])}..."
        )

    if metrics_to_report is None:
        metrics_to_report = [c for c in metrics_df.columns if c.startswith("corr_avg_") or c.startswith("corr_std_")]

    if not metrics_to_report:
        raise ValueError("No metrics columns to report")

    grouped = metrics_df.groupby(stratify_col, observed=True)

    rows = []
    for group_name, group_df in grouped:
        row = {"stratum": group_name, "n_cells": len(group_df)}
        for metric in metrics_to_report:
            if metric not in group_df.columns:
                row[f"{metric}_median"] = np.nan
                continue
            vals = group_df[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_median"] = float(np.nanmedian(vals))
            pcts = np.nanpercentile(vals, [10, 25, 75, 90])
            row[f"{metric}_p10"] = float(pcts[0])
            row[f"{metric}_p25"] = float(pcts[1])
            row[f"{metric}_p75"] = float(pcts[2])
            row[f"{metric}_p90"] = float(pcts[3])
        rows.append(row)

    return pd.DataFrame(rows)


def assemble_cross_model_metrics(
    per_model_metrics: dict[str, pd.DataFrame],
    shared_cell_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Combine per-model metrics into a single DataFrame with (model, metric) MultiIndex columns."""
    if not per_model_metrics:
        raise ValueError("per_model_metrics must be a non-empty dict")

    models = list(per_model_metrics.keys())

    if shared_cell_index is None:
        shared_cell_index = per_model_metrics[models[0]].index

    frames = {}
    for model_name, df in per_model_metrics.items():
        aligned = df.reindex(shared_cell_index)
        frames[model_name] = aligned

    result = pd.concat(frames, axis=1)
    result.columns = pd.MultiIndex.from_tuples(result.columns.tolist(), names=["model", "metric"])
    return result


def compute_best_achievable(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
) -> pd.Series:
    """Per-cell NaN-safe maximum of a metric across all models."""
    models = cross_model_df.columns.get_level_values("model").unique()
    stack = np.stack(
        [cross_model_df[(m, metric_name)].to_numpy() for m in models],
        axis=0,
    )
    best = np.nanmax(stack, axis=0)
    return pd.Series(best, index=cross_model_df.index, name="best_achievable")


def compute_integration_failure_rate(
    cross_model_df: pd.DataFrame,
    model_name: str,
    metric_name: str = "corr_avg_cross_dataset",
    threshold_high: float = 0.4,
    threshold_low: float = 0.2,
) -> float:
    """H13: fraction of **all cells** that are integrable by some model but fail for ``model_name``.

    Numerator: cells where ``best_achievable`` (across all models) is finite
    and > ``threshold_high``, AND this model's ``metric_name`` is NaN or
    below ``threshold_low``. Denominator: total cell count in
    ``cross_model_df`` (not restricted to the achievable subset).
    """
    best = compute_best_achievable(cross_model_df, metric_name).to_numpy()
    model_values = cross_model_df[(model_name, metric_name)].to_numpy()

    failure = np.isfinite(best) & (best > threshold_high) & (np.isnan(model_values) | (model_values < threshold_low))
    n_total = len(failure)
    if n_total == 0:
        return 0.0
    return float(failure.sum()) / n_total


def compute_model_pair_overlaps(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
    subset_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Pairwise distribution overlap (OVL) matrix between models for a given metric."""
    models = cross_model_df.columns.get_level_values("model").unique().tolist()

    arrays: dict[str, np.ndarray] = {}
    for m in models:
        vals = cross_model_df[(m, metric_name)].to_numpy(dtype=np.float64)
        if subset_mask is not None:
            mask_arr = np.asarray(subset_mask, dtype=bool)
            vals = vals[mask_arr]
        arrays[m] = vals

    n = len(models)
    ovl_matrix = np.ones((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            ovl = compute_distribution_overlap(arrays[models[i]], arrays[models[j]])
            ovl_matrix[i, j] = ovl
            ovl_matrix[j, i] = ovl

    return pd.DataFrame(ovl_matrix, index=models, columns=models)


def compute_contingency_per_cell(
    cross_model_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric_name: str = "corr_avg_cross_dataset",
    threshold_high: float = 0.4,
    threshold_low: float = 0.2,
) -> pd.DataFrame:
    """Per-cell 3x3 contingency classification for a model pair.

    Each cell is classified into one of 9 categories based on whether
    each model's metric is high (> *threshold_high*), low (<= *threshold_high*
    and finite), or absent (NaN — isolated, no neighbours).

    The category names are symmetrical:

    ======== ======== ========================
    A        B        Category
    ======== ======== ========================
    hi       hi       ``both_succeed``
    hi       low      ``A_ok_B_wrong_pairing``
    hi       NaN      ``A_ok_B_isolates``
    low      hi       ``B_ok_A_wrong_pairing``
    low      low      ``both_wrong_pairing``
    low      NaN      ``A_wrong_B_isolates``
    NaN      hi       ``B_ok_A_isolates``
    NaN      low      ``A_isolates_B_wrong``
    NaN      NaN      ``both_isolate_ambiguous``
    ======== ======== ========================

    Parameters
    ----------
    cross_model_df
        Output of :func:`assemble_cross_model_metrics`.
    model_a, model_b
        Model names to compare.
    metric_name
        Metric column.
    threshold_high
        Values strictly above this are "high" (succeed).
    threshold_low
        Not used as a separate threshold — the split is
        ``hi`` (> threshold_high) vs ``low`` (<= threshold_high, finite)
        vs ``NaN`` (isolated).  The parameter is accepted for API
        consistency with :func:`compute_integration_failure_rate`.

    Returns
    -------
    pd.DataFrame
        Single column ``category`` with string labels per cell.
    """
    a_vals = cross_model_df[(model_a, metric_name)].to_numpy(dtype=np.float64)
    b_vals = cross_model_df[(model_b, metric_name)].to_numpy(dtype=np.float64)

    a_hi = np.isfinite(a_vals) & (a_vals > threshold_high)
    a_lo = np.isfinite(a_vals) & ~a_hi
    a_nan = np.isnan(a_vals)

    b_hi = np.isfinite(b_vals) & (b_vals > threshold_high)
    b_lo = np.isfinite(b_vals) & ~b_hi
    b_nan = np.isnan(b_vals)

    conditions = [
        a_hi & b_hi,
        a_hi & b_lo,
        a_hi & b_nan,
        a_lo & b_hi,
        a_lo & b_lo,
        a_lo & b_nan,
        a_nan & b_hi,
        a_nan & b_lo,
        a_nan & b_nan,
    ]
    choices = [
        "both_succeed",
        "A_ok_B_wrong_pairing",
        "A_ok_B_isolates",
        "B_ok_A_wrong_pairing",
        "both_wrong_pairing",
        "A_wrong_B_isolates",
        "B_ok_A_isolates",
        "A_isolates_B_wrong",
        "both_isolate_ambiguous",
    ]

    category = np.select(conditions, choices, default="unclassified")

    return pd.DataFrame(
        {"category": category},
        index=cross_model_df.index,
    )


def compute_cross_technical_correlation(
    metrics_df: pd.DataFrame,
    min_neighbours: int = 1,
) -> float:
    """H14: median per-cell correlation between technical-covariate groups.

    Uses the ``corr_avg_cross_technical`` column produced by
    :func:`compute_marker_correlation` when ``technical_covariate_keys``
    was supplied. The ``cross_technical`` mask is the union of
    ``between_{tech_key}`` masks across all technical keys — neighbours
    are included if they differ in ANY technical covariate value.

    Parameters
    ----------
    metrics_df
        Per-cell DataFrame from :func:`compute_marker_correlation`.
    min_neighbours
        Filter: only cells with at least this many cross-technical
        neighbours contribute to the median.

    Returns
    -------
    float
        Median per-cell correlation on the ``cross_technical`` mask.
        ``np.nan`` if the column is missing (no technical keys were
        supplied) or no cells meet ``min_neighbours``.
    """
    corr_col = "corr_avg_cross_technical"
    nn_col = "n_neighbours_cross_technical"
    if corr_col not in metrics_df.columns:
        return np.nan

    vals = metrics_df[corr_col].to_numpy(dtype=np.float64)
    if nn_col in metrics_df.columns:
        valid = metrics_df[nn_col].to_numpy() >= min_neighbours
        vals = vals[valid]

    if len(vals) == 0 or not np.any(np.isfinite(vals)):
        return np.nan

    return float(np.nanmedian(vals))


def flag_consensus_isolated(
    cross_model_df: pd.DataFrame,
    metric_name: str = "corr_avg_cross_dataset",
    n_neighbours_col: str = "n_neighbours_cross_dataset",
    min_corr: float = 0.3,
) -> pd.Series:
    """Boolean flag for cells that NO model integrates cross-dataset.

    A cell is consensus-isolated when for ALL models either:

    - ``n_neighbours_cross_dataset == 0`` (no cross-dataset neighbours), OR
    - the correlation metric is NaN, OR
    - the correlation metric is below *min_corr*.

    These are candidate dataset-specific populations (label-free detection).

    The output is designed to feed into
    ``classify_failure_modes(..., model_comparison_result=consensus_flag)``
    to distinguish true dataset-specific populations (XD-0a) from
    integration failures fixable by a better model (XD-0b).

    Parameters
    ----------
    cross_model_df
        Output of :func:`assemble_cross_model_metrics`.
    metric_name
        Metric column to check.
    n_neighbours_col
        Column counting cross-dataset neighbours per cell.
    min_corr
        Minimum correlation to consider a cell "integrated" by a model.

    Returns
    -------
    pd.Series
        Boolean, ``True`` = consensus isolated (no model integrates this cell).
    """
    models = cross_model_df.columns.get_level_values("model").unique()
    n_cells = len(cross_model_df)

    all_fail = np.ones(n_cells, dtype=bool)

    for m in models:
        corr_vals = cross_model_df[(m, metric_name)].to_numpy(dtype=np.float64)

        if (m, n_neighbours_col) in cross_model_df.columns:
            nn_vals = cross_model_df[(m, n_neighbours_col)].to_numpy()
            model_fails = (nn_vals == 0) | np.isnan(corr_vals) | (corr_vals < min_corr)
        else:
            model_fails = np.isnan(corr_vals) | (corr_vals < min_corr)

        all_fail = all_fail & model_fails

    return pd.Series(
        all_fail,
        index=cross_model_df.index,
        name="consensus_isolated",
    )


def plot_marker_correlation_umap(
    adata,
    metrics_df: pd.DataFrame,
    columns: list[str] | None = None,
    leaf_df: pd.DataFrame | None = None,
    umap_key: str = "X_umap",
    figsize_per_panel: tuple[float, float] = (4.5, 4.0),
    cmap_divergent: str = "RdBu_r",
    cmap_sequential: str = "viridis",
    point_size: float = 0.5,
    nan_color: str = "#dddddd",
    nan_size: float = 0.3,
):
    """Grid of UMAPs coloured by per-cell neighbourhood correlation metrics.

    Parameters
    ----------
    adata
        AnnData with a UMAP embedding in ``adata.obsm[umap_key]``.
    metrics_df
        Per-cell metrics from :func:`compute_marker_correlation`.
        Must share (a subset of) the same index as ``adata.obs_names``.
    columns
        Metric columns to plot.  ``None`` selects all ``corr_avg_*``,
        ``corr_std_*``, ``corr_discrepancy_*``, ``n_neighbours_*``,
        ``corr_norm_by_library_*``, and ``corr_norm_by_all_*`` columns.
    leaf_df
        If provided, a DataFrame with a ``"leaf"`` column indexed like
        *metrics_df*.  An extra categorical panel is appended.
    umap_key
        Key in ``adata.obsm`` for UMAP coordinates.
    figsize_per_panel
        ``(width, height)`` per subplot.
    cmap_divergent
        Colourmap for correlation metrics (centred at 0).
    cmap_sequential
        Colourmap for count metrics (``n_neighbours_*``).
    point_size
        Scatter point size for valid cells.
    nan_color
        Colour for NaN cells.
    nan_size
        Scatter point size for NaN cells.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if columns is None:
        columns = [
            c
            for c in metrics_df.columns
            if c.startswith(
                (
                    "corr_avg_",
                    "corr_std_",
                    "corr_discrepancy_",
                    "n_neighbours_",
                    "corr_norm_by_library_",
                    "corr_norm_by_all_",
                )
            )
        ]

    n_panels = len(columns) + (1 if leaf_df is not None else 0)
    if n_panels == 0:
        raise ValueError("No columns to plot")

    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    umap = adata.obsm[umap_key]
    shared_idx = adata.obs_names.intersection(metrics_df.index)
    idx_pos = np.array([adata.obs_names.get_loc(i) for i in shared_idx])

    for panel_idx, col in enumerate(columns):
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        values = metrics_df.loc[shared_idx, col].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(values)

        nan_positions = idx_pos[~finite_mask]
        if len(nan_positions) > 0:
            ax.scatter(
                umap[nan_positions, 0],
                umap[nan_positions, 1],
                c=nan_color,
                s=nan_size,
                rasterized=True,
                zorder=1,
            )

        valid_positions = idx_pos[finite_mask]
        valid_values = values[finite_mask]

        is_count = col.startswith("n_neighbours_")
        if is_count:
            sc = ax.scatter(
                umap[valid_positions, 0],
                umap[valid_positions, 1],
                c=valid_values,
                cmap=cmap_sequential,
                s=point_size,
                rasterized=True,
                zorder=2,
            )
        else:
            if len(valid_values) > 0:
                vmax_abs = max(abs(np.nanmin(valid_values)), abs(np.nanmax(valid_values)))
                vmax_abs = max(vmax_abs, 1e-6)  # avoid degenerate norm
            else:
                vmax_abs = 1.0
            norm = TwoSlopeNorm(vcenter=0, vmin=-vmax_abs, vmax=vmax_abs)
            sc = ax.scatter(
                umap[valid_positions, 0],
                umap[valid_positions, 1],
                c=valid_values,
                cmap=cmap_divergent,
                norm=norm,
                s=point_size,
                rasterized=True,
                zorder=2,
            )

        fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(col, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if leaf_df is not None:
        panel_idx = len(columns)
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        leaf_values = leaf_df.reindex(shared_idx)["leaf"]
        unique_leaves = leaf_values.dropna().unique()
        cmap_cat = plt.colormaps.get_cmap("tab20").resampled(max(len(unique_leaves), 1))
        leaf_to_int = {leaf: i for i, leaf in enumerate(unique_leaves)}

        nan_leaf_mask = leaf_values.isna()
        nan_pos_leaf = idx_pos[nan_leaf_mask.values]
        if len(nan_pos_leaf) > 0:
            ax.scatter(
                umap[nan_pos_leaf, 0],
                umap[nan_pos_leaf, 1],
                c=nan_color,
                s=nan_size,
                rasterized=True,
                zorder=1,
            )
        valid_leaf_mask = ~nan_leaf_mask
        valid_pos_leaf = idx_pos[valid_leaf_mask.values]
        int_vals = leaf_values[valid_leaf_mask].map(leaf_to_int).to_numpy(dtype=np.float64)
        ax.scatter(
            umap[valid_pos_leaf, 0],
            umap[valid_pos_leaf, 1],
            c=int_vals,
            cmap=cmap_cat,
            s=point_size,
            rasterized=True,
            zorder=2,
        )
        ax.set_title("leaf", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_panels, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.tight_layout()
    return fig


def plot_metric_hist2d(
    metrics_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 100,
    range_: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    ax=None,
):
    """2D histogram of two per-cell metrics.

    NaN values are dropped before binning.  Colour scale uses
    ``LogNorm`` for visibility of sparse bins.

    Parameters
    ----------
    metrics_df
        Per-cell metrics DataFrame.
    x_col, y_col
        Column names for x and y axes.
    bins
        Number of bins per axis.
    range_
        ``((xmin, xmax), (ymin, ymax))`` for histogram.  ``None`` = auto.
    cmap
        Colourmap name.
    title
        Subplot title.  ``None`` uses ``"{y_col} vs {x_col}"``.
    ax
        Matplotlib Axes.  If ``None`` a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    x = metrics_df[x_col].to_numpy(dtype=np.float64)
    y = metrics_df[y_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) == 0:
        ax.text(0.5, 0.5, "No finite data", transform=ax.transAxes, ha="center", va="center")
        return ax

    ax.hist2d(x, y, bins=bins, range=range_, cmap=cmap, norm=LogNorm())
    ax.set_xlabel(x_col, fontsize=8)
    ax.set_ylabel(y_col, fontsize=8)
    ax.set_title(title or f"{y_col} vs {x_col}", fontsize=9)
    return ax


def plot_failure_mode_scatter(
    metrics_df: pd.DataFrame,
    figsize: tuple[float, float] = (20, 12),
    bins: int = 80,
):
    """Multi-panel hist2d grid covering key failure mode comparisons.

    Panels (one per mask where applicable):

    - ``corr_avg_same_library`` vs ``corr_avg_cross_library`` (within vs cross library)
    - ``corr_avg_same_library`` vs ``corr_avg_cross_dataset`` (within library vs cross dataset)
    - ``corr_avg_cross_library`` vs ``corr_avg_cross_dataset`` (cross library vs cross dataset)
    - ``corr_discrepancy_same_library`` vs ``corr_avg_same_library``
    - ``corr_discrepancy_cross_library`` vs ``corr_avg_cross_library``
    - ``corr_discrepancy_cross_dataset`` vs ``corr_avg_cross_dataset``
    - ``corr_avg_same_library`` vs ``corr_avg_random_same_library`` (model vs random)
    - ``corr_avg_cross_dataset`` vs ``corr_avg_random_cross_dataset`` (model vs random)

    Only panels whose columns exist in *metrics_df* are shown.

    Parameters
    ----------
    metrics_df
        Per-cell metrics DataFrame.
    figsize
        Figure size.
    bins
        Bins per axis for hist2d.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    panel_specs = [
        ("corr_avg_same_library", "corr_avg_cross_library", "within vs cross library"),
        ("corr_avg_same_library", "corr_avg_cross_dataset", "within library vs cross dataset"),
        ("corr_avg_cross_library", "corr_avg_cross_dataset", "cross library vs cross dataset"),
        ("corr_avg_same_library", "corr_discrepancy_same_library", "discrepancy within library"),
        ("corr_avg_cross_library", "corr_discrepancy_cross_library", "discrepancy cross library"),
        ("corr_avg_cross_dataset", "corr_discrepancy_cross_dataset", "discrepancy cross dataset"),
        ("corr_avg_same_library", "corr_avg_random_same_library", "model vs random (library)"),
        ("corr_avg_cross_dataset", "corr_avg_random_cross_dataset", "model vs random (dataset)"),
    ]

    valid_panels = [(x, y, t) for x, y, t in panel_specs if x in metrics_df.columns and y in metrics_df.columns]

    if not valid_panels:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No matching columns found", transform=ax.transAxes, ha="center", va="center")
        return fig

    ncols = min(4, len(valid_panels))
    nrows = int(np.ceil(len(valid_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (x_col, y_col, title) in enumerate(valid_panels):
        ax = axes[idx // ncols, idx % ncols]
        plot_metric_hist2d(metrics_df, x_col, y_col, bins=bins, title=title, ax=ax)

    for idx in range(len(valid_panels), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Failure mode scatter grid", fontsize=12)
    fig.tight_layout()
    return fig


def plot_distribution_overlap(
    metrics_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str | None = None,
    label_b: str | None = None,
    bins: int = 50,
    range_: tuple[float, float] = (-1.0, 1.0),
    ax=None,
):
    """Overlay two metric distributions for visual comparison.

    Useful for comparing same-batch vs cross-batch (V7) or
    within-library vs cross-dataset (V8) distributions.

    Parameters
    ----------
    metrics_df
        Per-cell metrics DataFrame.
    col_a, col_b
        Column names for the two distributions.
    label_a, label_b
        Legend labels (default: column names).
    bins
        Number of histogram bins.
    range_
        ``(min, max)`` range for histograms.
    ax
        Matplotlib Axes.  ``None`` creates a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    a_vals = metrics_df[col_a].dropna().to_numpy()
    b_vals = metrics_df[col_b].dropna().to_numpy()

    ax.hist(a_vals, bins=bins, range=range_, alpha=0.5, density=True, label=label_a or col_a, color="#1f77b4")
    ax.hist(b_vals, bins=bins, range=range_, alpha=0.5, density=True, label=label_b or col_b, color="#ff7f0e")
    ax.legend(fontsize=8)
    ax.set_xlabel("Correlation", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{col_a} vs {col_b}", fontsize=10)
    return ax


def plot_per_library_distributions(
    metrics_df: pd.DataFrame,
    adata,
    metric_col: str = "corr_avg_cross_library",
    library_key: str = "batch",
    dataset_key: str = "dataset",
    figsize_per_row: tuple[float, float] = (8, 2.5),
):
    """KDE lines per library, one row per dataset.

    For datasets with only one library the panel shows a note instead
    of an empty plot.

    Parameters
    ----------
    metrics_df
        Per-cell metrics DataFrame (index aligns to ``adata.obs_names``).
    adata
        AnnData with ``adata.obs[library_key]`` and
        ``adata.obs[dataset_key]``.
    metric_col
        Metric column in *metrics_df* to plot.
    library_key
        ``adata.obs`` column for library / batch.
    dataset_key
        ``adata.obs`` column for dataset.
    figsize_per_row
        ``(width, height)`` per dataset row.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    shared_idx = metrics_df.index.intersection(adata.obs_names)
    obs = adata.obs.loc[shared_idx, [library_key, dataset_key]].copy()
    obs[metric_col] = metrics_df.loc[shared_idx, metric_col]

    datasets = sorted(obs[dataset_key].dropna().unique())
    nrows = len(datasets)
    if nrows == 0:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No datasets found", transform=ax.transAxes, ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * nrows),
        squeeze=False,
    )

    cmap = plt.colormaps["tab10"]

    for row, ds in enumerate(datasets):
        ax = axes[row, 0]
        ds_mask = obs[dataset_key] == ds
        ds_obs = obs.loc[ds_mask]
        libraries = sorted(ds_obs[library_key].dropna().unique())

        if len(libraries) <= 1:
            ax.text(
                0.5, 0.5, f"{ds}: single library, skip", transform=ax.transAxes, ha="center", va="center", fontsize=9
            )
            ax.set_title(ds, fontsize=9)
            continue

        for lib_idx, lib in enumerate(libraries):
            vals = ds_obs.loc[ds_obs[library_key] == lib, metric_col].dropna()
            if len(vals) < 3:
                continue
            from scipy.stats import gaussian_kde

            try:
                kde = gaussian_kde(vals.to_numpy())
                x_grid = np.linspace(-1, 1, 200)
                ax.plot(x_grid, kde(x_grid), label=lib, color=cmap(lib_idx % 10), linewidth=1.2)
            except np.linalg.LinAlgError:
                ax.axvline(vals.median(), color=cmap(lib_idx % 10), linestyle="--", label=lib)

        ax.set_title(ds, fontsize=9)
        ax.set_xlabel(metric_col, fontsize=8)
        ax.legend(fontsize=6, ncol=2, loc="upper left")

    fig.tight_layout()
    return fig


def plot_isolation_bars(
    per_model_headlines: dict[str, pd.Series],
    metric: str = "isolation_norm_cross_dataset",
    figsize: tuple[float, float] = (8, 5),
):
    """Bar chart of a headline metric across models.

    Parameters
    ----------
    per_model_headlines
        ``{model_name: headline_series}`` where each Series is the output
        of :func:`summarise_marker_correlation`.
    metric
        Headline metric key to plot.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    models = []
    values = []
    for name, series in per_model_headlines.items():
        val = series.get(metric, np.nan)
        models.append(name)
        values.append(val)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    colors = plt.colormaps["tab10"](np.arange(len(models)) % 10)
    ax.bar(x, values, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(f"Isolation: {metric}", fontsize=10)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="random baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_leaf_distribution(
    per_model_leaves: dict[str, pd.Series],
    figsize: tuple[float, float] = (10, 5),
    max_leaves: int = 30,
):
    """Stacked bar chart showing fraction of cells per leaf, per model.

    Parameters
    ----------
    per_model_leaves
        ``{model_name: leaf_series}`` where each Series has one entry per
        cell with the leaf label (from :func:`assign_leaf_labels`).
    figsize
        Figure size.
    max_leaves
        Maximum number of leaves to show (by frequency). Remaining are
        grouped as ``"other"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fractions = {}
    all_leaves = set()
    for model_name, leaf_series in per_model_leaves.items():
        counts = leaf_series.value_counts(normalize=True)
        fractions[model_name] = counts
        all_leaves.update(counts.index.tolist())

    if len(all_leaves) > max_leaves:
        combined_counts = pd.Series(dtype=np.float64)
        for counts in fractions.values():
            combined_counts = combined_counts.add(counts, fill_value=0)
        top_leaves = combined_counts.nlargest(max_leaves).index.tolist()
    else:
        top_leaves = sorted(all_leaves)

    models = list(fractions.keys())
    leaf_data = {}
    for leaf in top_leaves:
        leaf_data[str(leaf)] = [fractions[m].get(leaf, 0.0) for m in models]

    other_vals = []
    for m in models:
        other = sum(v for k, v in fractions[m].items() if k not in top_leaves)
        other_vals.append(other)
    if any(v > 0 for v in other_vals):
        leaf_data["other"] = other_vals

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(leaf_data), 1))

    for i, (leaf_name, vals) in enumerate(leaf_data.items()):
        ax.bar(x, vals, bottom=bottom, label=leaf_name, color=cmap(i % 20), edgecolor="white", linewidth=0.3)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of cells", fontsize=9)
    ax.set_title("Leaf distribution per model", fontsize=10)
    ax.legend(fontsize=6, ncol=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def classify_cell_quality(
    metrics_df: pd.DataFrame,
    adata,
    library_key: str,
    gene_group_metrics: dict[str, pd.DataFrame] | None = None,
    ambient_frac: pd.Series | None = None,
    recon_perplexity: pd.Series | None = None,
    corr_deviation_threshold: float = 0.0,
    lineage_corr_threshold: float = 0.3,
    ambient_frac_threshold: float = 0.5,
    perplexity_threshold: float = 50.0,
) -> pd.DataFrame:
    """Classify cells as good, rare, poor-quality, or uncertain.

    Multi-axis classification that combines neighbourhood correlation
    deviation from library median with optional model-based QC metrics
    (ambient RNA fraction, reconstruction perplexity).

    The classification proceeds in four steps:

    1. **Correlation deviation**: ``corr_avg_same_library`` minus the median
       of ``corr_avg_same_library`` within the same library.  Cells with
       deviation >= *corr_deviation_threshold* match their peers and are
       classified as ``"good"``.

    2. **Best lineage correlation**: When *gene_group_metrics* is provided,
       the maximum ``corr_avg_same_library`` across lineage gene groups is
       computed.  If this exceeds *lineage_corr_threshold*, the cell is
       classified as ``"rare"`` (matches one lineage well despite low
       overall correlation).

    3. **Model QC** (optional): When *ambient_frac* and/or
       *recon_perplexity* are provided, cells with ``ambient_frac >
       ambient_frac_threshold`` OR ``recon_perplexity <
       perplexity_threshold`` are classified as ``"poor_quality"``.  When
       neither is provided, this step is skipped.

    4. All remaining cells are classified as ``"uncertain"``.

    Parameters
    ----------
    metrics_df
        Per-cell metrics DataFrame from :func:`compute_marker_correlation`.
        Must contain ``corr_avg_same_library``.
    adata
        AnnData with ``adata.obs[library_key]``.
    library_key
        Column in ``adata.obs`` identifying the library / batch for each cell.
    gene_group_metrics
        ``{group_name: per_cell_metrics_df}`` for lineage-level gene groups.
        Each value must contain ``corr_avg_same_library``.  Used in Step 2
        to identify rare cell types.  When ``None``, Step 2 is skipped.
    ambient_frac
        Per-cell ambient RNA fraction (``pd.Series``, index matching
        *metrics_df*).  From ``get_latent_qc_metrics()`` (not yet
        implemented).  ``None`` = skip ambient check.
    recon_perplexity
        Per-cell reconstruction perplexity (``pd.Series``, index matching
        *metrics_df*).  From ``get_latent_qc_metrics()`` (not yet
        implemented).  ``None`` = skip perplexity check.
    corr_deviation_threshold
        Minimum correlation deviation (from library median) for a cell to
        be classified as ``"good"``.  Default 0.0 (at or above median).
    lineage_corr_threshold
        Minimum best-lineage correlation for ``"rare"`` classification.
    ambient_frac_threshold
        Above this value, cell is flagged as ``"poor_quality"`` (Step 3).
    perplexity_threshold
        Below this value, cell is flagged as ``"poor_quality"`` (Step 3).

    Returns
    -------
    pd.DataFrame
        Indexed like *metrics_df*, with columns:

        - ``quality_corr_deviation`` : float — correlation deviation from
          library median.
        - ``quality_best_lineage_corr`` : float — max lineage group
          correlation (NaN if no gene groups).
        - ``quality_best_lineage`` : str — name of the best lineage group
          (NaN if no gene groups).
        - ``quality_ambient_frac`` : float — from input (NaN if not
          provided).
        - ``quality_recon_perplexity`` : float — from input (NaN if not
          provided).
        - ``quality_classification`` : categorical — one of ``"good"``,
          ``"rare"``, ``"poor_quality"``, ``"uncertain"``.
    """
    n = len(metrics_df)
    idx = metrics_df.index

    corr_col = "corr_avg_same_library"
    if corr_col not in metrics_df.columns:
        raise ValueError(f"metrics_df must contain '{corr_col}' column")

    corr_vals = metrics_df[corr_col].to_numpy(dtype=np.float64)

    obs_aligned = adata.obs.reindex(idx)
    lib_labels = obs_aligned[library_key].to_numpy()

    lib_series = pd.Series(corr_vals, index=idx)
    lib_cat = pd.Series(lib_labels, index=idx)
    library_medians = lib_series.groupby(lib_cat).transform("median").to_numpy()

    corr_deviation = corr_vals - library_medians

    best_lineage_corr = np.full(n, np.nan, dtype=np.float64)
    best_lineage_name = np.full(n, np.nan, dtype=object)

    if gene_group_metrics is not None and len(gene_group_metrics) > 0:
        group_names = []
        group_corrs = []
        for gname, gdf in gene_group_metrics.items():
            if corr_col in gdf.columns:
                aligned = gdf.reindex(idx)[corr_col].to_numpy(dtype=np.float64)
                group_names.append(gname)
                group_corrs.append(aligned)

        if len(group_corrs) > 0:
            corr_matrix = np.column_stack(group_corrs)  # (n_cells, n_groups)
            all_nan_mask = np.all(np.isnan(corr_matrix), axis=1)

            # nanmax/nanargmax raise on all-NaN slices; use -inf fill
            safe_matrix = corr_matrix.copy()
            safe_matrix[all_nan_mask, :] = -np.inf

            best_idx = np.argmax(safe_matrix, axis=1)
            best_lineage_corr = np.max(safe_matrix, axis=1)
            best_lineage_corr[all_nan_mask] = np.nan

            name_arr = np.array(group_names, dtype=object)
            best_lineage_name = np.where(all_nan_mask, np.nan, name_arr[best_idx])

    af_vals = np.full(n, np.nan, dtype=np.float64)
    if ambient_frac is not None:
        af_vals = ambient_frac.reindex(idx).to_numpy(dtype=np.float64)

    rp_vals = np.full(n, np.nan, dtype=np.float64)
    if recon_perplexity is not None:
        rp_vals = recon_perplexity.reindex(idx).to_numpy(dtype=np.float64)

    is_good = corr_deviation >= corr_deviation_threshold

    has_lineage = ~np.isnan(best_lineage_corr)
    is_rare = has_lineage & (best_lineage_corr > lineage_corr_threshold)

    has_model_qc = (ambient_frac is not None) or (recon_perplexity is not None)
    if has_model_qc:
        is_poor_af = np.where(np.isnan(af_vals), False, af_vals > ambient_frac_threshold)
        is_poor_rp = np.where(np.isnan(rp_vals), False, rp_vals < perplexity_threshold)
        is_poor = is_poor_af | is_poor_rp
    else:
        is_poor = np.zeros(n, dtype=bool)

    conditions = [is_good, is_rare, is_poor]
    choices = ["good", "rare", "poor_quality"]
    classification = np.select(conditions, choices, default="uncertain")

    result = pd.DataFrame(
        {
            "quality_corr_deviation": corr_deviation,
            "quality_best_lineage_corr": best_lineage_corr,
            "quality_best_lineage": best_lineage_name,
            "quality_ambient_frac": af_vals,
            "quality_recon_perplexity": rp_vals,
            "quality_classification": pd.Categorical(
                classification,
                categories=["good", "rare", "poor_quality", "uncertain"],
            ),
        },
        index=idx,
    )
    return result
