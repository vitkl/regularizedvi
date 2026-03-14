"""Gene and cell filtering utilities.

Gene filtering copied from cell2location (BayraktarLab/cell2location) to avoid a dependency.
Cell QC filtering for multi-dataset integration.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


def _filter_genes_naive(adata, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12):
    """Original filter_genes implementation (kept for testing).

    Materializes a full boolean sparse matrix via ``X > 0``, which uses
    O(nnz) extra memory. For large datasets (>1M cells), prefer
    :func:`filter_genes` which computes stats in batches.
    """
    adata.var["n_cells"] = np.array((adata.X > 0).sum(0)).flatten()
    adata.var["nonz_mean"] = np.array(adata.X.sum(0)).flatten() / adata.var["n_cells"]

    cell_count_cutoff = np.log10(cell_count_cutoff)
    cell_count_cutoff2 = np.log10(adata.shape[0] * cell_percentage_cutoff2)
    nonz_mean_cutoff = np.log10(nonz_mean_cutoff)

    gene_selection = (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2)) | (
        np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
        & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
    )
    return adata.var_names[gene_selection]


def _compute_gene_stats(X, batch_size=5000):
    """Compute per-gene n_cells and sum in batches to limit memory.

    Instead of ``(X > 0).sum(0)`` which allocates a full boolean sparse
    matrix, this iterates over cell-batches and uses
    ``chunk.getnnz(axis=0)`` (CSR pointer arithmetic, no copy).
    """
    n_cells_total = np.zeros(X.shape[1], dtype=np.int64)
    gene_sum_total = np.zeros(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[0], batch_size):
        chunk = X[start : start + batch_size]
        if sparse.issparse(chunk):
            n_cells_total += np.array(chunk.getnnz(axis=0))
            gene_sum_total += np.array(chunk.sum(axis=0)).flatten()
        else:
            n_cells_total += np.count_nonzero(chunk, axis=0)
            gene_sum_total += chunk.sum(axis=0)
    return n_cells_total, gene_sum_total


def filter_genes(adata, cell_count_cutoff=15, cell_percentage_cutoff2=0.05, nonz_mean_cutoff=1.12):
    r"""Plot the gene filter given a set of cutoffs and return resulting list of genes.

    Memory-efficient version that computes gene stats in batches of cells,
    avoiding the full boolean sparse matrix from ``X > 0``.

    Parameters
    ----------
    adata :
        anndata object with single cell / nucleus data.
    cell_count_cutoff :
        All genes detected in less than cell_count_cutoff cells will be excluded.
    cell_percentage_cutoff2 :
        All genes detected in at least this percentage of cells will be included.
    nonz_mean_cutoff :
        genes detected in the number of cells between the above mentioned cutoffs are selected
        only when their average expression in non-zero cells is above this cutoff.

    Returns
    -------
    a list of selected var_names
    """
    n_cells, gene_sum = _compute_gene_stats(adata.X)
    adata.var["n_cells"] = n_cells
    adata.var["nonz_mean"] = gene_sum / np.where(n_cells > 0, n_cells, 1)

    cell_count_cutoff = np.log10(cell_count_cutoff)
    cell_count_cutoff2 = np.log10(adata.shape[0] * cell_percentage_cutoff2)
    nonz_mean_cutoff = np.log10(nonz_mean_cutoff)

    gene_selection = (np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff2)) | (
        np.array(np.log10(adata.var["n_cells"]) > cell_count_cutoff)
        & np.array(np.log10(adata.var["nonz_mean"]) > nonz_mean_cutoff)
    )
    gene_selection = adata.var_names[gene_selection]
    adata_shape = adata[:, gene_selection].shape

    fig, ax = plt.subplots()
    ax.hist2d(
        np.log10(adata.var["nonz_mean"]),
        np.log10(adata.var["n_cells"]),
        bins=100,
        norm=matplotlib.colors.LogNorm(),
        range=[[0, 0.5], [1, 4.5]],
    )
    ax.axvspan(
        0,
        nonz_mean_cutoff,
        ymin=0.0,
        ymax=(cell_count_cutoff2 - 1) / 3.5,
        color="darkorange",
        alpha=0.3,
    )
    ax.axvspan(
        nonz_mean_cutoff,
        np.max(np.log10(adata.var["nonz_mean"])),
        ymin=0.0,
        ymax=(cell_count_cutoff - 1) / 3.5,
        color="darkorange",
        alpha=0.3,
    )
    plt.vlines(nonz_mean_cutoff, cell_count_cutoff, cell_count_cutoff2, color="darkorange")
    plt.hlines(cell_count_cutoff, nonz_mean_cutoff, 1, color="darkorange")
    plt.hlines(cell_count_cutoff2, 0, nonz_mean_cutoff, color="darkorange")
    plt.xlabel("Mean non-zero expression level of gene (log)")
    plt.ylabel("Number of cells expressing gene (log)")
    plt.title(f"Gene filter: {adata_shape[0]} cells x {adata_shape[1]} genes")
    plt.show()

    return gene_selection


def plot_qc_histograms(
    adata,
    count_lo: float = 1000,
    count_hi: float = 80000,
    gene_lo: float = 500,
    gene_hi: float = 10000,
    mt_threshold: float = 0.20,
    doublet_threshold: float = 0.18,
):
    """Plot histograms of total counts, genes, MT fraction, and doublet score.

    Parameters
    ----------
    adata
        AnnData with QC metrics. ``total_counts`` and ``n_genes`` are computed
        from ``X`` if not in ``.obs``. ``mt_frac`` and ``doublet_score`` are
        read from ``.obs``.
    count_lo, count_hi
        Vertical lines for total counts thresholds.
    gene_lo, gene_hi
        Vertical lines for gene count thresholds.
    mt_threshold
        Vertical line for MT fraction threshold.
    doublet_threshold
        Vertical line for doublet score threshold.
    """
    from matplotlib import rcParams

    rcParams["figure.figsize"] = 12, 3
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    total_counts = np.array(adata.X.sum(1)).squeeze()
    n_genes = np.array((adata.X > 0).sum(1)).squeeze()

    axes[0].hist(np.log10(total_counts[total_counts > 0]), bins=100)
    axes[0].axvline(np.log10(count_lo), color="red", linestyle="--")
    axes[0].axvline(np.log10(count_hi), color="red", linestyle="--")
    axes[0].set_xlabel("log10(UMI counts)")
    axes[0].set_title(f"Total counts (n={len(total_counts)}, zero={np.sum(total_counts == 0)})")

    axes[1].hist(np.log10(n_genes[n_genes > 0]), bins=100)
    axes[1].axvline(np.log10(gene_lo), color="red", linestyle="--")
    axes[1].axvline(np.log10(gene_hi), color="red", linestyle="--")
    axes[1].set_xlabel("log10(genes)")
    axes[1].set_title("Genes detected")

    if "mt_frac" in adata.obs.columns:
        axes[2].hist(adata.obs["mt_frac"].dropna(), bins=100)
        axes[2].axvline(mt_threshold, color="red", linestyle="--")
        axes[2].set_xlabel("MT fraction")
        axes[2].set_title("Mitochondrial fraction")

    if "doublet_score" in adata.obs.columns:
        axes[3].hist(adata.obs["doublet_score"].dropna(), bins=100)
        axes[3].axvline(doublet_threshold, color="red", linestyle="--")
        axes[3].set_xlabel("Doublet score")
        axes[3].set_title("Scrublet doublet score")

    plt.tight_layout()
    plt.show()
    return fig


def print_qc_summary(
    adata,
    dataset_key: str = "dataset",
    metrics: tuple[str, ...] = ("total_counts", "n_genes", "mt_frac", "doublet_score", "total_fragments"),
):
    """Print per-dataset QC metric distributions.

    Shows min, q10, q25, median, q75, q90, max for each metric and dataset,
    plus an ALL row aggregating across datasets.

    Parameters
    ----------
    adata
        AnnData with QC metrics in ``.obs``.
    dataset_key
        Column in ``.obs`` identifying datasets/studies.
    metrics
        Tuple of column names in ``.obs`` to summarize.
    """
    quantiles = [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    q_names = ["min", "q10", "q25", "median", "q75", "q90", "max"]

    datasets = sorted(adata.obs[dataset_key].unique())
    for metric in metrics:
        if metric not in adata.obs.columns:
            print(f"\n=== {metric} === (not in obs, skipping)")
            continue
        print(f"\n=== {metric} ===")
        rows = []
        fmt = ".4f" if metric in ("mt_frac", "doublet_score") else ".0f"
        for ds in datasets:
            vals = adata.obs.loc[adata.obs[dataset_key] == ds, metric].dropna()
            qs = np.quantile(vals, quantiles)
            rows.append([ds, len(vals)] + [f"{q:{fmt}}" for q in qs])
        vals = adata.obs[metric].dropna()
        qs = np.quantile(vals, quantiles)
        rows.append(["ALL", len(vals)] + [f"{q:{fmt}}" for q in qs])
        df = pd.DataFrame(rows, columns=["dataset", "n_cells"] + q_names)
        print(df.to_string(index=False))


def compound_qc_filter(
    adata,
    *,
    dataset_key: str = "dataset",
    batch_key: str = "batch",
    lower_quantile_metrics: tuple[str, ...] = ("total_counts", "n_genes", "total_fragments"),
    lower_quantile: float = 0.15,
    upper_quantile_metrics: tuple[str, ...] = ("doublet_score",),
    upper_quantile: float = 0.95,
    outlier_metrics: tuple[str, ...] = ("total_counts", "n_genes", "total_fragments"),
    outlier_threshold: float = 2.5,
    log_scale_metrics: tuple[str, ...] = ("total_counts", "n_genes", "total_fragments"),
    per_study_cutoffs: dict[str, dict] | None = None,
    global_cutoffs: dict | None = None,
    min_cells_per_batch: int = 100,
):
    """3-stage compound cell QC filter for multi-dataset integration.

    All quantiles and per-batch statistics are computed on the **unfiltered**
    data before any removal. The three stages build independent boolean masks
    that are combined (AND) at the end.

    Parameters
    ----------
    adata
        AnnData with QC metrics in ``.obs``.
    dataset_key
        Column in ``.obs`` identifying datasets/studies.
    batch_key
        Column in ``.obs`` identifying batches/samples.
    lower_quantile_metrics
        Metrics for per-study lower-quantile removal (count-like metrics).
    lower_quantile
        Per-study quantile below which cells are removed for ``lower_quantile_metrics``.
    upper_quantile_metrics
        Metrics for per-study upper-quantile removal (e.g. doublet_score).
        mt_frac is intentionally excluded — use absolute cutoffs in stage 3
        to avoid penalizing experiments with genuinely low MT%.
    upper_quantile
        Per-study quantile above which cells are removed for ``upper_quantile_metrics``.
    outlier_metrics
        Metrics for per-batch outlier removal (stage 2).
    outlier_threshold
        Number of standard deviations for outlier detection (mean +/- threshold*std).
    log_scale_metrics
        Metrics where mean/std are computed on log10 scale. Others use linear.
    per_study_cutoffs
        Per-study manual cutoffs. Keys: study names. Values: dicts with any of
        ``min_counts``, ``max_counts``, ``min_genes``, ``max_genes``,
        ``min_fragments``, ``max_fragments``, ``max_mt``, ``max_doublet``.
    global_cutoffs
        Fallback cutoffs for studies not in ``per_study_cutoffs``. Same keys as above.
    min_cells_per_batch
        Drop batches with fewer cells after all filtering stages.

    Returns
    -------
    Filtered AnnData (copy).
    """
    n_start = adata.n_obs
    datasets = sorted(adata.obs[dataset_key].unique())
    if per_study_cutoffs is None:
        per_study_cutoffs = {}
    if global_cutoffs is None:
        global_cutoffs = {}

    # Map cutoff keys to obs columns
    _cutoff_col = {
        "min_counts": "total_counts",
        "max_counts": "total_counts",
        "min_genes": "n_genes",
        "max_genes": "n_genes",
        "min_fragments": "total_fragments",
        "max_fragments": "total_fragments",
        "max_mt": "mt_frac",
        "max_doublet": "doublet_score",
    }

    # --- Stage 1: per-study quantile removal (computed on unfiltered data) ---
    mask_q = np.ones(adata.n_obs, dtype=bool)
    print("=== Stage 1: Per-study quantile removal ===")
    for ds in datasets:
        ds_idx = adata.obs[dataset_key] == ds
        n_ds = ds_idx.sum()
        ds_mask = np.ones(n_ds, dtype=bool)
        thresholds = []
        for metric in lower_quantile_metrics:
            if metric not in adata.obs.columns:
                continue
            vals = adata.obs.loc[ds_idx, metric]
            q_val = np.nanquantile(vals, lower_quantile)
            ds_mask &= vals.values >= q_val
            thresholds.append(f"{metric}>={q_val:.1f}")
        for metric in upper_quantile_metrics:
            if metric not in adata.obs.columns:
                continue
            vals = adata.obs.loc[ds_idx, metric]
            q_val = np.nanquantile(vals, upper_quantile)
            ds_mask &= vals.values <= q_val
            thresholds.append(f"{metric}<={q_val:.4f}")
        mask_q[ds_idx.values] = ds_mask
        n_removed = n_ds - ds_mask.sum()
        print(f"  {ds}: {n_removed:,} removed ({n_removed / n_ds * 100:.1f}%) — {', '.join(thresholds)}")
    n_s1 = mask_q.sum()
    print(f"  Stage 1 total: {n_start:,} -> {n_s1:,} ({n_s1 / n_start * 100:.1f}% kept)\n")

    # --- Stage 2: per-batch outlier removal (computed on unfiltered data) ---
    mask_o = np.ones(adata.n_obs, dtype=bool)
    print(f"=== Stage 2: Per-batch outlier removal (mean +/- {outlier_threshold:.1f}*std) ===")
    batch_stats = []
    for batch in sorted(adata.obs[batch_key].unique()):
        b_idx = adata.obs[batch_key] == batch
        n_b = b_idx.sum()
        b_mask = np.ones(n_b, dtype=bool)
        for metric in outlier_metrics:
            if metric not in adata.obs.columns:
                continue
            vals = adata.obs.loc[b_idx, metric].values.astype(np.float64)
            if metric in log_scale_metrics:
                vals_t = np.log10(np.maximum(vals, 1))
            else:
                vals_t = vals
            mu = np.nanmean(vals_t)
            sd = np.nanstd(vals_t)
            lo = mu - outlier_threshold * sd
            hi = mu + outlier_threshold * sd
            b_mask &= (vals_t >= lo) & (vals_t <= hi)
        mask_o[b_idx.values] = b_mask
        n_removed = n_b - b_mask.sum()
        ds = adata.obs.loc[b_idx, dataset_key].iloc[0]
        batch_stats.append((ds, batch, n_b, n_removed))

    # Summarize per dataset
    stats_df = pd.DataFrame(batch_stats, columns=["dataset", "batch", "n_cells", "n_removed"])
    for ds in datasets:
        ds_rows = stats_df[stats_df["dataset"] == ds]
        n_ds = ds_rows["n_cells"].sum()
        n_rem = ds_rows["n_removed"].sum()
        print(f"  {ds}: {n_rem:,} removed ({n_rem / n_ds * 100:.1f}%) across {len(ds_rows)} batches")
    n_s2 = mask_o.sum()
    print(f"  Stage 2 total: {n_start:,} -> {n_s2:,} ({n_s2 / n_start * 100:.1f}% kept)\n")

    # --- Stage 3: manual cutoffs ---
    mask_m = np.ones(adata.n_obs, dtype=bool)
    print("=== Stage 3: Manual per-study cutoffs ===")
    for ds in datasets:
        ds_idx = adata.obs[dataset_key] == ds
        n_ds = ds_idx.sum()
        cutoffs = per_study_cutoffs.get(ds, global_cutoffs)
        if not cutoffs:
            print(f"  {ds}: no cutoffs (stages 1-2 only)")
            continue
        ds_mask = np.ones(n_ds, dtype=bool)
        applied = []
        for key, col in _cutoff_col.items():
            if key not in cutoffs or col not in adata.obs.columns:
                continue
            vals = adata.obs.loc[ds_idx, col].values
            if key.startswith("min_"):
                ds_mask &= vals >= cutoffs[key]
                applied.append(f"{col}>={cutoffs[key]}")
            elif key.startswith("max_"):
                ds_mask &= vals <= cutoffs[key]
                applied.append(f"{col}<={cutoffs[key]}")
        mask_m[ds_idx.values] = ds_mask
        n_removed = n_ds - ds_mask.sum()
        print(f"  {ds}: {n_removed:,} removed ({n_removed / n_ds * 100:.1f}%) — {', '.join(applied)}")
    n_s3 = mask_m.sum()
    print(f"  Stage 3 total: {n_start:,} -> {n_s3:,} ({n_s3 / n_start * 100:.1f}% kept)\n")

    # --- Combine all stages ---
    mask_all = mask_q & mask_o & mask_m
    adata_filtered = adata[mask_all].copy()
    n_kept = adata_filtered.n_obs
    print(f"=== Combined: {n_start:,} -> {n_kept:,} ({n_kept / n_start * 100:.1f}% kept) ===")

    # Per-dataset summary
    print("\nPer-dataset summary:")
    for ds in datasets:
        n_before = (adata.obs[dataset_key] == ds).sum()
        n_after = (adata_filtered.obs[dataset_key] == ds).sum()
        print(f"  {ds}: {n_before:,} -> {n_after:,} ({n_after / n_before * 100:.1f}%)")

    # --- Post: drop small batches ---
    batch_counts = adata_filtered.obs[batch_key].value_counts()
    small_batches = batch_counts[batch_counts < min_cells_per_batch].index.tolist()
    if small_batches:
        n_before_drop = adata_filtered.n_obs
        adata_filtered = adata_filtered[~adata_filtered.obs[batch_key].isin(small_batches)].copy()
        n_dropped = n_before_drop - adata_filtered.n_obs
        print(
            f"\nDropped {len(small_batches)} batches with <{min_cells_per_batch} cells ({n_dropped:,} cells): {small_batches}"
        )
    else:
        print(f"\nAll batches have >= {min_cells_per_batch} cells.")

    print(f"\nFinal: {adata_filtered.n_obs:,} cells, {adata_filtered.obs[batch_key].nunique()} batches")
    return adata_filtered
