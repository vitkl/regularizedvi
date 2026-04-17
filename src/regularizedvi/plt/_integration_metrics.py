"""Integration quality metrics using sklearn (no scib dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _lisi_one(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute LISI (Local Inverse Simpson Index) for one label vector.

    For each cell, compute the effective number of label categories in its
    KNN neighborhood, measured as the inverse Simpson index of the label
    distribution weighted by a Gaussian kernel.

    Parameters
    ----------
    distances
        (n_query, k) KNN distance matrix.
    indices
        (n_query, k) KNN index matrix (indices into the KNN-fitted array).
    labels
        (n_fitted,) integer-encoded label vector for all cells in the
        KNN-fitted array.
    perplexity
        Perplexity for Gaussian kernel bandwidth.
    valid_mask
        (n_fitted,) boolean mask. If provided, only neighbors where
        ``valid_mask[neighbor_idx] == True`` contribute to the Simpson
        index (weights are re-normalized over valid neighbors).
        Use this to exclude unlabelled neighbors from cLISI.

    Returns
    -------
    (n_query,) LISI scores. Higher = more mixed labels in neighborhood.
    """
    n_cells, k = indices.shape
    lisi = np.zeros(n_cells, dtype=np.float64)

    for i in range(n_cells):
        nb_idx = indices[i]
        neighbor_labels = labels[nb_idx]
        d = distances[i]

        # Gaussian kernel weights (unnormalized)
        sigma = max(d[min(perplexity, k - 1)], 1e-10)
        weights = np.exp(-(d**2) / (2 * sigma**2))

        # Mask out invalid (unlabelled) neighbors if requested
        if valid_mask is not None:
            nb_valid = valid_mask[nb_idx]
            weights = weights * nb_valid
            if weights.sum() < 1e-10:
                lisi[i] = 1.0
                continue
            neighbor_labels = neighbor_labels[nb_valid]
            weights = weights[nb_valid]

        weights /= weights.sum()

        # Simpson index: sum of squared proportions per label
        unique_labels = np.unique(neighbor_labels)
        simpson = 0.0
        for lab in unique_labels:
            p = weights[neighbor_labels == lab].sum()
            simpson += p**2

        lisi[i] = 1.0 / max(simpson, 1e-10)

    return lisi


def compute_integration_metrics(
    adata,
    latent_key: str = "X_scVI",
    label_key: str = "level_1",
    batch_key: str = "batch",
    dataset_col: str = "dataset",
    tissue_col: str = "tissue",
    leiden_key: str = "leiden",
    subsample_n: int = 50000,
    lisi_k: int = 90,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute integration quality metrics.

    All metrics wrapped in try/except — partial results returned on failure.
    Uses sklearn (no scib dependency).

    Parameters
    ----------
    adata
        AnnData with latent representation and obs columns.
    latent_key
        Key in ``adata.obsm`` for latent representation.
    label_key
        Obs column with cell type labels for bio conservation.
    batch_key
        Obs column with batch/dataset labels for integration assessment.
    dataset_col
        Obs column with dataset identifiers.
    tissue_col
        Obs column with tissue type.
    leiden_key
        Obs column with leiden cluster assignments.
    subsample_n
        Number of cells to subsample for LISI (memory-intensive).
    lisi_k
        Number of neighbors for LISI computation.
    random_state
        Random seed for subsampling.

    Returns
    -------
    DataFrame with metric names and values.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors

    X = adata.obsm[latent_key]
    results = []

    def _add(name, value, category="global"):
        results.append({"metric": name, "value": value, "category": category})

    # Filter to cells with non-null labels (also catch string "nan") and non-null batch
    has_label = (
        adata.obs[label_key].notna() & (adata.obs[label_key] != "") & (adata.obs[label_key].astype(str) != "nan")
    )
    has_batch = adata.obs[batch_key].notna()
    has_label = has_label & has_batch
    n_labelled = has_label.sum()

    # --- Global metrics (on labelled cells) ---
    try:
        if n_labelled > 100:
            X_lab = X[has_label.values]
            labels = adata.obs.loc[has_label, label_key].values
            batches = adata.obs.loc[has_label, batch_key].values

            # Bio conservation: silhouette of cell types
            if len(np.unique(labels)) > 1:
                sil_label = silhouette_score(
                    X_lab, labels, sample_size=min(50000, len(X_lab)), random_state=random_state
                )
                _add("silhouette_label", sil_label)

            # Batch mixing: silhouette of batches (lower = better mixing)
            if len(np.unique(batches)) > 1:
                sil_batch = silhouette_score(
                    X_lab, batches, sample_size=min(50000, len(X_lab)), random_state=random_state
                )
                _add("silhouette_batch", sil_batch)
    except Exception as e:  # noqa: BLE001
        _add("silhouette_error", str(e))

    # --- ARI / NMI: leiden vs labels ---
    try:
        if leiden_key in adata.obs.columns and n_labelled > 100:
            lab_mask = has_label.values
            leiden_vals = adata.obs.loc[lab_mask, leiden_key].values
            label_vals = adata.obs.loc[lab_mask, label_key].values
            _add("ARI_leiden_vs_label", adjusted_rand_score(label_vals, leiden_vals))
            _add("NMI_leiden_vs_label", normalized_mutual_info_score(label_vals, leiden_vals))
    except Exception as e:  # noqa: BLE001
        _add("ARI_NMI_error", str(e))

    # --- LISI (subsampled, KNN on all cells) ---
    try:
        if n_labelled > 100:
            from sklearn.preprocessing import LabelEncoder

            rng = np.random.RandomState(random_state + 1)
            n_total = X.shape[0]

            # Subsample ALL cells (labelled + unlabelled) for KNN fitting
            n_knn = min(subsample_n * 2, n_total)
            if n_total > n_knn:
                all_idx = rng.choice(n_total, size=n_knn, replace=False)
            else:
                all_idx = np.arange(n_total)

            # Track which subsampled cells are labelled
            is_labelled_sub = has_label.values[all_idx]
            lab_positions = np.where(is_labelled_sub)[0]  # indices within all_idx

            # Cap labelled query cells at subsample_n
            if len(lab_positions) > subsample_n:
                lab_positions = rng.choice(lab_positions, size=subsample_n, replace=False)

            X_knn = X[all_idx]

            # Fit KNN on ALL subsampled cells (preserves latent geometry)
            nn = NearestNeighbors(n_neighbors=lisi_k, metric="euclidean")
            nn.fit(X_knn)
            # Query only labelled cells
            distances, indices = nn.kneighbors(X_knn[lab_positions])

            # iLISI: filter NaN batches from encoding
            batch_vals_knn = adata.obs[batch_key].values[all_idx]
            valid_batch_knn = pd.notna(batch_vals_knn)
            batches_knn = np.full(len(all_idx), -1, dtype=np.int64)
            if valid_batch_knn.all():
                batches_knn = LabelEncoder().fit_transform(batch_vals_knn)
            else:
                batches_knn[valid_batch_knn] = LabelEncoder().fit_transform(batch_vals_knn[valid_batch_knn])
            ilisi = _lisi_one(distances, indices, batches_knn)
            _add("iLISI_median", float(np.median(ilisi)))
            _add("iLISI_mean", float(np.mean(ilisi)))

            # cLISI: exclude unlabelled neighbors from weights
            label_vals_knn = adata.obs[label_key].values[all_idx]
            labelled_only = label_vals_knn[is_labelled_sub]
            le_label = LabelEncoder().fit(labelled_only)
            labels_int_knn = np.full(len(all_idx), -1, dtype=np.int64)
            labels_int_knn[is_labelled_sub] = le_label.transform(labelled_only)

            clisi = _lisi_one(distances, indices, labels_int_knn, valid_mask=is_labelled_sub)
            _add("cLISI_median", float(np.median(clisi)))
            _add("cLISI_mean", float(np.mean(clisi)))
            _add("LISI_subsample_n", int(len(lab_positions)))
    except Exception as e:  # noqa: BLE001
        _add("LISI_error", str(e))

    # --- Per-study silhouette (batch mixing within study) ---
    try:
        if dataset_col in adata.obs.columns:
            for ds in sorted(adata.obs[dataset_col].unique()):
                ds_mask = (adata.obs[dataset_col] == ds).values & has_label.values
                if ds_mask.sum() < 50:
                    continue
                ds_batches = adata.obs.loc[ds_mask, batch_key].values
                if len(np.unique(ds_batches)) > 1:
                    ds_labels = adata.obs.loc[ds_mask, label_key].values
                    if len(np.unique(ds_labels)) > 1:
                        sil = silhouette_score(
                            X[ds_mask],
                            ds_batches,
                            sample_size=min(10000, ds_mask.sum()),
                            random_state=random_state,
                        )
                        _add(f"silhouette_batch_{ds}", sil, category="per_study")
    except Exception as e:  # noqa: BLE001
        _add("per_study_error", str(e))

    # --- Organ integration (spleen, PBMC) ---
    try:
        if tissue_col in adata.obs.columns:
            organ_groups = {
                "spleen": ["lung_spleen_gse319044", "infant_adult_spleen"],
                "pbmc": ["pbmc_tea_seq", "crohns_pbmc", "covid_pbmc"],
            }
            for organ, datasets in organ_groups.items():
                organ_mask = adata.obs[dataset_col].isin(datasets).values & has_label.values
                if organ_mask.sum() < 50:
                    continue
                organ_datasets = adata.obs.loc[organ_mask, dataset_col].values
                organ_labels = adata.obs.loc[organ_mask, label_key].values
                if len(np.unique(organ_datasets)) > 1 and len(np.unique(organ_labels)) > 1:
                    sil = silhouette_score(
                        X[organ_mask],
                        organ_labels,
                        sample_size=min(20000, organ_mask.sum()),
                        random_state=random_state,
                    )
                    _add(f"silhouette_label_{organ}", sil, category="organ")
                    sil_batch = silhouette_score(
                        X[organ_mask],
                        organ_datasets,
                        sample_size=min(20000, organ_mask.sum()),
                        random_state=random_state,
                    )
                    _add(f"silhouette_batch_{organ}", sil_batch, category="organ")
    except Exception as e:  # noqa: BLE001
        _add("organ_error", str(e))

    df = pd.DataFrame(results)
    if len(df) > 0:
        print("\n=== Integration Metrics ===")
        for _, row in df.iterrows():
            v = row["value"]
            if isinstance(v, float):
                print(f"  {row['metric']}: {v:.4f}  [{row['category']}]")
            else:
                print(f"  {row['metric']}: {v}  [{row['category']}]")

    return df


# ---------------------------------------------------------------------------
# Integration metrics heatmap
# ---------------------------------------------------------------------------

# Metric group classification for column header coloring
_SUMMARY_METRICS = {"Total", "Bio conservation", "Batch correction"}
_BIO_METRICS = {
    "Isolated labels",
    "KMeans NMI",
    "KMeans ARI",
    "Silhouette label",
    "cLISI",
    "BRAS",
    # Alternative names from scib-metrics
    "NMI cluster/label",
    "ARI cluster/label",
    # Custom sklearn-based metrics
    "silhouette_label",
    "ARI_leiden_vs_label",
    "NMI_leiden_vs_label",
    "cLISI_median",
    "cLISI_mean",
}
_BATCH_METRICS = {
    "iLISI",
    "KBET",
    "Graph connectivity",
    "PCR comparison",
    "kBET",  # alternative capitalization
    # Custom sklearn-based metrics
    "silhouette_batch",
    "iLISI_median",
    "iLISI_mean",
}
# Prefixes for per-study/organ metrics (matched by startswith)
_BIO_PREFIXES = ("silhouette_label_",)
_BATCH_PREFIXES = ("silhouette_batch_",)

# Neighbourhood correlation headline metrics (H1-H14)
_NEIGHBOURHOOD_METRICS = {
    "corr_within_library",
    "corr_consistency",
    "corr_cross_library",
    "corr_gap_library",
    "isolation_norm_cross_library",
    "discrepancy_cross_library",
    "corr_cross_dataset",
    "corr_gap_dataset",
    "isolation_norm_cross_dataset",
    "discrepancy_cross_dataset",
    "distrib_overlap_library",
    "distrib_overlap_dataset",
    "integration_failure_rate",
    "cross_technical_correlation",
    "bio_conservation",
    "library_integration",
    "dataset_integration",
    "batch_correction",
    "total",
}
_NEIGHBOURHOOD_PREFIXES = (
    "corr_within_",
    "corr_cross_",
    "corr_gap_",
    "corr_consistency_",
    "isolation_norm_",
    "discrepancy_",
    "distrib_overlap_",
    "integration_failure_",
    "cross_technical_",
    "bio_conservation_",
    "library_integration_",
    "dataset_integration_",
    "batch_correction_",
)

# Colors for column header groups
_GROUP_COLORS = {
    "summary": "#808080",  # gray
    "bio": "#2ca02c",  # green
    "batch": "#ff7f0e",  # orange
    "neighbourhood": "#1f77b4",  # blue
    "hyperparam": "#9467bd",  # purple
}


def _classify_metric_col(col_name: str) -> str:
    """Return group name for a metric column."""
    if col_name in _SUMMARY_METRICS:
        return "summary"
    if col_name in _BIO_METRICS:
        return "bio"
    if col_name in _BATCH_METRICS:
        return "batch"
    # Prefix matching for per-study/organ metrics
    if col_name.startswith(_BIO_PREFIXES):
        return "bio"
    if col_name.startswith(_BATCH_PREFIXES):
        return "batch"
    # Neighbourhood correlation metrics (H1-H14)
    if col_name in _NEIGHBOURHOOD_METRICS:
        return "neighbourhood"
    if col_name.startswith(_NEIGHBOURHOOD_PREFIXES):
        return "neighbourhood"
    return "hyperparam"


def plot_integration_heatmap(
    scib_df: pd.DataFrame,
    experiments_df: pd.DataFrame | None = None,
    hyperparam_cols: list[str] | None = None,
    sort_by: str = "Total",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    dpi: int = 150,
) -> Figure:
    """Plot integration benchmark heatmap with metrics and hyperparameters.

    Parameters
    ----------
    scib_df
        DataFrame from ``Benchmarker.get_results()`` — rows = methods,
        columns = metrics.  Index = experiment names.
    experiments_df
        Pre-loaded DataFrame with experiment metadata (from
        ``integration_metrics_experiments.tsv``).  Joined on ``name`` column
        matching ``scib_df`` index for hyperparameter columns.
    hyperparam_cols
        Which hyperparameter columns from *experiments_df* to show.
        ``None`` → auto-detect columns with >1 unique value.
    sort_by
        Column to sort rows by (descending).  Default ``"Total"``.
    figsize
        ``(width, height)`` in inches.  ``None`` → auto-sized.
    save_path
        If given, save ``{save_path}.svg`` and ``{save_path}.png``.
    dpi
        Resolution for PNG output.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # ── 1. Clean scib_df: drop "Metric Type" row, keep numeric metric cols ──
    metric_type_mask = scib_df.index == "Metric Type"
    plot_df = scib_df[~metric_type_mask].copy()

    # Identify numeric metric columns
    metric_cols = []
    for c in plot_df.columns:
        try:
            plot_df[c] = pd.to_numeric(plot_df[c])
            metric_cols.append(c)
        except (ValueError, TypeError):
            pass

    # ── 2. Merge hyperparameters from experiments_df ──
    hp_cols_to_show = []
    if experiments_df is not None and hyperparam_cols is not None:
        exp_indexed = experiments_df.set_index("name")
        for hc in hyperparam_cols:
            if hc in exp_indexed.columns:
                # Match experiment names (scib_df index) to experiments_df
                matched = exp_indexed[hc].reindex(plot_df.index)
                plot_df[hc] = matched.values
                hp_cols_to_show.append(hc)
    elif experiments_df is not None and hyperparam_cols is None:
        # Auto-detect: columns with >1 unique value among matched experiments
        exp_indexed = experiments_df.set_index("name")
        matched_exp = exp_indexed.reindex(plot_df.index).dropna(how="all")
        for c in exp_indexed.columns:
            if c in ("name", "results_folder", "notebook", "label", "notes", "status"):
                continue
            vals = matched_exp[c].dropna().unique()
            if len(vals) > 1:
                plot_df[c] = exp_indexed[c].reindex(plot_df.index).values
                hp_cols_to_show.append(c)

    # ── 3. Sort rows ──
    if sort_by in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by, ascending=False)

    # ── 4. Define column order: summary → bio → batch → neighbourhood → hyperparams ──
    ordered_metric_cols = []
    for group_name in ["summary", "bio", "batch", "neighbourhood"]:
        for c in metric_cols:
            if _classify_metric_col(c) == group_name:
                ordered_metric_cols.append(c)
    all_display_cols = ordered_metric_cols + hp_cols_to_show
    n_metric = len(ordered_metric_cols)
    n_cols = len(all_display_cols)
    n_rows = len(plot_df)

    if n_cols == 0 or n_rows == 0:
        raise ValueError("No data to plot")

    # ── 5. Build figure ──
    if figsize is None:
        figsize = (max(10, n_cols * 1.1 + 3), max(4, n_rows * 0.45 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    # Allocate data matrix
    data = np.full((n_rows, n_cols), np.nan)

    # Fill metric columns (numeric)
    for j, c in enumerate(ordered_metric_cols):
        data[:, j] = plot_df[c].values.astype(float)

    # ── 6. Per-column color normalization for metrics ──
    col_norms = []
    for j in range(n_metric):
        col_vals = data[:, j]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmin == vmax:
                vmin, vmax = vmin - 0.01, vmax + 0.01
        else:
            vmin, vmax = 0, 1
        col_norms.append(Normalize(vmin=vmin, vmax=vmax))

    # ── 7. Build categorical colormaps for hyperparams ──
    hp_cmap = matplotlib.colormaps.get_cmap("tab10")
    hp_cat_maps = {}  # col_name -> {value: color_idx}
    for j, hc in enumerate(hp_cols_to_show):
        vals = plot_df[hc].fillna("—").astype(str).values
        unique_vals = sorted(set(vals))
        hp_cat_maps[hc] = {v: i for i, v in enumerate(unique_vals)}
        # Fill data matrix with category indices
        for i, v in enumerate(vals):
            data[i, n_metric + j] = hp_cat_maps[hc][v]

    # ── 8. Draw cells ──
    # Metric cells
    metric_cmap = matplotlib.colormaps.get_cmap("RdYlGn")
    for i in range(n_rows):
        for j in range(n_metric):
            val = data[i, j]
            if np.isnan(val):
                color = "white"
            else:
                color = metric_cmap(col_norms[j](val))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            if not np.isnan(val):
                text_color = "white" if col_norms[j](val) < 0.3 or col_norms[j](val) > 0.7 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color,
                    fontweight="bold",
                )

    # Hyperparameter cells
    for i in range(n_rows):
        for jj, hc in enumerate(hp_cols_to_show):
            j = n_metric + jj
            cat_idx = data[i, j]
            if np.isnan(cat_idx):
                color = "white"
            else:
                n_cats = len(hp_cat_maps[hc])
                color = hp_cmap(int(cat_idx) % 10) if n_cats > 1 else (0.9, 0.9, 0.9, 1.0)
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            # Print value text
            val_str = plot_df[hc].fillna("—").astype(str).values[i]
            ax.text(
                j + 0.5,
                i + 0.5,
                val_str,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if n_cats > 1 else "black",
                fontweight="bold",
            )

    # ── 9. Column header color bar ──
    for j, c in enumerate(all_display_cols):
        group = _classify_metric_col(c) if j < n_metric else "hyperparam"
        rect = plt.Rectangle(
            (j, -0.6), 1, 0.5, facecolor=_GROUP_COLORS[group], edgecolor="white", linewidth=0.5, clip_on=False
        )
        ax.add_patch(rect)

    # ── 10. Axes formatting ──
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, -0.6)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(all_display_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.set_frame_on(False)

    # Group labels at top
    group_positions = {"summary": [], "bio": [], "batch": [], "neighbourhood": [], "hyperparam": []}
    for j, c in enumerate(all_display_cols):
        g = _classify_metric_col(c) if j < n_metric else "hyperparam"
        group_positions[g].append(j)
    group_labels = {
        "summary": "Summary",
        "bio": "Bio conservation",
        "batch": "Batch correction",
        "neighbourhood": "Neighbourhood corr.",
        "hyperparam": "Hyperparameters",
    }
    for g, positions in group_positions.items():
        if positions:
            mid = (positions[0] + positions[-1]) / 2 + 0.5
            ax.text(
                mid,
                -1.0,
                group_labels[g],
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=_GROUP_COLORS[g],
                clip_on=False,
            )

    ax.set_title("Integration Benchmark", fontsize=12, pad=30)
    plt.tight_layout()

    # ── 11. Save ──
    if save_path is not None:
        for ext in ["svg", "png"]:
            fig.savefig(f"{save_path}.{ext}", dpi=dpi, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}.{{svg,png}}")

    return fig
