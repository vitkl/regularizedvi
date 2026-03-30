"""Integration quality metrics using sklearn (no scib dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    batch_key: str = "dataset",
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

    # Filter to cells with non-null labels (also catch string "nan")
    has_label = (
        adata.obs[label_key].notna() & (adata.obs[label_key] != "") & (adata.obs[label_key].astype(str) != "nan")
    )
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

            # iLISI: all neighbors have batch labels — no masking needed
            batches_knn = LabelEncoder().fit_transform(adata.obs[batch_key].values[all_idx])
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
                ds_batches = adata.obs.loc[ds_mask, "batch"].values
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
