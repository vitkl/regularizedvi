"""Plotting utilities for regularizedvi covariate analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def plot_covariate_umap_grid(
    adata,
    model,
    conditions: dict,
    cov_color_keys: list[str],
    size: float = 1,
):
    """Plot UMAP grid: modalities (cols) x conditions (rows), colored by covariates.

    Creates one figure per covariate key. Rows correspond to the original attribution
    plus each condition suffix; columns correspond to modality names.

    Parameters
    ----------
    adata
        AnnData with ``X_umap_attr_{mod}{suffix}`` keys in ``.obsm``.
    model
        :class:`~regularizedvi.RegularizedMultimodalVI` model instance
        (used for ``model.module.modality_names``).
    conditions
        Dict of ``{suffix: kwargs}`` as passed to
        :meth:`~regularizedvi.RegularizedMultimodalVI.get_modality_attribution`.
    cov_color_keys
        List of ``.obs`` column names to color by.
    size
        Point size for scatter plots.
    """
    modality_names = model.module.modality_names
    all_conditions = [""] + list(conditions.keys())

    for cov_key in cov_color_keys:
        n_rows = len(all_conditions)
        n_cols = len(modality_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for i, cond in enumerate(all_conditions):
            for j, mod in enumerate(modality_names):
                ax = axes[i, j]
                umap_key = f"X_umap_attr_{mod}{cond}"
                if umap_key in adata.obsm:
                    adata.obsm["X_umap"] = adata.obsm[umap_key]
                    sc.pl.umap(
                        adata,
                        color=cov_key,
                        ax=ax,
                        show=False,
                        title=f"{mod} {cond or '(original)'}",
                        frameon=False,
                        size=size,
                    )
                else:
                    ax.set_title(f"{mod} {cond} (missing)")
                    ax.axis("off")

        fig.suptitle(f"Attribution UMAPs colored by {cov_key}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


def plot_covariate_corr_heatmaps(cov_effects: dict):
    """Plot correlation heatmaps and boxplots for covariate parameters.

    For each ``feature_scaling`` and ``additive_background`` pathway, plots a
    hierarchically-clustered correlation heatmap of covariate levels alongside
    ordered boxplots of parameter values.

    Parameters
    ----------
    cov_effects
        Dict returned by
        :meth:`~regularizedvi.RegularizedMultimodalVI.get_covariate_effects`.
    """
    for pathway_name, pathway_data in [
        ("feature_scaling", cov_effects.get("feature_scaling", {})),
        ("additive_background", cov_effects.get("additive_background", {})),
    ]:
        for mod_name, mod_data in pathway_data.items():
            for cov_i, cov_info in enumerate(mod_data.get("per_covariate", [])):
                corr = cov_info.get("correlation_matrix")
                if corr is None or corr.shape[0] < 2:
                    continue

                n_cats = cov_info["n_cats"]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(4, n_cats * 0.3)))

                # Hierarchical clustering
                dist = 1 - corr
                np.fill_diagonal(dist, 0)
                dist = np.maximum(dist, 0)
                condensed = squareform(dist, checks=False)
                Z = linkage(condensed, method="average")
                dendro = dendrogram(Z, no_plot=True)
                order = dendro["leaves"]

                # Reordered correlation heatmap
                corr_ordered = corr[np.ix_(order, order)]
                im = ax1.imshow(corr_ordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax1.set_title(f"{pathway_name} {mod_name} cov{cov_i}\nCorrelation ({n_cats} levels)")
                plt.colorbar(im, ax=ax1, shrink=0.6)

                # Boxplot of parameter values per level (ordered by clustering)
                values = cov_info["values"]
                if pathway_name == "additive_background":
                    box_data = [values[:, k] for k in order]
                else:
                    box_data = [values[k, :] for k in order]
                ax2.boxplot(box_data, vert=True)
                ax2.set_title("Parameter values per level")
                ax2.set_xlabel("Level (clustered order)")

                plt.tight_layout()
                plt.show()
