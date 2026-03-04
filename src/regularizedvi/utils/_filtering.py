"""Gene filtering utilities.

Copied from cell2location (BayraktarLab/cell2location) to avoid a dependency.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
