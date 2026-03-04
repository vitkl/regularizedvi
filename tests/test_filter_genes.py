"""Tests for filter_genes: batched implementation vs naive."""

import numpy as np
import pytest
from scipy import sparse


@pytest.fixture()
def adata_for_filter():
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells, n_genes = 500, 200
    # Sparse counts with ~10% nonzero
    X = sparse.random(n_cells, n_genes, density=0.1, format="csr", random_state=rng)
    X.data = rng.integers(1, 100, size=X.data.shape).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    return ad.AnnData(X=X, obs={"_": obs_names}, var={"_": var_names})


def test_gene_stats_match_naive(adata_for_filter):
    """Batched _compute_gene_stats matches X>0 sum and X.sum."""
    from regularizedvi.utils._filtering import _compute_gene_stats

    X = adata_for_filter.X
    n_cells_batched, gene_sum_batched = _compute_gene_stats(X, batch_size=100)
    n_cells_naive = np.array((X > 0).sum(0)).flatten()
    gene_sum_naive = np.array(X.sum(0)).flatten()

    np.testing.assert_array_equal(n_cells_batched, n_cells_naive)
    np.testing.assert_allclose(gene_sum_batched, gene_sum_naive)


def test_filter_genes_matches_naive(adata_for_filter):
    """Efficient filter_genes returns same gene selection as naive."""
    import matplotlib

    matplotlib.use("Agg")
    from regularizedvi.utils._filtering import _filter_genes_naive, filter_genes

    cutoffs = {"cell_count_cutoff": 5, "cell_percentage_cutoff2": 0.03, "nonz_mean_cutoff": 1.05}

    naive_result = _filter_genes_naive(adata_for_filter.copy(), **cutoffs)
    efficient_result = filter_genes(adata_for_filter.copy(), **cutoffs)

    assert list(naive_result) == list(efficient_result)


def test_gene_stats_dense(adata_for_filter):
    """Batched stats work on dense arrays too."""
    from regularizedvi.utils._filtering import _compute_gene_stats

    X_dense = adata_for_filter.X.toarray()
    n_cells, gene_sum = _compute_gene_stats(X_dense, batch_size=100)
    n_cells_expected = np.count_nonzero(X_dense, axis=0)
    gene_sum_expected = X_dense.sum(axis=0)

    np.testing.assert_array_equal(n_cells, n_cells_expected)
    np.testing.assert_allclose(gene_sum, gene_sum_expected)
