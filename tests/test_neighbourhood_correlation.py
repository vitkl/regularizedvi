"""Tests for neighbourhood correlation metrics module."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adata_with_covariates():
    """Small AnnData with library/dataset/technical obs columns + KNN graph."""
    import anndata as ad

    rng = np.random.default_rng(42)
    n_cells, n_genes = 200, 80
    X = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=rng)
    X.data = rng.integers(1, 50, size=X.data.shape).astype(np.float32)

    obs = {
        "library": rng.choice(["lib_A", "lib_B", "lib_C", "lib_D"], size=n_cells),
        "dataset": np.where(
            np.isin(
                rng.choice(["lib_A", "lib_B", "lib_C", "lib_D"], size=n_cells),
                ["lib_A", "lib_B"],
            ),
            "ds1",
            "ds2",
        ),
        "technical": rng.choice(["tech_X", "tech_Y"], size=n_cells),
    }
    # Ensure library nests within dataset
    obs["dataset"] = np.where(np.isin(obs["library"], ["lib_A", "lib_B"]), "ds1", "ds2")

    adata = ad.AnnData(
        X=X,
        obs=obs,
        var={"SYMBOL": [f"gene_{i}" for i in range(n_genes)]},
    )

    # Build a small KNN graph (k=10)
    from sklearn.neighbors import NearestNeighbors

    X_dense = np.asarray(X.todense())
    nn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    nn.fit(X_dense)
    distances, indices = nn.kneighbors(X_dense)

    rows = np.repeat(np.arange(n_cells), 10)
    cols = indices.flatten()
    vals = np.ones(len(rows), dtype=np.float32)
    conn = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
    # Symmetrise
    conn = conn.maximum(conn.T)
    conn.setdiag(0)
    conn.eliminate_zeros()

    adata.obsp["connectivities"] = conn
    return adata


# ---------------------------------------------------------------------------
# Sub-plan 01: normalise_counts
# ---------------------------------------------------------------------------


class TestNormaliseCounts:
    def test_sparse_preserves_sparsity(self):
        X = sparse.random(100, 50, density=0.3, format="csr", dtype=np.float32)
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X_norm = normalise_counts(X)
        assert sparse.issparse(X_norm)
        assert isinstance(X_norm, sparse.csr_matrix)

    def test_row_sums_equal_n_vars(self):
        rng = np.random.default_rng(0)
        X = sparse.random(100, 50, density=0.3, format="csr", random_state=rng)
        X.data = rng.integers(1, 100, size=X.data.shape).astype(np.float32)
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X_norm = normalise_counts(X)
        sums = np.asarray(X_norm.sum(axis=1)).flatten()
        nonzero = np.asarray(X.sum(axis=1)).flatten() > 0
        np.testing.assert_allclose(sums[nonzero], 50.0, rtol=1e-5)

    def test_zero_count_row(self):
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X = sparse.lil_matrix((5, 10), dtype=np.float32)
        X[1, 3] = 5.0
        X = X.tocsr()
        X_norm = normalise_counts(X)
        assert np.asarray(X_norm[0].todense()).sum() == 0
        assert np.asarray(X_norm[1].todense()).sum() > 0

    def test_n_vars_override(self):
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X = sparse.random(50, 30, density=0.5, format="csr", dtype=np.float32)
        X.data = np.ones_like(X.data)
        X_norm = normalise_counts(X, n_vars=20000)
        sums = np.asarray(X_norm.sum(axis=1)).flatten()
        nonzero = np.asarray(X.sum(axis=1)).flatten() > 0
        np.testing.assert_allclose(sums[nonzero], 20000.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sub-plan 03: mask construction
# ---------------------------------------------------------------------------


class TestValidateCovariateHierarchy:
    def test_valid_hierarchy_passes(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            validate_covariate_hierarchy,
        )

        validate_covariate_hierarchy(adata_with_covariates, library_key="library", dataset_key="dataset")

    def test_invalid_hierarchy_raises(self):
        import anndata as ad

        obs = {
            "library": ["A", "A", "B", "B"],
            "dataset": ["ds1", "ds2", "ds1", "ds1"],
        }
        adata = ad.AnnData(
            X=sparse.eye(4, format="csr"),
            obs=obs,
        )
        from regularizedvi.plt._neighbourhood_correlation import (
            validate_covariate_hierarchy,
        )

        with pytest.raises(ValueError, match="hierarchy violation"):
            validate_covariate_hierarchy(adata, library_key="library", dataset_key="dataset")

    def test_no_dataset_passes_silently(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            validate_covariate_hierarchy,
        )

        validate_covariate_hierarchy(adata_with_covariates, library_key="library", dataset_key=None)


class TestConstructNeighbourMasks:
    def test_library_only_produces_two_masks(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
        )
        assert "same_library" in masks
        assert "between_libraries" in masks
        assert len(masks) == 2

    def test_with_dataset_produces_four_masks(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
            dataset_key="dataset",
        )
        assert "same_library" in masks
        assert "between_libraries" in masks
        assert "cross_library" in masks
        assert "cross_dataset" in masks
        assert len(masks) == 4

    def test_with_technical_adds_two_per_covariate(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
            dataset_key="dataset",
            technical_covariate_keys=["technical"],
        )
        assert "within_technical" in masks
        assert "between_technical" in masks
        # H14: a union "cross_technical" mask is always added when any technical key is given
        assert "cross_technical" in masks
        assert len(masks) == 7

    def test_masks_partition_connectivity(self, adata_with_covariates):
        """same_library + cross_library + cross_dataset == all non-self edges."""
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        conn = adata_with_covariates.obsp["connectivities"].copy()
        conn.setdiag(0)
        conn.eliminate_zeros()
        total_nnz = conn.nnz

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
            dataset_key="dataset",
        )
        sl_nnz = masks["same_library"].nnz
        xl_nnz = masks["cross_library"].nnz
        xd_nnz = masks["cross_dataset"].nnz
        assert sl_nnz + xl_nnz + xd_nnz == total_nnz, f"Partition failed: {sl_nnz} + {xl_nnz} + {xd_nnz} != {total_nnz}"

    def test_between_libraries_equals_cross_library_plus_cross_dataset(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
            dataset_key="dataset",
        )
        bl_nnz = masks["between_libraries"].nnz
        xl_nnz = masks["cross_library"].nnz
        xd_nnz = masks["cross_dataset"].nnz
        assert bl_nnz == xl_nnz + xd_nnz

    def test_masks_preserve_connectivity_weights(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        conn = adata_with_covariates.obsp["connectivities"]
        masks = construct_neighbour_masks(
            adata_with_covariates,
            conn,
            library_key="library",
        )
        sl = masks["same_library"]
        # Non-zero entries in same_library should have the same values as in conn
        sl_coo = sl.tocoo()
        for i, j, v in zip(sl_coo.row, sl_coo.col, sl_coo.data, strict=False):
            assert v == conn[i, j], f"Weight mismatch at ({i},{j})"

    def test_no_self_loops(self, adata_with_covariates):
        from regularizedvi.plt._neighbourhood_correlation import (
            construct_neighbour_masks,
        )

        masks = construct_neighbour_masks(
            adata_with_covariates,
            adata_with_covariates.obsp["connectivities"],
            library_key="library",
            dataset_key="dataset",
        )
        for name, m in masks.items():
            diag = m.diagonal()
            assert np.all(diag == 0), f"Self-loop found in mask '{name}'"


class TestListActiveMasks:
    def test_library_only(self):
        from regularizedvi.plt._neighbourhood_correlation import list_active_masks

        masks = list_active_masks(library_key="batch")
        assert masks == ["same_library", "between_libraries"]

    def test_with_dataset(self):
        from regularizedvi.plt._neighbourhood_correlation import list_active_masks

        masks = list_active_masks(library_key="batch", dataset_key="dataset")
        assert "same_library" in masks
        assert "between_libraries" in masks
        assert "cross_library" in masks
        assert "cross_dataset" in masks

    def test_with_technical(self):
        from regularizedvi.plt._neighbourhood_correlation import list_active_masks

        masks = list_active_masks(
            library_key="batch",
            dataset_key="dataset",
            technical_covariate_keys=["tissue", "experiment"],
        )
        assert "within_tissue" in masks
        assert "between_tissue" in masks
        assert "within_experiment" in masks
        assert "between_experiment" in masks


# ---------------------------------------------------------------------------
# Sub-plan 04: Sparse Pearson + correlation computation
# ---------------------------------------------------------------------------


class TestSparsePearsonRowStats:
    def test_matches_dense(self):
        """Row stats from sparse match numpy dense computation."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _sparse_pearson_row_stats,
        )

        rng = np.random.default_rng(42)
        X_dense = rng.poisson(2, size=(50, 30)).astype(np.float32)
        X_dense[rng.random(X_dense.shape) < 0.3] = 0
        X_sparse = sparse.csr_matrix(X_dense)

        mean_s, std_s = _sparse_pearson_row_stats(X_sparse)

        mean_d = X_dense.mean(axis=1)
        std_d = X_dense.std(axis=1)

        np.testing.assert_allclose(mean_s, mean_d, rtol=1e-5)
        np.testing.assert_allclose(std_s, std_d, rtol=1e-5, atol=1e-6)

    def test_zero_row(self):
        """All-zero row should give mean=0, std clipped to 1e-12."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _sparse_pearson_row_stats,
        )

        X = sparse.lil_matrix((3, 10), dtype=np.float32)
        X[1, 3] = 5.0
        X = X.tocsr()

        mean_x, std_x = _sparse_pearson_row_stats(X)
        assert mean_x[0] == 0.0
        assert std_x[0] > 0  # clipped, not zero


class TestApproachB:
    def test_matches_np_corrcoef(self, adata_with_covariates):
        """Approach B sparse Pearson matches np.corrcoef for each cell."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _approach_B_per_mask,
            _sparse_pearson_row_stats,
            normalise_counts,
        )

        adata = adata_with_covariates
        X = adata.X.tocsr().astype(np.float32)
        X_norm = normalise_counts(X)

        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()

        mean_x, std_x = _sparse_pearson_row_stats(X_norm)
        corr_avg = _approach_B_per_mask(X_norm, conn, mean_x, std_x)

        # Manually compute for a sample of cells using dense np.corrcoef
        X_dense = np.asarray(X_norm.todense())
        n_check = min(30, adata.n_obs)

        for cell_i in range(n_check):
            row = conn[cell_i]
            neigh_idx = row.indices
            neigh_weights = row.data

            if len(neigh_idx) == 0:
                assert np.isnan(corr_avg[cell_i])
                continue

            # Weighted average of neighbour profiles
            neigh_profiles = X_dense[neigh_idx]
            avg_profile = np.average(neigh_profiles, axis=0, weights=neigh_weights)

            # Check for zero variance
            cell_std = np.std(X_dense[cell_i])
            avg_std = np.std(avg_profile)
            if cell_std < 1e-6 or avg_std < 1e-6:
                assert np.isnan(corr_avg[cell_i]), f"Cell {cell_i}: zero-variance should give NaN"
                continue

            expected = np.corrcoef(X_dense[cell_i], avg_profile)[0, 1]
            np.testing.assert_allclose(
                corr_avg[cell_i],
                expected,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Cell {cell_i} mismatch",
            )

    def test_sparsity_preserved(self, adata_with_covariates):
        """Intermediate matrices stay sparse."""
        from regularizedvi.plt._neighbourhood_correlation import (
            normalise_counts,
        )

        X = adata_with_covariates.X.tocsr().astype(np.float32)
        X_norm = normalise_counts(X)
        assert sparse.issparse(X_norm)

        conn = adata_with_covariates.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()

        # Check that mask @ X stays sparse
        weighted_sum = conn @ X_norm
        assert sparse.issparse(weighted_sum)


class TestApproachA:
    def test_matches_np_corrcoef(self, adata_with_covariates):
        """Per-neighbour correlations match np.corrcoef."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _approach_A_per_mask,
            _sparse_pearson_row_stats,
            normalise_counts,
        )

        adata = adata_with_covariates
        X = adata.X.tocsr().astype(np.float32)
        X_norm = normalise_counts(X)

        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()

        mean_x, std_x = _sparse_pearson_row_stats(X_norm)
        agg = _approach_A_per_mask(X_norm, conn, mean_x, std_x, batch_size=50)

        X_dense = np.asarray(X_norm.todense())
        n_check = min(20, adata.n_obs)

        for cell_i in range(n_check):
            row = conn[cell_i]
            neigh_idx = row.indices

            if len(neigh_idx) == 0:
                assert np.isnan(agg["mean"][cell_i])
                continue

            # Compute per-neighbour correlations manually
            r_manual = []
            for j in neigh_idx:
                cell_std = np.std(X_dense[cell_i])
                neigh_std = np.std(X_dense[j])
                if cell_std < 1e-6 or neigh_std < 1e-6:
                    r_manual.append(np.nan)
                else:
                    r_manual.append(np.corrcoef(X_dense[cell_i], X_dense[j])[0, 1])

            r_manual = np.array(r_manual)
            valid = ~np.isnan(r_manual)
            if not np.any(valid):
                assert np.isnan(agg["mean"][cell_i])
                continue

            expected_mean = np.mean(r_manual[valid])
            np.testing.assert_allclose(
                agg["mean"][cell_i],
                expected_mean,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Cell {cell_i} mean mismatch",
            )

            expected_median = np.median(r_manual[valid])
            np.testing.assert_allclose(
                agg["median"][cell_i],
                expected_median,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"Cell {cell_i} median mismatch",
            )

    def test_zero_variance_gives_nan(self):
        """Constant row should produce NaN correlations."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _approach_A_per_mask,
            _sparse_pearson_row_stats,
        )

        # Cell 0: constant (5.0 everywhere), Cell 1: variable
        X = sparse.lil_matrix((3, 10), dtype=np.float32)
        for j in range(10):
            X[0, j] = 5.0  # constant row
        X[1, 0] = 1.0
        X[1, 5] = 10.0
        X[2, 3] = 3.0
        X[2, 7] = 7.0
        X = X.tocsr()

        # Connect cell 0 -> cell 1, cell 1 -> cell 2
        conn = sparse.csr_matrix(([1.0, 1.0], ([0, 1], [1, 2])), shape=(3, 3))

        mean_x, std_x = _sparse_pearson_row_stats(X)
        agg = _approach_A_per_mask(X, conn, mean_x, std_x, batch_size=10)

        # Cell 0 has constant expression -> NaN correlation
        assert np.isnan(agg["mean"][0]), "Constant row should give NaN"
        # Cell 2 has no neighbours -> NaN
        assert np.isnan(agg["mean"][2]), "No neighbours should give NaN"

    def test_zero_neighbours_gives_nan(self):
        """Cell with no neighbours in mask should get NaN."""
        from regularizedvi.plt._neighbourhood_correlation import (
            _approach_A_per_mask,
            _sparse_pearson_row_stats,
        )

        rng = np.random.default_rng(99)
        X = sparse.random(5, 10, density=0.5, format="csr", random_state=rng)
        X.data = rng.integers(1, 20, size=X.data.shape).astype(np.float32)

        # Empty mask
        mask = sparse.csr_matrix((5, 5), dtype=np.float32)

        mean_x, std_x = _sparse_pearson_row_stats(X)
        agg = _approach_A_per_mask(X, mask, mean_x, std_x, batch_size=10)

        assert np.all(np.isnan(agg["mean"]))
        assert np.all(np.isnan(agg["median"]))


class TestWeightedMedian:
    def test_uniform_weights(self):
        from regularizedvi.plt._neighbourhood_correlation import weighted_median

        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        wts = np.ones(5)
        result = weighted_median(vals, wts)
        assert result == 3.0

    def test_skewed_weights(self):
        from regularizedvi.plt._neighbourhood_correlation import weighted_median

        vals = np.array([1.0, 10.0])
        wts = np.array([9.0, 1.0])
        result = weighted_median(vals, wts)
        assert result == 1.0  # most weight on 1.0

    def test_empty(self):
        from regularizedvi.plt._neighbourhood_correlation import weighted_median

        result = weighted_median(np.array([]), np.array([]))
        assert np.isnan(result)


class TestComputeMarkerCorrelation:
    def test_basic_output_shape(self, adata_with_covariates):
        """Output DataFrame has expected shape and index."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == adata.n_obs
        assert df.index.equals(adata.obs_names)

    def test_expected_columns(self, adata_with_covariates):
        """All expected columns are present."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        # Mask-independent columns
        assert "n_neighbours_total" in df.columns
        assert "marker_gene_total_expression" in df.columns
        assert "corr_avg_all_neighbours" in df.columns
        assert "corr_weighted_mean_all_neighbours" in df.columns

        # Per-mask columns
        for mask_name in [
            "same_library",
            "between_libraries",
            "cross_library",
            "cross_dataset",
        ]:
            assert f"n_neighbours_{mask_name}" in df.columns
            assert f"frac_neighbours_{mask_name}" in df.columns
            assert f"corr_avg_{mask_name}" in df.columns
            assert f"corr_mean_{mask_name}" in df.columns
            assert f"corr_median_{mask_name}" in df.columns
            assert f"corr_weighted_mean_{mask_name}" in df.columns
            assert f"corr_weighted_median_{mask_name}" in df.columns
            assert f"corr_std_{mask_name}" in df.columns
            assert f"corr_cv_{mask_name}" in df.columns
            assert f"corr_discrepancy_{mask_name}" in df.columns
            assert f"corr_norm_by_library_{mask_name}" in df.columns
            assert f"corr_norm_by_all_{mask_name}" in df.columns

    def test_norm_by_library_same_library_is_one(self, adata_with_covariates):
        """corr_norm_by_library_same_library should be ~1.0 by construction."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        col = "corr_norm_by_library_same_library"
        valid = ~np.isnan(df[col].values)
        if valid.any():
            np.testing.assert_allclose(df[col].values[valid], 1.0, rtol=1e-10)

    def test_discrepancy_definition(self, adata_with_covariates):
        """corr_discrepancy = corr_avg - corr_mean."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        for mask_name in ["same_library", "between_libraries"]:
            disc = df[f"corr_discrepancy_{mask_name}"].values
            expected = df[f"corr_avg_{mask_name}"].values - df[f"corr_mean_{mask_name}"].values
            valid = ~np.isnan(disc) & ~np.isnan(expected)
            if valid.any():
                np.testing.assert_allclose(disc[valid], expected[valid], rtol=1e-10)

    def test_frac_neighbours_sums(self, adata_with_covariates):
        """Fraction neighbours across partition masks sums to ~1."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        # same_library + cross_library + cross_dataset should sum to 1
        frac_sum = (
            df["frac_neighbours_same_library"].values
            + df["frac_neighbours_cross_library"].values
            + df["frac_neighbours_cross_dataset"].values
        )
        valid = ~np.isnan(frac_sum)
        if valid.any():
            np.testing.assert_allclose(frac_sum[valid], 1.0, rtol=1e-10)

    def test_no_marker_genes_raises(self, adata_with_covariates):
        """Empty marker gene list should raise ValueError."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        with pytest.raises(ValueError, match="No marker genes"):
            compute_marker_correlation(
                adata_with_covariates,
                adata_with_covariates.obsp["connectivities"],
                ["nonexistent_gene_xyz"],
                library_key="library",
            )

    def test_library_only_no_dataset(self, adata_with_covariates):
        """Works with library_key only (no dataset_key)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        assert "corr_avg_same_library" in df.columns
        assert "corr_avg_between_libraries" in df.columns
        # Should NOT have cross_library/cross_dataset
        assert "corr_avg_cross_library" not in df.columns
        assert "corr_avg_cross_dataset" not in df.columns

    def test_with_technical_covariates(self, adata_with_covariates):
        """Technical covariate columns appear in output."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
            technical_covariate_keys=["technical"],
        )

        assert "corr_avg_within_technical" in df.columns
        assert "corr_avg_between_technical" in df.columns
        assert "n_neighbours_within_technical" in df.columns

    def test_correlation_values_in_range(self, adata_with_covariates):
        """Correlation values should be in [-1, 1] (where not NaN)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        corr_cols = [
            c
            for c in df.columns
            if c.startswith("corr_avg_") or c.startswith("corr_mean_") or c.startswith("corr_median_")
        ]
        for col in corr_cols:
            vals = df[col].dropna().values
            if len(vals) > 0:
                assert np.all(vals >= -1.0 - 1e-6), f"{col} has values < -1"
                assert np.all(vals <= 1.0 + 1e-6), f"{col} has values > 1"


# ---------------------------------------------------------------------------
# Sub-plan 05: Neighbourhood diagnostics + random KNN baseline
# ---------------------------------------------------------------------------


class TestComputeNeighbourhoodDiagnostics:
    def test_degree_matches_nnz(self, adata_with_covariates):
        """Degree series matches connectivities.getnnz(axis=1)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_neighbourhood_diagnostics,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        diag = compute_neighbourhood_diagnostics(adata, conn, library_key="library", dataset_key="dataset")

        # conn might have self-loops; diagnostics removes them
        conn_clean = conn.copy().tocsr()
        conn_clean.setdiag(0)
        conn_clean.eliminate_zeros()
        expected_degree_clean = np.asarray(conn_clean.getnnz(axis=1)).flatten()

        np.testing.assert_array_equal(diag["degree"].values, expected_degree_clean)

    def test_composition_sums_to_one(self, adata_with_covariates):
        """Per-cell composition fractions sum to 1 for each level."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_neighbourhood_diagnostics,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        diag = compute_neighbourhood_diagnostics(
            adata,
            conn,
            library_key="library",
            dataset_key="dataset",
            technical_covariate_keys=["technical"],
        )

        for key in ["composition_library", "composition_dataset", "composition_technical_technical"]:
            comp_df = diag[key]
            row_sums = comp_df.values.sum(axis=1)
            # Only check cells with neighbours
            has_neighbours = diag["degree"].values > 0
            np.testing.assert_allclose(
                row_sums[has_neighbours],
                1.0,
                rtol=1e-5,
                err_msg=f"Composition rows don't sum to 1 for {key}",
            )

    def test_penetration_threshold_zero(self, adata_with_covariates):
        """At threshold 0, fraction should be 1 (every cell trivially meets it)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_neighbourhood_diagnostics,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        diag = compute_neighbourhood_diagnostics(
            adata,
            conn,
            library_key="library",
            dataset_key="dataset",
            penetration_thresholds=(0, 5),
        )

        # At threshold 0, every cell has >= 0 cross-level neighbours
        pen_key = "penetration_between_libraries"
        pen_df = diag[pen_key]
        np.testing.assert_allclose(
            pen_df["threshold_0"].values,
            1.0,
            rtol=1e-10,
            err_msg="At threshold 0, all cells should have penetration fraction 1",
        )

    def test_high_degree_obs_means_has_expected_rows(self, adata_with_covariates):
        """High-degree comparison table has expected structure."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_neighbourhood_diagnostics,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        diag = compute_neighbourhood_diagnostics(
            adata,
            conn,
            library_key="library",
            k_reference=5,
            high_degree_multiplier=1.0,
        )

        hd_df = diag["high_degree_obs_means"]
        assert "high_degree_mean" in hd_df.columns
        assert "normal_mean" in hd_df.columns
        assert "diff" in hd_df.columns

    def test_composition_keys_present(self, adata_with_covariates):
        """All expected composition keys are in the diagnostics dict."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_neighbourhood_diagnostics,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        diag = compute_neighbourhood_diagnostics(
            adata,
            conn,
            library_key="library",
            dataset_key="dataset",
            technical_covariate_keys=["technical"],
        )

        assert "composition_library" in diag
        assert "composition_dataset" in diag
        assert "composition_technical_technical" in diag


class TestComputeRandomKnnBaseline:
    def test_random_baseline_output_shape(self, adata_with_covariates):
        """Random baseline returns DataFrame with expected shape."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_random_knn_baseline,
            normalise_counts,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()

        X_norm = normalise_counts(adata.X.tocsr().astype(np.float32))
        degree = np.asarray(conn.getnnz(axis=1)).flatten()

        df = compute_random_knn_baseline(
            adata,
            X_norm,
            degree,
            library_key="library",
            dataset_key="dataset",
            n_random_graphs=1,
            random_state=42,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == adata.n_obs
        assert "corr_avg_random_all_neighbours" in df.columns
        assert "corr_avg_random_same_library" in df.columns

    def test_random_baseline_lower_than_model(self, adata_with_covariates):
        """For a well-structured KNN, random correlation is lower than model."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            compute_random_knn_baseline,
            normalise_counts,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()

        marker_genes = adata.var_names[:20]

        # Model correlation
        model_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        # Random baseline
        marker_idx = adata.var_names.isin(marker_genes)
        X_norm = normalise_counts(
            adata[:, marker_idx].X.tocsr().astype(np.float32),
            n_vars=adata.n_vars,
        )
        degree = np.asarray(conn.getnnz(axis=1)).flatten()

        random_df = compute_random_knn_baseline(
            adata,
            X_norm,
            degree,
            library_key="library",
            n_random_graphs=2,
            random_state=0,
        )

        # Mean random correlation should be lower than mean model correlation
        model_mean = np.nanmean(model_df["corr_avg_same_library"].values)
        random_mean = np.nanmean(random_df["corr_avg_random_same_library"].values)

        assert random_mean < model_mean, (
            f"Random baseline ({random_mean:.4f}) should be lower than model correlation ({model_mean:.4f})"
        )


class TestComputeAnalyticalIsolationBaseline:
    def test_analytical_isolation_range(self, adata_with_covariates):
        """Isolation probabilities are in [0, 1]."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_analytical_isolation_baseline,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()
        degree = np.asarray(conn.getnnz(axis=1)).flatten()

        p_iso = compute_analytical_isolation_baseline(adata, degree, covariate_key="library")

        assert np.all(p_iso.values >= 0.0)
        assert np.all(p_iso.values <= 1.0)

    def test_analytical_isolation_vs_empirical_random(self, adata_with_covariates):
        """Analytical isolation matches empirical random within Monte Carlo error."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_analytical_isolation_baseline,
            construct_neighbour_masks,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"].copy().tocsr()
        conn.setdiag(0)
        conn.eliminate_zeros()
        degree = np.asarray(conn.getnnz(axis=1)).flatten()
        n_cells = adata.n_obs

        # Analytical isolation for dataset
        p_analytical = compute_analytical_isolation_baseline(adata, degree, covariate_key="dataset")

        # Empirical: generate many random graphs and count isolation fraction
        n_random = 200
        rng = np.random.default_rng(42)
        isolation_counts = np.zeros(n_cells, dtype=np.float64)
        max_k = int(degree.max())

        if max_k == 0:
            return  # skip if no neighbours

        for _ in range(n_random):
            samples = rng.integers(0, n_cells, size=(n_cells, max_k))
            self_mask = samples == np.arange(n_cells)[:, None]
            while self_mask.any():
                samples[self_mask] = rng.integers(0, n_cells, size=int(self_mask.sum()))
                self_mask = samples == np.arange(n_cells)[:, None]

            # Build random connectivity
            indptr = np.zeros(n_cells + 1, dtype=np.int64)
            indptr[1:] = np.cumsum(degree)
            total_nnz = int(indptr[-1])
            indices = np.empty(total_nnz, dtype=np.int64)
            for i in range(n_cells):
                start = indptr[i]
                end = indptr[i + 1]
                indices[start:end] = samples[i, : degree[i]]
            data = np.ones(total_nnz, dtype=np.float32)
            random_conn = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_cells))

            # Check which cells have 0 cross-dataset neighbours
            random_masks = construct_neighbour_masks(adata, random_conn, library_key="library", dataset_key="dataset")
            n_cross_ds = np.asarray(random_masks["cross_dataset"].getnnz(axis=1)).flatten()
            isolation_counts += (n_cross_ds == 0).astype(np.float64)

        empirical_isolation = isolation_counts / n_random

        # Compare: allow Monte Carlo noise (tolerance scales with 1/sqrt(n_random))
        # Use a generous tolerance for the small dataset
        np.testing.assert_allclose(
            p_analytical.values,
            empirical_isolation,
            atol=0.15,
            err_msg="Analytical isolation diverges from empirical random baseline",
        )

    def test_zero_degree_gives_one(self):
        """Cell with zero degree has isolation probability 1."""
        import anndata as ad

        from regularizedvi.plt._neighbourhood_correlation import (
            compute_analytical_isolation_baseline,
        )

        adata = ad.AnnData(
            X=sparse.eye(5, format="csr"),
            obs={"group": ["A", "A", "B", "B", "B"]},
        )
        degree = np.array([0, 5, 5, 5, 5])

        p_iso = compute_analytical_isolation_baseline(adata, degree, covariate_key="group")

        # Cell 0 has degree 0 -> (anything)^0 = 1
        assert p_iso.iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Sub-plan 06: Decision tree failure mode classification
# ---------------------------------------------------------------------------


def _make_metrics_df(
    n_cells=100,
    has_cross_library=True,
    has_cross_dataset=True,
    rng_seed=42,
):
    """Build a synthetic metrics_df for classify_failure_modes tests."""
    rng = np.random.default_rng(rng_seed)
    idx = pd.Index([f"cell_{i}" for i in range(n_cells)])

    data = {
        "n_neighbours_same_library": rng.integers(0, 20, size=n_cells),
        "corr_avg_same_library": rng.uniform(-0.2, 1.0, size=n_cells),
        "corr_std_same_library": rng.uniform(0.0, 0.5, size=n_cells),
    }

    if has_cross_library:
        data["n_neighbours_cross_library"] = rng.integers(0, 15, size=n_cells)
        data["corr_avg_cross_library"] = rng.uniform(-0.2, 1.0, size=n_cells)
        data["corr_std_cross_library"] = rng.uniform(0.0, 0.5, size=n_cells)

    if has_cross_dataset:
        data["n_neighbours_cross_dataset"] = rng.integers(0, 10, size=n_cells)
        data["corr_avg_cross_dataset"] = rng.uniform(-0.2, 1.0, size=n_cells)
        data["corr_std_cross_dataset"] = rng.uniform(0.0, 0.5, size=n_cells)

    return pd.DataFrame(data, index=idx)


class TestClassifyFailureModes:
    def test_all_ideal(self):
        """Cell with high correlation on all masks -> WL-1, XL-1, XD-1."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        # Single cell with extremely high, homogeneous correlations
        df = pd.DataFrame(
            {
                "n_neighbours_same_library": [20],
                "corr_avg_same_library": [0.95],
                "corr_std_same_library": [0.01],
                "n_neighbours_cross_library": [15],
                "corr_avg_cross_library": [0.93],
                "corr_std_cross_library": [0.01],
                "n_neighbours_cross_dataset": [10],
                "corr_avg_cross_dataset": [0.90],
                "corr_std_cross_dataset": [0.01],
            },
            index=pd.Index(["ideal_cell"]),
        )

        result = classify_failure_modes(df, threshold_high=0.5, std_threshold=0.1)

        assert result.loc["ideal_cell", "leaf_within_library"] == "WL-1_ideal"
        assert result.loc["ideal_cell", "leaf_cross_library"] == "XL-1_ideal"
        assert result.loc["ideal_cell", "leaf_cross_dataset"] == "XD-1_ideal"
        assert result.loc["ideal_cell", "failure_mode"] == "ideal"

    def test_orphan(self):
        """Cell with zero same_library neighbours -> WL-0."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = pd.DataFrame(
            {
                "n_neighbours_same_library": [0],
                "corr_avg_same_library": [np.nan],
                "corr_std_same_library": [np.nan],
            },
            index=pd.Index(["orphan_cell"]),
        )

        result = classify_failure_modes(df, threshold_high=0.5, std_threshold=0.1)

        assert result.loc["orphan_cell", "leaf_within_library"] == "WL-0_orphan"
        # No cross-library/dataset columns -> no XL/XD
        assert "leaf_cross_library" not in result.columns
        assert "leaf_cross_dataset" not in result.columns

    def test_leaf_columns_present(self):
        """Output has expected columns when all levels are available."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = _make_metrics_df(n_cells=50)
        result = classify_failure_modes(df)

        assert "leaf_within_library" in result.columns
        assert "leaf_cross_library" in result.columns
        assert "leaf_cross_dataset" in result.columns
        assert "failure_mode" in result.columns
        assert len(result) == 50

    def test_vectorised_speed(self):
        """Classify 100k cells in under 1 second."""
        import time

        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = _make_metrics_df(n_cells=100_000)

        start = time.perf_counter()
        classify_failure_modes(df)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"Classification took {elapsed:.2f}s, expected < 1s"

    def test_library_only_mode(self):
        """No XL/XD columns when dataset is absent."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = _make_metrics_df(n_cells=30, has_cross_library=False, has_cross_dataset=False)
        result = classify_failure_modes(df)

        assert "leaf_within_library" in result.columns
        assert "leaf_cross_library" not in result.columns
        assert "leaf_cross_dataset" not in result.columns
        # failure_mode should still exist
        assert "failure_mode" in result.columns

    def test_summarize_counts(self):
        """Sum of counts in summarize output equals total cells."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
            summarize_failure_modes,
        )

        df = _make_metrics_df(n_cells=200)
        leaf_df = classify_failure_modes(df)
        summary = summarize_failure_modes(leaf_df)

        # Check that for each level, counts sum to n_cells
        for level_name in summary["level"].unique():
            level_rows = summary[summary["level"] == level_name]
            total = level_rows["count"].sum()
            assert total == 200, f"Level '{level_name}': count sum {total} != 200"

    def test_all_wl_leaves_valid(self):
        """All WL leaves are from the expected set."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = _make_metrics_df(n_cells=500)
        result = classify_failure_modes(df)

        valid_wl = {
            "WL-0_orphan",
            "WL-1_ideal",
            "WL-2_merged_related",
            "WL-3_noisy",
            "WL-4_false_merge_confident",
            "WL-5_false_merge_partial",
            "WL-unknown",
        }
        unique_wl = set(result["leaf_within_library"].unique())
        assert unique_wl.issubset(valid_wl), f"Unexpected WL leaves: {unique_wl - valid_wl}"

    def test_xd_0a_vs_0b_with_model_comparison(self):
        """XD-0a/0b distinguished by model_comparison_result."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        # Two cells: both isolated cross-dataset, both ideal WL/XL
        df = pd.DataFrame(
            {
                "n_neighbours_same_library": [20, 20],
                "corr_avg_same_library": [0.95, 0.95],
                "corr_std_same_library": [0.01, 0.01],
                "n_neighbours_cross_library": [15, 15],
                "corr_avg_cross_library": [0.93, 0.93],
                "corr_std_cross_library": [0.01, 0.01],
                "n_neighbours_cross_dataset": [0, 0],
                "corr_avg_cross_dataset": [np.nan, np.nan],
                "corr_std_cross_dataset": [np.nan, np.nan],
            },
            index=pd.Index(["enriched", "under_int"]),
        )

        mc = pd.Series([False, True], index=pd.Index(["enriched", "under_int"]))
        result = classify_failure_modes(df, model_comparison_result=mc, threshold_high=0.5, std_threshold=0.1)

        assert result.loc["enriched", "leaf_cross_dataset"] == "XD-0a_dataset_enriched"
        assert result.loc["under_int", "leaf_cross_dataset"] == "XD-0b_under_integration"

    def test_xd_0_isolated_unknown_without_model_comparison(self):
        """XD-0_isolated_unknown when model_comparison_result is None."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = pd.DataFrame(
            {
                "n_neighbours_same_library": [20],
                "corr_avg_same_library": [0.95],
                "corr_std_same_library": [0.01],
                "n_neighbours_cross_library": [15],
                "corr_avg_cross_library": [0.93],
                "corr_std_cross_library": [0.01],
                "n_neighbours_cross_dataset": [0],
                "corr_avg_cross_dataset": [np.nan],
                "corr_std_cross_dataset": [np.nan],
            },
            index=pd.Index(["isolated_cell"]),
        )

        result = classify_failure_modes(df, model_comparison_result=None, threshold_high=0.5, std_threshold=0.1)

        assert result.loc["isolated_cell", "leaf_cross_dataset"] == "XD-0_isolated_unknown"

    def test_dataset_enriched_not_failure(self):
        """XD-0a should result in failure_mode = 'ideal' (not a failure)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_failure_modes,
        )

        df = pd.DataFrame(
            {
                "n_neighbours_same_library": [20],
                "corr_avg_same_library": [0.95],
                "corr_std_same_library": [0.01],
                "n_neighbours_cross_library": [15],
                "corr_avg_cross_library": [0.93],
                "corr_std_cross_library": [0.01],
                "n_neighbours_cross_dataset": [0],
                "corr_avg_cross_dataset": [np.nan],
                "corr_std_cross_dataset": [np.nan],
            },
            index=pd.Index(["enriched"]),
        )

        mc = pd.Series([False], index=pd.Index(["enriched"]))
        result = classify_failure_modes(df, model_comparison_result=mc, threshold_high=0.5, std_threshold=0.1)

        assert result.loc["enriched", "leaf_cross_dataset"] == "XD-0a_dataset_enriched"
        assert result.loc["enriched", "failure_mode"] == "ideal"


# ---------------------------------------------------------------------------
# Sub-plan 07: OVL, isolation norm, headline metrics, composite score
# ---------------------------------------------------------------------------


class TestComputeDistributionOverlap:
    def test_ovl_identical_distributions(self):
        """OVL of a distribution with itself should be 1."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_distribution_overlap,
        )

        rng = np.random.default_rng(42)
        x = rng.uniform(-0.5, 0.8, size=5000)
        ovl = compute_distribution_overlap(x, x, n_bins=50, range_=(-1.0, 1.0))
        assert ovl == pytest.approx(1.0, abs=1e-10)

    def test_ovl_disjoint_distributions(self):
        """OVL of completely disjoint distributions should be 0."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_distribution_overlap,
        )

        x = np.array([0.1, 0.2, 0.3])
        y = np.array([-0.9, -0.8, -0.7])
        ovl = compute_distribution_overlap(x, y, n_bins=50, range_=(-1.0, 1.0))
        assert ovl == 0.0

    def test_ovl_all_nan(self):
        """All-NaN input should return 0, not raise."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_distribution_overlap,
        )

        x = np.array([np.nan, np.nan])
        y = np.array([0.1, 0.2])
        assert compute_distribution_overlap(x, y) == 0.0
        assert compute_distribution_overlap(y, x) == 0.0
        assert compute_distribution_overlap(x, x) == 0.0

    def test_ovl_partial_overlap(self):
        """Partially overlapping distributions give OVL in (0, 1)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_distribution_overlap,
        )

        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 0.2, size=10000)
        y = rng.normal(0.3, 0.2, size=10000)
        ovl = compute_distribution_overlap(x, y, n_bins=100, range_=(-1.0, 1.0))
        assert 0.0 < ovl < 1.0


class TestComputeIsolationNorm:
    def test_isolation_norm_range(self, adata_with_covariates):
        """Isolation norm should be >= 0."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_isolation_norm,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        iso = compute_isolation_norm(
            metrics_df,
            adata,
            "cross_library",
            library_key="library",
            dataset_key="dataset",
        )
        assert not np.isnan(iso)
        assert iso >= 0.0

    def test_isolation_norm_missing_column(self, adata_with_covariates):
        """Returns NaN when the mask column is missing."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_isolation_norm,
        )

        adata = adata_with_covariates
        df = pd.DataFrame({"dummy": [1, 2, 3]})
        iso = compute_isolation_norm(df, adata, "nonexistent_mask", library_key="library")
        assert np.isnan(iso)


class TestSummariseMarkerCorrelation:
    def test_headline_metrics_count(self, adata_with_covariates):
        """Headline Series should have exactly 12 entries."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            summarise_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        headline = summarise_marker_correlation(
            metrics_df,
            adata,
            library_key="library",
            dataset_key="dataset",
        )

        assert isinstance(headline, pd.Series)
        # 12 H1-H12 metrics + H14 cross_technical_correlation (NaN when no technical keys)
        assert len(headline) == 13
        assert "cross_technical_correlation" in headline.index

    def test_single_dataset_mode(self, adata_with_covariates):
        """H7-H10, H12 are NaN when dataset_key is None."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            summarise_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        headline = summarise_marker_correlation(
            metrics_df,
            adata,
            library_key="library",
            dataset_key=None,
        )

        assert np.isnan(headline["corr_cross_dataset"])
        assert np.isnan(headline["corr_gap_dataset"])
        assert np.isnan(headline["isolation_norm_cross_dataset"])
        assert np.isnan(headline["discrepancy_cross_dataset"])
        assert np.isnan(headline["distrib_overlap_dataset"])

    def test_within_library_finite(self, adata_with_covariates):
        """H1 (corr_within_library) should be finite for reasonable data."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            summarise_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        headline = summarise_marker_correlation(
            metrics_df,
            adata,
            library_key="library",
            dataset_key="dataset",
        )

        assert np.isfinite(headline["corr_within_library"])


class TestComputeCompositeScore:
    def test_composite_score_range(self):
        """Total should be in [0, 1] for well-behaved input."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_composite_score,
        )

        headline = pd.Series(
            {
                "corr_within_library": 0.8,
                "corr_consistency": 0.05,
                "corr_cross_library": 0.7,
                "corr_gap_library": 0.1,
                "isolation_norm_cross_library": 0.5,
                "discrepancy_cross_library": 0.02,
                "corr_cross_dataset": 0.6,
                "corr_gap_dataset": 0.2,
                "isolation_norm_cross_dataset": 0.6,
                "discrepancy_cross_dataset": 0.03,
                "distrib_overlap_library": 0.7,
                "distrib_overlap_dataset": 0.6,
            }
        )

        composite = compute_composite_score(headline, has_dataset=True)
        assert 0.0 <= composite["total"] <= 1.0
        assert "bio_conservation" in composite
        assert "library_integration" in composite
        assert "dataset_integration" in composite
        assert "batch_correction" in composite

    def test_composite_single_dataset(self):
        """With has_dataset=False, batch_correction == library_integration."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_composite_score,
        )

        headline = pd.Series(
            {
                "corr_within_library": 0.8,
                "corr_cross_library": 0.7,
                "isolation_norm_cross_library": 0.5,
                "distrib_overlap_library": 0.7,
            }
        )

        composite = compute_composite_score(headline, has_dataset=False)
        assert composite["batch_correction"] == pytest.approx(composite["library_integration"])
        assert np.isnan(composite["dataset_integration"])

    def test_composite_deterministic(self):
        """Same input always gives same output."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_composite_score,
        )

        headline = pd.Series(
            {
                "corr_within_library": 0.75,
                "corr_cross_library": 0.65,
                "isolation_norm_cross_library": 0.4,
                "distrib_overlap_library": 0.8,
            }
        )

        c1 = compute_composite_score(headline, has_dataset=False)
        c2 = compute_composite_score(headline, has_dataset=False)
        pd.testing.assert_series_equal(c1, c2)


class TestStratifiedSummary:
    def test_stratified_summary_shape(self, adata_with_covariates):
        """Rows match unique values of stratify_by column."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            stratified_summary,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )
        # Add library column for stratification
        metrics_df["library"] = adata.obs["library"].values

        result = stratified_summary(metrics_df, stratify_by="library")
        n_libs = adata.obs["library"].nunique()
        assert len(result) == n_libs
        assert "n_cells" in result.columns

    def test_stratified_summary_list(self, adata_with_covariates):
        """List of stratify_by returns dict of DataFrames."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            stratified_summary,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        marker_genes = adata.var_names[:20]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )
        metrics_df["library"] = adata.obs["library"].values
        metrics_df["dataset"] = adata.obs["dataset"].values

        result = stratified_summary(metrics_df, stratify_by=["library", "dataset"])
        assert isinstance(result, dict)
        assert "library" in result
        assert "dataset" in result
        assert len(result["library"]) == adata.obs["library"].nunique()
        assert len(result["dataset"]) == adata.obs["dataset"].nunique()

    def test_stratified_summary_missing_column_raises(self):
        """Raises ValueError when stratify_by column is missing."""
        from regularizedvi.plt._neighbourhood_correlation import (
            stratified_summary,
        )

        df = pd.DataFrame({"corr_avg_same_library": [0.5, 0.6]})
        with pytest.raises(ValueError, match="not found"):
            stratified_summary(df, stratify_by="nonexistent")


# ---------------------------------------------------------------------------
# Sub-plan 08: Cross-model comparison
# ---------------------------------------------------------------------------


def _make_per_model_metrics(n_cells=100, seed=42):
    """Helper: create two per-model metric DataFrames for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"cell_{i}" for i in range(n_cells)], name="obs_names")

    df_a = pd.DataFrame(
        {
            "corr_avg_cross_dataset": rng.uniform(0.1, 0.9, n_cells),
            "corr_avg_same_library": rng.uniform(0.3, 0.95, n_cells),
            "n_neighbours_cross_dataset": rng.integers(0, 10, n_cells),
        },
        index=idx,
    )
    df_b = pd.DataFrame(
        {
            "corr_avg_cross_dataset": rng.uniform(0.1, 0.9, n_cells),
            "corr_avg_same_library": rng.uniform(0.3, 0.95, n_cells),
            "n_neighbours_cross_dataset": rng.integers(0, 10, n_cells),
        },
        index=idx,
    )
    return {"model_A": df_a, "model_B": df_b}


class TestAssembleCrossModelMetrics:
    def test_multi_index_structure(self):
        """Output has correct MultiIndex column levels."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
        )

        per_model = _make_per_model_metrics()
        result = assemble_cross_model_metrics(per_model)

        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["model", "metric"]
        models_in_cols = result.columns.get_level_values("model").unique().tolist()
        assert set(models_in_cols) == {"model_A", "model_B"}

    def test_shared_cell_index_alignment(self):
        """Reindexing to a subset introduces NaN correctly."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
        )

        per_model = _make_per_model_metrics(n_cells=50)
        subset_idx = per_model["model_A"].index[:30]
        result = assemble_cross_model_metrics(per_model, shared_cell_index=subset_idx)

        assert len(result) == 30


class TestBestAchievable:
    def test_uses_nanmax(self):
        """best_achievable ignores NaN and picks max across models."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_best_achievable,
        )

        idx = pd.Index(["c0", "c1", "c2"])
        df_a = pd.DataFrame({"corr_avg_cross_dataset": [0.8, np.nan, 0.3]}, index=idx)
        df_b = pd.DataFrame({"corr_avg_cross_dataset": [0.5, 0.6, np.nan]}, index=idx)
        cross = assemble_cross_model_metrics({"A": df_a, "B": df_b})
        best = compute_best_achievable(cross)

        assert best["c0"] == 0.8  # max(0.8, 0.5)
        assert best["c1"] == 0.6  # max(NaN, 0.6) = 0.6
        assert not np.isnan(best["c2"])  # max(0.3, NaN) = 0.3
        assert best["c2"] == 0.3

    def test_all_nan_gives_nan(self):
        """When all models are NaN, best_achievable is NaN."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_best_achievable,
        )

        idx = pd.Index(["c0"])
        df_a = pd.DataFrame({"corr_avg_cross_dataset": [np.nan]}, index=idx)
        df_b = pd.DataFrame({"corr_avg_cross_dataset": [np.nan]}, index=idx)
        cross = assemble_cross_model_metrics({"A": df_a, "B": df_b})
        best = compute_best_achievable(cross)

        assert np.isnan(best["c0"])


class TestIntegrationFailureRate:
    def test_identical_models_failure_rate_zero(self):
        """With identical metrics, failure_rate = 0."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_integration_failure_rate,
        )

        rng = np.random.default_rng(99)
        n = 100
        idx = pd.Index([f"c{i}" for i in range(n)])
        vals = rng.uniform(0.5, 0.9, n)  # all above threshold_high=0.4
        df = pd.DataFrame({"corr_avg_cross_dataset": vals}, index=idx)
        cross = assemble_cross_model_metrics({"m1": df, "m2": df.copy()})

        rate = compute_integration_failure_rate(cross, "m1", threshold_high=0.4, threshold_low=0.2)
        assert rate == 0.0

    def test_one_model_all_nan_high_failure(self):
        """If one model is all-NaN while the other succeeds, failure rate > 0."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_integration_failure_rate,
        )

        n = 50
        idx = pd.Index([f"c{i}" for i in range(n)])
        df_good = pd.DataFrame({"corr_avg_cross_dataset": np.full(n, 0.8)}, index=idx)
        df_bad = pd.DataFrame({"corr_avg_cross_dataset": np.full(n, np.nan)}, index=idx)
        cross = assemble_cross_model_metrics({"good": df_good, "bad": df_bad})

        rate = compute_integration_failure_rate(cross, "bad", threshold_high=0.4, threshold_low=0.2)
        assert rate == 1.0  # all cells fail for the bad model


class TestModelPairOverlaps:
    def test_identical_models_ovl_one(self):
        """Pairwise OVL between identical distributions = 1."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_model_pair_overlaps,
        )

        rng = np.random.default_rng(77)
        n = 500
        idx = pd.Index([f"c{i}" for i in range(n)])
        vals = rng.uniform(-0.5, 0.9, n)
        df = pd.DataFrame({"corr_avg_cross_dataset": vals}, index=idx)
        cross = assemble_cross_model_metrics({"m1": df, "m2": df.copy()})

        ovl = compute_model_pair_overlaps(cross)
        assert ovl.shape == (2, 2)
        np.testing.assert_allclose(ovl.loc["m1", "m2"], 1.0, atol=1e-10)
        np.testing.assert_allclose(ovl.loc["m1", "m1"], 1.0, atol=1e-10)


class TestContingencyPerCell:
    def test_categories_sum_to_n(self):
        """All 9 categories sum to n_cells."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            compute_contingency_per_cell,
        )

        per_model = _make_per_model_metrics(n_cells=200)
        # Inject some NaNs
        per_model["model_A"].iloc[0:10, 0] = np.nan
        per_model["model_B"].iloc[5:15, 0] = np.nan
        cross = assemble_cross_model_metrics(per_model)

        result = compute_contingency_per_cell(
            cross,
            "model_A",
            "model_B",
            threshold_high=0.4,
            threshold_low=0.2,
        )
        assert len(result) == 200
        assert result["category"].notna().all()
        # No "unclassified" — the 9 categories should be exhaustive
        valid_categories = {
            "both_succeed",
            "A_ok_B_wrong_pairing",
            "A_ok_B_isolates",
            "B_ok_A_wrong_pairing",
            "both_wrong_pairing",
            "A_wrong_B_isolates",
            "B_ok_A_isolates",
            "A_isolates_B_wrong",
            "both_isolate_ambiguous",
        }
        assert set(result["category"].unique()).issubset(valid_categories)


class TestConsensusIsolated:
    def test_all_nan_flagged_true(self):
        """Cells with all-NaN across all models are consensus isolated."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            flag_consensus_isolated,
        )

        idx = pd.Index(["c0", "c1", "c2"])
        df_a = pd.DataFrame(
            {
                "corr_avg_cross_dataset": [np.nan, 0.5, np.nan],
                "n_neighbours_cross_dataset": [0, 5, 0],
            },
            index=idx,
        )
        df_b = pd.DataFrame(
            {
                "corr_avg_cross_dataset": [np.nan, 0.6, 0.1],
                "n_neighbours_cross_dataset": [0, 3, 2],
            },
            index=idx,
        )
        cross = assemble_cross_model_metrics({"A": df_a, "B": df_b})

        flags = flag_consensus_isolated(cross, min_corr=0.3)
        assert flags["c0"] is np.True_  # both NaN / 0 neighbours
        assert flags["c1"] is np.False_  # both have high corr
        assert flags["c2"] is np.True_  # A=NaN, B=0.1 < min_corr

    def test_high_corr_not_flagged(self):
        """Cells where at least one model succeeds are NOT isolated."""
        from regularizedvi.plt._neighbourhood_correlation import (
            assemble_cross_model_metrics,
            flag_consensus_isolated,
        )

        n = 20
        idx = pd.Index([f"c{i}" for i in range(n)])
        df = pd.DataFrame(
            {
                "corr_avg_cross_dataset": np.full(n, 0.8),
                "n_neighbours_cross_dataset": np.full(n, 5, dtype=int),
            },
            index=idx,
        )
        cross = assemble_cross_model_metrics({"m1": df, "m2": df.copy()})

        flags = flag_consensus_isolated(cross, min_corr=0.3)
        assert not flags.any()


# ---------------------------------------------------------------------------
# Sub-plan 09: Visualisation functions
# ---------------------------------------------------------------------------


class TestPlotMarkerCorrelationUmap:
    """Tests for plot_marker_correlation_umap."""

    def test_returns_figure(self, adata_with_covariates):
        """Function returns a matplotlib Figure without error."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import (
            plot_marker_correlation_umap,
        )

        adata = adata_with_covariates
        n = adata.n_obs
        rng = np.random.default_rng(99)

        # Add UMAP coords
        adata.obsm["X_umap"] = rng.standard_normal((n, 2)).astype(np.float32)

        # Build minimal metrics_df
        metrics_df = pd.DataFrame(
            {
                "corr_avg_same_batch": rng.uniform(-0.5, 0.8, n).astype(np.float32),
                "n_neighbours_same_batch": rng.integers(0, 10, n),
            },
            index=adata.obs_names,
        )

        fig = plot_marker_correlation_umap(adata, metrics_df)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_handles_nan_values(self):
        """NaN metric values are plotted in grey without error."""
        import anndata as ad
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import (
            plot_marker_correlation_umap,
        )

        rng = np.random.default_rng(42)
        n = 50
        adata = ad.AnnData(
            X=sparse.random(n, 10, density=0.3, format="csr"),
        )
        adata.obsm["X_umap"] = rng.standard_normal((n, 2)).astype(np.float32)

        vals = rng.uniform(-0.5, 0.8, n).astype(np.float64)
        vals[:10] = np.nan  # 10 NaN cells
        metrics_df = pd.DataFrame(
            {"corr_avg_same_batch": vals},
            index=adata.obs_names,
        )

        fig = plot_marker_correlation_umap(adata, metrics_df)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotMetricHist2d:
    """Tests for plot_metric_hist2d."""

    def test_returns_axes(self):
        """Function returns an Axes object."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.axes import Axes

        from regularizedvi.plt._neighbourhood_correlation import plot_metric_hist2d

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "x": rng.uniform(-1, 1, n),
                "y": rng.uniform(-1, 1, n),
            }
        )
        ax = plot_metric_hist2d(df, "x", "y", bins=20)
        assert isinstance(ax, Axes)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_handles_nan(self):
        """NaN values are dropped without error."""
        import matplotlib

        matplotlib.use("Agg")

        from regularizedvi.plt._neighbourhood_correlation import plot_metric_hist2d

        n = 100
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, n)
        y = rng.uniform(-1, 1, n)
        x[:20] = np.nan
        y[10:30] = np.nan
        df = pd.DataFrame({"x": x, "y": y})

        ax = plot_metric_hist2d(df, "x", "y", bins=20)
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotFailureModeScatter:
    """Tests for plot_failure_mode_scatter."""

    def test_returns_figure(self):
        """Function returns a Figure when matching columns exist."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import (
            plot_failure_mode_scatter,
        )

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "corr_avg_same_batch": rng.uniform(-0.5, 0.8, n),
                "corr_avg_cross_batch": rng.uniform(-0.5, 0.8, n),
                "corr_avg_cross_dataset": rng.uniform(-0.5, 0.8, n),
                "corr_discrepancy_same_batch": rng.uniform(-0.3, 0.3, n),
            }
        )
        fig = plot_failure_mode_scatter(df, bins=20)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotDistributionOverlap:
    """Tests for plot_distribution_overlap."""

    def test_returns_axes(self):
        """Overlaid histogram returns Axes."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.axes import Axes

        from regularizedvi.plt._neighbourhood_correlation import (
            plot_distribution_overlap,
        )

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "a": rng.uniform(-1, 1, n),
                "b": rng.uniform(-0.5, 0.5, n),
            }
        )
        ax = plot_distribution_overlap(df, "a", "b")
        assert isinstance(ax, Axes)
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotIsolationBars:
    """Tests for plot_isolation_bars."""

    def test_returns_figure(self):
        """Bar chart returns Figure."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import plot_isolation_bars

        headlines = {
            "model_A": pd.Series({"isolation_norm_cross_dataset": 0.8}),
            "model_B": pd.Series({"isolation_norm_cross_dataset": 1.2}),
        }
        fig = plot_isolation_bars(headlines)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotLeafDistribution:
    """Tests for plot_leaf_distribution."""

    def test_returns_figure(self):
        """Stacked bar chart returns Figure."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import (
            plot_leaf_distribution,
        )

        rng = np.random.default_rng(42)
        leaves = {
            "model_A": pd.Series(rng.choice(["L1", "L2", "L3"], size=100)),
            "model_B": pd.Series(rng.choice(["L1", "L2", "L4"], size=100)),
        }
        fig = plot_leaf_distribution(leaves)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestHeatmapNeighbourhoodColumns:
    """Test that plot_integration_heatmap handles neighbourhood columns."""

    def test_neighbourhood_metrics_classified(self):
        """Neighbourhood metric names are classified correctly."""
        from regularizedvi.plt._integration_metrics import _classify_metric_col

        assert _classify_metric_col("corr_within_library") == "neighbourhood"
        assert _classify_metric_col("isolation_norm_cross_dataset") == "neighbourhood"
        assert _classify_metric_col("bio_conservation") == "neighbourhood"
        assert _classify_metric_col("distrib_overlap_library") == "neighbourhood"
        assert _classify_metric_col("total") == "neighbourhood"
        # Existing metrics still work
        assert _classify_metric_col("silhouette_label") == "bio"
        assert _classify_metric_col("iLISI") == "batch"
        assert _classify_metric_col("Total") == "summary"

    def test_heatmap_with_neighbourhood_columns_only(self):
        """Heatmap renders with only neighbourhood correlation columns."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._integration_metrics import plot_integration_heatmap

        scib_df = pd.DataFrame(
            {
                "corr_within_library": [0.7, 0.6, 0.8],
                "corr_cross_library": [0.5, 0.4, 0.6],
                "isolation_norm_cross_dataset": [0.9, 1.1, 0.85],
                "bio_conservation": [0.72, 0.65, 0.78],
            },
            index=["exp_A", "exp_B", "exp_C"],
        )

        fig = plot_integration_heatmap(scib_df, sort_by="bio_conservation")
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_heatmap_mixed_scib_and_neighbourhood(self):
        """Heatmap renders with both scIB and neighbourhood columns."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._integration_metrics import plot_integration_heatmap

        scib_df = pd.DataFrame(
            {
                "Total": [0.7, 0.65],
                "silhouette_label": [0.5, 0.45],
                "iLISI_median": [2.1, 1.9],
                "corr_within_library": [0.8, 0.7],
                "isolation_norm_cross_dataset": [0.9, 1.1],
            },
            index=["exp_A", "exp_B"],
        )

        fig = plot_integration_heatmap(scib_df)
        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# Sub-plan 02: select_marker_genes
# ---------------------------------------------------------------------------


class TestSelectMarkerGenes:
    def test_union_includes_curated(self, adata_with_covariates, tmp_path):
        """Union gene set includes genes from curated CSV."""
        from regularizedvi.plt._neighbourhood_correlation import select_marker_genes

        adata = adata_with_covariates
        # Add a label column
        rng = np.random.default_rng(7)
        adata.obs["cell_type"] = rng.choice(["typeA", "typeB", "typeC"], size=adata.n_obs)

        # Create a curated CSV referencing SYMBOL values (gene_0, gene_1...)
        # which map to var_names (0, 1, ...) via the SYMBOL column
        curated_csv = tmp_path / "curated.csv"
        curated_csv.write_text(
            "gene,cell_type,lineage,category\n"
            "gene_0,typeA,lineageX,cat1\n"
            "gene_1,typeB,lineageY,cat2\n"
            "gene_79,typeC,lineageX,cat1\n"
        )

        result = select_marker_genes(
            adata,
            label_columns=["cell_type"],
            dataset_col="dataset",
            curated_marker_csv=str(curated_csv),
            symbol_col="SYMBOL",
            per_dataset=False,
            return_per_level=True,
            mean_threshold=0.0,
            specificity_threshold=0.0,
        )

        assert "union" in result
        assert "curated" in result
        # Curated genes map to var_names via SYMBOL col: gene_0 -> "0", etc.
        expected_var_names = ["0", "1", "79"]
        for g in expected_var_names:
            assert g in result["union"], f"var_name '{g}' missing from union"
            assert g in result["curated"], f"var_name '{g}' missing from curated"

    def test_per_level_dict_has_expected_keys(self, adata_with_covariates):
        """When return_per_level=True, per_level dict has label column keys."""
        from regularizedvi.plt._neighbourhood_correlation import select_marker_genes

        adata = adata_with_covariates
        rng = np.random.default_rng(8)
        adata.obs["level_A"] = rng.choice(["a1", "a2"], size=adata.n_obs)
        adata.obs["level_B"] = rng.choice(["b1", "b2", "b3"], size=adata.n_obs)

        result = select_marker_genes(
            adata,
            label_columns=["level_A", "level_B"],
            dataset_col="dataset",
            per_dataset=False,
            return_per_level=True,
            mean_threshold=0.0,
            specificity_threshold=0.0,
        )

        assert "per_level" in result
        assert "level_A" in result["per_level"]
        assert "level_B" in result["per_level"]


# ---------------------------------------------------------------------------
# Sub-plan 01: compute_cluster_averages
# ---------------------------------------------------------------------------


class TestComputeClusterAverages:
    def test_output_shape(self, adata_with_covariates):
        """Output shape is (n_genes, n_clusters)."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_cluster_averages,
        )

        adata = adata_with_covariates
        rng = np.random.default_rng(10)
        adata.obs["cluster"] = rng.choice(["c1", "c2", "c3"], size=adata.n_obs)

        result = compute_cluster_averages(adata, "cluster", use_raw=False)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (adata.n_vars, 3)
        assert set(result.columns) == {"c1", "c2", "c3"}

    def test_matches_manual_groupby_mean(self, adata_with_covariates):
        """Averages match manual groupby mean computation."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_cluster_averages,
        )

        adata = adata_with_covariates
        rng = np.random.default_rng(11)
        adata.obs["cluster"] = rng.choice(["c1", "c2"], size=adata.n_obs)

        result = compute_cluster_averages(adata, "cluster", use_raw=False)

        # Manual computation
        X_dense = np.asarray(adata.X.todense())
        for cluster in ["c1", "c2"]:
            mask = adata.obs["cluster"].to_numpy() == cluster
            expected_mean = X_dense[mask].mean(axis=0)
            np.testing.assert_allclose(
                result[cluster].to_numpy(),
                expected_mean,
                rtol=1e-5,
                err_msg=f"Cluster {cluster} mean mismatch",
            )


# ---------------------------------------------------------------------------
# Sub-plan 08 (reworked as H14): compute_cross_technical_correlation
# ---------------------------------------------------------------------------


class TestComputeCrossTechnicalCorrelation:
    def test_returns_median_of_cross_technical_column(self):
        """Returns median of `corr_avg_cross_technical`, gated by min_neighbours."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_cross_technical_correlation,
        )

        rng = np.random.default_rng(20)
        n = 60
        idx = pd.Index([f"cell_{i}" for i in range(n)])
        corr = rng.uniform(0.3, 0.9, n)
        nn = rng.integers(1, 10, n)
        metrics_df = pd.DataFrame(
            {
                "corr_avg_cross_technical": corr,
                "n_neighbours_cross_technical": nn,
            },
            index=idx,
        )

        result = compute_cross_technical_correlation(metrics_df)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert 0.0 <= result <= 1.0
        assert result == float(np.nanmedian(corr))

    def test_nan_when_column_missing(self):
        """No technical keys were provided => column missing => NaN."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_cross_technical_correlation,
        )

        metrics_df = pd.DataFrame({"corr_avg_same_library": [0.5, 0.6]})
        assert np.isnan(compute_cross_technical_correlation(metrics_df))

    def test_min_neighbours_filter(self):
        """Only cells with at least `min_neighbours` contribute."""
        from regularizedvi.plt._neighbourhood_correlation import (
            compute_cross_technical_correlation,
        )

        metrics_df = pd.DataFrame(
            {
                "corr_avg_cross_technical": [0.2, 0.8, 0.9],
                "n_neighbours_cross_technical": [0, 1, 5],
            }
        )
        result = compute_cross_technical_correlation(metrics_df, min_neighbours=1)
        # Only rows 1 and 2 count: median(0.8, 0.9) = 0.85
        assert np.isclose(result, 0.85)


# ---------------------------------------------------------------------------
# Sub-plan 09: plot_per_library_distributions
# ---------------------------------------------------------------------------


class TestPlotPerLibraryDistributions:
    def test_returns_figure(self, adata_with_covariates):
        """Function returns a matplotlib Figure without error."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from regularizedvi.plt._neighbourhood_correlation import (
            compute_marker_correlation,
            plot_per_library_distributions,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
            dataset_key="dataset",
        )

        fig = plot_per_library_distributions(
            metrics_df,
            adata,
            metric_col="corr_avg_same_library",
            library_key="library",
            dataset_key="dataset",
        )

        assert isinstance(fig, Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# classify_cell_quality
# ---------------------------------------------------------------------------


class TestClassifyCellQuality:
    def test_good_cells_above_median(self, adata_with_covariates):
        """Cells at or above library median are classified as good."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        result = classify_cell_quality(
            metrics_df,
            adata,
            library_key="library",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == adata.n_obs
        assert "quality_classification" in result.columns
        assert "quality_corr_deviation" in result.columns
        assert "quality_best_lineage_corr" in result.columns
        assert "quality_best_lineage" in result.columns
        assert "quality_ambient_frac" in result.columns
        assert "quality_recon_perplexity" in result.columns

        # Cells with deviation >= 0 should be "good"
        good_mask = result["quality_corr_deviation"] >= 0
        assert (result.loc[good_mask, "quality_classification"] == "good").all()

    def test_no_model_qc_skips_poor_quality(self, adata_with_covariates):
        """Without ambient_frac/recon_perplexity, no poor_quality cells."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        result = classify_cell_quality(
            metrics_df,
            adata,
            library_key="library",
        )

        # Without model QC, no cells should be poor_quality
        assert (result["quality_classification"] != "poor_quality").all()

    def test_with_ambient_frac_flags_poor_quality(self, adata_with_covariates):
        """Providing ambient_frac enables poor_quality classification."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        # Create ambient_frac: set some cells very high
        rng = np.random.default_rng(99)
        af = pd.Series(
            rng.uniform(0.0, 0.3, adata.n_obs),
            index=metrics_df.index,
        )
        # Force a few cells to have high ambient_frac
        af.iloc[:5] = 0.8

        result = classify_cell_quality(
            metrics_df,
            adata,
            library_key="library",
            ambient_frac=af,
            ambient_frac_threshold=0.5,
        )

        # Cells with deviation < 0 AND high ambient_frac should be poor_quality
        poor = result["quality_classification"] == "poor_quality"
        # At least some poor_quality cells exist (the ones we forced)
        below_median = result["quality_corr_deviation"] < 0
        forced_poor = below_median & (af > 0.5)
        # Every forced_poor cell that isn't rare should be poor_quality
        # (rare check only applies if gene_group_metrics provided)
        if forced_poor.any():
            assert poor.any(), "Expected some poor_quality cells"

    def test_classification_is_categorical(self, adata_with_covariates):
        """Output classification column is pandas Categorical."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        marker_genes = adata.var_names[:20]
        conn = adata.obsp["connectivities"]

        metrics_df = compute_marker_correlation(
            adata,
            conn,
            marker_genes,
            library_key="library",
        )

        result = classify_cell_quality(
            metrics_df,
            adata,
            library_key="library",
        )

        assert hasattr(result["quality_classification"], "cat")
        categories = result["quality_classification"].cat.categories.tolist()
        assert set(categories) == {"good", "rare", "poor_quality", "uncertain"}

    def test_with_gene_groups_detects_rare(self, adata_with_covariates):
        """When gene_group_metrics provided, rare cells can be detected."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]

        # Compute metrics with two different gene subsets as "lineage groups"
        markers_a = adata.var_names[:15]
        markers_b = adata.var_names[15:30]

        metrics_main = compute_marker_correlation(
            adata,
            conn,
            adata.var_names[:30],
            library_key="library",
        )
        metrics_a = compute_marker_correlation(
            adata,
            conn,
            markers_a,
            library_key="library",
        )
        metrics_b = compute_marker_correlation(
            adata,
            conn,
            markers_b,
            library_key="library",
        )

        result = classify_cell_quality(
            metrics_main,
            adata,
            library_key="library",
            gene_group_metrics={"lineage_a": metrics_a, "lineage_b": metrics_b},
            lineage_corr_threshold=0.1,
        )

        # best_lineage_corr should not be NaN when gene groups are provided
        assert not result["quality_best_lineage_corr"].isna().all()
        # best_lineage should have values
        assert not result["quality_best_lineage"].isna().all()

    def test_nan_best_lineage_without_gene_groups(self, adata_with_covariates):
        """Without gene_group_metrics, best_lineage columns are all NaN."""
        from regularizedvi.plt._neighbourhood_correlation import (
            classify_cell_quality,
            compute_marker_correlation,
        )

        adata = adata_with_covariates
        conn = adata.obsp["connectivities"]
        metrics_df = compute_marker_correlation(
            adata,
            conn,
            adata.var_names[:20],
            library_key="library",
        )

        result = classify_cell_quality(
            metrics_df,
            adata,
            library_key="library",
        )

        assert result["quality_best_lineage_corr"].isna().all()


# ---------------------------------------------------------------------------
# F1: normalise_counts — total_counts parameter, equivalence, edge cases
# ---------------------------------------------------------------------------


class TestNormaliseCountsTotalCounts:
    def test_full_matrix_subset_equivalence(self):
        """normalise_counts on full X then subset == normalise_counts on marker-subset with total_counts_full."""
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        rng = np.random.default_rng(0)
        n_cells, n_genes, n_markers = 30, 50, 10
        X_full = sparse.csr_matrix(rng.integers(0, 10, (n_cells, n_genes)).astype(np.float32))
        marker_idx = np.arange(n_markers)

        X_norm_full = normalise_counts(X_full, n_vars=n_genes).toarray()
        expected = X_norm_full[:, marker_idx]

        X_sub = X_full[:, marker_idx]
        total_counts = np.asarray(X_full.sum(axis=1)).flatten()
        X_norm_sub = normalise_counts(X_sub, n_vars=n_genes, total_counts=total_counts).toarray()

        np.testing.assert_allclose(X_norm_sub, expected, rtol=1e-5)

    def test_length_mismatch_raises(self):
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X = sparse.csr_matrix(np.ones((5, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="total_counts length"):
            normalise_counts(X, n_vars=10, total_counts=np.array([1.0, 2.0]))

    def test_zero_total_row_yields_zero_row(self):
        """Cells with total_counts==0 produce an all-zero normalised row (no NaN/inf)."""
        from regularizedvi.plt._neighbourhood_correlation import normalise_counts

        X = sparse.csr_matrix(np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32))
        total_counts = np.array([0.0, 6.0])
        out = normalise_counts(X, n_vars=3, total_counts=total_counts).toarray()
        assert np.all(out[0] == 0)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# F3: compute_isolation_norm per-mask formula dispatch
# ---------------------------------------------------------------------------


class TestIsolationNormPerMask:
    def _tiny_adata(self):
        """Build a deterministic adata with 2 datasets, each with 2 libs, 1 tech key."""
        import anndata as ad

        n = 40
        obs = pd.DataFrame(
            {
                "library": (["lib_A"] * 10 + ["lib_B"] * 10 + ["lib_C"] * 10 + ["lib_D"] * 10),
                "dataset": (["ds1"] * 20 + ["ds2"] * 20),
                "tech": (["tX"] * 20 + ["tY"] * 20),
            },
            index=pd.Index([f"c{i}" for i in range(n)]),
        )
        X = sparse.eye(n, format="csr", dtype=np.float32)
        return ad.AnnData(X=X, obs=obs)

    def test_cross_library_restricts_to_within_dataset(self):
        """F3 fix: cross_library baseline must use n_dataset (not n_total)."""
        from regularizedvi.plt._neighbourhood_correlation import compute_isolation_norm

        adata = self._tiny_adata()
        n = adata.n_obs
        # synthetic metrics_df with degree=10 per cell and no cross_library neighbours (fully isolated)
        metrics_df = pd.DataFrame(
            {
                "n_neighbours_cross_library": np.zeros(n, dtype=int),
                "n_neighbours_total": np.full(n, 10, dtype=int),
            },
            index=adata.obs_names,
        )
        iso = compute_isolation_norm(
            metrics_df,
            adata,
            "cross_library",
            library_key="library",
            dataset_key="dataset",
        )
        # With per-mask fix, p_match = (n_ds - n_lib) / (n_total - 1) = (20 - 10) / 39 = 10/39
        # p_iso = (1 - 10/39)^10 ≈ 0.05; observed = 1.0; iso ≈ 1/0.05 = ~20
        p_match = 10 / 39.0
        expected = 1.0 / (1.0 - p_match) ** 10
        assert np.isclose(iso, expected, rtol=1e-6)

    def test_cross_dataset_formula(self):
        from regularizedvi.plt._neighbourhood_correlation import compute_isolation_norm

        adata = self._tiny_adata()
        n = adata.n_obs
        metrics_df = pd.DataFrame(
            {
                "n_neighbours_cross_dataset": np.zeros(n, dtype=int),
                "n_neighbours_total": np.full(n, 10, dtype=int),
            },
            index=adata.obs_names,
        )
        iso = compute_isolation_norm(metrics_df, adata, "cross_dataset", dataset_key="dataset")
        # p_match = (n_total - n_ds) / (n_total - 1) = 20/39
        p_match = 20 / 39.0
        expected = 1.0 / (1.0 - p_match) ** 10
        assert np.isclose(iso, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# F9: NaN covariate values produce warnings + NaN exclusion (no crash)
# ---------------------------------------------------------------------------


class TestNaNCovariateHandling:
    def test_isolation_baseline_warns_and_returns_nan_for_nan_rows(self, caplog):
        import logging

        import anndata as ad

        from regularizedvi.plt._neighbourhood_correlation import (
            compute_analytical_isolation_baseline,
        )

        n = 10
        obs = pd.DataFrame(
            {"library": ["A"] * 5 + ["B"] * 4 + [np.nan]},
            index=pd.Index([f"c{i}" for i in range(n)]),
        )
        adata = ad.AnnData(
            X=sparse.eye(n, format="csr", dtype=np.float32),
            obs=obs,
        )
        with caplog.at_level(logging.WARNING, logger="regularizedvi.plt._neighbourhood_correlation"):
            p_iso = compute_analytical_isolation_baseline(
                adata,
                np.full(n, 5, dtype=np.int64),
                "library",
            )
        assert any("NaN" in r.message for r in caplog.records)
        assert np.isnan(p_iso.iloc[-1])
        assert not np.isnan(p_iso.iloc[0])

    def test_isolation_norm_excludes_nan_cells(self, caplog):
        import logging

        import anndata as ad

        from regularizedvi.plt._neighbourhood_correlation import compute_isolation_norm

        n = 20
        obs = pd.DataFrame(
            {
                "library": ["A"] * 8 + ["B"] * 8 + [np.nan] * 4,
                "dataset": ["d1"] * 16 + [np.nan] * 4,
            },
            index=pd.Index([f"c{i}" for i in range(n)]),
        )
        adata = ad.AnnData(
            X=sparse.eye(n, format="csr", dtype=np.float32),
            obs=obs,
        )
        metrics_df = pd.DataFrame(
            {
                "n_neighbours_cross_library": np.zeros(n, dtype=int),
                "n_neighbours_total": np.full(n, 5, dtype=int),
            },
            index=adata.obs_names,
        )
        with caplog.at_level(logging.WARNING, logger="regularizedvi.plt._neighbourhood_correlation"):
            iso = compute_isolation_norm(
                metrics_df,
                adata,
                "cross_library",
                library_key="library",
                dataset_key="dataset",
            )
        # Should not crash; should emit NaN warning
        assert any("NaN" in r.message for r in caplog.records)
        # 16 valid cells, 4 NaN; answer should be finite and positive
        assert np.isfinite(iso)
        assert iso > 0


# ---------------------------------------------------------------------------
# F10: _compute_combined_failure_mode maps unknown leaves to 'unknown', not 'ideal'
# ---------------------------------------------------------------------------


class TestFailureModeUnknownDefault:
    def test_all_unknown_leaves_label_cell_unknown_not_ideal(self):
        from regularizedvi.plt._neighbourhood_correlation import (
            _compute_combined_failure_mode,
        )

        n = 5
        wl = np.array(["WL-unknown"] * n, dtype=object)
        xl = np.array(["XL-unknown"] * n, dtype=object)
        xd = np.array(["XD-unknown"] * n, dtype=object)
        out = _compute_combined_failure_mode(wl, xl, xd)
        assert all(label == "unknown" for label in out)
        assert not any(label == "ideal" for label in out)


# ---------------------------------------------------------------------------
# F12: compute_composite_score graceful NaN-aware reduction
# ---------------------------------------------------------------------------


class TestCompositeScoreGraceful:
    def test_single_nan_component_still_produces_score(self):
        from regularizedvi.plt._neighbourhood_correlation import compute_composite_score

        headline = pd.Series(
            {
                "corr_within_library": 0.8,
                "corr_cross_library": 0.6,
                "isolation_norm_cross_library": 0.5,
                "distrib_overlap_library": np.nan,  # one NaN component
                "corr_cross_dataset": 0.5,
                "isolation_norm_cross_dataset": 0.5,
                "distrib_overlap_dataset": 0.6,
            }
        )
        result = compute_composite_score(headline, has_dataset=True)
        # library_integration should still be finite: NaN weight renormalised
        assert np.isfinite(result["library_integration"])
        assert np.isfinite(result["total"])

    def test_all_nan_yields_nan(self):
        from regularizedvi.plt._neighbourhood_correlation import compute_composite_score

        headline = pd.Series(
            dict.fromkeys(
                [
                    "corr_within_library",
                    "corr_cross_library",
                    "isolation_norm_cross_library",
                    "distrib_overlap_library",
                    "corr_cross_dataset",
                    "isolation_norm_cross_dataset",
                    "distrib_overlap_dataset",
                ],
                np.nan,
            )
        )
        result = compute_composite_score(headline, has_dataset=True)
        assert np.isnan(result["total"])


# ---------------------------------------------------------------------------
# Sparse Pearson: high-sparsity equivalence to np.corrcoef
# ---------------------------------------------------------------------------


class TestSparsePearsonHighSparsity:
    def test_density_0_05(self):
        """95% zeros — must still match np.corrcoef for non-zero-variance cells."""
        from regularizedvi.plt._neighbourhood_correlation import _sparse_pearson_row_stats

        rng = np.random.default_rng(1)
        n_cells, n_markers = 40, 50
        X_dense = np.zeros((n_cells, n_markers), dtype=np.float32)
        # place ~2-3 non-zeros per cell
        for i in range(n_cells):
            positions = rng.choice(n_markers, size=rng.integers(2, 4), replace=False)
            X_dense[i, positions] = rng.integers(1, 20, size=len(positions))
        X = sparse.csr_matrix(X_dense)

        mean_x, std_x = _sparse_pearson_row_stats(X)
        # Per-row: mean = X_dense.mean(axis=1); std via E[X^2]-E[X]^2
        expected_mean = X_dense.mean(axis=1)
        expected_var = (X_dense**2).mean(axis=1) - expected_mean**2
        expected_std = np.sqrt(np.clip(expected_var, 1e-12, None))
        np.testing.assert_allclose(mean_x, expected_mean, rtol=1e-5)
        np.testing.assert_allclose(std_x, expected_std, rtol=1e-4)

    def test_extreme_sparsity_no_nan_inf(self):
        """Cells with only 1 non-zero marker — must not produce NaN/inf."""
        from regularizedvi.plt._neighbourhood_correlation import _sparse_pearson_row_stats

        n_cells, n_markers = 5, 20
        X_dense = np.zeros((n_cells, n_markers), dtype=np.float32)
        for i in range(n_cells):
            X_dense[i, i] = float(i + 1)  # one non-zero per cell
        X = sparse.csr_matrix(X_dense)

        mean_x, std_x = _sparse_pearson_row_stats(X)
        assert np.all(np.isfinite(mean_x))
        assert np.all(np.isfinite(std_x))
        assert np.all(std_x > 0)
