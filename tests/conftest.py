"""Shared test fixtures for regularizedvi."""

import anndata as ad
import mudata as mu
import numpy as np
import pytest


@pytest.fixture
def adata():
    """Create a small synthetic AnnData for testing."""
    n_obs, n_vars = 100, 50
    n_batches = 3
    rng = np.random.default_rng(42)

    # Simulate count data
    counts = rng.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts.copy()

    # Batch assignments
    adata.obs["batch"] = np.array([f"batch_{i % n_batches}" for i in range(n_obs)])
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    # Extra categorical covariates
    adata.obs["site"] = np.array([f"site_{i % 2}" for i in range(n_obs)])
    adata.obs["site"] = adata.obs["site"].astype("category")
    adata.obs["donor"] = np.array([f"donor_{i % 4}" for i in range(n_obs)])
    adata.obs["donor"] = adata.obs["donor"].astype("category")

    # Gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    return adata


@pytest.fixture
def mdata():
    """Create a small synthetic MuData with RNA + ATAC for testing."""
    n_obs = 100
    n_rna = 50
    n_atac = 30
    n_batches = 3
    rng = np.random.default_rng(42)

    # RNA modality
    rna_counts = rng.poisson(lam=5, size=(n_obs, n_rna)).astype(np.float32)
    adata_rna = ad.AnnData(X=rna_counts)
    adata_rna.var_names = [f"gene_{i}" for i in range(n_rna)]
    adata_rna.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata_rna.obs["batch"] = [f"batch_{i % n_batches}" for i in range(n_obs)]
    adata_rna.obs["batch"] = adata_rna.obs["batch"].astype("category")

    # ATAC modality
    atac_counts = rng.poisson(lam=2, size=(n_obs, n_atac)).astype(np.float32)
    adata_atac = ad.AnnData(X=atac_counts)
    adata_atac.var_names = [f"peak_{i}" for i in range(n_atac)]
    adata_atac.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata_atac.obs["batch"] = [f"batch_{i % n_batches}" for i in range(n_obs)]
    adata_atac.obs["batch"] = adata_atac.obs["batch"].astype("category")

    # Scaling covariate for testing (shared across modalities)
    for adata in [adata_rna, adata_atac]:
        adata.obs["technology"] = [f"tech_{i % 2}" for i in range(n_obs)]
        adata.obs["technology"] = adata.obs["technology"].astype("category")
        # Extra covariates for purpose-driven key testing
        adata.obs["site"] = [f"site_{i % 2}" for i in range(n_obs)]
        adata.obs["site"] = adata.obs["site"].astype("category")
        adata.obs["donor"] = [f"donor_{i % 4}" for i in range(n_obs)]
        adata.obs["donor"] = adata.obs["donor"].astype("category")
        adata.obs["pcr_well"] = [f"well_{i % 5}" for i in range(n_obs)]
        adata.obs["pcr_well"] = adata.obs["pcr_well"].astype("category")

    return mu.MuData({"rna": adata_rna, "atac": adata_atac})


@pytest.fixture
def adata_distinct_covs():
    """AnnData with distinct category counts per covariate (2,3,4,5,6,7,8).

    Each purpose-driven covariate has a unique N so shape mixing is immediately caught.
    """
    n_obs, n_vars = 100, 50
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=5, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts.copy()
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.obs["ambient_cov"] = [f"a{i % 2}" for i in range(n_obs)]  # 2 cats
    adata.obs["nn_cov1"] = [f"n{i % 3}" for i in range(n_obs)]  # 3 cats
    adata.obs["nn_cov2"] = [f"m{i % 4}" for i in range(n_obs)]  # 4 cats
    adata.obs["fs_cov1"] = [f"f{i % 5}" for i in range(n_obs)]  # 5 cats
    adata.obs["fs_cov2"] = [f"g{i % 6}" for i in range(n_obs)]  # 6 cats
    adata.obs["disp_cov"] = [f"d{i % 7}" for i in range(n_obs)]  # 7 cats
    adata.obs["library_cov"] = [f"l{i % 8}" for i in range(n_obs)]  # 8 cats
    for col in ["ambient_cov", "nn_cov1", "nn_cov2", "fs_cov1", "fs_cov2", "disp_cov", "library_cov"]:
        adata.obs[col] = adata.obs[col].astype("category")
    return adata


@pytest.fixture
def mdata_distinct_covs():
    """MuData with distinct category counts per covariate (2,3,4,5,6,7,8).

    Same obs columns on both RNA and ATAC modalities.
    """
    n_obs = 100
    n_rna, n_atac = 50, 30
    rng = np.random.default_rng(42)

    rna_counts = rng.poisson(lam=5, size=(n_obs, n_rna)).astype(np.float32)
    adata_rna = ad.AnnData(X=rna_counts)
    adata_rna.var_names = [f"gene_{i}" for i in range(n_rna)]
    adata_rna.obs_names = [f"cell_{i}" for i in range(n_obs)]

    atac_counts = rng.poisson(lam=2, size=(n_obs, n_atac)).astype(np.float32)
    adata_atac = ad.AnnData(X=atac_counts)
    adata_atac.var_names = [f"peak_{i}" for i in range(n_atac)]
    adata_atac.obs_names = [f"cell_{i}" for i in range(n_obs)]

    for a in [adata_rna, adata_atac]:
        a.obs["ambient_cov"] = [f"a{i % 2}" for i in range(n_obs)]
        a.obs["nn_cov1"] = [f"n{i % 3}" for i in range(n_obs)]
        a.obs["nn_cov2"] = [f"m{i % 4}" for i in range(n_obs)]
        a.obs["fs_cov1"] = [f"f{i % 5}" for i in range(n_obs)]
        a.obs["fs_cov2"] = [f"g{i % 6}" for i in range(n_obs)]
        a.obs["disp_cov"] = [f"d{i % 7}" for i in range(n_obs)]
        a.obs["library_cov"] = [f"l{i % 8}" for i in range(n_obs)]
        for col in ["ambient_cov", "nn_cov1", "nn_cov2", "fs_cov1", "fs_cov2", "disp_cov", "library_cov"]:
            a.obs[col] = a.obs[col].astype("category")

    return mu.MuData({"rna": adata_rna, "atac": adata_atac})


@pytest.fixture
def mdata_single_rna():
    """Create MuData with single RNA modality for N=1 testing."""
    n_obs, n_rna = 100, 50
    n_batches = 3
    rng = np.random.default_rng(42)

    rna_counts = rng.poisson(lam=5, size=(n_obs, n_rna)).astype(np.float32)
    adata_rna = ad.AnnData(X=rna_counts)
    adata_rna.var_names = [f"gene_{i}" for i in range(n_rna)]
    adata_rna.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata_rna.obs["batch"] = [f"batch_{i % n_batches}" for i in range(n_obs)]
    adata_rna.obs["batch"] = adata_rna.obs["batch"].astype("category")
    adata_rna.obs["site"] = [f"site_{i % 2}" for i in range(n_obs)]
    adata_rna.obs["site"] = adata_rna.obs["site"].astype("category")

    return mu.MuData({"rna": adata_rna})
