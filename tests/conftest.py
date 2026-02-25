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

    return mu.MuData({"rna": adata_rna, "atac": adata_atac})
