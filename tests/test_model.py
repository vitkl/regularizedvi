"""Tests for AmbientRegularizedSCVI and RegularizedMultimodalVI models."""

import math

import numpy as np
import pytest
import torch

import regularizedvi


def test_package_has_version():
    assert regularizedvi.__version__ is not None


class TestAmbientRegularizedSCVI:
    """Tests for the AmbientRegularizedSCVI model."""

    def test_setup_anndata(self, adata):
        """Test that setup_anndata registers fields correctly."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        # scvi-tools registers _scvi_uuid and _scvi_manager_uuid in uns
        assert hasattr(adata, "uns")
        assert "_scvi_uuid" in adata.uns or "_scvi" in adata.uns

    def test_model_init(self, adata):
        """Test model initialisation with default parameters."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata)
        assert model.module is not None
        assert isinstance(model.module, regularizedvi.RegularizedVAE)

    def test_model_init_with_covariates(self, adata):
        """Test model initialisation with categorical covariates."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
            nn_conditioning_covariate_keys=["site", "donor"],
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            n_layers=1,
        )
        assert model.module is not None

    def test_default_parameters(self, adata):
        """Test that regularizedvi defaults are applied."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata)
        module = model.module

        # Check regularizedvi-specific defaults
        assert module.gene_likelihood == "gamma_poisson"
        assert module.dispersion == "gene-batch"
        assert module.use_observed_lib_size is False
        assert module.use_additive_background is True
        assert module.use_batch_in_decoder is False
        assert module.regularise_dispersion is True

    def test_additive_background_shape(self, adata):
        """Test additive_background parameter has correct shape."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata)
        module = model.module

        n_vars = adata.shape[1]
        n_batch = len(adata.obs["batch"].cat.categories)

        assert hasattr(module, "additive_background")
        # With batch_key backward compat, one ambient covariate = batch_key
        # Single parameter with concatenated ambient categories
        assert module.additive_background.shape == (n_vars, n_batch)

    def test_dispersion_shape_gene_batch(self, adata):
        """Test dispersion parameter shape with gene-batch mode."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata)

        n_vars = adata.shape[1]
        n_batch = len(adata.obs["batch"].cat.categories)

        assert model.module.px_r_mu.shape == (n_vars, n_batch)
        assert model.module.px_r_log_sigma.shape == (n_vars, n_batch)

    def test_train_short(self, adata):
        """Test that training runs for a few epochs without error."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            n_layers=1,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_train_with_covariates(self, adata):
        """Test training with categorical covariates."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
            nn_conditioning_covariate_keys=["site", "donor"],
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            n_layers=1,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_get_latent_representation(self, adata):
        """Test latent representation extraction."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            n_layers=1,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        latent = model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 4)

    def test_no_additive_background(self, adata):
        """Test model works when additive background is disabled."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            use_additive_background=False,
        )
        assert not hasattr(model.module, "additive_background") or model.module.n_total_ambient_cats == 0
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_with_batch_in_decoder(self, adata):
        """Test model works with batch info in decoder."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            use_batch_in_decoder=True,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_no_dispersion_regularisation(self, adata):
        """Test model works without dispersion regularisation."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            regularise_dispersion=False,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_library_log_vars_weight(self, adata):
        """Test that library_log_vars are scaled by weight."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        # Build with default weight — learned library size is on by default
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            library_log_vars_weight=0.05,
        )
        # use_observed_lib_size should be False (learned library size)
        assert model.module.use_observed_lib_size is False
        # library_log_vars and library_log_means buffers should exist
        assert model.module.library_log_vars is not None
        assert model.module.library_log_means is not None

    def test_ambient_covariate_keys(self, adata):
        """Test setup_anndata with explicit ambient_covariate_keys."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch", "site"],
            dispersion_key="batch",
            library_size_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        module = model.module
        # 2 ambient covariates: batch (3 cats) + site (2 cats)
        assert len(module.n_cats_per_ambient_cov) == 2
        assert module.n_cats_per_ambient_cov[0] == 3  # batch
        assert module.n_cats_per_ambient_cov[1] == 2  # site
        # Single parameter with concatenated ambient categories (3 + 2 = 5)
        assert module.additive_background.shape == (adata.shape[1], 5)
        # Training should work
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_ambient_covariate_keys_required(self, adata):
        """Test that either batch_key or ambient_covariate_keys must be provided."""
        with pytest.raises(ValueError, match="Either batch_key or ambient_covariate_keys"):
            regularizedvi.AmbientRegularizedSCVI.setup_anndata(
                adata,
                layer="counts",
            )

    def test_batch_key_mutual_exclusion(self, adata):
        """batch_key cannot be combined with purpose-specific keys."""
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.AmbientRegularizedSCVI.setup_anndata(
                adata,
                layer="counts",
                batch_key="batch",
                ambient_covariate_keys=["batch"],
            )
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.AmbientRegularizedSCVI.setup_anndata(
                adata,
                layer="counts",
                batch_key="batch",
                dispersion_key="batch",
            )
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.AmbientRegularizedSCVI.setup_anndata(
                adata,
                layer="counts",
                batch_key="batch",
                library_size_key="batch",
            )

    def test_new_style_api_setup(self, adata):
        """Test setup_anndata with purpose-driven keys (no batch_key)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch", "site"],
            dispersion_key="batch",
            library_size_key="donor",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        module = model.module
        # 2 ambient covariates: batch (3 cats) + site (2 cats)
        assert len(module.n_cats_per_ambient_cov) == 2
        assert module.n_cats_per_ambient_cov[0] == 3  # batch
        assert module.n_cats_per_ambient_cov[1] == 2  # site
        # Dispersion: 3 categories (batch)
        assert module.n_dispersion_cats == 3
        # Library: 4 categories (donor)
        assert module.n_library_cats == 4

    def test_new_style_api_train(self, adata):
        """Test training with new-style purpose-driven API end-to-end."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch"],
            dispersion_key="batch",
            library_size_key="donor",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        # Verify training completed and we can get latent
        z = model.get_latent_representation()
        assert z.shape == (adata.n_obs, 4)

    def test_new_style_api_multiple_ambient_covs(self, adata):
        """Test multiple ambient covariates produce concatenated background."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch", "site", "donor"],
            dispersion_key="batch",
            library_size_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        module = model.module
        # 3 ambient covariates: batch(3) + site(2) + donor(4) = 9 total
        assert module.additive_background.shape == (adata.shape[1], 9)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_compute_pearson_single(self, adata):
        """Test that Pearson correlation metrics are logged when compute_pearson=True."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            compute_pearson=True,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

        history = model.history_
        # scvi-tools logs extra_metrics with _{mode} suffix
        assert "pearson_gene_train" in history, f"Missing pearson_gene_train. Keys: {list(history.keys())}"
        assert "pearson_cell_train" in history, f"Missing pearson_cell_train. Keys: {list(history.keys())}"

        # Values should be between -1 and 1
        gene_vals = history["pearson_gene_train"].values
        cell_vals = history["pearson_cell_train"].values
        assert np.all(gene_vals >= -1.0) and np.all(gene_vals <= 1.0), f"pearson_gene out of range: {gene_vals}"
        assert np.all(cell_vals >= -1.0) and np.all(cell_vals <= 1.0), f"pearson_cell out of range: {cell_vals}"

    def test_plot_training_diagnostics_single(self, adata):
        """Smoke test: plot_training_diagnostics returns a Figure for single-modal model."""
        import matplotlib.figure

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        fig = model.plot_training_diagnostics(skip_epochs=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_string_bool_raises(self, adata):
        """String boolean params raise TypeError (papermill -p passes strings)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        with pytest.raises(TypeError, match="must be bool"):
            regularizedvi.AmbientRegularizedSCVI(adata, regularise_background="false")

    def test_string_bool_all_params_raise(self, adata):
        """Each boolean param raises TypeError when passed as string."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        for param in [
            "use_additive_background",
            "use_batch_in_decoder",
            "regularise_dispersion",
            "regularise_background",
            "compute_pearson",
        ]:
            with pytest.raises(TypeError, match=param):
                regularizedvi.AmbientRegularizedSCVI(adata, **{param: "true"})

    def test_int_bool_accepted(self, adata):
        """Int 0/1 are accepted as boolean params (for papermill -r compatibility)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            regularise_background=0,
            compute_pearson=1,
        )
        assert model.module is not None

    def test_compute_latent_umap(self, adata):
        """Test compute_latent_umap populates obsm keys."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        model.compute_latent_umap(adata)
        assert "X_scVI" in adata.obsm
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_scVI"].shape == (100, 4)
        assert adata.obsm["X_umap"].shape == (100, 2)

    def test_compute_latent_umap_with_leiden(self, adata):
        """Test compute_latent_umap with Leiden clustering."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        model.compute_latent_umap(adata, add_leiden=True)
        assert "leiden" in adata.obs

    def test_save_analysis_outputs(self, adata, tmp_path):
        """Test save_analysis_outputs creates expected files."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        model.compute_latent_umap(adata, add_leiden=True)
        saved = model.save_analysis_outputs(str(tmp_path / "outputs"), adata)
        assert len(saved) > 0
        assert any("X_scVI" in p for p in saved)
        assert any("X_umap" in p for p in saved)
        assert any("leiden" in p for p in saved)
        assert any("distances" in p for p in saved)

    def test_save_load_roundtrip(self, adata, tmp_path):
        """Test model save/load round-trip works."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch"],
            dispersion_key="batch",
            library_size_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        save_dir = str(tmp_path / "model")
        model.save(save_dir, overwrite=True)
        loaded = regularizedvi.AmbientRegularizedSCVI.load(save_dir, adata=adata)
        assert loaded.is_trained_
        np.testing.assert_array_equal(model.get_latent_representation(), loaded.get_latent_representation())

    def test_save_load_roundtrip_encoder_covs(self, adata, tmp_path):
        """Test save/load with encoder_covariate_keys (regression test for pickle bug)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["batch"],
            dispersion_key="batch",
            library_size_key="batch",
            encoder_covariate_keys=["batch", "site"],
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        save_dir = str(tmp_path / "model")
        model.save(save_dir, overwrite=True)
        loaded = regularizedvi.AmbientRegularizedSCVI.load(save_dir, adata=adata)
        assert loaded.is_trained_
        np.testing.assert_array_equal(model.get_latent_representation(), loaded.get_latent_representation())

    def test_plot_umap_comparison(self, adata):
        """Test plot_umap_comparison returns a figure without mutating X_umap."""
        import matplotlib
        import matplotlib.pyplot as plt

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, n_layers=1)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        model.compute_latent_umap(adata)
        # Store original X_umap
        original_umap = adata.obsm["X_umap"].copy()
        fig = model.plot_umap_comparison(adata, color="batch")
        assert isinstance(fig, matplotlib.figure.Figure)
        # X_umap should not be mutated
        np.testing.assert_array_equal(adata.obsm["X_umap"], original_umap)
        plt.close(fig)

    def test_compute_latent_umap_with_precomputed_latent(self, adata):
        """Test compute_latent_umap skips GPU when latent is pre-stored."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        # Pre-store latent manually
        latent = model.get_latent_representation()
        adata.obsm["X_scVI"] = latent

        # compute_latent_umap should skip the GPU call
        model.compute_latent_umap(adata)

        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)


class TestRegularizedVAEModule:
    """Tests for the RegularizedVAE module directly."""

    def test_dispersion_regularisation_adds_to_loss(self, adata):
        """Test that dispersion regularisation contributes to the loss."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        # Model with regularisation
        model_reg = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            regularise_dispersion=True,
        )
        # Model without regularisation
        model_noreg = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            regularise_dispersion=False,
        )
        # Both should have the regularise_dispersion attribute set correctly
        assert model_reg.module.regularise_dispersion is True
        assert model_noreg.module.regularise_dispersion is False

        # Regularised model should have learnable prior rate parameter
        assert hasattr(model_reg.module, "dispersion_prior_rate_raw")
        assert isinstance(model_reg.module.dispersion_prior_rate_raw, torch.nn.Parameter)


class TestParameterInitialization:
    """Tests that learnable parameters are initialized at their prior means."""

    # --- RNA-only model (AmbientRegularizedSCVI) ---

    def test_px_r_init_at_prior_mean_gamma_poisson(self, adata):
        """px_r_mu should be initialized near log(rate^2) for gamma_poisson."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        # Default: gamma_poisson, regularise_dispersion_prior=3.0
        # Expected: px_r_mu ≈ log(9) = 2.197, theta = exp(px_r_mu) ≈ 9.0
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.mean().item() == pytest.approx(9.0, rel=0.2)
        # Noise scale: std of px_r_mu should be ~0.1
        assert model.module.px_r_mu.detach().std().item() == pytest.approx(0.1, abs=0.05)
        # log_sigma should be initialized at log(0.1)
        assert model.module.px_r_log_sigma.detach().mean().item() == pytest.approx(math.log(0.1), abs=0.01)

    def test_px_r_init_no_regularisation(self, adata):
        """Without regularisation, px_r_mu uses standard randn init."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, regularise_dispersion=False)
        # randn init: theta = exp(randn) has median ~1.0
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.median().item() < 3.0

    def test_additive_background_init_at_prior_mean(self, adata):
        """Additive background should be initialized near Gamma(1,100) mean=0.01."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        bg = torch.exp(model.module.additive_background).detach()
        assert bg.mean().item() == pytest.approx(0.01, rel=0.2)
        # Noise is very small
        assert model.module.additive_background.detach().std().item() < 0.05

    def test_dispersion_prior_rate_init(self, adata):
        """Dispersion prior rate should initialize at exactly regularise_dispersion_prior."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        rate = torch.nn.functional.softplus(model.module.dispersion_prior_rate_raw).detach()
        assert rate.mean().item() == pytest.approx(3.0, rel=0.01)

    # --- Multimodal model (RegularizedMultimodalVI) ---

    def test_px_r_init_multimodal_gamma_poisson(self, mdata):
        """Multimodal px_r_mu should be initialized near log(rate^2) per modality."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        for name in model.module.modality_names:
            theta = torch.exp(model.module.px_r_mu[name]).detach()
            assert theta.mean().item() == pytest.approx(9.0, rel=0.2), f"Failed for {name}"

    def test_additive_background_init_multimodal(self, mdata):
        """Multimodal additive background should be initialized near Gamma(1,100) mean=0.01."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"]
        )
        # additive_background["rna"] is a single Parameter with concatenated ambient categories.
        # With batch_key backward compat, there is one ambient covariate = batch_key.
        bg_param = model.module.additive_background["rna"]
        n_batch = len(mdata["rna"].obs["batch"].cat.categories)
        assert bg_param.shape == (50, n_batch), f"Expected (50, {n_batch}), got {bg_param.shape}"
        bg = torch.exp(bg_param).detach()
        assert bg.mean().item() == pytest.approx(0.01, rel=0.2)

    def test_region_factors_init_at_prior_mean(self, mdata):
        """Feature scaling softplus(0)/0.7 ≈ 0.99 should match Gamma(200,200) mean=1.0."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, feature_scaling_modalities=["atac"]
        )
        rf = torch.nn.functional.softplus(model.module.feature_scaling["atac"]).detach() / 0.7
        assert rf.mean().item() == pytest.approx(1.0, rel=0.02)

    def test_dispersion_prior_rate_init_multimodal(self, mdata):
        """Multimodal dispersion prior rate should initialize at 3.0."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        for name in model.module.modality_names:
            rate = torch.nn.functional.softplus(model.module.dispersion_prior_rate_raw[name]).detach()
            assert rate.mean().item() == pytest.approx(3.0, rel=0.01), f"Failed for {name}"

    def test_px_r_init_custom_prior(self, adata):
        """px_r_mu initialization adapts to custom regularise_dispersion_prior."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, regularise_dispersion_prior=5.0)
        # rate=5 → theta = 25 → px_r_mu ≈ log(25) = 3.219
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.mean().item() == pytest.approx(25.0, rel=0.2)

    def test_additive_bg_prior_stored(self, adata):
        """Additive background prior hyperparameters should be stored on module."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        assert model.module.additive_bg_prior_alpha == 1.0
        assert model.module.additive_bg_prior_beta == 100.0

    def test_additive_bg_custom_prior(self, adata):
        """Additive background init adapts to custom prior."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata, n_hidden=16, n_latent=4, additive_bg_prior_alpha=2.0, additive_bg_prior_beta=50.0
        )
        # Gamma(2, 50) → mean = 0.04
        bg = torch.exp(model.module.additive_background).detach()
        assert bg.mean().item() == pytest.approx(0.04, rel=0.2)

    def test_additive_bg_prior_multimodal_stored(self, mdata):
        """Multimodal additive background prior hyperparameters should be stored."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"]
        )
        assert model.module.additive_bg_prior_alpha == 1.0
        assert model.module.additive_bg_prior_beta == 100.0

    def test_regularise_background_false_single(self, adata):
        """Single-modal: regularise_background=False keeps param but skips penalty."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, regularise_background=False)
        assert model.module.regularise_background is False
        assert model.module.use_additive_background is True
        assert hasattr(model.module, "additive_background")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_regularise_background_false_multimodal(self, mdata):
        """Multimodal: regularise_background=False keeps param but skips penalty."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"], regularise_background=False
        )
        assert model.module.regularise_background is False
        assert "rna" in model.module.additive_background
        model.train(max_epochs=2, train_size=1.0, batch_size=32)


class TestGammaPoissonMode:
    """Tests for GammaPoisson likelihood (the only supported mode)."""

    def test_gamma_poisson_init(self, adata):
        """Test model initialisation uses GammaPoisson distribution."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        assert model.module.gene_likelihood == "gamma_poisson"

    def test_gamma_poisson_train(self, adata):
        """Test training with GammaPoisson runs without error."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_gamma_poisson_latent(self, adata):
        """Test latent representation with GammaPoisson."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 4)

    def test_feature_scaling_shape_default(self, adata):
        """Test feature_scaling param shape when no scaling covariates."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        # Without scaling covariates: single row, n_genes columns
        assert model.module.feature_scaling.shape == (1, adata.n_vars)

    def test_feature_scaling_shape_with_covs(self, adata):
        """Test feature_scaling param shape with scaling covariates."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
            feature_scaling_covariate_keys=["site", "donor"],
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        n_site = adata.obs["site"].nunique()
        n_donor = adata.obs["donor"].nunique()
        assert model.module.feature_scaling.shape == (n_site + n_donor, adata.n_vars)

    def test_feature_scaling_softplus_init(self, adata):
        """Test that feature_scaling uses softplus/0.7 (centered at ~1 at init)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        fs_val = torch.nn.functional.softplus(model.module.feature_scaling) / 0.7
        assert torch.allclose(fs_val, torch.ones_like(fs_val), atol=0.02)

    def test_train_with_feature_scaling_covs(self, adata):
        """Test training with feature scaling covariates."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
            feature_scaling_covariate_keys=["site", "donor"],
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_feature_scaling_prior_in_loss(self, adata):
        """Test that Gamma prior on feature_scaling contributes to finite loss."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        module = model.module
        module.eval()
        device = next(module.parameters()).device
        scdl = model._make_data_loader(adata=adata, batch_size=32)
        tensors = next(iter(scdl))
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        inf_inputs = module._get_inference_input(tensors)
        inf_outputs = module.inference(**inf_inputs)
        gen_inputs = module._get_generative_input(tensors, inf_outputs)
        gen_outputs = module.generative(**gen_inputs)
        loss_output = module.loss(tensors, inf_outputs, gen_outputs)
        assert loss_output.loss.isfinite()

    def test_parameter_shapes_single_modal(self, adata_distinct_covs):
        """Verify every parameter shape matches architecture expectations."""
        adata = adata_distinct_covs
        n_vars = 50
        n_hidden = 17
        n_latent = 13

        # Covariate counts (each unique!)
        n_ambient = 2
        n_nn1, n_nn2 = 3, 4
        n_fs1, n_fs2 = 5, 6
        n_disp = 7
        n_lib = 8

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["ambient_cov"],
            nn_conditioning_covariate_keys=["nn_cov1", "nn_cov2"],
            feature_scaling_covariate_keys=["fs_cov1", "fs_cov2"],
            dispersion_key="disp_cov",
            library_size_key="library_cov",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=n_hidden, n_latent=n_latent, n_layers=1)

        # --- Expected shapes from architecture ---
        # Encoder: no categoricals (encoder_covariate_keys=False default), no cont covs
        cat_dim_enc = 0
        n_enc_in = n_vars

        # Decoder: nn_conditioning cats [3, 4], no batch (use_batch_in_decoder=False default)
        cat_dim_dec = n_nn1 + n_nn2  # 3 + 4 = 7

        expected = {
            # Z encoder (n_layers=1)
            "z_encoder.encoder.fc_layers.Layer 0.0.weight": (n_hidden, n_enc_in + cat_dim_enc),
            "z_encoder.encoder.fc_layers.Layer 0.0.bias": (n_hidden,),
            "z_encoder.mean_encoder.weight": (n_latent, n_hidden),
            "z_encoder.mean_encoder.bias": (n_latent,),
            "z_encoder.var_encoder.weight": (n_latent, n_hidden),
            "z_encoder.var_encoder.bias": (n_latent,),
            # Library encoder (n_layers=1, n_output=1, n_hidden=DEFAULT_LIBRARY_N_HIDDEN=16)
            "l_encoder.encoder.fc_layers.Layer 0.0.weight": (16, n_enc_in + cat_dim_enc),
            "l_encoder.encoder.fc_layers.Layer 0.0.bias": (16,),
            "l_encoder.mean_encoder.weight": (1, 16),
            "l_encoder.mean_encoder.bias": (1,),
            "l_encoder.var_encoder.weight": (1, 16),
            "l_encoder.var_encoder.bias": (1,),
            # Decoder (n_layers=1)
            "decoder.px_decoder.fc_layers.Layer 0.0.weight": (n_hidden, n_latent + cat_dim_dec),
            "decoder.px_decoder.fc_layers.Layer 0.0.bias": (n_hidden,),
            "decoder.px_scale_decoder.0.weight": (n_vars, n_hidden),
            "decoder.px_scale_decoder.0.bias": (n_vars,),
            "decoder.px_r_decoder.weight": (n_vars, n_hidden),
            "decoder.px_r_decoder.bias": (n_vars,),
            "decoder.px_dropout_decoder.weight": (n_vars, n_hidden),
            "decoder.px_dropout_decoder.bias": (n_vars,),
            # Purpose-driven params
            "px_r_mu": (n_vars, n_disp),
            "px_r_log_sigma": (n_vars, n_disp),
            "dispersion_prior_rate_raw": (n_disp,),
            "additive_background": (n_vars, n_ambient),
            "feature_scaling": (n_fs1 + n_fs2, n_vars),
            # Buffers (library prior, shape (1, n_lib) from reshape(1, -1))
            "library_log_means": (1, n_lib),
            "library_log_vars": (1, n_lib),
        }

        sd = model.module.state_dict()

        for key, shape in expected.items():
            assert key in sd, f"Missing parameter: {key}"
            assert sd[key].shape == torch.Size(shape), f"{key}: expected {shape}, got {tuple(sd[key].shape)}"

        for key in sd:
            assert key in expected, f"Unexpected parameter in state_dict: {key} {tuple(sd[key].shape)}"


class TestRegularizedMultimodalVI:
    """Tests for the RegularizedMultimodalVI multi-modal model."""

    def test_setup_mudata(self, mdata):
        """Test that setup_mudata registers fields correctly."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        manager = regularizedvi.RegularizedMultimodalVI._get_most_recent_anndata_manager(mdata)
        registry_keys = list(manager.data_registry.keys())
        assert "X_rna" in registry_keys
        assert "X_atac" in registry_keys
        assert "batch" in registry_keys

    def test_model_init(self, mdata):
        """Test model initialisation with default parameters."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        assert model.module is not None
        assert isinstance(model.module, regularizedvi.RegularizedMultimodalVAE)

    def test_modality_discovery(self, mdata):
        """Test that modality names are correctly discovered from registered data."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        assert set(model.module.modality_names) == {"rna", "atac"}

    def test_per_modality_architecture(self, mdata):
        """Test per-modality encoders, decoders, and dispersion parameters."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module

        assert set(module.encoders.keys()) == {"rna", "atac"}
        assert set(module.decoders.keys()) == {"rna", "atac"}
        assert set(module.px_r_mu.keys()) == {"rna", "atac"}
        assert set(module.px_r_log_sigma.keys()) == {"rna", "atac"}
        n_batch = 3
        assert module.px_r_mu["rna"].shape == (50, n_batch)  # n_rna genes × n_batch (gene-batch default)
        assert module.px_r_mu["atac"].shape == (30, n_batch)  # n_atac peaks × n_batch

    def test_default_additive_background_and_region_factors(self, mdata):
        """Test default additive background on RNA, feature scaling on ATAC."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module

        assert "rna" in module.additive_background
        assert "atac" not in module.additive_background
        assert "atac" in module.feature_scaling
        assert "rna" not in module.feature_scaling

    def test_per_modality_n_hidden_n_latent(self, mdata):
        """Test per-modality architecture sizes via dict config."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden={"rna": 32, "atac": 16},
            n_latent={"rna": 8, "atac": 4},
        )
        module = model.module
        assert module.n_hidden_dict == {"rna": 32, "atac": 16}
        assert module.n_latent_dict == {"rna": 8, "atac": 4}
        assert module.total_latent_dim == 12  # 8 + 4

    def test_train_concatenation(self, mdata):
        """Test training with concatenation latent mode (default)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="concatenation")
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_train_single_encoder(self, mdata):
        """Test training with single_encoder latent mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="single_encoder")
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_train_weighted_mean(self, mdata):
        """Test training with weighted_mean latent mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="weighted_mean")
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_latent_concatenation(self, mdata):
        """Test latent representation shape for concatenation mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="concatenation")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (mdata.n_obs, 8)  # 4 + 4

    def test_latent_single_encoder(self, mdata):
        """Test latent representation shape for single_encoder mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="single_encoder")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (mdata.n_obs, 8)  # sum of per-modality n_latent

    def test_latent_weighted_mean(self, mdata):
        """Test latent representation shape for weighted_mean mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="weighted_mean")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (mdata.n_obs, 4)

    def test_latent_per_modality_sizes(self, mdata):
        """Test latent shape with per-modality n_latent in concatenation mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden={"rna": 32, "atac": 16},
            n_latent={"rna": 8, "atac": 4},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (mdata.n_obs, 12)  # 8 + 4

    def test_dispersion_regularisation(self, mdata):
        """Test learnable dispersion prior is present."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, regularise_dispersion=True)
        module = model.module
        assert hasattr(module, "dispersion_prior_rate_raw")
        assert "rna" in module.dispersion_prior_rate_raw
        assert "atac" in module.dispersion_prior_rate_raw

    def test_no_dispersion_regularisation(self, mdata):
        """Test model works without dispersion regularisation."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, regularise_dispersion=False)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_custom_modality_flags(self, mdata):
        """Test custom additive_background and feature_scaling modality lists."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        # Reverse the defaults: background on ATAC, feature scaling on RNA
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            additive_background_modalities=["atac"],
            feature_scaling_modalities=["rna"],
        )
        module = model.module
        assert "atac" in module.additive_background
        assert "rna" not in module.additive_background
        assert "rna" in module.feature_scaling
        assert "atac" not in module.feature_scaling
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_weighted_mean_universal_weights(self, mdata):
        """Test weighted_mean with universal weight mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            latent_mode="weighted_mean",
            modality_weights="universal",
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_weighted_mean_cell_weights(self, mdata):
        """Test weighted_mean with cell-level weight mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            latent_mode="weighted_mean",
            modality_weights="cell",
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_gene_batch_dispersion(self, mdata):
        """Test per-batch dispersion parameterization."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, dispersion="gene-batch")
        module = model.module
        n_batch = 3
        assert module.px_r_mu["rna"].shape == (50, n_batch)
        assert module.px_r_mu["atac"].shape == (30, n_batch)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_extra_metrics_in_loss(self, mdata):
        """Test that loss() returns per-modality extra_metrics."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        # Run one forward pass through the module to get the LossOutput
        module = model.module
        module.eval()
        device = next(module.parameters()).device
        scdl = model._make_data_loader(adata=mdata, batch_size=32)
        tensors = next(iter(scdl))
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        inf_inputs = module._get_inference_input(tensors)
        gen_inputs = module._get_generative_input(tensors, module.inference(**inf_inputs))
        inf_outputs = module.inference(**inf_inputs)
        gen_outputs = module.generative(**gen_inputs)
        loss_output = module.loss(tensors, inf_outputs, gen_outputs)

        em = loss_output.extra_metrics
        # Check per-modality recon_loss
        assert "recon_loss_rna" in em
        assert "recon_loss_atac" in em
        # Check per-modality kl_z (concatenation mode)
        assert "kl_z_rna" in em
        assert "kl_z_atac" in em
        # Check per-modality z_var
        assert "z_var_rna" in em
        assert "z_var_atac" in em
        # All should be 0-d tensors
        for key, val in em.items():
            assert val.dim() == 0, f"{key} should be a scalar tensor"

    def test_extra_metrics_logged_in_history(self, mdata):
        """Test that extra_metrics appear in model.history_ after training."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

        history = model.history_
        # scvi-tools logs extra_metrics with _{mode} suffix
        assert "recon_loss_rna_train" in history
        assert "recon_loss_atac_train" in history
        assert "kl_z_rna_train" in history
        assert "kl_z_atac_train" in history
        assert "z_var_rna_train" in history
        assert "z_var_atac_train" in history

    def test_get_modality_attribution(self, mdata):
        """Test get_modality_attribution returns correct shapes."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        result = model.get_modality_attribution(batch_size=32)

        assert "rna" in result
        assert "atac" in result

        n_obs = mdata.n_obs
        n_latent = model.module.total_latent_dim  # 4 + 4 = 8

        for name in ["rna", "atac"]:
            assert "attribution" in result[name]
            assert "weighted_z" in result[name]
            assert result[name]["attribution"].shape == (n_obs, n_latent)
            assert result[name]["weighted_z"].shape == (n_obs, n_latent)
            # Attribution values should be non-negative (mean of abs)
            assert np.all(result[name]["attribution"] >= 0)

    def test_get_modality_attribution_per_modality_latent(self, mdata):
        """Test attribution with different per-modality latent sizes."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent={"rna": 6, "atac": 3},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        result = model.get_modality_attribution(batch_size=32)

        n_latent = 6 + 3
        for name in ["rna", "atac"]:
            assert result[name]["attribution"].shape == (mdata.n_obs, n_latent)

    # --- Feature scaling: shape, activation, prior, scaling covariates ---

    def test_region_factors_shape_no_scaling_covs(self, mdata):
        """Test feature scaling shape without scaling covariates (backward compat)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # Default: ATAC gets feature scaling, shape (1, n_atac_features) without scaling covs
        assert "atac" in module.feature_scaling
        assert module.feature_scaling["atac"].shape == (1, 30)

    def test_region_factors_shape_with_scaling_covs(self, mdata):
        """Test feature scaling shape with scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            feature_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        n_tech = 2  # tech_0, tech_1
        assert module.feature_scaling["atac"].shape == (n_tech, 30)

    def test_region_factors_softplus_activation(self, mdata):
        """Test that feature scaling use softplus/0.7 activation (centered at ~1)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # At initialization (param=0), softplus(0)/0.7 ~ 0.693/0.7 ~ 0.99
        rf_val = torch.nn.functional.softplus(module.feature_scaling["atac"]) / 0.7
        assert torch.allclose(rf_val, torch.ones_like(rf_val), atol=0.02)

    def test_train_with_scaling_covs(self, mdata):
        """Test training with feature scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            feature_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_region_factors_prior_in_loss(self, mdata):
        """Test that Gamma prior on feature scaling contributes to finite loss."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        module = model.module
        module.eval()
        device = next(module.parameters()).device
        scdl = model._make_data_loader(adata=mdata, batch_size=32)
        tensors = next(iter(scdl))
        tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        inf_inputs = module._get_inference_input(tensors)
        inf_outputs = module.inference(**inf_inputs)
        gen_inputs = module._get_generative_input(tensors, inf_outputs)
        gen_outputs = module.generative(**gen_inputs)
        loss_output = module.loss(tensors, inf_outputs, gen_outputs)
        assert loss_output.loss.isfinite()

    def test_region_factors_with_multiple_scaling_covs(self, mdata):
        """Test feature scaling with multiple scaling covariates."""
        # Add a second covariate
        for mod_key in mdata.mod:
            mdata[mod_key].obs["site"] = [f"site_{i % 3}" for i in range(mdata.n_obs)]
            mdata[mod_key].obs["site"] = mdata[mod_key].obs["site"].astype("category")

        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            feature_scaling_covariate_keys=["technology", "site"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # 2 tech + 3 sites = 5 total rows
        assert module.feature_scaling["atac"].shape == (5, 30)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_attribution_with_scaling_covs(self, mdata):
        """Test get_modality_attribution works with scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            feature_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        result = model.get_modality_attribution(batch_size=32)
        n_latent = model.module.total_latent_dim
        for name in ["rna", "atac"]:
            assert result[name]["attribution"].shape == (mdata.n_obs, n_latent)

    def test_batch_key_mutual_exclusion(self, mdata):
        """batch_key cannot be combined with purpose-specific keys."""
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.RegularizedMultimodalVI.setup_mudata(
                mdata,
                batch_key="batch",
                ambient_covariate_keys=["batch"],
            )
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.RegularizedMultimodalVI.setup_mudata(
                mdata,
                batch_key="batch",
                dispersion_key="batch",
            )
        with pytest.raises(ValueError, match="batch_key cannot be combined"):
            regularizedvi.RegularizedMultimodalVI.setup_mudata(
                mdata,
                batch_key="batch",
                library_size_key="batch",
            )

    def test_new_style_api_setup(self, mdata):
        """Test setup_mudata with purpose-driven keys (no batch_key)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["batch", "site"],
            dispersion_key="batch",
            library_size_key="pcr_well",
            nn_conditioning_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"]
        )
        # 2 ambient covariates: batch (3 cats) + site (2 cats) = 5 total
        assert len(model.module.n_cats_per_ambient_cov) == 2
        assert model.module.n_cats_per_ambient_cov[0] == 3  # batch
        assert model.module.n_cats_per_ambient_cov[1] == 2  # site
        # Single parameter with concatenated ambient categories (3 + 2 = 5)
        assert model.module.additive_background["rna"].shape == (50, 5)
        # Dispersion: 3 categories (batch)
        assert model.module.n_dispersion_cats == 3
        # Library: 5 categories (pcr_well)
        assert model.module.n_library_cats == 5

    def test_new_style_api_train(self, mdata):
        """Test training with new-style purpose-driven API end-to-end."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["batch"],
            dispersion_key="batch",
            library_size_key="pcr_well",
        )
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"]
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        # Verify training completed and we can get latent
        z = model.get_latent_representation()
        assert z.shape == (mdata.n_obs, model.module.total_latent_dim)

    def test_new_style_api_multiple_ambient_covs(self, mdata):
        """Test multiple ambient covariates produce concatenated background."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["batch", "site", "donor"],
            dispersion_key="batch",
            library_size_key="batch",
        )
        model = regularizedvi.RegularizedMultimodalVI(
            mdata, n_hidden=16, n_latent=4, additive_background_modalities=["rna"]
        )
        # 3 ambient covariates: batch(3) + site(2) + donor(4) = 9 total
        bg_param = model.module.additive_background["rna"]
        assert bg_param.shape == (50, 9)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_attribution_with_categorical_covariates(self, mdata):
        """Test get_modality_attribution works with categorical covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            nn_conditioning_covariate_keys=["site", "donor"],
        )
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        result = model.get_modality_attribution(batch_size=32)

        assert "rna" in result
        assert "atac" in result

        n_obs = mdata.n_obs
        n_latent = model.module.total_latent_dim  # 4 + 4 = 8

        for name in ["rna", "atac"]:
            assert "attribution" in result[name]
            assert "weighted_z" in result[name]
            assert result[name]["attribution"].shape == (n_obs, n_latent)
            assert result[name]["weighted_z"].shape == (n_obs, n_latent)
            # Attribution values should be non-negative (mean of abs)
            assert np.all(result[name]["attribution"] >= 0)

    def test_compute_pearson_multimodal(self, mdata):
        """Test that per-modality Pearson correlation metrics are logged."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            compute_pearson=True,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

        history = model.history_
        # Per-modality Pearson metrics with _{mode} suffix
        for mod_name in ["rna", "atac"]:
            gene_key = f"pearson_gene_{mod_name}_train"
            cell_key = f"pearson_cell_{mod_name}_train"
            assert gene_key in history, f"Missing {gene_key}. Keys: {list(history.keys())}"
            assert cell_key in history, f"Missing {cell_key}. Keys: {list(history.keys())}"

            # Values should be between -1 and 1
            gene_vals = history[gene_key].values
            cell_vals = history[cell_key].values
            assert np.all(gene_vals >= -1.0) and np.all(gene_vals <= 1.0), f"{gene_key} out of range: {gene_vals}"
            assert np.all(cell_vals >= -1.0) and np.all(cell_vals <= 1.0), f"{cell_key} out of range: {cell_vals}"

    def test_get_modality_latents(self, mdata):
        """Test get_modality_latents returns dict with correct shapes."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent={"rna": 6, "atac": 4},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        result = model.get_modality_latents()

        # Should have modality keys + "__joint__"
        assert "rna" in result
        assert "atac" in result
        assert "__joint__" in result

        n_cells = mdata.n_obs

        # Check per-modality shapes
        assert result["rna"].shape == (n_cells, 6)
        assert result["atac"].shape == (n_cells, 4)
        assert result["__joint__"].shape == (n_cells, 10)  # 6 + 4

        # Joint should equal concatenation of per-modality latents in modality_names order
        parts = [result[name] for name in model.module.modality_names]
        joint_from_parts = np.concatenate(parts, axis=1)
        np.testing.assert_array_equal(result["__joint__"], joint_from_parts)

    def test_get_modality_latents_weighted_mean_raises(self, mdata):
        """Test get_modality_latents raises ValueError for weighted_mean mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            latent_mode="weighted_mean",
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        with pytest.raises(ValueError, match="weighted_mean"):
            model.get_modality_latents()

    def test_plot_training_diagnostics_multimodal(self, mdata):
        """Smoke test: plot_training_diagnostics returns a Figure for multimodal model."""
        import matplotlib.figure

        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

        fig = model.plot_training_diagnostics(skip_epochs=0)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_string_bool_raises_multimodal(self, mdata):
        """String boolean params raise TypeError in multimodal model."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(TypeError, match="must be bool"):
            regularizedvi.RegularizedMultimodalVI(mdata, regularise_background="false")

    def test_string_bool_all_params_raise_multimodal(self, mdata):
        """Each boolean param raises TypeError when passed as string."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        for param in [
            "use_batch_in_decoder",
            "regularise_dispersion",
            "regularise_background",
            "compute_pearson",
        ]:
            with pytest.raises(TypeError, match=param):
                regularizedvi.RegularizedMultimodalVI(mdata, **{param: "true"})

    def test_compute_latent_umap_multimodal(self, mdata):
        """Test compute_latent_umap populates joint + per-modality UMAPs."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        adata_rna = mdata["rna"]
        model.compute_latent_umap(adata_rna)
        assert "X_multiVI_joint" in adata_rna.obsm
        assert "X_umap_joint" in adata_rna.obsm
        assert "X_multiVI_rna" in adata_rna.obsm
        assert "X_umap_rna" in adata_rna.obsm
        assert "X_multiVI_atac" in adata_rna.obsm
        assert "X_umap_atac" in adata_rna.obsm
        # X_umap should be set to joint UMAP
        np.testing.assert_array_equal(adata_rna.obsm["X_umap"], adata_rna.obsm["X_umap_joint"])

    def test_save_analysis_outputs_multimodal(self, mdata, tmp_path):
        """Test save_analysis_outputs creates expected files for multimodal."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        adata_rna = mdata["rna"]
        model.compute_latent_umap(adata_rna)
        saved = model.save_analysis_outputs(str(tmp_path / "outputs"), adata_rna)
        assert len(saved) > 0
        assert any("joint" in p for p in saved)
        assert any("distances" in p for p in saved)

    def test_save_load_roundtrip_multimodal(self, mdata, tmp_path):
        """Test multimodal model save/load round-trip works."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        save_dir = str(tmp_path / "model")
        model.save(save_dir, overwrite=True)
        loaded = regularizedvi.RegularizedMultimodalVI.load(save_dir, adata=mdata)
        assert loaded.is_trained_
        np.testing.assert_array_equal(model.get_latent_representation(), loaded.get_latent_representation())

    def test_save_load_roundtrip_multimodal_encoder_covs(self, mdata, tmp_path):
        """Test multimodal save/load with encoder_covariate_keys (regression test for pickle bug)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["batch"],
            dispersion_key="batch",
            library_size_key="batch",
            nn_conditioning_covariate_keys=["site"],
            encoder_covariate_keys=["batch", "site"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        save_dir = str(tmp_path / "model")
        model.save(save_dir, overwrite=True)
        loaded = regularizedvi.RegularizedMultimodalVI.load(save_dir, adata=mdata)
        assert loaded.is_trained_
        np.testing.assert_array_equal(model.get_latent_representation(), loaded.get_latent_representation())

    def test_plot_modality_attribution(self, mdata):
        """Test plot_modality_attribution returns attribution dict and figure."""
        import matplotlib
        import matplotlib.pyplot as plt

        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        attribution, fig = model.plot_modality_attribution(batch_size=32)
        assert isinstance(attribution, dict)
        assert "rna" in attribution
        assert "atac" in attribution
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_plot_umap_comparison_multimodal(self, mdata):
        """Test plot_umap_comparison returns a figure without mutating X_umap."""
        import matplotlib
        import matplotlib.pyplot as plt

        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        adata_rna = mdata["rna"]
        model.compute_latent_umap(adata_rna)
        original_umap = adata_rna.obsm["X_umap"].copy()
        fig = model.plot_umap_comparison(adata_rna, color="batch")
        assert isinstance(fig, matplotlib.figure.Figure)
        np.testing.assert_array_equal(adata_rna.obsm["X_umap"], original_umap)
        plt.close(fig)

    def test_compute_latent_umap_with_precomputed_latents(self, mdata):
        """Test compute_latent_umap skips GPU when latents are pre-stored."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        adata_rna = mdata["rna"]

        # Pre-store latents manually (simulating GPU → save → CPU workflow)
        latent_dict = model.get_modality_latents()
        adata_rna.obsm["X_multiVI_joint"] = latent_dict["__joint__"]
        for name in model.module.modality_names:
            adata_rna.obsm[f"X_multiVI_{name}"] = latent_dict[name]

        # Now compute_latent_umap should skip the GPU call and just do KNN+UMAP
        model.compute_latent_umap(adata_rna)

        assert "X_umap_joint" in adata_rna.obsm
        assert adata_rna.obsm["X_umap_joint"].shape == (mdata["rna"].n_obs, 2)
        for name in model.module.modality_names:
            assert f"X_umap_{name}" in adata_rna.obsm

    def test_store_attribution_results(self, mdata):
        """Test store_attribution_results populates obsm + obs keys."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        adata_rna = mdata["rna"]
        attribution = model.store_attribution_results(adata_rna, batch_size=32)

        # Check return value
        assert isinstance(attribution, dict)
        for name in model.module.modality_names:
            assert name in attribution

        # Check obsm keys (weighted latents)
        for name in model.module.modality_names:
            assert f"X_multiVI_attr_{name}" in adata_rna.obsm
            assert adata_rna.obsm[f"X_multiVI_attr_{name}"].shape[0] == mdata["rna"].n_obs

        # Check obs columns
        for name in model.module.modality_names:
            assert f"{name}_decoder_total_attr" in adata_rna.obs
            assert f"{name}_decoder_own_attr" in adata_rna.obs

        # Log2 ratio (2 modalities: alphabetical order atac < rna)
        assert "log2_atac_vs_rna_attr" in adata_rna.obs

        # No UMAP keys yet (that's compute_attribution_umap's job)
        for name in model.module.modality_names:
            assert f"X_umap_attr_{name}" not in adata_rna.obsm

    def test_compute_attribution_umap(self, mdata):
        """Test compute_attribution_umap computes UMAPs on attribution latents."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        adata_rna = mdata["rna"]
        model.compute_latent_umap(adata_rna)  # populates X_umap_joint
        model.store_attribution_results(adata_rna, batch_size=32)
        model.compute_attribution_umap(adata_rna)

        # Check UMAP keys
        for name in model.module.modality_names:
            assert f"X_umap_attr_{name}" in adata_rna.obsm
            assert adata_rna.obsm[f"X_umap_attr_{name}"].shape == (mdata["rna"].n_obs, 2)

        # X_umap should be restored to joint
        np.testing.assert_array_equal(adata_rna.obsm["X_umap"], adata_rna.obsm["X_umap_joint"])

    def test_parameter_shapes_multimodal(self, mdata_distinct_covs):
        """Verify every parameter shape matches architecture expectations."""
        mdata = mdata_distinct_covs
        n_rna, n_atac = 50, 30
        n_hidden_rna, n_hidden_atac = 17, 19
        n_latent_rna, n_latent_atac = 13, 11
        total_latent = n_latent_rna + n_latent_atac  # 24
        lib_n_hidden = 16  # multimodal default library_n_hidden

        # Covariate counts (each unique!)
        n_ambient = 2
        n_nn1, n_nn2 = 3, 4
        n_fs1, n_fs2 = 5, 6
        n_disp = 7
        n_lib = 8

        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["ambient_cov"],
            nn_conditioning_covariate_keys=["nn_cov1", "nn_cov2"],
            feature_scaling_covariate_keys=["fs_cov1", "fs_cov2"],
            dispersion_key="disp_cov",
            library_size_key="library_cov",
        )
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden={"rna": n_hidden_rna, "atac": n_hidden_atac},
            n_latent={"rna": n_latent_rna, "atac": n_latent_atac},
            n_layers=1,
            additive_background_modalities=["rna", "atac"],
            feature_scaling_modalities=["rna", "atac"],
        )

        # --- Expected shapes from architecture ---
        # Encoder: no categoricals (encoder_covariate_keys=False default)
        cat_dim_enc = 0
        # Decoder: nn_conditioning cats [3, 4], no batch
        cat_dim_dec = n_nn1 + n_nn2  # 7

        expected = {}

        # Per-modality encoders, decoders, library encoders
        for name, n_feat, n_hid, n_lat in [
            ("atac", n_atac, n_hidden_atac, n_latent_atac),
            ("rna", n_rna, n_hidden_rna, n_latent_rna),
        ]:
            # Encoder (n_layers=1)
            expected[f"encoders.{name}.encoder.fc_layers.Layer 0.0.weight"] = (n_hid, n_feat + cat_dim_enc)
            expected[f"encoders.{name}.encoder.fc_layers.Layer 0.0.bias"] = (n_hid,)
            expected[f"encoders.{name}.mean_encoder.weight"] = (n_lat, n_hid)
            expected[f"encoders.{name}.mean_encoder.bias"] = (n_lat,)
            expected[f"encoders.{name}.var_encoder.weight"] = (n_lat, n_hid)
            expected[f"encoders.{name}.var_encoder.bias"] = (n_lat,)

            # Library encoder (n_layers=1, n_output=1, n_hidden=lib_n_hidden=16)
            expected[f"l_encoders.{name}.encoder.fc_layers.Layer 0.0.weight"] = (lib_n_hidden, n_feat + cat_dim_enc)
            expected[f"l_encoders.{name}.encoder.fc_layers.Layer 0.0.bias"] = (lib_n_hidden,)
            expected[f"l_encoders.{name}.mean_encoder.weight"] = (1, lib_n_hidden)
            expected[f"l_encoders.{name}.mean_encoder.bias"] = (1,)
            expected[f"l_encoders.{name}.var_encoder.weight"] = (1, lib_n_hidden)
            expected[f"l_encoders.{name}.var_encoder.bias"] = (1,)

            # Decoder (n_layers=1, input=total_latent)
            expected[f"decoders.{name}.px_decoder.fc_layers.Layer 0.0.weight"] = (n_hid, total_latent + cat_dim_dec)
            expected[f"decoders.{name}.px_decoder.fc_layers.Layer 0.0.bias"] = (n_hid,)
            expected[f"decoders.{name}.px_scale_decoder.0.weight"] = (n_feat, n_hid)
            expected[f"decoders.{name}.px_scale_decoder.0.bias"] = (n_feat,)
            expected[f"decoders.{name}.px_r_decoder.weight"] = (n_feat, n_hid)
            expected[f"decoders.{name}.px_r_decoder.bias"] = (n_feat,)
            expected[f"decoders.{name}.px_dropout_decoder.weight"] = (n_feat, n_hid)
            expected[f"decoders.{name}.px_dropout_decoder.bias"] = (n_feat,)

            # Purpose-driven params (ParameterDict uses dot notation)
            expected[f"px_r_mu.{name}"] = (n_feat, n_disp)
            expected[f"px_r_log_sigma.{name}"] = (n_feat, n_disp)
            expected[f"dispersion_prior_rate_raw.{name}"] = (n_disp,)
            expected[f"additive_background.{name}"] = (n_feat, n_ambient)
            expected[f"feature_scaling.{name}"] = (n_fs1 + n_fs2, n_feat)

            # Buffers (register_buffer uses underscore naming)
            expected[f"library_log_means_{name}"] = (1, n_lib)
            expected[f"library_log_vars_{name}"] = (1, n_lib)

        sd = model.module.state_dict()

        for key, shape in expected.items():
            assert key in sd, f"Missing parameter: {key}"
            assert sd[key].shape == torch.Size(shape), f"{key}: expected {shape}, got {tuple(sd[key].shape)}"

        for key in sd:
            assert key in expected, f"Unexpected parameter in state_dict: {key} {tuple(sd[key].shape)}"

    def test_initial_parameter_consistency(self, adata_distinct_covs, mdata_distinct_covs):
        """Compare initial parameter values between single-modal and multi-modal RNA."""
        # Single-modal setup
        adata = adata_distinct_covs
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            ambient_covariate_keys=["ambient_cov"],
            nn_conditioning_covariate_keys=["nn_cov1", "nn_cov2"],
            feature_scaling_covariate_keys=["fs_cov1", "fs_cov2"],
            dispersion_key="disp_cov",
            library_size_key="library_cov",
        )
        sm_model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=17, n_latent=13, n_layers=1)

        # Multi-modal setup (matching RNA architecture)
        mdata = mdata_distinct_covs
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            ambient_covariate_keys=["ambient_cov"],
            nn_conditioning_covariate_keys=["nn_cov1", "nn_cov2"],
            feature_scaling_covariate_keys=["fs_cov1", "fs_cov2"],
            dispersion_key="disp_cov",
            library_size_key="library_cov",
        )
        mm_model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden={"rna": 17, "atac": 19},
            n_latent={"rna": 13, "atac": 11},
            n_layers=1,
            additive_background_modalities=["rna"],
            feature_scaling_modalities=["rna"],
        )

        sm_sd = sm_model.module.state_dict()
        mm_sd = mm_model.module.state_dict()

        # Shape match for comparable params
        pairs = [
            ("px_r_mu", "px_r_mu.rna"),
            ("px_r_log_sigma", "px_r_log_sigma.rna"),
            ("dispersion_prior_rate_raw", "dispersion_prior_rate_raw.rna"),
            ("additive_background", "additive_background.rna"),
            ("feature_scaling", "feature_scaling.rna"),
        ]
        for sm_key, mm_key in pairs:
            assert sm_sd[sm_key].shape == mm_sd[mm_key].shape, (
                f"Shape mismatch {sm_key} vs {mm_key}: {tuple(sm_sd[sm_key].shape)} vs {tuple(mm_sd[mm_key].shape)}"
            )

        # Deterministic params — exact match
        assert torch.allclose(sm_sd["dispersion_prior_rate_raw"], mm_sd["dispersion_prior_rate_raw.rna"])
        assert torch.allclose(sm_sd["feature_scaling"], mm_sd["feature_scaling.rna"])
        assert torch.allclose(sm_sd["px_r_log_sigma"], mm_sd["px_r_log_sigma.rna"])

        # Stochastic params — mean should be close
        assert torch.allclose(sm_sd["px_r_mu"].mean(), mm_sd["px_r_mu.rna"].mean(), atol=0.5)
        assert torch.allclose(sm_sd["additive_background"].mean(), mm_sd["additive_background.rna"].mean(), atol=0.1)

        # Encoder shapes match (same n_hidden, n_input for RNA)
        sm_enc = sm_sd["z_encoder.encoder.fc_layers.Layer 0.0.weight"]
        mm_enc = mm_sd["encoders.rna.encoder.fc_layers.Layer 0.0.weight"]
        assert sm_enc.shape == mm_enc.shape

        # Encoder weight init distributions similar
        assert torch.allclose(sm_enc.std(), mm_enc.std(), rtol=0.5)

    # --- Learnable modality scaling ---

    def test_learnable_modality_scaling_default_off(self, mdata):
        """No modality_scale_raw params when learnable_modality_scaling=False (default)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        assert len(module.modality_scale_raw) == 0
        assert len(module.modality_scale_init) == 0
        assert module.learnable_modality_scaling is False

    def test_learnable_modality_scaling_enabled(self, mdata):
        """Params created with correct init, model trains, metrics in history."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            learnable_modality_scaling=True,
            library_log_means_centering_sensitivity={"rna": 1.0, "atac": 0.2},
        )
        module = model.module
        assert module.learnable_modality_scaling is True
        assert set(module.modality_scale_raw.keys()) == {"rna", "atac"}
        assert set(module.modality_scale_init.keys()) == {"rna", "atac"}

        # Check init values: softplus(raw)/0.7 ≈ init_val
        for name, expected in [("rna", 1.0), ("atac", 0.2)]:
            actual = torch.nn.functional.softplus(module.modality_scale_raw[name]) / 0.7
            assert abs(actual.item() - expected) < 0.01, f"{name}: expected {expected}, got {actual.item()}"

        # Train and check history
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        history = model.history_
        assert "modality_scale_rna_train" in history
        assert "modality_scale_atac_train" in history

    def test_learnable_modality_scaling_init_from_centering(self, mdata):
        """When centering sensitivity provided, init values match; without → default 1.0."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        # With centering sensitivity
        model1 = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            learnable_modality_scaling=True,
            library_log_means_centering_sensitivity={"rna": 0.5, "atac": 0.3},
        )
        assert model1.module.modality_scale_init["rna"] == 0.5
        assert model1.module.modality_scale_init["atac"] == 0.3

        # Without centering sensitivity → default 1.0 for all
        model2 = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            learnable_modality_scaling=True,
        )
        assert model2.module.modality_scale_init["rna"] == 1.0
        assert model2.module.modality_scale_init["atac"] == 1.0


class TestWandBUtilities:
    """Tests for W&B utility functions (no-op when wandb_project is None)."""

    def test_setup_wandb_none_is_noop(self):
        """setup_wandb_logger returns (None, None) when wandb_project is None."""
        from regularizedvi.utils import setup_wandb_logger

        logger_list, run = setup_wandb_logger(wandb_project=None)
        assert logger_list is None
        assert run is None

    def test_log_figure_no_run_is_noop(self):
        """log_figure_to_wandb is safe when no W&B run is active."""
        import matplotlib.pyplot as plt

        from regularizedvi.utils import log_figure_to_wandb

        fig, _ax = plt.subplots()
        log_figure_to_wandb("test_figure", fig)  # should not raise
        plt.close(fig)

    def test_finish_wandb_no_run_is_noop(self):
        """finish_wandb is safe when no W&B run is active."""
        from regularizedvi.utils import finish_wandb

        finish_wandb()  # should not raise


class TestCoercePapermillParams:
    """Tests for coerce_papermill_params utility."""

    def test_coerce_bool_string_zero(self):
        """bool("0") bug: papermill -r injects "0" as string."""
        from regularizedvi.utils import coerce_papermill_params

        result = coerce_papermill_params(flag=("0", bool))
        assert result["flag"] is False

    def test_coerce_bool_string_one(self):
        from regularizedvi.utils import coerce_papermill_params

        result = coerce_papermill_params(flag=("1", bool))
        assert result["flag"] is True

    def test_coerce_bool_int_passthrough(self):
        from regularizedvi.utils import coerce_papermill_params

        assert coerce_papermill_params(flag=(1, bool))["flag"] is True
        assert coerce_papermill_params(flag=(0, bool))["flag"] is False

    def test_coerce_bool_already_bool(self):
        from regularizedvi.utils import coerce_papermill_params

        assert coerce_papermill_params(flag=(True, bool))["flag"] is True
        assert coerce_papermill_params(flag=(False, bool))["flag"] is False

    def test_coerce_float_string(self):
        from regularizedvi.utils import coerce_papermill_params

        result = coerce_papermill_params(beta=("5.0", float))
        assert result["beta"] == 5.0
        assert isinstance(result["beta"], float)

    def test_coerce_float_passthrough(self):
        from regularizedvi.utils import coerce_papermill_params

        result = coerce_papermill_params(beta=(5.0, float))
        assert result["beta"] == 5.0

    def test_coerce_str_or_none_string_none(self):
        from regularizedvi.utils import coerce_papermill_params

        assert coerce_papermill_params(proj=("None", "str_or_none"))["proj"] is None
        assert coerce_papermill_params(proj=("none", "str_or_none"))["proj"] is None

    def test_coerce_str_or_none_passthrough(self):
        from regularizedvi.utils import coerce_papermill_params

        assert coerce_papermill_params(proj=(None, "str_or_none"))["proj"] is None
        assert coerce_papermill_params(proj=("myproject", "str_or_none"))["proj"] == "myproject"

    def test_coerce_multiple_params(self):
        from regularizedvi.utils import coerce_papermill_params

        result = coerce_papermill_params(
            regularise_background=("0", bool),
            additive_bg_prior_beta=("5.0", float),
            wandb_project=("None", "str_or_none"),
        )
        assert result["regularise_background"] is False
        assert result["additive_bg_prior_beta"] == 5.0
        assert result["wandb_project"] is None

    def test_coerce_invalid_bool_raises(self):
        from regularizedvi.utils import coerce_papermill_params

        with pytest.raises(TypeError, match="Cannot coerce"):
            coerce_papermill_params(flag=("notabool", bool))
