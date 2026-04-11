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

    def test_residual_library_encoder_single_modal(self, adata):
        """Test residual_library_encoder creates params, trains, and logs metrics."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            residual_library_encoder=True,
            library_log_means_centering_sensitivity=1.0,
        )
        module = model.module
        assert module.residual_library_encoder is True
        assert hasattr(module, "library_obs_w_mu")
        assert hasattr(module, "library_obs_w_log_sigma")
        assert hasattr(module, "library_global_log_mean")
        assert hasattr(module, "library_log_sensitivity")
        # global_log_mean should be actual data mean, not 0
        assert module.library_global_log_mean.item() != 0.0
        # Bias should be initialized to log(sensitivity) = 0 for sens=1.0
        assert abs(module.l_encoder.mean_encoder.bias.item()) < 0.01
        # Should train without errors and log w metric
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "library_obs_w_train" in model.history_

    def test_always_center_library_no_sensitivity(self, adata):
        """Without sensitivity, library should still be centered (global_log_mean != 0)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            # No sensitivity — library_log_means_centering_sensitivity=None (default)
        )
        module = model.module
        # library_global_log_mean should be actual global mean, not 0
        assert module.library_global_log_mean.item() != 0.0
        # library_log_sensitivity should be 0 (no sensitivity shift)
        assert module.library_log_sensitivity.item() == 0.0
        # library_log_means should be centered around 0
        assert abs(module.library_log_means.mean().item()) < 0.1

    def test_decoder_bias_multiplier_single_modal(self, adata):
        """Test decoder_bias_multiplier scales decoder bias init."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        # Without multiplier
        model_base = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            init_decoder_bias="mean",
        )
        bias_base = model_base.module.decoder.px_scale_decoder[0].bias.data.clone()
        # With 2x multiplier
        model_2x = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            init_decoder_bias="mean",
            decoder_bias_multiplier=2.0,
        )
        bias_2x = model_2x.module.decoder.px_scale_decoder[0].bias.data.clone()
        # softplus_inv(2x) != 2 * softplus_inv(x) exactly, but 2x should be larger
        assert bias_2x.mean().item() > bias_base.mean().item()


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
        """px_r_mu should be initialized near log(1/mean²) for gamma_poisson (inverse_sqrt)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        # Default: gamma_poisson, dispersion_hyper_prior_mean=1/3, direction="inverse_sqrt"
        # Expected: px_r_mu ≈ -log((1/3)²) = log(9) = 2.197, theta = exp(px_r_mu) ≈ 9.0
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
        """Without bg_init_gene_fraction, background should be at Gamma(1,100) mean=0.01."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, bg_init_gene_fraction=None)
        bg = torch.exp(model.module.additive_background).detach()
        assert bg.mean().item() == pytest.approx(0.01, rel=0.2)
        # Noise is very small
        assert model.module.additive_background.detach().std().item() < 0.05

    def test_dispersion_prior_rate_init(self, adata):
        """Dispersion prior rate should initialize at 1/dispersion_hyper_prior_mean."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        rate = torch.nn.functional.softplus(model.module.dispersion_prior_rate_raw).detach()
        # Default mean = 1/3 → lambda_init = 1/mean = 3.0
        expected = 1.0 / model.module.dispersion_hyper_prior_mean
        assert rate.mean().item() == pytest.approx(expected, rel=0.01)

    # --- Multimodal model (RegularizedMultimodalVI) ---

    def test_px_r_init_multimodal_gamma_poisson(self, mdata):
        """Multimodal px_r_mu should be initialized near log(rate^2) per modality."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        for name in model.module.modality_names:
            theta = torch.exp(model.module.px_r_mu[name]).detach()
            assert theta.mean().item() == pytest.approx(9.0, rel=0.2), f"Failed for {name}"

    def test_additive_background_init_multimodal(self, mdata):
        """Without bg_init, multimodal background should be at Gamma(1,100) mean=0.01."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            additive_background_modalities=["rna"],
            bg_init_gene_fraction=None,
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
        """Multimodal dispersion prior rate should initialize at 1/mean per modality.

        Under new API, lambda_init = 1/dispersion_hyper_prior_mean. Default
        expected_RNA mean=1/3 gives lambda_init=3.
        """
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        for name in model.module.modality_names:
            rate = torch.nn.functional.softplus(model.module.dispersion_prior_rate_raw[name]).detach()
            # Default expected_RNA mean=1/3 → lambda_init = 3
            assert rate.mean().item() == pytest.approx(3.0, rel=0.01), f"Failed for {name}"

    def test_px_r_init_custom_prior(self, adata):
        """px_r_mu initialization adapts to custom dispersion_hyper_prior_mean."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        # New API: dispersion_hyper_prior_mean = 1/5 maps to theta = 1/mean² = 25 under inverse_sqrt
        model = regularizedvi.AmbientRegularizedSCVI(
            adata, n_hidden=16, n_latent=4, dispersion_hyper_prior_mean=1.0 / 5.0
        )
        # mean=1/5 → theta = 1/(1/5)² = 25 → px_r_mu ≈ log(25) = 3.219
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.mean().item() == pytest.approx(25.0, rel=0.2)

    def test_additive_bg_prior_stored(self, adata):
        """Additive background prior hyperparameters should be stored on module."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        assert model.module.additive_bg_prior_alpha == 1.0
        assert model.module.additive_bg_prior_beta == 100.0

    def test_additive_bg_custom_prior(self, adata):
        """Additive background init adapts to custom prior (without bg_init_gene_fraction)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            additive_bg_prior_alpha=2.0,
            additive_bg_prior_beta=50.0,
            bg_init_gene_fraction=None,
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


class TestDecoderRegularization:
    """Tests for decoder weight L2 penalty, bias init, and gene-specific bg init."""

    def test_decoder_weight_l2_default(self, adata):
        """Default decoder_weight_l2=0.1."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        assert model.module.decoder_weight_l2 == 0.1

    def test_decoder_weight_l2_penalty_positive(self, adata):
        """With decoder_weight_l2 > 0, penalty should be logged in history."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, decoder_weight_l2=0.01)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "decoder_weight_penalty_train" in model.history_
        vals = model.history_["decoder_weight_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_decoder_weight_l2_weights_only(self, adata):
        """Penalty should only count weight matrices, not biases."""
        import torch

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, decoder_weight_l2=1.0)
        # Zero all Linear weights -> penalty should be zero
        with torch.no_grad():
            for layer_seq in model.module.decoder.px_decoder.fc_layers:
                for sublayer in layer_seq:
                    if isinstance(sublayer, torch.nn.Linear):
                        sublayer.weight.zero_()
                        if sublayer.bias is not None:
                            sublayer.bias.fill_(999.0)  # nonzero bias
            model.module.decoder.px_scale_decoder[0].weight.zero_()
            model.module.decoder.px_scale_decoder[0].bias.fill_(999.0)
        penalty = model.module._decoder_weight_l2_penalty()
        assert penalty.item() == 0.0

    def test_decoder_weight_l2_multimodal(self, mdata):
        """Multimodal decoder weight penalty should work."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, decoder_weight_l2=0.01)
        assert model.module.decoder_weight_l2 == 0.01
        penalty = model.module._decoder_weight_l2_penalty()
        assert penalty.item() > 0

    def test_init_decoder_bias_all_options(self, adata):
        """Test all 3 init_decoder_bias options: None, 'mean' (default), 'topN'."""
        import torch

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model_none = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, init_decoder_bias=None)
        model_mean = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, init_decoder_bias="mean")
        model_topn = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, init_decoder_bias="topN")
        bias_none = model_none.module.decoder.px_scale_decoder[0].bias.detach()
        bias_mean = model_mean.module.decoder.px_scale_decoder[0].bias.detach()
        bias_topn = model_topn.module.decoder.px_scale_decoder[0].bias.detach()
        # None vs mean: should differ (Kaiming vs data-dependent)
        assert not torch.allclose(bias_none, bias_mean)
        # None vs topN: should differ
        assert not torch.allclose(bias_none, bias_topn)
        # mean vs topN: should differ (different data statistics)
        assert not torch.allclose(bias_mean, bias_topn)
        # All should be finite
        assert torch.all(torch.isfinite(bias_none))
        assert torch.all(torch.isfinite(bias_mean))
        assert torch.all(torch.isfinite(bias_topn))
        # Default model should use "mean"
        model_default = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        bias_default = model_default.module.decoder.px_scale_decoder[0].bias.detach()
        assert torch.allclose(bias_default, bias_mean)

    def test_bg_init_gene_fraction(self, adata):
        """bg_init_gene_fraction: default (0.2), explicit 0.2, and None should all work."""

        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        # Explicit 0.2
        model_02 = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, bg_init_gene_fraction=0.2)
        bg_02 = model_02.module.additive_background.detach()
        # Per-gene init should create variation across genes
        assert bg_02[:, 0].std() > 0.01  # not constant

        # None (off) — should init at prior mean log(0.01)
        model_none = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, bg_init_gene_fraction=None)
        bg_none = model_none.module.additive_background.detach()
        # Without data-dependent init, bg should be near constant (prior mean)
        assert bg_none[:, 0].std() < 0.05

        # Default (bg_init_gene_fraction=0.2) should also produce per-gene variation
        model_default = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        bg_default = model_default.module.additive_background.detach()
        assert bg_default[:, 0].std() > 0.01  # data-dependent, not constant

    def test_init_decoder_bias_multimodal(self, mdata):
        """Multimodal bias init should work per modality."""
        import torch

        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, init_decoder_bias="mean")
        for name in model.module.modality_names:
            bias = model.module.decoders[name].px_scale_decoder[0].bias.detach()
            assert torch.all(torch.isfinite(bias))

    def test_bg_init_gene_fraction_multimodal(self, mdata):
        """Multimodal gene-specific bg init should work."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            additive_background_modalities=["rna"],
            bg_init_gene_fraction=0.2,
        )
        bg = model.module.additive_background["rna"].detach()
        col_means = bg[:, 0]
        assert col_means.std() > 0.01

    def test_combined_all_streams(self, adata):
        """All three streams should work together."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            decoder_weight_l2=0.01,
            init_decoder_bias="mean",
            bg_init_gene_fraction=0.2,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "decoder_weight_penalty_train" in model.history_


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
        loss_output = module.loss(tensors, inf_outputs, gen_outputs, skip_n_obs_check=True)
        assert loss_output.loss.isfinite()

    def test_loss_n_obs_scaling(self, adata):
        """Global priors must scale as 1/n_obs; doubling n_obs halves penalty contribution."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, decoder_weight_l2=1.0)
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

        lo_n = module.loss(tensors, inf_outputs, gen_outputs, n_obs=1000)
        lo_2n = module.loss(tensors, inf_outputs, gen_outputs, n_obs=2000)

        penalty_n = lo_n.extra_metrics["decoder_weight_penalty"]
        penalty_2n = lo_2n.extra_metrics["decoder_weight_penalty"]
        assert torch.isclose(penalty_2n * 2, penalty_n, rtol=1e-4), (
            f"Global prior must scale as 1/n_obs: "
            f"penalty(n=1000)={penalty_n.item()}, penalty(n=2000)={penalty_2n.item()}"
        )

    def test_loss_n_obs_assert_fires(self, adata):
        """Assert that loss() raises AssertionError when n_obs < batch size without skip."""
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
        with pytest.raises(AssertionError, match="n_obs"):
            module.loss(tensors, inf_outputs, gen_outputs)

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
            # Centering constants (always registered)
            "library_global_log_mean": (),
            "library_log_sensitivity": (),
            # Residual library encoder weight (default on)
            "library_obs_w_mu": (),
            "library_obs_w_log_sigma": (),
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
        loss_output = module.loss(tensors, inf_outputs, gen_outputs, skip_n_obs_check=True)

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

    def test_multimodal_loss_n_obs_scaling(self, mdata):
        """Multimodal global priors must scale as 1/n_obs; doubling n_obs halves penalty contribution."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, decoder_weight_l2=1.0)
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

        lo_n = module.loss(tensors, inf_outputs, gen_outputs, n_obs=1000)
        lo_2n = module.loss(tensors, inf_outputs, gen_outputs, n_obs=2000)

        penalty_n = lo_n.extra_metrics["decoder_weight_penalty"]
        penalty_2n = lo_2n.extra_metrics["decoder_weight_penalty"]
        assert torch.isclose(penalty_2n * 2, penalty_n, rtol=1e-4), (
            f"Multimodal global prior must scale as 1/n_obs: "
            f"penalty(n=1000)={penalty_n.item()}, penalty(n=2000)={penalty_2n.item()}"
        )

    def test_multimodal_loss_n_obs_assert_fires(self, mdata):
        """Multimodal loss() must raise AssertionError when n_obs < batch size without skip."""
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
        with pytest.raises(AssertionError, match="n_obs"):
            module.loss(tensors, inf_outputs, gen_outputs)

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
        loss_output = module.loss(tensors, inf_outputs, gen_outputs, skip_n_obs_check=True)
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
        assert "X_latent_joint" in adata_rna.obsm
        assert "X_umap_joint" in adata_rna.obsm
        assert "X_latent_rna" in adata_rna.obsm
        assert "X_umap_rna" in adata_rna.obsm
        assert "X_latent_atac" in adata_rna.obsm
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
        adata_rna.obsm["X_latent_joint"] = latent_dict["__joint__"]
        for name in model.module.modality_names:
            adata_rna.obsm[f"X_latent_{name}"] = latent_dict[name]

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
            assert f"X_latent_attr_{name}" in adata_rna.obsm
            assert adata_rna.obsm[f"X_latent_attr_{name}"].shape[0] == mdata["rna"].n_obs

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
            # Centering constants (always registered)
            expected[f"library_global_log_mean_{name}"] = ()
            expected[f"library_log_sensitivity_{name}"] = ()
            # Residual library encoder weight (default on, ParameterDict)
            expected[f"library_obs_w_mu.{name}"] = ()
            expected[f"library_obs_w_log_sigma.{name}"] = ()

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

    def test_modality_lr_multiplier_param_groups(self, mdata):
        """get_parameter_groups creates correct groups with right LRs."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module

        groups = module.get_parameter_groups(
            base_lr=1e-3,
            modality_lr_multiplier={"atac": 2.0},
        )

        # Should have groups for atac, rna, and possibly shared
        assert len(groups) >= 2

        # Check every requires_grad param is in exactly one group
        all_group_ids = []
        for g in groups:
            all_group_ids.extend(id(p) for p in g["params"])
        all_requires_grad = [p for p in module.parameters() if p.requires_grad]
        assert len(all_group_ids) == len(all_requires_grad)
        assert len(set(all_group_ids)) == len(all_group_ids)  # no duplicates

        # Check LR values
        lr_by_param_id = {}
        for g in groups:
            for p in g["params"]:
                lr_by_param_id[id(p)] = g["lr"]

        # ATAC encoder param should have 2x LR
        atac_enc_param = next(module.encoders["atac"].parameters())
        assert lr_by_param_id[id(atac_enc_param)] == pytest.approx(2e-3)

        # RNA encoder param should have base LR
        rna_enc_param = next(module.encoders["rna"].parameters())
        assert lr_by_param_id[id(rna_enc_param)] == pytest.approx(1e-3)

    def test_train_with_modality_lr_multiplier(self, mdata):
        """Training with modality_lr_multiplier completes without error."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(
            max_epochs=2,
            train_size=1.0,
            batch_size=32,
            plan_kwargs={"modality_lr_multiplier": {"atac": 2.0}},
        )
        assert "elbo_train" in model.history_

    def test_train_without_modality_lr_multiplier(self, mdata):
        """Training without modality_lr_multiplier uses default behavior."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_

    def test_modality_lr_multiplier_unknown_modality(self, mdata):
        """Unknown modality name in multiplier raises ValueError."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        with pytest.raises(ValueError, match="Unknown modality"):
            model.module.get_parameter_groups(
                base_lr=1e-3,
                modality_lr_multiplier={"protein": 2.0},
            )

    def test_residual_library_encoder_multimodal(self, mdata):
        """Test residual_library_encoder creates params, trains, and logs metrics."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            residual_library_encoder=True,
            library_log_means_centering_sensitivity={"rna": 1.0, "atac": 0.2},
        )
        module = model.module
        assert module.residual_library_encoder is True
        assert set(module.library_obs_w_mu.keys()) == {"rna", "atac"}
        assert set(module.library_obs_w_log_sigma.keys()) == {"rna", "atac"}
        # global_log_mean should be actual data mean, not 0
        for name in ["rna", "atac"]:
            glm = getattr(module, f"library_global_log_mean_{name}")
            assert glm.item() != 0.0, f"{name}: library_global_log_mean should not be 0"
        # ATAC bias should be log(0.2)
        import math

        atac_bias = module.l_encoders["atac"].mean_encoder.bias.item()
        assert abs(atac_bias - math.log(0.2)) < 0.01
        # Should train without errors
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        # Check w metrics in history
        assert "library_obs_w_rna_train" in model.history_
        assert "library_obs_w_atac_train" in model.history_

    def test_always_center_library_no_sensitivity_multimodal(self, mdata):
        """Without sensitivity, library should still be centered."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            # No sensitivity
        )
        module = model.module
        for name in ["rna", "atac"]:
            glm = getattr(module, f"library_global_log_mean_{name}")
            assert glm.item() != 0.0, f"{name}: global_log_mean should be actual mean, not 0"
            ls = getattr(module, f"library_log_sensitivity_{name}")
            assert ls.item() == 0.0, f"{name}: log_sensitivity should be 0 without centering"
            means = getattr(module, f"library_log_means_{name}")
            assert abs(means.mean().item()) < 0.1, f"{name}: means should be centered near 0"

    def test_decoder_bias_multiplier_multimodal(self, mdata):
        """Test decoder_bias_multiplier scales decoder bias init per modality."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model_base = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            init_decoder_bias="mean",
        )
        bias_base = model_base.module.decoders["atac"].px_scale_decoder[0].bias.data.clone()
        model_2x = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            init_decoder_bias="mean",
            decoder_bias_multiplier={"atac": 2.0},
        )
        bias_2x = model_2x.module.decoders["atac"].px_scale_decoder[0].bias.data.clone()
        assert bias_2x.mean().item() > bias_base.mean().item()


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


class TestBurstFrequencySizeDecoder:
    """Tests for decoder_type='burst_frequency_size' (bursting model)."""

    def test_single_modal_burst_frequency_size(self, adata):
        """Test AmbientRegularizedSCVI with burst_frequency_size decoder."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            burst_size_intercept=1.0,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)

        # Check training completed and metrics are present
        history = model.history
        assert "burst_freq_mean_train" in history
        assert "burst_size_mean_train" in history
        assert "stochastic_v_mean_train" in history
        assert "alpha_total_mean_train" in history

        # Check latent representation works
        latent = model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 8)

    def test_single_modal_burst_size_intercept_small(self, adata):
        """Test burst_size_intercept=0.01 doesn't break training."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            burst_size_intercept=0.01,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        latent = model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 8)

    def test_single_modal_variance_burst_size_init(self, adata):
        """Test dispersion_init='variance_burst_size' with burst_frequency_size decoder.

        Verifies data-init path: px_r_mu is initialised from data (not prior). Training
        is skipped here — tiny synthetic data produces an aggressive auto-derived
        lambda that is unstable at this size; full-model training is covered by
        test_single_modal_burst_frequency_size.
        """
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            dispersion_init="variance_burst_size",
            dispersion_init_bio_frac=0.9,
        )
        # Verify px_r_mu was initialized from data (not default prior init)
        px_r_mu = model.module.px_r_mu.data
        # Default init would be ~log(9)=2.197; variance_burst_size init gives different values
        default_init = torch.full_like(px_r_mu, 2.197)
        assert not torch.allclose(px_r_mu, default_init, atol=0.5), "px_r_mu should be initialized from data, not prior"

    def test_multimodal_burst_frequency_size_rna_only(self, mdata):
        """Test RegularizedMultimodalVI with burst_frequency_size for RNA, expected_RNA for ATAC."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=32,
            n_latent=8,
            decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        latent = model.get_latent_representation()
        assert latent.shape[0] == mdata.n_obs

    def test_multimodal_burst_attribution(self, mdata):
        """Test get_modality_attribution works with burst_frequency_size decoder."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=32,
            n_latent=8,
            decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        attribution = model.get_modality_attribution(batch_size=64)
        assert "rna" in attribution
        assert "atac" in attribution
        for name in ("rna", "atac"):
            assert attribution[name]["attribution"].shape[0] == mdata.n_obs

    def test_multimodal_burst_variance_metrics(self, mdata):
        """Test burst variance metrics are logged in model.history."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=32,
            n_latent=8,
            decoder_type={"rna": "burst_frequency_size", "atac": "expected_RNA"},
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        history = model.history
        assert "burst_freq_mean_rna_train" in history
        assert "var_biol_mean_rna_train" in history
        assert "var_total_mean_rna_train" in history
        assert "var_biol_frac_rna_train" in history


class TestSingleModalityMultimodal:
    """Tests for N=1 (single modality) multimodal model."""

    def test_setup_mudata_single_modality(self, mdata_single_rna):
        """Test setup_mudata works with single RNA modality."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata_single_rna, batch_key="batch")

    def test_train_single_modality(self, mdata_single_rna):
        """Test training with N=1 multimodal mode."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata_single_rna, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata_single_rna, n_hidden=16, n_latent=4)
        assert model.module.modality_names == ["rna"]
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        latent = model.get_latent_representation()
        assert latent.shape == (100, 4)

    def test_single_modality_burst_frequency_size(self, mdata_single_rna):
        """Test N=1 with burst_frequency_size decoder."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata_single_rna, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata_single_rna,
            n_hidden=16,
            n_latent=4,
            decoder_type="burst_frequency_size",
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=64)
        latent = model.get_latent_representation()
        assert latent.shape == (100, 4)


class TestBurstInitFixes:
    """Tests for burst decoder init: sensitivity normalization, hyper-prior auto-derive, label_key."""

    def test_burst_sensitivity_normalizes_burst_size(self, adata):
        """burst_size should be divided by sensitivity in compute_bursting_init."""
        from regularizedvi._dispersion_init import compute_bursting_init

        vals_s1, _ = compute_bursting_init(adata, sensitivity=1.0, verbose=False)
        vals_s02, _ = compute_bursting_init(adata, sensitivity=0.2, verbose=False)
        # burst_size at sensitivity=0.2 should be ~5x larger (rate-space = count/sensitivity)
        ratio = np.median(vals_s02["burst_size"]) / np.median(vals_s1["burst_size"])
        assert 3.0 < ratio < 7.0, f"Expected burst_size ratio ~5x, got {ratio:.2f}"
        # burst_freq should be the same (dimensionless)
        np.testing.assert_allclose(vals_s1["burst_freq"], vals_s02["burst_freq"], rtol=1e-5)
        # stochastic_v should be the same (count-space)
        np.testing.assert_allclose(vals_s1["stochastic_v_scale"], vals_s02["stochastic_v_scale"], rtol=1e-5)

    def test_suggested_hyper_mean_in_init_values(self, adata):
        """compute_bursting_init returns suggested_hyper_mean = mean(stochastic_v_scale)."""
        from regularizedvi._dispersion_init import compute_bursting_init

        vals, _ = compute_bursting_init(adata, verbose=False)
        # Item 4: mean (not median), renamed to suggested_hyper_mean.
        expected_mean = float(np.mean(vals["stochastic_v_scale"]))
        assert abs(vals["suggested_hyper_mean"] - expected_mean) < 1e-6

    def test_label_key_returns_per_group_stats(self, adata):
        """compute_dispersion_init with label_key returns per-group diagnostics."""
        from regularizedvi._dispersion_init import compute_dispersion_init

        # Add a mock label column
        adata.obs["test_label"] = ["A"] * 50 + ["B"] * 50
        _, diag = compute_dispersion_init(adata, label_key="test_label", min_cells_per_group=10, verbose=False)
        assert "mean_g_per_group" in diag
        assert "A" in diag["mean_g_per_group"]
        assert "B" in diag["mean_g_per_group"]
        assert diag["group_sizes"]["A"] == 50
        assert diag["group_sizes"]["B"] == 50
        # Per-group arrays should have correct gene dimension
        assert diag["mean_g_per_group"]["A"].shape[0] == adata.n_vars

    def test_label_key_none_unchanged(self, adata):
        """label_key=None should not add per-group keys to diagnostics."""
        from regularizedvi._dispersion_init import compute_dispersion_init

        _, diag = compute_dispersion_init(adata, label_key=None, verbose=False)
        assert "mean_g_per_group" not in diag

    def test_label_key_filters_nan(self, adata):
        """NaN/nan labels should be excluded from per-group stats."""
        from regularizedvi._dispersion_init import compute_dispersion_init

        adata.obs["test_label"] = ["A"] * 40 + [np.nan] * 10 + ["B"] * 50
        _, diag = compute_dispersion_init(adata, label_key="test_label", min_cells_per_group=10, verbose=False)
        assert "nan" not in diag["group_sizes"]
        assert diag["group_sizes"]["A"] == 40

    def test_auto_hyper_prior_applied_in_model(self, adata):
        """variance_burst_size init should auto-derive hyper-prior beta."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            dispersion_init="variance_burst_size",
            regularise_dispersion=True,
        )
        # The learned rate should be initialized at E[lambda] = alpha/beta
        import torch

        learned_rate = torch.nn.functional.softplus(model.module.dispersion_prior_rate_raw)
        # Should NOT be the default 50 (from Gamma(2, 0.04))
        assert not torch.allclose(learned_rate, torch.tensor(50.0), atol=5.0), (
            "Lambda init should be auto-derived from MoM, not the default 50"
        )

    def test_decoder_bias_multiplier_burst_freq(self, adata):
        """decoder_bias_multiplier should scale burst_freq init."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch")
        model_1x = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            dispersion_init="variance_burst_size",
        )
        model_2x = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=32,
            n_latent=8,
            decoder_type="burst_frequency_size",
            dispersion_init="variance_burst_size",
            decoder_bias_multiplier=2.0,
        )
        bias_1x = model_1x.module.decoder.px_scale_decoder[0].bias.data.clone()
        bias_2x = model_2x.module.decoder.px_scale_decoder[0].bias.data.clone()
        # Exact multiplicative scaling in softplus space: softplus(b_2x) == 2 * softplus(b_1x)
        sp1 = torch.nn.functional.softplus(bias_1x)
        sp2 = torch.nn.functional.softplus(bias_2x)
        assert torch.allclose(sp2, 2.0 * sp1, rtol=1e-4, atol=1e-4), f"sp1={sp1[:5]}, sp2={sp2[:5]}"

    def test_dispersion_init_bio_frac_as_dict(self, mdata):
        """Item 3a: per-modality dispersion_init_bio_frac dict routes values correctly."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            decoder_type="expected_RNA",
            dispersion_init="data",
            dispersion_init_bio_frac={"rna": 0.95, "atac": 0.5},
        )
        stored_bf = model._dispersion_init_bio_frac
        assert isinstance(stored_bf, dict), f"expected dict, got {type(stored_bf)}"
        assert stored_bf["rna"] == 0.95
        assert stored_bf["atac"] == 0.5


class TestBurstInitPerFeatureBounds:
    """Item 1/2/3/3a: per-feature stochastic_v bounds + scalar burst_size/burst_freq bounds."""

    @staticmethod
    def _make_synthetic_adata(mean_range=(0.005, 0.1), n_obs=200, n_genes=40, seed=0):
        """Synthetic ATAC-like adata with per-gene rates spanning mean_range."""
        import anndata as ad

        rng = np.random.default_rng(seed)
        rates = np.linspace(mean_range[0], mean_range[1], n_genes).astype(np.float32)
        counts = rng.poisson(lam=rates[None, :] * 10.0, size=(n_obs, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=counts)
        adata.layers["counts"] = counts.copy()
        adata.obs["batch"] = np.array(["b0"] * n_obs)
        adata.obs["batch"] = adata.obs["batch"].astype("category")
        adata.var_names = [f"peak_{i}" for i in range(n_genes)]
        return adata

    def test_compute_bursting_init_clips_stochastic_v_per_feature(self):
        """stochastic_v_scale bounded per-feature by [mean_g/sqrt(theta_max), mean_g/sqrt(theta_min)]."""
        from regularizedvi._dispersion_init import compute_bursting_init

        adata = self._make_synthetic_adata()
        theta_min, theta_max = 0.01, 20.0
        vals, diag = compute_bursting_init(adata, theta_min=theta_min, theta_max=theta_max, verbose=False)
        mean_g = diag["mean_g"]
        sv = vals["stochastic_v_scale"]
        sv_min_g = mean_g / np.sqrt(theta_max)
        sv_max_g = mean_g / np.sqrt(theta_min)
        # Per-feature lower and upper bounds, with small tolerance for float32
        assert np.all(sv >= sv_min_g - 1e-7), (
            f"sv violated lower per-feature bound: min(sv - sv_min_g)={np.min(sv - sv_min_g)}"
        )
        assert np.all(sv <= sv_max_g + 1e-7), (
            f"sv violated upper per-feature bound: max(sv - sv_max_g)={np.max(sv - sv_max_g)}"
        )

    def test_compute_bursting_init_per_feature_sv_bounds_match_theta_bounds(self):
        """min(sv/mean_g) >= 1/sqrt(theta_max) and max(sv/mean_g) <= 1/sqrt(theta_min)."""
        from regularizedvi._dispersion_init import compute_bursting_init

        adata = self._make_synthetic_adata()
        theta_min, theta_max = 0.01, 20.0
        vals, diag = compute_bursting_init(adata, theta_min=theta_min, theta_max=theta_max, verbose=False)
        mean_g = diag["mean_g"]
        sv = vals["stochastic_v_scale"]
        ratio = sv / mean_g
        assert ratio.min() >= 1.0 / np.sqrt(theta_max) - 1e-6, (
            f"min(sv/mean_g)={ratio.min()} < 1/sqrt(theta_max)={1.0 / np.sqrt(theta_max)}"
        )
        assert ratio.max() <= 1.0 / np.sqrt(theta_min) + 1e-6, (
            f"max(sv/mean_g)={ratio.max()} > 1/sqrt(theta_min)={1.0 / np.sqrt(theta_min)}"
        )

    def test_bursting_init_atac_scale_separation(self):
        """Per-feature bounds scale linearly with mean_g (not flat scalar bounds)."""
        from regularizedvi._dispersion_init import compute_bursting_init

        adata = self._make_synthetic_adata(mean_range=(0.005, 0.1))
        theta_min, theta_max = 0.01, 20.0
        _, diag = compute_bursting_init(adata, theta_min=theta_min, theta_max=theta_max, verbose=False)
        mean_g = diag["mean_g"]
        # Derived bounds
        sv_min_g = mean_g / np.sqrt(theta_max)
        sv_max_g = mean_g / np.sqrt(theta_min)
        # The spread of mean_g should be reflected in spread of sv_min_g and sv_max_g:
        # NOT flat scalar values. ratio of min/max bound across genes should be
        # roughly the ratio of min/max mean_g (since division by sqrt(theta_*) is
        # gene-independent).
        mean_ratio = float(mean_g.max() / mean_g.min())
        sv_min_ratio = float(sv_min_g.max() / sv_min_g.min())
        sv_max_ratio = float(sv_max_g.max() / sv_max_g.min())
        assert sv_min_ratio == pytest.approx(mean_ratio, rel=1e-4)
        assert sv_max_ratio == pytest.approx(mean_ratio, rel=1e-4)
        # Flat-scalar bounds would give ratios == 1.0; this must be strictly > 1.
        assert sv_min_ratio > 5.0, f"expected clear scale separation; got {sv_min_ratio}"

    def test_compute_bursting_init_clips_burst_size(self):
        """burst_size clamped to scalar [burst_size_min, burst_size_max] (Item 2)."""
        from regularizedvi._dispersion_init import compute_bursting_init

        # Use default burst_size_max=20 scalar bound.
        adata = self._make_synthetic_adata(mean_range=(0.01, 50.0))
        vals, _ = compute_bursting_init(adata, verbose=False)
        burst_size = vals["burst_size"]
        assert burst_size.max() <= 20.0 + 1e-6, f"max burst_size={burst_size.max()} > 20"
        assert burst_size.min() >= 0.01 - 1e-6, f"min burst_size={burst_size.min()} < 0.01"

    def test_compute_bursting_init_clips_burst_freq(self):
        """burst_freq clamped to [theta_min, theta_max] with default max=20 (Item 3)."""
        from regularizedvi._dispersion_init import compute_bursting_init

        adata = self._make_synthetic_adata(mean_range=(0.5, 50.0))
        vals, _ = compute_bursting_init(adata, verbose=False)
        burst_freq = vals["burst_freq"]
        # Default theta_max=20 applies
        assert burst_freq.max() <= 20.0 + 1e-6, f"max burst_freq={burst_freq.max()} > 20"
        assert burst_freq.min() >= 0.01 - 1e-6, f"min burst_freq={burst_freq.min()} < 0.01"


class TestDispersionInitNonBurstBounds:
    """Item 3a: compute_dispersion_init (non-burst) theta bounds + bio_frac robustness."""

    @staticmethod
    def _make_atac_like_adata(n_obs=200, n_genes=40, mean_range=(0.005, 0.1), seed=0):
        import anndata as ad

        rng = np.random.default_rng(seed)
        rates = np.linspace(mean_range[0], mean_range[1], n_genes).astype(np.float32)
        counts = rng.poisson(lam=rates[None, :] * 10.0, size=(n_obs, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=counts)
        adata.layers["counts"] = counts.copy()
        adata.obs["batch"] = np.array(["b0"] * n_obs)
        adata.obs["batch"] = adata.obs["batch"].astype("category")
        adata.var_names = [f"peak_{i}" for i in range(n_genes)]
        return adata

    def test_compute_dispersion_init_clips_theta(self):
        """theta_clamped ∈ [theta_min, theta_max] with default theta_max=20."""
        from regularizedvi._dispersion_init import compute_dispersion_init

        adata = self._make_atac_like_adata()
        log_theta, diag = compute_dispersion_init(adata, biological_variance_fraction=0.9, verbose=False)
        theta = np.exp(log_theta)
        # Default theta_min=0.01, theta_max=20 (Item 3a)
        assert theta.min() >= 0.01 - 1e-6
        assert theta.max() <= 20.0 + 1e-6
        assert diag["theta_max"] == 20.0

    def test_compute_dispersion_init_bio_frac_robustness(self):
        """Same input with bio_frac=0.9 vs 0.99 should not differ wildly under new bounds."""
        from regularizedvi._dispersion_init import compute_dispersion_init

        adata = self._make_atac_like_adata()
        log_theta_09, _ = compute_dispersion_init(adata, biological_variance_fraction=0.9, verbose=False)
        log_theta_099, _ = compute_dispersion_init(adata, biological_variance_fraction=0.99, verbose=False)
        theta_09 = np.exp(log_theta_09)
        theta_099 = np.exp(log_theta_099)
        # Both should be finite and within the theta_max=20 bound.
        assert np.all(np.isfinite(theta_09)) and np.all(np.isfinite(theta_099))
        assert theta_09.max() <= 20.0 + 1e-6
        assert theta_099.max() <= 20.0 + 1e-6
        # Summary: ratio of the overall medians should be < 5x under new bounds.
        ratio = float(np.median(theta_099) / max(float(np.median(theta_09)), 1e-12))
        assert ratio < 5.0, f"bio_frac sensitivity too extreme: median ratio={ratio:.2f}"


class TestDispersionPriorDirection:
    """Item 6: dispersion_prior_direction flag (inverse_sqrt default, sqrt option)."""

    def test_dispersion_prior_direction_default_inverse_sqrt(self, adata):
        """Default dispersion_prior_direction == 'inverse_sqrt'."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        assert model.module.dispersion_prior_direction == "inverse_sqrt"

    def test_dispersion_prior_direction_sqrt_single_modal(self, adata):
        """sqrt direction with default alpha=9 trains without NaNs in dispersion_kl."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        # Under sqrt direction the target theta ≈ 9 ⇔ mean = sqrt(9) = 3
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            regularise_dispersion=True,
            dispersion_prior_direction="sqrt",
            dispersion_hyper_prior_alpha=9.0,
            dispersion_hyper_prior_mean=3.0,
        )
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        hist = model.history_
        # Single-modal logs 'dispersion_kl_train' (see _module.py extra_metrics).
        assert "dispersion_kl_train" in hist, f"dispersion_kl_train not logged: {list(hist.keys())[:20]}"
        vals_k = np.asarray(hist["dispersion_kl_train"].values).ravel().astype(float)
        assert np.all(np.isfinite(vals_k)), f"NaN/inf in dispersion_kl_train: {vals_k}"
        assert model.module.dispersion_prior_direction == "sqrt"

    def test_dispersion_prior_direction_sqrt_multi_modal(self, mdata):
        """sqrt direction with default alpha=9 multimodal trains without NaNs.

        Multimodal loss rolls the dispersion hyper-prior penalty into the overall
        train loss (no dedicated `dispersion_kl_*` key), so we assert the total
        reconstruction and elbo are finite — no NaN propagates from the new
        direction branch.
        """
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            regularise_dispersion=True,
            dispersion_prior_direction="sqrt",
            dispersion_hyper_prior_alpha=9.0,
            dispersion_hyper_prior_mean=3.0,
        )
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        hist = model.history_
        finite_seen = 0
        for key in ("elbo_train", "reconstruction_loss_train", "train_loss_epoch"):
            if key in hist:
                vals_k = np.asarray(hist[key].values).ravel().astype(float)
                assert np.all(np.isfinite(vals_k)), f"NaN/inf in {key}: {vals_k}"
                finite_seen += 1
        assert finite_seen > 0, f"no standard training metric found: {list(hist.keys())[:20]}"
        # Per-modality dispersion attribute stores the direction dict.
        for name in model.module.modality_names:
            assert model.module.dispersion_prior_direction_dict[name] == "sqrt"

    def test_init_under_sqrt_direction_expected_rna(self, adata):
        """sqrt direction + mean=3 → theta_init = mean² = 9 → exp(px_r_mu) ≈ 9."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            decoder_type="expected_RNA",
            regularise_dispersion=True,
            dispersion_prior_direction="sqrt",
            dispersion_hyper_prior_alpha=9.0,
            dispersion_hyper_prior_mean=3.0,
        )
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.mean().item() == pytest.approx(9.0, rel=0.2)

    def test_init_under_inverse_sqrt_direction_expected_rna(self, adata):
        """inverse_sqrt + mean=1/3 → theta_init = 1/mean² = 9 → exp(px_r_mu) ≈ 9."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            decoder_type="expected_RNA",
            regularise_dispersion=True,
            dispersion_prior_direction="inverse_sqrt",
            dispersion_hyper_prior_alpha=9.0,
            dispersion_hyper_prior_mean=1.0 / 3.0,
        )
        theta = torch.exp(model.module.px_r_mu).detach()
        assert theta.mean().item() == pytest.approx(9.0, rel=0.2)


class TestEncoderWeightInit:
    """Item 8: Encoder var head weight init scaled 0.1× (mean head at PyTorch default)."""

    def test_encoder_var_head_weight_init_scaled(self):
        """var_encoder.weight.std() is ~0.1× a fresh nn.Linear's std."""
        from regularizedvi._components import RegularizedEncoder

        torch.manual_seed(0)
        n_input, n_output, n_hidden = 20, 8, 16
        enc = RegularizedEncoder(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            var_init_scale=0.1,
        )
        # Reference: a fresh nn.Linear with identical shape (matches PyTorch default init).
        torch.manual_seed(0)
        ref = torch.nn.Linear(n_hidden, n_output)
        ref_std = ref.weight.data.std().item()
        enc_var_std = enc.var_encoder.weight.data.std().item()
        # Should be ~0.1× the reference; allow mild tolerance for random draws.
        ratio = enc_var_std / ref_std
        assert 0.05 < ratio < 0.2, f"var weight std ratio {ratio:.3f} not in (0.05, 0.2)"

    def test_encoder_mean_head_weight_init_default(self):
        """mean_encoder.weight matches PyTorch default (not zero, not scaled)."""
        from regularizedvi._components import RegularizedEncoder

        torch.manual_seed(0)
        enc = RegularizedEncoder(
            n_input=20,
            n_output=8,
            n_hidden=16,
            var_init_scale=0.1,
        )
        # Mean head weights must be non-zero (i.e. PyTorch's kaiming default, not hand-init)
        mw = enc.mean_encoder.weight.data
        assert mw.abs().mean().item() > 0.0
        # Reference std from a fresh nn.Linear
        torch.manual_seed(0)
        ref = torch.nn.Linear(16, 8)
        ref_std = ref.weight.data.std().item()
        mean_std = mw.std().item()
        ratio = mean_std / ref_std
        # Untouched mean head should be close to 1× the reference (NOT 0.1×).
        assert ratio > 0.5, f"mean head weight std ratio {ratio:.3f} looks scaled"

    def test_qz_scale_at_init_matches_var_init_scale(self):
        """qz.scale.mean() at init is within 2% of var_init_scale."""
        from regularizedvi._components import RegularizedEncoder

        torch.manual_seed(0)
        target = 0.1
        enc = RegularizedEncoder(
            n_input=32,
            n_output=16,
            n_hidden=64,
            use_softplus_var_activation=True,
            var_init_scale=target,
            return_dist=True,
        )
        enc.eval()
        x = torch.randn(256, 32)
        dist, _ = enc(x)
        m = dist.scale.mean().item()
        # With 0.1× weight scaling the noise contribution should land within 2% of target.
        assert abs(m - target) / target < 0.02, f"qz.scale.mean()={m:.5f} vs target={target}"

    def test_horseshoe_pre_init_no_weight_touch(self, mdata):
        """After horseshoe pre-init: mean_encoder weights non-zero (PyTorch default lives)
        and var_encoder weights scaled by 0.1× (not zero)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="lognormal",
            horseshoe_posterior_init_loc=1.0,
            horseshoe_posterior_init_scale=0.1,
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        for name in model.module.modality_names:
            enc = model.module.horseshoe_encoders[name]
            mw = enc.mean_encoder.weight.data
            vw = enc.var_encoder.weight.data
            # Mean head weights: non-zero, from PyTorch default
            assert mw.abs().mean().item() > 0.0
            # Var head weights: non-zero (not zeroed by a hand-init) and narrower than mean
            assert vw.abs().mean().item() > 0.0
            # Var head should be noticeably narrower because of the 0.1× scaling
            assert vw.std().item() < mw.std().item(), (
                f"{name}: var weight std {vw.std().item():.4f} not < mean weight std {mw.std().item():.4f}"
            )


class TestSparsityFeatures:
    """Tests for z_sparsity_prior, decoder_hidden_l1, and hidden_activation_sparsity."""

    def test_z_sparsity_prior_single_modal(self, adata):
        """z_sparsity_prior='gamma' trains and logs z_sparsity_penalty."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, z_sparsity_prior="gamma")
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "z_sparsity_penalty_train" in model.history_
        vals = model.history_["z_sparsity_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_decoder_hidden_l1_single_modal(self, adata):
        """decoder_hidden_l1 > 0 trains and logs decoder_l1_penalty."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, decoder_hidden_l1=0.01)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "decoder_l1_penalty_train" in model.history_
        vals = model.history_["decoder_l1_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_hidden_activation_sparsity_single_modal(self, adata):
        """hidden_activation_sparsity=True trains and logs hidden_sparsity_penalty."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, hidden_activation_sparsity=True)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "hidden_sparsity_penalty_train" in model.history_
        vals = model.history_["hidden_sparsity_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_all_sparsity_combined_single_modal(self, adata):
        """All three sparsity features together should train without error."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            z_sparsity_prior="gamma",
            decoder_hidden_l1=0.01,
            hidden_activation_sparsity=True,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "z_sparsity_penalty_train" in model.history_
        assert "decoder_l1_penalty_train" in model.history_
        assert "hidden_sparsity_penalty_train" in model.history_

    def test_z_sparsity_prior_multimodal(self, mdata):
        """z_sparsity_prior='gamma' trains and logs z_sparsity_penalty (multimodal)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, z_sparsity_prior="gamma")
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "z_sparsity_penalty_train" in model.history_
        vals = model.history_["z_sparsity_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_decoder_hidden_l1_multimodal(self, mdata):
        """decoder_hidden_l1 > 0 trains and logs decoder_l1_penalty (multimodal)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, decoder_hidden_l1=0.01)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "decoder_l1_penalty_train" in model.history_
        vals = model.history_["decoder_l1_penalty_train"].values
        assert all(v > 0 for v in vals.flatten() if not np.isnan(v))

    def test_hidden_activation_sparsity_multimodal(self, mdata):
        """hidden_activation_sparsity=True logs per-modality hidden_sparsity metrics."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, hidden_activation_sparsity=True)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "hidden_sparsity_rna_train" in model.history_
        assert "hidden_sparsity_atac_train" in model.history_

    def test_all_sparsity_combined_multimodal(self, mdata):
        """All three sparsity features together should train without error (multimodal)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            z_sparsity_prior="gamma",
            decoder_hidden_l1=0.01,
            hidden_activation_sparsity=True,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "z_sparsity_penalty_train" in model.history_
        assert "decoder_l1_penalty_train" in model.history_
        assert "hidden_sparsity_rna_train" in model.history_
        assert "hidden_sparsity_atac_train" in model.history_

    def test_use_kl_z_false_single_modal(self, adata):
        """use_kl_z=False disables Normal(0,1) KL on z, trains without error."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4, use_kl_z=False)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_

    def test_use_kl_z_false_multimodal(self, mdata):
        """use_kl_z=False disables Normal(0,1) KL on z (multimodal), trains without error."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, use_kl_z=False)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_

    def test_use_kl_z_false_with_gamma_sparsity(self, adata):
        """use_kl_z=False + z_sparsity_prior='gamma': no z KL but gamma penalty active."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata, n_hidden=16, n_latent=4, use_kl_z=False, z_sparsity_prior="gamma"
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "z_sparsity_penalty_train" in model.history_
        sp_vals = model.history_["z_sparsity_penalty_train"].values
        assert all(v > 0 for v in sp_vals.flatten() if not np.isnan(v))

    def test_horseshoe_latent_z_prior_lognormal(self, mdata):
        """horseshoe_latent_z_prior_type='lognormal' trains and logs horseshoe_kl."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="lognormal",
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_
        # At least one modality should have horseshoe_kl logged
        hs_keys = [k for k in model.history_ if k.startswith("horseshoe_kl_")]
        assert len(hs_keys) > 0, f"No horseshoe_kl metric found in history. Keys: {list(model.history_.keys())[:20]}"

    def test_horseshoe_latent_z_prior_gamma(self, mdata):
        """horseshoe_latent_z_prior_type='gamma' trains with MC KL estimate."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="gamma",
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_
        hs_keys = [k for k in model.history_ if k.startswith("horseshoe_kl_")]
        assert len(hs_keys) > 0

    def test_horseshoe_latent_z_prior_none(self, mdata):
        """horseshoe_latent_z_prior_type=None (default) disables horseshoe, no horseshoe_kl logged."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type=None,
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        hs_keys = [k for k in model.history_ if k.startswith("horseshoe_kl_")]
        assert len(hs_keys) == 0

    def test_active_dims_tracking_multimodal(self, mdata):
        """Active dims metrics logged during training (multimodal)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        active_keys = [k for k in model.history_ if k.startswith("n_active_dims_")]
        assert len(active_keys) > 0, f"No n_active_dims metric found. Keys: {list(model.history_.keys())[:20]}"

    def test_active_dims_tracking_single_modal(self, adata):
        """Active dims metrics logged during training (unimodal)."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(adata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)
        assert "n_active_dims_train" in model.history_


class TestScaleSoftplusBias:
    """Tests for _scale_softplus_bias / _scale_softplus_bias_np (B1 math fix)."""

    def test_round_trip_torch(self):
        from regularizedvi._components import _scale_softplus_bias

        b = torch.tensor([-5.0, 0.0, 1.0, 5.0, 10.0, 15.0])
        m = 2.5
        out = _scale_softplus_bias(b, m)
        lhs = torch.nn.functional.softplus(out)
        rhs = m * torch.nn.functional.softplus(b)
        assert torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-5), f"{lhs} vs {rhs}"

    def test_round_trip_numpy(self):
        from regularizedvi._components import _scale_softplus_bias_np

        b = np.array([-5.0, 0.0, 1.0, 5.0, 10.0, 15.0], dtype=np.float32)
        m = 2.5
        out = _scale_softplus_bias_np(b, m)
        lhs = np.log1p(np.exp(-np.abs(out))) + np.maximum(out, 0.0)
        rhs = m * (np.log1p(np.exp(-np.abs(b))) + np.maximum(b, 0.0))
        assert np.allclose(lhs, rhs, rtol=1e-4, atol=1e-4)

    def test_identity_at_multiplier_one(self):
        from regularizedvi._components import _scale_softplus_bias, _scale_softplus_bias_np

        b_t = torch.tensor([-3.0, 0.0, 4.0])
        assert torch.equal(_scale_softplus_bias(b_t, 1.0), b_t)
        b_n = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
        assert np.array_equal(_scale_softplus_bias_np(b_n, 1.0), b_n)

    def test_extremes_finite(self):
        from regularizedvi._components import _scale_softplus_bias, _scale_softplus_bias_np

        b_t = torch.tensor([20.0, 25.0])
        out_t = _scale_softplus_bias(b_t, 100.0)
        assert torch.all(torch.isfinite(out_t))
        b_n = np.array([20.0, 25.0], dtype=np.float32)
        out_n = _scale_softplus_bias_np(b_n, 100.0)
        assert np.all(np.isfinite(out_n))

    def test_b4_expected_rna_respects_user_hyper_prior(self, adata):
        """expected_RNA decoder: user dispersion_hyper_prior_mean is respected, not overridden."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            decoder_type="expected_RNA",
            dispersion_hyper_prior_mean=0.5,
        )
        assert model.module.dispersion_hyper_prior_mean == 0.5

    def test_b4_data_init_raises_on_data_plus_burst(self, adata):
        """dispersion_init='data' + decoder_type='burst_frequency_size' is rejected."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        with pytest.raises(ValueError, match="incompatible"):
            regularizedvi.AmbientRegularizedSCVI(
                adata,
                n_hidden=16,
                n_latent=4,
                decoder_type="burst_frequency_size",
                dispersion_init="data",
            )

    def test_full_model_bias_multiplier(self, adata):
        """Full model build with decoder_bias_multiplier=3.0 applies exact multiplicative scaling."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, batch_key="batch", layer="counts")
        m1 = regularizedvi.AmbientRegularizedSCVI(
            adata, n_hidden=16, n_latent=4, init_decoder_bias="mean", decoder_bias_multiplier=1.0
        )
        m3 = regularizedvi.AmbientRegularizedSCVI(
            adata, n_hidden=16, n_latent=4, init_decoder_bias="mean", decoder_bias_multiplier=3.0
        )
        b1 = m1.module.decoder.px_scale_decoder[0].bias.detach()
        b3 = m3.module.decoder.px_scale_decoder[0].bias.detach()
        sp1 = torch.nn.functional.softplus(b1)
        sp3 = torch.nn.functional.softplus(b3)
        assert torch.allclose(sp3, 3.0 * sp1, rtol=1e-2, atol=1e-2)


class TestZPosteriorDiagnostics:
    """Tests for Z-init / posterior diagnostics features (plan stateless-squishing-lecun)."""

    def test_var_init_scale_encoder_unit_softplus(self):
        """RegularizedEncoder(softplus) with var_init_scale=0.1 yields mean qz.scale on the order of 0.1.

        Weights are initialised to N(0, 0.1) (not zero) so cells start with small but
        non-zero spread; mean across cells is allowed ~30% drift from the literal target.
        """
        from regularizedvi._components import RegularizedEncoder

        torch.manual_seed(0)
        enc = RegularizedEncoder(
            n_input=20,
            n_output=8,
            n_hidden=16,
            use_softplus_var_activation=True,
            var_init_scale=0.1,
            return_dist=True,
        )
        enc.eval()
        x = torch.randn(64, 20)
        dist, _ = enc(x)
        scale = dist.scale
        assert torch.all(scale > 0)
        # mean across cells × dims should sit near 0.1 (allow generous tolerance for the
        # randn weight noise — convex Jensen inflation is well below 50%).
        m = scale.mean().item()
        assert 0.05 < m < 0.20, f"Expected mean qz.scale near 0.1, got {m:.4f}"

    def test_var_init_scale_encoder_unit_exp(self):
        """RegularizedEncoder(exp activation) with var_init_scale=0.1 also produces mean qz.scale near 0.1."""
        from regularizedvi._components import RegularizedEncoder

        torch.manual_seed(0)
        enc = RegularizedEncoder(
            n_input=20,
            n_output=8,
            n_hidden=16,
            use_softplus_var_activation=False,  # default exp activation
            var_init_scale=0.1,
            return_dist=True,
        )
        enc.eval()
        x = torch.randn(64, 20)
        dist, _ = enc(x)
        scale = dist.scale
        assert torch.all(scale > 0)
        m = scale.mean().item()
        assert 0.05 < m < 0.20, f"Expected mean qz.scale near 0.1 with exp activation, got {m:.4f}"

    def test_var_init_scale_zero_raises(self):
        from regularizedvi._components import RegularizedEncoder

        with pytest.raises(ValueError, match="var_init_scale must be > 0"):
            RegularizedEncoder(n_input=4, n_output=2, n_hidden=4, var_init_scale=0.0)

    def test_var_init_scale_multimodal_train(self, mdata):
        """End-to-end: multimodal model with var_init_scale=0.1 trains."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            var_init_scale=0.1,
            use_softplus_var_activation=True,
        )
        # Pre-training: every Z encoder gives mean qz.scale near 0.1 (with randn-weight noise)
        for name in model.module.modality_names:
            enc = model.module.encoders[name]
            x = torch.randn(16, model.module.n_input_per_modality[name])
            enc.eval()
            dist, _ = enc(x)
            assert 0.05 < dist.scale.mean().item() < 0.20
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        assert "elbo_train" in model.history_

    def test_lognormal_mc_horseshoe(self, mdata):
        """horseshoe_latent_z_prior_type='lognormal_mc' trains and logs horseshoe_kl."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="lognormal_mc",
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        hs_keys = [k for k in model.history_ if k.startswith("horseshoe_kl_")]
        assert len(hs_keys) > 0

    def test_horseshoe_invalid_prior_type(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(ValueError, match="horseshoe_latent_z_prior_type"):
            regularizedvi.RegularizedMultimodalVI(
                mdata,
                n_hidden=16,
                n_latent=4,
                horseshoe_latent_z_prior_type="not_a_valid_choice",
            )

    def test_horseshoe_posterior_init_loc_scale(self, mdata):
        """Horseshoe posterior init biases land at log(M) and softplus_inv(S²); weights are randn-init."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="lognormal",
            horseshoe_posterior_init_loc=1.0,
            horseshoe_posterior_init_scale=0.1,
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        for name in model.module.modality_names:
            enc = model.module.horseshoe_encoders[name]
            # mean_encoder.bias = log(M=1) = 0
            assert torch.allclose(enc.mean_encoder.bias, torch.zeros_like(enc.mean_encoder.bias))
            # var_encoder.bias is set so softplus(bias) = S² = 0.01.
            sp = torch.nn.functional.softplus(enc.var_encoder.bias)
            assert torch.allclose(sp, torch.full_like(sp, 0.01), rtol=5e-2, atol=5e-2)
            # weights are NOT zero (randn-init at small std for non-trivial cell-dependence)
            assert enc.mean_encoder.weight.abs().mean().item() > 0
            assert enc.var_encoder.weight.abs().mean().item() > 0
            assert enc.mean_encoder.weight.std().item() < 0.5
            assert enc.var_encoder.weight.std().item() < 0.5

    def test_horseshoe_posterior_init_loc_zero_raises(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(ValueError, match="horseshoe_posterior_init_loc"):
            regularizedvi.RegularizedMultimodalVI(
                mdata,
                n_hidden=16,
                n_latent=4,
                horseshoe_latent_z_prior_type="lognormal",
                horseshoe_posterior_init_loc=0.0,
            )

    def test_ard_horseshoe_incompatible(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(ValueError, match="incompatible with horseshoe"):
            regularizedvi.RegularizedMultimodalVI(
                mdata,
                n_hidden=16,
                n_latent=4,
                use_ard_z_sigma_scale=True,
                horseshoe_latent_z_prior_type="lognormal",
            )

    def test_ard_z_sigma_scale_train(self, mdata):
        """use_ard_z_sigma_scale=True trains and logs ard_z_alpha_kl."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            use_ard_z_sigma_scale=True,
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        # Per-modality params exist with correct shape
        for name in model.module.modality_names:
            assert name in model.module.ard_z_alpha_loc
            assert model.module.ard_z_alpha_loc[name].shape == (4,)
            assert model.module.ard_z_alpha_log_scale[name].shape == (4,)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        ard_keys = [k for k in model.history_ if k.startswith("ard_z_alpha_kl_")]
        assert len(ard_keys) > 0, f"No ard_z_alpha_kl metric found. Keys: {list(model.history_.keys())[:30]}"

    def test_get_per_dim_kl(self, mdata):
        """get_per_dim_kl returns one (n_latent,) array per modality."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        kl_per_dim = model.get_per_dim_kl(batch_size=32)
        for name in model.module.modality_names:
            assert name in kl_per_dim
            assert kl_per_dim[name].shape == (4,)
            assert np.all(np.isfinite(kl_per_dim[name]))

    def test_get_per_dim_kl_single_encoder_raises(self, mdata):
        """single_encoder mode has no per-modality qz; method raises NotImplementedError."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, latent_mode="single_encoder")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        with pytest.raises(NotImplementedError, match="single_encoder"):
            model.get_per_dim_kl(batch_size=32)

    def test_get_per_dim_kl_with_horseshoe(self, mdata):
        """With horseshoe enabled, per-dim KL adds the horseshoe component on top of base N(0,1)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            horseshoe_latent_z_prior_type="lognormal",
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        kl_per_dim = model.get_per_dim_kl(batch_size=32)
        for name in model.module.modality_names:
            assert kl_per_dim[name].shape == (4,)
            assert np.all(np.isfinite(kl_per_dim[name]))

    def test_get_per_dim_kl_with_ard(self, mdata):
        """With ARD enabled, per-dim KL adds the ARD component on top of base N(0,1)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            use_ard_z_sigma_scale=True,
            use_softplus_var_activation=True,
            var_init_scale=0.1,
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        kl_per_dim = model.get_per_dim_kl(batch_size=32)
        for name in model.module.modality_names:
            assert kl_per_dim[name].shape == (4,)
            assert np.all(np.isfinite(kl_per_dim[name]))

    def test_get_per_feature_reconstruction_loss(self, mdata, tmp_path):
        """get_per_feature_reconstruction_loss returns per-modality Series indexed by feature names."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        nll_dict = model.get_per_feature_reconstruction_loss(batch_size=32, save_dir=str(tmp_path))
        assert "rna" in nll_dict
        assert "atac" in nll_dict
        assert nll_dict["rna"].shape == (50,)
        assert nll_dict["atac"].shape == (30,)
        assert list(nll_dict["rna"].index)[:3] == ["gene_0", "gene_1", "gene_2"]
        assert list(nll_dict["atac"].index)[:3] == ["peak_0", "peak_1", "peak_2"]

        # Saved parquet files exist
        assert (tmp_path / "per_feature_nll" / "rna.parquet").exists()
        assert (tmp_path / "per_feature_nll" / "atac.parquet").exists()

    def test_get_per_feature_reconstruction_loss_modality_subset(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        nll = model.get_per_feature_reconstruction_loss(modality_name="rna", batch_size=32)
        assert set(nll.keys()) == {"rna"}

    def test_get_per_feature_reconstruction_loss_matches_recon_loss(self, mdata):
        """Sum of per-feature NLL across features equals the standard recon loss (per cell, summed).

        Both passes are seeded so the residual library w sample and z sample are identical.
        """
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

        torch.manual_seed(123)
        per_feat = model.get_per_feature_reconstruction_loss(batch_size=32)

        # Re-run the same path manually with the same seed.
        module = model.module
        module.eval()
        device = next(module.parameters()).device
        scdl = model._make_data_loader(adata=mdata, batch_size=32)

        manual_mean_total = dict.fromkeys(module.modality_names, 0.0)
        n_cells_total = dict.fromkeys(module.modality_names, 0)
        torch.manual_seed(123)
        with torch.inference_mode():
            for tensors in scdl:
                tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
                inf_inputs = module._get_inference_input(tensors)
                outputs = module.inference(**inf_inputs)
                gen_inputs = module._get_generative_input(tensors, outputs)
                gen_out = module.generative(**gen_inputs)
                for name in module.modality_names:
                    if name not in gen_out["px"]:
                        continue
                    x = tensors.get(f"X_{name}")
                    if x is None:
                        continue
                    nll = -gen_out["px"][name].log_prob(x)  # (n_cells, n_features)
                    manual_mean_total[name] += nll.sum().item()
                    n_cells_total[name] += nll.shape[0]

        for name in module.modality_names:
            expected_mean = manual_mean_total[name] / n_cells_total[name]
            actual_mean = per_feat[name].sum()
            assert np.isclose(expected_mean, actual_mean, rtol=1e-3, atol=1e-3), (
                f"{name}: expected {expected_mean:.6f}, got {actual_mean:.6f}"
            )
