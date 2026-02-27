"""Tests for AmbientRegularizedSCVI and RegularizedMultimodalVI models."""

import numpy as np
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
            categorical_covariate_keys=["site", "donor"],
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

        assert model.module.px_r.shape == (n_vars, n_batch)

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
            categorical_covariate_keys=["site", "donor"],
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
        assert not hasattr(model.module, "additive_background") or not isinstance(
            getattr(model.module, "additive_background", None), torch.nn.Parameter
        )
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


class TestGammaPoissonMode:
    """Tests for likelihood_distribution='gamma_poisson' mode."""

    def test_gamma_poisson_init(self, adata):
        """Test model initialisation with gamma_poisson mode."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            likelihood_distribution="gamma_poisson",
        )
        assert model.module.likelihood_distribution == "gamma_poisson"

    def test_gamma_poisson_train(self, adata):
        """Test training with gamma_poisson mode runs without error."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            likelihood_distribution="gamma_poisson",
        )
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_gamma_poisson_latent(self, adata):
        """Test latent representation with gamma_poisson mode."""
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key="batch",
        )
        model = regularizedvi.AmbientRegularizedSCVI(
            adata,
            n_hidden=16,
            n_latent=4,
            likelihood_distribution="gamma_poisson",
        )
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        latent = model.get_latent_representation()
        assert latent.shape == (adata.n_obs, 4)

    def test_default_is_gamma_poisson(self, adata):
        """Test that the default likelihood_distribution is 'gamma_poisson'."""
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
        assert model.module.likelihood_distribution == "gamma_poisson"


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
        assert set(module.px_r.keys()) == {"rna", "atac"}
        n_batch = 3
        assert module.px_r["rna"].shape == (50, n_batch)  # n_rna genes × n_batch (gene-batch default)
        assert module.px_r["atac"].shape == (30, n_batch)  # n_atac peaks × n_batch

    def test_default_additive_background_and_region_factors(self, mdata):
        """Test default additive background on RNA, region factors on ATAC."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module

        assert "rna" in module.additive_background
        assert "atac" not in module.additive_background
        assert "atac" in module.region_factors
        assert "rna" not in module.region_factors

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

    def test_nb_likelihood(self, mdata):
        """Test model works with NB likelihood instead of default GammaPoisson."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4, likelihood_distribution="nb")
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_custom_modality_flags(self, mdata):
        """Test custom additive_background and region_factors modality lists."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        # Reverse the defaults: background on ATAC, region factors on RNA
        model = regularizedvi.RegularizedMultimodalVI(
            mdata,
            n_hidden=16,
            n_latent=4,
            additive_background_modalities=["atac"],
            region_factors_modalities=["rna"],
        )
        module = model.module
        assert "atac" in module.additive_background
        assert "rna" not in module.additive_background
        assert "rna" in module.region_factors
        assert "atac" not in module.region_factors
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
        assert module.px_r["rna"].shape == (50, n_batch)
        assert module.px_r["atac"].shape == (30, n_batch)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_extra_metrics_in_loss(self, mdata):
        """Test that loss() returns per-modality extra_metrics."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)

        # Run one forward pass through the module to get the LossOutput
        module = model.module
        module.eval()
        scdl = model._make_data_loader(adata=mdata, batch_size=32)
        tensors = next(iter(scdl))
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

    # --- Region factors: shape, activation, prior, scaling covariates ---

    def test_region_factors_shape_no_scaling_covs(self, mdata):
        """Test region factors shape without scaling covariates (backward compat)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # Default: ATAC gets region factors, shape (1, n_atac_features) without scaling covs
        assert "atac" in module.region_factors
        assert module.region_factors["atac"].shape == (1, 30)

    def test_region_factors_shape_with_scaling_covs(self, mdata):
        """Test region factors shape with scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            modality_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        n_tech = 2  # tech_0, tech_1
        assert module.region_factors["atac"].shape == (n_tech, 30)

    def test_region_factors_softplus_activation(self, mdata):
        """Test that region factors use softplus/0.7 activation (centered at ~1)."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # At initialization (param=0), softplus(0)/0.7 ~ 0.693/0.7 ~ 0.99
        rf_val = torch.nn.functional.softplus(module.region_factors["atac"]) / 0.7
        assert torch.allclose(rf_val, torch.ones_like(rf_val), atol=0.02)

    def test_train_with_scaling_covs(self, mdata):
        """Test training with modality scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            modality_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=3, train_size=1.0, batch_size=32)

    def test_region_factors_prior_in_loss(self, mdata):
        """Test that Gamma prior on region factors contributes to finite loss."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=1, train_size=1.0, batch_size=32)
        module = model.module
        module.eval()
        scdl = model._make_data_loader(adata=mdata, batch_size=32)
        tensors = next(iter(scdl))
        inf_inputs = module._get_inference_input(tensors)
        inf_outputs = module.inference(**inf_inputs)
        gen_inputs = module._get_generative_input(tensors, inf_outputs)
        gen_outputs = module.generative(**gen_inputs)
        loss_output = module.loss(tensors, inf_outputs, gen_outputs)
        assert loss_output.loss.isfinite()

    def test_region_factors_with_multiple_scaling_covs(self, mdata):
        """Test region factors with multiple scaling covariates."""
        # Add a second covariate
        for mod_key in mdata.mod:
            mdata[mod_key].obs["site"] = [f"site_{i % 3}" for i in range(mdata.n_obs)]
            mdata[mod_key].obs["site"] = mdata[mod_key].obs["site"].astype("category")

        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            modality_scaling_covariate_keys=["technology", "site"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        module = model.module
        # 2 tech + 3 sites = 5 total rows
        assert module.region_factors["atac"].shape == (5, 30)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)

    def test_attribution_with_scaling_covs(self, mdata):
        """Test get_modality_attribution works with scaling covariates."""
        regularizedvi.RegularizedMultimodalVI.setup_mudata(
            mdata,
            batch_key="batch",
            modality_scaling_covariate_keys=["technology"],
        )
        model = regularizedvi.RegularizedMultimodalVI(mdata, n_hidden=16, n_latent=4)
        model.train(max_epochs=2, train_size=1.0, batch_size=32)
        result = model.get_modality_attribution(batch_size=32)
        n_latent = model.module.total_latent_dim
        for name in ["rna", "atac"]:
            assert result[name]["attribution"].shape == (mdata.n_obs, n_latent)
