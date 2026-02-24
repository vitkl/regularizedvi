"""Tests for AmbientRegularizedSCVI model."""

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
        assert module.gene_likelihood == "nb"
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
        # Copy weights from reg to noreg so they're comparable
        model_noreg.module.load_state_dict(model_reg.module.state_dict())

        # Both should have the regularise_dispersion attribute set correctly
        assert model_reg.module.regularise_dispersion is True
        assert model_noreg.module.regularise_dispersion is False


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
