"""Tests asserting removed-API kwargs raise TypeError (Item 4 hard break).

Per plan `validated-knitting-zebra.md` Item 4 sub-item A: the kwargs
`regularise_dispersion_prior` and `dispersion_hyper_prior_beta` have been
DELETED from the constructors of:

- `regularizedvi._module.RegularizedVAE`
- `regularizedvi._multimodule.RegularizedMultimodalVAE`
- `regularizedvi._model.AmbientRegularizedSCVI`
- `regularizedvi._multimodel.RegularizedMultimodalVI`

They are replaced by the unified `dispersion_hyper_prior_mean` parameter.
This file asserts that passing the legacy kwargs now raises ``TypeError``
with Python's default "got an unexpected keyword argument" message — no
sentinel / custom error message is expected.
"""

from __future__ import annotations

import pytest

import regularizedvi
from regularizedvi._module import RegularizedVAE
from regularizedvi._multimodule import RegularizedMultimodalVAE


# ---------------------------------------------------------------------------
# AmbientRegularizedSCVI (high-level model class)
# ---------------------------------------------------------------------------
class TestAmbientRegularizedSCVIRemovedKwargs:
    """`regularise_dispersion_prior` / `dispersion_hyper_prior_beta` must be
    rejected by ``AmbientRegularizedSCVI`` (they flow through ``**kwargs``
    into ``RegularizedVAE`` which no longer accepts them)."""

    def test_regularise_dispersion_prior_removed_amb_scvi(self, adata):
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        with pytest.raises(TypeError, match="regularise_dispersion_prior"):
            regularizedvi.AmbientRegularizedSCVI(
                adata,
                n_hidden=8,
                n_latent=4,
                regularise_dispersion_prior=3.0,
            )

    def test_dispersion_hyper_prior_beta_removed_amb_scvi(self, adata):
        regularizedvi.AmbientRegularizedSCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        with pytest.raises(TypeError, match="dispersion_hyper_prior_beta"):
            regularizedvi.AmbientRegularizedSCVI(
                adata,
                n_hidden=8,
                n_latent=4,
                dispersion_hyper_prior_beta=3.0,
            )


# ---------------------------------------------------------------------------
# RegularizedMultimodalVI (high-level multimodal model class)
# ---------------------------------------------------------------------------
class TestRegularizedMultimodalVIRemovedKwargs:
    """Same two kwargs must be rejected by ``RegularizedMultimodalVI``
    (they flow through ``**kwargs`` into ``RegularizedMultimodalVAE``)."""

    def test_regularise_dispersion_prior_removed_multi(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(TypeError, match="regularise_dispersion_prior"):
            regularizedvi.RegularizedMultimodalVI(
                mdata,
                n_hidden=16,
                n_latent=4,
                regularise_dispersion_prior={"rna": 3.0, "atac": 3.0},
            )

    def test_dispersion_hyper_prior_beta_removed_multi(self, mdata):
        regularizedvi.RegularizedMultimodalVI.setup_mudata(mdata, batch_key="batch")
        with pytest.raises(TypeError, match="dispersion_hyper_prior_beta"):
            regularizedvi.RegularizedMultimodalVI(
                mdata,
                n_hidden=16,
                n_latent=4,
                dispersion_hyper_prior_beta={"rna": 3.0, "atac": 3.0},
            )


# ---------------------------------------------------------------------------
# Lower-level modules (direct construction)
# ---------------------------------------------------------------------------
class TestModuleClassRemovedKwargs:
    """Direct construction of the underlying ``nn.Module`` classes must also
    reject the removed kwargs (no sentinel; default Python TypeError).

    We do not need a functional module — we only need to reach the argument
    binding step, so minimal required args suffice.
    """

    def test_regularized_vae_regularise_dispersion_prior_removed(self):
        with pytest.raises(TypeError, match="regularise_dispersion_prior"):
            RegularizedVAE(
                n_input=20,
                n_batch=2,
                n_hidden=8,
                n_latent=4,
                regularise_dispersion_prior=3.0,
            )

    def test_regularized_vae_dispersion_hyper_prior_beta_removed(self):
        with pytest.raises(TypeError, match="dispersion_hyper_prior_beta"):
            RegularizedVAE(
                n_input=20,
                n_batch=2,
                n_hidden=8,
                n_latent=4,
                dispersion_hyper_prior_beta=3.0,
            )

    def test_regularized_multimodal_vae_regularise_dispersion_prior_removed(self):
        with pytest.raises(TypeError, match="regularise_dispersion_prior"):
            RegularizedMultimodalVAE(
                modality_names=["rna", "atac"],
                n_input_per_modality={"rna": 20, "atac": 15},
                n_batch=2,
                n_hidden=8,
                n_latent=4,
                regularise_dispersion_prior={"rna": 3.0, "atac": 3.0},
            )

    def test_regularized_multimodal_vae_dispersion_hyper_prior_beta_removed(self):
        with pytest.raises(TypeError, match="dispersion_hyper_prior_beta"):
            RegularizedMultimodalVAE(
                modality_names=["rna", "atac"],
                n_input_per_modality={"rna": 20, "atac": 15},
                n_batch=2,
                n_hidden=8,
                n_latent=4,
                dispersion_hyper_prior_beta={"rna": 3.0, "atac": 3.0},
            )
