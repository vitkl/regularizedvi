"""AmbientRegularizedSCVI model class.

Extends scvi.model.SCVI to use RegularizedVAE with regularizedvi defaults:
- gene_likelihood="nb"
- dispersion="gene-batch"
- use_observed_lib_size=False (learned library size)
- use_additive_background=True (ambient RNA correction)
- use_batch_in_decoder=False (batch-free decoder)
- regularise_dispersion=True (containment prior on overdispersion)
- use_batch_norm="none", use_layer_norm="both" (LayerNorm preferred)
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._constants import _ADATA_MINIFY_TYPE_UNS_KEY, ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    ArchesMixin,
    BaseMinifiedModeModelClass,
    EmbeddingMixin,
    RNASeqMixin,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.utils import setup_anndata_dsp

from regularizedvi._constants import (
    DEFAULT_DISPERSION,
    DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
    DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
    DEFAULT_GENE_LIKELIHOOD,
    DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
    DEFAULT_LIBRARY_N_HIDDEN,
    DEFAULT_LIKELIHOOD_DISTRIBUTION,
    DEFAULT_REGULARISE_DISPERSION,
    DEFAULT_REGULARISE_DISPERSION_PRIOR,
    DEFAULT_SCALE_ACTIVATION,
    DEFAULT_USE_ADDITIVE_BACKGROUND,
    DEFAULT_USE_BATCH_IN_DECODER,
    DEFAULT_USE_BATCH_NORM,
    DEFAULT_USE_LAYER_NORM,
)
from regularizedvi._module import RegularizedVAE

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData
    from scvi._types import MinifiedDataType
    from scvi.data.fields import (
        BaseAnnDataField,
    )

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"
_SCVI_OBSERVED_LIB_SIZE = "_scvi_observed_lib_size"

logger = logging.getLogger(__name__)


class AmbientRegularizedSCVI(
    EmbeddingMixin,
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
):
    """Regularized scVI with ambient RNA correction and overdispersion regularisation.

    Adapts `cell2location <https://doi.org/10.1038/s41587-021-01139-4>`_/`cell2fate <https://doi.org/10.1038/s41592-025-02608-3>`_ modelling principles to scVI:

    - **Ambient RNA**: per-gene, per-sample additive background captures ambient RNA,
      mirroring cell2location's ``(g_fg + b_eg) * h_e`` structure.
    - **Dispersion regularisation**: Exponential prior on the dispersion quantity
      ``sqrt(theta)`` penalises large theta, preventing NB collapse to Poisson
      during gradient-based training.
    - **Batch-free decoder**: batch correction through additive background and categorical
      covariates, not decoder conditioning.
    - **Learned library size**: with constrained prior (``library_log_vars_weight=0.05``).

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`setup_anndata`.
    n_hidden
        Number of nodes per hidden layer. High values (512-3000) are recommended —
        the hidden layer acts as a dictionary of atomic regulatory programmes.
    n_latent
        Dimensionality of the latent space. High values (128-700) give free capacity,
        removing competition between cell types for the same latent dimension.
    n_layers
        Number of hidden layers.
    dropout_rate
        Dropout rate for encoder.
    dispersion
        Dispersion parameter flexibility. Default ``"gene-batch"`` for per-gene,
        per-batch overdispersion.
    gene_likelihood
        Reconstruction distribution. Default ``"nb"`` (Negative Binomial).
    latent_distribution
        Latent space distribution.
    library_log_vars_weight
        Scale factor for library prior variance. Default ``0.05`` constrains
        the library size to prevent absorbing biological signal.
    library_n_hidden
        Hidden units in library encoder. Default ``16`` for low capacity.
    scale_activation
        Decoder scale activation. Default ``"softplus"`` (expression not on simplex).
    use_additive_background
        Enable ambient RNA correction. Default ``True``.
    use_batch_in_decoder
        Pass batch info to decoder. Default ``False`` (batch-free decoder).
    regularise_dispersion
        Enable overdispersion regularisation. Default ``True``.
    regularise_dispersion_prior
        Rate for Exponential prior on dispersion. Default ``3.0``.
    likelihood_distribution
        Distribution implementation for reconstruction loss.
        ``"nb"`` (default): scvi-tools NegativeBinomial with Exp prior on sqrt(theta).
        ``"gamma_poisson"``: Pyro GammaPoisson with Exp prior on 1/sqrt(theta)
        (cell2location direction, pushes theta toward Poisson).
    use_batch_norm
        Where to use BatchNorm. Default ``"none"``.
    use_layer_norm
        Where to use LayerNorm. Default ``"both"``.
    **kwargs
        Additional keyword arguments for :class:`RegularizedVAE`.

    Examples
    --------
    >>> import regularizedvi
    >>> regularizedvi.AmbientRegularizedSCVI.setup_anndata(
    ...     adata,
    ...     layer="counts",
    ...     batch_key="batch",
    ...     categorical_covariate_keys=["site", "donor"],
    ... )
    >>> model = regularizedvi.AmbientRegularizedSCVI(
    ...     adata,
    ...     n_hidden=512,
    ...     n_layers=1,
    ...     n_latent=128,
    ... )
    >>> model.train(train_size=1.0, max_epochs=2000, batch_size=1024)
    >>> latent = model.get_latent_representation()

    References
    ----------
    - Lopez et al. (2018). Deep generative modeling for single-cell transcriptomics.
    - Kleshchevnikov et al. (2022). Cell2location maps fine-grained cell types.
    - Aivazidis et al. (2025). Cell2fate infers RNA velocity modules.
    - Simpson et al. (2017). Penalising Model Component Complexity.
    """

    _module_cls = RegularizedVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = DEFAULT_DISPERSION,
        gene_likelihood: Literal["zinb", "nb", "poisson", "normal"] = DEFAULT_GENE_LIKELIHOOD,
        latent_distribution: Literal["normal", "ln"] = "normal",
        # regularizedvi defaults (from _constants.py)
        library_log_vars_weight: float = DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
        library_n_hidden: int = DEFAULT_LIBRARY_N_HIDDEN,
        scale_activation: str = DEFAULT_SCALE_ACTIVATION,
        use_additive_background: bool = DEFAULT_USE_ADDITIVE_BACKGROUND,
        use_batch_in_decoder: bool = DEFAULT_USE_BATCH_IN_DECODER,
        regularise_dispersion: bool = DEFAULT_REGULARISE_DISPERSION,
        regularise_dispersion_prior: float = DEFAULT_REGULARISE_DISPERSION_PRIOR,
        likelihood_distribution: Literal["nb", "gamma_poisson"] = DEFAULT_LIKELIHOOD_DISTRIBUTION,
        dispersion_hyper_prior_alpha: float = DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
        dispersion_hyper_prior_beta: float = DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_BATCH_NORM,
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_LAYER_NORM,
        **kwargs,
    ):
        super().__init__(adata)

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": use_layer_norm,
            "library_log_vars_weight": library_log_vars_weight,
            "library_n_hidden": library_n_hidden,
            "scale_activation": scale_activation,
            "use_additive_background": use_additive_background,
            "use_batch_in_decoder": use_batch_in_decoder,
            "regularise_dispersion": regularise_dispersion,
            "regularise_dispersion_prior": regularise_dispersion_prior,
            "likelihood_distribution": likelihood_distribution,
            "dispersion_hyper_prior_alpha": dispersion_hyper_prior_alpha,
            "dispersion_hyper_prior_beta": dispersion_hyper_prior_beta,
            **kwargs,
        }
        self._model_summary_string = (
            "AmbientRegularizedSCVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}, "
            f"likelihood_distribution: {likelihood_distribution}, "
            f"use_additive_background: {use_additive_background}, "
            f"use_batch_in_decoder: {use_batch_in_decoder}, "
            f"regularise_dispersion: {regularise_dispersion}."
        )

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `adata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            n_cats_per_cov = (
                self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
                if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
                else None
            )
            n_batch = self.summary_stats.n_batch
            use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
            library_log_means, library_log_vars = None, None
            if not use_size_factor_key and self.minified_data_type is None:
                library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)
            # Determine use_observed_lib_size:
            # If library_log_means/vars are provided (computed above), learn library size.
            # This is a key regularizedvi default — observed totals include ambient RNA.
            use_observed_lib_size = library_log_means is None
            self.module = self._module_cls(
                n_input=self.summary_stats.n_vars,
                n_batch=n_batch,
                n_labels=self.summary_stats.n_labels,
                n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
                n_cats_per_cov=n_cats_per_cov,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                dropout_rate=dropout_rate,
                dispersion=dispersion,
                gene_likelihood=gene_likelihood,
                latent_distribution=latent_distribution,
                use_size_factor_key=use_size_factor_key,
                use_observed_lib_size=use_observed_lib_size,
                library_log_means=library_log_means,
                library_log_vars=library_log_vars,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                library_log_vars_weight=library_log_vars_weight,
                library_n_hidden=library_n_hidden,
                scale_activation=scale_activation,
                use_additive_background=use_additive_background,
                use_batch_in_decoder=use_batch_in_decoder,
                regularise_dispersion=regularise_dispersion,
                regularise_dispersion_prior=regularise_dispersion_prior,
                likelihood_distribution=likelihood_distribution,
                dispersion_hyper_prior_alpha=dispersion_hyper_prior_alpha,
                dispersion_hyper_prior_beta=dispersion_hyper_prior_beta,
                **kwargs,
            )
            self.module.minified_data_type = self.minified_data_type

        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _get_fields_for_adata_minification(
        minified_data_type: MinifiedDataType,
    ) -> list[BaseAnnDataField]:
        """Return the fields required for adata minification of the given minified_data_type."""
        if minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            fields = [
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZM_KEY,
                    _SCVI_LATENT_QZM,
                ),
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZV_KEY,
                    _SCVI_LATENT_QZV,
                ),
                NumericalObsField(
                    REGISTRY_KEYS.OBSERVED_LIB_SIZE,
                    _SCVI_OBSERVED_LIB_SIZE,
                ),
            ]
        else:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")
        fields.append(
            StringUnsField(
                REGISTRY_KEYS.MINIFY_TYPE_KEY,
                _ADATA_MINIFY_TYPE_UNS_KEY,
            ),
        )
        return fields

    def minify_adata(
        self,
        minified_data_type: MinifiedDataType = ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ) -> None:
        """Minifies the model's adata.

        Parameters
        ----------
        minified_data_type
            How to minify the data.
        use_latent_qzm_key
            Key in ``adata.obsm`` for latent qzm params.
        use_latent_qzv_key
            Key in ``adata.obsm`` for latent qzv params.
        """
        from scvi.model.utils import get_minified_adata_scrna

        if minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")

        if self.module.use_observed_lib_size is False:
            raise ValueError("Cannot minify the data if `use_observed_lib_size` is False")

        minified_adata = get_minified_adata_scrna(self.adata, minified_data_type)
        minified_adata.obsm[_SCVI_LATENT_QZM] = self.adata.obsm[use_latent_qzm_key]
        minified_adata.obsm[_SCVI_LATENT_QZV] = self.adata.obsm[use_latent_qzv_key]
        counts = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        minified_adata.obs[_SCVI_OBSERVED_LIB_SIZE] = np.squeeze(np.asarray(counts.sum(axis=1)))
        self._update_adata_and_manager_post_minification(minified_adata, minified_data_type)
        self.module.minified_data_type = minified_data_type
