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

import matplotlib
import matplotlib.pyplot as plt
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
    AMBIENT_COVS_KEY,
    DEFAULT_ADDITIVE_BG_PRIOR_ALPHA,
    DEFAULT_ADDITIVE_BG_PRIOR_BETA,
    DEFAULT_COMPUTE_PEARSON,
    DEFAULT_DISPERSION,
    DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
    DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
    DEFAULT_GENE_LIKELIHOOD,
    DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
    DEFAULT_LIBRARY_N_HIDDEN,
    DEFAULT_REGULARISE_BACKGROUND,
    DEFAULT_REGULARISE_DISPERSION,
    DEFAULT_REGULARISE_DISPERSION_PRIOR,
    DEFAULT_SCALE_ACTIVATION,
    DEFAULT_USE_ADDITIVE_BACKGROUND,
    DEFAULT_USE_BATCH_IN_DECODER,
    DEFAULT_USE_BATCH_NORM,
    DEFAULT_USE_LAYER_NORM,
    DISPERSION_KEY,
    LIBRARY_SIZE_KEY,
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
        Rate for Exponential containment prior on ``1/sqrt(theta)``. Default ``3.0``.
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
        dispersion_hyper_prior_alpha: float = DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
        dispersion_hyper_prior_beta: float = DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
        additive_bg_prior_alpha: float = DEFAULT_ADDITIVE_BG_PRIOR_ALPHA,
        additive_bg_prior_beta: float = DEFAULT_ADDITIVE_BG_PRIOR_BETA,
        regularise_background: bool = DEFAULT_REGULARISE_BACKGROUND,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_BATCH_NORM,
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_LAYER_NORM,
        compute_pearson: bool = DEFAULT_COMPUTE_PEARSON,
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
            "dispersion_hyper_prior_alpha": dispersion_hyper_prior_alpha,
            "dispersion_hyper_prior_beta": dispersion_hyper_prior_beta,
            "additive_bg_prior_alpha": additive_bg_prior_alpha,
            "additive_bg_prior_beta": additive_bg_prior_beta,
            "regularise_background": regularise_background,
            "compute_pearson": compute_pearson,
            **kwargs,
        }
        self._model_summary_string = (
            "AmbientRegularizedSCVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}, "
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
            n_cats_per_ambient_cov = (
                self.adata_manager.get_state_registry(AMBIENT_COVS_KEY).n_cats_per_key
                if AMBIENT_COVS_KEY in self.adata_manager.data_registry
                else None
            )
            n_dispersion_cats = (
                len(self.adata_manager.get_state_registry(DISPERSION_KEY).categorical_mapping)
                if DISPERSION_KEY in self.adata_manager.data_registry
                else None
            )
            n_batch = self.summary_stats.n_batch
            # Extract n_library_cats from LIBRARY_SIZE_KEY registry
            if LIBRARY_SIZE_KEY in self.adata_manager.data_registry:
                n_library_cats = len(self.adata_manager.get_state_registry(LIBRARY_SIZE_KEY).categorical_mapping)
            else:
                n_library_cats = n_batch
            use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
            library_log_means, library_log_vars = None, None
            if not use_size_factor_key and self.minified_data_type is None:
                # Compute library priors per library_size_key group
                lib_indices = (
                    self.adata_manager.get_from_registry(LIBRARY_SIZE_KEY)
                    if LIBRARY_SIZE_KEY in self.adata_manager.data_registry
                    else self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
                )
                data = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
                library_log_means_arr = np.zeros(n_library_cats)
                library_log_vars_arr = np.ones(n_library_cats)
                for i_group in np.unique(lib_indices):
                    idx_group = np.squeeze(lib_indices == i_group)
                    group_data = data[idx_group.nonzero()[0]]
                    sum_counts = group_data.sum(axis=1)
                    masked_log_sum = np.ma.log(sum_counts)
                    if np.ma.is_masked(masked_log_sum):
                        logger.warning(
                            "Cells with zero total counts in library group %d. "
                            "Consider filtering with scanpy.pp.filter_cells().",
                            i_group,
                        )
                    log_counts = masked_log_sum.filled(0)
                    library_log_means_arr[i_group] = np.mean(log_counts).astype(np.float32)
                    library_log_vars_arr[i_group] = np.var(log_counts).astype(np.float32)
                library_log_means = library_log_means_arr.reshape(1, -1)
                library_log_vars = library_log_vars_arr.reshape(1, -1)
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
                n_cats_per_ambient_cov=n_cats_per_ambient_cov,
                n_dispersion_cats=n_dispersion_cats,
                n_library_cats=n_library_cats,
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
                dispersion_hyper_prior_alpha=dispersion_hyper_prior_alpha,
                dispersion_hyper_prior_beta=dispersion_hyper_prior_beta,
                additive_bg_prior_alpha=additive_bg_prior_alpha,
                additive_bg_prior_beta=additive_bg_prior_beta,
                regularise_background=regularise_background,
                compute_pearson=compute_pearson,
                **kwargs,
            )
            self.module.minified_data_type = self.minified_data_type

        self.init_params_ = self._get_init_params(locals())

    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        validation_batch_size: int | None = None,
        **trainer_kwargs,
    ):
        """Train the model with optional larger validation batch size.

        Parameters
        ----------
        validation_batch_size
            Batch size for the validation DataLoader. If ``None`` (default),
            uses the same ``batch_size`` as training. Larger values (e.g. 8192)
            reduce noise in Pearson correlation metrics computed on validation.
        **kwargs
            All other arguments are passed to
            :meth:`~scvi.model.base.UnsupervisedTrainingMixin.train`.
        """
        datasplitter_kwargs = datasplitter_kwargs or {}
        if validation_batch_size is not None:
            datasplitter_kwargs.setdefault("val_batch_size", validation_batch_size)

        return super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            load_sparse_tensor=load_sparse_tensor,
            batch_size=batch_size,
            early_stopping=early_stopping,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs=plan_kwargs,
            **trainer_kwargs,
        )

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
        ambient_covariate_keys: list[str] | None = None,
        dispersion_key: str | None = None,
        library_size_key: str | None = None,
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
        ambient_covariate_keys
            Optional list of categorical ``.obs`` keys whose categories define
            per-covariate additive background terms (ambient RNA correction).
            Each key produces a separate ``(n_features, n_cats)`` parameter.
            The per-cell background is the sum across all covariates.
            If None and ``batch_key`` is provided, defaults to ``[batch_key]``.
        dispersion_key
            Optional categorical ``.obs`` key whose categories define the
            per-group dispersion parameter ``px_r[g, s]``. When ``dispersion``
            is ``"gene-batch"``, the second dimension of ``px_r`` is sized by
            the number of categories in this key (instead of ``n_batch``).
            If None and ``batch_key`` is provided, defaults to ``batch_key``.
        library_size_key
            Optional categorical ``.obs`` key whose categories define groups
            for the library size prior ``N(mu_s, sigma_s)``. The prior mean
            and variance are computed per group from data at setup time.
            If None and ``batch_key`` is provided, defaults to ``batch_key``.
        """
        # Mutual exclusion: batch_key cannot be combined with purpose-specific keys
        if batch_key is not None and any([ambient_covariate_keys, dispersion_key, library_size_key]):
            raise ValueError(
                "batch_key cannot be combined with ambient_covariate_keys, dispersion_key, "
                "or library_size_key. Either use batch_key alone (backward compatible) or "
                "specify purpose-specific keys individually."
            )

        # Backward compat: batch_key fans out to purpose-specific keys
        if batch_key is not None:
            ambient_covariate_keys = [batch_key]
            dispersion_key = batch_key
            library_size_key = batch_key

        # Validation: must have ambient covariate source
        if ambient_covariate_keys is None:
            raise ValueError(
                "Either batch_key or ambient_covariate_keys must be provided. "
                "batch_key is used as the default ambient covariate for additive "
                "background correction."
            )

        # Validation: must have dispersion covariate source
        if dispersion_key is None:
            raise ValueError(
                "Either batch_key or dispersion_key must be provided. "
                "batch_key is used as the default dispersion grouping."
            )

        # Validation: must have library size covariate source
        if library_size_key is None:
            raise ValueError(
                "Either batch_key or library_size_key must be provided. "
                "batch_key is used as the default library size grouping."
            )

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            CategoricalJointObsField(AMBIENT_COVS_KEY, ambient_covariate_keys),
            CategoricalObsField(DISPERSION_KEY, dispersion_key),
            CategoricalObsField(LIBRARY_SIZE_KEY, library_size_key),
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

    def plot_training_diagnostics(
        self,
        skip_epochs: int = 80,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot single-modal training diagnostics.

        Creates a column of training curves for key metrics.
        Only rows with available data are shown.

        Parameters
        ----------
        skip_epochs
            Number of initial epochs to skip (early noisy epochs).
        figsize
            Figure size ``(width, height)``. Auto-computed if None.

        Returns
        -------
        matplotlib.figure.Figure
        """
        history = self.history_

        # Define metric rows: (row_label, train_key, val_key)
        metric_rows = [
            ("Reconstruction loss", "reconstruction_loss_train", "reconstruction_loss_validation"),
            ("Pearson r (gene-wise)", "pearson_gene_train", "pearson_gene_validation"),
            ("Pearson r (cell-wise)", "pearson_cell_train", "pearson_cell_validation"),
            ("KL divergence (Z)", "kl_local_train", "kl_local_validation"),
            ("Total ELBO", "elbo_train", "elbo_validation"),
        ]

        # Filter to rows that have at least one available key
        available_rows = []
        for row_label, train_key, val_key in metric_rows:
            has_train = train_key is not None and train_key in history
            has_val = val_key is not None and val_key in history
            if has_train or has_val:
                available_rows.append((row_label, train_key, val_key))

        n_rows = len(available_rows)
        if n_rows == 0:
            raise ValueError("No training metrics found in model.history_. Train the model first.")

        if figsize is None:
            figsize = (5.0, 3.0 * n_rows)

        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)

        for row_idx, (row_label, train_key, val_key) in enumerate(available_rows):
            ax = axes[row_idx, 0]

            if train_key is not None and train_key in history:
                df = history[train_key]
                values = df.iloc[skip_epochs:].values.ravel()
                epochs = range(skip_epochs, skip_epochs + len(values))
                ax.plot(epochs, values, color="tab:blue", label="train")
            if val_key is not None and val_key in history:
                df = history[val_key]
                values = df.iloc[skip_epochs:].values.ravel()
                epochs = range(skip_epochs, skip_epochs + len(values))
                ax.plot(epochs, values, color="tab:orange", label="validation")

            ax.set_ylabel(row_label)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize="small")

        fig.tight_layout()
        return fig
