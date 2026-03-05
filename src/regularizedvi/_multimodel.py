"""RegularizedMultimodalVI model class.

N-modality extensible model with symmetric regularized components.
Supports RNA + ATAC (and extensible to more modalities) with:
- GammaPoisson likelihood for all modalities
- Learnable hierarchical dispersion prior
- Per-modality additive background and feature scaling
- Three Z combination strategies: concatenation, single_encoder, weighted_mean
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mudata import MuData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from scvi.utils import setup_anndata_dsp

from regularizedvi._constants import (
    AMBIENT_COVS_KEY,
    DEFAULT_ADDITIVE_BG_PRIOR_ALPHA,
    DEFAULT_ADDITIVE_BG_PRIOR_BETA,
    DEFAULT_COMPUTE_PEARSON,
    DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
    DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
    DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
    DEFAULT_FEATURE_SCALING_PRIOR_BETA,
    DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
    DEFAULT_LIBRARY_N_HIDDEN,
    DEFAULT_REGULARISE_BACKGROUND,
    DEFAULT_REGULARISE_DISPERSION,
    DEFAULT_REGULARISE_DISPERSION_PRIOR,
    DEFAULT_SCALE_ACTIVATION,
    DEFAULT_USE_BATCH_IN_DECODER,
    DEFAULT_USE_BATCH_NORM,
    DEFAULT_USE_LAYER_NORM,
    DISPERSION_KEY,
    ENCODER_COVS_KEY,
    FEATURE_SCALING_COVS_KEY,
    LIBRARY_SIZE_KEY,
)
from regularizedvi._model import AmbientRegularizedSCVI
from regularizedvi._multimodule import RegularizedMultimodalVAE

if TYPE_CHECKING:
    from typing import Literal


logger = logging.getLogger(__name__)


class RegularizedMultimodalVI(
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """Regularized multi-modal VI for paired RNA + ATAC (and more).

    N-modality extensible model using symmetric regularized components.
    Each modality uses the same encoder/decoder architecture with per-modality
    configuration for architecture size, additive background, and feature scaling.

    Parameters
    ----------
    mdata
        MuData object with modalities registered via :meth:`setup_mudata`.
    n_hidden
        Hidden units per modality. Int (shared) or dict per modality.
    n_latent
        Latent dim per modality. Int (shared) or dict per modality.
    n_layers
        Hidden layers per modality. Int (shared) or dict per modality.
    dropout_rate
        Dropout rate for encoders.
    latent_mode
        Z combination strategy: ``"concatenation"`` (default),
        ``"single_encoder"``, or ``"weighted_mean"``.
    modality_weights
        For weighted_mean: ``"equal"``, ``"universal"``, or ``"cell"``.
    dispersion
        Dispersion parameterization. Str (shared) or dict per modality.
    library_log_vars_weight
        Scale for library prior variance. Default 0.05.
    library_n_hidden
        Hidden units for library encoder. Default 16.
    scale_activation
        Decoder scale activation. Default ``"softplus"``.
    use_batch_in_decoder
        If False (default), batch-free decoder.
    additive_background_modalities
        Modalities with additive ambient background. Default ``["rna"]``.
    feature_scaling_modalities
        Modalities with per-feature feature scaling. Default ``["atac"]``.
    feature_scaling_prior_alpha
        Gamma prior alpha on feature scaling. Default 200 (tight prior, mean=1).
    feature_scaling_prior_beta
        Gamma prior beta on feature scaling. Default 200.
    regularise_dispersion
        Enable dispersion regularization. Default True.
    regularise_dispersion_prior
        Initialization for Exponential containment prior rate. Default 3.0.
    dispersion_hyper_prior_alpha
        Gamma hyper-prior alpha. Default 9.0.
    dispersion_hyper_prior_beta
        Gamma hyper-prior beta. Default 3.0.
    use_batch_norm
        Where to use BatchNorm. Default ``"none"``.
    use_layer_norm
        Where to use LayerNorm. Default ``"both"``.
    **kwargs
        Additional kwargs for :class:`RegularizedMultimodalVAE`.
    """

    _module_cls = RegularizedMultimodalVAE

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(cls, adata, **kwargs):
        """Not supported. Use :meth:`setup_mudata` instead."""
        raise NotImplementedError(
            "RegularizedMultimodalVI requires MuData input. Use RegularizedMultimodalVI.setup_mudata() instead."
        )

    def __init__(
        self,
        mdata: MuData | None = None,
        n_hidden: dict[str, int] | int = 128,
        n_latent: dict[str, int] | int = 10,
        n_layers: dict[str, int] | int = 1,
        dropout_rate: float = 0.1,
        latent_mode: Literal["concatenation", "single_encoder", "weighted_mean"] = "concatenation",
        modality_weights: Literal["equal", "universal", "cell"] = "equal",
        dispersion: dict[str, str] | str = "gene-batch",
        library_log_vars_weight: float = DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
        library_n_hidden: int = DEFAULT_LIBRARY_N_HIDDEN,
        scale_activation: str = DEFAULT_SCALE_ACTIVATION,
        use_batch_in_decoder: bool = DEFAULT_USE_BATCH_IN_DECODER,
        additive_background_modalities: list[str] | None = None,
        feature_scaling_modalities: list[str] | None = None,
        feature_scaling_prior_alpha: float = DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
        feature_scaling_prior_beta: float = DEFAULT_FEATURE_SCALING_PRIOR_BETA,
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
        AmbientRegularizedSCVI._validate_bool_params(
            use_batch_in_decoder=use_batch_in_decoder,
            regularise_dispersion=regularise_dispersion,
            regularise_background=regularise_background,
            compute_pearson=compute_pearson,
        )
        super().__init__(mdata)

        # Discover modality names from the registered MuData
        if mdata is not None:
            modality_names = self._get_modality_names()
        else:
            modality_names = []

        # Set defaults based on discovered modalities
        if additive_background_modalities is None:
            additive_background_modalities = [m for m in modality_names if m == "rna"]
        if feature_scaling_modalities is None:
            feature_scaling_modalities = [m for m in modality_names if m == "atac"]

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "latent_mode": latent_mode,
            "modality_weights": modality_weights,
            "dispersion": dispersion,
            "library_log_vars_weight": library_log_vars_weight,
            "library_n_hidden": library_n_hidden,
            "scale_activation": scale_activation,
            "use_batch_in_decoder": use_batch_in_decoder,
            "additive_background_modalities": additive_background_modalities,
            "feature_scaling_modalities": feature_scaling_modalities,
            "feature_scaling_prior_alpha": feature_scaling_prior_alpha,
            "feature_scaling_prior_beta": feature_scaling_prior_beta,
            "regularise_dispersion": regularise_dispersion,
            "regularise_dispersion_prior": regularise_dispersion_prior,
            "dispersion_hyper_prior_alpha": dispersion_hyper_prior_alpha,
            "dispersion_hyper_prior_beta": dispersion_hyper_prior_beta,
            "additive_bg_prior_alpha": additive_bg_prior_alpha,
            "additive_bg_prior_beta": additive_bg_prior_beta,
            "regularise_background": regularise_background,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": use_layer_norm,
            "compute_pearson": compute_pearson,
            **kwargs,
        }

        if self._module_init_on_train:
            self.module = None
            warnings.warn(
                "Model was initialized without `mdata`. The module will be initialized when "
                "calling `train`. This behavior is experimental and may change in the future.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        else:
            self._init_module(modality_names)

        self._model_summary_string = (
            f"RegularizedMultimodalVI with modalities: {modality_names}, "
            f"latent_mode={latent_mode}, n_hidden={n_hidden}, n_latent={n_latent}"
        )
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

    def _get_modality_names(self) -> list[str]:
        """Extract modality names from registered data."""
        registry = self.adata_manager.data_registry
        names = []
        for key in sorted(registry.keys()):
            if key.startswith("X_") and key != "X_KEY":
                names.append(key[2:])  # Remove "X_" prefix
        return names

    def _init_module(self, modality_names: list[str]):
        """Initialize the module with data-dependent parameters."""
        n_batch = self.summary_stats.n_batch

        # Get n_input per modality from summary stats
        # MuDataLayerField stores n_vars as n_X_{name} in summary_stats
        n_input_per_modality = {}
        for name in modality_names:
            # Try the key format that MuDataLayerField uses
            for key_fmt in [f"n_X_{name}", f"n_vars_{name}"]:
                if key_fmt in self.summary_stats:
                    n_input_per_modality[name] = self.summary_stats[key_fmt]
                    break
            else:
                # Fallback: get shape from registered data
                reg_key = f"X_{name}"
                if reg_key in self.adata_manager.data_registry:
                    n_input_per_modality[name] = self.adata_manager.get_from_registry(reg_key).shape[1]

        # Compute per-modality library size priors (log-scale mean and variance per group)
        # Use library_size_key indices if registered, otherwise fall back to batch_key
        n_library_cats = None
        if LIBRARY_SIZE_KEY in self.adata_manager.data_registry:
            lib_indices = self.adata_manager.get_from_registry(LIBRARY_SIZE_KEY)
            n_library_cats = len(self.adata_manager.get_state_registry(LIBRARY_SIZE_KEY).categorical_mapping)
        else:
            lib_indices = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
            n_library_cats = n_batch

        library_log_means = {}
        library_log_vars = {}
        for name in modality_names:
            reg_key = f"X_{name}"
            data = self.adata_manager.get_from_registry(reg_key)
            log_means = np.zeros(n_library_cats)
            log_vars = np.ones(n_library_cats)
            for i_group in np.unique(lib_indices):
                idx_group = np.squeeze(lib_indices == i_group)
                group_data = data[idx_group.nonzero()[0]]
                sum_counts = group_data.sum(axis=1)
                masked_log_sum = np.ma.log(sum_counts)
                if np.ma.is_masked(masked_log_sum):
                    logger.warning(
                        "Modality '%s' has cells with zero total counts in library group %d. "
                        "Consider filtering with scanpy.pp.filter_cells().",
                        name,
                        i_group,
                    )
                log_counts = masked_log_sum.filled(0)
                log_means[i_group] = np.mean(log_counts).astype(np.float32)
                log_vars[i_group] = np.var(log_counts).astype(np.float32)
            library_log_means[name] = log_means.reshape(1, -1)
            library_log_vars[name] = log_vars.reshape(1, -1)

        n_cats_per_cov = None
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_cov = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key

        n_cats_per_feature_scaling_cov = None
        if FEATURE_SCALING_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_feature_scaling_cov = self.adata_manager.get_state_registry(
                FEATURE_SCALING_COVS_KEY
            ).n_cats_per_key

        n_cats_per_ambient_cov = None
        if AMBIENT_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_ambient_cov = self.adata_manager.get_state_registry(AMBIENT_COVS_KEY).n_cats_per_key

        n_dispersion_cats = None
        if DISPERSION_KEY in self.adata_manager.data_registry:
            n_dispersion_cats = len(self.adata_manager.get_state_registry(DISPERSION_KEY).categorical_mapping)

        n_cats_per_encoder_cov = None
        if ENCODER_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_encoder_cov = self.adata_manager.get_state_registry(ENCODER_COVS_KEY).n_cats_per_key

        kwargs = dict(self._module_kwargs)
        # Remove keys that are passed separately
        for k in ["n_hidden", "n_latent", "n_layers", "dropout_rate"]:
            kwargs.pop(k, None)

        self.module = self._module_cls(
            modality_names=modality_names,
            n_input_per_modality=n_input_per_modality,
            n_batch=n_batch,
            n_hidden=self._module_kwargs["n_hidden"],
            n_latent=self._module_kwargs["n_latent"],
            n_layers=self._module_kwargs["n_layers"],
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=self._module_kwargs["dropout_rate"],
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_cats_per_feature_scaling_cov=n_cats_per_feature_scaling_cov,
            n_cats_per_ambient_cov=n_cats_per_ambient_cov,
            n_cats_per_encoder_cov=n_cats_per_encoder_cov,
            n_dispersion_cats=n_dispersion_cats,
            n_library_cats=n_library_cats,
            **kwargs,
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        modalities: dict[str, str] | None = None,
        batch_key: str | None = None,
        ambient_covariate_keys: list[str] | None = None,
        dispersion_key: str | None = None,
        library_size_key: str | None = None,
        nn_conditioning_covariate_keys: list[str] | None = None,
        nn_continuous_covariate_keys: list[str] | None = None,
        feature_scaling_covariate_keys: list[str] | None = None,
        encoder_covariate_keys: list[str] | None | bool = False,
        **kwargs,
    ):
        """Set up MuData for RegularizedMultimodalVI.

        Parameters
        ----------
        mdata
            MuData object with modalities.
        modalities
            Dict mapping registry key to MuData modality key.
            E.g., ``{"rna": "rna", "atac": "atac"}``.
            If None, uses all modalities in the MuData.
        %(param_batch_key)s
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
        nn_conditioning_covariate_keys
            keys in ``adata.obs`` that correspond to categorical data.
            One-hot encoded and fed into the decoder neural network as
            conditioning input.
        nn_continuous_covariate_keys
            keys in ``adata.obs`` that correspond to continuous data.
            Fed into the encoder and decoder neural networks as conditioning input.
        feature_scaling_covariate_keys
            Optional list of categorical ``.obs`` keys whose categories define
            per-feature scaling factors (cell2location-style ``y_{t,g}``).
            Registered separately from ``nn_conditioning_covariate_keys`` — these
            do NOT feed into encoder/decoder injection layers, they only
            control the per-feature multiplicative feature scaling.
        encoder_covariate_keys
            Categorical ``.obs`` keys to inject into the encoder as one-hot
            covariates. Default is ``False`` (no encoder categoricals), which
            matches standard scVI/MultiVI/PeakVI behavior and keeps the latent
            space batch-free. Setting this to a list or ``None`` is non-standard
            and may leak batch information into the latent space.
        """
        # Encoder covariates: False = no encoder categoricals (default, matches scVI)
        _encoder_cov_keys = None  # None → empty CategoricalJointObsField
        if encoder_covariate_keys is not False:
            import warnings as _warn

            _warn.warn(
                "encoder_covariate_keys is set to a non-default value. "
                "Injecting categorical covariates into the encoder is non-standard "
                "(scVI, MultiVI, PeakVI all default to no encoder categoricals) "
                "and may leak batch information into the latent space, hurting "
                "batch correction. Use with caution.",
                UserWarning,
                stacklevel=2,
            )
            _encoder_cov_keys = encoder_covariate_keys  # list[str] or None

        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            modalities = {key: key for key in mdata.mod.keys()}

        # Mutual exclusion: batch_key cannot be combined with purpose-specific keys
        if batch_key is not None and any([ambient_covariate_keys, dispersion_key, library_size_key]):
            raise ValueError(
                "batch_key cannot be combined with ambient_covariate_keys, dispersion_key, "
                "or library_size_key. Either use batch_key alone (backward compatible) or "
                "specify purpose-specific keys individually."
            )

        # Backward compat: batch_key fans out to all purpose-specific keys
        if batch_key is not None:
            ambient_covariate_keys = [batch_key]
            dispersion_key = batch_key
            library_size_key = batch_key

        # Default: modality_scaling mirrors categorical if not explicitly set
        if nn_conditioning_covariate_keys is not None and feature_scaling_covariate_keys is None:
            feature_scaling_covariate_keys = nn_conditioning_covariate_keys

        anndata_fields = []

        # Register each modality's data matrix
        for registry_name, mod_key in modalities.items():
            anndata_fields.append(
                fields.MuDataLayerField(
                    f"X_{registry_name}",
                    layer=None,
                    mod_key=mod_key,
                    is_count_data=True,
                )
            )

        # Batch key (shared across modalities, still needed for dispersion/library)
        anndata_fields.append(
            fields.MuDataCategoricalObsField(
                REGISTRY_KEYS.BATCH_KEY,
                batch_key,
                mod_key=list(modalities.values())[0],
            )
        )

        # Dispersion key (controls per-group px_r, separate from batch_key)
        if dispersion_key:
            anndata_fields.append(
                fields.MuDataCategoricalObsField(
                    DISPERSION_KEY,
                    dispersion_key,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Library size key (controls per-group library prior, separate from batch_key)
        if library_size_key:
            anndata_fields.append(
                fields.MuDataCategoricalObsField(
                    LIBRARY_SIZE_KEY,
                    library_size_key,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Ambient covariates (for additive background)
        if ambient_covariate_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    AMBIENT_COVS_KEY,
                    ambient_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Covariates
        if nn_conditioning_covariate_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY,
                    nn_conditioning_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )
        if nn_continuous_covariate_keys:
            anndata_fields.append(
                fields.MuDataNumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY,
                    nn_continuous_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Modality scaling covariates (separate from encoder/decoder covariates)
        if feature_scaling_covariate_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    FEATURE_SCALING_COVS_KEY,
                    feature_scaling_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Encoder covariates (dedicated key, default=None → no encoder categoricals)
        if _encoder_cov_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    ENCODER_COVS_KEY,
                    _encoder_cov_keys,
                    mod_key=list(modalities.values())[0],
                )
            )

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata=None,
        indices=None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return the latent representation for each cell.

        Parameters
        ----------
        adata
            MuData to use. Defaults to the MuData used to initialize the model.
        indices
            Indices of cells to use.
        give_mean
            If True, return the mean of the posterior. If False, return a sample.
        batch_size
            Batch size for data loading.

        Returns
        -------
        np.ndarray of shape (n_cells, n_latent_total).
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            if give_mean and self.module.latent_mode != "single_encoder":
                # For concatenation mode, use per-modality means
                qz = outputs["qz_per_modality"]
                if self.module.latent_mode == "concatenation":
                    z_means = []
                    for name in self.module.modality_names:
                        if name in qz:
                            z_means.append(qz[name].loc)
                        else:
                            z_means.append(
                                torch.zeros(
                                    z.shape[0],
                                    self.module.n_latent_dict[name],
                                    device=z.device,
                                )
                            )
                    z = torch.cat(z_means, dim=-1)
                elif self.module.latent_mode == "weighted_mean":
                    # Use means in mixing
                    z_means = {name: qz[name].loc for name in qz}
                    z = self.module._mix_modalities(
                        z_means,
                        qz,
                        outputs["masks"],
                        tensors[REGISTRY_KEYS.BATCH_KEY],
                    )
            elif give_mean and self.module.latent_mode == "single_encoder":
                qz = outputs["qz_per_modality"]["_joint"]
                z = qz.loc

            latent.append(z.cpu().numpy())

        return np.concatenate(latent, axis=0)

    def get_modality_latents(self, **kwargs) -> dict[str, np.ndarray]:
        """Return per-modality latent representations.

        Calls :meth:`get_latent_representation` to obtain the joint latent
        (n_cells, total_latent), then splits it by modality using the
        per-modality ``n_latent`` configuration.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to :meth:`get_latent_representation`.

        Returns
        -------
        dict mapping modality names to their latent arrays, plus
        ``"__joint__"`` for the full concatenated representation.
        For example::

            {
                "rna": np.ndarray of shape (n_cells, n_latent_rna),
                "atac": np.ndarray of shape (n_cells, n_latent_atac),
                "__joint__": np.ndarray of shape (n_cells, total_latent),
            }

        Raises
        ------
        ValueError
            If the model uses ``latent_mode="weighted_mean"`` (per-modality
            latents share the same dimensions and cannot be meaningfully split).
        """
        if self.module.latent_mode == "weighted_mean":
            raise ValueError(
                "get_modality_latents() is not supported for latent_mode='weighted_mean' "
                "because all modalities share the same latent dimensions. "
                "Use get_latent_representation() instead."
            )

        joint = self.get_latent_representation(**kwargs)
        result = {"__joint__": joint}

        # Split joint latent into per-modality slices
        offset = 0
        for name in self.module.modality_names:
            n_lat = self.module.n_latent_dict[name]
            result[name] = joint[:, offset : offset + n_lat]
            offset += n_lat

        return result

    def plot_training_diagnostics(
        self,
        skip_epochs: int = 80,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot per-modality training diagnostics.

        Creates a grid of training curves with modalities as columns and
        metric types as rows. Only rows with available data are shown.

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
        modality_names = self.module.modality_names
        n_modalities = len(modality_names)

        # Define metric rows: (row_label, key_template_train, key_template_val)
        # Templates use {name} for modality name
        metric_rows = [
            ("Reconstruction loss", "recon_loss_{name}_train", "recon_loss_{name}_validation"),
            ("Pearson r (gene-wise)", "pearson_gene_{name}_train", "pearson_gene_{name}_validation"),
            ("Pearson r (cell-wise)", "pearson_cell_{name}_train", "pearson_cell_{name}_validation"),
            ("KL divergence (Z)", "kl_z_{name}_train", "kl_z_{name}_validation"),
            ("Z variance", "z_var_{name}_train", "z_var_{name}_validation"),
        ]

        # Filter to rows that have at least one available key across modalities
        available_rows = []
        for row_label, train_tmpl, val_tmpl in metric_rows:
            has_any = False
            for name in modality_names:
                train_key = train_tmpl.format(name=name)
                val_key = val_tmpl.format(name=name)
                if train_key in history or val_key in history:
                    has_any = True
                    break
            if has_any:
                available_rows.append((row_label, train_tmpl, val_tmpl))

        # Check for ELBO row (not per-modality)
        has_elbo = "elbo_train" in history or "elbo_validation" in history

        n_metric_rows = len(available_rows)
        n_total_rows = n_metric_rows + (1 if has_elbo else 0)

        if n_total_rows == 0:
            raise ValueError("No training metrics found in model.history_. Train the model first.")

        if figsize is None:
            figsize = (4.0 * n_modalities, 3.0 * n_total_rows)

        fig, axes = plt.subplots(
            n_total_rows,
            n_modalities,
            figsize=figsize,
            squeeze=False,
        )

        # Plot per-modality metric rows
        for row_idx, (row_label, train_tmpl, val_tmpl) in enumerate(available_rows):
            for col_idx, name in enumerate(modality_names):
                ax = axes[row_idx, col_idx]
                train_key = train_tmpl.format(name=name)
                val_key = val_tmpl.format(name=name)

                plotted = False
                if train_key in history:
                    df = history[train_key]
                    values = df.iloc[skip_epochs:].values.ravel().astype(float)
                    epochs = range(skip_epochs, skip_epochs + len(values))
                    ax.plot(epochs, values, color="tab:blue", label="train")
                    plotted = True
                if val_key in history:
                    df = history[val_key]
                    values = df.iloc[skip_epochs:].values.ravel().astype(float)
                    epochs = range(skip_epochs, skip_epochs + len(values))
                    ax.plot(epochs, values, color="tab:orange", label="validation")
                    plotted = True

                if not plotted:
                    ax.set_visible(False)
                else:
                    if row_idx == 0:
                        ax.set_title(name)
                    if col_idx == 0:
                        ax.set_ylabel(row_label)
                    ax.set_xlabel("Epoch")
                    ax.legend(fontsize="small")

        # Plot ELBO row (spans all columns, but use leftmost subplot)
        if has_elbo:
            elbo_row = n_metric_rows
            # Hide all but the first axis in the ELBO row
            for col_idx in range(n_modalities):
                if col_idx > 0:
                    axes[elbo_row, col_idx].set_visible(False)

            ax = axes[elbo_row, 0]
            if "elbo_train" in history:
                df = history["elbo_train"]
                values = df.iloc[skip_epochs:].values.ravel()
                epochs = range(skip_epochs, skip_epochs + len(values))
                ax.plot(epochs, values, color="tab:blue", label="train")
            if "elbo_validation" in history:
                df = history["elbo_validation"]
                values = df.iloc[skip_epochs:].values.ravel()
                epochs = range(skip_epochs, skip_epochs + len(values))
                ax.plot(epochs, values, color="tab:orange", label="validation")
            ax.set_ylabel("Total ELBO")
            ax.set_xlabel("Epoch")
            ax.legend(fontsize="small")

        fig.tight_layout()
        return fig

    @torch.no_grad()
    def get_modality_attribution(
        self,
        adata=None,
        indices=None,
        batch_size: int | None = None,
        eps: float = 1e-3,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute decoder Jacobian attribution per modality.

        For each modality's decoder, computes the mean absolute Jacobian
        ``|d(px_rate)/d(z)|`` per latent dimension via finite differences,
        revealing which Z dimensions each decoder actually uses. This allows
        creating modality-specific views of the shared latent space.

        Uses forward finite differences with ``n_latent`` decoder forward passes
        per modality (e.g. 192 passes for 192 latent dims).

        Parameters
        ----------
        adata
            MuData to use. Defaults to the MuData used to initialize the model.
        indices
            Indices of cells to use.
        batch_size
            Batch size for data loading.
        eps
            Finite difference step size for Jacobian approximation.

        Returns
        -------
        dict mapping modality names to dicts with keys:
            ``'attribution'``: np.ndarray (n_cells, n_latent) — mean |Jacobian| per dim
            ``'weighted_z'``: np.ndarray (n_cells, n_latent) — z * attribution
        """
        from torch.nn.functional import one_hot

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        self.module.eval()
        device = next(self.module.parameters()).device
        n_latent = self.module.total_latent_dim

        all_z = []
        importances = {name: [] for name in self.module.modality_names}

        for tensors in scdl:
            # Move all tensors to model device (dataloader returns CPU tensors)
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

            # Run inference to get z (posterior means)
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            # Use posterior means for stable Jacobian
            qz = outputs["qz_per_modality"]
            if self.module.latent_mode == "concatenation":
                z_means = []
                for name in self.module.modality_names:
                    if name in qz:
                        z_means.append(qz[name].loc)
                    else:
                        z_means.append(
                            torch.zeros(
                                outputs["z"].shape[0],
                                self.module.n_latent_dict[name],
                                device=outputs["z"].device,
                            )
                        )
                z = torch.cat(z_means, dim=-1)
            elif self.module.latent_mode == "single_encoder":
                z = qz["_joint"].loc
            else:
                z = outputs["z"]

            all_z.append(z.cpu().numpy())

            library = outputs["library"]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
            cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
            ambient_covs = tensors.get(AMBIENT_COVS_KEY)

            # Prepare categorical inputs (independent of z)
            if cat_covs is not None:
                categorical_input = torch.split(cat_covs, 1, dim=1)
            else:
                categorical_input = ()

            # Build scaling covariate indicator (cell2location obs2extra_categoricals)
            feature_scaling_covs = tensors.get(FEATURE_SCALING_COVS_KEY)
            if feature_scaling_covs is not None and self.module.n_total_feature_scaling_cats > 0:
                feature_scaling_indicator = torch.cat(
                    [
                        one_hot(feature_scaling_covs[:, i].long(), n_cats).float()
                        for i, n_cats in enumerate(self.module.n_cats_per_feature_scaling_cov)
                    ],
                    dim=-1,
                )
            else:
                feature_scaling_indicator = None

            def _make_decode_rate_fn(
                module,
                name,
                disp,
                lib,
                batch_index,
                categorical_input,
                bg,
                cont_covs,
                feature_scaling_indicator,
            ):
                """Create a closure that computes decoder rate for a given z."""

                def _decode_rate(z_input):
                    if cont_covs is None:
                        dec_input = z_input
                    elif z_input.dim() != cont_covs.dim():
                        dec_input = torch.cat(
                            [z_input, cont_covs.unsqueeze(0).expand(z_input.size(0), -1, -1)],
                            dim=-1,
                        )
                    else:
                        dec_input = torch.cat([z_input, cont_covs], dim=-1)

                    if module.use_batch_in_decoder:
                        _, _, px_rate, _ = module.decoders[name](
                            disp,
                            dec_input,
                            lib,
                            batch_index,
                            *categorical_input,
                            additive_background=bg,
                        )
                    else:
                        _, _, px_rate, _ = module.decoders[name](
                            disp,
                            dec_input,
                            lib,
                            *categorical_input,
                            additive_background=bg,
                        )

                    if name in module.feature_scaling:
                        rf_transformed = torch.nn.functional.softplus(module.feature_scaling[name]) / 0.7
                        if feature_scaling_indicator is not None:
                            scaling = torch.matmul(feature_scaling_indicator, rf_transformed)
                        else:
                            scaling = rf_transformed
                        px_rate = px_rate * scaling

                    return px_rate

                return _decode_rate

            for name in self.module.modality_names:
                if name not in library:
                    continue

                disp = self.module.dispersion_dict[name]
                lib = library[name]

                # Additive background (independent of z)
                bg = None
                if name in self.module.additive_background and ambient_covs is not None:
                    concat_ambient = torch.cat(
                        [
                            one_hot(ambient_covs[:, i].long(), int(n_cats_i)).float()
                            for i, n_cats_i in enumerate(self.module.n_cats_per_ambient_cov)
                        ],
                        dim=-1,
                    )
                    bg = torch.matmul(concat_ambient, torch.exp(self.module.additive_background[name]).T)

                decode_rate = _make_decode_rate_fn(
                    self.module,
                    name,
                    disp,
                    lib,
                    batch_index,
                    categorical_input,
                    bg,
                    cont_covs,
                    feature_scaling_indicator,
                )

                # Baseline decoder rate
                base_rate = decode_rate(z)

                # Jacobian column-wise via finite differences
                importance = torch.zeros(z.shape[0], n_latent, device=z.device)
                for j in range(n_latent):
                    z_perturbed = z.clone()
                    z_perturbed[:, j] += eps
                    rate_perturbed = decode_rate(z_perturbed)
                    jac_col = (rate_perturbed - base_rate) / eps
                    importance[:, j] = jac_col.abs().mean(dim=-1)

                importances[name].append(importance.cpu().numpy())

        z_all = np.concatenate(all_z, axis=0)

        result = {}
        for name in self.module.modality_names:
            if importances[name]:
                attr = np.concatenate(importances[name], axis=0)
                result[name] = {
                    "attribution": attr,
                    "weighted_z": z_all * attr,
                }

        return result

    def compute_latent_umap(
        self,
        adata,
        n_neighbors: int = 50,
        min_dist: float = 0.4,
        spread: float = 1.3,
        per_modality: bool = True,
        add_leiden: bool = False,
        leiden_resolution: float = 1.0,
    ) -> None:
        """Compute joint and per-modality latent representations + UMAPs.

        Stores results in ``adata.obsm``:
        - ``X_multiVI_joint``, ``X_umap_joint``
        - ``X_multiVI_{modality}``, ``X_umap_{modality}`` (if ``per_modality=True``)

        If latent representations are already stored in ``adata.obsm``
        (e.g. from a previous call to ``get_modality_latents()``), the
        GPU forward pass is skipped and only KNN/UMAP is computed. This
        enables a split workflow: compute latents on GPU, save ``adata``,
        then compute UMAPs on CPU.

        Parameters
        ----------
        adata
            AnnData object to store results in (typically the first modality's
            ``adata``, e.g. ``mdata["rna"]``).
        n_neighbors
            Number of neighbors for KNN graph.
        min_dist
            UMAP min_dist parameter.
        spread
            UMAP spread parameter.
        per_modality
            If True, also compute per-modality UMAPs.
        add_leiden
            If True, compute Leiden clustering on joint UMAP.
        leiden_resolution
            Resolution for Leiden clustering.
        """
        import scanpy as sc

        # GPU part: skip if latents already stored (enables CPU-only workflow)
        if "X_multiVI_joint" not in adata.obsm:
            latent_dict = self.get_modality_latents()
            adata.obsm["X_multiVI_joint"] = latent_dict["__joint__"]
            if per_modality:
                for name in self.module.modality_names:
                    adata.obsm[f"X_multiVI_{name}"] = latent_dict[name]

        sc.pp.neighbors(
            adata, use_rep="X_multiVI_joint", n_neighbors=n_neighbors, metric="euclidean", key_added="joint"
        )
        sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key="joint")
        adata.obsm["X_umap_joint"] = adata.obsm["X_umap"].copy()

        if add_leiden:
            sc.tl.leiden(adata, resolution=leiden_resolution, flavor="igraph", neighbors_key="joint")

        # Per-modality UMAPs (CPU-only: latents already in obsm)
        if per_modality:
            for name in self.module.modality_names:
                if f"X_multiVI_{name}" in adata.obsm:
                    sc.pp.neighbors(
                        adata,
                        use_rep=f"X_multiVI_{name}",
                        n_neighbors=n_neighbors,
                        metric="euclidean",
                        key_added=name,
                    )
                    sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key=name)
                    adata.obsm[f"X_umap_{name}"] = adata.obsm["X_umap"].copy()

        # Set X_umap to joint UMAP (sc.tl.umap leaves the last one computed)
        adata.obsm["X_umap"] = adata.obsm["X_umap_joint"]

    def store_attribution_results(
        self,
        adata,
        attribution: dict | None = None,
        batch_size: int = 256,
    ) -> dict:
        """Compute attribution and store results in ``adata``.

        Stores attribution-weighted latents in ``adata.obsm`` and per-modality
        importance scores in ``adata.obs``. Does **not** compute KNN/UMAP —
        use :meth:`compute_attribution_umap` for that (can run on CPU separately).

        For each modality stores:

        - ``X_multiVI_attr_{name}`` in obsm — attribution-weighted latent
        - ``{name}_decoder_total_attr`` in obs — total attribution per cell
        - ``{name}_decoder_own_attr`` in obs — attribution from own Z dims

        If exactly 2 modalities, also stores ``log2_{mod0}_vs_{mod1}_attr``.

        Parameters
        ----------
        adata
            AnnData object to store results in.
        attribution
            Pre-computed attribution dict from :meth:`get_modality_attribution`.
            If None, computes it (requires GPU).
        batch_size
            Batch size for attribution computation (if not pre-computed).

        Returns
        -------
        dict
            The attribution dict.
        """
        import numpy as np

        if attribution is None:
            attribution = self.get_modality_attribution(batch_size=batch_size)

        # Store attribution-weighted latents
        for name in self.module.modality_names:
            adata.obsm[f"X_multiVI_attr_{name}"] = attribution[name]["weighted_z"]

        # Per-modality importance scores
        total_attrs = {}
        for name in self.module.modality_names:
            attr = attribution[name]["attribution"]
            total = attr.sum(axis=1)
            adata.obs[f"{name}_decoder_total_attr"] = total
            total_attrs[name] = total

            # Own-modality attribution: find column offset for this modality's Z dims
            offset = 0
            for mod_name in self.module.modality_names:
                n = self.module.n_latent_dict[mod_name]
                if mod_name == name:
                    adata.obs[f"{name}_decoder_own_attr"] = attr[:, offset : offset + n].sum(axis=1)
                    break
                offset += n

        # Log2 ratio for 2-modality case
        if len(self.module.modality_names) == 2:
            mod0, mod1 = self.module.modality_names
            adata.obs[f"log2_{mod0}_vs_{mod1}_attr"] = np.log2(total_attrs[mod0] / (total_attrs[mod1] + 1e-10))

        return attribution

    def compute_attribution_umap(
        self,
        adata,
        n_neighbors: int = 50,
        min_dist: float = 0.4,
        spread: float = 1.3,
    ) -> None:
        """Compute KNN graphs and UMAPs on attribution-weighted latents.

        Requires :meth:`store_attribution_results` to have been called first,
        which populates ``X_multiVI_attr_{name}`` keys in ``adata.obsm``.

        For each modality stores ``X_umap_attr_{name}`` in ``adata.obsm``.

        Parameters
        ----------
        adata
            AnnData with ``X_multiVI_attr_{name}`` keys in ``.obsm``.
        n_neighbors
            Number of neighbors for KNN graph.
        min_dist
            UMAP min_dist parameter.
        spread
            UMAP spread parameter.
        """
        import scanpy as sc

        for name in self.module.modality_names:
            obsm_key = f"X_multiVI_attr_{name}"
            if obsm_key not in adata.obsm:
                msg = f"{obsm_key!r} not found in adata.obsm. Call store_attribution_results() first."
                raise KeyError(msg)
            sc.pp.neighbors(
                adata,
                use_rep=obsm_key,
                n_neighbors=n_neighbors,
                metric="euclidean",
                key_added=f"attr_{name}",
            )
            sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key=f"attr_{name}")
            adata.obsm[f"X_umap_attr_{name}"] = adata.obsm["X_umap"].copy()

        # Restore X_umap to joint UMAP if available
        if "X_umap_joint" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_umap_joint"]

    def save_analysis_outputs(
        self,
        output_dir: str,
        adata,
        n_neighbors: int = 50,
        save_latents: bool = True,
        save_umaps: bool = True,
        save_knn_graphs: bool = True,
        save_attribution: bool = False,
        attribution: dict | None = None,
    ) -> list[str]:
        """Save multimodal analysis outputs to CSV/npz files.

        Parameters
        ----------
        output_dir
            Directory to save outputs. Created if it doesn't exist.
        adata
            AnnData object with computed latents, UMAPs, etc.
        n_neighbors
            Number of neighbors (used in output filenames).
        save_latents
            Save latent representations (joint + per-modality) to CSV.
        save_umaps
            Save UMAP coordinates to CSV.
        save_knn_graphs
            Save KNN distance/connectivity matrices to npz.
        save_attribution
            Save attribution data to CSV.
        attribution
            Attribution dict from ``get_modality_attribution()``.

        Returns
        -------
        list[str]
            List of saved file paths.
        """
        import os

        import pandas as pd
        import scipy.sparse

        os.makedirs(output_dir, exist_ok=True)
        saved = []

        # Latent representations
        if save_latents:
            latent_keys = [("X_multiVI_joint", "joint")]
            for name in self.module.modality_names:
                latent_keys.append((f"X_multiVI_{name}", name))
            for obsm_key, label in latent_keys:
                if obsm_key in adata.obsm:
                    path = f"{output_dir}/X_multiVI_{label}.csv"
                    pd.DataFrame(
                        adata.obsm[obsm_key],
                        index=adata.obs_names,
                        columns=range(adata.obsm[obsm_key].shape[1]),
                    ).to_csv(path)
                    saved.append(path)

        # UMAP coordinates
        if save_umaps:
            umap_keys = [("X_umap_joint", "joint")]
            for name in self.module.modality_names:
                umap_keys.append((f"X_umap_{name}", name))
                umap_keys.append((f"X_umap_attr_{name}", f"attr_{name}"))
            for obsm_key, label in umap_keys:
                if obsm_key in adata.obsm:
                    path = f"{output_dir}/X_umap_{label}_k{n_neighbors}.csv"
                    pd.DataFrame(
                        adata.obsm[obsm_key],
                        index=adata.obs_names,
                        columns=range(2),
                    ).to_csv(path)
                    saved.append(path)

        # KNN graphs (joint only — stored with key_added="joint" prefix)
        if save_knn_graphs:
            for key in ("distances", "connectivities"):
                obsp_key = f"joint_{key}"
                if obsp_key in adata.obsp:
                    path = f"{output_dir}/{key}_euclidean_k{n_neighbors}.npz"
                    scipy.sparse.save_npz(path, adata.obsp[obsp_key], compressed=True)
                    saved.append(path)

        # Attribution data
        if save_attribution and attribution is not None:
            for name in self.module.modality_names:
                if name in attribution:
                    for key in ("attribution", "weighted_z"):
                        data = attribution[name][key]
                        path = f"{output_dir}/attribution_{name}_{key}.csv"
                        pd.DataFrame(
                            data,
                            index=adata.obs_names,
                            columns=range(data.shape[1]),
                        ).to_csv(path)
                        saved.append(path)

        return saved

    def plot_modality_attribution(
        self,
        attribution: dict | None = None,
        batch_size: int = 256,
        figsize: tuple[float, float] = (14, 5),
    ) -> tuple[dict, matplotlib.figure.Figure]:
        """Plot attribution bar chart showing per-modality decoder sensitivity.

        Parameters
        ----------
        attribution
            Pre-computed attribution dict. If None, computes via
            ``get_modality_attribution()``.
        batch_size
            Batch size for attribution computation (if not pre-computed).
        figsize
            Figure size for the bar chart.

        Returns
        -------
        tuple[dict, Figure]
            ``(attribution_dict, figure)``
        """
        from matplotlib.patches import Patch

        if attribution is None:
            attribution = self.get_modality_attribution(batch_size=batch_size)

        n_modalities = len(self.module.modality_names)
        fig, axes = plt.subplots(1, n_modalities, figsize=figsize)
        if n_modalities == 1:
            axes = [axes]

        # Color palette for modalities
        mod_colors = {}
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, name in enumerate(self.module.modality_names):
            mod_colors[name] = palette[i % len(palette)]

        for ax, name in zip(axes, self.module.modality_names, strict=False):
            attr = attribution[name]["attribution"]
            mean_attr = attr.mean(axis=0)

            # Color bars by which modality's Z dims they belong to
            colors = []
            for mod_name in self.module.modality_names:
                n = self.module.n_latent_dict[mod_name]
                colors.extend([mod_colors[mod_name]] * n)

            ax.bar(range(len(mean_attr)), mean_attr, color=colors, alpha=0.7)
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel("Mean |Jacobian|")
            ax.set_title(f"{name.upper()} decoder attribution")

            # Add modality boundary lines
            offset = 0
            for mod_name in self.module.modality_names[:-1]:
                offset += self.module.n_latent_dict[mod_name]
                ax.axvline(offset - 0.5, color="red", linestyle="--", alpha=0.5)

            # Legend
            legend_elements = [
                Patch(facecolor=mod_colors[m], alpha=0.7, label=f"Z_{m}") for m in self.module.modality_names
            ]
            ax.legend(handles=legend_elements)

        fig.tight_layout()

        # Print cross-modality attribution fractions
        for name in self.module.modality_names:
            attr = attribution[name]["attribution"]
            total = attr.mean(axis=0).sum()
            offset = 0
            for mod_name in self.module.modality_names:
                n = self.module.n_latent_dict[mod_name]
                frac = attr.mean(axis=0)[offset : offset + n].sum() / total
                logger.info(f"{name.upper()} decoder: {frac:.1%} from Z_{mod_name}")
                offset += n

        return attribution, fig

    def plot_attribution_scatter(
        self,
        adata,
        basis: str = "X_umap_joint",
        figsize_per_panel: tuple[float, float] = (5, 5),
        size: float = 2,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Plot per-cell attribution values on UMAP.

        Requires :meth:`store_attribution_results` to have been called first.
        Shows one panel per modality (total_attr) plus log2 ratio for 2-modality case.

        Parameters
        ----------
        adata
            AnnData with attribution columns in ``.obs``.
        basis
            Embedding key in ``adata.obsm`` to use as coordinates.
        figsize_per_panel
            Size per panel.
        size
            Point size.
        **kwargs
            Passed to ``sc.pl.embedding``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import scanpy as sc

        names = self.module.modality_names
        color_keys = [f"{name}_decoder_total_attr" for name in names]
        # Check required columns exist
        for key in color_keys:
            if key not in adata.obs:
                raise ValueError(f"{key!r} not found in adata.obs. Call store_attribution_results() first.")

        # Add log2 ratio panel for 2-modality case
        has_ratio = len(names) == 2 and f"log2_{names[0]}_vs_{names[1]}_attr" in adata.obs
        if has_ratio:
            color_keys.append(f"log2_{names[0]}_vs_{names[1]}_attr")

        n_panels = len(color_keys)
        fig, axes = plt.subplots(1, n_panels, figsize=(figsize_per_panel[0] * n_panels, figsize_per_panel[1]))
        if n_panels == 1:
            axes = [axes]

        for ax, key in zip(axes, color_keys, strict=False):
            plot_kwargs = {"size": size, "show": False, "ax": ax, **kwargs}
            if "log2_" in key:
                plot_kwargs.setdefault("vcenter", 0)
                plot_kwargs.setdefault("cmap", "RdBu_r")
            sc.pl.embedding(adata, basis=basis, color=key, **plot_kwargs)

        fig.tight_layout()
        return fig

    def plot_umap_comparison(
        self,
        adata,
        color: str | list[str] = "l2_cell_type",
        umap_keys: list[tuple[str, str]] | None = None,
        size: float = 2,
        show_legend: bool = False,
        figsize_per_panel: tuple[float, float] = (7, 7),
    ) -> matplotlib.figure.Figure:
        """Side-by-side UMAP comparison across latent representations.

        Parameters
        ----------
        adata
            AnnData object with precomputed UMAP embeddings in ``.obsm``.
        color
            Column(s) in ``adata.obs`` to color by.
        umap_keys
            List of ``(obsm_key, title)`` tuples. If None, auto-detects
            available UMAP keys.
        size
            Point size in UMAP plots.
        show_legend
            Whether to show legend on each panel.
        figsize_per_panel
            Size of each individual panel.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import scanpy as sc

        if umap_keys is None:
            umap_keys = []
            for key in adata.obsm:
                if key.startswith("X_umap"):
                    title = key.replace("X_umap_", "").replace("X_umap", "UMAP")
                    umap_keys.append((key, title))

        if isinstance(color, str):
            color = [color]

        n_umaps = len(umap_keys)
        n_colors = len(color)
        fig, axes = plt.subplots(
            n_colors,
            n_umaps,
            figsize=(figsize_per_panel[0] * n_umaps, figsize_per_panel[1] * n_colors),
            squeeze=False,
        )

        for row_idx, color_key in enumerate(color):
            for col_idx, (umap_key, title) in enumerate(umap_keys):
                ax = axes[row_idx, col_idx]
                sc.pl.embedding(
                    adata,
                    basis=umap_key,
                    color=color_key,
                    size=size,
                    ax=ax,
                    show=False,
                    title=f"{title} - {color_key}" if n_colors > 1 else title,
                    legend_loc="right margin" if show_legend else "none",
                )

        fig.tight_layout()
        return fig
