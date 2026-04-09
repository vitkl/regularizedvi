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
import math
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
    DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
    DEFAULT_FEATURE_SCALING_PRIOR_BETA,
    DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
    DEFAULT_LIBRARY_N_HIDDEN,
    DEFAULT_MODALITY_SCALE_PRIOR_CONCENTRATION,
    DEFAULT_REGULARISE_BACKGROUND,
    DEFAULT_REGULARISE_DISPERSION,
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
from regularizedvi._training_plan import MultimodalTrainingPlan

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd


logger = logging.getLogger(__name__)


class RegularizedMultimodalVI(
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    _training_plan_cls = MultimodalTrainingPlan

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
        Scale for library prior variance. Float (all modalities) or dict per
        modality (e.g. ``{"rna": 0.2, "atac": 1.5}``). Default 0.05.
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
        library_log_vars_weight: float | dict[str, float] = DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
        library_log_means_centering_sensitivity: dict[str, float] | float | None = None,
        library_n_hidden: int = DEFAULT_LIBRARY_N_HIDDEN,
        scale_activation: str = DEFAULT_SCALE_ACTIVATION,
        use_batch_in_decoder: bool = DEFAULT_USE_BATCH_IN_DECODER,
        additive_background_modalities: list[str] | None = None,
        feature_scaling_modalities: list[str] | None = None,
        feature_scaling_prior_alpha: float = DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
        feature_scaling_prior_beta: float = DEFAULT_FEATURE_SCALING_PRIOR_BETA,
        regularise_dispersion: bool = DEFAULT_REGULARISE_DISPERSION,
        regularise_dispersion_prior: dict[str, float] | float | None = None,
        dispersion_hyper_prior_alpha: dict[str, float] | float | None = None,
        dispersion_hyper_prior_beta: dict[str, float] | float | None = None,
        additive_bg_prior_alpha: float = DEFAULT_ADDITIVE_BG_PRIOR_ALPHA,
        additive_bg_prior_beta: float = DEFAULT_ADDITIVE_BG_PRIOR_BETA,
        regularise_background: bool = DEFAULT_REGULARISE_BACKGROUND,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_BATCH_NORM,
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_LAYER_NORM,
        compute_pearson: bool = DEFAULT_COMPUTE_PEARSON,
        learnable_modality_scaling: bool = False,
        modality_scale_prior_concentration: float = DEFAULT_MODALITY_SCALE_PRIOR_CONCENTRATION,
        # Decoder weight regularization
        decoder_weight_l2: float = 0.1,
        decoder_cov_weight_l2: float = 0.0,
        # Data-dependent initialization
        init_decoder_bias: str | None = "mean",
        bg_init_gene_fraction: float | None = 0.2,
        decoder_bias_multiplier: dict[str, float] | None = None,
        # Residual library encoder
        residual_library_encoder: bool = True,
        library_obs_w_prior_rate: float = 1.0,
        # Data-driven dispersion initialization
        dispersion_init: Literal["prior", "data", "variance_burst_size"] = "prior",
        dispersion_init_bio_frac: float = 0.9,
        dispersion_init_theta_min: float = 0.01,
        dispersion_init_theta_max: float = 10.0,
        # Bursting model decoder type (per modality)
        decoder_type: dict[str, str] | str = "expected_RNA",
        burst_size_intercept: dict[str, float] | float = 1.0,
        burst_size_n_hidden: dict[str, int] | int | None = None,
        # Sparsity priors
        z_sparsity_prior: str | None = None,
        n_active_latent_per_cell: float = 20.0,
        decoder_hidden_l1: float = 0.0,
        hidden_activation_sparsity: bool = False,
        n_active_hidden_per_cell: float = 40.0,
        use_kl_z: bool = True,
        # Horseshoe latent Z prior (multiplicative shrinkage on Z)
        horseshoe_latent_z_prior_type: str | None = None,
        horseshoe_latent_z_encoder_fraction: float = 1.0,
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

        # Store data-dependent init params for _init_module
        self._init_decoder_bias = init_decoder_bias
        self._bg_init_gene_fraction = bg_init_gene_fraction
        self._decoder_bias_multiplier = decoder_bias_multiplier
        self._dispersion_init = dispersion_init
        self._dispersion_init_bio_frac = dispersion_init_bio_frac
        self._dispersion_init_theta_min = dispersion_init_theta_min
        self._dispersion_init_theta_max = dispersion_init_theta_max

        # B4 validation: dispersion_init="data" is incompatible with burst_frequency_size
        # decoders (which require variance_burst_size for proper burst-parameter init).
        if dispersion_init == "data":
            _dt_check = decoder_type if isinstance(decoder_type, dict) else {"_default": decoder_type}
            _bad = [mn for mn, dt in _dt_check.items() if dt == "burst_frequency_size"]
            if _bad:
                raise ValueError(
                    f"dispersion_init='data' is incompatible with decoder_type='burst_frequency_size' "
                    f"(modalities: {_bad}). Use dispersion_init='variance_burst_size' for burst decoders."
                )

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "latent_mode": latent_mode,
            "modality_weights": modality_weights,
            "dispersion": dispersion,
            "library_log_vars_weight": library_log_vars_weight,
            "library_log_means_centering_sensitivity": library_log_means_centering_sensitivity,
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
            "learnable_modality_scaling": learnable_modality_scaling,
            "modality_scale_prior_concentration": modality_scale_prior_concentration,
            "decoder_weight_l2": decoder_weight_l2,
            "decoder_cov_weight_l2": decoder_cov_weight_l2,
            "residual_library_encoder": residual_library_encoder,
            "library_obs_w_prior_rate": library_obs_w_prior_rate,
            "decoder_type": decoder_type,
            "burst_size_intercept": burst_size_intercept,
            "burst_size_n_hidden": burst_size_n_hidden,
            "z_sparsity_prior": z_sparsity_prior,
            "n_active_latent_per_cell": n_active_latent_per_cell,
            "decoder_hidden_l1": decoder_hidden_l1,
            "hidden_activation_sparsity": hidden_activation_sparsity,
            "n_active_hidden_per_cell": n_active_hidden_per_cell,
            "use_kl_z": use_kl_z,
            "horseshoe_latent_z_prior_type": horseshoe_latent_z_prior_type,
            "horseshoe_latent_z_encoder_fraction": horseshoe_latent_z_encoder_fraction,
            **kwargs,
        }

        if hidden_activation_sparsity and any(
            v > 1
            for v in (
                self._module_kwargs["n_layers"].values()
                if isinstance(self._module_kwargs["n_layers"], dict)
                else [self._module_kwargs["n_layers"]]
            )
        ):
            warnings.warn(
                "hidden_activation_sparsity=True with n_layers>1: the sparsity penalty is applied "
                "to the LAST hidden layer activations only. Multi-layer decoders may not behave as "
                "expected — consider using n_layers=1.",
                UserWarning,
                stacklevel=2,
            )

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
        early_stopping_min_delta_per_feature: float | None = None,
        **trainer_kwargs,
    ):
        """Train the model with optional larger validation batch size.

        Parameters
        ----------
        validation_batch_size
            Batch size for the validation DataLoader. If ``None`` (default),
            uses the same ``batch_size`` as training. Larger values (e.g. 8192)
            reduce noise in Pearson correlation metrics computed on validation.
        early_stopping_min_delta_per_feature
            Per-feature scaling factor for auto-computed ``early_stopping_min_delta``.
            The total ``min_delta = n_features * early_stopping_min_delta_per_feature``.
            Only used when ``early_stopping=True`` and ``early_stopping_min_delta``
            is not explicitly set in ``trainer_kwargs``. Default from constants.
        **kwargs
            All other arguments are passed to
            :meth:`~scvi.model.base.UnsupervisedTrainingMixin.train`.
        """
        from regularizedvi._constants import DEFAULT_EARLY_STOPPING_MIN_DELTA_PER_FEATURE

        datasplitter_kwargs = datasplitter_kwargs or {}
        if validation_batch_size is not None:
            datasplitter_kwargs.setdefault("val_batch_size", validation_batch_size)

        # Auto-scale early_stopping_min_delta by total n_features if not explicitly set
        if early_stopping and "early_stopping_min_delta" not in trainer_kwargs:
            _per_feat = early_stopping_min_delta_per_feature or DEFAULT_EARLY_STOPPING_MIN_DELTA_PER_FEATURE
            n_features = sum(self.summary_stats.get(f"n_X_{name}", 0) for name in self.module.modality_names)
            trainer_kwargs["early_stopping_min_delta"] = n_features * _per_feat

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

        # Data-dependent initialization (Streams B+C): per-modality bias init + bg init
        import scipy.sparse as sp

        decoder_bias_init_dict = {}
        bg_init_per_gene_dict = {}
        if self._init_decoder_bias is not None or self._bg_init_gene_fraction is not None:
            for name in modality_names:
                reg_key = f"X_{name}"
                data_mod = self.adata_manager.get_from_registry(reg_key)
                if sp.issparse(data_mod):
                    data_mod = data_mod.astype(np.float32)
                else:
                    data_mod = np.asarray(data_mod, dtype=np.float32)
                lib_sizes = np.array(data_mod.sum(axis=1)).flatten()
                lib_sizes = np.maximum(lib_sizes, 1.0)
                _centering = self._module_kwargs.get("library_log_means_centering_sensitivity")
                if _centering is None:
                    sensitivity = 1.0
                elif isinstance(_centering, dict):
                    sensitivity = _centering.get(name, 1.0)
                else:
                    sensitivity = float(_centering)
                norm_target = lib_sizes.mean() / sensitivity
                norm_data = (
                    data_mod.multiply(norm_target / lib_sizes[:, None])
                    if sp.issparse(data_mod)
                    else data_mod * (norm_target / lib_sizes[:, None])
                )
                if sp.issparse(norm_data):
                    norm_data = norm_data.tocsc()
                del data_mod

                if self._init_decoder_bias is not None:
                    if self._init_decoder_bias == "mean":
                        decoder_bias_init_dict[name] = np.array(norm_data.mean(axis=0)).flatten().astype(np.float32)
                    elif self._init_decoder_bias == "topN":
                        n_top = min(int(0.01 * norm_data.shape[0]), 500)
                        vals = np.zeros(norm_data.shape[1], dtype=np.float32)
                        for g in range(norm_data.shape[1]):
                            col = (
                                np.array(norm_data[:, g].todense()).flatten()
                                if sp.issparse(norm_data)
                                else norm_data[:, g]
                            )
                            top_vals = np.partition(col, -n_top)[-n_top:]
                            vals[g] = top_vals.mean()
                        decoder_bias_init_dict[name] = vals
                    decoder_bias_init_dict[name] = np.nan_to_num(
                        decoder_bias_init_dict[name], nan=0.01, posinf=0.01, neginf=0.01
                    )
                    # Change 2: optional per-modality decoder bias multiplier — exact in softplus space
                    if self._decoder_bias_multiplier is not None:
                        multiplier = self._decoder_bias_multiplier.get(name, 1.0)
                        if multiplier != 1.0:
                            from regularizedvi._components import _scale_softplus_bias_np

                            decoder_bias_init_dict[name] = _scale_softplus_bias_np(
                                decoder_bias_init_dict[name], float(multiplier)
                            )

                _bg_modalities = self._module_kwargs.get("additive_background_modalities", [])
                if self._bg_init_gene_fraction is not None and name in _bg_modalities:
                    # Batch-specific: mirrors forward pass one-hot encoding of ambient covs
                    ambient_raw = np.asarray(
                        self.adata_manager.get_from_registry(AMBIENT_COVS_KEY)
                        if AMBIENT_COVS_KEY in self.adata_manager.data_registry
                        else self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
                    )
                    if ambient_raw.ndim == 1:
                        ambient_raw = ambient_raw.reshape(-1, 1)
                    _amb_cov = (
                        self.adata_manager.get_state_registry(AMBIENT_COVS_KEY).n_cats_per_key
                        if AMBIENT_COVS_KEY in self.adata_manager.data_registry
                        else None
                    )
                    cats_per_key = list(_amb_cov) if _amb_cov is not None else [n_batch]
                    _n_amb = sum(int(c) for c in cats_per_key)
                    n_feat = norm_data.shape[1]
                    bg_arr = np.full((n_feat, _n_amb), math.log(1e-8), dtype=np.float32)
                    col_offset = 0
                    for key_idx, n_cats_i in enumerate(cats_per_key):
                        key_col = ambient_raw[:, key_idx]
                        for i_cat in range(int(n_cats_i)):
                            cat_idx = np.where(key_col == i_cat)[0]
                            if len(cat_idx) == 0:
                                col_offset += 1
                                continue
                            cat_data = norm_data[cat_idx]
                            mean_expr_cat = np.asarray(cat_data.mean(axis=0)).flatten()
                            del cat_data
                            bg_arr[:, col_offset] = np.log(
                                np.maximum(self._bg_init_gene_fraction * mean_expr_cat, 1e-8)
                            ).astype(np.float32)
                            col_offset += 1
                    bg_init_per_gene_dict[name] = bg_arr

                del norm_data

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

        # Data-driven dispersion initialization (per-modality)
        px_r_init_mean_dict = None
        if self._dispersion_init == "data":
            from regularizedvi._dispersion_init import compute_dispersion_init

            px_r_init_mean_dict = {}
            for mod_name in modality_names:
                mod_adata = self.adata.mod[mod_name]
                logger.info(f"Computing data-driven dispersion init for modality '{mod_name}'...")
                log_theta_init, _diag = compute_dispersion_init(
                    mod_adata,
                    biological_variance_fraction=self._dispersion_init_bio_frac,
                    theta_min=self._dispersion_init_theta_min,
                    theta_max=self._dispersion_init_theta_max,
                    verbose=False,
                )
                px_r_init_mean_dict[mod_name] = log_theta_init
                logger.info(
                    f"  {mod_name}: median theta={np.exp(np.median(log_theta_init)):.3f}, "
                    f"CV²(L)={_diag['cv2_L']:.3f}, sub-Poisson={_diag['n_sub_poisson']}/{len(log_theta_init)}"
                )

        # Bursting model init (per-modality, only for burst_frequency_size modalities)
        _bursting_init_per_modality = {}
        if self._dispersion_init == "variance_burst_size":
            from regularizedvi._constants import DEFAULT_DECODER_TYPE
            from regularizedvi._dispersion_init import compute_bursting_init, compute_dispersion_init

            _decoder_type = self._module_kwargs.get("decoder_type", DEFAULT_DECODER_TYPE)
            _decoder_type_resolved = (
                _decoder_type if isinstance(_decoder_type, dict) else dict.fromkeys(modality_names, _decoder_type)
            )
            _burst_intercept = self._module_kwargs.get("burst_size_intercept", 1.0)
            _burst_intercept_resolved = (
                _burst_intercept
                if isinstance(_burst_intercept, dict)
                else dict.fromkeys(modality_names, _burst_intercept)
            )

            # Resolve per-modality sensitivity
            _centering = self._module_kwargs.get("library_log_means_centering_sensitivity")
            _sensitivity_resolved = {}
            for _mn in modality_names:
                if _centering is None:
                    _sensitivity_resolved[_mn] = 1.0
                elif isinstance(_centering, dict):
                    _sensitivity_resolved[_mn] = _centering.get(_mn, 1.0)
                else:
                    _sensitivity_resolved[_mn] = float(_centering)

            # Resolve per-modality hyper-prior alpha
            _hp_alpha = self._module_kwargs.get("dispersion_hyper_prior_alpha")
            _hp_alpha_resolved = {}
            for _mn in modality_names:
                if _hp_alpha is None:
                    from regularizedvi._constants import DECODER_TYPE_DEFAULTS

                    _dt = _decoder_type_resolved.get(_mn, "expected_RNA")
                    _hp_alpha_resolved[_mn] = DECODER_TYPE_DEFAULTS.get(_dt, DECODER_TYPE_DEFAULTS["expected_RNA"])[
                        "dispersion_hyper_prior_alpha"
                    ]
                elif isinstance(_hp_alpha, dict):
                    _hp_alpha_resolved[_mn] = _hp_alpha.get(_mn, 2.0)
                else:
                    _hp_alpha_resolved[_mn] = float(_hp_alpha)

            px_r_init_mean_dict = px_r_init_mean_dict or {}
            for mod_name in modality_names:
                if _decoder_type_resolved.get(mod_name) == "burst_frequency_size":
                    mod_adata = self.adata.mod[mod_name]
                    logger.info(f"Computing bursting model init for modality '{mod_name}'...")
                    _init_vals, _diag = compute_bursting_init(
                        mod_adata,
                        biological_variance_fraction=self._dispersion_init_bio_frac,
                        burst_size_intercept=_burst_intercept_resolved.get(mod_name, 1.0),
                        sensitivity=_sensitivity_resolved.get(mod_name, 1.0),
                        dispersion_hyper_prior_alpha=_hp_alpha_resolved.get(mod_name, 2.0),
                        theta_min=self._dispersion_init_theta_min,
                        theta_max=self._dispersion_init_theta_max,
                        verbose=True,
                    )
                    px_r_init_mean_dict[mod_name] = _init_vals["log_theta"]
                    _bursting_init_per_modality[mod_name] = _init_vals
                    logger.info(
                        f"  {mod_name}: median burst_freq={np.median(_init_vals['burst_freq']):.3f}, "
                        f"median stochastic_v={np.median(_init_vals['stochastic_v_scale']):.4f}"
                    )
            # Fallback: non-burst modalities get data-driven MoM theta init
            for mod_name in modality_names:
                if mod_name not in px_r_init_mean_dict:
                    mod_adata = self.adata.mod[mod_name]
                    logger.info(f"Computing data-driven dispersion init for non-burst modality '{mod_name}'...")
                    log_theta_init, _diag = compute_dispersion_init(
                        mod_adata,
                        biological_variance_fraction=self._dispersion_init_bio_frac,
                        theta_min=self._dispersion_init_theta_min,
                        theta_max=self._dispersion_init_theta_max,
                        verbose=False,
                    )
                    px_r_init_mean_dict[mod_name] = log_theta_init

        # B4: per-modality hyper-prior routing.
        # Rule: if a modality uses a data-init decoder (e.g. burst_frequency_size) AND
        # dispersion_init is a data-driven mode, user hyper-prior overrides for that
        # modality are IGNORED and replaced with MoM-derived values (warned).
        if _bursting_init_per_modality:
            import warnings as _warnings

            from regularizedvi._constants import (
                AUTO_HYPER_PRIOR_LAMBDA_MAX,
                AUTO_HYPER_PRIOR_LAMBDA_MIN,
                DECODER_TYPE_DEFAULTS,
            )

            _user_hp_beta = self._module_kwargs.get("dispersion_hyper_prior_beta")
            _auto_beta = {}
            _auto_prior = {}
            for _mn in modality_names:
                if _mn in _bursting_init_per_modality:
                    _suggested = _bursting_init_per_modality[_mn]["suggested_hyper_beta"]
                    _suggested = float(
                        np.clip(
                            _suggested,
                            _hp_alpha_resolved[_mn] / AUTO_HYPER_PRIOR_LAMBDA_MAX,
                            _hp_alpha_resolved[_mn] / AUTO_HYPER_PRIOR_LAMBDA_MIN,
                        )
                    )
                    # Determine if a user override exists for this modality
                    _user_val = None
                    if isinstance(_user_hp_beta, dict):
                        _user_val = _user_hp_beta.get(_mn)
                    elif _user_hp_beta is not None:
                        _user_val = float(_user_hp_beta)
                    if _user_val is not None:
                        _warnings.warn(
                            f"[data-init decoder @ modality '{_mn}'] Ignoring user "
                            f"dispersion_hyper_prior_beta={_user_val}; using MoM-derived "
                            f"{_suggested:.4f} instead.",
                            stacklevel=2,
                        )
                    _auto_beta[_mn] = _suggested
                    _auto_prior[_mn] = _hp_alpha_resolved[_mn] / _auto_beta[_mn]
                    logger.info(
                        f"  {_mn}: auto hyper-prior beta={_auto_beta[_mn]:.4f}, lambda_init={_auto_prior[_mn]:.1f}"
                    )
                else:
                    _dt = _decoder_type_resolved.get(_mn, "expected_RNA")
                    _dt_def = DECODER_TYPE_DEFAULTS.get(_dt, DECODER_TYPE_DEFAULTS["expected_RNA"])
                    if isinstance(_user_hp_beta, dict) and _mn in _user_hp_beta:
                        _auto_beta[_mn] = _user_hp_beta[_mn]
                    elif isinstance(_user_hp_beta, (int, float)):
                        _auto_beta[_mn] = float(_user_hp_beta)
                    else:
                        _auto_beta[_mn] = _dt_def["dispersion_hyper_prior_beta"]
                    _auto_prior[_mn] = _hp_alpha_resolved[_mn] / _auto_beta[_mn]

        kwargs = dict(self._module_kwargs)
        # Remove keys that are passed separately
        for k in ["n_hidden", "n_latent", "n_layers", "dropout_rate"]:
            kwargs.pop(k, None)

        # Inject per-modality dispersion init if computed
        if px_r_init_mean_dict is not None:
            kwargs["px_r_init_mean"] = px_r_init_mean_dict

        # Inject auto-derived hyper-prior for burst modalities
        if _bursting_init_per_modality and _auto_beta:
            kwargs["dispersion_hyper_prior_beta"] = _auto_beta
            kwargs["regularise_dispersion_prior"] = _auto_prior

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
            decoder_bias_init=decoder_bias_init_dict or None,
            additive_bg_init_per_gene=bg_init_per_gene_dict or None,
            **kwargs,
        )

        # Apply bursting model init values to px_r (reused as stochastic_v) and decoder biases
        if _bursting_init_per_modality:
            import torch

            for mod_name, _init_vals in _bursting_init_per_modality.items():
                # px_r_mu stores stochastic_v init: px_r = exp(px_r_mu) ≈ v_scale^2
                if mod_name in self.module.px_r_mu:
                    stochastic_v_scale = torch.tensor(_init_vals["stochastic_v_scale"], dtype=torch.float32)
                    sv_log_init = torch.log(torch.clamp(stochastic_v_scale.pow(2), min=1e-8))
                    _px_r_param = self.module.px_r_mu[mod_name]
                    if _px_r_param.dim() == 1:
                        _px_r_param.data.copy_(sv_log_init)
                    else:
                        _px_r_param.data.copy_(sv_log_init.unsqueeze(1).expand_as(_px_r_param))

                # Initialize burst_freq decoder bias (apply multiplier if set)
                bf_bias = torch.tensor(_init_vals["burst_freq_bias"], dtype=torch.float32)
                if self._decoder_bias_multiplier is not None:
                    _mult = (
                        self._decoder_bias_multiplier.get(mod_name, 1.0)
                        if isinstance(self._decoder_bias_multiplier, dict)
                        else float(self._decoder_bias_multiplier)
                    )
                    if _mult != 1.0:
                        from regularizedvi._components import _scale_softplus_bias

                        bf_bias = _scale_softplus_bias(bf_bias, float(_mult))
                        logger.info(f"  Applied decoder_bias_multiplier={_mult} to '{mod_name}' burst_freq bias")
                self.module.decoders[mod_name].px_scale_decoder[0].bias.data.copy_(bf_bias)

                # Initialize burst_size decoder bias
                bs_bias = torch.tensor(_init_vals["burst_size_bias"], dtype=torch.float32)
                self.module.decoders[mod_name].burst_size_head[0].bias.data.copy_(bs_bias)

                logger.info(f"Applied bursting model init to '{mod_name}' px_r (stochastic_v) and decoder biases.")

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
            warnings.warn(
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
        remove_covariates: str | bool = False,
        covariate_reference: str = "zero",
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute decoder Jacobian attribution per modality.

        For each modality's decoder, computes the mean absolute Jacobian
        ``|d(px_rate)/d(z)|`` per latent dimension, revealing which Z
        dimensions each decoder actually uses. This allows creating
        modality-specific views of the shared latent space.

        Uses ``torch.func.vmap(jacfwd(...))`` to compute exact per-cell
        Jacobians, vectorized across the cell batch dimension. Forward-mode
        autodiff is preferred because n_genes >> n_latent. ``eps`` is retained
        for backward compatibility but is no longer used (Jacobian is exact).

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
        remove_covariates
            Which covariates to remove from decoder computation so that the
            Jacobian reflects only the z -> rate mapping without covariate
            confounds. Options:

            - ``False`` (default): keep all covariates as-is (original behavior).
            - ``"cat_covs"``: replace categorical covariates only.
            - ``"feature_scaling"``: replace feature scaling covariates only.
            - ``"all"`` or ``True``: replace both categorical and feature scaling.
        covariate_reference
            How to replace removed covariates. Only used when
            ``remove_covariates`` is not ``False``. Options:

            - ``"zero"``: zero out one-hot encodings (or skip feature scaling).
            - ``"mean"``: use uniform distribution across categories.
            - ``"reference"``: use the most common level per covariate column.

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

        # --- Validate and parse remove_covariates ---
        if remove_covariates is True:
            remove_covariates = "all"
        if remove_covariates not in (False, "cat_covs", "feature_scaling", "all"):
            msg = (
                f"remove_covariates must be False, 'cat_covs', 'feature_scaling', 'all', or True. "
                f"Got {remove_covariates!r}"
            )
            raise ValueError(msg)
        if covariate_reference not in ("zero", "mean", "reference"):
            msg = f"covariate_reference must be 'zero', 'mean', or 'reference'. Got {covariate_reference!r}"
            raise ValueError(msg)

        _remove_cat = remove_covariates in ("cat_covs", "all")
        _remove_fs = remove_covariates in ("feature_scaling", "all")

        # --- For "reference" mode, find most common level per covariate column ---
        _cat_ref_indices = None
        _fs_ref_indices = None
        if covariate_reference == "reference":
            if _remove_cat and REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
                cat_data = np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY))
                _cat_ref_indices = []
                for col_i in range(cat_data.shape[1]):
                    col = cat_data[:, col_i]
                    if hasattr(col, "numpy"):
                        col = col.numpy()
                    vals, counts = np.unique(col, return_counts=True)
                    _cat_ref_indices.append(int(vals[np.argmax(counts)]))

            if _remove_fs and FEATURE_SCALING_COVS_KEY in self.adata_manager.data_registry:
                fs_data = np.asarray(self.adata_manager.get_from_registry(FEATURE_SCALING_COVS_KEY))
                _fs_ref_indices = []
                for col_i in range(fs_data.shape[1]):
                    col = fs_data[:, col_i]
                    if hasattr(col, "numpy"):
                        col = col.numpy()
                    vals, counts = np.unique(col, return_counts=True)
                    _fs_ref_indices.append(int(vals[np.argmax(counts)]))

        # --- Get decoder n_cat_list for building replacement categoricals ---
        _decoder_n_cat_list = None
        if _remove_cat:
            first_mod = self.module.modality_names[0]
            _decoder_n_cat_list = list(self.module.decoders[first_mod].px_decoder.n_cat_list)

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
            bs = z.shape[0]

            # Prepare categorical inputs (independent of z)
            if _remove_cat and _decoder_n_cat_list is not None and cat_covs is not None:
                # Build replacement categorical tensors that bypass one-hot encoding
                # in FCLayers.forward() by passing tensors with size(1) == n_cat
                categorical_input = []
                cov_i = 0
                for n_cat in _decoder_n_cat_list:
                    if n_cat > 1:
                        if covariate_reference == "zero":
                            categorical_input.append(torch.zeros(bs, n_cat, device=device))
                        elif covariate_reference == "mean":
                            categorical_input.append(torch.full((bs, n_cat), 1.0 / n_cat, device=device))
                        elif covariate_reference == "reference":
                            ref_tensor = torch.zeros(bs, n_cat, device=device)
                            ref_idx = _cat_ref_indices[cov_i] if _cat_ref_indices is not None else 0
                            ref_tensor[:, ref_idx] = 1.0
                            categorical_input.append(ref_tensor)
                        cov_i += 1
                    else:
                        # n_cat < 2: pass original (ignored by FCLayers anyway)
                        if cat_covs is not None:
                            categorical_input.append(cat_covs[:, cov_i : cov_i + 1])
                        cov_i += 1
                categorical_input = tuple(categorical_input)
            elif cat_covs is not None:
                categorical_input = torch.split(cat_covs, 1, dim=1)
            else:
                categorical_input = ()

            # Build scaling covariate indicator (cell2location obs2extra_categoricals)
            skip_feature_scaling = False
            feature_scaling_covs = tensors.get(FEATURE_SCALING_COVS_KEY)
            if _remove_fs:
                if covariate_reference == "zero":
                    feature_scaling_indicator = None
                    skip_feature_scaling = True
                elif covariate_reference == "mean":
                    if self.module.n_total_feature_scaling_cats > 0:
                        feature_scaling_indicator = torch.cat(
                            [
                                torch.full((bs, n_cats), 1.0 / n_cats, device=device)
                                for n_cats in self.module.n_cats_per_feature_scaling_cov
                            ],
                            dim=-1,
                        )
                    else:
                        feature_scaling_indicator = None
                elif covariate_reference == "reference":
                    if self.module.n_total_feature_scaling_cats > 0 and _fs_ref_indices is not None:
                        indicator_parts = []
                        for col_i, n_cats in enumerate(self.module.n_cats_per_feature_scaling_cov):
                            ref_oh = torch.zeros(bs, n_cats, device=device)
                            ref_oh[:, _fs_ref_indices[col_i]] = 1.0
                            indicator_parts.append(ref_oh)
                        feature_scaling_indicator = torch.cat(indicator_parts, dim=-1)
                    else:
                        feature_scaling_indicator = None
            elif feature_scaling_covs is not None and self.module.n_total_feature_scaling_cats > 0:
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
                skip_feature_scaling=False,
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
                        _dec_out = module.decoders[name](
                            disp,
                            dec_input,
                            lib,
                            batch_index,
                            *categorical_input,
                            additive_background=bg,
                        )
                    else:
                        _dec_out = module.decoders[name](
                            disp,
                            dec_input,
                            lib,
                            *categorical_input,
                            additive_background=bg,
                        )
                    # px_rate is always the 3rd element (burst_frequency_size returns 6, others 4)
                    px_rate = _dec_out[2]

                    if not skip_feature_scaling and name in module.feature_scaling:
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

                # Additive background (independent of z, d(bg)/dz=0)
                bg = None
                if _remove_cat or remove_covariates == "all":
                    # bg doesn't affect Jacobian (additive, z-independent), skip
                    bg = None
                elif name in self.module.additive_background and ambient_covs is not None:
                    concat_ambient = torch.cat(
                        [
                            one_hot(ambient_covs[:, i].long(), int(n_cats_i)).float()
                            for i, n_cats_i in enumerate(self.module.n_cats_per_ambient_cov)
                        ],
                        dim=-1,
                    )
                    bg = torch.matmul(concat_ambient, torch.exp(self.module.additive_background[name]).T)

                # Exact per-cell Jacobian via torch.func.vmap(jacfwd(...)).
                # jacfwd is preferred over jacrev because n_genes >> n_latent
                # (e.g. 28k vs 192), so forward-mode is ~150x cheaper.
                from torch.func import jacfwd, vmap

                # Pack per-cell tensors and vmap over them. None-valued inputs
                # (bg, cont_covs, feature_scaling_indicator) are replaced with
                # zero-width stand-ins because vmap requires real tensors;
                # _decode_rate_one rebuilds the original None via Python flags.
                _module = self.module
                _name = name
                _disp = disp
                _skip_fs = skip_feature_scaling
                _has_bg = bg is not None
                _has_cc = cont_covs is not None
                _has_fs = feature_scaling_indicator is not None

                bs_z = z.shape[0]
                bg_pack = bg if _has_bg else torch.empty(bs_z, 0, device=z.device)
                cc_pack = cont_covs if _has_cc else torch.empty(bs_z, 0, device=z.device)
                fs_pack = feature_scaling_indicator if _has_fs else torch.empty(bs_z, 0, device=z.device)
                # Pre-one-hot categoricals so FCLayers takes the (bs, n_cat)
                # branch and skips its data-dependent nn.functional.one_hot
                # call (which is not vmap-traceable).
                _dec_n_cat_list = list(self.module.decoders[name].px_decoder.n_cat_list)
                cat_pack_list = []
                _ci = 0
                for n_cat in _dec_n_cat_list:
                    if n_cat > 1:
                        c = categorical_input[_ci]
                        if c.size(1) != n_cat:
                            c = one_hot(c.squeeze(-1).long(), n_cat).float()
                        cat_pack_list.append(c)
                        _ci += 1
                    else:
                        if _ci < len(categorical_input):
                            cat_pack_list.append(categorical_input[_ci])
                            _ci += 1
                cat_pack = tuple(cat_pack_list)

                def _decode_rate_one(
                    z_s,
                    lib_s,
                    bi_s,
                    bg_s,
                    cc_s,
                    fs_s,
                    cat_s,
                    _module=_module,
                    _name=_name,
                    _disp=_disp,
                    _skip_fs=_skip_fs,
                    _has_bg=_has_bg,
                    _has_cc=_has_cc,
                    _has_fs=_has_fs,
                ):
                    lib_i = lib_s.unsqueeze(0)
                    bi_i = bi_s.unsqueeze(0)
                    bg_i = bg_s.unsqueeze(0) if _has_bg else None
                    cc_i = cc_s.unsqueeze(0) if _has_cc else None
                    fs_i = fs_s.unsqueeze(0) if _has_fs else None
                    cat_i = tuple(c.unsqueeze(0) for c in cat_s)
                    one_cell_decode = _make_decode_rate_fn(
                        _module,
                        _name,
                        _disp,
                        lib_i,
                        bi_i,
                        cat_i,
                        bg_i,
                        cc_i,
                        fs_i,
                        skip_feature_scaling=_skip_fs,
                    )
                    return one_cell_decode(z_s.unsqueeze(0)).squeeze(0)

                jac_per_cell = vmap(
                    jacfwd(_decode_rate_one, argnums=0),
                    in_dims=(0, 0, 0, 0, 0, 0, tuple(0 for _ in cat_pack)),
                )
                J = jac_per_cell(z, lib, batch_index, bg_pack, cc_pack, fs_pack, cat_pack)  # (bs, n_genes, n_latent)
                importance = J.abs().mean(dim=1)  # (bs, n_latent)

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
        - ``X_latent_joint``, ``X_umap_joint``
        - ``X_latent_{modality}``, ``X_umap_{modality}`` (if ``per_modality=True``)

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
        if "X_latent_joint" not in adata.obsm:
            latent_dict = self.get_modality_latents()
            adata.obsm["X_latent_joint"] = latent_dict["__joint__"]
            if per_modality:
                for name in self.module.modality_names:
                    adata.obsm[f"X_latent_{name}"] = latent_dict[name]

        sc.pp.neighbors(adata, use_rep="X_latent_joint", n_neighbors=n_neighbors, metric="euclidean", key_added="joint")
        sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key="joint")
        adata.obsm["X_umap_joint"] = adata.obsm["X_umap"].copy()

        if add_leiden:
            sc.tl.leiden(adata, resolution=leiden_resolution, flavor="igraph", neighbors_key="joint")

        # Per-modality UMAPs (CPU-only: latents already in obsm)
        if per_modality:
            for name in self.module.modality_names:
                if f"X_latent_{name}" in adata.obsm:
                    sc.pp.neighbors(
                        adata,
                        use_rep=f"X_latent_{name}",
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
        suffix: str = "",
    ) -> dict:
        """Compute attribution and store results in ``adata``.

        Stores attribution-weighted latents in ``adata.obsm`` and per-modality
        importance scores in ``adata.obs``. Does **not** compute KNN/UMAP —
        use :meth:`compute_attribution_umap` for that (can run on CPU separately).

        For each modality stores:

        - ``X_latent_attr_{name}{suffix}`` in obsm — attribution-weighted latent
        - ``{name}_decoder_total_attr{suffix}`` in obs — total attribution per cell
        - ``{name}_decoder_own_attr{suffix}`` in obs — attribution from own Z dims

        If exactly 2 modalities, also stores ``log2_{mod0}_vs_{mod1}_attr{suffix}``.

        Parameters
        ----------
        adata
            AnnData object to store results in.
        attribution
            Pre-computed attribution dict from :meth:`get_modality_attribution`.
            If None, computes it (requires GPU).
        batch_size
            Batch size for attribution computation (if not pre-computed).
        suffix
            Suffix to append to all stored keys. Use e.g. ``"_nocov"`` to store
            covariate-removed attribution alongside the default results.

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
            adata.obsm[f"X_latent_attr_{name}{suffix}"] = attribution[name]["weighted_z"]

        # Per-modality importance scores
        total_attrs = {}
        for name in self.module.modality_names:
            attr = attribution[name]["attribution"]
            total = attr.sum(axis=1)
            adata.obs[f"{name}_decoder_total_attr{suffix}"] = total
            total_attrs[name] = total

            # Own-modality attribution: find column offset for this modality's Z dims
            offset = 0
            for mod_name in self.module.modality_names:
                n = self.module.n_latent_dict[mod_name]
                if mod_name == name:
                    adata.obs[f"{name}_decoder_own_attr{suffix}"] = attr[:, offset : offset + n].sum(axis=1)
                    break
                offset += n

        # Log2 ratio for 2-modality case
        if len(self.module.modality_names) == 2:
            mod0, mod1 = self.module.modality_names
            adata.obs[f"log2_{mod0}_vs_{mod1}_attr{suffix}"] = np.log2(total_attrs[mod0] / (total_attrs[mod1] + 1e-10))

        return attribution

    def compute_attribution_umap(
        self,
        adata,
        n_neighbors: int = 50,
        min_dist: float = 0.4,
        spread: float = 1.3,
        suffix: str = "",
    ) -> None:
        """Compute KNN graphs and UMAPs on attribution-weighted latents.

        Requires :meth:`store_attribution_results` to have been called first,
        which populates ``X_latent_attr_{name}{suffix}`` keys in ``adata.obsm``.

        For each modality stores ``X_umap_attr_{name}{suffix}`` in ``adata.obsm``.

        Parameters
        ----------
        adata
            AnnData with ``X_latent_attr_{name}{suffix}`` keys in ``.obsm``.
        n_neighbors
            Number of neighbors for KNN graph.
        min_dist
            UMAP min_dist parameter.
        spread
            UMAP spread parameter.
        suffix
            Suffix that was used in :meth:`store_attribution_results`. Must match
            to find the correct obsm keys and store UMAPs with the same suffix.
        """
        import scanpy as sc

        for name in self.module.modality_names:
            obsm_key = f"X_latent_attr_{name}{suffix}"
            if obsm_key not in adata.obsm:
                msg = f"{obsm_key!r} not found in adata.obsm. Call store_attribution_results() first."
                raise KeyError(msg)
            sc.pp.neighbors(
                adata,
                use_rep=obsm_key,
                n_neighbors=n_neighbors,
                metric="euclidean",
                key_added=f"attr_{name}{suffix}",
            )
            sc.tl.umap(adata, min_dist=min_dist, spread=spread, neighbors_key=f"attr_{name}{suffix}")
            adata.obsm[f"X_umap_attr_{name}{suffix}"] = adata.obsm["X_umap"].copy()

        # Restore X_umap to joint UMAP if available
        if "X_umap_joint" in adata.obsm:
            adata.obsm["X_umap"] = adata.obsm["X_umap_joint"]

    @torch.no_grad()
    def get_covariate_effects(
        self,
        adata=None,
        indices=None,
        batch_size: int = 256,
        n_px_scale_batches: int | None = None,
    ) -> dict[str, dict]:
        """Extract and quantify all covariate pathway effects in the model.

        Computes metrics for the 5 purpose-based covariate pathways (decoder weights,
        feature scaling, additive background, dispersion, library) plus decoder px_scale
        statistics. No-forward-pass components are extracted directly from parameters;
        library and px_scale require a forward pass over the data.

        Parameters
        ----------
        adata
            MuData to use. Defaults to the MuData used to initialize the model.
        indices
            Indices of cells to use.
        batch_size
            Batch size for the forward-pass loop (library + px_scale).
        n_px_scale_batches
            If set, stop accumulating px_scale after this many batches
            (saves memory for large datasets). Library is always collected for all cells.

        Returns
        -------
        dict with keys:
            ``"decoder_weights"`` : per-modality Frobenius norm decomposition (z vs cov dims)
            ``"feature_scaling"`` : per-modality transformed scaling params + per-covariate stats
            ``"additive_background"`` : per-modality transformed background + per-covariate stats
            ``"dispersion"`` : per-modality dispersion parameters and correlations
            ``"library"`` : per-modality log-library and library arrays from encoder
            ``"px_scale_stats"`` : per-modality mean px_scale per gene
            ``"covariate_names"`` : registry-derived covariate names and mappings
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        self.module.eval()
        device = next(self.module.parameters()).device

        result = {}

        # =====================================================================
        # 1. Decoder weights: Frobenius norm decomposition (z vs cov dims)
        # =====================================================================
        decoder_weights = {}
        for name in self.module.modality_names:
            decoder = self.module.decoders[name]
            # First FC layer in px_decoder
            first_layer = decoder.px_decoder.fc_layers[0]  # nn.Sequential for "Layer 0"
            # Find the nn.Linear sublayer
            linear = None
            for sublayer in first_layer:
                if sublayer is not None and isinstance(sublayer, torch.nn.Linear):
                    linear = sublayer
                    break
            if linear is None:
                logger.warning("Could not find Linear layer in decoder for modality '%s'", name)
                continue

            weight = linear.weight.detach().cpu()  # (n_hidden, n_input + cat_dim)
            n_cat_list = decoder.px_decoder.n_cat_list
            cat_dim = sum(n_cat_list) if n_cat_list else 0
            n_z_dims = weight.shape[1] - cat_dim

            if cat_dim > 0:
                z_weight = weight[:, :n_z_dims]
                cov_weight = weight[:, n_z_dims:]
                z_norm = torch.linalg.norm(z_weight).item()
                cov_norm = torch.linalg.norm(cov_weight).item()
            else:
                z_norm = torch.linalg.norm(weight).item()
                cov_norm = 0.0

            total = z_norm + cov_norm
            decoder_weights[name] = {
                "z_norm": z_norm,
                "cov_norm": cov_norm,
                "cov_fraction": cov_norm / total if total > 0 else 0.0,
                "n_z_dims": n_z_dims,
                "n_cov_dims": cat_dim,
            }
        result["decoder_weights"] = decoder_weights

        # =====================================================================
        # 2. Feature scaling (per modality, per covariate block)
        # =====================================================================
        feature_scaling_result = {}
        for name in self.module.feature_scaling_modalities:
            if name not in self.module.feature_scaling:
                continue
            param = self.module.feature_scaling[name].detach().cpu()
            transformed = (torch.nn.functional.softplus(param) / 0.7).numpy()

            per_covariate = []
            n_cats_list = self.module.n_cats_per_feature_scaling_cov
            if n_cats_list:
                offset = 0
                for n_cats in n_cats_list:
                    block = transformed[offset : offset + n_cats, :]
                    cov_info = {
                        "n_cats": int(n_cats),
                        "values": block,
                        "mean_abs_deviation": np.mean(np.abs(block - 1.0), axis=0),
                    }
                    if n_cats > 1:
                        cov_info["correlation_matrix"] = np.corrcoef(block)
                    else:
                        cov_info["correlation_matrix"] = None
                    per_covariate.append(cov_info)
                    offset += n_cats
            else:
                # Single shared factor (no covariates)
                per_covariate.append(
                    {
                        "n_cats": transformed.shape[0],
                        "values": transformed,
                        "mean_abs_deviation": np.mean(np.abs(transformed - 1.0), axis=0),
                        "correlation_matrix": None,
                    }
                )

            feature_scaling_result[name] = {
                "transformed": transformed,
                "per_covariate": per_covariate,
                "relative_to_px_scale": None,  # filled after forward pass
            }
        result["feature_scaling"] = feature_scaling_result

        # =====================================================================
        # 3. Additive background (per modality, per covariate block)
        # =====================================================================
        additive_bg_result = {}
        for name in self.module.additive_background_modalities:
            if name not in self.module.additive_background:
                continue
            param = self.module.additive_background[name].detach().cpu()
            transformed = torch.exp(param).numpy()  # (n_features, n_total_ambient_cats)

            per_covariate = []
            n_cats_list = self.module.n_cats_per_ambient_cov
            if n_cats_list:
                col_offset = 0
                for n_cats in n_cats_list:
                    block = transformed[:, col_offset : col_offset + n_cats]
                    cov_info = {
                        "n_cats": int(n_cats),
                        "values": block,
                    }
                    if n_cats > 1:
                        cov_info["correlation_matrix"] = np.corrcoef(block.T)
                    else:
                        cov_info["correlation_matrix"] = None
                    per_covariate.append(cov_info)
                    col_offset += n_cats

            additive_bg_result[name] = {
                "transformed": transformed,
                "per_covariate": per_covariate,
                "relative_to_px_scale": None,  # filled after forward pass
            }
        result["additive_background"] = additive_bg_result

        # =====================================================================
        # 4. Dispersion parameters (per modality)
        # =====================================================================
        dispersion_result = {}
        for name in self.module.modality_names:
            if name not in self.module.px_r_mu:
                continue
            mu = self.module.px_r_mu[name].detach().cpu().numpy()
            disp_type = self.module.dispersion_dict[name]

            disp_info = {
                "px_r_mu": mu,
                "px_r_mean": np.exp(mu),
                "dispersion_type": disp_type,
            }
            # For gene-batch: correlations between batch groups
            if disp_type in ("gene-batch", "region-batch") and mu.ndim == 2 and mu.shape[1] > 1:
                disp_info["correlation_matrix"] = np.corrcoef(mu.T)
            else:
                disp_info["correlation_matrix"] = None

            dispersion_result[name] = disp_info
        result["dispersion"] = dispersion_result

        # =====================================================================
        # 5. Forward pass: library + px_scale accumulation
        # =====================================================================
        library_lists = {name: [] for name in self.module.modality_names}
        px_scale_sum = dict.fromkeys(self.module.modality_names)
        px_scale_n_cells = dict.fromkeys(self.module.modality_names, 0)
        px_scale_done = False
        n_batches_seen = 0

        for tensors in scdl:
            # Move tensors to device
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

            # Run inference
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            # Collect library per modality
            for name in self.module.modality_names:
                if name in outputs["library"]:
                    library_lists[name].append(outputs["library"][name].cpu().numpy())

            # Accumulate px_scale (optional early stop)
            if not px_scale_done:
                gen_inputs = self.module._get_generative_input(tensors, outputs)
                gen_out = self.module.generative(**gen_inputs)

                for name in self.module.modality_names:
                    if name in gen_out["px"]:
                        px_scale = gen_out["px"][name].scale.cpu().numpy()
                        if px_scale_sum[name] is None:
                            px_scale_sum[name] = px_scale.sum(axis=0)
                        else:
                            px_scale_sum[name] += px_scale.sum(axis=0)
                        px_scale_n_cells[name] += px_scale.shape[0]

                n_batches_seen += 1
                if n_px_scale_batches is not None and n_batches_seen >= n_px_scale_batches:
                    px_scale_done = True

        # Assemble library results
        library_result = {}
        for name in self.module.modality_names:
            if library_lists[name]:
                log_lib = np.concatenate(library_lists[name], axis=0)
                library_result[name] = {
                    "log_library": log_lib,
                    "library": np.exp(log_lib),
                }
        result["library"] = library_result

        # Assemble px_scale stats
        px_scale_stats = {}
        for name in self.module.modality_names:
            if px_scale_sum[name] is not None and px_scale_n_cells[name] > 0:
                px_scale_stats[name] = {
                    "mean_per_gene": px_scale_sum[name] / px_scale_n_cells[name],
                }
        result["px_scale_stats"] = px_scale_stats

        # =====================================================================
        # 6. Relative metrics: feature scaling and additive bg vs px_scale
        # =====================================================================
        for name in feature_scaling_result:
            if name in px_scale_stats:
                mean_px = px_scale_stats[name]["mean_per_gene"]
                all_vals = feature_scaling_result[name]["transformed"]
                mean_abs_dev = np.mean(np.abs(all_vals - 1.0), axis=0)
                safe_px = np.where(mean_px > 0, mean_px, 1.0)
                feature_scaling_result[name]["relative_to_px_scale"] = mean_abs_dev / safe_px

        for name in additive_bg_result:
            if name in px_scale_stats:
                mean_px = px_scale_stats[name]["mean_per_gene"]
                bg_vals = additive_bg_result[name]["transformed"]
                mean_bg = np.mean(bg_vals, axis=1)  # mean across covariate levels
                safe_px = np.where(mean_px > 0, mean_px, 1.0)
                additive_bg_result[name]["relative_to_px_scale"] = mean_bg / safe_px

        # =====================================================================
        # 7. Covariate names from registry
        # =====================================================================
        covariate_names = {}
        for reg_key, label in [
            (FEATURE_SCALING_COVS_KEY, "feature_scaling"),
            (AMBIENT_COVS_KEY, "ambient"),
            (ENCODER_COVS_KEY, "encoder"),
        ]:
            if reg_key in self.adata_manager.data_registry:
                try:
                    state = self.adata_manager.get_state_registry(reg_key)
                    info = {"n_cats_per_key": list(state.n_cats_per_key)}
                    if hasattr(state, "field_keys"):
                        info["field_keys"] = list(state.field_keys)
                    if hasattr(state, "mappings"):
                        info["mappings"] = {k: list(v) for k, v in state.mappings.items()}
                    covariate_names[label] = info
                except (AttributeError, KeyError):
                    logger.debug("Could not extract registry info for %s", reg_key)

        for reg_key, label in [
            (DISPERSION_KEY, "dispersion"),
            (LIBRARY_SIZE_KEY, "library_size"),
        ]:
            if reg_key in self.adata_manager.data_registry:
                try:
                    state = self.adata_manager.get_state_registry(reg_key)
                    covariate_names[label] = {
                        "categorical_mapping": list(state.categorical_mapping),
                    }
                except (AttributeError, KeyError):
                    logger.debug("Could not extract registry info for %s", reg_key)

        result["covariate_names"] = covariate_names

        return result

    def compute_covariate_lisi(
        self,
        adata,
        covariate_keys: list[str],
        obsm_keys: list[str] | None = None,
        n_neighbors: int = 50,
        subsample_n: int = 40000,
        subsample_seed: int = 42,
        stratify_key: str = "cell_type_lvl2_new",
    ) -> pd.DataFrame:
        """Compute LISI scores for covariates on attribution embeddings.

        Uses ``scib_metrics.lisi_knn`` to measure mixing of each covariate
        on KNN graphs built from attribution-weighted latent embeddings.

        Parameters
        ----------
        adata
            AnnData with ``X_latent_attr_{name}*`` keys in ``.obsm``.
        covariate_keys
            Obs columns to compute LISI for (e.g. ``['Embryo', '10x_kit']``).
        obsm_keys
            obsm keys to evaluate. If ``None``, auto-discovers all
            ``X_latent_attr_*`` keys in ``adata.obsm``.
        n_neighbors
            Number of neighbors for KNN graph.
        subsample_n
            Number of cells to subsample (stratified). LISI only.
        subsample_seed
            Random seed for subsampling.
        stratify_key
            Obs column for stratified subsampling.

        Returns
        -------
        pd.DataFrame
            Rows = obsm_keys, columns = covariate_keys, values = mean LISI.
        """
        import pandas as pd
        from scib_metrics import lisi_knn
        from scib_metrics.nearest_neighbors import NeighborsResults
        from sklearn.neighbors import NearestNeighbors

        # Auto-discover obsm keys
        if obsm_keys is None:
            obsm_keys = sorted(k for k in adata.obsm if k.startswith("X_latent_attr_"))

        # Stratified subsample
        n_total = adata.n_obs
        if subsample_n is not None and subsample_n < n_total:
            rng = np.random.RandomState(subsample_seed)
            if stratify_key in adata.obs.columns:
                groups = adata.obs[stratify_key].astype(str)
                unique_groups = groups.unique()
                per_group = max(1, subsample_n // len(unique_groups))
                indices = []
                for g in unique_groups:
                    g_idx = np.where(groups.values == g)[0]
                    n_take = min(len(g_idx), per_group)
                    indices.append(rng.choice(g_idx, size=n_take, replace=False))
                indices = np.concatenate(indices)
                # If we have fewer than subsample_n, top up randomly
                if len(indices) < subsample_n:
                    remaining = np.setdiff1d(np.arange(n_total), indices)
                    extra = rng.choice(
                        remaining,
                        size=min(subsample_n - len(indices), len(remaining)),
                        replace=False,
                    )
                    indices = np.concatenate([indices, extra])
                indices = np.sort(indices)
            else:
                indices = np.sort(rng.choice(n_total, size=subsample_n, replace=False))
            adata_sub = adata[indices].copy()
        else:
            adata_sub = adata
            indices = np.arange(n_total)

        # Compute LISI for each obsm_key × covariate_key
        results = {}
        for obsm_key in obsm_keys:
            if obsm_key not in adata_sub.obsm:
                logger.warning("obsm key %r not found in adata, skipping", obsm_key)
                continue

            X = np.asarray(adata_sub.obsm[obsm_key], dtype=np.float32)
            # Build KNN
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            nn.fit(X)
            distances, indices_knn = nn.kneighbors(X)
            knn_results = NeighborsResults(
                indices=indices_knn.astype(np.int32),
                distances=distances.astype(np.float32),
            )

            row = {}
            for cov_key in covariate_keys:
                if cov_key not in adata_sub.obs.columns:
                    row[cov_key] = np.nan
                    continue
                labels = adata_sub.obs[cov_key].astype("category").cat.codes.values
                per_cell_lisi = lisi_knn(knn_results, labels)
                row[cov_key] = float(np.nanmean(per_cell_lisi))

            results[obsm_key] = row

        return pd.DataFrame(results).T

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
            latent_keys = [("X_latent_joint", "joint")]
            for name in self.module.modality_names:
                latent_keys.append((f"X_latent_{name}", name))
            for obsm_key, label in latent_keys:
                if obsm_key in adata.obsm:
                    path = f"{output_dir}/X_latent_{label}.csv"
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
        palette: list[str] | None = None,
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
        palette
            Color palette for categorical variables. If None, uses an extended
            150-color palette from scanpy.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import scanpy as sc

        if palette is None:
            palette = sc.pl.palettes.default_102 + sc.pl.palettes.zeileis_28 + sc.pl.palettes.vega_20_scanpy

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
                    palette=palette,
                )

        fig.tight_layout()
        return fig
