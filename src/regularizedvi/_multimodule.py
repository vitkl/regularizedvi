"""RegularizedMultimodalVAE module for multi-modal single-cell data.

N-modality extensible VAE with symmetric regularized components.
Each modality uses the same encoder/decoder architecture (RegularizedEncoder,
RegularizedDecoderSCVI) with per-modality configuration for:
- n_hidden, n_latent (architecture sizing)
- additive_background (ambient correction, default ON for RNA only)
- feature_scaling (per-feature bias, default ON for ATAC only)
- dispersion parameterization (gene/region, gene-batch/region-batch)
- GammaPoisson likelihood (default for all modalities)

Supports three latent combination strategies:
- "concatenation" (default): z = [z_mod1; z_mod2; ...], preserves modality-specific signal
- "single_encoder": one encoder for concatenated input
- "weighted_mean": MultiVI-style weighted mixing
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Exponential, Gamma, LogNormal, Normal
from torch.distributions import kl_divergence as kld

from regularizedvi._constants import (
    AMBIENT_COVS_KEY,
    DEFAULT_COMPUTE_PEARSON,
    DISPERSION_KEY,
    ENCODER_COVS_KEY,
    FEATURE_SCALING_COVS_KEY,
    LIBRARY_SIZE_KEY,
)
from regularizedvi._module import _pearson_corr_rows

if TYPE_CHECKING:
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


def _resolve_per_modality(value, modality_names: list[str]) -> dict[str, any]:
    """Resolve a scalar or dict into a per-modality dict.

    Parameters
    ----------
    value
        Either a scalar (applied to all modalities) or a dict keyed by modality name.
    modality_names
        List of modality names.

    Returns
    -------
    Dict mapping each modality name to its value.
    """
    if isinstance(value, dict):
        missing = set(modality_names) - set(value.keys())
        if missing:
            raise ValueError(f"Missing modality keys in config: {missing}")
        return {name: value[name] for name in modality_names}
    return dict.fromkeys(modality_names, value)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax with masking for missing modalities."""
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=dim)


class RegularizedMultimodalVAE(BaseModuleClass):
    """N-modality extensible VAE with symmetric regularized components.

    All modalities share the same encoder/decoder class (RegularizedEncoder,
    RegularizedDecoderSCVI) but can differ in architecture size, normalization,
    and per-modality flags (additive_background, feature_scaling).

    Parameters
    ----------
    modality_names
        Ordered list of modality names (e.g., ``["rna", "atac"]``).
    n_input_per_modality
        Dict mapping modality name to number of input features.
    n_batch
        Number of batches.
    n_hidden
        Number of hidden units per modality. Scalar (shared) or dict per modality.
    n_latent
        Latent dimensionality per modality. Scalar (shared) or dict per modality.
    n_layers
        Number of hidden layers. Scalar (shared) or dict per modality.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        Number of categories per categorical covariate.
    dropout_rate
        Dropout rate for encoders.
    latent_mode
        How to combine per-modality encoder outputs:
        ``"concatenation"`` (default): z = [z_mod1; z_mod2; ...].
        ``"single_encoder"``: one encoder for all concatenated inputs.
        ``"weighted_mean"``: weighted average of per-modality z.
    modality_weights
        For weighted_mean mode: ``"equal"``, ``"universal"``, or ``"cell"``.
    dispersion
        Dispersion parameterization per modality. Scalar or dict.
        Options: ``"gene"``, ``"gene-batch"`` (or ``"region"``, ``"region-batch"``).
    log_variational
        If True, use log1p on input before encoding.
    use_size_factor_key
        Not supported for multimodal model. Must be False.
    use_observed_lib_size
        Not supported for multimodal model. Must be False.
        Library size is always learned with a constrained LogNormal prior.
    library_log_means
        Prior means for log library sizes, dict per modality.
    library_log_vars
        Prior variances for log library sizes, dict per modality.
    library_log_vars_weight
        Scale factor for library prior variance. If a dict, per-modality weights
        (e.g. ``{"rna": 0.2, "atac": 1.5}``). If a float, applied to all modalities.
    library_n_hidden
        Hidden units for library encoder.
    scale_activation
        Decoder scale activation.
    use_batch_in_decoder
        If False, batch-free decoder.
    additive_background_modalities
        List of modality names that get additive ambient background.
    additive_bg_prior_alpha
        Alpha for Gamma prior on ``exp(additive_background)``. Default ``1.0``.
    additive_bg_prior_beta
        Beta for Gamma prior on ``exp(additive_background)``. Default ``100.0``.
        Together gives Gamma(1, 100) with mean=0.01 (cell2location-style).
    feature_scaling_modalities
        List of modality names that get per-feature feature scaling.
    regularise_dispersion
        Enable dispersion regularization.
    regularise_dispersion_prior
        Initialization for the Exponential containment prior rate parameter.
    dispersion_hyper_prior_alpha
        Alpha for Gamma hyper-prior on learned rate.
    dispersion_hyper_prior_beta
        Beta for Gamma hyper-prior on learned rate.
    use_batch_norm
        Where to use BatchNorm.
    use_layer_norm
        Where to use LayerNorm.
    encode_covariates
        If True, include covariates in encoder input.
    deeply_inject_covariates
        If True, inject covariates into all decoder layers.
    extra_encoder_kwargs
        Additional kwargs for RegularizedEncoder.
    extra_decoder_kwargs
        Additional kwargs for RegularizedDecoderSCVI.
    """

    def __init__(
        self,
        modality_names: list[str],
        n_input_per_modality: dict[str, int],
        n_batch: int = 0,
        n_hidden: dict[str, int] | int = 128,
        n_latent: dict[str, int] | int = 10,
        n_layers: dict[str, int] | int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        latent_mode: Literal["concatenation", "single_encoder", "weighted_mean"] = "concatenation",
        modality_weights: Literal["equal", "universal", "cell"] = "equal",
        dispersion: dict[str, str] | str = "gene-batch",
        log_variational: bool = True,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = False,
        library_log_means: dict[str, np.ndarray] | None = None,
        library_log_vars: dict[str, np.ndarray] | None = None,
        library_log_vars_weight: float | dict[str, float] | None = None,
        library_log_means_centering_sensitivity: dict[str, float] | float | None = None,
        library_n_hidden: int = 16,
        scale_activation: str = "softplus",
        use_batch_in_decoder: bool = False,
        additive_background_modalities: list[str] | None = None,
        feature_scaling_modalities: list[str] | None = None,
        regularise_dispersion: bool = True,
        regularise_dispersion_prior: dict[str, float] | float | None = None,
        dispersion_hyper_prior_alpha: dict[str, float] | float | None = None,
        dispersion_hyper_prior_beta: dict[str, float] | float | None = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        # Feature scaling (cell2location-style detection_tech_gene)
        n_cats_per_feature_scaling_cov: list[int] | None = None,
        feature_scaling_prior_alpha: float = 200.0,
        feature_scaling_prior_beta: float = 200.0,
        additive_bg_prior_alpha: float = 1.0,
        additive_bg_prior_beta: float = 100.0,
        regularise_background: bool = True,
        # Ambient covariates (for additive background, decoupled from batch_key)
        n_cats_per_ambient_cov: list[int] | None = None,
        # Encoder covariates (what categoricals the encoder sees, default=None → nothing)
        n_cats_per_encoder_cov: list[int] | None = None,
        # Dispersion covariate (controls per-group px_r, decoupled from batch_key)
        n_dispersion_cats: int | None = None,
        # Library size covariate (controls per-group library prior, decoupled from batch_key)
        n_library_cats: int | None = None,
        # Training metrics
        compute_pearson: bool = DEFAULT_COMPUTE_PEARSON,
        # Learnable per-modality scaling on size factors
        learnable_modality_scaling: bool = False,
        modality_scale_prior_concentration: float = 5.0,
        # Decoder weight regularization
        decoder_weight_l2: float = 0.1,
        decoder_cov_weight_l2: float = 0.0,
        # Data-dependent initialization
        decoder_bias_init: dict[str, np.ndarray] | None = None,
        additive_bg_init_per_gene: dict[str, np.ndarray] | None = None,
        # Residual library encoder: library = log(sens) + w*(obs-log(sens)) + encoder
        residual_library_encoder: bool = True,
        library_obs_w_prior_rate: float = 1.0,
        # Dispersion initialization override
        px_r_init_mean: float | np.ndarray | dict | None = None,
        px_r_init_std: float | None = None,
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
    ):
        from regularizedvi._components import RegularizedDecoderSCVI, RegularizedEncoder

        super().__init__()

        self.compute_pearson = compute_pearson
        self.modality_names = list(modality_names)
        self.n_modalities = len(self.modality_names)
        self.n_input_per_modality = n_input_per_modality
        self.n_batch = n_batch
        self.log_variational = log_variational
        self.latent_mode = latent_mode
        self.modality_weights_mode = modality_weights
        if use_size_factor_key:
            raise ValueError(
                "use_size_factor_key=True is not supported for RegularizedMultimodalVAE. Library size must be learned."
            )
        if use_observed_lib_size:
            raise ValueError(
                "use_observed_lib_size=True is not supported for RegularizedMultimodalVAE. "
                "Library size must be learned with a constrained LogNormal prior."
            )
        self.use_observed_lib_size = False
        self.use_batch_in_decoder = use_batch_in_decoder
        self.residual_library_encoder = residual_library_encoder
        self.regularise_dispersion = regularise_dispersion
        self.encode_covariates = encode_covariates
        self.z_sparsity_prior = z_sparsity_prior
        self.n_active_latent_per_cell = n_active_latent_per_cell
        self.decoder_hidden_l1 = decoder_hidden_l1
        self.hidden_activation_sparsity = hidden_activation_sparsity
        self.n_active_hidden_per_cell = n_active_hidden_per_cell

        # Resolve decoder_type early (needed for hyper-prior defaults)
        decoder_type_dict = _resolve_per_modality(decoder_type, self.modality_names)
        self.decoder_type_dict = decoder_type_dict

        # Per-modality dispersion hyper-prior:
        # None → use decoder-type defaults from DECODER_TYPE_DEFAULTS
        # scalar → broadcast to all modalities (user override)
        # dict → per-modality (user override)
        from regularizedvi._constants import DECODER_TYPE_DEFAULTS

        self.dispersion_hyper_prior_alpha_dict = {}
        self.dispersion_hyper_prior_beta_dict = {}
        self.regularise_dispersion_prior_dict = {}

        # Resolve user values (None stays None, scalar→dict, dict stays dict)
        _alpha_user = (
            _resolve_per_modality(dispersion_hyper_prior_alpha, self.modality_names)
            if dispersion_hyper_prior_alpha is not None
            else None
        )
        _beta_user = (
            _resolve_per_modality(dispersion_hyper_prior_beta, self.modality_names)
            if dispersion_hyper_prior_beta is not None
            else None
        )
        _prior_user = (
            _resolve_per_modality(regularise_dispersion_prior, self.modality_names)
            if regularise_dispersion_prior is not None
            else None
        )

        for name in self.modality_names:
            dt = decoder_type_dict[name]
            dt_defaults = DECODER_TYPE_DEFAULTS.get(dt, DECODER_TYPE_DEFAULTS["expected_RNA"])
            self.dispersion_hyper_prior_alpha_dict[name] = (
                _alpha_user[name] if _alpha_user is not None else dt_defaults["dispersion_hyper_prior_alpha"]
            )
            self.dispersion_hyper_prior_beta_dict[name] = (
                _beta_user[name] if _beta_user is not None else dt_defaults["dispersion_hyper_prior_beta"]
            )
            self.regularise_dispersion_prior_dict[name] = (
                _prior_user[name] if _prior_user is not None else dt_defaults["regularise_dispersion_prior"]
            )

        # Feature scaling scaling covariates (cell2location-style detection_tech_gene)
        self.n_cats_per_feature_scaling_cov = n_cats_per_feature_scaling_cov or []
        self.n_total_feature_scaling_cats = (
            sum(self.n_cats_per_feature_scaling_cov) if self.n_cats_per_feature_scaling_cov else 0
        )
        self.feature_scaling_prior_alpha = feature_scaling_prior_alpha
        self.feature_scaling_prior_beta = feature_scaling_prior_beta
        self.additive_bg_prior_alpha = additive_bg_prior_alpha
        self.additive_bg_prior_beta = additive_bg_prior_beta
        self.regularise_background = regularise_background
        self.decoder_weight_l2 = decoder_weight_l2
        self.decoder_cov_weight_l2 = decoder_cov_weight_l2

        # Dispersion covariate (decoupled from batch_key, fallback to n_batch)
        self.n_dispersion_cats = n_dispersion_cats if n_dispersion_cats is not None else n_batch
        # Library size covariate (decoupled from batch_key, fallback to n_batch)
        self.n_library_cats = n_library_cats if n_library_cats is not None else n_batch

        additive_background_modalities = additive_background_modalities or []
        feature_scaling_modalities = feature_scaling_modalities or []
        self.additive_background_modalities = additive_background_modalities
        self.feature_scaling_modalities = feature_scaling_modalities

        # Resolve per-modality configs
        n_hidden_dict = _resolve_per_modality(n_hidden, self.modality_names)
        n_latent_dict = _resolve_per_modality(n_latent, self.modality_names)
        n_layers_dict = _resolve_per_modality(n_layers, self.modality_names)
        dispersion_dict = _resolve_per_modality(dispersion, self.modality_names)

        burst_size_intercept_dict = _resolve_per_modality(burst_size_intercept, self.modality_names)
        burst_size_n_hidden_dict = _resolve_per_modality(burst_size_n_hidden, self.modality_names)

        self.n_hidden_dict = n_hidden_dict
        self.n_latent_dict = n_latent_dict
        self.dispersion_dict = dispersion_dict

        use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        use_layer_norm_decoder = use_layer_norm in ("decoder", "both")

        # Encoder/decoder covariate composition (purpose-driven keys)
        # Encoder sees: only encoder_covs (from dedicated encoder_covariate_keys registry)
        # Default: no encoder categoricals (matches scVI/MultiVI/PeakVI default)
        # Decoder sees: [cat_covs...] only (no batch injection by default)
        n_total_ambient_cats = sum(int(c) for c in n_cats_per_ambient_cov) if n_cats_per_ambient_cov else 0
        self.n_cats_per_encoder_cov = list(n_cats_per_encoder_cov) if n_cats_per_encoder_cov else []
        self.n_total_encoder_cats = (
            sum(int(c) for c in self.n_cats_per_encoder_cov) if self.n_cats_per_encoder_cov else 0
        )
        encoder_cat_list = list(self.n_cats_per_encoder_cov) if self.n_cats_per_encoder_cov else None
        _cat_cats = list(n_cats_per_cov or [])

        if not use_batch_in_decoder:
            decoder_cat_list = _cat_cats
        else:
            decoder_cat_list = _cat_cats  # batch injection via use_batch_in_decoder is deprecated

        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        _extra_decoder_kwargs = extra_decoder_kwargs or {}

        # Compute total latent dim for decoder input
        if latent_mode == "concatenation":
            self.total_latent_dim = sum(n_latent_dict.values())
        elif latent_mode == "single_encoder":
            # Single encoder outputs into a shared latent space
            # Use sum of per-modality n_latent as total
            self.total_latent_dim = sum(n_latent_dict.values())
        elif latent_mode == "weighted_mean":
            # All modalities must have the same n_latent for weighted mean
            latent_vals = list(n_latent_dict.values())
            if len(set(latent_vals)) != 1:
                raise ValueError(
                    f"All modalities must have the same n_latent for weighted_mean mode. Got: {n_latent_dict}"
                )
            self.total_latent_dim = latent_vals[0]

        n_input_decoder = self.total_latent_dim + n_continuous_cov

        # ---- Per-modality encoders ----
        self.encoders = nn.ModuleDict()
        for name in self.modality_names:
            n_in = n_input_per_modality[name] + n_continuous_cov
            self.encoders[name] = RegularizedEncoder(
                n_input=n_in,
                n_output=n_latent_dict[name],
                n_cat_list=encoder_cat_list,
                n_layers=n_layers_dict[name],
                n_hidden=n_hidden_dict[name],
                dropout_rate=dropout_rate,
                distribution="normal",
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                return_dist=True,
                **_extra_encoder_kwargs,
            )

        # ---- Single encoder (for single_encoder mode) ----
        if latent_mode == "single_encoder":
            total_input = sum(n_input_per_modality.values()) + n_continuous_cov
            # Use max n_hidden across modalities for single encoder
            max_n_hidden = max(n_hidden_dict.values())
            max_n_layers = max(n_layers_dict.values())
            self.joint_encoder = RegularizedEncoder(
                n_input=total_input,
                n_output=self.total_latent_dim,
                n_cat_list=encoder_cat_list,
                n_layers=max_n_layers,
                n_hidden=max_n_hidden,
                dropout_rate=dropout_rate,
                distribution="normal",
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                return_dist=True,
                **_extra_encoder_kwargs,
            )

        # ---- Modality weights (for weighted_mean mode) ----
        if latent_mode == "weighted_mean":
            if modality_weights == "universal":
                self.modality_weight_params = nn.Parameter(torch.zeros(self.n_modalities))
            elif modality_weights == "cell":
                # Small network to predict per-cell weights
                self.modality_weight_net = nn.Linear(self.total_latent_dim * self.n_modalities, self.n_modalities)

        # ---- Per-modality decoders ----
        self.decoders = nn.ModuleDict()
        for name in self.modality_names:
            self.decoders[name] = RegularizedDecoderSCVI(
                n_input=n_input_decoder,
                n_output=n_input_per_modality[name],
                n_cat_list=decoder_cat_list,
                n_layers=n_layers_dict[name],
                n_hidden=n_hidden_dict[name],
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation=scale_activation,
                decoder_type=decoder_type_dict[name],
                burst_size_n_hidden=burst_size_n_hidden_dict[name],
                burst_size_intercept=burst_size_intercept_dict[name],
                **_extra_decoder_kwargs,
            )

        # Data-dependent decoder bias initialization (Stream B)
        if decoder_bias_init is not None:
            for name in self.modality_names:
                if name in decoder_bias_init:
                    init_vals = torch.from_numpy(decoder_bias_init[name]).float()
                    init_vals = torch.clamp(init_vals, min=0.01)
                    # softplus_inv(x) = log(exp(x) - 1), numerically stable for large x
                    bias_init = torch.where(
                        init_vals > 20.0,
                        init_vals,
                        torch.log(torch.expm1(init_vals)),
                    )
                    with torch.no_grad():
                        self.decoders[name].px_scale_decoder[0].bias.copy_(bias_init)

        # ---- Per-modality library encoders (variational, low-capacity) ----
        if library_n_hidden > 32:
            warnings.warn(
                f"library_n_hidden={library_n_hidden} is large. The library encoder "
                f"should be low-capacity (16-32 hidden units) to prevent overfitting "
                f"the library size. Consider setting library_n_hidden=16.",
                UserWarning,
                stacklevel=2,
            )
        self.l_encoders = nn.ModuleDict()
        for name in self.modality_names:
            n_in = n_input_per_modality[name] + n_continuous_cov
            self.l_encoders[name] = RegularizedEncoder(
                n_input=n_in,
                n_output=1,
                n_layers=1,
                n_cat_list=encoder_cat_list,
                n_hidden=library_n_hidden,
                dropout_rate=dropout_rate,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_encoder,
                use_layer_norm=use_layer_norm_encoder,
                return_dist=True,
                **_extra_encoder_kwargs,
            )

        # ---- Per-modality library priors ----
        library_log_means = library_log_means or {}
        library_log_vars = library_log_vars or {}
        if isinstance(library_log_means_centering_sensitivity, dict):
            _sensitivity = library_log_means_centering_sensitivity
        elif library_log_means_centering_sensitivity is not None:
            _sensitivity = dict.fromkeys(self.modality_names, float(library_log_means_centering_sensitivity))
        else:
            _sensitivity = {}
        # Normalize library_log_vars_weight to dict
        if isinstance(library_log_vars_weight, dict):
            _vars_weight = library_log_vars_weight
        elif library_log_vars_weight is not None:
            _vars_weight = dict.fromkeys(self.modality_names, library_log_vars_weight)
        else:
            _vars_weight = {}
        for name in self.modality_names:
            if name in library_log_means and name in library_log_vars:
                means = torch.from_numpy(library_log_means[name]).float()
                # ALWAYS center: subtract global mean, optionally shift by log(sensitivity)
                _sens = _sensitivity.get(name, None)
                global_log_mean = means.mean()
                log_sens = np.log(_sens) if _sens is not None else 0.0
                means = means - global_log_mean + log_sens

                # Change 1: initialize library encoder bias to centered scale
                with torch.no_grad():
                    self.l_encoders[name].mean_encoder.bias.fill_(log_sens)

                # Always store centering constants
                self.register_buffer(
                    f"library_global_log_mean_{name}",
                    torch.tensor(global_log_mean.item()),
                )
                self.register_buffer(
                    f"library_log_sensitivity_{name}",
                    torch.tensor(log_sens),
                )
                vars_ = torch.from_numpy(library_log_vars[name]).float()
                if name in _vars_weight:
                    vars_ = vars_ * _vars_weight[name]
                self.register_buffer(f"library_log_means_{name}", means)
                self.register_buffer(f"library_log_vars_{name}", vars_)

        # ---- Residual library encoder: variational weight w ~ LogNormal ----
        if self.residual_library_encoder:
            self.library_obs_w_prior_rate = library_obs_w_prior_rate
            self.library_obs_w_mu = nn.ParameterDict()
            self.library_obs_w_log_sigma = nn.ParameterDict()
            for name in self.modality_names:
                self.library_obs_w_mu[name] = nn.Parameter(torch.tensor(0.0))  # E[w] ≈ 1
                self.library_obs_w_log_sigma[name] = nn.Parameter(torch.tensor(-2.0))  # tight init

        # ---- Per-modality dispersion parameters ----
        # Initialize px_r at prior equilibrium when regularisation is active.
        # Containment prior: Exp(rate) on 1/sqrt(theta) → theta = rate² at equilibrium.
        # px_r_init_mean can be a dict {modality_name: np.ndarray} for per-gene init
        _px_r_init_dict = {}
        if isinstance(px_r_init_mean, dict):
            _px_r_init_dict = px_r_init_mean
        _px_r_init_scalar = None
        if px_r_init_mean is not None and not isinstance(px_r_init_mean, dict):
            _px_r_init_scalar = px_r_init_mean
        _px_r_std = px_r_init_std if px_r_init_std is not None else 0.1

        _log_sigma_init = math.log(0.1)
        self.px_r_mu = nn.ParameterDict()
        self.px_r_log_sigma = nn.ParameterDict()
        for name in self.modality_names:
            n_feat = n_input_per_modality[name]
            disp = dispersion_dict[name]
            # Resolve per-modality init: dict entry > scalar > from prior rate
            if name in _px_r_init_dict:
                _mod_init = _px_r_init_dict[name]
            elif _px_r_init_scalar is not None:
                _mod_init = _px_r_init_scalar
            elif self.regularise_dispersion:
                _mod_rate = self.regularise_dispersion_prior_dict[name]
                _mod_init = math.log(_mod_rate**2)
            else:
                _mod_init = None

            if disp in ("gene", "region"):
                if _mod_init is not None:
                    if isinstance(_mod_init, np.ndarray):
                        _init_tensor = torch.tensor(_mod_init, dtype=torch.float32)
                    else:
                        _init_tensor = torch.full((n_feat,), _mod_init)
                    self.px_r_mu[name] = nn.Parameter(_init_tensor + _px_r_std * torch.randn(n_feat))
                else:
                    self.px_r_mu[name] = nn.Parameter(torch.randn(n_feat))
                self.px_r_log_sigma[name] = nn.Parameter(torch.full((n_feat,), _log_sigma_init))
            elif disp in ("gene-batch", "region-batch"):
                n_disp = self.n_dispersion_cats
                if _mod_init is not None:
                    if isinstance(_mod_init, np.ndarray):
                        _init_tensor = torch.tensor(_mod_init, dtype=torch.float32).unsqueeze(1).expand(n_feat, n_disp)
                    else:
                        _init_tensor = torch.full((n_feat, n_disp), _mod_init)
                    self.px_r_mu[name] = nn.Parameter(_init_tensor.clone() + _px_r_std * torch.randn(n_feat, n_disp))
                else:
                    self.px_r_mu[name] = nn.Parameter(torch.randn(n_feat, n_disp))
                self.px_r_log_sigma[name] = nn.Parameter(torch.full((n_feat, n_disp), _log_sigma_init))

        # ---- Learnable dispersion prior rates (per modality, decoder-type defaults) ----
        if self.regularise_dispersion:
            self.dispersion_prior_rate_raw = nn.ParameterDict()
            for name in self.modality_names:
                _mod_rate = self.regularise_dispersion_prior_dict[name]
                _raw_init = torch.log(torch.expm1(torch.tensor(_mod_rate)))
                disp = dispersion_dict[name]
                if disp in ("gene-batch", "region-batch"):
                    self.dispersion_prior_rate_raw[name] = nn.Parameter(
                        _raw_init.expand(self.n_dispersion_cats).clone()
                    )
                else:
                    self.dispersion_prior_rate_raw[name] = nn.Parameter(_raw_init.unsqueeze(0).clone())

        # ---- Ambient covariates (for additive background, decoupled from batch_key) ----
        self.n_cats_per_ambient_cov = list(n_cats_per_ambient_cov) if n_cats_per_ambient_cov is not None else []

        # ---- Additive background (per selected modality, concatenated ambient covariates) ----
        # Each modality gets a single (n_feat, n_total_ambient_cats) parameter.
        # Ambient covariates are concatenated into one one-hot vector per cell.
        # Initialized at log(prior_mean) = log(alpha/beta) so exp(init) matches Gamma prior mean.
        # Default Gamma(1, 100) → mean = 0.01, small relative to px_scale.
        _bg_init_scalar = math.log(additive_bg_prior_alpha / additive_bg_prior_beta)
        self.n_total_ambient_cats = n_total_ambient_cats
        self.additive_background = nn.ParameterDict()
        for name in additive_background_modalities:
            n_feat = n_input_per_modality[name]
            if self.n_total_ambient_cats > 0:
                if additive_bg_init_per_gene is not None and name in additive_bg_init_per_gene:
                    init_vals = torch.from_numpy(additive_bg_init_per_gene[name]).float()
                    if init_vals.dim() == 1:
                        init_vals = init_vals.unsqueeze(1).expand(-1, self.n_total_ambient_cats)
                    self.additive_background[name] = nn.Parameter(
                        init_vals + 0.01 * torch.randn(n_feat, self.n_total_ambient_cats)
                    )
                else:
                    self.additive_background[name] = nn.Parameter(
                        torch.full((n_feat, self.n_total_ambient_cats), _bg_init_scalar)
                        + 0.01 * torch.randn(n_feat, self.n_total_ambient_cats)
                    )

        # ---- Feature scaling (cell2location-style per-covariate scaling) ----
        # Shape: (n_feature_scaling_rows, n_features) where n_feature_scaling_rows is either:
        #   - sum(n_cats_per_feature_scaling_cov) when scaling covariates are provided
        #   - 1 when no scaling covariates (single shared factor, backward compatible)
        # Initialized at 0; softplus(0)/0.7 ≈ 0.99 (centered at ~1.0)
        self.feature_scaling = nn.ParameterDict()
        n_feature_scaling_rows = self.n_total_feature_scaling_cats if self.n_total_feature_scaling_cats > 0 else 1
        for name in feature_scaling_modalities:
            n_feat = n_input_per_modality[name]
            self.feature_scaling[name] = nn.Parameter(torch.zeros(n_feature_scaling_rows, n_feat))

        # ---- Learnable per-modality scaling on size factors ----
        self.learnable_modality_scaling = learnable_modality_scaling
        self.modality_scale_prior_concentration = modality_scale_prior_concentration
        self.modality_scale_raw = nn.ParameterDict()
        self.modality_scale_init = {}
        if learnable_modality_scaling:
            if isinstance(library_log_means_centering_sensitivity, dict):
                _sensitivity2 = library_log_means_centering_sensitivity
            elif library_log_means_centering_sensitivity is not None:
                _sensitivity2 = dict.fromkeys(self.modality_names, float(library_log_means_centering_sensitivity))
            else:
                _sensitivity2 = {}
            for name in self.modality_names:
                init_val = _sensitivity2.get(name, 1.0)
                self.modality_scale_init[name] = init_val
                # softplus(raw)/0.7 = init_val → raw = inverse_softplus(init_val * 0.7)
                target = init_val * 0.7
                raw_init = math.log(math.exp(target) - 1) if target < 20 else target
                self.modality_scale_raw[name] = nn.Parameter(torch.tensor(raw_init))

    def _decoder_weight_l2_penalty(self) -> torch.Tensor:
        """Sum of squared weights in all decoder FC layers (excludes biases)."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for decoder in self.decoders.values():
            for layer_seq in decoder.px_decoder.fc_layers:
                for sublayer in layer_seq:
                    if isinstance(sublayer, nn.Linear):
                        penalty = penalty + sublayer.weight.pow(2).sum()
            penalty = penalty + decoder.px_scale_decoder[0].weight.pow(2).sum()
        return penalty

    def _decoder_hidden_l1_penalty(self) -> torch.Tensor:
        """Sum of absolute weights in all decoder FC layers (excludes biases)."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for decoder in self.decoders.values():
            for layer_seq in decoder.px_decoder.fc_layers:
                for sublayer in layer_seq:
                    if isinstance(sublayer, nn.Linear):
                        penalty = penalty + sublayer.weight.abs().sum()
        return penalty

    def _decoder_cov_weight_l2_penalty(self) -> torch.Tensor:
        """Sum of squared weights for covariate columns of decoder layer 0 (all modalities)."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for decoder in self.decoders.values():
            cat_dim = sum(decoder.px_decoder.n_cat_list)
            if cat_dim == 0:
                continue
            layer0 = decoder.px_decoder.fc_layers[0]
            for sublayer in layer0:
                if isinstance(sublayer, nn.Linear):
                    W = sublayer.weight
                    assert W.shape[1] > cat_dim, (
                        f"Decoder layer 0 weight has {W.shape[1]} columns but cat_dim={cat_dim}"
                    )
                    cov_weights = W[:, -cat_dim:]
                    penalty = penalty + cov_weights.pow(2).sum()
                    break
        return penalty

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for inference."""
        result = {
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
            "cont_covs": tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            "cat_covs": tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            "encoder_covs": tensors.get(ENCODER_COVS_KEY, None),
            "library_size_index": tensors.get(LIBRARY_SIZE_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
        }
        for name in self.modality_names:
            key = f"X_{name}"
            result[f"x_{name}"] = tensors.get(key, None)
        return result

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        return {
            "z": inference_outputs["z"],
            "library": inference_outputs["library"],
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
            "dispersion_index": tensors.get(DISPERSION_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
            "library_size_index": tensors.get(LIBRARY_SIZE_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
            "cont_covs": tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            "cat_covs": tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            "feature_scaling_covs": tensors.get(FEATURE_SCALING_COVS_KEY, None),
            "ambient_covs": tensors.get(AMBIENT_COVS_KEY, None),
        }

    @auto_move_data
    def inference(
        self,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        encoder_covs: torch.Tensor | None = None,
        library_size_index: torch.Tensor | None = None,
        n_samples: int = 1,
        **modality_inputs,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Per-modality encoding and Z combination.

        Parameters
        ----------
        batch_index
            Batch indices (kept for backward compat, not used by encoder).
        cont_covs
            Continuous covariates.
        cat_covs
            Categorical covariates (decoder injection only).
        encoder_covs
            Encoder covariate indices (from dedicated encoder_covariate_keys).
        library_size_index
            Library size group index (unused, kept for API compat).
        n_samples
            Number of samples from posterior.
        **modality_inputs
            Per-modality inputs as x_{modality_name} tensors.
        """
        # Build encoder categorical inputs from dedicated encoder_covariate_keys
        encoder_categorical_input = ()
        if encoder_covs is not None and self.n_cats_per_encoder_cov:
            encoder_categorical_input += tuple(t.long() for t in torch.split(encoder_covs, 1, dim=1))

        # Detect which modalities are present per cell (for masking)
        masks = {}
        for name in self.modality_names:
            x = modality_inputs.get(f"x_{name}")
            if x is not None:
                masks[name] = x.sum(dim=1) > 0  # (n_cells,)
            else:
                masks[name] = torch.zeros(batch_index.shape[0], dtype=torch.bool, device=batch_index.device)

        # Per-modality encoding
        qz_per_modality = {}
        z_per_modality = {}

        if self.latent_mode == "single_encoder":
            # Concatenate all modality inputs
            inputs = []
            for name in self.modality_names:
                x = modality_inputs.get(f"x_{name}")
                if x is None:
                    x = torch.zeros(
                        batch_index.shape[0],
                        self.n_input_per_modality[name],
                        device=batch_index.device,
                    )
                if self.log_variational:
                    x = torch.log1p(x)
                if cont_covs is not None:
                    x = torch.cat([x, cont_covs], dim=-1)
                inputs.append(x)
            joint_input = torch.cat(inputs, dim=-1)
            qz, z = self.joint_encoder(joint_input, *encoder_categorical_input)
            # Store as single "joint" modality for KL computation
            qz_per_modality["_joint"] = qz
            z_per_modality["_joint"] = z
        else:
            for name in self.modality_names:
                x = modality_inputs.get(f"x_{name}")
                if x is None:
                    continue
                x_ = torch.log1p(x) if self.log_variational else x
                if cont_covs is not None:
                    x_ = torch.cat([x_, cont_covs], dim=-1)
                qz, z = self.encoders[name](x_, *encoder_categorical_input)
                qz_per_modality[name] = qz
                z_per_modality[name] = z

        # Combine Z based on latent_mode
        if self.latent_mode == "concatenation":
            z_parts = []
            for name in self.modality_names:
                if name in z_per_modality:
                    z_parts.append(z_per_modality[name])
                else:
                    # Missing modality: use zeros
                    z_parts.append(
                        torch.zeros(
                            batch_index.shape[0],
                            self.n_latent_dict[name],
                            device=batch_index.device,
                        )
                    )
            z = torch.cat(z_parts, dim=-1)
        elif self.latent_mode == "single_encoder":
            z = z_per_modality["_joint"]
        elif self.latent_mode == "weighted_mean":
            z = self._mix_modalities(z_per_modality, qz_per_modality, masks, batch_index)

        # Per-modality library encoding (always learned, never observed)
        assert not self.use_observed_lib_size, (
            "use_observed_lib_size=True is not supported. Library size must be learned."
        )
        library = {}
        ql_per_modality = {}
        for name in self.modality_names:
            x = modality_inputs.get(f"x_{name}")
            if x is None:
                continue
            x_ = torch.log1p(x) if self.log_variational else x
            if cont_covs is not None:
                x_ = torch.cat([x_, cont_covs], dim=-1)
            ql_enc, lib_enc = self.l_encoders[name](x_, *encoder_categorical_input)

            if self.residual_library_encoder:
                # Centered observed log-library (same scale as prior)
                log_obs = torch.log(x.sum(dim=-1, keepdim=True).clamp(min=1.0))
                glm = getattr(self, f"library_global_log_mean_{name}", None)
                ls = getattr(self, f"library_log_sensitivity_{name}", None)
                if glm is not None and ls is not None:
                    log_obs_centered = log_obs - glm + ls
                else:
                    log_obs_centered = log_obs
                log_sens = ls if ls is not None else torch.tensor(0.0, device=x.device)

                # Sample w from LogNormal variational posterior
                w_mu = self.library_obs_w_mu[name]
                w_sigma = torch.exp(self.library_obs_w_log_sigma[name])
                w = LogNormal(w_mu, w_sigma).rsample()

                # Shrink deviation from global mean: log(sens) + w*(obs-log(sens)) + enc
                obs_contribution = log_sens + w * (log_obs_centered - log_sens)
                lib = obs_contribution + lib_enc
                ql = Normal(obs_contribution + ql_enc.loc, ql_enc.scale)
                library[name] = lib
                ql_per_modality[name] = ql
            else:
                library[name] = lib_enc
                ql_per_modality[name] = ql_enc

        return {
            "z": z,
            "qz_per_modality": qz_per_modality,
            "z_per_modality": z_per_modality,
            "library": library,
            "ql_per_modality": ql_per_modality,
            "masks": masks,
        }

    def _mix_modalities(
        self,
        z_per_modality: dict[str, torch.Tensor],
        qz_per_modality: dict[str, Normal],
        masks: dict[str, torch.Tensor],
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted mean mixing of per-modality latent representations."""
        n_cells = batch_index.shape[0]
        device = batch_index.device
        n_latent = self.total_latent_dim

        # Stack z values: (n_cells, n_modalities, n_latent)
        z_stack = torch.zeros(n_cells, self.n_modalities, n_latent, device=device)
        mask_stack = torch.zeros(n_cells, self.n_modalities, dtype=torch.bool, device=device)

        for i, name in enumerate(self.modality_names):
            if name in z_per_modality:
                z_stack[:, i, :] = z_per_modality[name]
                mask_stack[:, i] = masks.get(name, torch.ones(n_cells, dtype=torch.bool, device=device))

        if self.modality_weights_mode == "equal":
            # Equal weights, masked for missing modalities
            weights = mask_stack.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
        elif self.modality_weights_mode == "universal":
            logits = self.modality_weight_params.unsqueeze(0).expand(n_cells, -1)
            weights = masked_softmax(logits, mask_stack, dim=-1)
        elif self.modality_weights_mode == "cell":
            # Concatenate all z for weight prediction
            z_cat = z_stack.view(n_cells, -1)
            logits = self.modality_weight_net(z_cat)
            weights = masked_softmax(logits, mask_stack, dim=-1)

        # Weighted mean: (n_cells, n_latent)
        z = (z_stack * weights.unsqueeze(-1)).sum(dim=1)
        return z

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: dict[str, torch.Tensor],
        batch_index: torch.Tensor,
        dispersion_index: torch.Tensor | None = None,
        library_size_index: torch.Tensor | None = None,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        feature_scaling_covs: torch.Tensor | None = None,
        ambient_covs: torch.Tensor | None = None,
    ) -> dict[str, dict]:
        """Per-modality decoding from shared Z.

        Returns dict with keys per modality, each containing the distribution.
        """
        from torch.nn.functional import one_hot

        from regularizedvi._module import GammaPoissonWithScale

        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        # Build one-hot indicator for scaling covariates (cell2location obs2extra_categoricals)
        if feature_scaling_covs is not None and self.n_total_feature_scaling_cats > 0:
            feature_scaling_indicator = torch.cat(
                [
                    one_hot(feature_scaling_covs[:, i].long(), n_cats).float()
                    for i, n_cats in enumerate(self.n_cats_per_feature_scaling_cov)
                ],
                dim=-1,
            )  # (n_cells, n_total_feature_scaling_cats)
        else:
            feature_scaling_indicator = None

        px_dict = {}
        px_r_mu_dict = {}
        burst_outputs = {}
        px_r_log_sigma_dict = {}
        px_r_sampled_dict = {}
        hidden_act_dict = {}
        for name in self.modality_names:
            if name not in library:
                continue

            disp = self.dispersion_dict[name]
            lib = library[name]

            # Compute additive background (concatenated one-hot @ single parameter)
            bg = None
            if name in self.additive_background and ambient_covs is not None:
                concat_ambient = torch.cat(
                    [
                        one_hot(ambient_covs[:, i].long(), int(n_cats_i)).float()
                        for i, n_cats_i in enumerate(self.n_cats_per_ambient_cov)
                    ],
                    dim=-1,
                )
                bg = torch.matmul(concat_ambient, torch.exp(self.additive_background[name]).T)

            # Decode
            _mod_decoder_type = self.decoder_type_dict[name]
            _burst_freq = _burst_size = None
            if self.use_batch_in_decoder:
                _dec_out = self.decoders[name](
                    disp,
                    decoder_input,
                    lib,
                    batch_index,
                    *categorical_input,
                    additive_background=bg,
                )
            else:
                _dec_out = self.decoders[name](
                    disp,
                    decoder_input,
                    lib,
                    *categorical_input,
                    additive_background=bg,
                )
            if _mod_decoder_type == "burst_frequency_size":
                px_scale, px_r_cell, px_rate, px_dropout, hidden_act, _burst_freq, _burst_size = _dec_out
            else:
                px_scale, px_r_cell, px_rate, px_dropout, hidden_act = _dec_out

            # Feature scaling (cell2location-style per-covariate scaling)
            # softplus(param)/0.7: centered at ~1.0 when param=0, positive, unbounded
            _feature_scaling_factor = None
            if name in self.feature_scaling:
                rf_transformed = torch.nn.functional.softplus(self.feature_scaling[name]) / 0.7
                if feature_scaling_indicator is not None:
                    _feature_scaling_factor = torch.matmul(feature_scaling_indicator, rf_transformed)
                else:
                    _feature_scaling_factor = rf_transformed
                px_rate = px_rate * _feature_scaling_factor

            # Learnable per-modality scaling (global scale factor on expected counts)
            if name in self.modality_scale_raw:
                mod_scale = torch.nn.functional.softplus(self.modality_scale_raw[name]) / 0.7
                px_rate = px_rate * mod_scale

            # Resolve dispersion via variational LogNormal posterior
            _disp_idx = dispersion_index if dispersion_index is not None else batch_index
            if disp in ("gene-batch", "region-batch"):
                _oh = one_hot(_disp_idx.squeeze(-1).long(), self.n_dispersion_cats).float()
                _px_r_mu = torch.nn.functional.linear(_oh, self.px_r_mu[name])
                _px_r_log_sigma = torch.nn.functional.linear(_oh, self.px_r_log_sigma[name])
            elif disp in ("gene-cell", "region-cell"):
                raise ValueError(
                    f"{disp} dispersion is deprecated. Use 'gene'/'region' or 'gene-batch'/'region-batch'."
                )
            else:
                _px_r_mu = self.px_r_mu[name]
                _px_r_log_sigma = self.px_r_log_sigma[name]

            if _px_r_mu is not None:
                _px_r_sigma = torch.exp(_px_r_log_sigma)
                if self.training:
                    px_r = torch.exp(_px_r_mu + _px_r_sigma * torch.randn_like(_px_r_mu))
                else:
                    px_r = torch.exp(_px_r_mu)
            else:
                px_r = torch.exp(px_r)

            # Build likelihood
            if _mod_decoder_type == "burst_frequency_size" and _burst_freq is not None:
                # px_r (from LogNormal posterior) reused as stochastic_v (technical variance)
                stochastic_v_cg = px_r

                # sensitivity = exp(library) * feature_scaling
                _sensitivity = torch.exp(lib)
                if _feature_scaling_factor is not None:
                    _sensitivity = _sensitivity * _feature_scaling_factor

                # Form 2: alpha = mu^2 / var
                _var_biol = _burst_freq * _burst_size.pow(2)
                _var = _sensitivity.pow(2) * _var_biol + stochastic_v_cg
                _alpha = px_rate.pow(2) / (_var + 1e-8)

                # scale = burst_freq * burst_size (biological rate, for get_normalized_expression)
                _bio_rate = _burst_freq * _burst_size
                px = GammaPoissonWithScale(concentration=_alpha, rate=_alpha / px_rate, scale=_bio_rate)
            else:
                # Standard expected_RNA: concentration=theta, rate=theta/mu
                px = GammaPoissonWithScale(concentration=px_r, rate=px_r / px_rate, scale=px_scale)

            px_dict[name] = px
            if _px_r_mu is not None:
                px_r_mu_dict[name] = _px_r_mu
                px_r_log_sigma_dict[name] = _px_r_log_sigma
            px_r_sampled_dict[name] = px_r if _mod_decoder_type != "burst_frequency_size" else _alpha

            if _mod_decoder_type == "burst_frequency_size" and _burst_freq is not None:
                burst_outputs[name] = {
                    "burst_freq": _burst_freq,
                    "burst_size": _burst_size,
                    "stochastic_v_cg": stochastic_v_cg,
                    "alpha_total": _alpha,
                    "var_biol": _var_biol,
                    "var_total": _var,
                }

            hidden_act_dict[name] = hidden_act

        # Prior on z
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        # Library priors (always learned, indexed by library_size_index)
        _lib_idx = library_size_index if library_size_index is not None else batch_index
        pl_dict = {}
        for name in self.modality_names:
            if name not in library:
                continue
            means_buf = getattr(self, f"library_log_means_{name}", None)
            vars_buf = getattr(self, f"library_log_vars_{name}", None)
            if means_buf is not None and vars_buf is not None:
                local_means = torch.nn.functional.linear(
                    one_hot(_lib_idx.squeeze(-1).long(), self.n_library_cats).float(),
                    means_buf,
                )
                local_vars = torch.nn.functional.linear(
                    one_hot(_lib_idx.squeeze(-1).long(), self.n_library_cats).float(),
                    vars_buf,
                )
                pl_dict[name] = Normal(local_means, local_vars.sqrt())

        return {
            "px": px_dict,
            "pz": pz,
            "pl": pl_dict,
            "px_r_mu": px_r_mu_dict,
            "px_r_log_sigma": px_r_log_sigma_dict,
            "px_r_sampled": px_r_sampled_dict,
            "burst_outputs": burst_outputs,
            "hidden_activations": hidden_act_dict,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, dict],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute multi-modal loss.

        Sum of per-modality reconstruction losses + KL divergence + hierarchical
        dispersion priors.
        """
        masks = inference_outputs["masks"]
        px_dict = generative_outputs["px"]
        pz = generative_outputs["pz"]
        pl_dict = generative_outputs.get("pl", {})
        z_per_modality = inference_outputs.get("z_per_modality", {})

        extra_metrics = {}

        # ---- Reconstruction loss (per modality) ----
        recon_loss = torch.zeros(
            tensors[REGISTRY_KEYS.BATCH_KEY].shape[0], device=tensors[REGISTRY_KEYS.BATCH_KEY].device
        )
        for name in self.modality_names:
            if name not in px_dict:
                continue
            key = f"X_{name}"
            x = tensors.get(key)
            if x is None:
                continue
            px = px_dict[name]
            rl = -px.log_prob(x).sum(-1)
            # Mask: zero out loss for cells missing this modality
            if name in masks:
                rl = rl * masks[name].float()
            extra_metrics[f"recon_loss_{name}"] = rl.mean().detach()
            recon_loss = recon_loss + rl

        # ---- Modality scaling metrics (log current values) ----
        if self.learnable_modality_scaling:
            for name in self.modality_names:
                if name in self.modality_scale_raw:
                    w = torch.nn.functional.softplus(self.modality_scale_raw[name]) / 0.7
                    extra_metrics[f"modality_scale_{name}"] = w.detach()

        # ---- KL divergence on Z ----
        qz_per_modality = inference_outputs["qz_per_modality"]
        if self.latent_mode == "single_encoder":
            qz = qz_per_modality["_joint"]
            kl_z = kld(qz, pz).sum(dim=-1)
        elif self.latent_mode == "concatenation":
            # Sum KL for each modality's encoder
            kl_z = torch.zeros_like(recon_loss)
            for name in self.modality_names:
                if name not in qz_per_modality:
                    continue
                qz = qz_per_modality[name]
                n_lat = self.n_latent_dict[name]
                prior = Normal(
                    torch.zeros(n_lat, device=recon_loss.device),
                    torch.ones(n_lat, device=recon_loss.device),
                )
                kl_mod = kld(qz, prior).sum(dim=-1)
                if name in masks:
                    kl_mod = kl_mod * masks[name].float()
                extra_metrics[f"kl_z_{name}"] = kl_mod.mean().detach()
                kl_z = kl_z + kl_mod
        elif self.latent_mode == "weighted_mean":
            # KL on the mixed z (approximate): penalise each modality's encoder
            kl_z = torch.zeros_like(recon_loss)
            for name in self.modality_names:
                if name not in qz_per_modality:
                    continue
                qz = qz_per_modality[name]
                prior = Normal(
                    torch.zeros(self.total_latent_dim, device=recon_loss.device),
                    torch.ones(self.total_latent_dim, device=recon_loss.device),
                )
                kl_mod = kld(qz, prior).sum(dim=-1)
                if name in masks:
                    kl_mod = kl_mod * masks[name].float()
                extra_metrics[f"kl_z_{name}"] = kl_mod.mean().detach()
                kl_z = kl_z + kl_mod

        # ---- Z variance per modality (latent utilization) ----
        for name in self.modality_names:
            if name in z_per_modality:
                z_mod = z_per_modality[name]
                extra_metrics[f"z_var_{name}"] = z_mod.var(dim=0).mean().detach()

        # ---- KL divergence on library (always learned) ----
        kl_l = torch.zeros_like(recon_loss)
        ql_per_modality = inference_outputs.get("ql_per_modality", {})
        for name in self.modality_names:
            if name not in ql_per_modality or name not in pl_dict:
                continue
            kl_lib = kld(ql_per_modality[name], pl_dict[name]).sum(dim=1)
            if name in masks:
                kl_lib = kl_lib * masks[name].float()
            kl_l = kl_l + kl_lib

        # ---- KL on residual library weight w (Change 3) ----
        kl_w_total = torch.tensor(0.0, device=recon_loss.device)
        if self.residual_library_encoder:
            for name in self.modality_names:
                if name not in ql_per_modality:
                    continue
                w_mu = self.library_obs_w_mu[name]
                w_sigma = torch.exp(self.library_obs_w_log_sigma[name])
                # KL(LogNormal(mu, sigma) || Exponential(rate))
                # = -H(q) + rate * E_q[w] - log(rate)
                q_w = LogNormal(w_mu, w_sigma)
                exp_rate = self.library_obs_w_prior_rate
                kl_w = -q_w.entropy() + exp_rate * q_w.mean - np.log(exp_rate)
                kl_w_total = kl_w_total + kl_w
                extra_metrics[f"library_obs_w_{name}"] = q_w.mean.detach()

        # ---- Z sparsity: Gamma prior on |z| ----
        if self.z_sparsity_prior == "gamma":
            z = inference_outputs["z"]
            _shape_z = torch.tensor(
                self.n_active_latent_per_cell / self.total_latent_dim,
                device=z.device,
                dtype=z.dtype,
            )
            _gamma_z = Gamma(
                concentration=_shape_z,
                rate=torch.tensor(1.0, device=z.device, dtype=z.dtype),
            )
            z_sparsity_penalty = -_gamma_z.log_prob(z.abs() + 1e-8).sum(dim=-1)
            extra_metrics["z_sparsity_penalty"] = z_sparsity_penalty.mean().detach()
            kl_z = kl_z + z_sparsity_penalty

        # ---- Hidden activation sparsity: Gamma prior on decoder hidden activations (per-cell) ----
        if self.hidden_activation_sparsity:
            hidden_act_dict = generative_outputs["hidden_activations"]
            for name in self.modality_names:
                if name not in hidden_act_dict:
                    continue
                _hidden_act = hidden_act_dict[name]
                _shape_h = torch.tensor(
                    self.n_active_hidden_per_cell / _hidden_act.shape[-1],
                    device=_hidden_act.device,
                    dtype=_hidden_act.dtype,
                )
                _gamma_h = Gamma(
                    concentration=_shape_h,
                    rate=torch.tensor(1.0, device=_hidden_act.device, dtype=_hidden_act.dtype),
                )
                _h_penalty = -_gamma_h.log_prob(_hidden_act + 1e-8).sum(dim=-1)
                kl_z = kl_z + _h_penalty
                extra_metrics[f"hidden_sparsity_{name}"] = _h_penalty.mean().detach()

        # ---- Weighted loss ----
        loss = torch.mean(recon_loss + kl_weight * kl_z + kl_l) + kl_w_total

        # ---- Variational dispersion regularisation (replaces MAP with proper posterior) ----
        if self.regularise_dispersion:
            n_obs = recon_loss.shape[0]
            dispersion_penalty = torch.tensor(0.0, device=recon_loss.device)

            for name in self.modality_names:
                if name not in self.px_r_mu:
                    continue

                # Use raw parameter tensors for KL (not per-cell resolved)
                px_r_log_sigma = self.px_r_log_sigma[name]
                px_r_sigma = torch.exp(px_r_log_sigma)

                # Level 1: learned rate with Gamma hyper-prior (per-modality)
                learned_rate = torch.nn.functional.softplus(self.dispersion_prior_rate_raw[name])
                _hp_alpha = self.dispersion_hyper_prior_alpha_dict[name]
                _hp_beta = self.dispersion_hyper_prior_beta_dict[name]
                neg_log_hyper = -Gamma(_hp_alpha, _hp_beta).log_prob(learned_rate).sum()

                # Level 2: Exponential prior on dispersion
                disp = self.dispersion_dict[name]
                if disp in ("gene-batch", "region-batch"):
                    rate = learned_rate.unsqueeze(0).expand_as(self.px_r_mu[name])
                elif disp in ("gene-label", "region-label"):
                    rate = learned_rate.unsqueeze(0).expand_as(self.px_r_mu[name])
                else:
                    rate = learned_rate.expand_as(self.px_r_mu[name])

                # Analytic LogNormal entropy
                entropy = (px_r_log_sigma + 0.5 * math.log(2 * math.pi * math.e)).sum()

                # MC estimate of E_q[log p(transform(theta))] using fresh sample
                px_r_sample = torch.exp(self.px_r_mu[name] + px_r_sigma * torch.randn_like(self.px_r_mu[name]))
                if self.decoder_type_dict.get(name) == "burst_frequency_size":
                    # Prior on sqrt(v) ~ Exp(lambda): pushes technical variance toward zero
                    px_r_transformed = px_r_sample.pow(0.5)
                else:
                    # Prior on 1/sqrt(theta) ~ Exp(lambda): pushes theta toward large (Poisson)
                    px_r_transformed = (1.0 / px_r_sample).pow(0.5)
                log_prior = Exponential(rate).log_prob(px_r_transformed).sum()

                # KL = -entropy - E_q[log p]
                dispersion_kl = -entropy - log_prior
                dispersion_penalty = dispersion_penalty + dispersion_kl + neg_log_hyper

            loss = loss + dispersion_penalty / n_obs

        # ---- Feature scaling Gamma prior (cell2location-style) ----
        if self.feature_scaling_modalities:
            n_obs = recon_loss.shape[0]
            rf_penalty = torch.tensor(0.0, device=recon_loss.device)
            for name in self.feature_scaling_modalities:
                if name not in self.feature_scaling:
                    continue
                rf_transformed = torch.nn.functional.softplus(self.feature_scaling[name]) / 0.7
                neg_log_prior = (
                    -Gamma(
                        self.feature_scaling_prior_alpha,
                        self.feature_scaling_prior_beta,
                    )
                    .log_prob(rf_transformed)
                    .sum()
                )
                rf_penalty = rf_penalty + neg_log_prior
            loss = loss + rf_penalty / n_obs

        # ---- Additive background Gamma prior (cell2location-style s_g_gene_add) ----
        if self.regularise_background and self.additive_background_modalities:
            n_obs = recon_loss.shape[0]
            bg_penalty = torch.tensor(0.0, device=recon_loss.device)
            for name in self.additive_background_modalities:
                if name not in self.additive_background:
                    continue
                bg_transformed = torch.exp(self.additive_background[name])
                neg_log_prior = (
                    -Gamma(self.additive_bg_prior_alpha, self.additive_bg_prior_beta).log_prob(bg_transformed).sum()
                )
                bg_penalty = bg_penalty + neg_log_prior
            loss = loss + bg_penalty / n_obs

        # ---- Decoder weight L2 penalty (Normal prior on decoder weights, excludes biases) ----
        if self.decoder_weight_l2 > 0.0:
            n_obs = recon_loss.shape[0]
            decoder_w_penalty = self.decoder_weight_l2 * self._decoder_weight_l2_penalty()
            loss = loss + decoder_w_penalty / n_obs
            extra_metrics["decoder_weight_penalty"] = (decoder_w_penalty / n_obs).detach()

        # ---- Decoder covariate weight L2 penalty ----
        if self.decoder_cov_weight_l2 > 0.0:
            decoder_cov_penalty = self.decoder_cov_weight_l2 * self._decoder_cov_weight_l2_penalty()
            loss = loss + decoder_cov_penalty / n_obs
            extra_metrics["decoder_cov_weight_penalty"] = (decoder_cov_penalty / n_obs).detach()

        # ---- Decoder weight L1 penalty (sparsity-inducing) ----
        if self.decoder_hidden_l1 > 0.0:
            n_obs = recon_loss.shape[0]
            decoder_l1_penalty = self.decoder_hidden_l1 * self._decoder_hidden_l1_penalty()
            loss = loss + decoder_l1_penalty / n_obs
            extra_metrics["decoder_l1_penalty"] = (decoder_l1_penalty / n_obs).detach()

        # ---- Modality scaling Gamma prior ----
        if self.learnable_modality_scaling and self.modality_scale_init:
            n_obs = recon_loss.shape[0]
            ms_penalty = torch.tensor(0.0, device=recon_loss.device)
            for name, init_val in self.modality_scale_init.items():
                if name not in self.modality_scale_raw:
                    continue
                w = torch.nn.functional.softplus(self.modality_scale_raw[name]) / 0.7
                conc = self.modality_scale_prior_concentration
                # Gamma(conc, conc/init_val) has mean=init_val
                neg_log_prior = -Gamma(conc, conc / init_val).log_prob(w).sum()
                ms_penalty = ms_penalty + neg_log_prior
            loss = loss + ms_penalty / n_obs

        # ---- Pearson correlation metrics (per modality) ----
        # Normalize to per-cell proportions to remove library size confound:
        # px_rate is library-scaled, so raw correlation is dominated by cell size.
        if self.compute_pearson:
            for name in self.modality_names:
                if name not in px_dict:
                    continue
                key = f"X_{name}"
                x_obs = tensors.get(key)
                if x_obs is None:
                    continue
                px_rate = px_dict[name].mean.detach()
                x_obs = x_obs.detach()
                px_props = px_rate / px_rate.sum(dim=1, keepdim=True)
                x_props = x_obs / x_obs.sum(dim=1, keepdim=True)
                # Gene-wise: transpose so genes/features are rows, cells are columns
                pearson_gene = _pearson_corr_rows(px_props.T, x_props.T).mean()
                # Cell-wise: cells are rows, genes/features are columns
                pearson_cell = _pearson_corr_rows(px_props, x_props).mean()
                extra_metrics[f"pearson_gene_{name}"] = pearson_gene
                extra_metrics[f"pearson_cell_{name}"] = pearson_cell

        # ---- Burst frequency/size metrics (for burst_frequency_size modalities) ----
        px_r_sampled_dict = generative_outputs.get("px_r_sampled", {})
        burst_outputs = generative_outputs.get("burst_outputs", {})
        for name in self.modality_names:
            if name in px_r_sampled_dict:
                extra_metrics[f"theta_mean_{name}"] = px_r_sampled_dict[name].detach().mean()
            if name in burst_outputs:
                bo = burst_outputs[name]
                extra_metrics[f"burst_freq_mean_{name}"] = bo["burst_freq"].detach().mean()
                extra_metrics[f"burst_size_mean_{name}"] = bo["burst_size"].detach().mean()
                extra_metrics[f"stochastic_v_mean_{name}"] = bo["stochastic_v_cg"].detach().mean()
                extra_metrics[f"alpha_total_mean_{name}"] = bo["alpha_total"].detach().mean()
                extra_metrics[f"var_biol_mean_{name}"] = bo["var_biol"].detach().mean()
                extra_metrics[f"var_total_mean_{name}"] = bo["var_total"].detach().mean()
                extra_metrics[f"var_biol_frac_{name}"] = (bo["var_biol"] / (bo["var_total"] + 1e-8)).detach().mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local={
                "kl_divergence_z": kl_z,
                "kl_divergence_l": kl_l,
            },
            extra_metrics=extra_metrics,
        )

    _PER_MODALITY_CONTAINERS = [
        "encoders",
        "l_encoders",
        "decoders",
        "px_r_mu",
        "px_r_log_sigma",
        "dispersion_prior_rate_raw",
        "additive_background",
        "feature_scaling",
        "modality_scale_raw",
    ]

    def get_parameter_groups(
        self,
        base_lr: float,
        modality_lr_multiplier: dict[str, float],
    ) -> list[dict]:
        """Build optimizer parameter groups with per-modality learning rates.

        Parameters
        ----------
        base_lr
            Base learning rate for shared parameters and modalities without a multiplier.
        modality_lr_multiplier
            Mapping from modality name to LR multiplier (e.g., ``{"atac": 2.0}``).
            Modalities not listed use ``base_lr``.

        Returns
        -------
        List of dicts compatible with PyTorch optimizer param groups.
        """
        unknown = set(modality_lr_multiplier) - set(self.modality_names)
        if unknown:
            msg = f"Unknown modality names in modality_lr_multiplier: {unknown}. Available: {self.modality_names}"
            raise ValueError(msg)

        # Collect param ids per modality
        modality_param_ids: dict[str, set[int]] = {name: set() for name in self.modality_names}
        for container_name in self._PER_MODALITY_CONTAINERS:
            container = getattr(self, container_name, None)
            if container is None:
                continue
            for name in self.modality_names:
                if name not in container:
                    continue
                entry = container[name]
                if isinstance(entry, nn.Module):
                    for p in entry.parameters():
                        if p.requires_grad:
                            modality_param_ids[name].add(id(p))
                elif isinstance(entry, nn.Parameter):
                    if entry.requires_grad:
                        modality_param_ids[name].add(id(entry))

        # All assigned param ids
        all_assigned = set()
        for ids in modality_param_ids.values():
            all_assigned |= ids

        # Build groups
        param_groups = []
        for name in self.modality_names:
            params = [p for p in self.parameters() if p.requires_grad and id(p) in modality_param_ids[name]]
            if params:
                lr = base_lr * modality_lr_multiplier.get(name, 1.0)
                param_groups.append({"params": params, "lr": lr})

        # Shared params (not assigned to any modality)
        shared_params = [p for p in self.parameters() if p.requires_grad and id(p) not in all_assigned]
        if shared_params:
            param_groups.append({"params": shared_params, "lr": base_lr})

        return param_groups
