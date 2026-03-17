"""RegularizedVAE module for regularizedvi.

Modified from scvi-tools VAE (scvi.module._vae) with the following changes:

1. Uses RegularizedEncoder and RegularizedDecoderSCVI from _components.py
2. Additive ambient RNA background: per-gene, per-batch learnable parameter
3. Overdispersion regularisation: Exponential prior pushing NB toward Poisson
4. Batch-free decoder: batch info removed from decoder when use_batch_in_decoder=False
5. Learned library size with constrained prior (library_log_vars_weight)
6. Separate library_n_hidden for low-capacity library encoder
7. Support for dropout_on_input via extra_encoder_kwargs/extra_decoder_kwargs
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import (
    BaseMinifiedModeModuleClass,
    EmbeddingModuleMixin,
    LossOutput,
    auto_move_data,
)
from torch.distributions import Exponential, Normal
from torch.nn.functional import one_hot

from regularizedvi._constants import (
    AMBIENT_COVS_KEY,
    DEFAULT_COMPUTE_PEARSON,
    DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
    DEFAULT_FEATURE_SCALING_PRIOR_BETA,
    DEFAULT_USE_FEATURE_SCALING,
    DISPERSION_KEY,
    ENCODER_COVS_KEY,
    FEATURE_SCALING_COVS_KEY,
    LIBRARY_SIZE_KEY,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)


def _pearson_corr_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson r between x and y (both shape [N, M])."""
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    num = (x_centered * y_centered).sum(dim=1)
    den = (x_centered.norm(dim=1) * y_centered.norm(dim=1)).clamp(min=1e-8)
    return num / den


class GammaPoissonWithScale:
    """Thin wrapper around Pyro GammaPoisson for scvi-tools compatibility.

    scvi-tools expects distributions to have a ``.scale`` property and
    ``get_normalized()`` method. This wrapper adds those while delegating
    ``log_prob`` and other methods to Pyro's ``GammaPoisson``.
    """

    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, scale: torch.Tensor | None = None):
        from pyro.distributions import GammaPoisson

        self._dist = GammaPoisson(concentration=concentration, rate=rate)
        self._scale = scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(value)

    @property
    def scale(self) -> torch.Tensor | None:
        return self._scale

    @property
    def mean(self) -> torch.Tensor:
        return self._dist.mean

    def get_normalized(self, key: str) -> torch.Tensor:
        if key == "scale":
            return self._scale
        if key == "mu":
            return self.mean
        raise ValueError(f"Normalized key {key!r} not recognized")

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._dist.sample(sample_shape)


class RegularizedVAE(EmbeddingModuleMixin, BaseMinifiedModeModuleClass):
    """Regularized variational auto-encoder with ambient RNA correction.

    Extends the standard scVI VAE :cite:p:`Lopez18` with:

    - **Ambient RNA correction**: per-gene, per-batch additive background term
      ``b_{g,s_n} = exp(beta_{g,s_n})``, mirroring cell2location's
      ``(g_fg + b_eg) * h_e`` structure :cite:p:`Kleshchevnikov22`
      and cell2fate's Bayesian modelling principles :cite:p:`Aivazidis25`.
    - **Dispersion regularisation**: Exponential containment prior on
      ``1/sqrt(theta)`` regularising against excessive overdispersion
      :cite:p:`Simpson17` :cite:p:`Kleshchevnikov22` :cite:p:`Aivazidis25`.
    - **Batch-free decoder**: batch correction through additive background
      and categorical covariates rather than decoder conditioning.
    - **Learned library size**: with constrained prior variance to prevent
      library size from absorbing biological signal.

    Parameters
    ----------
    n_input
        Number of input features (genes).
    n_batch
        Number of batches. If ``0``, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer for encoder and decoder.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers for encoder and decoder.
    n_continuous_cov
        Number of continuous covariates.
    n_cats_per_cov
        Number of categories for each categorical covariate.
    dropout_rate
        Dropout rate for encoder.
    dispersion
        Flexibility of the dispersion parameter.
    log_variational
        If ``True``, use log1p on input before encoding.
    gene_likelihood
        Distribution for reconstruction.
    latent_distribution
        Distribution for the latent space.
    encode_covariates
        If ``True``, covariates are concatenated to gene expression for encoding.
    deeply_inject_covariates
        If ``True``, covariates injected into each hidden layer.
    batch_representation
        Method for encoding batch information.
    use_batch_norm
        Where to use BatchNorm.
    use_layer_norm
        Where to use LayerNorm.
    use_size_factor_key
        If ``True``, use size factor from anndata.
    use_observed_lib_size
        If ``True``, use observed library size.
    library_log_means
        Prior means for log library sizes.
    library_log_vars
        Prior variances for log library sizes.
    var_activation
        Callable for variance positivity.
    extra_encoder_kwargs
        Additional kwargs for encoder (e.g., ``dropout_on_input=True``).
    extra_decoder_kwargs
        Additional kwargs for decoder.
    batch_embedding_kwargs
        Kwargs for batch embedding.
    library_log_vars_weight
        Scale factor for library log variance prior. Smaller values constrain
        the library size more tightly, preventing it from absorbing biological signal.
    library_n_hidden
        Hidden units in library encoder. Small values (16-32) prevent the
        library encoder from becoming a pathway for biological signal.
        If None, uses n_hidden.
    scale_activation
        Override for decoder scale activation. If provided, overrides the
        default logic based on use_size_factor_key. Use "softplus" for
        regularizedvi (expression is no longer on the simplex).
    use_additive_background
        If True, add per-gene, per-batch learnable ambient RNA background.
    use_batch_in_decoder
        If False, remove batch info from decoder. Batch correction handled
        through additive background and categorical covariates.
    regularise_dispersion
        If True, add Exponential containment prior on ``1/sqrt(theta)``.
    regularise_dispersion_prior
        Rate parameter for the Exponential containment prior on ``1/sqrt(theta)``.
    dispersion_hyper_prior_alpha
        Alpha parameter for the Gamma hyper-prior on the learned dispersion
        rate parameter. Default ``9.0`` (from cell2location).
    dispersion_hyper_prior_beta
        Beta parameter for the Gamma hyper-prior on the learned dispersion
        rate parameter. Default ``3.0`` (from cell2location).
        Together with alpha, gives Gamma(9, 3) with mean=3.0.
    additive_bg_prior_alpha
        Alpha parameter for the Gamma prior on ``exp(additive_background)``.
        Default ``1.0`` (cell2location-style ``gene_add_mean_hyp_prior``).
    additive_bg_prior_beta
        Beta parameter for the Gamma prior. Default ``100.0``.
        Together with alpha, gives Gamma(1, 100) with mean=0.01.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene-batch",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        batch_representation: Literal["one-hot", "embedding"] = "one-hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = False,
        extra_payload_autotune: bool = False,
        library_log_means: np.ndarray | None = None,
        library_log_vars: np.ndarray | None = None,
        var_activation: Callable[[torch.Tensor], torch.Tensor] = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
        batch_embedding_kwargs: dict | None = None,
        # regularizedvi-specific parameters
        library_log_vars_weight: float | None = None,
        library_log_means_centering_sensitivity: float | None = None,
        library_n_hidden: int | None = None,
        scale_activation: str | None = None,
        use_additive_background: bool = False,
        use_batch_in_decoder: bool = True,
        regularise_dispersion: bool = False,
        regularise_dispersion_prior: float = 3.0,
        dispersion_hyper_prior_alpha: float = 9.0,
        dispersion_hyper_prior_beta: float = 3.0,
        additive_bg_prior_alpha: float = 1.0,
        additive_bg_prior_beta: float = 100.0,
        regularise_background: bool = True,
        # Ambient covariates (for additive background, decoupled from batch_key)
        n_cats_per_ambient_cov: list[int] | None = None,
        # Feature scaling covariates (cell2location-style per-feature multiplicative scaling)
        n_cats_per_feature_scaling_cov: list[int] | None = None,
        # Encoder covariates (what categoricals the encoder sees, default=None → nothing)
        n_cats_per_encoder_cov: list[int] | None = None,
        use_feature_scaling: bool = DEFAULT_USE_FEATURE_SCALING,
        feature_scaling_prior_alpha: float = DEFAULT_FEATURE_SCALING_PRIOR_ALPHA,
        feature_scaling_prior_beta: float = DEFAULT_FEATURE_SCALING_PRIOR_BETA,
        # Dispersion covariate (controls per-group px_r, decoupled from batch_key)
        n_dispersion_cats: int | None = None,
        # Library size covariate (controls per-group library prior, decoupled from batch_key)
        n_library_cats: int | None = None,
        # Parameter initialization control
        px_r_init_mean: float | None = None,
        px_r_init_std: float | None = None,
        additive_bg_init_mean: float | None = None,
        additive_bg_init_std: float | None = None,
        # Decoder weight regularization
        decoder_weight_l2: float = 0.0,
        # Data-dependent initialization
        decoder_bias_init: np.ndarray | None = None,
        additive_bg_init_per_gene: np.ndarray | None = None,
        # Training metrics
        compute_pearson: bool = DEFAULT_COMPUTE_PEARSON,
    ):
        from regularizedvi._components import RegularizedDecoderSCVI, RegularizedEncoder

        super().__init__()

        self.compute_pearson = compute_pearson
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.extra_payload_autotune = extra_payload_autotune
        self.use_additive_background = use_additive_background
        self.use_batch_in_decoder = use_batch_in_decoder
        self.regularise_dispersion = regularise_dispersion
        self.regularise_dispersion_prior = regularise_dispersion_prior
        self.dispersion_hyper_prior_alpha = dispersion_hyper_prior_alpha
        self.dispersion_hyper_prior_beta = dispersion_hyper_prior_beta
        self.additive_bg_prior_alpha = additive_bg_prior_alpha
        self.additive_bg_prior_beta = additive_bg_prior_beta
        self.regularise_background = regularise_background
        self.decoder_weight_l2 = decoder_weight_l2

        # Dispersion covariate (decoupled from batch_key, fallback to n_batch)
        self.n_dispersion_cats = n_dispersion_cats if n_dispersion_cats is not None else n_batch
        # Library size covariate (decoupled from batch_key, fallback to n_batch)
        self.n_library_cats = n_library_cats if n_library_cats is not None else n_batch

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError("If not using observed_lib_size, must provide library_log_means and library_log_vars.")

            # Optionally center library_log_means: subtract global mean, shift by log(sensitivity)
            # This makes exp(library) ≈ sensitivity at initialization instead of raw total counts.
            log_means = torch.from_numpy(library_log_means).float()
            if library_log_means_centering_sensitivity is not None:
                global_log_mean = log_means.mean()
                log_means = log_means - global_log_mean + math.log(library_log_means_centering_sensitivity)
            self.register_buffer("library_log_means", log_means)
            # Scale library_log_vars by weight to constrain the prior
            log_vars = torch.from_numpy(library_log_vars).float()
            if library_log_vars_weight is not None:
                log_vars = log_vars * library_log_vars_weight
            self.register_buffer("library_log_vars", log_vars)

        # Initialize px_r variational posterior at prior equilibrium when regularisation is active.
        # Variational LogNormal posterior: q(theta) = LogNormal(mu, sigma)
        # mu is initialized at log(rate^2) so E[theta] ≈ rate^2 at init.
        # log_sigma is initialized at log(0.1) for small initial variance.
        # Override with px_r_init_mean/px_r_init_std for ablation experiments.
        if px_r_init_mean is not None:
            _px_r_init = px_r_init_mean
        elif self.regularise_dispersion:
            _rate = regularise_dispersion_prior
            _px_r_init = math.log(_rate**2)  # log(9) ≈ 2.197
        else:
            _px_r_init = None
        _px_r_std = px_r_init_std if px_r_init_std is not None else (0.1 if _px_r_init is not None else 1.0)
        _log_sigma_init = math.log(0.1)

        if self.dispersion == "gene":
            if _px_r_init is not None:
                self.px_r_mu = torch.nn.Parameter(torch.full((n_input,), _px_r_init) + _px_r_std * torch.randn(n_input))
            else:
                self.px_r_mu = torch.nn.Parameter(_px_r_std * torch.randn(n_input))
            self.px_r_log_sigma = torch.nn.Parameter(torch.full((n_input,), _log_sigma_init))
        elif self.dispersion == "gene-batch":
            n_disp = self.n_dispersion_cats
            if _px_r_init is not None:
                self.px_r_mu = torch.nn.Parameter(
                    torch.full((n_input, n_disp), _px_r_init) + _px_r_std * torch.randn(n_input, n_disp)
                )
            else:
                self.px_r_mu = torch.nn.Parameter(_px_r_std * torch.randn(n_input, n_disp))
            self.px_r_log_sigma = torch.nn.Parameter(torch.full((n_input, n_disp), _log_sigma_init))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError("`dispersion` must be one of 'gene', 'gene-batch', 'gene-cell'.")

        # Learnable dispersion prior rate (cell2location-style hierarchical prior)
        # Initialized at inverse_softplus(regularise_dispersion_prior) so that
        # softplus(raw) = regularise_dispersion_prior at initialization.
        # Gamma(alpha, beta) hyper-prior keeps the rate near the default.
        if self.regularise_dispersion:
            _init_rate = regularise_dispersion_prior
            # inverse softplus: log(exp(x) - 1)
            _raw_init = torch.log(torch.expm1(torch.tensor(_init_rate)))
            if self.dispersion == "gene-batch":
                self.dispersion_prior_rate_raw = torch.nn.Parameter(_raw_init.expand(self.n_dispersion_cats).clone())
            else:
                self.dispersion_prior_rate_raw = torch.nn.Parameter(_raw_init.unsqueeze(0).clone())

        # Encoder covariates (dedicated registry key, default=None → no encoder categoricals)
        self.n_cats_per_encoder_cov = list(n_cats_per_encoder_cov) if n_cats_per_encoder_cov else []
        self.n_total_encoder_cats = (
            sum(int(c) for c in self.n_cats_per_encoder_cov) if self.n_cats_per_encoder_cov else 0
        )

        # Ambient covariates (for additive background, decoupled from batch_key)
        self.n_cats_per_ambient_cov = list(n_cats_per_ambient_cov) if n_cats_per_ambient_cov is not None else []

        # Additive background: per-gene learnable parameter with concatenated ambient covariates.
        # Shape: (n_input, sum(n_cats_per_ambient_cov)) — single parameter, one-hot covariates
        # are concatenated and multiplied in one matmul.
        # Initialized at log(prior_mean) = log(alpha/beta) so exp(init) matches prior mean.
        # Default Gamma(1, 100) → mean = 0.01, small relative to px_scale.
        self.n_total_ambient_cats = (
            sum(int(c) for c in self.n_cats_per_ambient_cov) if self.n_cats_per_ambient_cov else 0
        )
        if self.use_additive_background and self.n_total_ambient_cats > 0:
            _bg_std = additive_bg_init_std if additive_bg_init_std is not None else 0.01
            if additive_bg_init_per_gene is not None:
                init_vals = torch.from_numpy(additive_bg_init_per_gene).float()
                init_vals = init_vals.unsqueeze(1).expand(-1, self.n_total_ambient_cats)
                self.additive_background = torch.nn.Parameter(
                    init_vals + _bg_std * torch.randn(n_input, self.n_total_ambient_cats)
                )
            else:
                _bg_mean = (
                    additive_bg_init_mean
                    if additive_bg_init_mean is not None
                    else math.log(additive_bg_prior_alpha / additive_bg_prior_beta)
                )
                self.additive_background = torch.nn.Parameter(
                    torch.full((n_input, self.n_total_ambient_cats), _bg_mean)
                    + _bg_std * torch.randn(n_input, self.n_total_ambient_cats)
                )

        # Feature scaling (cell2location-style per-covariate multiplicative scaling)
        self.use_feature_scaling = use_feature_scaling
        self.n_cats_per_feature_scaling_cov = (
            list(n_cats_per_feature_scaling_cov) if n_cats_per_feature_scaling_cov else []
        )
        self.n_total_feature_scaling_cats = (
            sum(self.n_cats_per_feature_scaling_cov) if self.n_cats_per_feature_scaling_cov else 0
        )
        if self.use_feature_scaling:
            n_feature_scaling_rows = self.n_total_feature_scaling_cats if self.n_total_feature_scaling_cats > 0 else 1
            self.feature_scaling = torch.nn.Parameter(torch.zeros(n_feature_scaling_rows, n_input))
        self.feature_scaling_prior_alpha = feature_scaling_prior_alpha
        self.feature_scaling_prior_beta = feature_scaling_prior_beta

        self.batch_representation = batch_representation
        if self.batch_representation == "embedding":
            self.init_embedding(REGISTRY_KEYS.BATCH_KEY, n_batch, **(batch_embedding_kwargs or {}))
            batch_dim = self.get_embedding(REGISTRY_KEYS.BATCH_KEY).embedding_dim
        elif self.batch_representation != "one-hot":
            raise ValueError("`batch_representation` must be one of 'one-hot', 'embedding'.")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        n_input_encoder = n_input + n_continuous_cov
        if self.batch_representation == "embedding" and encode_covariates:
            n_input_encoder += batch_dim

        # Encoder/decoder covariate composition (purpose-driven keys)
        # Encoder sees: only encoder_covs (from dedicated encoder_covariate_keys registry)
        # Default: no encoder categoricals (matches scVI/MultiVI/PeakVI default)
        # Decoder sees: [cat_covs...] only (no batch injection by default)
        encoder_cat_list = list(self.n_cats_per_encoder_cov) if self.n_cats_per_encoder_cov else None
        _cat_cats = list(n_cats_per_cov or [])
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = RegularizedEncoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        # Use library_n_hidden for low-capacity library encoder (default 16)
        from regularizedvi._constants import DEFAULT_LIBRARY_N_HIDDEN

        _library_n_hidden = library_n_hidden if library_n_hidden is not None else DEFAULT_LIBRARY_N_HIDDEN
        if _library_n_hidden > 32:
            warnings.warn(
                f"library_n_hidden={_library_n_hidden} is large. The library encoder "
                f"should be low-capacity (16-32 hidden units) to prevent overfitting "
                f"the library size. Consider setting library_n_hidden=16.",
                UserWarning,
                stacklevel=2,
            )
        self.l_encoder = RegularizedEncoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=_library_n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )

        # Decoder setup
        n_input_decoder = n_latent + n_continuous_cov
        if self.use_batch_in_decoder:
            if self.batch_representation == "embedding":
                n_input_decoder += batch_dim
                decoder_cat_list = _cat_cats
            else:
                decoder_cat_list = [n_batch] + _cat_cats
        else:
            # Batch-free decoder: only categorical covariates, no batch
            decoder_cat_list = _cat_cats

        # Determine scale activation
        if scale_activation is not None:
            _scale_activation = scale_activation
        elif use_size_factor_key:
            _scale_activation = "softplus"
        else:
            _scale_activation = "softmax"

        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = RegularizedDecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=decoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation=_scale_activation,
            **_extra_decoder_kwargs,
        )

        # Data-dependent decoder bias initialization (Stream B)
        if decoder_bias_init is not None:
            init_vals = torch.from_numpy(decoder_bias_init).float()
            init_vals = torch.clamp(init_vals, min=0.01)
            # softplus_inv(x) = log(exp(x) - 1), numerically stable for large x
            bias_init = torch.where(
                init_vals > 20.0,
                init_vals,  # softplus_inv(x) ≈ x for large x
                torch.log(torch.expm1(init_vals)),
            )
            with torch.no_grad():
                self.decoder.px_scale_decoder[0].bias.copy_(bias_init)

    def _decoder_weight_l2_penalty(self) -> torch.Tensor:
        """Sum of squared weights in decoder FC layers (excludes biases)."""
        penalty = torch.tensor(0.0, device=next(self.decoder.parameters()).device)
        for layer_seq in self.decoder.px_decoder.fc_layers:
            for sublayer in layer_seq:
                if isinstance(sublayer, torch.nn.Linear):
                    penalty = penalty + sublayer.weight.pow(2).sum()
        penalty = penalty + self.decoder.px_scale_decoder[0].weight.pow(2).sum()
        return penalty

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        from scvi.data._constants import ADATA_MINIFY_TYPE

        if self.minified_data_type is None:
            return {
                MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],
                MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
                MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
                MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
                "encoder_covs": tensors.get(ENCODER_COVS_KEY, None),
                "library_size_index": tensors.get(LIBRARY_SIZE_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
            }
        elif self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            return {
                MODULE_KEYS.QZM_KEY: tensors[REGISTRY_KEYS.LATENT_QZM_KEY],
                MODULE_KEYS.QZV_KEY: tensors[REGISTRY_KEYS.LATENT_QZV_KEY],
                REGISTRY_KEYS.OBSERVED_LIB_SIZE: tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE],
            }
        else:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the generative process."""
        size_factor = tensors.get(REGISTRY_KEYS.SIZE_FACTOR_KEY, None)
        if size_factor is not None:
            size_factor = torch.log(size_factor)

        return {
            MODULE_KEYS.Z_KEY: inference_outputs[MODULE_KEYS.Z_KEY],
            MODULE_KEYS.LIBRARY_KEY: inference_outputs[MODULE_KEYS.LIBRARY_KEY],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
            MODULE_KEYS.CONT_COVS_KEY: tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            MODULE_KEYS.CAT_COVS_KEY: tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
            MODULE_KEYS.SIZE_FACTOR_KEY: size_factor,
            "ambient_covs": tensors.get(AMBIENT_COVS_KEY, None),
            "feature_scaling_covs": tensors.get(FEATURE_SCALING_COVS_KEY, None),
            "dispersion_index": tensors.get(DISPERSION_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
            "library_size_index": tensors.get(LIBRARY_SIZE_KEY, tensors[REGISTRY_KEYS.BATCH_KEY]),
        }

    def _compute_local_library_params(
        self,
        library_size_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes local library parameters.

        Compute two tensors of shape (library_size_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the library_size_key group the cell corresponds to.
        """
        from torch.nn.functional import linear

        local_library_log_means = linear(
            one_hot(library_size_index.squeeze(-1).long(), self.n_library_cats).float(),
            self.library_log_means,
        )
        local_library_log_vars = linear(
            one_hot(library_size_index.squeeze(-1).long(), self.n_library_cats).float(),
            self.library_log_vars,
        )

        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        encoder_covs: torch.Tensor | None = None,
        library_size_index: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Run the regular inference process."""
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log1p(x_)

        if cont_covs is not None:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_

        # Build encoder categorical inputs from dedicated encoder_covariate_keys
        # Default: no encoder categoricals (encoder_covs=None, matches scVI)
        encoder_categorical_input = ()
        if encoder_covs is not None and self.n_cats_per_encoder_cov:
            encoder_categorical_input += tuple(t.long() for t in torch.split(encoder_covs, 1, dim=1))

        if self.batch_representation == "embedding" and self.encode_covariates:
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            encoder_input = torch.cat([encoder_input, batch_rep], dim=-1)

        qz, z = self.z_encoder(encoder_input, *encoder_categorical_input)

        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, *encoder_categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZ_KEY: qz,
            MODULE_KEYS.QL_KEY: ql,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data
    def _cached_inference(
        self,
        qzm: torch.Tensor,
        qzv: torch.Tensor,
        observed_lib_size: torch.Tensor,
        n_samples: int = 1,
    ) -> dict[str, torch.Tensor | None]:
        """Run the cached inference process."""
        from scvi.data._constants import ADATA_MINIFY_TYPE
        from torch.distributions import Normal

        if self.minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise NotImplementedError(f"Unknown minified-data type: {self.minified_data_type}")

        dist = Normal(qzm, qzv.sqrt())
        untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
        z = self.z_encoder.z_transformation(untran_z)
        library = torch.log(observed_lib_size)
        if n_samples > 1:
            library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))

        return {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZM_KEY: qzm,
            MODULE_KEYS.QZV_KEY: qzv,
            MODULE_KEYS.QL_KEY: None,
            MODULE_KEYS.LIBRARY_KEY: library,
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
        ambient_covs: torch.Tensor | None = None,
        feature_scaling_covs: torch.Tensor | None = None,
        dispersion_index: torch.Tensor | None = None,
        library_size_index: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process."""
        from torch.nn.functional import linear

        # Likelihood distribution
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

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        # Compute additive background if enabled (concatenated one-hot @ single parameter)
        bg = None
        if self.use_additive_background and self.n_cats_per_ambient_cov and ambient_covs is not None:
            concat_ambient = torch.cat(
                [
                    one_hot(ambient_covs[:, i].long(), int(n_cats_i)).float()
                    for i, n_cats_i in enumerate(self.n_cats_per_ambient_cov)
                ],
                dim=-1,
            )
            bg = torch.matmul(concat_ambient, torch.exp(self.additive_background).T)

        # Decoder call: batch-free or with batch
        if self.use_batch_in_decoder:
            if self.batch_representation == "embedding":
                batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
                decoder_input = torch.cat([decoder_input, batch_rep], dim=-1)
                px_scale, px_r, px_rate, px_dropout = self.decoder(
                    self.dispersion,
                    decoder_input,
                    size_factor,
                    *categorical_input,
                    additive_background=bg,
                )
            else:
                px_scale, px_r, px_rate, px_dropout = self.decoder(
                    self.dispersion,
                    decoder_input,
                    size_factor,
                    batch_index,
                    *categorical_input,
                    additive_background=bg,
                )
        else:
            # Batch-free decoder: omit batch_index, only pass categorical covariates
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                *categorical_input,
                additive_background=bg,
            )

        if self.dispersion == "gene-batch":
            _disp_idx = dispersion_index if dispersion_index is not None else batch_index
            _oh = one_hot(_disp_idx.squeeze(-1).long(), self.n_dispersion_cats).float()
            px_r_mu = linear(_oh, self.px_r_mu)
            px_r_log_sigma = linear(_oh, self.px_r_log_sigma)
        elif self.dispersion == "gene":
            px_r_mu = self.px_r_mu
            px_r_log_sigma = self.px_r_log_sigma

        # Variational LogNormal posterior: sample during training, use mean at inference
        px_r_sigma = torch.exp(px_r_log_sigma)
        if self.training:
            px_r = torch.exp(px_r_mu + px_r_sigma * torch.randn_like(px_r_mu))
        else:
            px_r = torch.exp(px_r_mu)

        # Feature scaling (cell2location-style per-covariate multiplicative scaling)
        if self.use_feature_scaling:
            fs_transformed = torch.nn.functional.softplus(self.feature_scaling) / 0.7
            if feature_scaling_covs is not None and self.n_total_feature_scaling_cats > 0:
                feature_scaling_indicator = torch.cat(
                    [
                        one_hot(feature_scaling_covs[:, i].long(), int(n)).float()
                        for i, n in enumerate(self.n_cats_per_feature_scaling_cov)
                    ],
                    dim=-1,
                )
                scaling = torch.matmul(feature_scaling_indicator, fs_transformed)
            else:
                scaling = fs_transformed  # (1, n_genes) broadcasts
            px_rate = px_rate * scaling

        # GammaPoisson (= NB): concentration=theta, rate=theta/mu
        px = GammaPoissonWithScale(concentration=px_r, rate=px_r / px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            _lib_idx = library_size_index if library_size_index is not None else batch_index
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(_lib_idx)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
            "px_r_mu": px_r_mu,
            "px_r_log_sigma": px_r_log_sigma,
            "px_r_sampled": px_r,
        }

    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor | Distribution | None],
        generative_outputs: dict[str, Distribution | None],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """Compute the loss with optional dispersion regularisation."""
        from torch.distributions import kl_divergence

        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl_divergence(
            inference_outputs[MODULE_KEYS.QZ_KEY], generative_outputs[MODULE_KEYS.PZ_KEY]
        ).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl_divergence(
                inference_outputs[MODULE_KEYS.QL_KEY], generative_outputs[MODULE_KEYS.PL_KEY]
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.zeros_like(kl_divergence_z)

        reconst_loss = -generative_outputs[MODULE_KEYS.PX_KEY].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        # Variational dispersion regularisation (replaces MAP with proper posterior).
        # Two-level prior:
        #   Level 1: alpha_g_phi_hyp ~ Gamma(alpha_prior, beta_prior)  [learned rate]
        #   Level 2: 1/sqrt(theta) ~ Exponential(alpha_g_phi_hyp)      [per-gene dispersion]
        # Variational posterior: q(log_theta) = Normal(mu, sigma) → theta = exp(mu + sigma*eps)
        # KL = -entropy(q) - E_q[log p(1/sqrt(theta))]
        # Scaled by 1/N (number of observations in mini-batch).
        if self.regularise_dispersion:
            from torch.distributions import Gamma

            n_obs = x.shape[0]

            # Use raw parameter tensors for KL (not per-cell resolved)
            px_r_log_sigma = self.px_r_log_sigma
            px_r_sigma = torch.exp(px_r_log_sigma)

            # Level 1: learned rate with Gamma hyper-prior
            learned_rate = torch.nn.functional.softplus(self.dispersion_prior_rate_raw)
            neg_log_hyper_prior = (
                -Gamma(
                    self.dispersion_hyper_prior_alpha,
                    self.dispersion_hyper_prior_beta,
                )
                .log_prob(learned_rate)
                .sum()
            )

            # Level 2: Exponential prior on dispersion with learned rate
            # Broadcast rate to match px_r_mu shape
            if self.dispersion == "gene-batch":
                rate = learned_rate.unsqueeze(0).expand_as(self.px_r_mu)
            elif self.dispersion == "gene-label":
                rate = learned_rate.unsqueeze(0).expand_as(self.px_r_mu)
            else:
                rate = learned_rate.expand_as(self.px_r_mu)

            # Analytic LogNormal entropy: H[q] = sum(log_sigma + 0.5*log(2*pi*e))
            entropy = (px_r_log_sigma + 0.5 * math.log(2 * math.pi * math.e)).sum()

            # MC estimate of E_q[log p(1/sqrt(theta))] using fresh sample from parameter-level posterior
            px_r_sample = torch.exp(self.px_r_mu + px_r_sigma * torch.randn_like(self.px_r_mu))
            px_r_transformed = (1.0 / px_r_sample).pow(0.5)
            log_prior = Exponential(rate).log_prob(px_r_transformed).sum()

            # KL = -entropy - E_q[log p]
            dispersion_kl = -entropy - log_prior
            loss = loss + (dispersion_kl + neg_log_hyper_prior) / n_obs

        # Additive background Gamma prior (cell2location-style s_g_gene_add).
        # Gamma(alpha, beta) on exp(additive_background) pushes background small.
        if self.regularise_background and self.use_additive_background and self.n_total_ambient_cats > 0:
            from torch.distributions import Gamma

            n_obs = x.shape[0]
            bg_transformed = torch.exp(self.additive_background)
            bg_penalty = (
                -Gamma(self.additive_bg_prior_alpha, self.additive_bg_prior_beta).log_prob(bg_transformed).sum()
            )
            loss = loss + bg_penalty / n_obs

        # Feature scaling Gamma prior (cell2location-style)
        if self.use_feature_scaling:
            from torch.distributions import Gamma

            n_obs = x.shape[0]
            fs_transformed = torch.nn.functional.softplus(self.feature_scaling) / 0.7
            fs_penalty = (
                -Gamma(self.feature_scaling_prior_alpha, self.feature_scaling_prior_beta).log_prob(fs_transformed).sum()
            )
            loss = loss + fs_penalty / n_obs

        # Decoder weight L2 penalty (Normal prior on decoder weights, excludes biases)
        if self.decoder_weight_l2 > 0.0:
            n_obs = x.shape[0]
            decoder_w_penalty = self.decoder_weight_l2 * self._decoder_weight_l2_penalty()
            loss = loss + decoder_w_penalty / n_obs

        # a payload to be used during autotune
        if self.extra_payload_autotune:
            extra_metrics_payload = {
                "z": inference_outputs["z"],
                "batch": tensors[REGISTRY_KEYS.BATCH_KEY],
            }
        else:
            extra_metrics_payload = {}

        if self.decoder_weight_l2 > 0.0:
            extra_metrics_payload["decoder_weight_penalty"] = (decoder_w_penalty / n_obs).detach()

        # Pearson correlation metrics (gene-wise and cell-wise)
        # Normalize to per-cell proportions to remove library size confound:
        # px_rate is library-scaled, so raw correlation is dominated by cell size.
        if self.compute_pearson:
            px_rate = generative_outputs[MODULE_KEYS.PX_KEY].mean.detach()
            x_obs = x.detach()
            px_props = px_rate / px_rate.sum(dim=1, keepdim=True)
            x_props = x_obs / x_obs.sum(dim=1, keepdim=True)
            # Gene-wise: transpose so genes are rows, cells are columns
            pearson_gene = _pearson_corr_rows(px_props.T, x_props.T).mean()
            # Cell-wise: cells are rows, genes are columns (natural layout)
            pearson_cell = _pearson_corr_rows(px_props, x_props).mean()
            extra_metrics_payload["pearson_gene"] = pearson_gene
            extra_metrics_payload["pearson_cell"] = pearson_cell

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local={
                MODULE_KEYS.KL_L_KEY: kl_divergence_l,
                MODULE_KEYS.KL_Z_KEY: kl_divergence_z,
            },
            extra_metrics=extra_metrics_payload,
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors: dict[str, torch.Tensor],
        n_samples: int = 1,
        max_poisson_rate: float = 1e8,
    ) -> torch.Tensor:
        r"""Generate predictive samples from the posterior predictive distribution.

        Parameters
        ----------
        tensors
            Dictionary of tensors passed into forward.
        n_samples
            Number of Monte Carlo samples to draw.
        max_poisson_rate
            Maximum value for Poisson rate parameter.

        Returns
        -------
        Tensor on CPU with shape ``(n_obs, n_vars)`` if ``n_samples == 1``, else
        ``(n_obs, n_vars,)``.
        """
        from scvi.distributions import Poisson

        inference_kwargs = {"n_samples": n_samples}
        _, generative_outputs = self.forward(tensors, inference_kwargs=inference_kwargs, compute_loss=False)

        dist = generative_outputs[MODULE_KEYS.PX_KEY]
        if self.gene_likelihood == "poisson":
            dist = Poisson(torch.clamp(dist.rate, max=max_poisson_rate))

        samples = dist.sample()
        samples = torch.permute(samples, (1, 2, 0)) if n_samples > 1 else samples

        return samples.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(
        self,
        tensors: dict[str, torch.Tensor],
        n_mc_samples: int,
        return_mean: bool = False,
        n_mc_samples_per_pass: int = 1,
    ):
        """Compute the marginal log-likelihood of the data under the model."""
        from torch import logsumexp
        from torch.distributions import Normal

        library_size_index = tensors.get(LIBRARY_SIZE_KEY, tensors[REGISTRY_KEYS.BATCH_KEY])

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            warnings.warn(
                "Number of chunks is larger than the total number of samples, setting it to the number of samples",
                RuntimeWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            inference_outputs, _, losses = self.forward(tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass})
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            ql = inference_outputs[MODULE_KEYS.QL_KEY]
            z = inference_outputs[MODULE_KEYS.Z_KEY]
            library = inference_outputs[MODULE_KEYS.LIBRARY_KEY]

            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            p_z = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(library_size_index)

                p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x
            if n_mc_samples_per_pass == 1:
                log_prob_sum = log_prob_sum.unsqueeze(0)

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl
