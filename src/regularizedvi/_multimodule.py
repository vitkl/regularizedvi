"""RegularizedMultimodalVAE module for multi-modal single-cell data.

N-modality extensible VAE with symmetric regularized components.
Each modality uses the same encoder/decoder architecture (RegularizedEncoder,
RegularizedDecoderSCVI) with per-modality configuration for:
- n_hidden, n_latent (architecture sizing)
- additive_background (ambient correction, default ON for RNA only)
- region_factors (per-feature bias, default ON for ATAC only)
- dispersion parameterization (gene/region, gene-batch/region-batch)
- GammaPoisson likelihood (default for all modalities)

Supports three latent combination strategies:
- "concatenation" (default): z = [z_mod1; z_mod2; ...], preserves modality-specific signal
- "single_encoder": one encoder for concatenated input
- "weighted_mean": MultiVI-style weighted mixing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Exponential, Gamma, Normal
from torch.distributions import kl_divergence as kld

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
    and per-modality flags (additive_background, region_factors).

    Parameters
    ----------
    modality_names
        Ordered list of modality names (e.g., ``["rna", "atac"]``).
    n_input_per_modality
        Dict mapping modality name to number of input features.
    n_batch
        Number of batches.
    n_labels
        Number of labels.
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
        If True, use size_factor from anndata.
    use_observed_lib_size
        If True, use observed library size (sum of counts).
    library_log_means
        Prior means for log library sizes, dict per modality.
    library_log_vars
        Prior variances for log library sizes, dict per modality.
    library_log_vars_weight
        Scale factor for library prior variance.
    library_n_hidden
        Hidden units for library encoder.
    scale_activation
        Decoder scale activation.
    use_batch_in_decoder
        If False, batch-free decoder.
    additive_background_modalities
        List of modality names that get additive ambient background.
    region_factors_modalities
        List of modality names that get per-feature region factors.
    regularise_dispersion
        Enable dispersion regularization.
    regularise_dispersion_prior
        Initialization for the Exponential rate parameter.
    likelihood_distribution
        ``"gamma_poisson"`` (default) or ``"nb"``.
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
        n_labels: int = 0,
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
        use_observed_lib_size: bool = True,
        library_log_means: dict[str, np.ndarray] | None = None,
        library_log_vars: dict[str, np.ndarray] | None = None,
        library_log_vars_weight: float | None = None,
        library_n_hidden: int = 16,
        scale_activation: str = "softplus",
        use_batch_in_decoder: bool = False,
        additive_background_modalities: list[str] | None = None,
        region_factors_modalities: list[str] | None = None,
        regularise_dispersion: bool = True,
        regularise_dispersion_prior: float = 3.0,
        likelihood_distribution: Literal["nb", "gamma_poisson"] = "gamma_poisson",
        dispersion_hyper_prior_alpha: float = 9.0,
        dispersion_hyper_prior_beta: float = 3.0,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
    ):
        from regularizedvi._components import RegularizedDecoderSCVI, RegularizedEncoder

        super().__init__()

        self.modality_names = list(modality_names)
        self.n_modalities = len(self.modality_names)
        self.n_input_per_modality = n_input_per_modality
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.log_variational = log_variational
        self.latent_mode = latent_mode
        self.modality_weights_mode = modality_weights
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        self.use_batch_in_decoder = use_batch_in_decoder
        self.regularise_dispersion = regularise_dispersion
        self.likelihood_distribution = likelihood_distribution
        self.dispersion_hyper_prior_alpha = dispersion_hyper_prior_alpha
        self.dispersion_hyper_prior_beta = dispersion_hyper_prior_beta
        self.encode_covariates = encode_covariates

        additive_background_modalities = additive_background_modalities or []
        region_factors_modalities = region_factors_modalities or []
        self.additive_background_modalities = additive_background_modalities
        self.region_factors_modalities = region_factors_modalities

        # Resolve per-modality configs
        n_hidden_dict = _resolve_per_modality(n_hidden, self.modality_names)
        n_latent_dict = _resolve_per_modality(n_latent, self.modality_names)
        n_layers_dict = _resolve_per_modality(n_layers, self.modality_names)
        dispersion_dict = _resolve_per_modality(dispersion, self.modality_names)

        self.n_hidden_dict = n_hidden_dict
        self.n_latent_dict = n_latent_dict
        self.dispersion_dict = dispersion_dict

        use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        use_layer_norm_decoder = use_layer_norm in ("decoder", "both")

        # Batch/covariate handling
        cat_list = [n_batch] + list(n_cats_per_cov or [])
        encoder_cat_list = cat_list if encode_covariates else None

        if not use_batch_in_decoder:
            decoder_cat_list = list(n_cats_per_cov or [])
        else:
            decoder_cat_list = cat_list

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
            n_in = n_input_per_modality[name] + n_continuous_cov * encode_covariates
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
            total_input = sum(n_input_per_modality.values()) + n_continuous_cov * encode_covariates
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
                **_extra_decoder_kwargs,
            )

        # ---- Per-modality library encoders (variational, low-capacity) ----
        self.l_encoders = nn.ModuleDict()
        if not self.use_observed_lib_size:
            for name in self.modality_names:
                n_in = n_input_per_modality[name] + n_continuous_cov * encode_covariates
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
        if not self.use_observed_lib_size:
            library_log_means = library_log_means or {}
            library_log_vars = library_log_vars or {}
            for name in self.modality_names:
                if name in library_log_means and name in library_log_vars:
                    means = torch.from_numpy(library_log_means[name]).float()
                    vars_ = torch.from_numpy(library_log_vars[name]).float()
                    if library_log_vars_weight is not None:
                        vars_ = vars_ * library_log_vars_weight
                    self.register_buffer(f"library_log_means_{name}", means)
                    self.register_buffer(f"library_log_vars_{name}", vars_)

        # ---- Per-modality dispersion parameters ----
        self.px_r = nn.ParameterDict()
        for name in self.modality_names:
            n_feat = n_input_per_modality[name]
            disp = dispersion_dict[name]
            if disp in ("gene", "region"):
                self.px_r[name] = nn.Parameter(torch.randn(n_feat))
            elif disp in ("gene-batch", "region-batch"):
                self.px_r[name] = nn.Parameter(torch.randn(n_feat, n_batch))
            elif disp in ("gene-label", "region-label"):
                self.px_r[name] = nn.Parameter(torch.randn(n_feat, n_labels))

        # ---- Learnable dispersion prior rates (per modality) ----
        if self.regularise_dispersion:
            _raw_init = torch.log(torch.expm1(torch.tensor(regularise_dispersion_prior)))
            self.dispersion_prior_rate_raw = nn.ParameterDict()
            for name in self.modality_names:
                disp = dispersion_dict[name]
                if disp in ("gene-batch", "region-batch"):
                    self.dispersion_prior_rate_raw[name] = nn.Parameter(_raw_init.expand(n_batch).clone())
                elif disp in ("gene-label", "region-label"):
                    self.dispersion_prior_rate_raw[name] = nn.Parameter(_raw_init.expand(n_labels).clone())
                else:
                    self.dispersion_prior_rate_raw[name] = nn.Parameter(_raw_init.unsqueeze(0).clone())

        # ---- Additive background (per selected modality) ----
        self.additive_background = nn.ParameterDict()
        for name in additive_background_modalities:
            n_feat = n_input_per_modality[name]
            self.additive_background[name] = nn.Parameter(torch.randn(n_feat, n_batch))

        # ---- Region factors (per selected modality) ----
        self.region_factors = nn.ParameterDict()
        for name in region_factors_modalities:
            n_feat = n_input_per_modality[name]
            self.region_factors[name] = nn.Parameter(torch.zeros(n_feat))

    def _get_inference_input(
        self,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for inference."""
        result = {
            "batch_index": tensors[REGISTRY_KEYS.BATCH_KEY],
            "cont_covs": tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            "cat_covs": tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
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
            "cont_covs": tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None),
            "cat_covs": tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None),
        }

    @auto_move_data
    def inference(
        self,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        n_samples: int = 1,
        **modality_inputs,
    ) -> dict[str, torch.Tensor | Distribution | None]:
        """Per-modality encoding and Z combination.

        Parameters
        ----------
        batch_index
            Batch indices.
        cont_covs
            Continuous covariates.
        cat_covs
            Categorical covariates.
        n_samples
            Number of samples from posterior.
        **modality_inputs
            Per-modality inputs as x_{modality_name} tensors.
        """
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

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
                if cont_covs is not None and self.encode_covariates:
                    x = torch.cat([x, cont_covs], dim=-1)
                inputs.append(x)
            joint_input = torch.cat(inputs, dim=-1)
            qz, z = self.joint_encoder(joint_input, batch_index, *categorical_input)
            # Store as single "joint" modality for KL computation
            qz_per_modality["_joint"] = qz
            z_per_modality["_joint"] = z
        else:
            for name in self.modality_names:
                x = modality_inputs.get(f"x_{name}")
                if x is None:
                    continue
                x_ = torch.log1p(x) if self.log_variational else x
                if cont_covs is not None and self.encode_covariates:
                    x_ = torch.cat([x_, cont_covs], dim=-1)
                qz, z = self.encoders[name](x_, batch_index, *categorical_input)
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

        # Per-modality library encoding
        library = {}
        ql_per_modality = {}
        for name in self.modality_names:
            x = modality_inputs.get(f"x_{name}")
            if x is None:
                continue
            if self.use_observed_lib_size:
                library[name] = torch.log(x.sum(1)).unsqueeze(1)
            else:
                x_ = torch.log1p(x) if self.log_variational else x
                if cont_covs is not None and self.encode_covariates:
                    x_ = torch.cat([x_, cont_covs], dim=-1)
                ql, lib = self.l_encoders[name](x_, batch_index, *categorical_input)
                library[name] = lib
                ql_per_modality[name] = ql

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
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
    ) -> dict[str, dict]:
        """Per-modality decoding from shared Z.

        Returns dict with keys per modality, each containing the distribution.
        """
        from scvi.distributions import NegativeBinomial
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

        px_dict = {}
        for name in self.modality_names:
            if name not in library:
                continue

            disp = self.dispersion_dict[name]
            lib = library[name]

            # Compute additive background
            bg = None
            if name in self.additive_background:
                bg = torch.matmul(
                    one_hot(batch_index.squeeze(-1), self.n_batch).float(),
                    torch.exp(self.additive_background[name]).T,
                )

            # Decode
            if self.use_batch_in_decoder:
                px_scale, px_r_cell, px_rate, px_dropout = self.decoders[name](
                    disp,
                    decoder_input,
                    lib,
                    batch_index,
                    *categorical_input,
                    additive_background=bg,
                )
            else:
                px_scale, px_r_cell, px_rate, px_dropout = self.decoders[name](
                    disp,
                    decoder_input,
                    lib,
                    *categorical_input,
                    additive_background=bg,
                )

            # Region factors
            if name in self.region_factors:
                px_rate = px_rate * torch.sigmoid(self.region_factors[name])

            # Resolve dispersion
            if disp in ("gene-batch", "region-batch"):
                px_r = torch.nn.functional.linear(
                    one_hot(batch_index.squeeze(-1), self.n_batch).float(),
                    self.px_r[name],
                )
            elif disp in ("gene-label", "region-label"):
                px_r = torch.nn.functional.linear(
                    one_hot(batch_index.squeeze(-1), self.n_labels).float(),
                    self.px_r[name],
                )
            elif disp in ("gene-cell", "region-cell"):
                px_r = px_r_cell
            else:
                px_r = self.px_r[name]

            px_r = torch.exp(px_r)

            # Build distribution
            if self.likelihood_distribution == "gamma_poisson":
                px = GammaPoissonWithScale(concentration=px_r, rate=px_r / px_rate, scale=px_scale)
            else:
                px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)

            px_dict[name] = px

        # Prior on z
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        # Library priors
        pl_dict = {}
        if not self.use_observed_lib_size:
            for name in self.modality_names:
                if name not in library:
                    continue
                means_buf = getattr(self, f"library_log_means_{name}", None)
                vars_buf = getattr(self, f"library_log_vars_{name}", None)
                if means_buf is not None and vars_buf is not None:
                    local_means = torch.nn.functional.linear(
                        one_hot(batch_index.squeeze(-1), self.n_batch).float(),
                        means_buf,
                    )
                    local_vars = torch.nn.functional.linear(
                        one_hot(batch_index.squeeze(-1), self.n_batch).float(),
                        vars_buf,
                    )
                    pl_dict[name] = Normal(local_means, local_vars.sqrt())

        return {
            "px": px_dict,
            "pz": pz,
            "pl": pl_dict,
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
            recon_loss = recon_loss + rl

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
                kl_z = kl_z + kl_mod

        # ---- KL divergence on library ----
        kl_l = torch.zeros_like(recon_loss)
        if not self.use_observed_lib_size:
            ql_per_modality = inference_outputs.get("ql_per_modality", {})
            for name in self.modality_names:
                if name not in ql_per_modality or name not in pl_dict:
                    continue
                kl_lib = kld(ql_per_modality[name], pl_dict[name]).sum(dim=1)
                if name in masks:
                    kl_lib = kl_lib * masks[name].float()
                kl_l = kl_l + kl_lib

        # ---- Weighted loss ----
        loss = torch.mean(recon_loss + kl_weight * kl_z + kl_l)

        # ---- Hierarchical dispersion priors ----
        if self.regularise_dispersion:
            n_obs = recon_loss.shape[0]
            dispersion_penalty = torch.tensor(0.0, device=recon_loss.device)

            for name in self.modality_names:
                if name not in self.px_r:
                    continue
                # Level 1: learned rate with Gamma hyper-prior
                learned_rate = torch.nn.functional.softplus(self.dispersion_prior_rate_raw[name])
                neg_log_hyper = (
                    -Gamma(self.dispersion_hyper_prior_alpha, self.dispersion_hyper_prior_beta)
                    .log_prob(learned_rate)
                    .sum()
                )

                # Level 2: Exponential prior on dispersion
                disp = self.dispersion_dict[name]
                if disp in ("gene-batch", "region-batch"):
                    rate = learned_rate.unsqueeze(0).expand_as(self.px_r[name])
                elif disp in ("gene-label", "region-label"):
                    rate = learned_rate.unsqueeze(0).expand_as(self.px_r[name])
                else:
                    rate = learned_rate.expand_as(self.px_r[name])

                if self.likelihood_distribution == "gamma_poisson":
                    px_r_transformed = torch.exp(-self.px_r[name]).pow(0.5)
                else:
                    px_r_transformed = torch.exp(self.px_r[name]).pow(0.5)

                neg_log_prior = -Exponential(rate).log_prob(px_r_transformed).sum()
                dispersion_penalty = dispersion_penalty + neg_log_prior + neg_log_hyper

            loss = loss + dispersion_penalty / n_obs

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local={
                "kl_divergence_z": kl_z,
                "kl_divergence_l": kl_l,
            },
        )
