"""RegularizedMultimodalVI model class.

N-modality extensible model with symmetric regularized components.
Supports RNA + ATAC (and extensible to more modalities) with:
- GammaPoisson likelihood for all modalities
- Learnable hierarchical dispersion prior
- Per-modality additive background and region factors
- Three Z combination strategies: concatenation, single_encoder, weighted_mean
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

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
    DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
    DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
    DEFAULT_LIBRARY_LOG_VARS_WEIGHT,
    DEFAULT_LIBRARY_N_HIDDEN,
    DEFAULT_REGION_FACTORS_PRIOR_ALPHA,
    DEFAULT_REGION_FACTORS_PRIOR_BETA,
    DEFAULT_REGULARISE_DISPERSION,
    DEFAULT_REGULARISE_DISPERSION_PRIOR,
    DEFAULT_SCALE_ACTIVATION,
    DEFAULT_USE_BATCH_IN_DECODER,
    DEFAULT_USE_BATCH_NORM,
    DEFAULT_USE_LAYER_NORM,
    DISPERSION_KEY,
    MODALITY_SCALING_COVS_KEY,
)
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
    configuration for architecture size, additive background, and region factors.

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
    region_factors_modalities
        Modalities with per-feature region factors. Default ``["atac"]``.
    region_factors_prior_alpha
        Gamma prior alpha on region factors. Default 200 (tight prior, mean=1).
    region_factors_prior_beta
        Gamma prior beta on region factors. Default 200.
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
        region_factors_modalities: list[str] | None = None,
        region_factors_prior_alpha: float = DEFAULT_REGION_FACTORS_PRIOR_ALPHA,
        region_factors_prior_beta: float = DEFAULT_REGION_FACTORS_PRIOR_BETA,
        regularise_dispersion: bool = DEFAULT_REGULARISE_DISPERSION,
        regularise_dispersion_prior: float = DEFAULT_REGULARISE_DISPERSION_PRIOR,
        dispersion_hyper_prior_alpha: float = DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,
        dispersion_hyper_prior_beta: float = DEFAULT_DISPERSION_HYPER_PRIOR_BETA,
        additive_bg_prior_alpha: float = DEFAULT_ADDITIVE_BG_PRIOR_ALPHA,
        additive_bg_prior_beta: float = DEFAULT_ADDITIVE_BG_PRIOR_BETA,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_BATCH_NORM,
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = DEFAULT_USE_LAYER_NORM,
        **kwargs,
    ):
        super().__init__(mdata)

        # Discover modality names from the registered MuData
        if mdata is not None:
            modality_names = self._get_modality_names()
        else:
            modality_names = []

        # Set defaults based on discovered modalities
        if additive_background_modalities is None:
            additive_background_modalities = [m for m in modality_names if m == "rna"]
        if region_factors_modalities is None:
            region_factors_modalities = [m for m in modality_names if m == "atac"]

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
            "region_factors_modalities": region_factors_modalities,
            "region_factors_prior_alpha": region_factors_prior_alpha,
            "region_factors_prior_beta": region_factors_prior_beta,
            "regularise_dispersion": regularise_dispersion,
            "regularise_dispersion_prior": regularise_dispersion_prior,
            "dispersion_hyper_prior_alpha": dispersion_hyper_prior_alpha,
            "dispersion_hyper_prior_beta": dispersion_hyper_prior_beta,
            "additive_bg_prior_alpha": additive_bg_prior_alpha,
            "additive_bg_prior_beta": additive_bg_prior_beta,
            "use_batch_norm": use_batch_norm,
            "use_layer_norm": use_layer_norm,
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
        n_labels = self.summary_stats.get("n_labels", 1)

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

        # Compute per-modality library size priors (log-scale mean and variance per batch)
        library_log_means = {}
        library_log_vars = {}
        batch_indices = self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        for name in modality_names:
            reg_key = f"X_{name}"
            data = self.adata_manager.get_from_registry(reg_key)
            log_means = np.zeros(n_batch)
            log_vars = np.ones(n_batch)
            for i_batch in np.unique(batch_indices):
                idx_batch = np.squeeze(batch_indices == i_batch)
                batch_data = data[idx_batch.nonzero()[0]]
                sum_counts = batch_data.sum(axis=1)
                masked_log_sum = np.ma.log(sum_counts)
                if np.ma.is_masked(masked_log_sum):
                    logger.warning(
                        "Modality '%s' has cells with zero total counts in batch %d. "
                        "Consider filtering with scanpy.pp.filter_cells().",
                        name,
                        i_batch,
                    )
                log_counts = masked_log_sum.filled(0)
                log_means[i_batch] = np.mean(log_counts).astype(np.float32)
                log_vars[i_batch] = np.var(log_counts).astype(np.float32)
            library_log_means[name] = log_means.reshape(1, -1)
            library_log_vars[name] = log_vars.reshape(1, -1)

        n_cats_per_cov = None
        if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_cov = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key

        n_cats_per_scaling_cov = None
        if MODALITY_SCALING_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_scaling_cov = self.adata_manager.get_state_registry(MODALITY_SCALING_COVS_KEY).n_cats_per_key

        n_cats_per_ambient_cov = None
        if AMBIENT_COVS_KEY in self.adata_manager.data_registry:
            n_cats_per_ambient_cov = self.adata_manager.get_state_registry(AMBIENT_COVS_KEY).n_cats_per_key

        n_dispersion_cats = None
        if DISPERSION_KEY in self.adata_manager.data_registry:
            n_dispersion_cats = len(self.adata_manager.get_state_registry(DISPERSION_KEY).categorical_mapping)

        kwargs = dict(self._module_kwargs)
        # Remove keys that are passed separately
        for k in ["n_hidden", "n_latent", "n_layers", "dropout_rate"]:
            kwargs.pop(k, None)

        self.module = self._module_cls(
            modality_names=modality_names,
            n_input_per_modality=n_input_per_modality,
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=self._module_kwargs["n_hidden"],
            n_latent=self._module_kwargs["n_latent"],
            n_layers=self._module_kwargs["n_layers"],
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=self._module_kwargs["dropout_rate"],
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_cats_per_scaling_cov=n_cats_per_scaling_cov,
            n_cats_per_ambient_cov=n_cats_per_ambient_cov,
            n_dispersion_cats=n_dispersion_cats,
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
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        modality_scaling_covariate_keys: list[str] | None = None,
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
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        modality_scaling_covariate_keys
            Optional list of categorical ``.obs`` keys whose categories define
            per-feature scaling factors (cell2location-style ``y_{t,g}``).
            Registered separately from ``categorical_covariate_keys`` — these
            do NOT feed into encoder/decoder injection layers, they only
            control the per-feature multiplicative region factor.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        if modalities is None:
            modalities = {key: key for key in mdata.mod.keys()}

        # Backward compat: batch_key fans out to ambient covariates and dispersion
        if batch_key is not None and ambient_covariate_keys is None:
            ambient_covariate_keys = [batch_key]
        if batch_key is not None and dispersion_key is None:
            dispersion_key = batch_key

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
        if categorical_covariate_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    REGISTRY_KEYS.CAT_COVS_KEY,
                    categorical_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )
        if continuous_covariate_keys:
            anndata_fields.append(
                fields.MuDataNumericalJointObsField(
                    REGISTRY_KEYS.CONT_COVS_KEY,
                    continuous_covariate_keys,
                    mod_key=list(modalities.values())[0],
                )
            )

        # Modality scaling covariates (separate from encoder/decoder covariates)
        if modality_scaling_covariate_keys:
            anndata_fields.append(
                fields.MuDataCategoricalJointObsField(
                    MODALITY_SCALING_COVS_KEY,
                    modality_scaling_covariate_keys,
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
        n_latent = self.module.total_latent_dim

        all_z = []
        importances = {name: [] for name in self.module.modality_names}

        for tensors in scdl:
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
            scaling_covs = tensors.get(MODALITY_SCALING_COVS_KEY)
            if scaling_covs is not None and self.module.n_total_scaling_cats > 0:
                scaling_indicator = torch.cat(
                    [
                        one_hot(scaling_covs[:, i].long(), n_cats).float()
                        for i, n_cats in enumerate(self.module.n_cats_per_scaling_cov)
                    ],
                    dim=-1,
                )
            else:
                scaling_indicator = None

            def _make_decode_rate_fn(
                module,
                name,
                disp,
                lib,
                batch_index,
                categorical_input,
                bg,
                cont_covs,
                scaling_indicator,
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

                    if name in module.region_factors:
                        rf_transformed = torch.nn.functional.softplus(module.region_factors[name]) / 0.7
                        if scaling_indicator is not None:
                            scaling = torch.matmul(scaling_indicator, rf_transformed)
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
                    bg = sum(
                        torch.matmul(
                            one_hot(ambient_covs[:, i].long(), int(n_cats_i)).float(),
                            torch.exp(self.module.additive_background[name][i]).T,
                        )
                        for i, n_cats_i in enumerate(self.module.n_cats_per_ambient_cov)
                    )

                decode_rate = _make_decode_rate_fn(
                    self.module,
                    name,
                    disp,
                    lib,
                    batch_index,
                    categorical_input,
                    bg,
                    cont_covs,
                    scaling_indicator,
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
