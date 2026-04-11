"""Default hyperparameters for regularizedvi.

These defaults encode the regularizedvi modelling choices, which differ from
standard scVI defaults. They reflect cell2location/cell2fate modelling
principles adapted to the scVI framework.
"""

# --- Likelihood and dispersion ---
DEFAULT_GENE_LIKELIHOOD = "gamma_poisson"
DEFAULT_DISPERSION = "gene-batch"

# --- Library size ---
DEFAULT_USE_OBSERVED_LIB_SIZE = False
DEFAULT_LIBRARY_LOG_VARS_WEIGHT = 0.5
DEFAULT_LIBRARY_N_HIDDEN = 16

# --- Decoder ---
DEFAULT_SCALE_ACTIVATION = "softplus"
DEFAULT_USE_ADDITIVE_BACKGROUND = True
DEFAULT_USE_BATCH_IN_DECODER = False
DEFAULT_DECODER_WEIGHT_L2 = 0.0

# --- Dispersion regularisation (containment prior) ---
DEFAULT_REGULARISE_DISPERSION = True

# --- Hierarchical dispersion hyper-prior (cell2location-style) ---
# Two-level prior: MoM ~ Exp(lambda); lambda ~ Gamma(alpha, beta).
# Parameterised by (alpha, mean) where `mean` is the prior mean of the MoM
# estimate itself (= 1/lambda), NOT the mean of the Gamma on lambda.
# Conversion: beta = alpha * mean (NOT alpha / mean) — see the docstring of
# resolve_dispersion_hyper_prior_params in _dispersion_init.py for the full
# Semantics block. Default mean = 1/3 encodes E[1/sqrt(theta)] ≈ 0.333 →
# theta ≈ 9 at unit mu, preserving the old Gamma(9, 3) behaviour numerically.
DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA = 9.0
DEFAULT_DISPERSION_HYPER_PRIOR_MEAN = 1.0 / 3.0

# --- Additive background prior (cell2location-style s_g_gene_add) ---
# Gamma(alpha, beta) prior on exp(additive_background); mean = alpha/beta = 0.01.
# Pushes ambient/background contribution to be small relative to biological signal.
DEFAULT_ADDITIVE_BG_PRIOR_ALPHA = 1.0
DEFAULT_ADDITIVE_BG_PRIOR_BETA = 100.0
DEFAULT_REGULARISE_BACKGROUND = False

# --- Feature scaling (cell2location-style detection_tech_gene) ---
# Gamma(alpha, beta) prior on softplus(param)/0.7; mean = alpha/beta = 1.0.
# High alpha (200) gives tight prior: factors stay near 1 unless data demands otherwise.
DEFAULT_FEATURE_SCALING_PRIOR_ALPHA = 200.0
DEFAULT_FEATURE_SCALING_PRIOR_BETA = 200.0
DEFAULT_USE_FEATURE_SCALING = True

# --- Learnable per-modality scaling (global scale factor on size factors) ---
DEFAULT_MODALITY_SCALE_PRIOR_CONCENTRATION = 5.0

# Registry key for feature scaling covariates (separate from encoder/decoder covariates)
FEATURE_SCALING_COVS_KEY = "feature_scaling_covs"

# Registry key for ambient covariates (controls additive background, separate from batch_key)
AMBIENT_COVS_KEY = "ambient_covs"

# Registry key for dispersion covariate (controls per-group px_r, separate from batch_key)
DISPERSION_KEY = "dispersion_cov"

# Registry key for library size covariate (controls N(mu_s, sigma_s) prior, separate from batch_key)
LIBRARY_SIZE_KEY = "library_size_cov"

# Registry key for encoder covariates (controls what categoricals the encoder sees)
ENCODER_COVS_KEY = "encoder_covs"

# --- Decoder type switching ---
DECODER_TYPES = ("expected_RNA", "Kon_Koff", "burst_frequency_size", "probability")
DEFAULT_DECODER_TYPE = "expected_RNA"
DEFAULT_BURST_SIZE_INTERCEPT = 1.0

# --- Per-decoder-type default hyperparameters ---
# Auto-applied when user doesn't override. Different decoder types need different priors.
# Preserves old numerical behaviour for inverse_sqrt expected_RNA; burst default
# updated to match expected_RNA's encoding of theta=9 at unit mu (see plan Item 4
# for derivation).
DECODER_TYPE_DEFAULTS = {
    "expected_RNA": {
        # inverse_sqrt path; preserves old Gamma(9,3) → E[lambda]=3 behaviour:
        # mean=1/3 → β=9·(1/3)=3, λ_init=3, px_r_mu_init=log(9)=+2.197
        "dispersion_hyper_prior_alpha": 9.0,
        "dispersion_hyper_prior_mean": 1.0 / 3.0,
    },
    "burst_frequency_size": {
        # Matches expected_RNA's encoding of theta=9 at reference mu=1.
        # Both decoders share the SAME excess_technical decomposition in
        # _dispersion_init.py:108-111 and 634-638:
        #   NB:    theta_g = mu² / excess_technical
        #   burst: stochastic_v² = excess_technical, stochastic_v_scale = sqrt(excess_technical)
        # So at mu=1: theta=9 ⟺ excess_technical = 1/9 ⟺ stochastic_v_scale = 1/3.
        # Setting mean = 1/3 here gives β = α · mean = 2/3 ≈ 0.667 (NOT the old
        # 0.04). The old default 0.04 corresponded to mean=0.02 which encoded
        # ~280× tighter variance — a latent bug, since the burst path was always
        # used with data-driven init in practice and the fallback was rarely hit.
        "dispersion_hyper_prior_alpha": 2.0,
        "dispersion_hyper_prior_mean": 1.0 / 3.0,
    },
}

# --- Data-driven init routing ---
# Decoder types that initialise decoder params from data (MoM / variance decomposition).
# For modalities using these decoder types, user-supplied dispersion hyper-prior
# values are IGNORED and replaced with MoM-derived values (with a log warning).
DATA_INIT_DECODER_TYPES = frozenset({"burst_frequency_size"})

# dispersion_init modes that actually run MoM / variance-decomposition and produce
# auto-derived hyper-prior suggestions.
DATA_DRIVEN_DISPERSION_INIT = frozenset({"data", "variance_burst_size"})

# --- Network architecture ---
DEFAULT_USE_BATCH_NORM = "none"
DEFAULT_USE_LAYER_NORM = "both"
DEFAULT_DROPOUT_ON_INPUT = True

# --- Active dimensions tracking ---
DEFAULT_ACTIVE_DIM_KL_THRESHOLD = 0.01
DEFAULT_ACTIVE_DIM_Z_GAMMA_THRESHOLD = 0.1

# --- Training metrics ---
DEFAULT_COMPUTE_PEARSON = False

# --- Early stopping ---
DEFAULT_EARLY_STOPPING_MIN_DELTA_PER_FEATURE = 0.0002

# --- Recommended training defaults (documented, not enforced) ---
# These are not enforced in code but recommended in docs/tutorial:
# n_hidden=512, n_latent=128, n_layers=1
# train_size=1.0, max_epochs=2000, batch_size=1024
