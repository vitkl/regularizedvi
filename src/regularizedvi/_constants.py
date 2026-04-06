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
DEFAULT_REGULARISE_DISPERSION_PRIOR = 3.0

# --- Hierarchical dispersion hyper-prior (cell2location-style) ---
# Gamma(alpha, beta) hyper-prior on the Exponential rate parameter.
# Mean = alpha/beta = 9/3 = 3.0, concentrated around the default rate.
DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA = 9.0
DEFAULT_DISPERSION_HYPER_PRIOR_BETA = 3.0

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
DECODER_TYPE_DEFAULTS = {
    "expected_RNA": {
        "dispersion_hyper_prior_alpha": DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA,  # 9.0
        "dispersion_hyper_prior_beta": DEFAULT_DISPERSION_HYPER_PRIOR_BETA,  # 3.0
        "regularise_dispersion_prior": DEFAULT_REGULARISE_DISPERSION_PRIOR,  # 3.0
    },
    "burst_frequency_size": {
        # Gamma(2, 0.04) hyper-prior: E[lambda]=50, marginal median(v_std)=0.017
        # Wider than expected_RNA because technical variance spans ~3 orders of magnitude.
        "dispersion_hyper_prior_alpha": 2.0,
        "dispersion_hyper_prior_beta": 0.04,
        "regularise_dispersion_prior": 3.0,
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

# --- Auto-derived hyper-prior lambda bounds (sanity filter only) ---
# When auto-deriving dispersion_hyper_prior_beta from MoM stochastic_v,
# clamp E[lambda] = alpha/beta to reject only ridiculous values. Lambda is the
# Exponential rate for sqrt(stochastic_v), so target sqrt(v) scale = 1/lambda.
# LAMBDA_MIN=1e-4  → rejects scale > 10000 (ridiculously large technical variance)
# LAMBDA_MAX=100   → rejects scale < 0.01   (ridiculously small technical variance)
AUTO_HYPER_PRIOR_LAMBDA_MIN = 1e-4
AUTO_HYPER_PRIOR_LAMBDA_MAX = 100.0

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
