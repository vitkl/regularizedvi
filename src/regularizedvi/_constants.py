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
DEFAULT_LIBRARY_LOG_VARS_WEIGHT = 0.05
DEFAULT_LIBRARY_N_HIDDEN = 16

# --- Decoder ---
DEFAULT_SCALE_ACTIVATION = "softplus"
DEFAULT_USE_ADDITIVE_BACKGROUND = True
DEFAULT_USE_BATCH_IN_DECODER = False

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
DEFAULT_REGULARISE_BACKGROUND = True

# --- Region factors (cell2location-style detection_tech_gene) ---
# Gamma(alpha, beta) prior on softplus(param)/0.7; mean = alpha/beta = 1.0.
# High alpha (200) gives tight prior: factors stay near 1 unless data demands otherwise.
DEFAULT_REGION_FACTORS_PRIOR_ALPHA = 200.0
DEFAULT_REGION_FACTORS_PRIOR_BETA = 200.0

# Registry key for modality scaling covariates (separate from encoder/decoder covariates)
MODALITY_SCALING_COVS_KEY = "modality_scaling_covs"

# Registry key for ambient covariates (controls additive background, separate from batch_key)
AMBIENT_COVS_KEY = "ambient_covs"

# Registry key for dispersion covariate (controls per-group px_r, separate from batch_key)
DISPERSION_KEY = "dispersion_key"

# Registry key for library size covariate (controls N(mu_s, sigma_s) prior, separate from batch_key)
LIBRARY_SIZE_KEY = "library_size_key"

# --- Network architecture ---
DEFAULT_USE_BATCH_NORM = "none"
DEFAULT_USE_LAYER_NORM = "both"
DEFAULT_DROPOUT_ON_INPUT = True

# --- Training metrics ---
DEFAULT_COMPUTE_PEARSON = False

# --- Recommended training defaults (documented, not enforced) ---
# These are not enforced in code but recommended in docs/tutorial:
# n_hidden=512, n_latent=128, n_layers=1
# train_size=1.0, max_epochs=2000, batch_size=1024
