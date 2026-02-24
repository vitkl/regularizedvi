"""Default hyperparameters for regularizedvi.

These defaults encode the regularizedvi modelling choices, which differ from
standard scVI defaults. They reflect cell2location/cell2fate modelling
principles adapted to the scVI framework.
"""

# --- Likelihood and dispersion ---
DEFAULT_GENE_LIKELIHOOD = "nb"
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
DEFAULT_LIKELIHOOD_DISTRIBUTION = "gamma_poisson"

# --- Hierarchical dispersion hyper-prior (cell2location-style) ---
# Gamma(alpha, beta) hyper-prior on the Exponential rate parameter.
# Mean = alpha/beta = 9/3 = 3.0, concentrated around the default rate.
DEFAULT_DISPERSION_HYPER_PRIOR_ALPHA = 9.0
DEFAULT_DISPERSION_HYPER_PRIOR_BETA = 3.0

# --- Network architecture ---
DEFAULT_USE_BATCH_NORM = "none"
DEFAULT_USE_LAYER_NORM = "both"
DEFAULT_DROPOUT_ON_INPUT = True

# --- Recommended training defaults (documented, not enforced) ---
# These are not enforced in code but recommended in docs/tutorial:
# n_hidden=512, n_latent=128, n_layers=1
# train_size=1.0, max_epochs=2000, batch_size=1024
