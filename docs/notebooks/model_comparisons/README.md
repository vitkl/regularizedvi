# Model comparisons (bone marrow dataset)

Systematic comparison of regularizedvi vs standard scVI on the NeurIPS 2021 bone marrow multiome dataset (46,534 cells, 5 batches, 4 sites, 8 donors).

## Key findings

- **Large scVI overfits at 2000 epochs**: Standard scVI with large architecture (512/128/1) shows double descent and overfitting when trained for 2000 epochs. The same model with default training settings (~400 epochs, batch_size=128) performs better.
- **regularizedvi works across model sizes**: GammaPoisson mode produces good integration at all tested architectures (128/10, 256/30, 512/128) without hyperparameter tuning.
- **All regularizedvi models still improving at 2000 epochs**: Training curves suggest regularizedvi benefits from longer training, motivating the early stopping experiment.
- **Dispersion (theta)**: Unconstrained scVI learns theta ~1-2 (high overdispersion), while regularizedvi GammaPoisson learns theta ~53 (closer to Poisson). The prior prevents NB collapse during training.
- **Batch-free decoder**: Separating constrained additive (ambient RNA per batch) from flexible multiplicative (categorical covariates for donor/protocol) correction is the key architectural benefit.

## Notebook inventory

| Notebook | Model | Architecture | Epochs | Likelihood | Notes |
|----------|-------|-------------|--------|------------|-------|
| `bone_marrow_regularizedvi_nb.ipynb` | regularizedvi | 512/128/1 | 2000 | NB | Original NB default |
| `bone_marrow_gamma_poisson.ipynb` | regularizedvi | 512/128/1 | 2000 | GammaPoisson | Now the default |
| `bone_marrow_gamma_poisson_1375ep.ipynb` | regularizedvi | 512/128/1 | 1375 | GammaPoisson | Lueken-adjusted epochs |
| `bone_marrow_gamma_poisson_early_stopping.ipynb` | regularizedvi | 512/128/1 | ≤4000 | GammaPoisson | Early stopping + checkpoints |
| `bone_marrow_gamma_poisson_small.ipynb` | regularizedvi | 128/10/1 | 2000 | GammaPoisson | Small architecture |
| `bone_marrow_gamma_poisson_medium.ipynb` | regularizedvi | 256/30/1 | 2000 | GammaPoisson | Medium architecture |
| `bone_marrow_gamma_poisson_small_default_train.ipynb` | regularizedvi | 128/10/1 | default | GammaPoisson | Small + default scVI training |
| `bone_marrow_scvi_default.ipynb` | scVI | 128/30/1 | default | ZINB | scVI defaults |
| `bone_marrow_scvi_custom.ipynb` | scVI | 512/128/1 | 2000 | NB | Large scVI, 2000 epochs |
| `bone_marrow_scvi_custom_zinb.ipynb` | scVI | 512/128/1 | 2000 | ZINB | Large scVI, ZINB |
| `bone_marrow_scvi_custom_default_train.ipynb` | scVI | 512/128/1 | default | NB | Large scVI, default training |
| `bone_marrow_scvi_lueken.ipynb` | scVI | 128/30/2 | 1375 | ZINB | Lueken lung atlas settings |
| `model_comparison_theta.ipynb` | — | — | — | — | Theta distribution analysis |
| `distribution_comparison.ipynb` | — | — | — | — | NB vs GammaPoisson comparison |

Architecture notation: `n_hidden/n_latent/n_layers`. "default" training = scVI defaults (~400 epochs, batch_size=128).

Epochs for Lueken-adjusted training: `round(400 * (20000/N) * (batch_size/128))` = 1375 for N=46534, batch_size=1024.
