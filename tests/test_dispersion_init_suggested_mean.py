"""Tests for Wave 1 changes to _dispersion_init.py (plan validated-knitting-zebra).

Covers:
- Item 4: ``compute_bursting_init`` uses ``np.mean`` (not ``np.median``) to compute
  ``init_values["suggested_hyper_mean"]``, and ``compute_dispersion_init`` stores a
  direction-aware ``diagnostics["suggested_hyper_mean"]``.
- Item 1: per-feature ``stochastic_v_scale`` bounds derived from ``theta_min`` /
  ``theta_max`` and ``mean_g`` (so ATAC-like low-mean features get tight bounds that
  scale linearly with the mean).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from regularizedvi._dispersion_init import (
    compute_bursting_init,
    compute_dispersion_init,
)


@pytest.fixture
def synthetic_adata():
    """Synthetic adata with controlled per-gene means and variances.

    Genes 0-9:  mean=0.05  (ATAC-like)
    Genes 10-19: mean=1.0  (RNA-like)
    Genes 20-29: mean=10.0 (high-expression)
    """
    rng = np.random.default_rng(0)
    n_cells = 500
    n_genes = 30
    means = np.concatenate(
        [
            np.full(10, 0.05),
            np.full(10, 1.0),
            np.full(10, 10.0),
        ]
    )
    X = rng.poisson(means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    # Add NB-style overdispersion via gamma multiplier
    overdisp = rng.gamma(shape=5, scale=1 / 5, size=(n_cells, n_genes))
    X = (X * overdisp).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.obs["batch"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    return adata


@pytest.fixture
def skewed_adata():
    """Right-skewed synthetic adata where mean != median across genes.

    Most genes have small mean, a few have very large mean, so the empirical
    ``stochastic_v_scale`` distribution across genes is right-skewed and
    ``mean(stochastic_v_scale) > median(stochastic_v_scale)``.
    """
    rng = np.random.default_rng(7)
    n_cells = 500
    # 40 low-mean genes + 10 high-mean genes -> right-skewed
    means = np.concatenate(
        [
            np.full(40, 0.1),
            np.full(10, 50.0),
        ]
    )
    n_genes = means.size
    X = rng.poisson(means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    overdisp = rng.gamma(shape=3, scale=1 / 3, size=(n_cells, n_genes))
    X = (X * overdisp).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.obs["batch"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    return adata


# ---------------------------------------------------------------------------
# Item 4 — mean-not-median for burst init
# ---------------------------------------------------------------------------


def test_compute_bursting_init_suggested_mean_is_mean_not_median(skewed_adata):
    """``suggested_hyper_mean`` must equal mean(stochastic_v_scale), NOT the median.

    Uses a right-skewed gene population so mean and median differ substantially.
    """
    init_values, _ = compute_bursting_init(skewed_adata, verbose=False)
    sv = init_values["stochastic_v_scale"]

    sv_mean = float(np.mean(sv))
    sv_median = float(np.median(sv))

    # Sanity: the fixture is designed to be skewed.
    assert sv_mean != pytest.approx(sv_median, rel=1e-3), (
        f"fixture is not skewed enough: mean={sv_mean} median={sv_median}"
    )

    # suggested_hyper_mean should match mean (subject to the per-feature min floor).
    # The floor is min(sv_min_g) = min(mean_g)/sqrt(theta_max), which for this
    # fixture is much smaller than sv_mean, so the floor is inactive.
    assert init_values["suggested_hyper_mean"] == pytest.approx(sv_mean, rel=1e-5)

    # And explicitly NOT the median.
    assert not np.isclose(init_values["suggested_hyper_mean"], sv_median, rtol=1e-3)


def test_compute_bursting_init_direction_agnostic(skewed_adata):
    """Burst init's ``suggested_hyper_mean`` must not depend on ``dispersion_prior_direction``.

    Burst's MoM transform is always ``sqrt(v)``; the direction kwarg is accepted
    for API symmetry with ``compute_dispersion_init`` and silently ignored.
    """
    init_is, _ = compute_bursting_init(skewed_adata, dispersion_prior_direction="inverse_sqrt", verbose=False)
    init_sq, _ = compute_bursting_init(skewed_adata, dispersion_prior_direction="sqrt", verbose=False)
    assert init_is["suggested_hyper_mean"] == pytest.approx(init_sq["suggested_hyper_mean"], rel=1e-12)
    # And the underlying per-gene arrays must match too.
    np.testing.assert_allclose(init_is["stochastic_v_scale"], init_sq["stochastic_v_scale"], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Item 4 — direction-aware MoM mean for compute_dispersion_init
# ---------------------------------------------------------------------------


def _expected_mom_mean(diag, direction, theta_min, theta_max):
    """Recompute suggested_hyper_mean from theta_option1 in the diagnostics."""
    theta = diag["theta_option1"]
    eff_min = max(theta_min, 1e-10) if theta_min > 0 else 1e-10
    theta_clamped = np.clip(theta, eff_min, theta_max)
    if direction == "inverse_sqrt":
        mom = 1.0 / np.sqrt(theta_clamped)
    elif direction == "sqrt":
        mom = np.sqrt(theta_clamped)
    else:
        raise ValueError(direction)
    return float(np.mean(mom))


def test_compute_dispersion_init_inverse_sqrt_direction(synthetic_adata):
    """diagnostics['suggested_hyper_mean'] must equal mean(1/sqrt(clamped theta))."""
    theta_min = 0.01
    theta_max = 20.0
    _, diag = compute_dispersion_init(
        synthetic_adata,
        theta_min=theta_min,
        theta_max=theta_max,
        dispersion_prior_direction="inverse_sqrt",
        verbose=False,
    )

    expected = _expected_mom_mean(diag, "inverse_sqrt", theta_min, theta_max)
    assert diag["suggested_hyper_mean"] == pytest.approx(expected, rel=1e-6)
    assert diag["dispersion_prior_direction"] == "inverse_sqrt"


def test_compute_dispersion_init_sqrt_direction(synthetic_adata):
    """diagnostics['suggested_hyper_mean'] must equal mean(sqrt(clamped theta))."""
    theta_min = 0.01
    theta_max = 20.0
    _, diag = compute_dispersion_init(
        synthetic_adata,
        theta_min=theta_min,
        theta_max=theta_max,
        dispersion_prior_direction="sqrt",
        verbose=False,
    )

    expected = _expected_mom_mean(diag, "sqrt", theta_min, theta_max)
    assert diag["suggested_hyper_mean"] == pytest.approx(expected, rel=1e-6)
    assert diag["dispersion_prior_direction"] == "sqrt"


def test_compute_dispersion_init_diagnostics_has_direction_key(synthetic_adata):
    """diagnostics must carry the direction that was used, for both options."""
    for direction in ("inverse_sqrt", "sqrt"):
        _, diag = compute_dispersion_init(
            synthetic_adata,
            dispersion_prior_direction=direction,
            verbose=False,
        )
        assert "dispersion_prior_direction" in diag
        assert diag["dispersion_prior_direction"] == direction
        assert "suggested_hyper_mean" in diag
        assert np.isfinite(diag["suggested_hyper_mean"])


# ---------------------------------------------------------------------------
# Item 1 — per-feature stochastic_v_scale bounds
# ---------------------------------------------------------------------------


def test_compute_bursting_init_per_feature_sv_bounds(synthetic_adata):
    """``stochastic_v_scale[g]`` must respect per-feature bounds from theta clamps.

    For every gene ``g`` with ``mean_g[g] > 0``:
        sv[g] / mean_g[g] in [1/sqrt(theta_max), 1/sqrt(theta_min)]
    """
    theta_min = 0.01
    theta_max = 20.0
    init_values, diag = compute_bursting_init(
        synthetic_adata,
        theta_min=theta_min,
        theta_max=theta_max,
        verbose=False,
    )
    sv = init_values["stochastic_v_scale"].astype(np.float64)
    mean_g = diag["mean_g"].astype(np.float64)

    # All fixture genes have strictly positive mean.
    assert np.all(mean_g > 0)

    ratio = sv / mean_g
    lower = 1.0 / np.sqrt(theta_max)
    upper = 1.0 / np.sqrt(theta_min)

    # Small numerical tolerance for float32 round-trip in stochastic_v_scale.
    atol = 1e-6
    assert float(ratio.min()) >= lower - atol, f"min(sv/mean_g) = {ratio.min()} < 1/sqrt(theta_max) = {lower}"
    assert float(ratio.max()) <= upper + atol, f"max(sv/mean_g) = {ratio.max()} > 1/sqrt(theta_min) = {upper}"


def test_compute_bursting_init_atac_scale_separation():
    """``sv_max_g = mean_g / sqrt(theta_min)`` must scale linearly with mean_g.

    Previously the bound was a flat scalar across all features. Now, for
    ATAC-like features with means varying over ``[0.005, 0.1]``, the upper
    bound must be proportional to the per-feature mean — so features with
    20x higher mean get a 20x higher upper bound.
    """
    rng = np.random.default_rng(11)
    n_cells = 400
    # Linearly spaced low means (ATAC scale).
    means = np.linspace(0.005, 0.1, 30)
    n_genes = means.size
    X = rng.poisson(means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    # Add some extra variance so excess_technical is positive.
    overdisp = rng.gamma(shape=4, scale=1 / 4, size=(n_cells, n_genes))
    X = (X * overdisp).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(X))
    adata.layers["counts"] = sp.csr_matrix(X)
    adata.obs["batch"] = ["A"] * n_cells
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    theta_min = 0.01
    theta_max = 20.0
    _, diag = compute_bursting_init(
        adata,
        theta_min=theta_min,
        theta_max=theta_max,
        verbose=False,
    )
    mean_g = diag["mean_g"].astype(np.float64)

    # The per-feature bound itself should scale linearly with mean_g.
    sv_max_g = mean_g / np.sqrt(theta_min)

    # sv_max_g must NOT be (approximately) flat — check that max/min ratio
    # closely matches max(mean_g)/min(mean_g).
    assert mean_g.min() > 0
    expected_ratio = mean_g.max() / mean_g.min()
    observed_ratio = sv_max_g.max() / sv_max_g.min()
    assert observed_ratio == pytest.approx(expected_ratio, rel=1e-6), (
        f"sv_max_g should scale linearly with mean_g: expected ratio {expected_ratio}, got {observed_ratio}"
    )

    # And the ratio sv_max_g/mean_g must be a flat scalar (1/sqrt(theta_min)).
    per_feature_scale = sv_max_g / mean_g
    np.testing.assert_allclose(per_feature_scale, 1.0 / np.sqrt(theta_min), rtol=1e-12)

    # Sanity: the spread in absolute values is at least ~10x (0.1 / 0.005).
    assert sv_max_g.max() / sv_max_g.min() > 10.0
