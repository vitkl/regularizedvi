"""Unit tests for resolve_dispersion_hyper_prior_params (Item 4).

These tests lock in the single-source-of-truth math for the two-level
dispersion containment prior:

    MoM ~ Exp(lambda)         [containment prior, lambda is the RATE]
    lambda ~ Gamma(alpha, beta)   [hyper-prior, PyTorch rate form]

Under the plan's semantics, `dispersion_hyper_prior_mean` is the prior mean
of the MoM estimate itself (i.e. mean = E[MoM] = 1/lambda), so

    lambda_init = 1 / mean
    beta        = alpha * mean     (NOT alpha / mean)

See [docs for plan `validated-knitting-zebra.md`, Item 4].
"""

from __future__ import annotations

import math

import pytest

from regularizedvi._dispersion_init import (
    DispersionHyperPriorParams,
    resolve_dispersion_hyper_prior_params,
)


class TestResolveExpectedRNAInverseSqrt:
    """expected_RNA decoder, direction='inverse_sqrt' (legacy default path)."""

    def test_old_default(self):
        """(mean=1/3, alpha=9, inverse_sqrt, dispersion) reproduces legacy defaults.

        Legacy defaults were alpha=9, beta=3, lambda_init=3, px_r_mu_init=log(9).
        The new resolve function must be numerically identical.
        """
        hp = resolve_dispersion_hyper_prior_params(
            mean=1.0 / 3.0,
            alpha=9.0,
            direction="inverse_sqrt",
            init_mode="dispersion",
        )
        assert hp.alpha == pytest.approx(9.0, rel=1e-12)
        assert hp.beta == pytest.approx(3.0, rel=1e-12)
        assert hp.lambda_init == pytest.approx(3.0, rel=1e-12)
        assert hp.px_r_mu_init == pytest.approx(math.log(9.0), rel=1e-12)

    def test_inverse_sqrt_flips_sign_of_px_r_mu_init(self):
        """Under inverse_sqrt, theta = 1/mean^2 so px_r_mu_init = -2*log(mean)."""
        hp = resolve_dispersion_hyper_prior_params(
            mean=2.0,
            alpha=5.0,
            direction="inverse_sqrt",
            init_mode="dispersion",
        )
        # 1/sqrt(theta) = 2 → theta = 1/4 → log(theta) = -log(4)
        assert hp.px_r_mu_init == pytest.approx(-2.0 * math.log(2.0), rel=1e-12)
        assert hp.px_r_mu_init == pytest.approx(math.log(0.25), rel=1e-12)


class TestResolveExpectedRNASqrt:
    """expected_RNA decoder, direction='sqrt' (Item 6 new path)."""

    def test_sqrt_direction(self):
        """(mean=3, alpha=9, sqrt, dispersion) → beta=27, lambda=1/3, px_r_mu=log(9)."""
        hp = resolve_dispersion_hyper_prior_params(
            mean=3.0,
            alpha=9.0,
            direction="sqrt",
            init_mode="dispersion",
        )
        assert hp.alpha == pytest.approx(9.0, rel=1e-12)
        assert hp.beta == pytest.approx(27.0, rel=1e-12)
        assert hp.lambda_init == pytest.approx(1.0 / 3.0, rel=1e-12)
        assert hp.px_r_mu_init == pytest.approx(math.log(9.0), rel=1e-12)

    def test_sqrt_preserves_sign_of_px_r_mu_init(self):
        """Under sqrt, theta = mean^2 so px_r_mu_init = +2*log(mean)."""
        hp = resolve_dispersion_hyper_prior_params(
            mean=2.0,
            alpha=5.0,
            direction="sqrt",
            init_mode="dispersion",
        )
        assert hp.px_r_mu_init == pytest.approx(2.0 * math.log(2.0), rel=1e-12)
        assert hp.px_r_mu_init == pytest.approx(math.log(4.0), rel=1e-12)


class TestResolveBurstVariance:
    """burst_frequency_size decoder, init_mode='variance'."""

    def test_burst_old_default(self):
        """(mean=0.02, alpha=2, inverse_sqrt, variance) → beta=0.04, lambda=50, px_r_mu=log(0.0004)."""
        hp = resolve_dispersion_hyper_prior_params(
            mean=0.02,
            alpha=2.0,
            direction="inverse_sqrt",
            init_mode="variance",
        )
        assert hp.alpha == pytest.approx(2.0, rel=1e-12)
        assert hp.beta == pytest.approx(0.04, rel=1e-12)
        assert hp.lambda_init == pytest.approx(50.0, rel=1e-12)
        assert hp.px_r_mu_init == pytest.approx(math.log(0.0004), rel=1e-12)
        # Also equal to 2*log(0.02)
        assert hp.px_r_mu_init == pytest.approx(2.0 * math.log(0.02), rel=1e-12)

    def test_direction_is_ignored_for_variance_mode(self):
        """init_mode='variance' must ignore direction entirely."""
        hp_inv = resolve_dispersion_hyper_prior_params(
            mean=0.02,
            alpha=2.0,
            direction="inverse_sqrt",
            init_mode="variance",
        )
        hp_sqrt = resolve_dispersion_hyper_prior_params(
            mean=0.02,
            alpha=2.0,
            direction="sqrt",
            init_mode="variance",
        )
        assert hp_inv.alpha == hp_sqrt.alpha
        assert hp_inv.beta == hp_sqrt.beta
        assert hp_inv.lambda_init == hp_sqrt.lambda_init
        assert hp_inv.px_r_mu_init == hp_sqrt.px_r_mu_init


class TestResolveInvalidInputs:
    """Argument validation."""

    @pytest.mark.parametrize("bad_mean", [0.0, -1.0, -1e-9, -100.0])
    def test_invalid_mean_raises(self, bad_mean):
        with pytest.raises(ValueError, match="dispersion_hyper_prior_mean"):
            resolve_dispersion_hyper_prior_params(
                mean=bad_mean,
                alpha=9.0,
                direction="inverse_sqrt",
                init_mode="dispersion",
            )

    @pytest.mark.parametrize("bad_alpha", [0.0, -1.0, -1e-9, -100.0])
    def test_invalid_alpha_raises(self, bad_alpha):
        with pytest.raises(ValueError, match="dispersion_hyper_prior_alpha"):
            resolve_dispersion_hyper_prior_params(
                mean=1.0,
                alpha=bad_alpha,
                direction="inverse_sqrt",
                init_mode="dispersion",
            )

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction"):
            resolve_dispersion_hyper_prior_params(
                mean=1.0,
                alpha=9.0,
                direction="bogus",  # type: ignore[arg-type]
                init_mode="dispersion",
            )

    def test_invalid_init_mode_raises(self):
        with pytest.raises(ValueError, match="init_mode"):
            resolve_dispersion_hyper_prior_params(
                mean=1.0,
                alpha=9.0,
                direction="inverse_sqrt",
                init_mode="other",  # type: ignore[arg-type]
            )


class TestResolveReturnType:
    """Return type contract."""

    def test_returns_namedtuple(self):
        hp = resolve_dispersion_hyper_prior_params(
            mean=1.0,
            alpha=9.0,
            direction="inverse_sqrt",
            init_mode="dispersion",
        )
        assert isinstance(hp, DispersionHyperPriorParams)
        # NamedTuple contract: field access
        assert hp.alpha == 9.0
        assert hasattr(hp, "alpha")
        assert hasattr(hp, "beta")
        assert hasattr(hp, "lambda_init")
        assert hasattr(hp, "px_r_mu_init")
        # Explicit field name tuple check
        assert DispersionHyperPriorParams._fields == (
            "alpha",
            "beta",
            "lambda_init",
            "px_r_mu_init",
        )
        # Also unpackable as a 4-tuple
        a, b, lam, px = hp
        assert (a, b, lam, px) == (hp.alpha, hp.beta, hp.lambda_init, hp.px_r_mu_init)


class TestResolveBetaFormula:
    """Document and lock in the unusual β = α · mean formula."""

    @pytest.mark.parametrize(
        ("mean", "alpha"),
        [
            (0.5, 2.0),
            (2.0, 5.0),
            (0.1, 10.0),
            (1.0 / 3.0, 9.0),
            (3.0, 9.0),
            (0.02, 2.0),
        ],
    )
    def test_beta_is_alpha_times_mean(self, mean, alpha):
        """β = α · mean for all (mean, alpha) pairs, regardless of direction/mode."""
        for direction in ("inverse_sqrt", "sqrt"):
            for init_mode in ("variance", "dispersion"):
                hp = resolve_dispersion_hyper_prior_params(
                    mean=mean,
                    alpha=alpha,
                    direction=direction,
                    init_mode=init_mode,
                )
                assert hp.beta == pytest.approx(alpha * mean, rel=1e-12), (
                    f"beta != alpha*mean for (mean={mean}, alpha={alpha}, direction={direction}, init_mode={init_mode})"
                )


class TestResolveLambdaFormula:
    """Document and lock in λ_init = 1 / mean."""

    @pytest.mark.parametrize(
        ("mean", "alpha"),
        [
            (0.5, 2.0),
            (2.0, 5.0),
            (0.1, 10.0),
            (1.0 / 3.0, 9.0),
            (3.0, 9.0),
            (0.02, 2.0),
        ],
    )
    def test_lambda_init_is_one_over_mean(self, mean, alpha):
        """λ_init = 1 / mean regardless of direction/mode."""
        for direction in ("inverse_sqrt", "sqrt"):
            for init_mode in ("variance", "dispersion"):
                hp = resolve_dispersion_hyper_prior_params(
                    mean=mean,
                    alpha=alpha,
                    direction=direction,
                    init_mode=init_mode,
                )
                assert hp.lambda_init == pytest.approx(1.0 / mean, rel=1e-12), (
                    f"lambda_init != 1/mean for (mean={mean}, alpha={alpha}, "
                    f"direction={direction}, init_mode={init_mode})"
                )
