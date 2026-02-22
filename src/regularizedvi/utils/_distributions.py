"""Distribution comparison utilities.

Functions to compare scvi-tools NegativeBinomial with Pyro GammaPoisson
and visualise the dispersion prior penalty landscape.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Exponential


def compare_nb_gammapoisson(
    x_range: tuple[int, int] = (0, 100),
    mu: float = 5.0,
    theta_range: tuple[float, float] = (0.01, 1000),
    n_theta: int = 50,
) -> dict[str, np.ndarray]:
    """Compare log_prob from scvi-tools NB vs Pyro GammaPoisson over a grid.

    Parameters
    ----------
    x_range
        Range of observed counts (inclusive).
    mu
        Mean parameter for the NB distribution.
    theta_range
        Range of theta (inverse dispersion / GammaPoisson concentration).
    n_theta
        Number of theta values (log-spaced).

    Returns
    -------
    Dict with arrays: x_grid, theta_grid, nb_logprob, gp_logprob, diff.
    """
    from pyro.distributions import GammaPoisson
    from scvi.distributions import NegativeBinomial

    x_vals = torch.arange(x_range[0], x_range[1] + 1, dtype=torch.float32)
    theta_vals = torch.logspace(np.log10(theta_range[0]), np.log10(theta_range[1]), n_theta)

    # Create grids: (n_theta, n_x)
    x_grid, theta_grid = torch.meshgrid(x_vals, theta_vals, indexing="xy")
    # x_grid shape: (n_theta, n_x), theta_grid shape: (n_theta, n_x)

    mu_t = torch.full_like(x_grid, mu)

    # scvi-tools NB
    nb = NegativeBinomial(mu=mu_t, theta=theta_grid)
    nb_lp = nb.log_prob(x_grid).detach().numpy()

    # Pyro GammaPoisson
    gp = GammaPoisson(concentration=theta_grid, rate=theta_grid / mu_t)
    gp_lp = gp.log_prob(x_grid).detach().numpy()

    return {
        "x_vals": x_vals.numpy(),
        "theta_vals": theta_vals.numpy(),
        "nb_logprob": nb_lp,
        "gp_logprob": gp_lp,
        "diff": nb_lp - gp_lp,
        "mu": mu,
    }


def compare_prior_directions(
    theta_range: tuple[float, float] = (0.01, 1000),
    rate: float = 3.0,
    n_points: int = 200,
) -> dict[str, np.ndarray]:
    """Compare the two prior penalty directions over theta range.

    Direction 1 (current regularizedvi): Exp prior on sqrt(theta)
    Direction 2 (cell2location-like): Exp prior on 1/sqrt(theta)

    Parameters
    ----------
    theta_range
        Range of theta values (log-spaced).
    rate
        Exponential distribution rate parameter.
    n_points
        Number of theta values.

    Returns
    -------
    Dict with arrays: theta, phi, penalty_current, penalty_flipped,
    gradient_current, gradient_flipped.
    """
    theta_vals = torch.logspace(np.log10(theta_range[0]), np.log10(theta_range[1]), n_points)
    exp_dist = Exponential(torch.tensor(rate))

    # Direction 1: Exp on sqrt(theta) — current code
    sqrt_theta = theta_vals.pow(0.5)
    lp_current = exp_dist.log_prob(sqrt_theta)
    penalty_current = -lp_current  # neg log prior (added to loss)

    # Direction 2: Exp on 1/sqrt(theta) — cell2location direction
    inv_sqrt_theta = theta_vals.pow(-0.5)
    lp_flipped = exp_dist.log_prob(inv_sqrt_theta)
    penalty_flipped = -lp_flipped

    # Gradients w.r.t. phi = log(theta)
    grad_current = []
    grad_flipped = []
    for theta_val in theta_vals:
        phi = torch.log(theta_val).requires_grad_(True)
        # Current direction
        t = torch.exp(phi).pow(0.5)
        p = -exp_dist.log_prob(t)
        p.backward()
        grad_current.append(phi.grad.item())

        phi2 = torch.log(theta_val).requires_grad_(True)
        # Flipped direction
        t2 = torch.exp(-phi2).pow(0.5)
        p2 = -exp_dist.log_prob(t2)
        p2.backward()
        grad_flipped.append(phi2.grad.item())

    return {
        "theta": theta_vals.numpy(),
        "phi": torch.log(theta_vals).numpy(),
        "penalty_current": penalty_current.detach().numpy(),
        "penalty_flipped": penalty_flipped.detach().numpy(),
        "gradient_current": np.array(grad_current),
        "gradient_flipped": np.array(grad_flipped),
        "rate": rate,
    }


def nb_variance(mu: float, theta: np.ndarray) -> np.ndarray:
    """Compute NB variance: Var = mu + mu^2/theta."""
    return mu + mu**2 / theta


def plot_nb_vs_gammapoisson(results: dict, figsize: tuple[int, int] = (16, 5)):
    """Plot heatmap comparison of NB vs GammaPoisson log_prob.

    Parameters
    ----------
    results
        Output of :func:`compare_nb_gammapoisson`.
    figsize
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    x_vals = results["x_vals"]
    theta_vals = results["theta_vals"]
    mu = results["mu"]

    vmin = min(results["nb_logprob"].min(), results["gp_logprob"].min())
    vmax = max(results["nb_logprob"].max(), results["gp_logprob"].max())

    for ax, data, title in [
        (axes[0], results["nb_logprob"], f"scvi-tools NB (mu={mu})"),
        (axes[1], results["gp_logprob"], f"Pyro GammaPoisson (mu={mu})"),
    ]:
        im = ax.pcolormesh(x_vals, theta_vals, data, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_yscale("log")
        ax.set_xlabel("Observed count x")
        ax.set_ylabel("theta (= concentration)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="log_prob")

    # Difference
    diff = results["diff"]
    max_diff = max(abs(diff.min()), abs(diff.max()), 1e-10)
    im = axes[2].pcolormesh(
        x_vals,
        theta_vals,
        diff,
        shading="auto",
        cmap="RdBu_r",
        vmin=-max_diff,
        vmax=max_diff,
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Observed count x")
    axes[2].set_ylabel("theta (= concentration)")
    axes[2].set_title(f"|NB - GP| (max={np.abs(diff).max():.2e})")
    plt.colorbar(im, ax=axes[2], label="NB - GP")

    plt.tight_layout()
    return fig


def plot_prior_comparison(results: dict, figsize: tuple[int, int] = (14, 5)):
    """Plot prior penalty and gradient for both directions.

    Parameters
    ----------
    results
        Output of :func:`compare_prior_directions`.
    figsize
        Figure size.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    theta = results["theta"]
    rate = results["rate"]

    # Panel 1: Penalty (neg log prior) vs theta
    ax = axes[0]
    ax.plot(theta, results["penalty_current"], label="Current: Exp on sqrt(theta)", color="C0")
    ax.plot(theta, results["penalty_flipped"], label="Flipped: Exp on 1/sqrt(theta)", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel("theta")
    ax.set_ylabel("neg_log_prior (added to loss)")
    ax.set_title(f"Penalty landscape (rate={rate})")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Panel 2: Gradient w.r.t. phi = log(theta)
    ax = axes[1]
    ax.plot(theta, results["gradient_current"], label="Current: d(penalty)/d(phi)", color="C0")
    ax.plot(theta, results["gradient_flipped"], label="Flipped: d(penalty)/d(phi)", color="C1")
    ax.set_xscale("log")
    ax.set_xlabel("theta")
    ax.set_ylabel("d(penalty)/d(phi)")
    ax.set_title("Gradient direction")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        "positive = pushes theta down", xy=(0.5, 0.95), xycoords="axes fraction", ha="center", fontsize=8, color="gray"
    )
    ax.annotate(
        "negative = pushes theta up", xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=8, color="gray"
    )

    # Panel 3: NB variance vs theta
    ax = axes[2]
    mu_vals = [2, 5, 10]
    for mu in mu_vals:
        var = nb_variance(mu, theta)
        ax.plot(theta, var, label=f"mu={mu}")
        ax.axhline(mu, color="gray", linestyle=":", alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("theta")
    ax.set_ylabel("Var(X)")
    ax.set_title("NB variance = mu + mu²/theta")
    ax.legend()
    ax.annotate("← overdispersed", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8, color="gray")
    ax.annotate("Poisson-like →", xy=(0.75, 0.05), xycoords="axes fraction", fontsize=8, color="gray")

    plt.tight_layout()
    return fig
