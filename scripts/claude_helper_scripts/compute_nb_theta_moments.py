#!/usr/bin/env python3
"""Compute NB dispersion theta from method of moments on h5ad files.

Wrapper around regularizedvi._dispersion_init.compute_dispersion_init().
Saves raw diagnostics to npz so theta can be recomputed for different
biological_variance_fraction values without re-running Welford.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add package src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from regularizedvi._dispersion_init import compute_dispersion_init


def main():
    """CLI wrapper for compute_dispersion_init."""
    parser = argparse.ArgumentParser(description="Compute NB theta from method of moments")
    parser.add_argument("path", help="Path to .h5ad file")
    parser.add_argument("--layer", default=None, help="Layer name (default: X)")
    parser.add_argument("--feature-type", default=None, help="Filter by feature_types (e.g. GEX, ATAC)")
    parser.add_argument("--dispersion-key", default=None, help="obs column for batch grouping")
    parser.add_argument("--bio-frac", type=float, default=10.0, help="Biological variance fraction (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size (default: 5000)")
    parser.add_argument("--save-npz", default=None, help="Save all diagnostics to .npz file")
    args = parser.parse_args()

    log_theta, diag = compute_dispersion_init(
        args.path,
        layer=args.layer,
        dispersion_key=args.dispersion_key,
        biological_variance_fraction=args.bio_frac,
        theta_min=1e-10,
        theta_max=np.inf,
        chunk_size=args.chunk_size,
        feature_type=args.feature_type,
        verbose=True,
    )

    print(f"\nlog_theta shape: {log_theta.shape}")

    # Print theta for multiple bio_frac values
    bio_fracs = [1, 5, 10, 20]
    cv2_L = diag["cv2_L"]
    nb_inflation = 1 + cv2_L
    excess_raw = diag["excess_raw"]
    mean_g = diag["mean_g"]
    eps = 1e-10

    print(f"\n{'bio_frac':>10s} {'θ_5%':>10s} {'θ_25%':>10s} {'θ_50%':>10s} {'θ_75%':>10s} {'θ_95%':>10s}")
    print("-" * 65)
    for bf in bio_fracs:
        excess_tech = excess_raw / (nb_inflation * bf)
        theta = (mean_g**2) / np.maximum(excess_tech, eps)
        finite_t = theta[np.isfinite(theta) & (theta > 0)]
        print(
            f"{bf:>10.0f} "
            f"{np.quantile(finite_t, 0.05):>10.3f} "
            f"{np.quantile(finite_t, 0.25):>10.3f} "
            f"{np.quantile(finite_t, 0.50):>10.3f} "
            f"{np.quantile(finite_t, 0.75):>10.3f} "
            f"{np.quantile(finite_t, 0.95):>10.3f}"
        )

    if args.save_npz:
        np.savez(
            args.save_npz,
            mean_g=diag["mean_g"],
            var_g=diag["var_g"],
            cv2_L=np.array([diag["cv2_L"]]),
            cv2_L_within_batch=np.array([diag["cv2_L_within_batch"]]),
            cv2_L_between_batch=np.array([diag["cv2_L_between_batch"]]),
            excess_raw=diag["excess_raw"],
            excess_adjusted=diag["excess_adjusted"],
            poisson_var=diag["poisson_var"],
            library_var=diag["library_var"],
            n_sub_poisson=np.array([diag["n_sub_poisson"]]),
        )
        print(f"\nSaved diagnostics to {args.save_npz}")


if __name__ == "__main__":
    main()
