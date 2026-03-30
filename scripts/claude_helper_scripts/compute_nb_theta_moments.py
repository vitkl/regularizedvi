#!/usr/bin/env python3
"""Compute NB dispersion theta from method of moments on h5ad files.

Wrapper around regularizedvi._dispersion_init.compute_dispersion_init().
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
    parser.add_argument("--theta-min", type=float, default=0.01, help="Theta clip min (default: 0.01)")
    parser.add_argument("--theta-max", type=float, default=10.0, help="Theta clip max (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size (default: 5000)")
    parser.add_argument("--no-clip", action="store_true", help="Disable clipping (set min=1e-10, max=inf)")
    parser.add_argument("--save-npz", default=None, help="Save per-gene arrays to .npz file")
    args = parser.parse_args()

    theta_min = 1e-10 if args.no_clip else args.theta_min
    theta_max = np.inf if args.no_clip else args.theta_max

    log_theta, diag = compute_dispersion_init(
        args.path,
        layer=args.layer,
        dispersion_key=args.dispersion_key,
        biological_variance_fraction=args.bio_frac,
        theta_min=theta_min,
        theta_max=theta_max,
        chunk_size=args.chunk_size,
        feature_type=args.feature_type,
        verbose=True,
    )

    print(f"\nlog_theta shape: {log_theta.shape}")

    if args.save_npz:
        np.savez(
            args.save_npz,
            log_theta=log_theta,
            theta_option1=diag["theta_option1"],
            theta_option2=diag["theta_option2"],
            mean_g=diag["mean_g"],
            var_g=diag["var_g"],
        )
        print(f"Saved to {args.save_npz}")


if __name__ == "__main__":
    main()
