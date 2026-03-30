#!/usr/bin/env python3
"""Extract learned dispersion (theta) parameters from trained model checkpoints.

Loads model.pt files, extracts px_r_mu/px_r_log_sigma, computes theta statistics.
Reports quantiles across genes for each experiment.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def extract_dispersion(model_path, name=None):
    """Extract dispersion parameters from a model checkpoint.

    Returns dict with theta statistics and raw arrays.
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # scvi-tools saves as {'model_state_dict': ..., 'var_names': ..., 'attr_dict': ...}
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Find px_r_mu — could be top-level or nested in module
    if "px_r_mu" in state_dict:
        px_r_mu = state_dict["px_r_mu"].numpy()
        px_r_log_sigma = state_dict["px_r_log_sigma"].numpy()
    elif "module.px_r_mu" in state_dict:
        px_r_mu = state_dict["module.px_r_mu"].numpy()
        px_r_log_sigma = state_dict["module.px_r_log_sigma"].numpy()
    else:
        # Search for the key
        mu_keys = [k for k in state_dict if "px_r_mu" in k and "log_sigma" not in k]
        sigma_keys = [k for k in state_dict if "px_r_log_sigma" in k]
        if not mu_keys:
            raise KeyError(f"No px_r_mu found in {model_path}. Keys: {list(state_dict.keys())[:20]}")
        px_r_mu = state_dict[mu_keys[0]].numpy()
        px_r_log_sigma = state_dict[sigma_keys[0]].numpy()

    # Learned prior rate
    rate_keys = [k for k in state_dict if "dispersion_prior_rate_raw" in k]
    if rate_keys:
        rate_raw = state_dict[rate_keys[0]].numpy()
        learned_rate = np.log1p(np.exp(rate_raw))  # softplus
    else:
        learned_rate = np.array([np.nan])

    # Compute theta
    px_r_sigma = np.exp(px_r_log_sigma)

    # For gene-batch: px_r_mu has shape (n_genes, n_batches)
    # Compute statistics across genes (and batches if present)
    is_gene_batch = px_r_mu.ndim == 2

    if is_gene_batch:
        # Per-gene: average across batches first for summary
        px_r_mu_avg = px_r_mu.mean(axis=1)
        px_r_sigma_avg = px_r_sigma.mean(axis=1)
    else:
        px_r_mu_avg = px_r_mu
        px_r_sigma_avg = px_r_sigma

    theta_median = np.exp(px_r_mu_avg)  # LogNormal median (what model uses at inference)
    theta_mean = np.exp(px_r_mu_avg + (px_r_sigma_avg**2) / 2)  # true LogNormal mean

    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    q_labels = ["min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]

    result = {
        "name": name or Path(model_path).parent.parent.name,
        "model_path": str(model_path),
        "n_genes": px_r_mu_avg.shape[0],
        "is_gene_batch": is_gene_batch,
        "n_batches": px_r_mu.shape[1] if is_gene_batch else 1,
        "learned_rate": float(learned_rate.item() if learned_rate.size == 1 else learned_rate[0]),
        "theta_median_quantiles": {
            q_labels[i]: float(np.quantile(theta_median, quantiles[i])) for i in range(len(quantiles))
        },
        "theta_mean_quantiles": {
            q_labels[i]: float(np.quantile(theta_mean, quantiles[i])) for i in range(len(quantiles))
        },
        "log_theta_median_quantiles": {
            q_labels[i]: float(np.quantile(np.log(theta_median), quantiles[i])) for i in range(len(quantiles))
        },
        "sigma_quantiles": {
            q_labels[i]: float(np.quantile(px_r_sigma_avg, quantiles[i])) for i in range(len(quantiles))
        },
        "ratio_mean_over_median_quantiles": {
            q_labels[i]: float(np.quantile(theta_mean / theta_median, quantiles[i])) for i in range(len(quantiles))
        },
    }

    # Per-batch stats if gene-batch
    if is_gene_batch:
        batch_medians = []
        for b in range(px_r_mu.shape[1]):
            theta_b = np.exp(px_r_mu[:, b])
            batch_medians.append(float(np.median(theta_b)))
        result["per_batch_median_theta"] = batch_medians

    return result


def format_quantile_row(label, q_dict):
    """Format a quantile dict as a table row."""
    vals = [q_dict[k] for k in q_dict]
    return f"  {label:<25s} " + " ".join(f"{v:>8.3f}" for v in vals)


def print_results(results):
    """Print formatted results table."""
    q_labels = ["min", "1%", "5%", "10%", "25%", "50%", "75%", "90%", "95%", "99%", "max"]
    header = f"  {'':25s} " + " ".join(f"{q:>8s}" for q in q_labels)

    for r in results:
        print(f"\n{'=' * 120}")
        print(f"  Experiment: {r['name']}")
        print(f"  Genes: {r['n_genes']}, Gene-batch: {r['is_gene_batch']}, Batches: {r['n_batches']}")
        print(f"  Learned rate (softplus): {r['learned_rate']:.4f}")
        print(f"  Implied prior E[theta] = rate^2 = {r['learned_rate'] ** 2:.4f}")
        print()
        print(header)
        print(f"  {'-' * 115}")
        print(format_quantile_row("theta (median/inference)", r["theta_median_quantiles"]))
        print(format_quantile_row("theta (LogNormal mean)", r["theta_mean_quantiles"]))
        print(format_quantile_row("log(theta_median)", r["log_theta_median_quantiles"]))
        print(format_quantile_row("px_r_sigma", r["sigma_quantiles"]))
        print(format_quantile_row("ratio mean/median", r["ratio_mean_over_median_quantiles"]))

    # Summary comparison table
    print(f"\n\n{'=' * 120}")
    print("SUMMARY: Median theta (50th percentile) across experiments")
    print(f"{'=' * 120}")
    print(f"  {'Experiment':<55s} {'median_θ':>10s} {'mean_θ':>10s} {'σ_50%':>10s} {'rate':>8s} {'rate²':>8s}")
    print(f"  {'-' * 105}")
    for r in results:
        print(
            f"  {r['name']:<55s} "
            f"{r['theta_median_quantiles']['50%']:>10.3f} "
            f"{r['theta_mean_quantiles']['50%']:>10.3f} "
            f"{r['sigma_quantiles']['50%']:>10.4f} "
            f"{r['learned_rate']:>8.4f} "
            f"{r['learned_rate'] ** 2:>8.4f}"
        )


def main():
    """CLI entry point for extracting learned dispersion from model checkpoints."""
    parser = argparse.ArgumentParser(description="Extract learned dispersion from model checkpoints")
    parser.add_argument(
        "--results-dir",
        default="/nfs/team205/vk7/sanger_projects/my_packages/regularizedvi/results",
        help="Base results directory",
    )
    parser.add_argument("--output-json", default=None, help="Save results as JSON")
    parser.add_argument("--pattern", default=None, help="Glob pattern for result directories")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Define experiments: P5 (disp1/disp2) + default prior=3 baselines
    experiments = [
        # Default prior=3 baselines (for comparison)
        ("no_tea_small", "immune_integration_rna_embryo_es2_no_tea_small"),
        ("no_tea", "immune_integration_rna_embryo_es2_no_tea"),
        ("no_tea_small_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_small_dwl2_1"),
        ("no_tea_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_dwl2_1"),
        # P5: disp_mean=1
        ("P5-1 no_tea_small_disp1", "immune_integration_rna_embryo_es2_no_tea_small_disp1"),
        ("P5-2 no_tea_disp1", "immune_integration_rna_embryo_es2_no_tea_disp1"),
        ("P5-5 no_tea_small_disp1_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_small_disp1_dwl2_1"),
        ("P5-6 no_tea_disp1_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_disp1_dwl2_1"),
        # P5: disp_mean=2
        ("P5-3 no_tea_small_disp2", "immune_integration_rna_embryo_es2_no_tea_small_disp2"),
        ("P5-4 no_tea_disp2", "immune_integration_rna_embryo_es2_no_tea_disp2"),
        ("P5-7 no_tea_small_disp2_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_small_disp2_dwl2_1"),
        ("P5-8 no_tea_disp2_dwl2_1", "immune_integration_rna_embryo_es2_no_tea_disp2_dwl2_1"),
    ]

    results = []
    for name, dirname in experiments:
        model_path = results_dir / dirname / "model" / "model.pt"
        if not model_path.exists():
            print(f"  SKIP {name}: {model_path} not found", file=sys.stderr)
            continue
        try:
            r = extract_dispersion(model_path, name=name)
            results.append(r)
        except (KeyError, RuntimeError, OSError) as e:
            print(f"  ERROR {name}: {e}", file=sys.stderr)

    if not results:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
