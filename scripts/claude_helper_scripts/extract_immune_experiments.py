"""Extract hyperparameters from immune experiment checkpoints and generate TSV.

Usage:
    bash scripts/helper_scripts/run_python_cmd.sh scripts/claude_helper_scripts/extract_immune_experiments.py

Outputs:
    docs/notebooks/immune_integration/integration_metrics_experiments.tsv
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch

RESULTS_BASE = Path("results")
OUTPUT_TSV = Path("docs/notebooks/immune_integration/integration_metrics_experiments.tsv")
BURST_JOBS_TSV = Path("docs/notebooks/model_comparisons/burst_jobs.tsv")

# Folders to skip (not individual experiments)
SKIP_FOLDERS = {"immune_integration", "immune_integration_metrics_comparison"}

# Era assignment rules (order matters — first match wins)
ERA_RULES = [
    # Era V: burst/baseline/v2 models (newest)
    (5, lambda n: n.startswith(("immune_burst_", "immune_baseline_", "immune_rna_v2_"))),
    # Era IV: decoder/dispersion sweep
    (
        4,
        lambda n: (
            n.startswith("immune_integration_rna_dwl2")
            or n.startswith("immune_integration_rna_disp_data")
            or n.startswith("immune_integration_rna_baseline_")
        ),
    ),
    # Era III: embryo-param transfer
    (3, lambda n: "embryo_es" in n),
    # Era II: QC-focused
    (2, lambda n: "studyqc" in n or n == "immune_integration_rna_bg_flat" or n == "immune_integration_rna_v2"),
    # Era I: early exploratory (everything else)
    (1, lambda n: True),
]

# Columns for output TSV
TSV_COLUMNS = [
    "name",
    "type",
    "era",
    "results_folder",
    "label",
    "notebook",
    "status",
    "n_hidden",
    "n_latent",
    "decoder_type",
    "decoder_weight_l2",
    "dispersion_prior_mean",
    "dispersion_prior_alpha",
    "dispersion_init",
    "burst_size_intercept",
    "library_log_vars_weight",
    "library_log_means_centering_sensitivity_rna",
    "regularise_background",
    "residual_library_encoder",
    "library_obs_w_prior_rate",
    "early_stopping_min_delta_per_feature",
    "stratify_validation_key",
    "max_epochs",
    "exclude_datasets",
    "notes",
]


def assign_era(name: str) -> int:  # noqa: D103
    for era, rule in ERA_RULES:
        if rule(name):
            return era
    return 1


def extract_dict_value(val):
    """Extract scalar from dict-valued param (multimodal models store per-modality)."""
    if isinstance(val, dict):
        if "rna" in val:
            return val["rna"]
        # Return first value if no 'rna' key
        return next(iter(val.values()))
    return val


def load_checkpoint_params(model_pt_path: str) -> dict:
    """Load hyperparameters from model.pt checkpoint."""
    try:
        d = torch.load(model_pt_path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR loading {model_pt_path}: {e}", file=sys.stderr)
        return {}

    nk = d.get("attr_dict", {}).get("init_params_", {}).get("non_kwargs", {})

    params = {}
    # Architecture
    params["n_hidden"] = extract_dict_value(nk.get("n_hidden"))
    params["n_latent"] = extract_dict_value(nk.get("n_latent"))

    # Decoder type
    dt = nk.get("decoder_type")
    if dt is not None:
        params["decoder_type"] = extract_dict_value(dt)

    # Regularization
    params["decoder_weight_l2"] = nk.get("decoder_weight_l2")

    # Dispersion prior
    rdp = nk.get("regularise_dispersion_prior")
    if rdp is not None:
        params["dispersion_prior_mean"] = extract_dict_value(rdp)
    dhpa = nk.get("dispersion_hyper_prior_alpha")
    if dhpa is not None:
        params["dispersion_prior_alpha"] = extract_dict_value(dhpa)

    # Dispersion init
    params["dispersion_init"] = nk.get("dispersion_init")

    # Burst
    params["burst_size_intercept"] = nk.get("burst_size_intercept")

    # Library
    params["library_log_vars_weight"] = nk.get("library_log_vars_weight")
    lcs = nk.get("library_log_means_centering_sensitivity")
    if lcs is not None:
        params["library_log_means_centering_sensitivity_rna"] = extract_dict_value(lcs)

    # Background / feature scaling
    params["regularise_background"] = nk.get("regularise_background")
    params["residual_library_encoder"] = nk.get("residual_library_encoder")
    params["library_obs_w_prior_rate"] = nk.get("library_obs_w_prior_rate")

    # Filter out None values
    return {k: v for k, v in params.items() if v is not None}


def load_burst_jobs_metadata() -> dict:
    """Load burst_jobs.tsv for Era V experiment metadata."""
    if not BURST_JOBS_TSV.exists():
        return {}
    df = pd.read_csv(BURST_JOBS_TSV, sep="\t")
    metadata = {}
    for _, row in df.iterrows():
        name = row.get("name", "")
        if name.startswith("immune_"):
            metadata[name] = row.to_dict()
    return metadata


def find_notebook_path(name: str) -> str:
    """Try to find the output notebook for an experiment."""
    # Check common patterns
    candidates = [
        f"docs/notebooks/immune_integration/bm_pbmc_rna_{name.replace('immune_integration_rna_', '')}_out.ipynb",
        f"docs/notebooks/immune_integration/bm_pbmc_rna_training_{name.replace('immune_integration_rna_', '')}_out.ipynb",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return ""


def detect_model_type(name: str, params: dict) -> str:
    """Detect if this is a standard scVI or regularizedvi model."""
    if name == "immune_integration_rna_scvi_baseline":
        return "scvi"
    return "rna"


def has_latent_output(name: str) -> bool:
    """Check if experiment has latent representation output."""
    outputs_dir = RESULTS_BASE / name / "model" / "outputs"
    return (outputs_dir / "X_scVI.csv").exists() or (outputs_dir / "X_regularizedvi.csv").exists()


def detect_status(name: str) -> str:
    """Check if model training completed."""
    model_pt = RESULTS_BASE / name / "model" / "model.pt"
    if not model_pt.exists():
        return "no_checkpoint"
    if has_latent_output(name):
        return "completed"
    return "trained_no_outputs"


def main():  # noqa: D103
    # Load burst_jobs metadata for cross-referencing
    burst_meta = load_burst_jobs_metadata()

    rows = []
    # Scan results directories
    immune_dirs = sorted(
        d
        for d in os.listdir(RESULTS_BASE)
        if d.startswith("immune") and d not in SKIP_FOLDERS and (RESULTS_BASE / d / "model").exists()
    )

    print(f"Found {len(immune_dirs)} immune experiment directories")

    for name in immune_dirs:
        model_pt = RESULTS_BASE / name / "model" / "model.pt"
        status = detect_status(name)

        row = dict.fromkeys(TSV_COLUMNS, "")
        row["name"] = name
        row["type"] = detect_model_type(name, {})
        row["era"] = assign_era(name)
        row["results_folder"] = f"results/{name}/model"
        row["status"] = status

        # Load checkpoint params
        if model_pt.exists():
            cp_params = load_checkpoint_params(str(model_pt))
            for k, v in cp_params.items():
                if k in TSV_COLUMNS:
                    row[k] = v

        # Cross-reference burst_jobs.tsv for Era V (fill only if checkpoint didn't provide)
        if name in burst_meta:
            bm = burst_meta[name]
            # Fill in training params not in checkpoint
            if not row.get("decoder_type") and bm.get("decoder_type_rna", "-") != "-":
                row["decoder_type"] = bm["decoder_type_rna"]
            if not row.get("n_hidden") and "n_hidden" in bm and bm["n_hidden"] != "-":
                row["n_hidden"] = bm["n_hidden"]
            if not row.get("n_latent") and "n_latent" in bm and bm["n_latent"] != "-":
                row["n_latent"] = bm["n_latent"]
            # notebook path from burst_jobs
            if "output" in bm:
                row["notebook"] = bm["output"]

        # Try to find notebook
        if not row["notebook"]:
            row["notebook"] = find_notebook_path(name)

        # Generate label
        label_parts = [f"Era {row['era']}"]
        if row.get("n_hidden"):
            label_parts.append(f"h={row['n_hidden']}")
        if row.get("n_latent"):
            label_parts.append(f"l={row['n_latent']}")
        if row.get("decoder_type") and row["decoder_type"] != "expected_RNA":
            label_parts.append(row["decoder_type"])
        row["label"] = ", ".join(label_parts)

        rows.append(row)
        print(
            f"  {name}: era={row['era']}, status={status}, "
            f"n_hidden={row.get('n_hidden', '?')}, n_latent={row.get('n_latent', '?')}"
        )

    # Create DataFrame
    df = pd.DataFrame(rows, columns=TSV_COLUMNS)

    # Convert boolean columns
    for col in ["regularise_background", "residual_library_encoder"]:
        df[col] = df[col].apply(lambda x: int(x) if isinstance(x, bool) else x)

    # Save
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"\nSaved {len(df)} experiments to {OUTPUT_TSV}")

    # Summary
    for era in sorted(df["era"].unique()):
        era_df = df[df["era"] == era]
        completed = (era_df["status"] == "completed").sum()
        print(f"  Era {era}: {len(era_df)} experiments ({completed} with outputs)")


if __name__ == "__main__":
    main()
